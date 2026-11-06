#!/usr/bin/env python3
"""Contracts for scripts/architecture_model.py (analytical estimates).

Hand-checkable arithmetic, profile validation, support rules, memory rejection,
ranking honesty, and an exact cross-check against the C++ engine's real command
streams (transfer bytes, synchronizations, capability decisions).
Exit 77 when onnx is unavailable.
"""
import argparse
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
BINARY = None
DIGITS = ROOT / 'examples/models/digits'
EDGE = ROOT / 'examples/models/edge-cnn'

try:
    import onnx  # noqa: F401
    import architecture_model as A
    from onnx_frontend import import_model, inspect_model
except ImportError:
    print('onnx unavailable: architecture model contracts skipped')
    sys.exit(77)


def matmul_node(m, k, n):
    shapes = {'x': [m, k], 'w': [k, n], 'y': [m, n]}
    node = {'operator': 'MatMul', 'inputs': ['x', 'w'], 'output': 'y', 'shape': [m, n], 'parameters': [],
            'generated_node': 'n0', 'source_index': 0, 'source_name': 'mm', 'logical_shape': [m, n]}
    return node, shapes


def tiny_report(m=4, k=8, n=2):
    """One MatMul then one ReLU, in the importer's normalized form."""
    node, shapes = matmul_node(m, k, n)
    relu = {'operator': 'Relu', 'inputs': ['y'], 'output': 'z', 'shape': [m, n], 'parameters': [],
            'generated_node': 'n1', 'source_index': 1, 'source_name': 'act', 'logical_shape': [m, n]}
    shapes['z'] = [m, n]
    return {'normalized_nodes': [node, relu],
            'normalized_tensors': [{'name': k_, 'shape': v} for k_, v in shapes.items()],
            'tensor_mapping': {'in': 'x', 'out': 'z'}, 'inputs': [{'source_name': 'in'}],
            'outputs': [{'source_name': 'out'}]}


class Arithmetic(unittest.TestCase):
    def test_matmul_work_and_ideal_traffic_by_hand(self):
        node, shapes = matmul_node(4, 8, 2)
        w = A.work(node, shapes)
        self.assertEqual(w['operations'], 2 * 4 * 2 * 8)           # 128: multiply-add counts as 2
        self.assertEqual(w['ideal_traffic_bytes'], 4 * (32 + 16 + 8))  # x + w + y, f32
        self.assertEqual(w['matrix_mnk'], (4, 2, 8))

    def test_tile_utilization_and_roofline_by_hand(self):
        p = copy.deepcopy(A.load('compact-vector'))
        p['accelerator'].update(peak_gops=1, bandwidth_gbps=1, efficiency=[1, 1, 1], dispatch_us=0, tile_mnk=[8, 8, 8])
        node, shapes = matmul_node(4, 8, 2)
        c = A.kernel_cost(node, shapes, p, 'A', 1)
        self.assertAlmostEqual(c['tile_utilization'], (4 * 2 * 8) / (8 * 8 * 8))  # 64/512 = 12.5%
        # 128 ops at 1 GOPS * 12.5% utilization = 128 / (1e6 * 0.125) ms
        self.assertAlmostEqual(c['compute_ms'], 128 / 125000)
        self.assertAlmostEqual(c['memory_ms'], c['traffic_bytes'] / 1e6)
        self.assertEqual(c['total_ms'], max(c['compute_ms'], c['memory_ms']))

    def test_tiled_traffic_rereads_when_weights_do_not_fit(self):
        p = copy.deepcopy(A.load('compact-vector'))
        node, shapes = matmul_node(64, 64, 64)
        tA = A.tile_scratch_bytes([16, 16, 16])
        p['accelerator'].update(tile_mnk=[16, 16, 16], local_bytes=tA)
        w = A.work(node, shapes)
        traffic, mode = A.tiled_traffic(node, shapes, w, p['accelerator'])
        tile_rows = tile_cols = 64 // 16
        self.assertEqual(traffic, 4 * 64 * 64 * tile_cols + 4 * 64 * 64 * tile_rows + 4 * 64 * 64)
        self.assertIn('output-stationary', mode)
        p['accelerator']['local_bytes'] = tA + 4 * 64 * 64
        traffic, mode = A.tiled_traffic(node, shapes, w, p['accelerator'])
        self.assertEqual(traffic, 4 * 64 * 64 * tile_cols + 4 * 64 * 64 + 4 * 64 * 64)
        self.assertIn('weight-stationary', mode)

    def test_schedule_counts_transfers_syncs_and_lifetimes(self):
        p = A.load('wide-array')
        r = tiny_report()
        s = A.schedule(r, p, 'AC')
        # H2D x (128 B) and w (64 B); D2H y (32 B) before the CPU ReLU -> one sync.
        self.assertEqual((s['h2d_bytes'], s['d2h_bytes'], s['syncs']), (192, 32, 1))
        # Device holds x, w and y together (224 B) plus the kernel workspace.
        self.assertEqual(s['device_peak_bytes'], 224 + p['accelerator']['workspace_bytes'])
        cpu = A.schedule(r, p, 'CC')
        self.assertEqual((cpu['transfer_bytes'], cpu['syncs'], cpu['device_peak_bytes']), (0, 0, 0))
        resident = A.schedule(r, p, 'AC', resident_weights=True)
        self.assertEqual(resident['h2d_bytes'], 128)  # weights were uploaded before the request

    def test_memory_limit_rejects_plan_without_latency(self):
        p = copy.deepcopy(A.load('wide-array'))
        p['accelerator'].update(memory_bytes=240, workspace_bytes=0)  # each node fits (<= 224 B); the whole plan (256 B) does not
        s = A.schedule(tiny_report(), p, 'AA')
        self.assertFalse(s['feasible_under_assumptions'])
        self.assertIsNone(s['estimated_ms'])
        self.assertIn('exceeds declared memory', s['failures'][0]['reasons'][0])


class Profiles(unittest.TestCase):
    def test_registered_profiles_validate_and_are_versioned(self):
        ids = [p['id'] for p in A.registered()]
        self.assertGreaterEqual(len(ids), 3)
        for p in A.registered():
            self.assertFalse(p['evidence']['hardware_validated'])
            self.assertGreaterEqual(p['revision'], 1)

    def test_invalid_profiles_rejected(self):
        base = A.load('wide-array')
        mutations = [lambda p: p.update(extra=1), lambda p: p['evidence'].update(hardware_validated=True),
                     lambda p: p['cpu'].update(efficiency=[0.8, 0.5, 0.3]), lambda p: p['support'].update(operators=['Softmax']),
                     lambda p: p['accelerator'].update(local_bytes=16), lambda p: p['link'].update(bandwidth_gbps=0),
                     lambda p: p.pop('revision')]
        for mutate in mutations:
            p = copy.deepcopy(base)
            mutate(p)
            with self.assertRaises(ValueError):
                A.validate(p)

    def test_duplicate_json_fields_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'dup.json'
            path.write_text('{"id": "a", "id": "b"}')
            with self.assertRaises(ValueError):
                A.load(str(path))

    def test_execution_flags_translate_support_rules(self):
        self.assertNotIn('--relu-supported', A.execution_flags(A.load('fragmented-array')))
        self.assertIn('--device-no-grouped-conv', A.execution_flags(A.load('dense-conv-array')))


class Models(unittest.TestCase):
    def test_unsupported_model_gets_no_latency(self):
        with tempfile.TemporaryDirectory() as d:
            from onnx import TensorProto, helper
            g = helper.make_graph([helper.make_node('Softmax', ['x'], ['y'])], 'g',
                                  [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2])],
                                  [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])])
            path = Path(d) / 'softmax.onnx'
            onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]), path)
            r = A.analyze(path, A.load('wide-array'))
            self.assertIsNone(r['analysis'])
            self.assertEqual(r['issues'][0]['code'], 'OPERATOR')
            c = A.compare_targets(path, [A.load('wide-array'), A.load('compact-vector')])
            self.assertEqual(c['nominal_order'], [])
            self.assertIsNone(c['robust_winner'])

    def test_analysis_scope_accepts_larger_model_but_marks_it_not_executable(self):
        model = EDGE / 'edge-standard-160.onnx'
        self.assertFalse(inspect_model(model)['supported'])  # execution scope rejects it
        r = A.analyze(model, A.load('wide-array'), with_sensitivity=False)
        self.assertTrue(r['import_supported'])
        self.assertFalse(r['functionally_executable'])
        self.assertTrue(any(x['code'] == 'TENSOR_LIMIT' for x in r['execution_limits']))
        self.assertIsNotNone(r['analysis']['selected']['estimated_ms'])

    def test_grouped_conv_rule_causes_fallback_and_more_movement(self):
        model = EDGE / 'edge-depthwise-64.onnx'
        wide = A.analyze(model, A.load('wide-array'), with_sensitivity=False)['analysis']['maximal_supported']
        dense = A.analyze(model, A.load('dense-conv-array'), with_sensitivity=False)['analysis']['maximal_supported']
        fallback = [n for n in dense['nodes'] if any('grouped' in r for r in n['device_rejection_reasons'])]
        self.assertEqual(len(fallback), 2)
        self.assertGreater(dense['syncs'], wide['syncs'])
        self.assertGreater(dense['transfer_bytes'], wide['transfer_bytes'])

    def test_estimates_are_deterministic_and_search_mode_is_labelled(self):
        a = A.analyze(DIGITS / 'mlp.onnx', A.load('wide-array'))
        b = A.analyze(DIGITS / 'mlp.onnx', A.load('wide-array'))
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertTrue(a['analysis']['search']['optimum_of_this_model'])
        big = A.analyze(EDGE / 'edge-depthwise-160.onnx', A.load('dense-conv-array'), with_sensitivity=False)
        self.assertFalse(big['analysis']['search']['optimum_of_this_model'])
        self.assertEqual(len(a['sensitivity']), 2 * len(A.SENSITIVITY))
        lo, hi = a['analysis']['selected']['scenario_range_ms']
        self.assertLessEqual(lo, a['analysis']['selected']['estimated_ms'])
        self.assertLessEqual(a['analysis']['selected']['estimated_ms'], hi)

    def test_identical_targets_tie_and_no_winner_is_claimed(self):
        twin = copy.deepcopy(A.load('wide-array'))
        twin['id'] = 'wide-array-twin'
        c = A.compare_targets(DIGITS / 'mlp.onnx', [A.load('wide-array'), twin])
        self.assertIn('Tie', c['conclusion'])
        self.assertIsNone(c['robust_winner'])

    def test_comparison_requires_same_cpu(self):
        other = copy.deepcopy(A.load('compact-vector'))
        other['cpu']['peak_gops'] = 99
        with self.assertRaises(ValueError):
            A.compare_targets(DIGITS / 'mlp.onnx', [A.load('wide-array'), other])

    def test_cpu_perturbations_apply_to_every_target_together(self):
        profiles = [A.load(t) for t in ('compact-vector', 'wide-array', 'dense-conv-array')]
        c = A.compare_targets(EDGE / 'edge-depthwise-64.onnx', profiles)
        for f in c['winner_flips']:
            if f['parameter'].startswith('cpu.'):
                self.assertEqual(f['target_changed'], 'all targets (shared CPU)')
        self.assertFalse(any(f['parameter'].startswith('cpu.') and f['target_changed'] != 'all targets (shared CPU)'
                             for f in c['winner_flips']))
        # wide-array and dense-conv-array tie exactly: rendered with '=', never as a flip between them.
        self.assertEqual(c['nominal_order_text'], 'compact-vector < dense-conv-array = wide-array')
        self.assertIn('dense-conv-array = wide-array', A.render(c))

    def test_exact_tie_reordering_is_not_a_flip(self):
        profiles = [A.load(t) for t in ('compact-vector', 'wide-array', 'dense-conv-array')]
        c = A.compare_targets(EDGE / 'edge-depthwise-160.onnx', profiles)
        self.assertIn('Tie', c['conclusion'])
        tied = {'wide-array', 'dense-conv-array'}
        self.assertFalse(any(set(f['new_winner'].split(' = ')) & tied for f in c['winner_flips']))

    def test_small_device_memory_rejects_node_like_the_engine(self):
        p = copy.deepcopy(A.load('wide-array'))
        p['accelerator']['memory_bytes'] = 140000
        report = inspect_model(DIGITS / 'mlp.onnx')
        shapes = {x['name']: x['shape'] for x in report['normalized_tensors']}
        reasons = [r for n in report['normalized_nodes'] for r in A.support(n, shapes, A.validate(p))]
        self.assertIn('operands plus output exceed device memory budget', reasons)

    def test_cli_input_errors(self):
        with self.assertRaisesRegex(ValueError, 'Unknown target'):
            A.load('no-such-target')
        with self.assertRaisesRegex(ValueError, 'distinct model files'):
            A.compare_models([DIGITS / 'mlp.onnx', DIGITS / 'mlp.onnx'], A.load('wide-array'))

    def test_analysis_node_cap_and_time_budget(self):
        import time
        from onnx import TensorProto, helper

        def chain(n, path):
            nodes = [helper.make_node('Relu', ['X' if i == 0 else f't{i - 1}'], ['Y' if i == n - 1 else f't{i}'], name=f'r{i}')
                     for i in range(n)]
            g = helper.make_graph(nodes, 'chain', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [4, 4])],
                                  [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4, 4])])
            onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]), path)
            return path
        with tempfile.TemporaryDirectory() as d:
            over = A.analyze(chain(129, Path(d) / 'c129.onnx'), A.load('wide-array'), with_sensitivity=False)
            self.assertIsNone(over['analysis'])
            self.assertTrue(any('1..128' in x['message'] for x in over['issues']))
            start = time.monotonic()
            at_cap = A.analyze(chain(128, Path(d) / 'c128.onnx'), A.load('wide-array'))
            self.assertIsNotNone(at_cap['analysis'])
            self.assertLess(time.monotonic() - start, 90, 'estimate at the analysis cap exceeded its time budget')

    def test_model_comparison_never_invents_accuracy(self):
        c = A.compare_models([EDGE / 'edge-standard-16.onnx', EDGE / 'edge-depthwise-16.onnx'], A.load('wide-array'))
        self.assertTrue(all(r['task_accuracy'] == 'not evaluated' for r in c['rows']))


@unittest.skipUnless(BINARY is not None or '--binary' in sys.argv, 'engine binary not supplied')
class EngineCrossCheck(unittest.TestCase):
    """Analytical movement must equal the engine's emitted commands for every placement."""

    def check(self, model, npz, profile_id, memory_policy='release'):
        p = profile_id if isinstance(profile_id, dict) else A.load(profile_id)
        flags = A.execution_flags(p, memory_policy)
        with tempfile.TemporaryDirectory() as d:
            import_model(model, npz, Path(d) / 'imp')
            search = ['--search', 'bounded'] if len(inspect_model(model)['normalized_nodes']) > 12 else []
            subprocess.run([str(BINARY), 'verify', '--graph', str(Path(d) / 'imp/graph.plg'), '--json',
                            str(Path(d) / 'core.json')] + search + flags, check=True, capture_output=True)
            core = json.loads((Path(d) / 'core.json').read_text())
        report = inspect_model(model)
        shapes = {x['name']: x['shape'] for x in report['normalized_tensors']}
        engine_caps = {x['node_id']: x['device_supported'] for x in core['capabilities']}
        model_caps = {n['generated_node']: not A.support(n, shapes, p) for n in report['normalized_nodes']}
        self.assertEqual(engine_caps, model_caps)
        self.assertGreater(len(core['results']), 1)
        for r in core['results']:
            self.assertTrue(r['correct'])
            s = A.schedule(report, p, r['placement'].replace('D', 'A'), memory_policy=memory_policy)
            workspace = p['accelerator']['workspace_bytes'] if 'A' in s['placement'] else 0
            self.assertEqual((r['h2d_bytes'], r['d2h_bytes'], r['synchronizations'], r['peak_live_device_bytes']),
                             (s['h2d_bytes'], s['d2h_bytes'], s['syncs'], s['device_peak_bytes'] - workspace),
                             f"{model.name} {profile_id} {memory_policy} {r['placement']}")
        return len(core['results'])

    def test_trained_mlp_all_profiles_and_memory_policies(self):
        for pid in ('wide-array', 'fragmented-array', 'compact-vector'):
            for policy in ('release', 'retain'):
                self.check(DIGITS / 'mlp.onnx', DIGITS / 'mlp_test_inputs.npz', pid, policy)

    def test_skip_connection_sync_and_lifetime_rules(self):
        # Regression: a device copy is freed after its last DEVICE reader (the CPU
        # consumer uses the host copy), and the first CPU op after device work waits once.
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper
        with tempfile.TemporaryDirectory() as d:
            w = numpy_helper.from_array(np.eye(4, dtype=np.float32), 'W')
            g = helper.make_graph([helper.make_node('MatMul', ['X', 'W'], ['A'], name='mm'),
                                   helper.make_node('Relu', ['A'], ['B'], name='r1'),
                                   helper.make_node('Relu', ['B'], ['C'], name='r2'),
                                   helper.make_node('Add', ['C', 'A'], ['Y'], name='skip')], 'skip',
                                  [helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 4])],
                                  [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 4])], [w])
            model = Path(d) / 'skip.onnx'
            onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]), model)
            np.savez(Path(d) / 'in.npz', X=np.arange(8, dtype=np.float32).reshape(2, 4) - 3)
            for policy in ('release', 'retain'):
                self.assertEqual(self.check(model, Path(d) / 'in.npz', 'wide-array', policy), 16)

    def test_small_device_memory_profile(self):
        # Per-node rule: distinct operands + output above memory_bytes falls back in both models.
        p = copy.deepcopy(A.load('wide-array'))
        p['accelerator']['memory_bytes'] = 140000
        self.check(DIGITS / 'mlp.onnx', DIGITS / 'mlp_test_inputs.npz', A.validate(p))

    def test_memory_feasibility_boundary_matches_engine(self):
        # The maximal MLP plan needs 329,152 B payload + 4,096 B workspace = 333,248 B.
        report = inspect_model(DIGITS / 'mlp.onnx')
        with tempfile.TemporaryDirectory() as d:
            import_model(DIGITS / 'mlp.onnx', DIGITS / 'mlp_test_inputs.npz', Path(d) / 'imp')
            for memory, feasible in ((333248, True), (333247, False)):
                p = copy.deepcopy(A.load('wide-array'))
                p['accelerator']['memory_bytes'] = memory
                model = A.schedule(report, p, 'AAAAA')['feasible_under_assumptions']
                run = subprocess.run([str(BINARY), 'run', '--policy', 'maximal', '--graph', str(Path(d) / 'imp/graph.plg')]
                                     + A.execution_flags(p), capture_output=True, text=True)
                self.assertEqual(model, feasible, memory)
                self.assertEqual(run.returncode == 0, feasible, run.stdout + run.stderr)

    def test_trained_cnn(self):
        self.check(DIGITS / 'cnn.onnx', DIGITS / 'cnn_test_chunk0.npz', 'wide-array')

    def test_depthwise_cnn_under_grouped_conv_rule(self):
        self.check(EDGE / 'edge-depthwise-16.onnx', EDGE / 'edge-depthwise-16.inputs.npz', 'dense-conv-array')


@unittest.skipUnless(BINARY is not None or '--binary' in sys.argv, 'engine binary not supplied')
class CapstoneCli(unittest.TestCase):
    """Documented CLI commands reach the authoritative modules and exit honestly."""

    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'scripts/capstone.py'), *map(str, args)],
                              capture_output=True, text=True, timeout=600)

    def test_estimate_compare_and_execute(self):
        p = self.cli('targets')
        self.assertEqual(p.returncode, 0)
        self.assertIn('dense-conv-array', p.stdout)
        p = self.cli('estimate', DIGITS / 'mlp.onnx', '--target', 'wide-array', '--json')
        self.assertEqual(p.returncode, 0, p.stderr)
        direct = A.analyze(DIGITS / 'mlp.onnx', A.load('wide-array'))
        self.assertEqual(json.loads(p.stdout)['analysis']['selected']['placement'], direct['analysis']['selected']['placement'])
        p = self.cli('compare-models', EDGE / 'edge-standard-16.onnx', EDGE / 'edge-depthwise-16.onnx', '--target', 'wide-array')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('accuracy: not evaluated', p.stdout)
        with tempfile.TemporaryDirectory() as d:
            p = self.cli('execute', DIGITS / 'mlp.onnx', '--inputs', DIGITS / 'mlp_test_inputs.npz', '--target',
                         'fragmented-array', '--out', Path(d) / 'run', '--binary', BINARY, '--json')
            self.assertEqual(p.returncode, 0, p.stderr)
            result = json.loads(p.stdout)
            self.assertTrue(result['all_match_original_onnx'])
            self.assertTrue(result['all_movement_agrees_with_model'])
            self.assertTrue(any('ReLU' in f['reason'] for f in result['cpu_fallback_by_declared_rules']))
            p = self.cli('execute', DIGITS / 'mlp.onnx', '--inputs', DIGITS / 'mlp_test_inputs.npz', '--target',
                         'wide-array', '--out', Path(d) / 'run', '--binary', BINARY)
            self.assertEqual(p.returncode, 1)  # existing evidence directory is never overwritten

    def test_unsupported_and_analysis_only_paths(self):
        p = self.cli('inspect', EDGE / 'edge-standard-160.onnx')
        self.assertEqual(p.returncode, 2)
        p = self.cli('inspect', EDGE / 'edge-standard-160.onnx', '--analysis-scope')
        self.assertEqual(p.returncode, 0)
        self.assertIn('execution limit', p.stdout)
        p = self.cli('estimate', DIGITS / 'mlp.onnx', '--target', 'no-such-target')
        self.assertEqual(p.returncode, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path)
    known, rest = parser.parse_known_args()
    BINARY = known.binary
    unittest.main(argv=[sys.argv[0]] + rest)
