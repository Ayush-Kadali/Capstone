#!/usr/bin/env python3
"""Independent end-to-end/model-reference gates for the ONNX experiment wrapper."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

for dependency in ('onnx', 'onnxruntime', 'numpy'):
    if not importlib.util.find_spec(dependency):
        print('SKIP: model explorer tests require ' + dependency)
        sys.exit(77)
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as H, numpy_helper as N, TensorProto as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import explore_model as E
import study
import bounded_study
from make_onnx_examples import create_examples

BINARY = None


class OutputContracts(unittest.TestCase):
    def test_logical_rank_four_reference_is_compared_without_losing_shape(self):
        reference = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
        result = [{'placement': 'A', 'correct': True, 'output': list(range(12))}]
        comparisons = E.compare_outputs(result, reference, [1, 2, 2, 3])
        self.assertTrue(comparisons[0]['original_onnx_match'])
        with self.assertRaises(ValueError):
            E.compare_outputs(result, reference, [1, 2, 3, 2])

    def test_empty_results_rejected(self):
        with self.assertRaises(ValueError):
            E.compare_outputs([], np.ones((2, 2), np.float32))

    def test_invalid_reference_rejected(self):
        result = [{'placement': 'C', 'correct': True, 'output': [1, 1, 1, 1]}]
        for reference in (np.ones((2, 2), np.float64), np.ones(4, np.float32),
                          np.full((2, 2), np.nan, np.float32), np.full((2, 2), np.inf, np.float32)):
            with self.subTest(dtype=str(reference.dtype), shape=reference.shape):
                with self.assertRaises(ValueError):
                    E.compare_outputs(result, reference)

    def test_mismatch_missing_values_nonfinite_and_core_failure(self):
        reference = np.array([[1, 2], [3, 4]], np.float32)
        for output in ([1, 2, 3, 5], [1, 2, 3], [1, 2, 3, float('inf')], [[1, 2], [3, 4]]):
            with self.subTest(output=output):
                checks = E.compare_outputs([{'placement': 'C', 'correct': True, 'output': output}], reference)
                self.assertFalse(checks[0]['original_onnx_match'])
        checks = E.compare_outputs([{'placement': 'C', 'correct': False, 'output': [1, 2, 3, 4]}], reference)
        self.assertTrue(checks[0]['original_onnx_match'])
        self.assertFalse(checks[0]['core_reference_match'])


class ExplorerIntegration(unittest.TestCase):
    def setUp(self):
        if BINARY is None:
            self.skipTest('pass --binary for actual C++ integration')
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        create_examples(self.root / 'examples')
        self.model = self.root / 'examples/supported_dense.onnx'
        self.inputs = self.root / 'examples/supported_dense.inputs.npz'
        self.profile = ROOT / 'profiles/software-device-relu-on.json'
        self.out = self.root / 'experiment'

    def tearDown(self):
        if hasattr(self, 'temp'):
            self.temp.cleanup()

    def explore(self, measured=False):
        return E.explore(self.model, self.inputs, self.profile, BINARY, self.out, measured)

    def alter_reference(self, transform):
        original = ort.InferenceSession
        class ChangedSession:
            def __init__(self, *args, **kwargs):
                self.real = original(*args, **kwargs)
            def run(self, *args, **kwargs):
                return [transform(x) for x in self.real.run(*args, **kwargs)]
            def get_outputs(self):
                return self.real.get_outputs()
            def get_providers(self):
                return self.real.get_providers()
        return mock.patch.object(ort, 'InferenceSession', ChangedSession)

    def test_real_original_onnx_vs_all_core_placements(self):
        result = self.explore()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['placements_checked'], 8)
        self.assertTrue(result['numerical_execution_validated'])
        self.assertFalse(result['hardware_validated'])
        self.assertFalse(result['vendor_compilation_performed'])
        self.assertFalse(result['performance_conclusions_valid'])
        self.assertTrue(all(x['original_onnx_match'] and x['core_reference_match'] for x in result['comparisons']))
        self.assertEqual((self.out / 'model.onnx').read_bytes(), self.model.read_bytes())
        self.assertEqual((self.out / 'inputs.npz').read_bytes(), self.inputs.read_bytes())

    def test_gemm_target_diagnostics_follow_every_normalized_operation(self):
        x = np.array([[1, -2, 3], [4, 5, -6]], np.float32)
        w = np.array([[1, 2, 3], [-2, 1, 4]], np.float32)
        graph = H.make_graph([H.make_node('Gemm', ['x', 'w', 'b'], ['y'], name='linear', transB=1)],
                            'linear', [H.make_tensor_value_info('x', T.FLOAT, [2, 3])],
                            [H.make_tensor_value_info('y', T.FLOAT, [2, 2])],
                            [N.from_array(w, 'w'), N.from_array(np.array([.5, -.5], np.float32), 'b')])
        onnx.save(H.make_model(graph, opset_imports=[H.make_opsetid('', 13)], ir_version=10), self.model)
        np.savez(self.inputs, x=x)
        result = self.explore()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['placements_checked'], 4)
        analysis = json.loads((self.out / 'target-analysis.json').read_text())
        self.assertEqual(len(analysis['nodes']), 2)
        self.assertEqual({n['operator'] for n in analysis['nodes']}, {'MatMul', 'Add'})
        self.assertTrue(all(n['source_name'] == 'linear' and n['source_index'] == 0 for n in analysis['nodes']))
        self.assertTrue(all(n['accepted'] for n in analysis['nodes']))

    def test_rank_four_conv_original_reference_and_cpu_fallback(self):
        x = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        graph = H.make_graph([H.make_node('Conv', ['x', 'w'], ['y'], name='conv')], 'cnn',
                            [H.make_tensor_value_info('x', T.FLOAT, [1, 1, 3, 3])],
                            [H.make_tensor_value_info('y', T.FLOAT, [1, 1, 2, 2])],
                            [N.from_array(np.ones((1, 1, 2, 2), np.float32), 'w')])
        onnx.save(H.make_model(graph, opset_imports=[H.make_opsetid('', 13)], ir_version=10), self.model)
        np.savez(self.inputs, x=x)
        result = self.explore()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['placements_checked'], 1)
        self.assertEqual(result['comparisons'][0]['placement'], 'C')
        analysis = json.loads((self.out / 'target-analysis.json').read_text())
        self.assertFalse(analysis['nodes'][0]['accepted'])
        self.assertEqual(analysis['nodes'][0]['operator'], 'Conv2D')
        with np.load(self.out / 'onnx-reference.npz', allow_pickle=False) as archive:
            self.assertEqual(archive[archive.files[0]].shape, (1, 1, 2, 2))

    def deep_model(self, relu_only=False):
        nodes = []
        for index in range(29):
            previous = 'x' if index == 0 else f'h{index-1}'
            output = 'y' if index == 28 else f'h{index}'
            nodes.append(H.make_node('Relu' if relu_only else 'Add',
                                     [previous] if relu_only else [previous, 'b'], [output]))
        graph = H.make_graph(nodes, 'deep', [H.make_tensor_value_info('x', T.FLOAT, [2, 2])],
                            [H.make_tensor_value_info('y', T.FLOAT, [2, 2])],
                            [] if relu_only else [N.from_array(np.ones((2, 2), np.float32), 'b')])
        onnx.save(H.make_model(graph, opset_imports=[H.make_opsetid('', 13)], ir_version=10), self.model)
        np.savez(self.inputs, x=np.array([[1, -2], [3, 4]], np.float32))

    def test_large_model_uses_bounded_study_and_original_reference(self):
        self.deep_model()
        result = self.explore(measured=True)
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['search_scope'], 'bounded candidates; no global oracle')
        self.assertLessEqual(result['placements_checked'], 10)
        self.assertTrue(result['performance_conclusions_valid'])
        bundle = bounded_study.validate_bundle(json.loads((self.out / result['study_bundle']).read_text()))
        self.assertEqual(bundle['artifact_type'], 'bounded-placement-study')
        self.assertFalse(bundle['exhaustive'])
        self.assertFalse(bundle['oracle_available'])
        self.assertFalse(bundle['global_regret_available'])
        self.assertNotIn('oracle', bundle['policies'])
        reference = np.array([[30, 27], [32, 33]], np.float32)
        for phase, count in [('warmup', 1), ('calibration', 3), ('evaluation', 5)]:
            for index in range(count):
                raw = json.loads((self.out / 'study' / f'{phase}-{index:03d}.json').read_text())
                comparisons = E.compare_outputs(raw['results'], reference, [2, 2])
                self.assertTrue(all(row['original_onnx_match'] for row in comparisons))

    def test_large_model_unsupported_device_ops_fall_back_to_cpu(self):
        self.deep_model(relu_only=True)
        self.profile = ROOT / 'profiles/software-device-relu-off.json'
        result = self.explore()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['placements_checked'], 1)
        self.assertEqual(result['comparisons'][0]['placement'], 'C' * 29)
        raw = json.loads((self.out / 'core.json').read_text())
        self.assertFalse(raw['search']['exhaustive'])
        self.assertTrue(raw['search']['maximal_feasible'])
        self.assertEqual(raw['search']['maximal_placement'], 'C' * 29)

    def test_large_measured_external_reference_failure_is_quarantined(self):
        self.deep_model()
        original = bounded_study.run_study
        def changed_run(*args, **kwargs):
            bundle = original(*args, **kwargs)
            report = Path(args[2]) / 'evaluation-000.json'
            raw = json.loads(report.read_text())
            raw['results'][0]['output'][0] += 100
            report.write_text(json.dumps(raw))
            return bundle
        with mock.patch.object(bounded_study, 'run_study', changed_run):
            with self.assertRaises(ValueError):
                self.explore(measured=True)
        result = json.loads((self.out / 'result.json').read_text())
        self.assertEqual(result['status'], 'error')
        self.assertFalse(result['performance_conclusions_valid'])
        self.assertFalse((self.out / 'study/bundle.json').exists())
        pending = json.loads((self.out / 'study/bundle.pending-external-reference.json').read_text())
        self.assertFalse(pending['conclusions_valid'])
        self.assertFalse(pending['external_reference_validated'])

    def test_real_measured_study_and_original_reference(self):
        result = self.explore(measured=True)
        self.assertEqual(result['status'], 'verified')
        self.assertTrue(result['performance_conclusions_valid'])
        bundle = study.validate_bundle(json.loads((self.out / result['study_bundle']).read_text()))
        self.assertTrue(bundle['conclusions_valid'])
        runs = [p for p in (self.out / 'study').glob('*.json')
                if p.name.startswith(('warmup-', 'calibration-', 'evaluation-'))]
        self.assertEqual(len(runs), 9)
        with np.load(self.out / 'onnx-reference.npz', allow_pickle=False) as archive:
            reference = archive[archive.files[0]]
        for path in runs:
            comparisons = E.compare_outputs(json.loads(path.read_text())['results'], reference)
            self.assertEqual(len(comparisons), 8)
            self.assertTrue(all(x['original_onnx_match'] for x in comparisons))

    def test_unsupported_model_produces_diagnostic_without_core_execution(self):
        self.model = self.root / 'runtime-broadcast.onnx'
        self.inputs = self.root / 'runtime-broadcast.npz'
        graph = H.make_graph([H.make_node('Add', ['x', 'bias'], ['y'])], 'runtime-broadcast',
                            [H.make_tensor_value_info('x', T.FLOAT, [2, 2]),
                             H.make_tensor_value_info('bias', T.FLOAT, [1, 2])],
                            [H.make_tensor_value_info('y', T.FLOAT, [2, 2])])
        onnx.save(H.make_model(graph, opset_imports=[H.make_opsetid('', 13)], ir_version=10), self.model)
        np.savez(self.inputs, x=np.ones((2, 2), np.float32), bias=np.ones((1, 2), np.float32))
        with mock.patch.object(E.subprocess, 'run', side_effect=AssertionError('should not execute')):
            result = self.explore()
        self.assertEqual(result['status'], 'unsupported_by_importer')
        self.assertFalse(result['performance_conclusions_valid'])
        self.assertFalse((self.out / 'core.json').exists())
        self.assertFalse((self.out / 'import').exists())
        self.assertIn('ADD_BROADCAST', [x['code'] for x in result['issues']])

    def test_existing_output_preserved(self):
        self.out.mkdir()
        marker = self.out / 'result.json'
        marker.write_text('existing evidence')
        with self.assertRaises(FileExistsError):
            self.explore()
        self.assertEqual(marker.read_text(), 'existing evidence')

    def test_invalid_profile_rejected_before_output_creation(self):
        bad = json.loads(self.profile.read_text())
        bad['evidence']['hardware_validated'] = True
        self.profile = self.root / 'bad-profile.json'
        self.profile.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            self.explore()
        self.assertFalse(self.out.exists())

    def test_original_reference_mismatch_suppresses_measurement(self):
        with self.alter_reference(lambda x: x + np.float32(1)), \
                mock.patch.object(study, 'run_study', side_effect=AssertionError('must not measure mismatch')):
            result = self.explore(measured=True)
        self.assertEqual(result['status'], 'numerical_mismatch')
        self.assertFalse(result['numerical_execution_validated'])
        self.assertFalse(result['performance_conclusions_valid'])
        self.assertFalse((self.out / 'study').exists())

    def test_reference_shape_with_same_element_count_rejected(self):
        with self.alter_reference(lambda x: x.reshape(1, 4)):
            with self.assertRaises(ValueError):
                self.explore()
        result = json.loads((self.out / 'result.json').read_text())
        self.assertEqual(result['status'], 'error')
        self.assertFalse(result['performance_conclusions_valid'])

    def test_nonfinite_reference_rejected(self):
        with self.alter_reference(lambda x: np.full_like(x, np.nan)):
            with self.assertRaises(ValueError):
                self.explore()
        self.assertFalse(json.loads((self.out / 'result.json').read_text())['performance_conclusions_valid'])

    def test_measured_original_reference_failure_never_publishes_valid_bundle(self):
        original = study.run_study
        def changed_run(*args, **kwargs):
            bundle = original(*args, **kwargs)
            directory = Path(args[2])
            report = directory / 'evaluation-000.json'
            raw = json.loads(report.read_text())
            raw['results'][0]['output'][0] += 100
            report.write_text(json.dumps(raw))
            return bundle
        with mock.patch.object(study, 'run_study', changed_run):
            with self.assertRaises(ValueError):
                self.explore(measured=True)
        result = json.loads((self.out / 'result.json').read_text())
        self.assertEqual(result['status'], 'error')
        self.assertFalse(result['performance_conclusions_valid'])
        bundle = self.out / 'study/bundle.json'
        self.assertFalse(bundle.exists())
        pending = self.out / 'study/bundle.pending-external-reference.json'
        self.assertTrue(pending.exists())
        quarantined = json.loads(pending.read_text())
        self.assertFalse(quarantined['conclusions_valid'])
        self.assertFalse(quarantined['external_reference_validated'])

    def test_missing_measured_raw_report_prevents_external_validation(self):
        original = study.run_study
        def incomplete_run(*args, **kwargs):
            bundle = original(*args, **kwargs)
            (Path(args[2]) / 'evaluation-000.json').unlink()
            return bundle
        with mock.patch.object(study, 'run_study', incomplete_run):
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.explore(measured=True)
        self.assertFalse(json.loads((self.out / 'result.json').read_text())['performance_conclusions_valid'])
        self.assertFalse((self.out / 'study/bundle.json').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve() if args.binary else None
    unittest.main(argv=[sys.argv[0]] + remaining)
