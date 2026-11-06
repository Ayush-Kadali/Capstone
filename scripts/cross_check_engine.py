#!/usr/bin/env python3
"""Compare the analytical model with the C++ engine on every executed placement.

For each model x architecture profile x memory policy, run `placementlens verify`
with the profile's engine flags, then check per executed placement that
H2D bytes, D2H bytes, synchronizations and peak device payload are equal, that
per-node support decisions are equal, and that the engine result is correct.
Exit 1 on any mismatch. Usage: cross_check_engine.py --binary build/placementlens --out report.json
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def skip_graph(directory):
    """Y = relu(relu(X@W)) + X@W: the case that exposed the sync/lifetime defect."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.eye(4, dtype=np.float32), 'W')
    g = helper.make_graph([helper.make_node('MatMul', ['X', 'W'], ['A'], name='mm'),
                           helper.make_node('Relu', ['A'], ['B'], name='r1'),
                           helper.make_node('Relu', ['B'], ['C'], name='r2'),
                           helper.make_node('Add', ['C', 'A'], ['Y'], name='skip')], 'skip',
                          [helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 4])],
                          [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 4])], [w])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]), directory / 'skip.onnx')
    np.savez(directory / 'skip.npz', X=np.arange(8, dtype=np.float32).reshape(2, 4) - 3)
    return directory / 'skip.onnx', directory / 'skip.npz'


def main():
    import architecture_model as A
    from onnx_frontend import import_model, inspect_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp())
    edge, digits = ROOT / 'examples/models/edge-cnn', ROOT / 'examples/models/digits'
    cases = [skip_graph(work), (digits / 'mlp.onnx', digits / 'mlp_test_inputs.npz'),
             (digits / 'cnn.onnx', digits / 'cnn_test_chunk0.npz')]
    cases += [(edge / f'edge-{v}-16.onnx', edge / f'edge-{v}-16.inputs.npz') for v in ('standard', 'depthwise', 'residual')]
    rows, total, mismatches = [], 0, 0
    for model, inputs in cases:
        report = inspect_model(model)
        shapes = {x['name']: x['shape'] for x in report['normalized_tensors']}
        search = ['--search', 'bounded'] if len(report['normalized_nodes']) > 12 else []
        for profile_id in ('wide-array', 'dense-conv-array', 'fragmented-array'):
            profile = A.load(profile_id)
            for policy in ('retain', 'release'):
                run = Path(tempfile.mkdtemp(dir=work))
                import_model(model, inputs, run / 'import')
                subprocess.run([str(args.binary), 'verify', '--graph', str(run / 'import/graph.plg'), '--json', str(run / 'core.json')]
                               + search + A.execution_flags(profile, policy), check=True, capture_output=True)
                core = json.loads((run / 'core.json').read_text())
                support_equal = ({c['node_id']: c['device_supported'] for c in core['capabilities']}
                                 == {n['generated_node']: not A.support(n, shapes, profile) for n in report['normalized_nodes']})
                bad = []
                for r in core['results']:
                    s = A.schedule(report, profile, r['placement'].replace('D', 'A'), memory_policy=policy)
                    workspace = profile['accelerator']['workspace_bytes'] if 'A' in s['placement'] else 0
                    engine = (r['h2d_bytes'], r['d2h_bytes'], r['synchronizations'], r['peak_live_device_bytes'])
                    model_ = (s['h2d_bytes'], s['d2h_bytes'], s['syncs'], s['device_peak_bytes'] - workspace)
                    if engine != model_ or not r['correct']:
                        bad.append({'placement': r['placement'], 'engine': engine, 'model': model_, 'correct': r['correct']})
                total += len(core['results'])
                mismatches += len(bad) + (0 if support_equal else 1)
                rows.append({'model': model.name, 'profile': profile_id, 'memory_policy': policy,
                             'placements': len(core['results']), 'support_decisions_equal': support_equal, 'mismatches': bad})
                print(f"{model.name:<24} {profile_id:<17} {policy:<8} {len(core['results']):>4} placements, "
                      f"{len(bad)} mismatches, support equal: {support_equal}")
    args.out.write_text(json.dumps({'compared': ['h2d_bytes', 'd2h_bytes', 'synchronizations', 'peak device payload',
                                                 'support decisions', 'engine correctness'],
                                    'placement_checks': total, 'mismatches': mismatches, 'cases': rows}, indent=2) + '\n')
    print(f'{total} placement checks, {mismatches} mismatches')
    return 1 if mismatches else 0


if __name__ == '__main__':
    sys.exit(main())
