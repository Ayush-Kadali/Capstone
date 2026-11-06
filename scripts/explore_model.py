#!/usr/bin/env python3
"""Run a bounded ONNX model through the capstone and an independent ONNX runtime.

This first integration uses hypothetical software profiles. It does not establish
vendor compile acceptance or real accelerator behavior. Output is never replaced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from onnx_frontend import import_model, inspect_model, MAX_FILE_BYTES
from target_profile import load_profile, engine_flags


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def seal_evidence(out, receipt):
    """Bind explanations to these saved bytes, not an unrelated valid study."""
    names = ['model.onnx', 'inputs.npz', 'profile.json', 'compatibility.json',
             'import/import.json', 'import/graph.plg', 'onnx-reference.npz',
             'core.json', 'target-analysis.json']
    if receipt.get('performance_conclusions_valid') is True:
        names.append('study/bundle.json')
    receipt['evidence_sha256'] = {name: sha(out/name) for name in names if (out/name).is_file()}


def copy_bounded(source, destination):
    with Path(source).open('rb') as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('Model/input file exceeds 16 MiB')
    Path(destination).write_bytes(raw)


def compare_outputs(results, reference, expected_shape=None):
    """Independent ONNX output is authoritative; reject missing/nonfinite results."""
    import numpy as np
    ref = np.asarray(reference)
    if ref.dtype != np.float32 or ref.ndim not in (2, 4) or not np.isfinite(ref).all():
        raise ValueError('ONNX reference must be finite rank-2 or rank-4 float32')
    if expected_shape is not None and list(ref.shape) != list(expected_shape):
        raise ValueError('ONNX reference shape does not match the imported model output')
    if not results:
        raise ValueError('No core placements were executed')
    comparisons = []
    for result in results:
        values = np.asarray(result['output'], dtype=np.float64)
        expected = ref.reshape(-1).astype(np.float64)
        same_size = values.shape == expected.shape
        finite = bool(np.isfinite(values).all())
        error = np.abs(values - expected) if same_size and finite else None
        matches = bool(error is not None and np.all(error <= 1e-5 + 1e-5 * np.abs(expected)))
        comparisons.append({'placement': result['placement'],
                            'core_reference_match': result['correct'] is True,
                            'original_onnx_match': matches,
                            'max_abs_error': float(error.max()) if error is not None else None})
    return comparisons


def explore(model, inputs, profile_path, binary, out, measured=False):
    # Resolve optional dependencies before creating a partially populated experiment.
    import numpy as np
    import onnxruntime as ort
    profile = load_profile(profile_path)
    flags = engine_flags(profile)
    binary = Path(binary).resolve(strict=True)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'started', 'hardware_validated': False,
               'vendor_compilation_performed': False, 'accuracy_evaluated': False,
               'execution_scope': 'host software with hypothetical device profile',
               'platform': platform.platform(), 'python': sys.version,
               'onnxruntime_version': ort.__version__, 'numpy_version': np.__version__,
               'binary_sha256': sha(binary), 'profile': profile, 'core_flags': flags,
               'atol': 1e-5, 'rtol': 1e-5, 'performance_conclusions_valid': False}
    try:
        copy_bounded(model, out/'model.onnx')
        copy_bounded(inputs, out/'inputs.npz')
        receipt.update(model_sha256=sha(out/'model.onnx'), inputs_sha256=sha(out/'inputs.npz'))
        save(out/'profile.json', profile)
        inspection = inspect_model(out/'model.onnx')
        save(out/'compatibility.json', inspection)
        if not inspection['supported']:
            receipt['status'] = 'unsupported_by_importer'
            receipt['issues'] = inspection['issues']
            seal_evidence(out, receipt)
            save(out/'result.json', receipt)
            return receipt
        imported = import_model(out/'model.onnx', out/'inputs.npz', out/'import')
        receipt['import_manifest'] = 'import/import.json'
        receipt['graph_sha256'] = imported['graph_sha256']
        receipt['tool_source_sha256'] = {name: sha(Path(__file__).parent/name)
                                        for name in ('explore_model.py', 'onnx_frontend.py', 'target_profile.py', 'study.py', 'bounded_study.py')}
        # ORT consumes the original ONNX snapshot, never our generated graph.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(out/'model.onnx'), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        with np.load(out/'inputs.npz', allow_pickle=False) as archive:
            feeds = {name: archive[name] for name in archive.files}
        reference = session.run(None, feeds)
        if len(reference) != 1:
            raise ValueError('Reference output count violates importer contract')
        expected_shape = imported['outputs'][0]['shape']
        if list(reference[0].shape) != expected_shape:
            raise ValueError('ONNX reference shape does not match the imported model output')
        np.savez(out/'onnx-reference.npz', **{session.get_outputs()[0].name: reference[0]})
        receipt['reference_provider'] = session.get_providers()
        bounded = len(imported['normalized_nodes']) > 12
        receipt['search_scope'] = 'bounded candidates; no global oracle' if bounded else 'exhaustive feasible placements'
        search_flags = ['--search', 'bounded', '--search-budget', '128', '--beam-width', '8'] if bounded else []
        command = [str(binary), 'verify', '--graph', str(out/'import/graph.plg'),
                   *flags, *search_flags, '--json', str(out/'core.json')]
        receipt['command'] = command
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (out/'core.log').write_text(process.stdout + process.stderr)
        receipt['core_exit'] = process.returncode
        if process.returncode not in (0, 2) or not (out/'core.json').is_file():
            raise RuntimeError('Core verification failed; see core.log')
        report = json.loads((out/'core.json').read_text())
        if bounded:
            from bounded_study import validate_report
        else:
            from study import validate_report
        validate_report(report)
        # The core is authoritative for target capability, including CNN attributes.
        capabilities = {node['node_id']: node for node in report['capabilities']}
        node_analysis = []
        for node in imported['normalized_nodes']:
            capability = capabilities[node['generated_node']]
            node_analysis.append({**node, 'accepted': capability['device_supported'],
                                  'reason': capability['reason'], 'evidence_level': 'hypothetical',
                                  'hardware_validated': False,
                                  'cpu_fallback_available': not capability['device_supported']})
        save(out/'target-analysis.json', {'schema_version': 2, 'nodes': node_analysis,
             'profile_id': profile['id'], 'hardware_validated': False,
             'scope': 'Authoritative per-node core capability; whole-plan memory is checked by the core'})
        receipt['target_analysis'] = 'target-analysis.json'
        comparisons = compare_outputs(report['results'], reference[0], expected_shape)
        receipt['comparisons'] = comparisons
        receipt['placements_checked'] = len(comparisons)
        passed = process.returncode == 0 and all(x['core_reference_match'] and x['original_onnx_match'] for x in comparisons)
        receipt['status'] = 'verified' if passed else 'numerical_mismatch'
        receipt['numerical_execution_validated'] = passed
        # Study samples are checked against original ONNX as well, not just the IR.
        if measured and passed:
            if bounded:
                from bounded_study import run_study, validate_bundle
            else:
                from study import run_study, validate_bundle
            search_options = {"search_budget": 128, "beam_width": 8} if bounded else {}
            bundle = run_study(binary, out/'import/graph.plg', out/'study', flags=flags,
                               warmup=1, calibration=3, samples=5, **search_options)
            pending = out/'study/bundle.pending-external-reference.json'
            (out/'study/bundle.json').rename(pending)
            validate_bundle(bundle)
            if not bundle['conclusions_valid']:
                raise ValueError('Measured study has invalid core correctness conclusions')
            for phase, count in (('warmup', 1), ('calibration', 3), ('evaluation', 5)):
                for index in range(count):
                    path = out/'study'/f'{phase}-{index:03d}.json'
                    raw = validate_report(json.loads(path.read_text()))
                    if not all(x['original_onnx_match'] and x['core_reference_match']
                               for x in compare_outputs(raw['results'], reference[0], expected_shape)):
                        raise ValueError('Measured study disagrees with original ONNX: '+path.name)
            pending.rename(out/'study/bundle.json')
            receipt['study_bundle'] = 'study/bundle.json'
            receipt['performance_conclusions_valid'] = True
            receipt['performance_scope'] = 'Measured host software only; no hardware speedup inference'
        seal_evidence(out, receipt)
        save(out/'result.json', receipt)
        return receipt
    except Exception as error:
        receipt['status'] = 'error'
        receipt['performance_conclusions_valid'] = False
        receipt['error'] = str(error)
        # Preserve raw measurements but never leave a publishable study artifact
        # whose numerical validity was checked only against our translated IR.
        for candidate in (out/'study/bundle.json', out/'study/bundle.pending-external-reference.json'):
            if candidate.is_file():
                pending = out/'study/bundle.pending-external-reference.json'
                unvalidated = json.loads(candidate.read_text())
                unvalidated.update(conclusions_valid=False, external_reference_validated=False,
                                   external_reference_error=str(error))
                save(pending, unvalidated)
                if candidate != pending:
                    candidate.unlink()
        save(out/'result.json', receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('model', 'inputs', 'profile', 'binary', 'out'):
        parser.add_argument('--'+arg, required=True, type=Path)
    parser.add_argument('--study', action='store_true', help='separate calibration/evaluation with original-ONNX checks')
    args = parser.parse_args()
    try:
        result = explore(args.model, args.inputs, args.profile, args.binary, args.out, args.study)
        print(json.dumps({'status': result['status'], 'placements_checked': result.get('placements_checked', 0),
                          'result': str(args.out/'result.json'), 'hardware_validated': False}))
        return 0 if result['status'] == 'verified' else 2
    except Exception as error:
        print('MODEL_EXPLORER: '+str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
