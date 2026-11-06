#!/usr/bin/env python3
"""One CLI for the model-to-accelerator exploration system (PlacementLens is the C++ compiler/runtime).

Typical route:  build -> targets -> inspect -> estimate -> execute -> compare-* -> explain/demo.
No web server is required. Physical-board support is never inferred from simulator results.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / 'build/placementlens'


def target_path(value):
    from target_profile import load_profile
    candidate = ROOT/'profiles'/f'{value}.json'
    if not candidate.is_file():
        candidate = Path(value).expanduser()
    if not candidate.is_file():
        raise ValueError('Unknown target. Run targets to list hypothetical profiles; no physical accelerator profile is validated yet.')
    load_profile(candidate)
    return candidate


def write_outputs(out, report, text, stem='estimate'):
    if out:
        out.mkdir(parents=True, exist_ok=False)
        (out/f'{stem}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        (out/f'{stem}.txt').write_text(text)


def execute(model, inputs, target, binary, out, runtime='serial', memory_policy='retain'):
    """Run the real engine under an architecture profile's rules and check against ONNX Runtime.

    Exact: placements, commands, transfer bytes, syncs, correctness. Modeled: the
    analytical ms for the same placement. Measured: host software time (not hardware).
    """
    import numpy as np
    import onnxruntime as ort
    import architecture_model as A
    from explore_model import compare_outputs
    from onnx_frontend import import_model, inspect_model
    profile = A.load(target)
    binary = Path(binary).resolve(strict=True)
    out.mkdir(parents=True, exist_ok=False)
    try:  # a failed import/validation removes --out so the corrected command can be rerun
        inspection = inspect_model(model)
        (out/'compatibility.json').write_text(json.dumps(inspection, indent=2)+'\n')
        if not inspection['supported']:
            return {'status': 'unsupported_by_importer', 'issues': inspection['issues'], 'model': str(model)}
        imported = import_model(model, inputs, out/'import')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        with np.load(inputs, allow_pickle=False) as archive:
            reference = session.run(None, {k: archive[k] for k in archive.files})[0]
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    search = ['--search', 'bounded'] if len(imported['normalized_nodes']) > 12 else []
    command = [str(binary), 'verify', '--graph', str(out/'import/graph.plg'), '--json', str(out/'core.json'),
               *search, *A.execution_flags(profile, memory_policy)]
    if runtime != 'serial':
        command += ['--runtime', runtime]
    process = subprocess.run(command, capture_output=True, text=True, timeout=300)
    (out/'core.log').write_text(' '.join(command) + '\n' + process.stdout + process.stderr)
    if process.returncode not in (0, 2):
        raise RuntimeError('engine failed; see ' + str(out/'core.log'))
    core = json.loads((out/'core.json').read_text())
    checks = {c['placement']: c for c in compare_outputs(core['results'], reference, imported['outputs'][0]['shape'])}
    rows = []
    for r in core['results']:
        placement = r['placement'].replace('D', 'A')
        modeled = A.schedule(inspection, profile, placement, memory_policy=memory_policy)
        rows.append({'placement': placement, 'policy': r['policy'],
                     'matches_original_onnx': checks[r['placement']]['original_onnx_match'],
                     'max_abs_error_vs_onnx': checks[r['placement']]['max_abs_error'],
                     'exact': {'h2d_bytes': r['h2d_bytes'], 'd2h_bytes': r['d2h_bytes'], 'syncs': r['synchronizations'],
                               'commands': len(r['commands']), 'islands': r['islands']},
                     'modeled': {'estimated_ms': modeled['estimated_ms'],
                                 'movement_agrees_with_engine': (modeled['h2d_bytes'], modeled['d2h_bytes'], modeled['syncs'],
                                                                 modeled['device_peak_bytes'] - (profile['accelerator']['workspace_bytes'] if 'A' in placement else 0))
                                 == (r['h2d_bytes'], r['d2h_bytes'], r['synchronizations'], r['peak_live_device_bytes'])},
                     'measured_host_software_ms': r['host_execution_ms'], 'execution_backend': r['execution_backend']})
    fallback = [{'node': c['node_id'], 'reason': c['reason']} for c in core['capabilities'] if not c['device_supported']]
    result = {'schema_version': 1, 'model': str(model), 'target': profile['id'], 'target_revision': profile['revision'],
              'engine_flags': A.execution_flags(profile, memory_policy), 'runtime': runtime, 'memory_policy': memory_policy, 'command': command,
              'search': core.get('search'), 'cpu_fallback_by_declared_rules': fallback, 'placements': rows,
              'all_match_original_onnx': all(r['matches_original_onnx'] for r in rows),
              'all_movement_agrees_with_model': all(r['modeled']['movement_agrees_with_engine'] for r in rows),
              'labels': {'exact': 'engine command stream', 'modeled': 'analytical estimate',
                         'measured_host_software_ms': 'host software timing, not accelerator latency'},
              'hardware_validated': False, 'task_accuracy': 'not evaluated by this command'}
    (out/'execute.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def render_execute(result):
    from developer_report import clean
    if 'placements' not in result:
        return 'Importer rejected the model; nothing executed.\n' + '\n'.join(
            '  ' + clean(f"{i['code']}: {i['message']}") for i in result['issues']) + '\n'
    lines = [f"EXECUTED {clean(result['model'])} under {result['target']} r{result['target_revision']} rules "
             f"({len(result['placements'])} placements, runtime {result['runtime']})"]
    search = result.get('search') or {}
    rejected = search.get('infeasible') or search.get('rejected')
    if rejected:
        lines.append(f"  {rejected} placements rejected by declared support rules or the memory budget (not executed; see execute.json search)")
    for f in result['cpu_fallback_by_declared_rules']:
        lines.append(f"  fallback {f['node']}: {clean(f['reason'])}")
    lines.append('  placement            ONNX-match  H2D B   D2H B  syncs  cmds  modeled ms  host-sw ms')
    for r in sorted(result['placements'], key=lambda r: (r['modeled']['estimated_ms'] or 1e18))[:12]:
        e, m = r['exact'], r['modeled']
        ms = 'infeasible' if m['estimated_ms'] is None else f"{m['estimated_ms']:.6f}"
        lines.append(f"  {r['placement']:<20} {str(r['matches_original_onnx']):<11} {e['h2d_bytes']:<7} {e['d2h_bytes']:<6} "
                     f"{e['syncs']:<6} {e['commands']:<5} {ms:<11} {r['measured_host_software_ms']:.4f}")
    lines += [f"All placements match original ONNX: {result['all_match_original_onnx']}",
              f"Analytical bytes, syncs and peak memory equal engine commands: {result['all_movement_agrees_with_model']}",
              'Exact = engine commands; modeled = analytical; host-sw = software timing, not accelerator latency.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('build', help='configure and build the C++ engine into build/')
    build.add_argument('--build-dir', type=Path, default=ROOT/'build')
    sub.add_parser('targets', help='list execution and architecture profiles with evidence levels')
    sub.add_parser('architecture-targets', help='list analytical architecture profiles only')
    inspect = sub.add_parser('inspect', help='check ONNX import semantics without running kernels')
    inspect.add_argument('model', type=Path)
    inspect.add_argument('--analysis-scope', action='store_true', help='allow models above executor size limits')
    estimate = sub.add_parser('estimate', help='shape-based work, traffic, memory, placement and sensitivity; no execution')
    estimate.add_argument('model', type=Path)
    estimate.add_argument('--target', required=True)
    comparison = sub.add_parser('compare-targets', help='same model on 2..8 architecture profiles; ranking robustness')
    comparison.add_argument('model', type=Path)
    comparison.add_argument('--targets', nargs='+', required=True)
    variants = sub.add_parser('compare-models', help='2..8 model variants on one architecture profile')
    variants.add_argument('models', type=Path, nargs='+')
    variants.add_argument('--target', required=True)
    for command in (estimate, comparison, variants):
        command.add_argument('--resident-weights', action='store_true', help='assume weights were uploaded before the request')
        command.add_argument('--memory-policy', choices=('retain', 'release'), default='retain',
                             help='retain all device allocations (engine default) or release after last use (engine --reuse-buffers)')
        command.add_argument('--json', action='store_true')
        command.add_argument('--out', type=Path, help='fresh directory for JSON and text evidence')
    run = sub.add_parser('execute', help='run the C++ engine under an architecture profile and check against ONNX Runtime')
    run.add_argument('model', type=Path)
    run.add_argument('--inputs', required=True, type=Path, help='NPZ keyed by ONNX input names')
    run.add_argument('--target', required=True, help='architecture profile ID or path')
    run.add_argument('--out', required=True, type=Path)
    run.add_argument('--binary', type=Path, default=DEFAULT_BINARY)
    run.add_argument('--runtime', choices=('serial', 'command-queue'), default='serial', help='serial (default) or command-queue')
    run.add_argument('--memory-policy', choices=('retain', 'release'), default='retain')
    run.add_argument('--json', action='store_true')
    analyze = sub.add_parser('analyze', help='execution-profile study: original ONNX checks, candidates, measured host runs')
    analyze.add_argument('model', type=Path)
    analyze.add_argument('--inputs', required=True, type=Path, help='NPZ keyed by ONNX runtime input names')
    analyze.add_argument('--target', required=True, help='registered execution profile ID or profile JSON path')
    analyze.add_argument('--out', required=True, type=Path, help='fresh directory for evidence and readable reports')
    analyze.add_argument('--binary', type=Path, default=DEFAULT_BINARY)
    analyze.add_argument('--measure', action='store_true', help='separate warmup, calibration and held-out host measurements')
    explain = sub.add_parser('explain', help='read a bound analyze result without reexecuting the model')
    explain.add_argument('run', type=Path)
    for command in (analyze, explain):
        command.add_argument('--details', action='store_true', help='include individual transfers and cost components')
        command.add_argument('--include-host-baseline', action='store_true', help='also show CPU-only; not an optimized laptop benchmark')
        command.add_argument('--json', action='store_true', help='print structured explanation')
    demo = sub.add_parser('demo', help='run the six-step demonstration and save a transcript')
    demo.add_argument('--out', required=True, type=Path)
    demo.add_argument('--binary', type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args(argv)
    try:
        if args.action == 'build':
            subprocess.run(['cmake', '-S', str(ROOT), '-B', str(args.build_dir), '-DCMAKE_BUILD_TYPE=Release',
                            f'-DPython3_EXECUTABLE={sys.executable}'], check=True)
            subprocess.run(['cmake', '--build', str(args.build_dir), '-j', '4'], check=True)
            print(f'Built {args.build_dir/"placementlens"}. Run: ctest --test-dir {args.build_dir} --output-on-failure')
            return 0
        if args.action in ('architecture-targets', 'targets'):
            import architecture_model as A
            print('Architecture profiles (estimate / compare / execute; hardware + compiler rules):')
            for profile in A.registered():
                print(f"  {profile['id']} r{profile['revision']} | {profile['evidence']['level']} | {profile['description']}")
            if args.action == 'targets':
                from target_profile import load_profile
                print('Execution profiles (analyze; older software-device cost settings):')
                for path in sorted((ROOT/'profiles').glob('*.json')):
                    p = load_profile(path)
                    print(f"  {p['id']} | {p['evidence']['level']} | device budget {p['memory']['device_bytes']} B")
            print('\nPhysical GPU/NPU/TPU support and latency: not established by these hypothetical profiles.')
            return 0
        if args.action in ('estimate', 'compare-targets', 'compare-models'):
            import architecture_model as A
            if args.action == 'estimate':
                report = A.analyze(args.model, A.load(args.target), resident_weights=args.resident_weights, memory_policy=args.memory_policy)
                ok = report['analysis'] is not None
            elif args.action == 'compare-targets':
                report = A.compare_targets(args.model, [A.load(t) for t in args.targets], resident_weights=args.resident_weights, memory_policy=args.memory_policy)
                ok = bool(report['nominal_order'])
            else:
                report = A.compare_models(args.models, A.load(args.target), resident_weights=args.resident_weights, memory_policy=args.memory_policy)
                ok = all(r['estimated_ms'] is not None for r in report['rows'])
            text = A.render(report)
            write_outputs(args.out, report, text)
            print(json.dumps(report, indent=2, allow_nan=False) if args.json else text, end='\n' if args.json else '')
            return 0 if ok else 2
        if args.action == 'inspect':
            from onnx_frontend import inspect_model
            from developer_report import clean
            r = inspect_model(args.model, 'analysis' if args.analysis_scope else 'execution')
            print('Importer: '+('SUPPORTED STATIC SUBSET' if r['supported'] else 'UNSUPPORTED'))
            for n in r.get('nodes', []):
                print(f"  #{n['source_index']} {clean(n['source_name'])}: {n['operator']}")
            for issue in r['issues']:
                print('  '+clean(f"{issue['code']}: {issue['message']}"))
            for limit in r.get('execution_limits', []):
                print('  execution limit (analysis only): '+clean(limit['message']))
            print('This checks importer semantics only. Target compilation, execution and performance are not established.')
            return 0 if r['supported'] else 2
        if args.action == 'execute':
            result = execute(args.model, args.inputs, args.target, args.binary, args.out, args.runtime, args.memory_policy)
            print(json.dumps(result, indent=2, allow_nan=False) if args.json else render_execute(result), end='\n' if args.json else '')
            return 0 if result.get('all_match_original_onnx') else 2
        if args.action == 'demo':
            from demo import run_demo
            return run_demo(args.out, args.binary)
        if args.action == 'analyze':
            from explore_model import explore
            print('Checking original ONNX and core candidates'+('; measuring held-out host runs' if args.measure else '')+'…', file=sys.stderr)
            explore(args.model, args.inputs, target_path(args.target), args.binary, args.out, args.measure)
            directory = args.out
        else:
            directory = args.run
        from developer_report import build_report, render
        report = build_report(directory, args.include_host_baseline)
        if args.action == 'analyze':
            (directory/'developer-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            (directory/'developer-report.txt').write_text(render(report, details=True))
        print(json.dumps(report, indent=2, allow_nan=False) if args.json else render(report, args.details), end='\n' if args.json else '')
        return 0 if report['simulator_verdict'] == 'verified_on_supplied_inputs' else 2
    except Exception as error:
        from developer_report import clean
        print('CAPSTONE: '+clean(error), file=sys.stderr)
        print('No execution/performance conclusion is established. For ONNX dependencies see requirements-onnx.txt.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
