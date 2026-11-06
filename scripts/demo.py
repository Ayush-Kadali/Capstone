"""Six-step demonstration. Every runnable claim comes from executing the software now.

1. Import a real (trained) ONNX model.
2. A supported operation becomes CPU fallback under a declared target rule.
3. That fallback changes transfers and synchronizations (exact engine counts).
4. A target change and a model change improve explicitly stated metrics.
5. Memory and support constraints reject plans.
6. Numerical correctness against ONNX Runtime and inspectable evidence files,
   plus the command-level queue runtime (synthetic overlap clearly labelled).

Labels: EXACT = engine command stream / importer output; MODELED = analytical
estimate; MEASURED = host software timing; EVALUATED = held-out dataset accuracy.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIGITS = ROOT / 'examples/models/digits'
EDGE = ROOT / 'examples/models/edge-cnn'


def run_demo(out, binary):
    import architecture_model as A
    from capstone import execute
    from onnx_frontend import inspect_model, import_model
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    binary = Path(binary).resolve(strict=True)
    log = []

    def say(text=''):
        print(text)
        log.append(text)

    # 1 ---------------------------------------------------------------------------------
    say('# Demonstration transcript\n')
    say('## 1. Import a real ONNX model (EXACT)')
    manifest = json.loads((DIGITS / 'manifest.json').read_text())
    report = inspect_model(DIGITS / 'cnn.onnx')
    (out / '1-import.json').write_text(json.dumps(report, indent=2) + '\n')
    say(f"Model examples/models/digits/cnn.onnx: PyTorch-trained digit classifier, opset 13, supported={report['supported']}")
    for n in report['normalized_nodes']:
        say(f"  {n['generated_node']:<11} {n['operator']:<10} from ONNX node {n['source_name']} -> shape {n['logical_shape']}")

    # 2 + 3 -----------------------------------------------------------------------------
    say('\n## 2. A supported operation becomes CPU fallback (EXACT, executed)')
    runs = {}
    for target in ('wide-array', 'fragmented-array'):
        runs[target] = execute(DIGITS / 'cnn.onnx', DIGITS / 'cnn_test_chunk0.npz', target, binary, out / f'2-execute-{target}')
    for f in runs['fragmented-array']['cpu_fallback_by_declared_rules']:
        say(f"  fragmented-array: {f['node']} -> CPU because: {f['reason']}")
    say('  wide-array: no declared-rule fallback' if not runs['wide-array']['cpu_fallback_by_declared_rules'] else '')

    say('\n## 3. Effect on transfers and synchronization (EXACT engine counts, maximal-support plans)')
    rows = {}
    for target, result in runs.items():
        maximal = A.analyze(DIGITS / 'cnn.onnx', A.load(target), with_sensitivity=False)['analysis']['maximal_supported']['placement']
        row = next(r for r in result['placements'] if r['placement'] == maximal)
        rows[target] = row
        e = row['exact']
        say(f"  {target:<17} plan {maximal}: H2D {e['h2d_bytes']} B, D2H {e['d2h_bytes']} B, syncs {e['syncs']}, "
            f"commands {e['commands']}, device islands {e['islands']}")
    w, f = rows['wide-array']['exact'], rows['fragmented-array']['exact']
    say(f"  Fallback adds {f['h2d_bytes'] + f['d2h_bytes'] - w['h2d_bytes'] - w['d2h_bytes']} B of movement "
        f"and {f['syncs'] - w['syncs']} synchronization(s): the ReLU's input comes back to the host, and its output goes out to the device again.")

    # 4 ---------------------------------------------------------------------------------
    say('\n## 4. Changes that improve a stated metric')
    say(f"  Target change (EXACT metric: bytes moved by the maximal plan): fragmented-array -> wide-array saves "
        f"{f['h2d_bytes'] + f['d2h_bytes'] - w['h2d_bytes'] - w['d2h_bytes']} B and {f['syncs'] - w['syncs']} sync(s).")
    models = [EDGE / 'edge-standard-160.onnx', EDGE / 'edge-depthwise-160.onnx']
    cmp = A.compare_models(models, A.load('compact-vector'))
    (out / '4-compare-models.json').write_text(json.dumps(cmp, indent=2) + '\n')
    (out / '4-compare-models.txt').write_text(A.render(cmp))
    a, b = cmp['rows']
    say(f"  Model change (MODELED metric: single-request latency on compact-vector, 160x160 input, analysis-only size):")
    say(f"    standard  {a['estimated_ms']:.4f} ms (range {a['scenario_range_ms'][0]:.4f}..{a['scenario_range_ms'][1]:.4f}), {a['operations']} ops")
    say(f"    depthwise {b['estimated_ms']:.4f} ms (range {b['scenario_range_ms'][0]:.4f}..{b['scenario_range_ms'][1]:.4f}), {b['operations']} ops")
    say('    Task accuracy of these variants: not evaluated (untrained architecture examples).')
    targets = A.compare_targets(EDGE / 'edge-depthwise-64.onnx', [A.load(t) for t in ('compact-vector', 'wide-array', 'dense-conv-array')])
    (out / '4-compare-targets.json').write_text(json.dumps(targets, indent=2) + '\n')
    (out / '4-compare-targets.txt').write_text(A.render(targets))
    say(f"  Target ranking for edge-depthwise-64 (MODELED): {targets['nominal_order_text']}")
    say(f"    {targets['conclusion']}")

    # 5 ---------------------------------------------------------------------------------
    say('\n## 5. Constraints that reject plans')
    with_limit = out / '5-memory'
    import_model(DIGITS / 'mlp.onnx', DIGITS / 'mlp_test_inputs.npz', with_limit / 'import')
    command = [str(binary), 'run', '--policy', 'maximal', '--graph', str(with_limit / 'import/graph.plg'),
               '--relu-supported', '--device-bytes', '200000']
    p = subprocess.run(command, capture_output=True, text=True)
    (with_limit / 'run.log').write_text(' '.join(command) + f'\nexit {p.returncode}\n' + p.stdout + p.stderr)
    reason = [line for line in (p.stdout + p.stderr).splitlines() if 'E_' in line or 'REJECT' in line or 'infeasible' in line.lower()]
    say(f"  EXACT (engine): maximal plan for the 360-image MLP with a 200,000 B device budget -> exit {p.returncode}")
    for line in reason[:4]:
        say('    ' + line.strip())
    est = A.analyze(EDGE / 'edge-standard-160.onnx', A.load('wide-array'), with_sensitivity=False)
    stem = est['analysis']['selected']['nodes'][0]
    say(f"  MODELED (support rule): {stem['source_name']} on wide-array -> {'; '.join(stem['device_rejection_reasons'])}")
    say(f"  Importer: edge-standard-160 is analysis-only; execution limits: "
        f"{', '.join(sorted({x['code'] for x in est['execution_limits']}))}")

    # 6 ---------------------------------------------------------------------------------
    say('\n## 6. Numerical correctness and evidence')
    for target, result in runs.items():
        say(f"  {target}: {len(result['placements'])} executed placements; all match original ONNX Runtime: "
            f"{result['all_match_original_onnx']}; analytical movement equals engine commands: {result['all_movement_agrees_with_model']}")
    for name, info in manifest['models'].items():
        say(f"  EVALUATED {name}: held-out test accuracy {info['onnxruntime_test_accuracy']:.4f} (ONNX Runtime, 360 images; "
            f"recorded in examples/models/digits/manifest.json)")
    say('  Numerical agreement shows the engine computes the same function; accuracy is a property of the trained model.')
    queued = execute(DIGITS / 'mlp.onnx', DIGITS / 'mlp_test_inputs.npz', 'fragmented-array', binary, out / '6-command-queue',
                     runtime='command-queue')
    say(f"  Command-level queue runtime (copy + compute engines, per-command events): {len(queued['placements'])} placements, "
        f"all match ONNX: {queued['all_match_original_onnx']}")
    trace = out / '6-overlap'
    trace.mkdir()
    command = [str(binary), 'run', '--graph', str(ROOT / 'examples/fragmented.plg'), '--policy', 'maximal', '--relu-supported',
               '--runtime', 'command-queue', '--emulate-boundaries', '--boundary-ms', '20', '--launch-ms', '20',
               '--json', str(trace / 'run.json'), '--trace', str(trace / 'trace.json')]
    p = subprocess.run(command, capture_output=True, text=True)
    (trace / 'run.log').write_text(' '.join(command) + f'\nexit {p.returncode}\n' + p.stdout + p.stderr)
    overlap = [line for line in p.stdout.splitlines() if 'overlap' in line]
    say('  SYNTHETIC overlap demo (injected 20 ms waits, not hardware time): ' + (overlap[0].strip() if overlap else 'no overlap line'))
    say(f"  Evidence directory: {out}")
    say('\nNothing here is a physical-hardware measurement; hardware validation is deferred.')
    (out / 'transcript.md').write_text('\n'.join(log) + '\n')
    (out / 'manifest.json').write_text(json.dumps({'binary': str(binary), 'python': sys.version,
                                                    'files': sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())},
                                                   indent=2) + '\n')
    return 0
