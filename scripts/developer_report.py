"""Explain saved model evidence without inventing target support or hardware latency."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from target_profile import validate_profile, engine_flags

BASE_FILES = {'model.onnx', 'inputs.npz', 'profile.json', 'compatibility.json'}
RUN_FILES = BASE_FILES | {'import/import.json', 'import/graph.plg', 'onnx-reference.npz',
                         'core.json', 'target-analysis.json'}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_evidence(directory):
    directory = Path(directory).resolve()
    receipt = read_json(directory/'result.json')
    verified = receipt.get('status') == 'verified'
    measured = receipt.get('performance_conclusions_valid') is True
    expected = RUN_FILES if verified else BASE_FILES
    if measured:
        require(verified, 'A failed model cannot have valid performance conclusions')
        expected = expected | {'study/bundle.json'}
    seals = receipt.get('evidence_sha256', {})
    require(expected <= set(seals), 'Run lacks bound explanation evidence; analyze again into a fresh directory')
    require(set(seals) <= RUN_FILES | {'study/bundle.json'}, 'Unexpected evidence path')
    for name, checksum in seals.items():
        path = directory/name
        require(path.resolve().is_relative_to(directory), 'Evidence escapes its run directory')
        require(digest(path) == checksum, 'Evidence changed: '+name)
    profile = validate_profile(read_json(directory/'profile.json'))
    require(receipt['profile'] == profile and receipt['core_flags'] == engine_flags(profile), 'Target/flags differ from saved profile')
    require(receipt.get('hardware_validated') is False and receipt.get('vendor_compilation_performed') is False,
            'Software explanation cannot assert physical hardware or compiler acceptance')
    require(receipt.get('model_sha256') == seals['model.onnx'] and receipt.get('inputs_sha256') == seals['inputs.npz'], 'Model/input identity differs')
    imported = read_json(directory/'compatibility.json')
    core = bundle = None
    if verified:
        import study
        import bounded_study
        core = read_json(directory/'core.json')
        validator = study if core['search']['exhaustive'] else bounded_study
        validator.validate_report(core)
        config = core['config']
        expected_config = {**profile['memory'],
                           'relu_supported': profile['operators']['Relu'],
                           'cnn_supported': profile['operators'].get('Conv2D', False),
                           'transfer_fixed_ms': profile['costs']['transfer_fixed_ms'],
                           'bandwidth_bytes_per_ms': profile['costs']['bandwidth_bytes_per_ms'],
                           'launch_ms': profile['costs']['launch_ms'],
                           'synchronization_ms': profile['costs']['sync_ms']}
        require(all(config[k] == v for k, v in expected_config.items()), 'Core target configuration differs from profile')
        imported = read_json(directory/'import/import.json')
        require(imported['model_sha256'] == seals['model.onnx'] and imported['inputs_sha256'] == seals['inputs.npz']
                and imported['graph_sha256'] == seals['import/graph.plg'] == receipt['graph_sha256'], 'Imported source identity differs')
        require(receipt.get('numerical_execution_validated') is True, 'Original-model validation missing')
        comparisons = receipt['comparisons']
        require(len(comparisons) == len(core['results']) == receipt['placements_checked'] and
                {x['placement'] for x in comparisons} == {x['placement'] for x in core['results']} and
                all(x['original_onnx_match'] is True and x['core_reference_match'] is True for x in comparisons),
                'Original-model checks do not cover all reported placements')
        require(all(r['correct'] is True for r in core['results']), 'Failed numerical result')
        normalized = imported['normalized_nodes']
        require([n['generated_node'] for n in normalized] == [n['id'] for n in core['compiled_graph']['nodes']], 'Source-node mapping differs')
        if measured:
            bundle = read_json(directory/'study/bundle.json')
            validator.validate_bundle(bundle)
            require(bundle['conclusions_valid'] is True and bundle['experiment']['config'] == core['config'] and
                    bundle['experiment']['graph_sha256'] == receipt['graph_sha256'] and
                    bundle['experiment']['binary_sha256'] == receipt['binary_sha256'] and
                    bundle['experiment']['flags'] == receipt['core_flags'], 'Measured study belongs to a different run/configuration')
            require({p['placement'] for p in bundle['plans']} == {p['placement'] for p in core['results']}, 'Measured candidate set differs')
    return receipt, imported, core, bundle


def describe_plan(result, nodes, capabilities):
    commands = result['commands']
    copies = []
    for command in commands:
        if command['kind'] != 'COPY':
            continue
        output = command['output']
        tensor = output[2:]
        producer = next((n for n in nodes if n['output'] == tensor), None)
        copies.append({'command_id': command['id'], 'from': command['inputs'][0], 'to': output,
                       'bytes': math.prod(command['shape'])*4,
                       'reason': ('Move an intermediate across the CPU/device boundary' if producer else
                                  'Make an input or constant available in the selected memory domain'),
                       'source_node': producer['source_name'] if producer else None})
    segments = []
    for letter, node in zip(result['placement'], nodes):
        if not segments or segments[-1]['lane'] != letter:
            segments.append({'lane': letter, 'nodes': []})
        cap = capabilities[node['generated_node']]
        reason = ('Target accepts this operation' if letter == 'A' else
                  'CPU fallback: '+cap['reason'] if not cap['device_supported'] else
                  'CPU selected by this policy although target capability accepts the operation')
        segments[-1]['nodes'].append({'id': node['generated_node'], 'source': node['source_name'],
                                      'source_index': node['source_index'], 'operator': node['operator'], 'reason': reason})
    costs = Counter()
    for command in commands:
        category = ('compute' if command['kind'] == 'COMPUTE' else
                    'transfers' if command['kind'] == 'COPY' else
                    'synchronization' if command['kind'] == 'SYNC' else 'allocation_and_lifetime')
        costs[category] += command['modeled_ms']
    # Command compute costs already include launch charges where applicable.
    return {'placement': result['placement'], 'segments': segments, 'copies': copies,
            'h2d_bytes': result['h2d_bytes'], 'd2h_bytes': result['d2h_bytes'],
            'reserved_device_bytes': result['peak_allocated_device_bytes'],
            'live_device_bytes': result['peak_live_device_bytes'],
            'syncs': result['synchronizations'], 'device_regions': result['islands'],
            'command_count': len(commands), 'illustrative_ms': result['modeled_ms'],
            'illustrative_cost_components_ms': dict(costs), 'measured': None}


def build_report(directory, include_host=False):
    receipt, imported, core, bundle = load_evidence(directory)
    profile = receipt['profile']
    report = {'schema_version': 1, 'artifact_type': 'developer-model-explanation',
              'model_sha256': receipt['model_sha256'], 'target': profile['id'],
              'target_evidence': profile['evidence'], 'status': receipt['status'],
              'simulator_verdict': 'not_verified', 'physical_accelerator_verdict': 'unknown',
              'physical_accelerator_latency_ms': None, 'task_accuracy': 'not_evaluated',
              'issues': imported.get('issues', []), 'plans': [], 'nodes': [], 'findings': [],
              'limits': ['Device buffers are host allocations in this backend.',
                         'No vendor workspace, cache, contention, thermal or vendor scheduling model.',
                         'Per-operation illustrative compute costs are placeholders, not calibrated accelerator predictions.'],
              'next_steps': ['Identify the actual accelerator, SDK/image and compiler/runtime versions.',
                             'Compile the original model with that vendor toolchain; retain accepted/rejected subgraphs.',
                             'Run representative held-out inputs on the board, check outputs and retain repeated timings.',
                             'Calibrate target costs from those observations and state where the calibration applies.']}
    if core is None:
        report['simulator_verdict'] = ('unsupported_by_importer' if receipt['status'] == 'unsupported_by_importer' else 'validation_failed')
        report['findings'].append('This implementation cannot establish complete-model execution/performance for this input. Importer rejection is not proof of accelerator rejection.')
        return report
    import study
    nodes = imported['normalized_nodes']
    caps = {c['node_id']: c for c in core['capabilities']}
    report['simulator_verdict'] = 'verified_on_supplied_inputs'
    report['inputs'] = imported['inputs']
    report['outputs'] = imported['outputs']
    report['search'] = core['search']
    report['device_budget_bytes'] = core['config']['device_bytes']
    report['profile_assumptions'] = profile['costs']
    for node in nodes:
        cap = caps[node['generated_node']]
        report['nodes'].append({'source': node['source_name'], 'source_index': node['source_index'],
                                'generated_node': node['generated_node'], 'operator': node['operator'],
                                'device_supported': cap['device_supported'], 'reason': cap['reason']})
    results = {r['placement']: r for r in core['results']}
    maximal = study.strict_maximal_placement(core)
    chosen = core['search']['ranked_feasible'][0]['placement']
    selections = [('Illustrative-cost choice', chosen), ('Strict maximal supported', maximal)]
    if bundle is not None:
        selections.append(('Host-calibrated robust choice', bundle['policies']['robust']['placement']))
    if include_host:
        selections.append(('CPU reference baseline (same host executor)', 'C'*len(nodes)))
    for label, placement in selections:
        if placement not in results:
            report['plans'].append({'label': label, 'placement': placement, 'feasible': False,
                                   'reason': core['search'].get('maximal_rejection') or
                                             'Locally accepted operations do not form a feasible complete plan under this configuration'})
            continue
        plan = describe_plan(results[placement], nodes, caps)
        plan.update(label=label, feasible=True)
        if bundle is not None:
            measured_plan = next(p for p in bundle['plans'] if p['placement'] == placement)
            m, prediction = measured_plan['metrics'], measured_plan['prediction']
            plan['measured'] = {'scope': 'host software executor; not physical accelerator latency',
                                'samples': m['count'], 'p50_ms': m['p50_ms'], 'p95_ms': m['p95_ms'],
                                'calibrated_estimate_ms': prediction['boundary_ms'],
                                'empirical_envelope_ms': [prediction['lower_ms'], prediction['upper_ms']],
                                'end_to_end_p50_ms': m['end_to_end']['p50_ms']}
        report['plans'].append(plan)
    fallback = [n for n in report['nodes'] if not n['device_supported']]
    report['device_only_verdict'] = ('unavailable_unsupported_operations' if fallback else
                                    'feasible_in_simulator' if 'A'*len(nodes) in results else
                                    'infeasible_complete_plan')
    report['findings'].append(f'{len(nodes)-len(fallback)}/{len(nodes)} normalized operations are locally accepted by this hypothetical profile; counts are not a compute-coverage percentage.')
    if fallback:
        report['findings'].append('Required CPU fallback: '+', '.join(f"#{n['source_index']} {n['source']} ({n['operator']})" for n in fallback))
    best, maximal_plan = report['plans'][:2]
    if maximal_plan['feasible']:
        saved = maximal_plan['h2d_bytes'] + maximal_plan['d2h_bytes'] - best['h2d_bytes'] - best['d2h_bytes']
        report['findings'].append(f"Compared with strict maximal support, the illustrative-cost choice changes transferred bytes by {-saved:+d}, syncs by {best['syncs']-maximal_plan['syncs']:+d}, and reserved device bytes by {best['reserved_device_bytes']-maximal_plan['reserved_device_bytes']:+d}.")
        if best['placement'] != maximal:
            report['findings'].append('Supported operations may stay on CPU because selection scores the complete plan, including boundary costs. More device operations do not guarantee a better plan.')
    else:
        report['findings'].append('Strict maximal placement is infeasible. Per-node support does not establish whole-model memory fit; another feasible plan does not replace that baseline.')
    report['findings'].append('A favorable illustrative score or a host timing difference does not establish a hardware speedup or model accuracy.')
    return report


def clean(value):
    # ONNX source names are untrusted text; do not emit terminal control sequences.
    return ''.join(c if c.isprintable() else '?' for c in str(value))


def render(report, details=False):
    lines = ['MODEL EXECUTION EXPLANATION', f"Target: {clean(report['target'])} | hypothetical software device",
             '', 'Will my model run?', f"  Simulator: {report['simulator_verdict']}",
             '  Physical accelerator: UNKNOWN — vendor compilation and board execution are not validated.',
             '  Physical accelerator latency: UNKNOWN — no calibrated or measured board estimate.',
             '  Task accuracy: not evaluated; output agreement is a numerical check.']
    if not report['plans']:
        lines += ['', 'Why analysis stopped:'] + ['  '+clean(x) for x in report['issues']]
        lines += ['  '+x for x in report['findings']]
    else:
        lines += ['', 'Inputs: '+', '.join(clean(x['source_name'])+' '+str(x['shape']) for x in report['inputs']),
                  'Output: '+', '.join(clean(x['source_name'])+' '+str(x['shape']) for x in report['outputs'])]
        lines.append('Device-only execution in this simulator: '+report['device_only_verdict'])
        search = report['search']
        lines.append('Search: '+('exhaustive feasible placements' if search['exhaustive'] else
                                f"bounded candidates ({search['evaluated']} evaluations; no global optimum claim)"))
        lines += ['', 'Compatibility (original ONNX source -> lowered operation):']
        for n in report['nodes']:
            lines.append(f"  #{n['source_index']} {clean(n['source'])} -> {n['operator']}: "+
                         ('device accepted' if n['device_supported'] else 'CPU fallback')+'; '+clean(n['reason']))
        lines += ['', 'How it executes (A = logical device, C = CPU):']
        for plan in report['plans']:
            lines.append(f"  {plan['label']}: {plan['placement']}")
            if not plan['feasible']:
                lines.append('    INFEASIBLE: '+clean(plan['reason']))
                continue
            path = ' -> '.join(('device' if s['lane']=='A' else 'CPU')+'['+', '.join(n['operator'] for n in s['nodes'])+']' for s in plan['segments'])
            lines.append('    '+path)
            lines.append(f"    Transfer H->D {plan['h2d_bytes']} B / D->H {plan['d2h_bytes']} B; {plan['device_regions']} device regions; {plan['syncs']} syncs; {plan['command_count']} commands.")
            lines.append(f"    Device payload: reserved {plan['reserved_device_bytes']} B / budget {report['device_budget_bytes']} B; live peak {plan['live_device_bytes']} B. Vendor workspace excluded.")
            lines.append(f"    Illustrative score: {plan['illustrative_ms']:.6g} ms (uncalibrated assumptions, NOT accelerator latency).")
            m = plan['measured']
            if m:
                lines.append(f"    Host-only measurement: p50 {m['p50_ms']:.6g} ms / p95 {m['p95_ms']:.6g} ms, {m['samples']} held-out runs.")
                lines.append(f"    Host-calibrated estimate: {m['calibrated_estimate_ms']:.6g} ms; empirical range {m['empirical_envelope_ms'][0]:.6g}–{m['empirical_envelope_ms'][1]:.6g} ms (not a confidence interval).")
            else:
                lines.append('    Repeated performance measurement: not requested; use analyze --measure.')
            if details:
                for copy in plan['copies']:
                    lines.append(f"      COPY #{copy['command_id']}: {clean(copy['from'])} -> {clean(copy['to'])}, {copy['bytes']} B. {copy['reason']}.")
                lines.append('    Illustrative cost breakdown (compute includes launch where charged): '+clean(plan['illustrative_cost_components_ms']))
        lines += ['', 'Why plans differ:'] + ['  '+clean(x) for x in report['findings']]
        lines += ['', 'Runtime sequence: allocate distinct buffers; copy missing operands; run kernels in order;',
                  'synchronize before CPU reads; return outputs; compare against original ONNX Runtime.',
                  'Capability says where an operation CAN run. Placement decides where it SHOULD run under the selected scoring assumptions.']
    lines += ['', 'What is needed for an accelerator performance answer:'] + ['  '+x for x in report['next_steps']]
    lines += ['', 'Simulator limits:'] + ['  '+x for x in report['limits']]
    return '\n'.join(lines)+'\n'
