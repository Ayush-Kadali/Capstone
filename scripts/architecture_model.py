"""Shape-aware analytical performance model for declared accelerator architectures.

This module ESTIMATES; it never executes a model or emulates a vendor chip.
It reads the importer's normalized operations (scripts/onnx_frontend.py) and an
architecture profile (profiles/architectures/*.json), then:

1. counts arithmetic work and memory traffic for every operation from its shapes;
2. applies the profile's declared support rules (which ops may use the accelerator);
3. walks a candidate CPU/accelerator placement in program order, tracking where each
   tensor lives, which transfers and synchronizations occur, and peak device memory;
4. converts work and bytes into time with simple roofline formulas;
5. searches placements (exhaustive when small, beam otherwise) and reports ranges,
   one-parameter sensitivity and whether a target ranking is robust.

Units: operations (1 multiply-add = 2 ops), bytes (f32 = 4 B), milliseconds.
GOPS = 1e9 ops/s = 1e6 ops/ms; GB/s = 1e9 B/s = 1e6 B/ms; microseconds / 1000 = ms.

The same profile also maps to functional-engine flags (execution_flags), so the
fallbacks shown here are the fallbacks the C++ core actually executes.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / 'profiles' / 'architectures'
OPS = ('MatMul', 'MatMulRelu', 'Add', 'Relu', 'RowSum', 'Conv2D',
       'MaxPool2D', 'AveragePool2D', 'GlobalAveragePool', 'Reshape')
# Importer operator name -> C++ pl::name(Op) spelling used by --device-unsupported.
CORE_NAMES = {'Relu': 'ReLU', 'MatMulRelu': 'MatMulReLU'}
VERSION = 'serial-roofline-2'
EXHAUSTIVE_LIMIT = 10      # at most 2^10 = 1024 candidate placements
BEAM_WIDTH = 8
SENSITIVITY = (('accelerator', 'peak_gops'), ('accelerator', 'bandwidth_gbps'),
               ('accelerator', 'dispatch_us'), ('link', 'bandwidth_gbps'),
               ('link', 'transfer_us'), ('link', 'sync_us'), ('cpu', 'peak_gops'))
FACTORS = (0.5, 2.0)
LIMITS = [
    'Analytical scenario: not measured latency, not vendor compiler acceptance, not a physical-chip prediction.',
    'Static f32 importer subset only; task accuracy is not evaluated by this estimate.',
    'Kernels run one at a time; no fusion, copy/compute overlap, cache simulation, contention, DVFS or thermal effects.',
    'CPU traffic assumes each operand is read once (ideal caching); accelerator MatMul/Conv traffic uses the declared tile re-read model.',
    'Default request starts with inputs and weights on the host; --resident-weights assumes weights were uploaded earlier.',
    'Default memory policy "retain" keeps every device allocation for the request (as the engine does by default); "release" frees after last device use and waits for the device before each batch of frees.',
    'Workspace is a declared per-kernel reserve, not an inferred vendor allocation.',
    'Ranges come from declared efficiency scenarios; they are assumption ranges, not statistical confidence intervals.',
]


# ---------------------------------------------------------------- profiles

def _exact(obj, keys, where):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError(f'{where}: expected exactly the fields {sorted(keys)}')


def _number(x, where, *, zero=False):
    low = 0 if zero else 1e-12
    if type(x) not in (int, float) or not math.isfinite(x) or not low <= x <= 1e15:
        raise ValueError(f'{where}: expected a finite {"nonnegative" if zero else "positive"} number <= 1e15')


def validate(profile):
    """Reject anything outside the documented schema; returns the profile unchanged."""
    _exact(profile, {'schema_version', 'id', 'revision', 'description', 'evidence',
                     'cpu', 'accelerator', 'link', 'support'}, 'architecture profile')
    if type(profile['schema_version']) is not int or profile['schema_version'] != 1:
        raise ValueError('architecture profile schema_version must be 1')
    if type(profile['revision']) is not int or not 1 <= profile['revision'] <= 1000:
        raise ValueError('revision must be an integer in 1..1000')
    for k in ('id', 'description'):
        if not isinstance(profile[k], str) or not 1 <= len(profile[k]) <= 1000 or any(ord(c) < 32 for c in profile[k]):
            raise ValueError(f'{k}: expected printable nonempty text')
    evidence = profile['evidence']
    _exact(evidence, {'level', 'source', 'hardware_validated'}, 'evidence')
    if evidence['level'] not in ('hypothetical', 'documentation_based') or evidence['hardware_validated'] is not False:
        raise ValueError('Only unvalidated hypothetical or documentation_based analytical profiles are supported')
    if not isinstance(evidence['source'], str) or not evidence['source'].strip():
        raise ValueError('evidence.source is required')
    for side in ('cpu', 'accelerator'):
        unit = profile[side]
        fields = {'peak_gops', 'bandwidth_gbps', 'efficiency', 'dispatch_us'}
        if side == 'accelerator':
            fields |= {'memory_bytes', 'local_bytes', 'workspace_bytes', 'tile_mnk'}
        _exact(unit, fields, side)
        _number(unit['peak_gops'], f'{side}.peak_gops')
        _number(unit['bandwidth_gbps'], f'{side}.bandwidth_gbps')
        _number(unit['dispatch_us'], f'{side}.dispatch_us', zero=True)
        eff = unit['efficiency']
        if (not isinstance(eff, list) or len(eff) != 3 or eff != sorted(eff)
                or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 < x <= 1 for x in eff)):
            raise ValueError(f'{side}.efficiency must be [low, nominal, high] ordered in (0, 1]')
    acc = profile['accelerator']
    for k in ('memory_bytes', 'local_bytes', 'workspace_bytes'):
        if type(acc[k]) is not int or not 0 <= acc[k] <= 2**40:
            raise ValueError(f'accelerator.{k} must be an integer in [0, 2^40]')
    if acc['workspace_bytes'] >= acc['memory_bytes']:
        raise ValueError('accelerator.workspace_bytes must be smaller than memory_bytes')
    tile = acc['tile_mnk']
    if not isinstance(tile, list) or len(tile) != 3 or any(type(x) is not int or not 1 <= x <= 1024 for x in tile):
        raise ValueError('tile_mnk requires three integers in [1, 1024]')
    if tile_scratch_bytes(tile) > acc['local_bytes']:
        raise ValueError(f'declared tile scratch {tile_scratch_bytes(tile)} B exceeds local_bytes {acc["local_bytes"]} B')
    link = profile['link']
    _exact(link, {'bandwidth_gbps', 'transfer_us', 'sync_us'}, 'link')
    _number(link['bandwidth_gbps'], 'link.bandwidth_gbps')
    _number(link['transfer_us'], 'link.transfer_us', zero=True)
    _number(link['sync_us'], 'link.sync_us', zero=True)
    rules = profile['support']
    _exact(rules, {'operators', 'max_tensor_bytes', 'conv_groups'}, 'support')
    ops = rules['operators']
    if not isinstance(ops, list) or len(set(ops)) != len(ops) or not all(isinstance(x, str) and x in OPS for x in ops):
        raise ValueError('support.operators must be distinct normalized operator names: ' + ', '.join(OPS))
    if type(rules['max_tensor_bytes']) is not int or not 1 <= rules['max_tensor_bytes'] <= 2**40:
        raise ValueError('support.max_tensor_bytes must be a positive integer <= 2^40')
    if rules['conv_groups'] not in ('any', 'one'):
        raise ValueError('support.conv_groups must be "any" or "one"')
    return profile


def _no_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate profile field: ' + key)
        obj[key] = value
    return obj


def load(value):
    """Load by registered ID (profiles/architectures/ID.json) or by file path."""
    path = PROFILE_DIR / f'{value}.json'
    if not path.is_file():
        path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f'Unknown target {value!r}: not a registered profile ID or profile file; run "capstone.py architecture-targets"')
    with path.open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('architecture profile exceeds 64 KiB')
    return validate(json.loads(raw, object_pairs_hook=_no_duplicates))


def registered():
    return [load(p) for p in sorted(PROFILE_DIR.glob('*.json'))]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def execution_flags(profile, memory_policy='retain'):
    """Translate the support rules and memory limits into C++ engine flags.

    This is what makes the functional simulator honour the same declared rules.
    Timing parameters are NOT translated: the engine's illustrative costs are a
    separate, older model, and its host timings are software measurements.
    """
    rules, acc = profile['support'], profile['accelerator']
    allowed = set(rules['operators'])
    cnn = {'Conv2D', 'MaxPool2D', 'AveragePool2D', 'GlobalAveragePool', 'Reshape'}
    # The engine budgets tensor payload only; reserving the declared workspace here makes
    # its feasibility test identical to the model's "payload + workspace <= memory".
    flags = ['--device-bytes', str(acc['memory_bytes'] - acc['workspace_bytes']), '--local-bytes', str(acc['local_bytes']),
             '--device-max-tensor-bytes', str(rules['max_tensor_bytes'])]
    if memory_policy == 'release':
        flags.append('--reuse-buffers')
    if 'Relu' in allowed:
        flags.append('--relu-supported')
    if allowed & cnn:
        flags.append('--cnn-supported')
    if rules['conv_groups'] == 'one':
        flags.append('--device-no-grouped-conv')
    for op in OPS:
        relu_gated = op in ('Relu', 'MatMulRelu') and 'Relu' not in allowed  # already off
        cnn_gated = op in cnn and not (allowed & cnn)
        if op not in allowed and not relu_gated and not cnn_gated:
            flags += ['--device-unsupported', CORE_NAMES.get(op, op)]
    return flags


# ---------------------------------------------------------------- work and traffic

def tile_scratch_bytes(tile):
    """Local memory for one A tile (m x k), one B tile (k x n) and one C tile (m x n)."""
    m, n, k = tile
    return 4 * (m * k + k * n + m * n)


def _elements(shape):
    return math.prod(shape)


def work(node, shapes):
    """Exact operation count and ideal traffic (each distinct operand read once, output written once).

    MatMul [M,K]x[K,N]: 2*M*N*K ops (multiply-add = 2). Conv2D as implicit GEMM:
    M = batch*OH*OW, N = filters/group, K = (channels/group)*KH*KW, repeated per group.
    Pooling: one compare/add per window element per output (+1 divide for average).
    """
    op = node['operator']
    out = _elements(node['shape'])
    ins = node['inputs']
    p = node['parameters']
    matrix, groups = None, 1
    if op in ('MatMul', 'MatMulRelu'):
        m, k = shapes[ins[0]]
        n = shapes[ins[1]][1]
        matrix = (m, n, k)
        ops = 2 * m * n * k + (out if op == 'MatMulRelu' else 0)
    elif op == 'Conv2D':
        batch, channels, filters, kh, kw, groups = p[0], p[1], p[4], p[5], p[6], p[13]
        _, _, oh, ow = node['logical_shape']
        k = channels // groups * kh * kw
        matrix = (batch * oh * ow, filters // groups, k)
        ops = 2 * out * k
    elif op in ('MaxPool2D', 'AveragePool2D'):
        ops = out * p[4] * p[5] + (out if op == 'AveragePool2D' else 0)
    elif op in ('RowSum', 'GlobalAveragePool'):
        ops = _elements(shapes[ins[0]]) - out + (out if op == 'GlobalAveragePool' else 0)
    elif op in ('Add', 'Relu'):
        ops = out
    elif op == 'Reshape':
        ops = 0  # the functional runtime materializes a contiguous copy: traffic only
    else:
        raise ValueError('unmodeled normalized operator ' + op)
    traffic = 4 * (sum(_elements(shapes[x]) for x in dict.fromkeys(ins)) + out)
    return {'operations': ops, 'ideal_traffic_bytes': traffic, 'matrix_mnk': matrix, 'groups': groups}


def tiled_traffic(node, shapes, w, acc):
    """Accelerator external traffic for MatMul/Conv under the declared output tiles.

    Output-stationary tiling: activation (A) is re-read once per column tile,
    weights (B) once per row tile, output written once. If the whole weight
    operand plus one tile set fits local memory, weights are read once
    (weight-stationary). Other operations keep ideal traffic.
    """
    if not w['matrix_mnk']:
        return w['ideal_traffic_bytes'], 'ideal (not a tiled operation)'
    m, n, _ = w['matrix_mnk']
    tm, tn, _ = acc['tile_mnk']
    a_bytes = 4 * _elements(shapes[node['inputs'][0]])
    b_bytes = 4 * _elements(shapes[node['inputs'][1]])
    out_bytes = 4 * _elements(node['shape'])
    a_reads = math.ceil(n / tn)
    if b_bytes + tile_scratch_bytes(acc['tile_mnk']) <= acc['local_bytes']:
        b_reads, mode = 1, 'weight-stationary (weights fit local memory)'
    else:
        b_reads, mode = math.ceil(m / tm), 'output-stationary tiles (weights re-read per row tile)'
    return a_bytes * a_reads + b_bytes * b_reads + out_bytes, mode


def support(node, shapes, profile):
    """Declared-rule reasons why this op may NOT use the accelerator ([] = permitted)."""
    rules = profile['support']
    reasons = []
    if node['operator'] not in rules['operators']:
        reasons.append('operator excluded by the declared target support rules')
    if any(4 * _elements(shapes[t]) > rules['max_tensor_bytes'] for t in node['inputs'] + [node['output']]):
        reasons.append(f'tensor exceeds the declared per-tensor device limit of {rules["max_tensor_bytes"]} B')
    if node['operator'] == 'Conv2D' and rules['conv_groups'] == 'one' and node['parameters'][13] != 1:
        reasons.append('grouped convolution excluded by the declared target support rules')
    # Same per-node check as the engine's capability() (src/core.cpp): distinct operands + output.
    if 4 * sum(_elements(shapes[t]) for t in dict.fromkeys(node['inputs'] + [node['output']])) > profile['accelerator']['memory_bytes'] - profile['accelerator']['workspace_bytes']:
        reasons.append('operands plus output exceed device memory budget')
    return reasons


def kernel_cost(node, shapes, profile, side, scenario):
    """Roofline time for one kernel: max(compute, memory) + dispatch.

    scenario indexes the [low, nominal, high] efficiency list (0, 1, 2).
    compute_ms = ops / (peak_gops * 1e6 * efficiency * tile_utilization)
    memory_ms  = bytes / (bandwidth_gbps * 1e6 * efficiency)
    """
    w = work(node, shapes)
    unit = profile['cpu' if side == 'C' else 'accelerator']
    eff = unit['efficiency'][scenario]
    utilization, traffic, mode = 1.0, w['ideal_traffic_bytes'], 'ideal'
    if side == 'A' and w['matrix_mnk']:
        dims, tile = w['matrix_mnk'], unit['tile_mnk']
        padded = math.prod(math.ceil(d / t) * t for d, t in zip(dims, tile))
        utilization = math.prod(dims) / padded
        traffic, mode = tiled_traffic(node, shapes, w, unit)
    compute = w['operations'] / (unit['peak_gops'] * 1e6 * eff * utilization)
    memory = traffic / (unit['bandwidth_gbps'] * 1e6 * eff)
    dispatch = unit['dispatch_us'] / 1000
    bound = 'compute' if compute > memory else 'memory'
    if dispatch > max(compute, memory):
        bound = 'dispatch'
    return dict(w, traffic_bytes=traffic, traffic_model=mode, tile_utilization=utilization,
                compute_ms=compute, memory_ms=memory, dispatch_ms=dispatch,
                total_ms=max(compute, memory) + dispatch, bottleneck=bound)


# ---------------------------------------------------------------- placement walk

def _context(report):
    nodes = report['normalized_nodes']
    shapes = {x['name']: x['shape'] for x in report['normalized_tensors']}
    produced = {n['output'] for n in nodes}
    runtime = {report['tensor_mapping'][x['source_name']] for x in report['inputs']}
    weights = set(shapes) - produced - runtime
    output = report['tensor_mapping'][report['outputs'][0]['source_name']]
    last_use = {t: -1 for t in shapes}
    for i, n in enumerate(nodes):
        for t in n['inputs']:
            last_use[t] = i
    last_use[output] = len(nodes)
    return nodes, shapes, weights, output, last_use


def schedule(report, profile, placement, scenario=1, resident_weights=False, memory_policy='retain'):
    """Walk one placement in program order and account every cost component.

    Rules (documented in docs/runtime/ANALYTICAL_MODEL.md):
    * Tensors are immutable; a transfer copies and leaves the source valid.
    * H2D happens before an accelerator op needs a host-only operand.
    * D2H happens before a CPU op needs a device-only operand.
    * Synchronization follows the engine's lowering (src/core.cpp insert_transfers):
      the first CPU op after any device work waits once for the device, whether or
      not it pulls data; later pulls while the device is idle need no new wait. The
      request ends with one wait if device work is still pending, then the output
      returns to the host.
    * memory_policy='retain' (default; matches the engine without --reuse-buffers):
      nothing is freed during the request, so the peak is every device allocation.
    * memory_policy='release' (engine --reuse-buffers): a device copy is
      freed once no later accelerator op reads it and no later D2H needs it, and freeing waits for pending device work first
      (one sync before a batch of frees). Resident weights are never freed.
    * Peak device memory = live device tensors + workspace while an accelerator op runs.
    """
    nodes, shapes, weights, output, last_use = _context(report)
    if memory_policy not in ('release', 'retain'):
        raise ValueError('memory_policy must be release or retain')
    if len(placement) != len(nodes) or any(c not in 'CA' for c in placement):
        raise ValueError('placement must give one C (CPU) or A (accelerator) per normalized node')
    acc, link = profile['accelerator'], profile['link']
    host = set(shapes) - {n['output'] for n in nodes}
    device = set(weights) if resident_weights else set()
    pinned = set(device)
    transfers, rows, failures = [], [], []
    totals = {'compute_ms': 0.0, 'transfer_ms': 0.0, 'sync_ms': 0.0}
    syncs = 0
    pending = False  # device work submitted since the last host wait
    live = sum(4 * _elements(shapes[t]) for t in device)
    peak = live

    consumers = {}
    for j, n in enumerate(nodes):
        for t in dict.fromkeys(n['inputs']):
            consumers.setdefault(t, []).append(j)

    def device_copy_needed(t, i):
        """The device copy is still read later by an accelerator op, or by a D2H copy
        because a later CPU consumer (or the final output) cannot use a host copy."""
        later = [j for j in consumers.get(t, []) if j > i]
        if any(placement[j] == 'A' for j in later):
            return True
        return t not in host and (bool(later) or t == output)

    def move(tensor, to, index):
        nonlocal live, peak
        size = 4 * _elements(shapes[tensor])
        cost = link['transfer_us'] / 1000 + size / (link['bandwidth_gbps'] * 1e6)
        if to == 'A':
            device.add(tensor)
            live += size
            peak = max(peak, live)
        else:
            host.add(tensor)
        transfers.append({'tensor': tensor, 'bytes': size, 'direction': 'H2D' if to == 'A' else 'D2H',
                          'before_node': index, 'ms': cost})
        totals['transfer_ms'] += cost

    for i, (node, side) in enumerate(zip(nodes, placement)):
        reasons = support(node, shapes, profile)
        if side == 'A' and reasons:
            failures.append({'node': node['generated_node'], 'reasons': reasons})
        if side == 'C' and pending:
            syncs += 1
            pending = False
        for t in dict.fromkeys(node['inputs']):
            if side == 'A' and t not in device:
                move(t, 'A', i)
            elif side == 'C' and t not in host:
                move(t, 'C', i)
        if side == 'A':
            pending = True
            device.add(node['output'])
            live += 4 * _elements(node['shape'])
            peak = max(peak, live + acc['workspace_bytes'])
        else:
            host.add(node['output'])
        cost = kernel_cost(node, shapes, profile, side, scenario)
        totals['compute_ms'] += cost['total_ms']
        rows.append(dict(cost, node=node['generated_node'], source_index=node['source_index'],
                         source_name=node['source_name'], operator=node['operator'], backend=side,
                         device_rejection_reasons=reasons))
        freed = [t for t in device if t not in pinned and not device_copy_needed(t, i)] if memory_policy == 'release' else []
        if freed and pending:
            syncs += 1  # a buffer may be reused only after the device finished with it
            pending = False
        for t in freed:
            device.discard(t)
            live -= 4 * _elements(shapes[t])
        for t in [t for t in host if last_use[t] <= i]:
            host.discard(t)
    if pending:
        syncs += 1
    if output not in host:
        move(output, 'C', len(nodes))
    totals['sync_ms'] = syncs * link['sync_us'] / 1000
    if peak > acc['memory_bytes']:
        failures.append({'node': None, 'reasons': [
            f'peak device payload + workspace {peak} B exceeds declared memory {acc["memory_bytes"]} B']})
    feasible = not failures
    total = totals['compute_ms'] + totals['transfer_ms'] + totals['sync_ms']
    return {'placement': placement, 'memory_policy': memory_policy, 'feasible_under_assumptions': feasible,
            'estimated_ms': total if feasible else None,
            'components_ms': {k: v for k, v in totals.items()},
            'device_peak_bytes': peak, 'transfer_bytes': sum(t['bytes'] for t in transfers),
            'h2d_bytes': sum(t['bytes'] for t in transfers if t['direction'] == 'H2D'),
            'd2h_bytes': sum(t['bytes'] for t in transfers if t['direction'] == 'D2H'),
            'transfer_ms': totals['transfer_ms'], 'syncs': syncs, 'sync_ms': totals['sync_ms'],
            'accelerator_nodes': placement.count('A'), 'nodes': rows, 'transfers': transfers,
            'failures': failures, 'dominant_cost': _dominant(rows, totals)}


def _dominant(rows, totals):
    buckets = {'transfers': totals['transfer_ms'], 'synchronization': totals['sync_ms']}
    for r in rows:
        key = f"{'accelerator' if r['backend'] == 'A' else 'CPU'} {r['bottleneck']}"
        if r['bottleneck'] == 'dispatch':
            buckets[key] = buckets.get(key, 0) + r['total_ms']
        else:
            buckets[key] = buckets.get(key, 0) + max(r['compute_ms'], r['memory_ms'])
            buckets_key = f"{'accelerator' if r['backend'] == 'A' else 'CPU'} dispatch"
            buckets[buckets_key] = buckets.get(buckets_key, 0) + r['dispatch_ms']
    name = max(sorted(buckets), key=lambda k: buckets[k])
    return {'component': name, 'ms': buckets[name], 'breakdown_ms': dict(sorted(buckets.items()))}


# ---------------------------------------------------------------- search

def _score(result):
    ms = result['estimated_ms']
    return (math.inf if ms is None else ms, result['transfer_bytes'], result['placement'].count('A'), result['placement'])


def choose(report, profile, resident_weights=False, memory_policy='retain'):
    """Pick the lowest nominal estimate among declared-legal placements.

    <= EXHAUSTIVE_LIMIT legal accelerator nodes: every legal placement is scored
    (an exact optimum *of this model*). Larger graphs: a deterministic beam keeps
    the BEAM_WIDTH best prefixes (completed with CPU) -- a candidate, not an optimum.
    Ties break toward fewer transferred bytes, then fewer accelerator nodes.
    """
    nodes, shapes, *_ = _context(report)
    legal = [not support(n, shapes, profile) for n in nodes]
    count, total = sum(legal), len(nodes)
    cache = {}

    def evaluate(p):
        if p not in cache:
            cache[p] = schedule(report, profile, p, 1, resident_weights, memory_policy)
        return cache[p]

    cpu = 'C' * total
    maximal = ''.join('A' if ok else 'C' for ok in legal)
    evaluate(cpu)
    evaluate(maximal)
    if count <= EXHAUSTIVE_LIMIT:
        for bits in itertools.product('CA', repeat=count):
            it = iter(bits)
            evaluate(''.join(next(it) if ok else 'C' for ok in legal))
        mode, width = 'exhaustive over declared-legal placements', None
    else:
        beam, width, mode = [''], BEAM_WIDTH, 'beam search candidates (not exhaustive)'
        for ok in legal:
            grown = [p + c for p in beam for c in ('CA' if ok else 'C')]
            beam = sorted(grown, key=lambda p: _score(evaluate(p + 'C' * (total - len(p)))))[:width]
    ranked = sorted(cache.values(), key=_score)
    best = ranked[0]
    ties = [r['placement'] for r in ranked[1:] if best['estimated_ms'] is not None and r['estimated_ms'] is not None
            and abs(r['estimated_ms'] - best['estimated_ms']) <= 1e-9 * max(1.0, best['estimated_ms'])]
    selected = copy.deepcopy(best)
    low = schedule(report, profile, selected['placement'], 2, resident_weights, memory_policy)['estimated_ms']
    high = schedule(report, profile, selected['placement'], 0, resident_weights, memory_policy)['estimated_ms']
    selected['scenario_range_ms'] = [low, high] if low is not None else None
    infeasible = [r['placement'] for r in cache.values() if not r['feasible_under_assumptions']]
    return {'selected': selected, 'cpu_only': evaluate(cpu), 'maximal_supported': evaluate(maximal),
            'ties_with_selected': ties[:20], 'infeasible_candidates': len(infeasible),
            'search': {'mode': mode, 'evaluated': len(cache), 'beam_width': width,
                       'legal_accelerator_nodes': count,
                       'optimum_of_this_model': count <= EXHAUSTIVE_LIMIT}}


def _vary(profile, section, field, factor):
    varied = copy.deepcopy(profile)
    varied[section][field] *= factor
    return varied


def sensitivity(report, profile, resident_weights=False, memory_policy='retain'):
    """Change one parameter at a time (x0.5, x2), reselect the placement each time."""
    base = choose(report, profile, resident_weights, memory_policy)['selected']
    rows = []
    for section, field in SENSITIVITY:
        for factor in FACTORS:
            s = choose(report, _vary(profile, section, field, factor), resident_weights, memory_policy)['selected']
            rows.append({'parameter': f'{section}.{field}', 'factor': factor, 'placement': s['placement'],
                         'estimated_ms': s['estimated_ms'], 'changed_placement': s['placement'] != base['placement'],
                         'change_ms': None if s['estimated_ms'] is None or base['estimated_ms'] is None
                         else s['estimated_ms'] - base['estimated_ms']})
    return rows


# ---------------------------------------------------------------- public analyses

def _inspect(model):
    from onnx_frontend import inspect_model
    return inspect_model(model, scope='analysis')


def analyze(model, profile, *, resident_weights=False, with_sensitivity=True, memory_policy='retain'):
    validate(profile)
    report = _inspect(model)
    result = {'schema_version': 2, 'estimator': VERSION, 'model': str(model), 'model_sha256': report.get('model_sha256'),
              'profile_sha256': digest(profile), 'estimator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'target': profile['id'], 'target_revision': profile['revision'], 'evidence': profile['evidence'],
              'value_kinds': {'operations/bytes/placement/transfers': 'derived exactly from shapes and rules',
                              'milliseconds': 'modeled under declared assumptions',
                              'hardware latency': 'unavailable (no physical measurement)'},
              'profile': profile, 'resident_weights': resident_weights, 'memory_policy': memory_policy,
              'limitations': LIMITS,
              'import_supported': report['supported'], 'issues': report['issues'],
              'functionally_executable': report.get('functionally_executable', False),
              'execution_limits': report.get('execution_limits', []),
              'execution_flags': execution_flags(profile, memory_policy),
              'hardware_execution_validated': False, 'analysis': None, 'sensitivity': []}
    if not report['supported']:
        return result
    result['analysis'] = choose(report, profile, resident_weights, memory_policy)
    if with_sensitivity:
        result['sensitivity'] = sensitivity(report, profile, resident_weights, memory_policy)
    return result


def compare_targets(model, profiles, *, resident_weights=False, memory_policy='retain'):
    """Same model, several targets. A winner is called robust only if it survives
    the efficiency scenarios AND every one-parameter perturbation of every target."""
    if not 2 <= len(profiles) <= 8:
        raise ValueError('compare-targets requires 2..8 architecture profiles')
    if len({p['id'] for p in profiles}) != len(profiles):
        raise ValueError('comparison target IDs must be distinct')
    if any(p['cpu'] != profiles[0]['cpu'] for p in profiles[1:]):
        raise ValueError('target comparison requires identical CPU assumptions (otherwise host changes confound it)')
    reports = [analyze(model, p, resident_weights=resident_weights, with_sensitivity=False, memory_policy=memory_policy) for p in profiles]
    result = {'schema_version': 2, 'kind': 'target comparison', 'objective': 'modeled single-request latency (ms)',
              'scope': 'Same ONNX model, f32, identical CPU assumptions, one request; analytical estimates only.',
              'reports': reports, 'nominal_order': [], 'nominal_order_text': '', 'separated_by_efficiency_scenarios': False,
              'winner_flips': [], 'robust_winner': None, 'hardware_recommendation_validated': False}
    if any(not r['analysis'] for r in reports):
        result['conclusion'] = 'No ranking: the importer rejected the model, so no target has a latency.'
        return result
    est = {r['target']: r['analysis']['selected']['estimated_ms'] for r in reports}
    if any(v is None for v in est.values()):
        result['conclusion'] = 'No ranking: a target has no feasible placement under its declared limits.'
        return result
    order = sorted(est, key=lambda t: (est[t], t))
    result['nominal_order'] = order

    def tied(a, b):
        return abs(a - b) <= 1e-9 * max(1.0, min(a, b))

    def leaders(values):
        low = min(values.values())
        return {t for t, v in values.items() if tied(v, low)}

    result['nominal_order_text'] = order[0] + ''.join(
        (' = ' if tied(est[a], est[b]) else ' < ') + b for a, b in zip(order, order[1:]))
    ranges = {r['target']: r['analysis']['selected']['scenario_range_ms'] for r in reports}
    best = order[0]
    co_leaders = leaders(est)
    separated = len(co_leaders) == 1 and all(ranges[best][1] < ranges[t][0] for t in order[1:])
    result['separated_by_efficiency_scenarios'] = separated
    report_of = _inspect(model)

    def estimate(p):
        v = choose(report_of, p, resident_weights, memory_policy)['selected']['estimated_ms']
        return math.inf if v is None else v

    def record(changed, parameter, factor, trial):
        # A flip needs a new leader outside the nominal (co-)leaders; reordering
        # inside an exact tie is a tie-break artifact, not a ranking change.
        new = leaders(trial)
        if not new & co_leaders:
            flips.append({'target_changed': changed, 'parameter': parameter, 'factor': factor,
                          'new_winner': ' = '.join(sorted(new)) or 'none feasible'})

    flips = []
    # Accelerator and link assumptions belong to one target: vary them one target at a time.
    # The CPU is the shared host (identical by construction), so a CPU change applies to
    # every target at once; changing it on one target only would be the host confound
    # this comparison forbids.
    for section, field in SENSITIVITY:
        for factor in FACTORS:
            if section == 'cpu':
                trial = {p['id']: estimate(_vary(p, section, field, factor)) for p in profiles}
                record('all targets (shared CPU)', f'{section}.{field}', factor, trial)
                continue
            for p in profiles:
                trial = dict(est, **{p['id']: estimate(_vary(p, section, field, factor))})
                record(p['id'], f'{section}.{field}', factor, trial)
    result['winner_flips'] = flips
    if len(co_leaders) > 1:
        tie = ' and '.join(sorted(co_leaders))
        result['conclusion'] = f'Tie: {tie} have equal nominal estimates; no ranking among them follows.'
        if flips:
            result['conclusion'] += f' {len(flips)} one-parameter perturbation(s) move the lead outside the tie.'
    elif separated and not flips:
        result['robust_winner'] = best
        result['conclusion'] = (f'{best} is lowest under every declared efficiency scenario and every one-parameter '
                                'x0.5/x2 perturbation. This is a statement about the model, not measured hardware.')
    else:
        why = [] if separated else ['efficiency scenario ranges overlap']
        if flips:
            why.append(f'{len(flips)} one-parameter perturbation(s) change the winner')
        result['conclusion'] = f'Nominal leader {best}, but no reliable ranking: ' + '; '.join(why) + '.'
    return result


def compare_models(models, profile, *, resident_weights=False, accuracy=None, memory_policy='retain'):
    """Model variants on one target. Accuracy is reported only when supplied from a real evaluation."""
    if not 2 <= len(models) <= 8:
        raise ValueError('compare-models requires 2..8 ONNX models')
    if len({Path(m).resolve() for m in models}) != len(models):
        raise ValueError('compare-models requires distinct model files')
    reports = [analyze(m, profile, resident_weights=resident_weights, with_sensitivity=False, memory_policy=memory_policy) for m in models]
    rows = []
    for r in reports:
        a = r['analysis']
        s = a['selected'] if a else None
        rows.append({'model': r['model'], 'import_supported': r['import_supported'],
                     'estimated_ms': s['estimated_ms'] if s else None,
                     'scenario_range_ms': s['scenario_range_ms'] if s else None,
                     'operations': sum(n['operations'] for n in s['nodes']) if s else None,
                     'transfer_bytes': s['transfer_bytes'] if s else None, 'syncs': s['syncs'] if s else None,
                     'cpu_fallback_nodes': sum(1 for n in s['nodes'] if n['backend'] == 'C') if s else None,
                     'device_peak_bytes': s['device_peak_bytes'] if s else None,
                     'placement': s['placement'] if s else None,
                     'task_accuracy': (accuracy or {}).get(r['model'], 'not evaluated')})
    return {'schema_version': 2, 'kind': 'model comparison', 'target': profile['id'],
            'objective': 'modeled single-request latency (ms)', 'rows': rows, 'reports': reports,
            'note': 'Variants can differ in function; latency differences say nothing about task accuracy unless measured.',
            'hardware_recommendation_validated': False}


# ---------------------------------------------------------------- text rendering

def _ms(x):
    return 'infeasible' if x is None else f'{x:.6f} ms'


def render(report):
    from developer_report import clean
    kind = report.get('kind')
    if kind == 'target comparison':
        lines = ['TARGET COMPARISON (analytical estimates, not measurements)', report['scope']]
        for r in report['reports']:
            a = r['analysis']
            if not a:
                lines.append(f"  {clean(r['target'])}: importer rejected the model; no latency")
                continue
            s = a['selected']
            rng = s['scenario_range_ms']
            lines.append(f"  {clean(r['target'])} r{r['target_revision']}: {_ms(s['estimated_ms'])}"
                         + (f" [range {rng[0]:.6f}..{rng[1]:.6f}]" if rng else '')
                         + f"; plan {s['placement']}; {s['transfer_bytes']} B moved; {s['syncs']} syncs;"
                         f" dominant: {s['dominant_cost']['component']}")
        lines.append('Nominal order: ' + (report.get('nominal_order_text') or 'none'))
        for f in report['winner_flips'][:6]:
            lines.append(f"  flip: {f['target_changed']} {f['parameter']} x{f['factor']} -> {f['new_winner']} wins")
        return '\n'.join(lines + [report['conclusion'], 'No physical board execution or ranking validation is implied.']) + '\n'
    if kind == 'model comparison':
        lines = [f"MODEL COMPARISON on {clean(report['target'])} (analytical estimates)"]
        for r in report['rows']:
            if r['estimated_ms'] is None and not r['import_supported']:
                lines.append(f"  {clean(r['model'])}: importer rejected; no latency")
                continue
            lines.append(f"  {clean(r['model'])}: {_ms(r['estimated_ms'])}; {r['operations']} ops; "
                         f"{r['transfer_bytes']} B moved; {r['syncs']} syncs; {r['cpu_fallback_nodes']} CPU nodes; "
                         f"accuracy: {r['task_accuracy']}")
        return '\n'.join(lines + [report['note']]) + '\n'
    lines = [f"MODEL-TO-TARGET ESTIMATE: {clean(report['target'])} revision {report['target_revision']}"
             f" ({report['evidence']['level']}; not hardware-validated)"]
    if report['execution_limits']:
        lines.append(f"Functional execution: not possible within executor limits ({len(report['execution_limits'])} limit(s)); estimate only.")
    a = report['analysis']
    if not a:
        return '\n'.join(lines + ['No latency: the importer rejected these semantics.']
                         + ['  ' + clean(f"{x['code']}: {x['message']}") for x in report['issues']]) + '\n'
    s = a['selected']
    rng = s['scenario_range_ms']
    c = s['components_ms']
    lines += [f"Selected {s['placement']} (C = CPU, A = accelerator): {_ms(s['estimated_ms'])}"
              + (f"; efficiency scenarios {rng[0]:.6f}..{rng[1]:.6f} ms" if rng else ''),
              f"  kernels {c['compute_ms']:.6f} + transfers {c['transfer_ms']:.6f} + syncs {c['sync_ms']:.6f} ms;"
              f" dominant: {s['dominant_cost']['component']}",
              f"  moved H2D {s['h2d_bytes']} B, D2H {s['d2h_bytes']} B; {s['syncs']} syncs;"
              f" peak device {s['device_peak_bytes']} / {report['profile']['accelerator']['memory_bytes']} B"
              + ('; weights resident' if report['resident_weights'] else ''),
              f"Search: {a['search']['mode']}, {a['search']['evaluated']} placements scored;"
              f" {a['infeasible_candidates']} infeasible; ties: {len(a['ties_with_selected'])}"]
    for label in ('cpu_only', 'maximal_supported'):
        b = a[label]
        lines.append(f"  {label}: {b['placement']} -> " + (_ms(b['estimated_ms']) if b['estimated_ms'] is not None
                     else 'infeasible: ' + clean('; '.join(r for f in b['failures'] for r in f['reasons']))))
    for row in s['nodes']:
        lines.append(f"  {row['node']} {clean(row['source_name'])} [{row['operator']}] on {row['backend']}: "
                     f"{row['operations']} ops, {row['traffic_bytes']} B, tile use {row['tile_utilization']:.0%}, "
                     f"{row['bottleneck']}-bound, {row['total_ms']:.6f} ms")
        if row['device_rejection_reasons']:
            lines.append('    CPU fallback: ' + '; '.join(row['device_rejection_reasons']))
    if report.get('sensitivity'):
        lines.append('Sensitivity (one parameter at a time; placement reselected):')
        for x in report['sensitivity']:
            note = ' (placement changed)' if x['changed_placement'] else ''
            lines.append(f"  {x['parameter']} x{x['factor']}: {_ms(x['estimated_ms'])} {x['placement']}{note}")
    return '\n'.join(lines + ['Assumptions and limits:'] + ['  ' + x for x in report['limitations']]) + '\n'
