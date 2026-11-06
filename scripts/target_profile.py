#!/usr/bin/env python3
"""Strict software-target profiles: illustrative constraints, never hardware prediction."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import sys

CNN_OPS = ('Conv2D', 'MaxPool2D', 'AveragePool2D', 'GlobalAveragePool', 'Reshape')
OPS = ('MatMul', 'Add', 'Relu', 'RowSum', 'MatMulRelu')
LIMITATION = ('Node footprint counts operands and output only; it excludes graph lifetimes, '
              'idle pool reservations and vendor workspace. Acceptance is not a whole-model '
              'memory fit, vendor compilation, hardware compatibility or latency guarantee.')

class ProfileError(ValueError):
    pass

def _keys(obj, keys, where):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ProfileError(f'{where}: expected exactly {", ".join(keys)}')

def _integer(value, low, high, where):
    if type(value) is not int or not low <= value <= high:
        raise ProfileError(f'{where}: expected integer in [{low}, {high}]')

def _number(value, low, high, where):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ProfileError(f'{where}: expected finite number in [{low}, {high}]')

def validate_profile(profile):
    """Validate the entire contract; unknown fields cannot silently imply extra support."""
    _keys(profile, ('schema_version', 'id', 'description', 'target', 'evidence', 'tensor',
                    'operators', 'memory', 'costs'), 'profile')
    if type(profile['schema_version']) is not int or profile['schema_version'] not in (1, 2):
        raise ProfileError('schema_version: only 1 and 2 are supported')
    if not isinstance(profile['id'], str) or not re.fullmatch(r'software-[a-z0-9-]{1,80}', profile['id']):
        raise ProfileError('id: must start software- and use lowercase letters, digits or hyphens')
    if not isinstance(profile['description'], str) or not 1 <= len(profile['description']) <= 1000:
        raise ProfileError('description: expected 1..1000 characters')
    target = profile['target']
    _keys(target, ('kind', 'backend'), 'target')
    if target != {'kind': 'software-device', 'backend': 'software'}:
        raise ProfileError('target: version 1 supports software-device/software only, not GPU/NPU/TPU emulation')
    evidence = profile['evidence']
    _keys(evidence, ('level', 'source', 'hardware_validated'), 'evidence')
    if evidence['level'] != 'hypothetical' or evidence['hardware_validated'] is not False:
        raise ProfileError('evidence: version 1 permits hypothetical, hardware_validated=false only')
    if not isinstance(evidence['source'], str) or not 1 <= len(evidence['source']) <= 2000:
        raise ProfileError('evidence.source: expected nonempty explanatory provenance')
    if profile['tensor'] != {'dtype': 'f32', 'rank': 2, 'layout': 'contiguous', 'max_elements': 65536}:
        raise ProfileError('tensor: engine requires f32, rank 2, contiguous, max_elements 65536')
    # Equality would otherwise accept True == 1 or 2.0 == 2.
    if type(profile['tensor']['rank']) is not int or type(profile['tensor']['max_elements']) is not int:
        raise ProfileError('tensor: rank and max_elements must be integers')
    ops = profile['operators']
    _keys(ops, OPS + (CNN_OPS if profile['schema_version'] == 2 else ()), 'operators')
    if any(type(value) is not bool for value in ops.values()):
        raise ProfileError('operators: support values must be booleans')
    if not all(ops[op] for op in ('MatMul', 'Add', 'RowSum')) or ops['Relu'] != ops['MatMulRelu']:
        raise ProfileError('operators: engine only honors a shared Relu/MatMulRelu toggle; other operators remain enabled')
    if profile['schema_version'] == 2 and len({ops[op] for op in CNN_OPS}) != 1:
        raise ProfileError('operators: engine requires a shared CNN/Reshape capability toggle')
    memory = profile['memory']
    _keys(memory, ('device_bytes', 'local_bytes', 'tile_size'), 'memory')
    # Portable bound fits 32-bit size_t too, independent of the development host.
    _integer(memory['device_bytes'], 0, 2**32-1, 'device_bytes')
    _integer(memory['local_bytes'], 0, 2**32-1, 'local_bytes')
    _integer(memory['tile_size'], 0, 64, 'tile_size')
    if 12 * memory['tile_size'] ** 2 > memory['local_bytes']:
        raise ProfileError('memory: tiled MatMul requires 3 * tile_size^2 * 4 scratch bytes')
    costs = profile['costs']
    _keys(costs, ('transfer_fixed_ms', 'bandwidth_bytes_per_ms', 'launch_ms', 'sync_ms'), 'costs')
    for key in ('transfer_fixed_ms', 'launch_ms', 'sync_ms'):
        _number(costs[key], 0, 1e9, key)
    _number(costs['bandwidth_bytes_per_ms'], 1e-9, 1e300, 'bandwidth_bytes_per_ms')
    return profile

def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ProfileError(f'duplicate JSON key: {key}')
        obj[key] = value
    return obj

def load_profile(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ProfileError('profile exceeds 65536 bytes')
    return validate_profile(json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object))

def engine_flags(profile):
    """Arguments for the existing engine; profile cannot select external hardware."""
    validate_profile(profile)
    flags = ['--backend', 'software']
    for key, flag in (('device_bytes', '--device-bytes'), ('local_bytes', '--local-bytes'), ('tile_size', '--tile-size')):
        flags += [flag, str(profile['memory'][key])]
    for key, flag in (('transfer_fixed_ms', '--boundary-ms'), ('bandwidth_bytes_per_ms', '--bandwidth'),
                      ('launch_ms', '--launch-ms'), ('sync_ms', '--sync-ms')):
        flags += [flag, str(profile['costs'][key])]
    if profile['operators']['Relu']:
        flags += ['--relu-supported']
    if profile['schema_version'] == 2 and profile['operators']['Conv2D']:
        flags += ['--cnn-supported']
    return flags

def _shape(shape):
    return (isinstance(shape, (list, tuple)) and len(shape) == 2
            and all(type(x) is int and x > 0 for x in shape) and math.prod(shape) <= 65536)

def evaluate_node(op, shape, input_shapes, profile, *, dtype='f32', layout='contiguous',
                  input_names=None, output_name=None):
    """Evaluate normalized IR semantics, not unvalidated ONNX attributes or broadcasting.

    Names are optional. Without names, repeated operands are counted separately as a
    conservative upper bound. With names, alias shapes must agree and inputs are deduped.
    """
    validate_profile(profile)
    result = {'accepted': False, 'reason': '', 'profile_id': profile['id'],
              'evidence_level': 'hypothetical', 'hardware_validated': False,
              'operand_output_bytes': None, 'memory_estimate': 'distinct_operands' if input_names is not None else 'operand_upper_bound',
              'limitations': LIMITATION}
    def reject(reason):
        result['reason'] = reason
        return result
    if op in CNN_OPS:
        return reject('CNN attributes require authoritative core capability analysis; use explore_model.py')
    if op not in OPS:
        return reject('unsupported normalized IR operator; ONNX attributes must be validated by the importer')
    if dtype != 'f32' or layout != 'contiguous':
        return reject('only contiguous f32 tensors are supported')
    if not isinstance(input_shapes, (list, tuple)) or not _shape(shape) or not all(_shape(s) for s in input_shapes):
        return reject('static positive rank-2 shapes with at most 65536 elements are required')
    arity = 1 if op in ('Relu', 'RowSum') else 2
    if len(input_shapes) != arity:
        return reject('wrong input arity')
    a = list(input_shapes[0])
    if op in ('MatMul', 'MatMulRelu'):
        b = input_shapes[1]
        if a[1] != b[0]:
            return reject('MatMul inner dimensions do not match')
        expected = [a[0], b[1]]
    elif op == 'Add':
        if a != list(input_shapes[1]):
            return reject('Add requires identical shapes; broadcasting is not implemented')
        expected = a
    else:
        expected = [a[0], 1] if op == 'RowSum' else a
    if list(shape) != expected:
        return reject('output shape does not match operator semantics')
    if (input_names is None) != (output_name is None):
        return reject('provide both input_names and output_name or neither')
    shapes = list(input_shapes) + [shape]
    if input_names is not None:
        if not isinstance(input_names, (list, tuple)) or len(input_names) != arity:
            return reject('input_names must match input arity')
        names = list(input_names) + [output_name]
        if any(not isinstance(n, str) or not n for n in names) or output_name in input_names:
            return reject('tensor names must be nonempty and output must be distinct from inputs')
        distinct = {}
        for name, s in zip(names, shapes):
            if name in distinct and list(s) != distinct[name]:
                return reject('repeated input name has contradictory shapes')
            distinct[name] = list(s)
        shapes = list(distinct.values())
    footprint = sum(math.prod(s) * 4 for s in shapes)
    result['operand_output_bytes'] = footprint
    if not profile['operators'][op]:
        return reject('ReLU kernel is absent from this software-device profile')
    if footprint > profile['memory']['device_bytes']:
        return reject('operand/output footprint exceeds device budget' if input_names is not None else
                      'conservative operand/output estimate exceeds budget; provide tensor identities to resolve repeated operands')
    result.update(accepted=True, reason='normalized operator semantics and node operand/output footprint accepted')
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'flags', 'analyze-node'))
    parser.add_argument('profile')
    parser.add_argument('--node', help='JSON file with op, shape, input_shapes and optional dtype/layout/input_names/output_name')
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
        if args.action == 'validate':
            output = {'valid': True, 'profile_id': profile['id'], 'evidence': profile['evidence'], 'limitations': LIMITATION}
        elif args.action == 'flags':
            output = {'profile_id': profile['id'], 'argv': engine_flags(profile), 'execution_backend': 'software',
                      'performance_claim': 'Illustrative cost assumptions; not predicted physical accelerator latency.'}
        else:
            if not args.node:
                raise ProfileError('analyze-node requires --node JSON_FILE')
            path = Path(args.node)
            if path.stat().st_size > 65536:
                raise ProfileError('node JSON exceeds 65536 bytes')
            node = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
            if not isinstance(node, dict) or not {'op', 'shape', 'input_shapes'} <= set(node) or set(node) - {'op', 'shape', 'input_shapes', 'dtype', 'layout', 'input_names', 'output_name'}:
                raise ProfileError('invalid node fields')
            output = evaluate_node(profile=profile, **node)
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(f'E_PROFILE: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
