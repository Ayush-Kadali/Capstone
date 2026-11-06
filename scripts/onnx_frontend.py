#!/usr/bin/env python3
"""Strict, bounded ONNX opset-13 import into the existing execution graph.

inspect_model(path) returns JSON-safe diagnostics; import_model(path, npz, new_dir)
creates graph.plg and import.json only after all model/input checks pass.
Constant-only bias broadcasting and Gemm weight transposition are explicit lowerings.
Runtime inputs are never baked into weights; external tensors are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_ELEMENTS = 65536
MAX_NODES = 64
MAX_GRAPH_BYTES = 16_000_000
MAX_CNN_VISITS = 64_000_000
# Analysis scope (static estimates only) keeps semantics checks but records the
# executor's size limits as non-fatal execution_limits, within these hard caps.
EXECUTION_LIMIT_CODES = {'SOURCE_NODE_LIMIT', 'NODE_LIMIT', 'TENSOR_LIMIT', 'INPUT_LIMIT', 'CNN_WORK', 'GENERATED_NODE_LIMIT'}
ANALYSIS_MAX_ELEMENTS = 1 << 26
ANALYSIS_MAX_SOURCE_NODES = 1024
# ponytail: estimate time grows ~cubically in generated nodes (beam search + 14
# sensitivity re-searches); 128 keeps one estimate well under a minute. Raise only after measuring.
ANALYSIS_MAX_NODES = 128


class FrontendError(ValueError):
    pass


def _deps():
    try:
        import numpy as np
        import onnx
    except ImportError as exc:
        raise FrontendError("ONNX frontend requires numpy and onnx in this Python environment") from exc
    return np, onnx


def _read(path):
    path = Path(path)
    with path.open('rb') as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise FrontendError('file exceeds 16 MiB import limit')
    return data


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _analyze(path, scope='execution'):
    if scope not in ('execution', 'analysis'):
        raise ValueError('scope must be execution or analysis')
    np, onnx = _deps()
    report = {'schema_version': 1, 'frontend': 'onnx-opset13-static-f32-packed-nchw',
              'supported': False, 'execution_validated': False,
              'hardware_compatibility_validated': False, 'issues': [], 'nodes': [],
              'inputs': [], 'outputs': [], 'tensors': [], 'scope': scope, 'execution_limits': []}

    def issue(code, message, node_index=None):
        item = {'code': code, 'message': message}
        if node_index is not None:
            item['node_index'] = node_index
        if scope == 'analysis' and code in EXECUTION_LIMIT_CODES:
            report['execution_limits'].append(item)
            return False
        report['issues'].append(item)
        return True

    try:
        data = _read(path)
        report['model_sha256'] = _hash(data)
        model = onnx.load_model_from_string(data)
    except Exception as exc:
        issue('MODEL_READ', str(exc))
        return report, None
    report['ir_version'] = model.ir_version
    report['opsets'] = [{'domain': o.domain, 'version': o.version} for o in model.opset_import]
    if len(model.opset_import) != 1 or model.opset_import[0].domain != '' or model.opset_import[0].version != 13:
        issue('OPSET', 'Exactly one standard-domain (empty domain) opset import, version 13, is required')
    if model.functions or model.training_info:
        issue('MODEL_FEATURE', 'Local functions and training graphs are unsupported')
    graph = model.graph
    if graph.sparse_initializer or graph.quantization_annotation:
        issue('MODEL_FEATURE', 'Sparse initializers and quantization annotations are unsupported')
    if len(graph.output) != 1:
        issue('OUTPUT_COUNT', 'Exactly one graph output is required')
    if len(graph.node) > 128 and (issue('SOURCE_NODE_LIMIT', 'The executor accepts at most 128 source nodes')
                                  or len(graph.node) > ANALYSIS_MAX_SOURCE_NODES):
        if scope == 'analysis':
            report['issues'].append({'code': 'ANALYSIS_LIMIT', 'message': f'At most {ANALYSIS_MAX_SOURCE_NODES} source nodes are analyzed'})
        return report, None
    operational = [n for n in graph.node if n.op_type != 'Constant']
    if not operational or len(operational) > ANALYSIS_MAX_NODES:
        report['issues'].append({'code': 'NODE_LIMIT', 'message': f'Analysis requires 1..{ANALYSIS_MAX_NODES} computational nodes (the executor accepts at most {MAX_NODES})'})
    elif len(operational) > MAX_NODES:
        issue('NODE_LIMIT', 'The bounded executor requires 1..64 computational nodes; exhaustive search is limited to 12')
    tensors = list(graph.initializer)
    for idx, node in enumerate(graph.node):
        report['nodes'].append({'source_index': idx, 'source_name': node.name,
                                'operator': node.op_type, 'domain': node.domain,
                                'inputs': list(node.input), 'outputs': list(node.output)})
        if node.domain != '' or node.op_type not in {'Constant', 'MatMul', 'Relu', 'Add', 'ReduceSum', 'Gemm', 'Conv', 'MaxPool', 'AveragePool', 'GlobalAveragePool', 'Flatten', 'Reshape'}:
            issue('OPERATOR', f'Unsupported operator {node.domain or "ai.onnx"}::{node.op_type}', idx)
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS,
                             onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS):
                issue('ATTRIBUTE', 'Graph and sparse-tensor attributes are unsupported', idx)
            if attr.type == onnx.AttributeProto.TENSOR:
                tensors.append(attr.t)
            tensors.extend(attr.tensors)
    for tensor in tensors:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            issue('EXTERNAL_DATA', 'External tensor data is unsupported and is never loaded')
        count = math.prod(tensor.dims)
        if count < 1 or count > ANALYSIS_MAX_ELEMENTS or any(d <= 0 for d in tensor.dims):
            report['issues'].append({'code': 'TENSOR_LIMIT', 'message': f'Tensor {tensor.name!r} must contain 1..{ANALYSIS_MAX_ELEMENTS} elements for analysis'})
        elif count > MAX_ELEMENTS:
            issue('TENSOR_LIMIT', f'Tensor {tensor.name!r} must contain 1..65536 elements')
    if report['issues']:
        # No shape inference/checker on unsupported graphs or external references.
        return report, None
    try:
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    except Exception as exc:
        issue('ONNX_VALIDATION', str(exc))
        return report, None
    values = {}
    constants = {}
    try:
        for tensor in graph.initializer:
            constants[tensor.name] = onnx.numpy_helper.to_array(tensor)
        for idx, node in enumerate(graph.node):
            if node.op_type == 'Constant':
                attrs = {a.name: a for a in node.attribute}
                if node.input or len(node.output) != 1 or set(attrs) != {'value'} or attrs['value'].type != onnx.AttributeProto.TENSOR:
                    issue('CONSTANT', 'Only Constant with one tensor-valued value attribute is supported', idx)
                else:
                    constants[node.output[0]] = onnx.numpy_helper.to_array(attrs['value'].t)
    except Exception as exc:
        issue('TENSOR_DATA', str(exc))
        return report, None
    for vi in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tt = vi.type.tensor_type
        shape = [int(d.dim_value) if d.HasField('dim_value') else None for d in tt.shape.dim]
        values[vi.name] = (tt.elem_type, shape)
    for name, array in constants.items():
        values[name] = (onnx.helper.np_dtype_to_tensor_dtype(array.dtype), list(array.shape))
        if array.dtype.kind not in 'fiu':
            issue('DTYPE', f'Constant {name!r} must have a supported numeric tensor dtype')
        elif not np.isfinite(array).all():
            issue('NONFINITE', f'Constant {name!r} contains nonfinite values')
    axes_names = {n.input[1] for n in graph.node if n.op_type in ('ReduceSum', 'Reshape') and len(n.input) == 2}
    data_names = set()
    for n in graph.node:
        if n.op_type == 'Constant':
            continue
        data_names.update(x for x in (n.input[:1] if n.op_type in ('ReduceSum', 'Reshape') else n.input) if x)
        data_names.update(n.output)
    data_names.update(x.name for x in graph.input)
    data_names.update(x.name for x in graph.output)
    # Unused constants are also checked and retained unless they are axis metadata.
    data_names.update(set(constants) - axes_names)
    for name in sorted(data_names):
        dtype, shape = values.get(name, (0, []))
        if dtype != onnx.TensorProto.FLOAT:
            issue('DTYPE', f'Tensor {name!r} must be float32')
        flexible_constant = name in constants and len(shape) <= 4
        if (len(shape) not in (2, 4) and not flexible_constant) or any(d is None or d <= 0 for d in shape):
            issue('SHAPE', f'Tensor {name!r} must have static positive rank-2 or NCHW rank-4 dimensions; found {shape}')
        elif math.prod(shape) > ANALYSIS_MAX_ELEMENTS:
            report['issues'].append({'code': 'TENSOR_LIMIT', 'message': f'Tensor {name!r} exceeds the analysis limit of {ANALYSIS_MAX_ELEMENTS} elements'})
        elif math.prod(shape) > MAX_ELEMENTS:
            issue('TENSOR_LIMIT', f'Tensor {name!r} exceeds 65536 elements')
        report['tensors'].append({'source_name': name, 'shape': shape, 'dtype': 'float32' if dtype == 1 else str(dtype)})
    if set(x.name for x in graph.input) & set(constants):
        issue('OVERRIDABLE_INITIALIZER', 'Initializers also declared as graph inputs are unsupported')
    if len(set(x.name for x in graph.input) | (set(constants) - axes_names)) > 64:
        issue('INPUT_LIMIT', 'At most 64 materialized input/weight tensors are supported')
    for vi in graph.input:
        report['inputs'].append({'source_name': vi.name, 'shape': values.get(vi.name, (0, []))[1], 'dtype': 'float32'})
    for vi in graph.output:
        report['outputs'].append({'source_name': vi.name, 'shape': values.get(vi.name, (0, []))[1]})
    if report['issues']:
        return report, None
    names = {name: f't{i:03d}' for i, name in enumerate(sorted(data_names))}
    normalized = []
    arrays = {}
    shapes = {}
    tensor_lowering = []
    node_mapping = []

    def packed(shape):
        return [shape[0], math.prod(shape[1:])]

    def materialize(name, idx):
        logical_shape = values[name][1]
        if len(logical_shape) not in (2, 4):
            issue('SHAPE', f'Tensor {name!r} must be rank 2 or 4 for this operation', idx)
            return None
        generated = names[name]
        shape = packed(logical_shape)
        shapes[generated] = shape
        if name in constants:
            arrays[generated] = constants[name].reshape(shape)
        return generated

    def transformed(name, array, idx, suffix, action):
        generated = f'c{idx:03d}_{suffix}'
        arrays[generated] = np.ascontiguousarray(array).reshape(packed(array.shape))
        shapes[generated] = packed(array.shape)
        tensor_lowering.append({'source_name': name, 'generated_tensor': generated,
                                'source_shape': values[name][1], 'logical_shape': list(array.shape), 'shape': packed(array.shape),
                                'transformation': action, 'source_index': idx})
        return generated

    def expand(name, output_shape, idx, suffix):
        if name not in constants:
            issue('RUNTIME_BROADCAST', 'Broadcast bias must be a constant, never a runtime input', idx)
            return None
        try:
            expanded = np.broadcast_to(constants[name], output_shape)
        except ValueError:
            issue('BIAS_SHAPE', 'Constant bias cannot broadcast to the output shape', idx)
            return None
        return transformed(name, expanded, idx, suffix, 'constant_bias_broadcast')

    def spatial_valid(logical, output_shape, channels, kernel, dilation, stride, pad, idx):
        effective = [(kernel[i] - 1) * dilation[i] + 1 for i in range(2)]
        if any(x > MAX_ELEMENTS for x in effective):
            issue('CNN_EFFECTIVE_KERNEL', 'Effective spatial kernel dimensions must not exceed 65536', idx)
            return False
        if any(pad[i] >= effective[i % 2] for i in range(4)) or any(logical[i + 2] + pad[i] + pad[i + 2] < effective[i] for i in range(2)):
            issue('CNN_PADDING', 'Padding must be smaller than the effective kernel and produce a nonempty output', idx)
            return False
        expected = [logical[0], channels] + [(logical[i + 2] + pad[i] + pad[i + 2] - effective[i]) // stride[i] + 1 for i in range(2)]
        if expected != output_shape:
            issue('CNN_OUTPUT', 'Inferred ONNX shape differs from native spatial-kernel dimensions', idx)
            return False
        return True

    def emit(op, inputs, output, shape, idx, node, suffix='', parameters=None):
        generated_node = f'n{idx:03d}' + suffix
        normalized.append({'generated_node': generated_node, 'operator': op, 'inputs': inputs,
                           'output': output, 'shape': packed(shape), 'logical_shape': shape,
                           'source_index': idx, 'source_name': node.name, 'parameters': parameters or []})
        shapes[output] = packed(shape)
        return generated_node

    for vi in graph.input:
        materialize(vi.name, None)
    for idx, node in enumerate(graph.node):
        attrs = {a.name: a for a in node.attribute}
        mapped = {'source_index': idx, 'source_name': node.name, 'operator': node.op_type,
                  'generated_node': None, 'generated_nodes': []}
        node_mapping.append(mapped)
        if node.op_type == 'Constant':
            mapped['lowering'] = 'materialized_initializer'
            continue
        arities = (2, 3) if node.op_type in ('Gemm', 'Conv') else (1,) if node.op_type in ('Relu', 'MaxPool', 'AveragePool', 'GlobalAveragePool', 'Flatten') else (2,)
        if len(node.input) not in arities or not all(node.input[:2] if node.op_type in ('Gemm', 'Conv') else node.input) or len(node.output) != 1:
            issue('ARITY', 'Operator input/output arity is outside the supported subset', idx)
            continue
        out_name = node.output[0]
        output_shape = values[out_name][1]
        output = names[out_name]
        before = len(normalized)
        if node.op_type not in ('ReduceSum', 'Gemm', 'Conv', 'MaxPool', 'AveragePool', 'Flatten') and attrs:
            issue('ATTRIBUTE', 'No attributes are supported on this operator', idx)
            continue
        if node.op_type == 'Gemm':
            attr_values = {key: onnx.helper.get_attribute_value(a) for key, a in attrs.items()}
            if set(attrs) - {'alpha', 'beta', 'transA', 'transB'}:
                issue('ATTRIBUTE', 'Unsupported Gemm attributes', idx)
                continue
            if attr_values.get('alpha', 1.0) != 1.0 or attr_values.get('beta', 1.0) != 1.0:
                issue('GEMM_SCALE', 'Gemm currently requires alpha=1 and beta=1', idx)
                continue
            if attr_values.get('transA', 0) != 0 or attr_values.get('transB', 0) not in (0, 1):
                issue('GEMM_TRANSPOSE', 'Gemm requires transA=0 and transB=0 or 1', idx)
                continue
            if node.input[1] not in constants or len(values[node.input[1]][1]) != 2:
                issue('GEMM_WEIGHT', 'Gemm B must be a constant rank-2 weight matrix', idx)
                continue
            if len(values[node.input[0]][1]) != 2:
                issue('GEMM_RANK', 'Gemm requires rank-2 A', idx)
                continue
            left = materialize(node.input[0], idx)
            if left is None:
                continue
            weight = constants[node.input[1]]
            if attr_values.get('transB', 0):
                right = transformed(node.input[1], weight.T, idx, 'weight', 'constant_weight_transpose')
            else:
                right = materialize(node.input[1], idx)
            bias_name = node.input[2] if len(node.input) == 3 else ''
            bias = expand(bias_name, output_shape, idx, 'bias') if bias_name else None
            if bias_name and bias is None:
                continue
            intermediate = f'g{idx:03d}_matmul' if bias else output
            emit('MatMul', [left, right], intermediate, output_shape, idx, node, '_matmul')
            if bias:
                emit('Add', [intermediate, bias], output, output_shape, idx, node, '_bias')
        elif node.op_type in ('Conv', 'MaxPool', 'AveragePool', 'GlobalAveragePool'):
            logical = values[node.input[0]][1]
            if len(logical) != 4 or len(output_shape) != 4:
                issue('CNN_RANK', 'Spatial operators require static NCHW rank-4 input and output', idx)
                continue
            attr_values = {key: onnx.helper.get_attribute_value(a) for key, a in attrs.items()}
            allowed = {'auto_pad', 'kernel_shape', 'strides', 'pads'}
            if node.op_type == 'Conv':
                allowed |= {'group', 'dilations'}
            elif node.op_type == 'MaxPool':
                allowed |= {'ceil_mode', 'dilations', 'storage_order'}
            elif node.op_type == 'AveragePool':
                allowed |= {'ceil_mode', 'count_include_pad'}
            else:
                allowed = set()
            if set(attrs) - allowed:
                issue('CNN_ATTRIBUTE', 'Unsupported spatial-operator attribute', idx)
                continue
            if attr_values.get('auto_pad', b'NOTSET') not in (b'NOTSET', b''):
                issue('AUTO_PAD', 'Only explicit padding (auto_pad=NOTSET) is supported', idx)
                continue
            operand = materialize(node.input[0], idx)
            if node.op_type == 'GlobalAveragePool':
                emit('GlobalAveragePool', [operand], output, output_shape, idx, node, parameters=logical)
            else:
                stride = list(attr_values.get('strides', [1, 1]))
                pad = list(attr_values.get('pads', [0, 0, 0, 0]))
                dilation = list(attr_values.get('dilations', [1, 1]))
                if len(stride) != 2 or len(pad) != 4 or len(dilation) != 2 or any(not 1 <= x <= 64 for x in stride + dilation) or any(not 0 <= x <= 65536 for x in pad):
                    issue('CNN_PARAMETERS', 'Expected two strides/dilations in 1..64 and four pads in 0..65536', idx)
                    continue
                if node.op_type == 'Conv':
                    weight_name = node.input[1]
                    if weight_name not in constants or constants[weight_name].ndim != 4:
                        issue('CONV_WEIGHT', 'Conv requires constant OIHW rank-4 weights', idx)
                        continue
                    weight_shape = list(constants[weight_name].shape)
                    kernel = list(attr_values.get('kernel_shape', weight_shape[2:]))
                    group = attr_values.get('group', 1)
                    if kernel != weight_shape[2:] or group < 1 or logical[1] % group or weight_shape[0] % group or weight_shape[1] != logical[1] // group:
                        issue('CONV_PARAMETERS', 'Conv kernel/group/channel dimensions are inconsistent', idx)
                        continue
                    if not spatial_valid(logical, output_shape, weight_shape[0], kernel, dilation, stride, pad, idx):
                        continue
                    if (math.prod(output_shape) * math.prod(kernel) * (logical[1] // group) > MAX_CNN_VISITS
                            and issue('CNN_WORK', 'Conv exceeds 64000000 kernel visits per execution', idx)):
                        continue
                    weight_id = materialize(weight_name, idx)
                    bias_name = node.input[2] if len(node.input) == 3 else ''
                    bias = None
                    if bias_name:
                        if bias_name not in constants or constants[bias_name].shape != (weight_shape[0],):
                            issue('CONV_BIAS', 'Conv bias must be a constant vector with one value per output channel', idx)
                            continue
                        expanded = np.broadcast_to(constants[bias_name].reshape(1, weight_shape[0], 1, 1), output_shape)
                        bias = transformed(bias_name, expanded, idx, 'bias', 'constant_channel_bias_broadcast')
                    intermediate = f'g{idx:03d}_conv' if bias else output
                    parameters = logical + [weight_shape[0]] + kernel + stride + pad + [group] + dilation
                    emit('Conv2D', [operand, weight_id], intermediate, output_shape, idx, node, '_conv', parameters)
                    if bias:
                        emit('Add', [intermediate, bias], output, output_shape, idx, node, '_bias')
                else:
                    kernel = list(attr_values.get('kernel_shape', []))
                    count_pad = attr_values.get('count_include_pad', 0)
                    if len(kernel) != 2 or math.prod(kernel) > MAX_ELEMENTS or any(not 1 <= x <= 65536 for x in kernel) or dilation != [1, 1] or attr_values.get('ceil_mode', 0) != 0 or attr_values.get('storage_order', 0) != 0 or count_pad not in (0, 1):
                        issue('POOL_PARAMETERS', 'Pooling requires positive kernel area <=65536, dilation1, ceil_mode0, storage_order0, and count_include_pad0/1', idx)
                        continue
                    if not spatial_valid(logical, output_shape, logical[1], kernel, dilation, stride, pad, idx):
                        continue
                    if math.prod(output_shape) * math.prod(kernel) > MAX_CNN_VISITS:
                        if issue('CNN_WORK', 'Pooling exceeds 64000000 kernel visits per execution', idx):
                            continue
                    op = 'MaxPool2D' if node.op_type == 'MaxPool' else 'AveragePool2D'
                    emit(op, [operand], output, output_shape, idx, node, parameters=logical + kernel + stride + pad + [count_pad])
        elif node.op_type in ('Flatten', 'Reshape'):
            logical = values[node.input[0]][1]
            if node.op_type == 'Flatten':
                if set(attrs) - {'axis'} or (attrs.get('axis') is not None and attrs['axis'].i != 1):
                    issue('FLATTEN_AXIS', 'Flatten currently supports axis=1 only', idx)
                    continue
            else:
                target = constants.get(node.input[1])
                if target is None or target.dtype != np.dtype('int64') or target.ndim != 1 or len(target) not in (2, 4):
                    issue('RESHAPE_SHAPE', 'Reshape requires a constant int64 shape vector describing rank2 or rank4', idx)
                    continue
            if math.prod(logical) != math.prod(output_shape):
                issue('RESHAPE_SIZE', 'Reshape must preserve the contiguous element count', idx)
                continue
            operand = materialize(node.input[0], idx)
            if operand is not None:
                emit('Reshape', [operand], output, output_shape, idx, node, parameters=packed(output_shape))
        elif node.op_type == 'Add':
            left_name, right_name = node.input
            if values[left_name] == values[right_name] and len(values[left_name][1]) in (2, 4):
                left, right = materialize(left_name, idx), materialize(right_name, idx)
            else:
                constant_operands = [name in constants for name in node.input]
                if sum(constant_operands) != 1:
                    issue('ADD_BROADCAST', 'Broadcast Add requires one constant operand and an unchanged rank-2 or rank-4 runtime operand', idx)
                    continue
                const_pos = constant_operands.index(True)
                runtime_name = node.input[1 - const_pos]
                if values[runtime_name][1] != output_shape or len(output_shape) not in (2, 4):
                    issue('ADD_BROADCAST', 'Broadcasting the runtime operand is unsupported', idx)
                    continue
                constant = expand(node.input[const_pos], output_shape, idx, 'bias')
                runtime = materialize(runtime_name, idx)
                if constant is None or runtime is None:
                    continue
                left, right = (constant, runtime) if const_pos == 0 else (runtime, constant)
            emit('Add', [left, right], output, output_shape, idx, node)
        elif node.op_type == 'ReduceSum':
            if set(attrs) - {'keepdims', 'noop_with_empty_axes'}:
                issue('ATTRIBUTE', 'Unsupported ReduceSum attributes', idx)
                continue
            keep = attrs.get('keepdims')
            noop = attrs.get('noop_with_empty_axes')
            if (keep is not None and keep.i != 1) or (noop is not None and noop.i != 0):
                issue('REDUCE_ATTRIBUTES', 'ReduceSum requires keepdims=1 and noop_with_empty_axes=0', idx)
                continue
            if len(values[node.input[0]][1]) != 2:
                issue('REDUCE_RANK', 'ReduceSum currently requires rank-2 input', idx)
                continue
            axis = constants.get(node.input[1])
            if axis is None or axis.dtype != np.dtype('int64') or axis.shape != (1,) or int(axis[0]) not in (1, -1):
                issue('REDUCE_AXES', 'ReduceSum requires a constant int64 axes tensor [1] (or [-1])', idx)
                continue
            operand = materialize(node.input[0], idx)
            if operand is not None:
                emit('RowSum', [operand], output, output_shape, idx, node)
        else:
            if node.op_type == 'MatMul' and any(len(values[name][1]) != 2 for name in node.input):
                issue('MATMUL_RANK', 'MatMul currently requires two rank-2 operands; batched MatMul is unsupported', idx)
                continue
            operands = [materialize(name, idx) for name in node.input]
            if all(x is not None for x in operands):
                emit(node.op_type, operands, output, output_shape, idx, node)
        mapped['generated_nodes'] = [n['generated_node'] for n in normalized[before:]]
        mapped['generated_node'] = mapped['generated_nodes'][0] if mapped['generated_nodes'] else None
        mapped['generated_operators'] = [n['operator'] for n in normalized[before:]]
    for vi in graph.output:
        materialize(vi.name, None)
    if len(normalized) > ANALYSIS_MAX_NODES:
        report['issues'].append({'code': 'GENERATED_NODE_LIMIT', 'message': f'Lowering produces {len(normalized)} computational nodes; analysis maximum is {ANALYSIS_MAX_NODES}'})
    elif len(normalized) > MAX_NODES:
        issue('GENERATED_NODE_LIMIT', f'Lowering produces {len(normalized)} computational nodes; maximum is 64')
    if len(arrays) + len(graph.input) > 64:
        issue('INPUT_LIMIT', 'Lowering exceeds 64 materialized input/weight tensors')
    report['normalized_nodes'] = normalized
    report['normalized_tensors'] = [{'name': name, 'shape': shape} for name, shape in sorted(shapes.items())]
    report['node_mapping'] = node_mapping
    report['tensor_mapping'] = {name: generated for name, generated in names.items() if generated in shapes}
    report['tensor_lowering'] = tensor_lowering
    report['generated_node_count'] = len(normalized)
    report['supported'] = not report['issues']
    report['functionally_executable'] = report['supported'] and not report['execution_limits']
    report['storage_contract'] = 'Logical rank-4 tensors use contiguous NCHW storage packed as [N,C*H*W]; rank-2 tensors use row-major storage. Native spatial operations retain logical dimensions in command parameters.'
    report['support_scope'] = 'Importability into this software executor; not a hardware feasibility or correctness claim'
    return report, {'model': model, 'constants': constants, 'values': values, 'axes_names': axes_names,
                    'model_bytes': data, 'arrays': arrays} if report['supported'] else None


def inspect_model(model_path, scope='execution'):
    """scope='analysis' accepts larger static models for estimates; never for execution."""
    return _analyze(model_path, scope)[0]


def import_model(model_path, inputs_path, out_dir):
    np, onnx = _deps()
    report, state = _analyze(model_path)
    if not report['supported']:
        raise FrontendError(json.dumps(report, ensure_ascii=True))
    data = _read(inputs_path)
    required = {i['source_name'] for i in report['inputs']}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > 64 or sum(m.file_size for m in members) > MAX_FILE_BYTES:
                raise FrontendError('NPZ exceeds 64 arrays or 16 MiB uncompressed limit')
            if len({m.filename for m in members}) != len(members):
                raise FrontendError('NPZ contains duplicate members')
            if {m.filename for m in members} != {name + '.npy' for name in required}:
                raise FrontendError(f'NPZ keys must exactly match graph inputs: {sorted(required)}')
            # NumPy can allocate based on a forged NPY header before discovering
            # a truncated payload. Validate headers and byte lengths first.
            for member in members:
                name = member.filename[:-4]
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise FrontendError('Only NPY versions 1.0 and 2.0 are supported')
                    if dtype != np.dtype('float32') or list(shape) != state['values'][name][1]:
                        raise FrontendError(f'Input {name!r} must be float32 with shape {state["values"][name][1]}')
                    if member.file_size - stream.tell() != math.prod(shape) * dtype.itemsize:
                        raise FrontendError(f'Input {name!r} NPY payload length differs from its validated shape')
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != required or len(archive.files) != len(required):
                raise FrontendError(f'NPZ keys must exactly match graph inputs: {sorted(required)}')
            supplied = {name: archive[name] for name in archive.files}
    except FrontendError:
        raise
    except Exception as exc:
        raise FrontendError(f'Invalid non-pickle NPZ input: {exc}') from exc
    for name, array in supplied.items():
        if array.dtype != np.dtype('float32') or list(array.shape) != state['values'][name][1]:
            raise FrontendError(f'Input {name!r} must be float32 with shape {state["values"][name][1]}')
        if not np.isfinite(array).all():
            raise FrontendError(f'Input {name!r} contains nonfinite values')
    names = report['tensor_mapping']
    arrays = {**state['arrays'], **{names[k]: v.reshape(v.shape[0], -1) for k, v in supplied.items()}}
    lines = ['# Generated from validated ONNX opset 13; source mapping is in import.json.']
    for name in sorted(arrays):
        array = arrays[name]
        lines.append(f'tensor {name} f32 {array.shape[0]} {array.shape[1]} ' +
                     ' '.join(format(float(x), '.9g') for x in array.flat))
    for node in report['normalized_nodes']:
        op = {'MatMul': 'matmul', 'Relu': 'relu', 'Add': 'add', 'RowSum': 'rowsum', 'Conv2D': 'conv2d', 'MaxPool2D': 'maxpool2d',
              'AveragePool2D': 'averagepool2d', 'GlobalAveragePool': 'globalaveragepool', 'Reshape': 'reshape'}[node['operator']]
        lines.append(' '.join([op, node['generated_node']] + node['inputs'] + [node['output']] + [str(x) for x in node['parameters']]))
    lines.append(f'output {names[report["outputs"][0]["source_name"]]}')
    plg = ('\n'.join(lines) + '\n').encode('utf-8')
    if len(plg) > MAX_GRAPH_BYTES:
        raise FrontendError('Materialized graph exceeds the existing executor 16000000-byte file limit')
    manifest = {**report, 'inputs_sha256': _hash(data), 'graph_sha256': _hash(plg),
                'graph_file': 'graph.plg',
                'tool_versions': {'onnx': onnx.__version__, 'numpy': np.__version__},
                'checks': ['onnx_checker_full', 'strict_shape_inference', 'bounded_subset_semantics',
                           'finite_float32_inputs', 'exact_input_names_and_shapes'],
                'numerical_execution_performed': False}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        (out / 'graph.plg').write_bytes(plg)
        (out / 'import.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n')
    except Exception:
        for name in ('graph.plg', 'import.json'):
            (out / name).unlink(missing_ok=True)
        out.rmdir()
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--model', required=True)
    inspect.add_argument('--out', required=True)
    imp = sub.add_parser('import')
    imp.add_argument('--model', required=True)
    imp.add_argument('--inputs', required=True)
    imp.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            result = inspect_model(args.model)
            with Path(args.out).open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
        else:
            result = import_model(args.model, args.inputs, args.out)
        print(json.dumps({'supported': result['supported'], 'out': args.out,
                          'execution_validated': False, 'hardware_compatibility_validated': False}))
        return 0
    except (FrontendError, OSError) as exc:
        print(f'ONNX_FRONTEND: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
