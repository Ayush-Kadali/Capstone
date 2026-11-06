#!/usr/bin/env python3
"""Semantic/rejection contracts for the bounded ONNX importer.

Run with a Python containing numpy/onnx. Optional --binary exercises the real
C++ parser/executor and compares final tensor values with ONNX ReferenceEvaluator.
"""
import argparse
import importlib.util
import io
import zipfile
from unittest import mock
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
if not importlib.util.find_spec('onnx') or not importlib.util.find_spec('numpy'):
    print('SKIP: onnx_frontend_tests requires optional onnx and numpy dependencies')
    sys.exit(77)
import numpy as np
import onnx
from onnx import TensorProto as T, helper as H, numpy_helper as N
from onnx.reference import ReferenceEvaluator
from onnx_frontend import inspect_model, import_model, FrontendError
from make_onnx_examples import create_examples

BINARY = None


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        create_examples(self.root / 'examples')
        self.model = self.root / 'examples/supported_dense.onnx'
        self.inputs = self.root / 'examples/supported_dense.inputs.npz'

    def tearDown(self):
        self.temp.cleanup()

    def save(self, model):
        path = self.root / 'variant.onnx'
        onnx.save(model, path)
        return path

    def reject(self, model, code=None):
        path = self.save(model)
        report = inspect_model(path)
        self.assertFalse(report['supported'], report)
        if code:
            self.assertIn(code, [i['code'] for i in report['issues']], report)
        with self.assertRaises(FrontendError):
            import_model(path, self.inputs, self.root / 'failed')
        self.assertFalse((self.root / 'failed').exists())
        return report

    def test_supported_provenance_safe_names_and_no_overwrite(self):
        model = onnx.load(self.model)
        model.graph.node[0].name = 'unsafe\n# source node'
        self.model = self.save(model)
        report = inspect_model(self.model)
        self.assertTrue(report['supported'], report)
        result = import_model(self.model, self.inputs, self.root / 'imported')
        self.assertFalse(result['execution_validated'])
        self.assertFalse(result['hardware_compatibility_validated'])
        plg = (self.root / 'imported/graph.plg').read_text()
        self.assertNotIn('unsafe', plg)
        self.assertEqual(result['node_mapping'][0]['source_name'], 'unsafe\n# source node')
        self.assertEqual(result, json.loads((self.root / 'imported/import.json').read_text()))
        with self.assertRaises(FileExistsError):
            import_model(self.model, self.inputs, self.root / 'imported')
        self.assertEqual(plg, (self.root / 'imported/graph.plg').read_text())

    def test_broadcast_is_valid_onnx_but_rejected(self):
        model = onnx.load(self.root / 'examples/unsupported_broadcast.onnx')
        onnx.checker.check_model(model, full_check=True)
        self.reject(model, 'ADD_BROADCAST')

    def test_opset_and_custom_domain(self):
        for version in (12, 14, 18):
            with self.subTest(version=version):
                model = onnx.load(self.model)
                model.opset_import[0].version = version
                self.reject(model, 'OPSET')
        model = onnx.load(self.model)
        model.graph.node[0].domain = 'custom'
        self.reject(model, 'OPERATOR')

    def test_external_initializer_is_not_followed(self):
        model = onnx.load(self.model)
        tensor = model.graph.initializer[0]
        tensor.ClearField('raw_data')
        tensor.data_location = T.EXTERNAL
        tensor.external_data.add(key='location', value='/nonexistent/private/weights.bin')
        self.reject(model, 'EXTERNAL_DATA')

    def test_unsupported_operator_and_extra_attribute(self):
        model = onnx.load(self.model)
        model.graph.node[1].op_type = 'Sigmoid'
        self.reject(model, 'OPERATOR')
        model = onnx.load(self.model)
        model.graph.node[1].attribute.append(H.make_attribute('alpha', 0.2))
        self.reject(model, 'ONNX_VALIDATION')

    def test_dynamic_rank_dtype_and_size(self):
        for kind in ('dynamic', 'rank', 'dtype', 'size'):
            with self.subTest(kind=kind):
                xshape = ['batch', 2] if kind == 'dynamic' else ([1, 2, 2] if kind == 'rank' else [257, 256] if kind == 'size' else [2, 2])
                dtype = T.DOUBLE if kind == 'dtype' else T.FLOAT
                model = H.make_model(H.make_graph([H.make_node('Relu', ['X'], ['Y'])], 'bad',
                    [H.make_tensor_value_info('X', dtype, xshape)],
                    [H.make_tensor_value_info('Y', dtype, xshape)]),
                    opset_imports=[H.make_opsetid('', 13)], ir_version=9)
                self.reject(model, {'dynamic': 'SHAPE', 'rank': 'SHAPE', 'dtype': 'DTYPE', 'size': 'TENSOR_LIMIT'}[kind])

    def test_input_archive_contract(self):
        cases = [dict(wrong=np.ones((2, 3), dtype=np.float32)),
                 dict(features=np.ones((2, 3), dtype=np.float64)),
                 dict(features=np.ones((3, 2), dtype=np.float32)),
                 dict(features=np.full((2, 3), np.inf, dtype=np.float32)),
                 dict(features=np.array([[object()]], dtype=object))]
        for idx, arrays in enumerate(cases):
            with self.subTest(idx=idx):
                path = self.root / f'bad-{idx}.npz'
                np.savez(path, **arrays)
                with self.assertRaises(FrontendError):
                    import_model(self.model, path, self.root / f'bad-{idx}')
                self.assertFalse((self.root / f'bad-{idx}').exists())

    def test_forged_npy_header_rejected_before_numpy_load(self):
        payload = io.BytesIO()
        np.lib.format.write_array_header_1_0(payload, {
            'descr': '<f4', 'fortran_order': False, 'shape': (1000000000, 1)})
        forged = self.root / 'forged.npz'
        with zipfile.ZipFile(forged, 'w') as archive:
            archive.writestr('features.npy', payload.getvalue())
        with mock.patch('numpy.load', side_effect=AssertionError('unsafe allocation path reached')):
            with self.assertRaisesRegex(FrontendError, 'must be float32 with shape'):
                import_model(self.model, forged, self.root / 'forged-output')
        self.assertFalse((self.root / 'forged-output').exists())

    def test_truncated_payload_and_duplicate_members(self):
        valid = io.BytesIO()
        np.save(valid, np.ones((2, 3), dtype=np.float32))
        for label, payload in [('truncated', valid.getvalue()[:-1]), ('trailing', valid.getvalue() + b'x')]:
            archive_path = self.root / f'{label}.npz'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('features.npy', payload)
            with mock.patch('numpy.load', side_effect=AssertionError('unsafe allocation path reached')):
                with self.assertRaisesRegex(FrontendError, 'payload length'):
                    import_model(self.model, archive_path, self.root / label)
        import warnings
        duplicate = self.root / 'duplicate.npz'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(duplicate, 'w') as archive:
                archive.writestr('features.npy', valid.getvalue())
                archive.writestr('features.npy', valid.getvalue())
        with self.assertRaisesRegex(FrontendError, 'duplicate members'):
            import_model(self.model, duplicate, self.root / 'duplicate')

    def test_string_initializer_produces_diagnostic(self):
        model = onnx.load(self.model)
        model.graph.initializer.append(H.make_tensor('unused_string', T.STRING, [1, 1], ['hello']))
        self.reject(model, 'DTYPE')

    def test_nonfinite_weight(self):
        model = onnx.load(self.model)
        model.graph.initializer[0].CopyFrom(N.from_array(np.full((3, 2), np.nan, dtype=np.float32), 'weights'))
        self.reject(model, 'NONFINITE')

    def reduce_model(self, axis=1, keep=1, noop=0, constant=False):
        axes = N.from_array(np.array([axis], dtype=np.int64), 'axes')
        nodes = [H.make_node('ReduceSum', ['X', 'axes'], ['Y'], keepdims=keep, noop_with_empty_axes=noop)]
        if constant:
            nodes.insert(0, H.make_node('Constant', [], ['axes'], value=axes))
        shape = [2, 1] if axis in (1, -1) and keep else [1, 3] if keep else [2]
        return H.make_model(H.make_graph(nodes, 'reduce',
            [H.make_tensor_value_info('X', T.FLOAT, [2, 3])],
            [H.make_tensor_value_info('Y', T.FLOAT, shape)], [] if constant else [axes]),
            opset_imports=[H.make_opsetid('', 13)], ir_version=9)

    def test_reduce_axes_and_constant_lowering(self):
        for axis, constant in ((1, False), (-1, True)):
            with self.subTest(axis=axis):
                path = self.save(self.reduce_model(axis=axis, constant=constant))
                report = inspect_model(path)
                self.assertTrue(report['supported'], report)
                np.savez(self.root / 'reduce.npz', X=np.arange(6, dtype=np.float32).reshape(2, 3))
                manifest = import_model(path, self.root / 'reduce.npz', self.root / f'reduce-{axis}')
                text = (self.root / f'reduce-{axis}/graph.plg').read_text()
                self.assertIn('rowsum ', text)
                self.assertNotIn('axes', manifest['tensor_mapping'])
        self.reject(self.reduce_model(axis=0), 'REDUCE_AXES')
        self.reject(self.reduce_model(keep=0), 'SHAPE')
        self.reject(self.reduce_model(noop=1), 'REDUCE_ATTRIBUTES')

    def test_float_constant_materialization_and_reference(self):
        model = onnx.load(self.model)
        bias = onnx.TensorProto()
        bias.CopyFrom(model.graph.initializer[1])
        del model.graph.initializer[1]
        model.graph.node.insert(0, H.make_node('Constant', [], ['bias'], value=bias, name='constant/bias'))
        path = self.save(model)
        manifest = import_model(path, self.inputs, self.root / 'constant-import')
        self.assertIsNone(manifest['node_mapping'][0]['generated_node'])
        self.assertIn('bias', manifest['tensor_mapping'])
        if BINARY:
            self.compare_runtime(path, self.inputs, self.root / 'constant-import')

    def test_rowsum_runtime_reference(self):
        if not BINARY:
            self.skipTest('supply --binary for real runtime integration')
        path = self.save(self.reduce_model(axis=-1, constant=True))
        inputs = self.root / 'reduce-inputs.npz'
        np.savez(inputs, X=np.array([[1, 2, -4], [0.5, 0.25, -0.125]], dtype=np.float32))
        imported = self.root / 'reduce-import'
        import_model(path, inputs, imported)
        self.compare_runtime(path, inputs, imported)

    def gemm_model(self, shape=(2,), trans_b=1, **attrs):
        weight = np.arange(6, dtype=np.float32).reshape(2, 3) / 4
        if not trans_b:
            weight = weight.T.copy()
        bias = np.full(shape, .25, dtype=np.float32)
        node = H.make_node('Gemm', ['X', 'W', 'B'], ['Y'], transB=trans_b, name='dense', **attrs)
        return H.make_model(H.make_graph([node], 'gemm',
            [H.make_tensor_value_info('X', T.FLOAT, [2, 3])],
            [H.make_tensor_value_info('Y', T.FLOAT, [2, 2])],
            [N.from_array(weight, 'W'), N.from_array(bias, 'B')]),
            opset_imports=[H.make_opsetid('', 13)], ir_version=9)

    def test_gemm_bias_forms_and_normalized_provenance(self):
        for idx, (shape, transpose) in enumerate([((), 1), ((2,), 1), ((1, 2), 1), ((2, 1), 0), ((2, 2), 0)]):
            with self.subTest(shape=shape, transpose=transpose):
                path = self.save(self.gemm_model(shape, transpose))
                inputs = self.root / f'gemm{idx}.npz'
                np.savez(inputs, X=np.array([[1, -2, 3], [.25, .5, -.25]], dtype=np.float32))
                imported = self.root / f'gemm{idx}'
                report = import_model(path, inputs, imported)
                self.assertEqual([n['operator'] for n in report['normalized_nodes']], ['MatMul', 'Add'])
                self.assertEqual(len(report['node_mapping'][0]['generated_nodes']), 2)
                tensor_names = {t['name'] for t in report['normalized_tensors']}
                for node in report['normalized_nodes']:
                    self.assertTrue(set(node['inputs']) <= tensor_names)
                    self.assertIn(node['output'], tensor_names)
                self.assertEqual(report['inputs'][0]['shape'], [2, 3])
                if BINARY:
                    self.compare_runtime(path, inputs, imported)

    def test_gemm_scale_and_runtime_weight_rejected(self):
        self.reject(self.gemm_model(alpha=.5), 'GEMM_SCALE')
        self.reject(self.gemm_model(beta=0.), 'GEMM_SCALE')
        model = self.gemm_model()
        del model.graph.initializer[0]
        model.graph.input.append(H.make_tensor_value_info('W', T.FLOAT, [2, 3]))
        self.reject(model, 'GEMM_WEIGHT')
        model = self.gemm_model()
        del model.graph.initializer[1]
        model.graph.input.append(H.make_tensor_value_info('B', T.FLOAT, [2]))
        self.reject(model, 'SHAPE')

    def test_add_constant_broadcast_both_operand_orders(self):
        for idx, (shape, swap) in enumerate([((), False), ((2,), True), ((1, 2), False), ((2, 1), True)]):
            model = H.make_model(H.make_graph([H.make_node('Add', ['B', 'X'] if swap else ['X', 'B'], ['Y'])],
                'bias', [H.make_tensor_value_info('X', T.FLOAT, [2, 2])],
                [H.make_tensor_value_info('Y', T.FLOAT, [2, 2])],
                [N.from_array(np.full(shape, .25, dtype=np.float32), 'B')]),
                opset_imports=[H.make_opsetid('', 13)], ir_version=9)
            path = self.save(model)
            inputs = self.root / f'add{idx}.npz'
            np.savez(inputs, X=np.arange(4, dtype=np.float32).reshape(2, 2))
            imported = self.root / f'add{idx}'
            report = import_model(path, inputs, imported)
            self.assertEqual(report['tensor_lowering'][0]['transformation'], 'constant_bias_broadcast')
            if BINARY:
                self.compare_runtime(path, inputs, imported)

    def test_lowered_node_limit(self):
        nodes = [H.make_node('Gemm', ['X' if i == 0 else f'v{i-1}', 'W', 'B'], [f'v{i}']) for i in range(32)]
        nodes.append(H.make_node('Relu', ['v31'], ['v32']))
        model = H.make_model(H.make_graph(nodes, 'limit',
            [H.make_tensor_value_info('X', T.FLOAT, [2, 2])],
            [H.make_tensor_value_info('v32', T.FLOAT, [2, 2])],
            [N.from_array(np.eye(2, dtype=np.float32), 'W'), N.from_array(np.ones((2,), dtype=np.float32), 'B')]),
            opset_imports=[H.make_opsetid('', 13)], ir_version=9)
        self.reject(model, 'GENERATED_NODE_LIMIT')

    def spatial_model(self, op, xshape, yshape, initializers=(), inputs=None, **attrs):
        node = H.make_node(op, inputs or ['X'], ['Y'], **attrs)
        return H.make_model(H.make_graph([node], 'spatial',
            [H.make_tensor_value_info('X', T.FLOAT, xshape)],
            [H.make_tensor_value_info('Y', T.FLOAT, yshape)], list(initializers)),
            opset_imports=[H.make_opsetid('', 13)], ir_version=9)

    def test_native_grouped_dilated_convolution(self):
        for idx, (kernel, dilation, stride, outshape) in enumerate([
                (3, 1, [1, 1], [1, 2, 4, 4]), (2, 2, [2, 1], [1, 2, 2, 4])]):
            weights = np.arange(2 * kernel * kernel, dtype=np.float32).reshape(2, 1, kernel, kernel) / 16 - .25
            model = self.spatial_model('Conv', [1, 2, 4, 4], outshape,
                [N.from_array(weights, 'W'), N.from_array(np.array([.125, -.25], dtype=np.float32), 'B')],
                inputs=['X', 'W', 'B'], group=2, strides=stride, dilations=[dilation, dilation], pads=[1, 1, 1, 1])
            path = self.save(model)
            inputs = self.root / f'conv{idx}.npz'
            np.savez(inputs, X=np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4) / 16 - 1)
            imported = self.root / f'conv{idx}'
            report = import_model(path, inputs, imported)
            self.assertEqual(report['inputs'][0]['shape'], [1, 2, 4, 4])
            self.assertEqual(report['normalized_nodes'][0]['operator'], 'Conv2D')
            self.assertEqual(len(report['normalized_nodes'][0]['parameters']), 16)
            self.assertEqual(report['normalized_nodes'][0]['logical_shape'], outshape)
            self.assertEqual(report['normalized_nodes'][0]['shape'], [outshape[0], int(np.prod(outshape[1:]))])
            if BINARY:
                self.compare_runtime(path, inputs, imported, cnn=True)

    def test_native_pooling_and_global_average(self):
        cases = [('MaxPool', 0), ('AveragePool', 0), ('AveragePool', 1), ('GlobalAveragePool', 0)]
        for idx, (op, include_pad) in enumerate(cases):
            attrs = {} if op == 'GlobalAveragePool' else dict(kernel_shape=[2, 3], strides=[2, 1], pads=[0, 1, 0, 1])
            if op == 'AveragePool':
                attrs['count_include_pad'] = include_pad
            outshape = [2, 2, 1, 1] if op == 'GlobalAveragePool' else [2, 2, 2, 5]
            path = self.save(self.spatial_model(op, [2, 2, 4, 5], outshape, **attrs))
            inputs = self.root / f'pool{idx}.npz'
            np.savez(inputs, X=np.arange(80, dtype=np.float32).reshape(2, 2, 4, 5) / 32 - 1)
            imported = self.root / f'pool{idx}'
            import_model(path, inputs, imported)
            if BINARY:
                self.compare_runtime(path, inputs, imported, cnn=True)

    def test_native_flatten_and_static_reshape(self):
        for idx, (op, target, yshape) in enumerate([
                ('Flatten', None, [1, 12]), ('Reshape', [2, 6], [2, 6]),
                ('Reshape', [0, -1], [1, 12]), ('Reshape', [1, 1, 3, 4], [1, 1, 3, 4])]):
            initializers = [] if target is None else [N.from_array(np.array(target, dtype=np.int64), 'shape')]
            model = self.spatial_model(op, [1, 2, 2, 3], yshape, initializers,
                inputs=['X'] if target is None else ['X', 'shape'])
            path = self.save(model)
            inputs = self.root / f'reshape{idx}.npz'
            np.savez(inputs, X=np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3))
            imported = self.root / f'reshape{idx}'
            report = import_model(path, inputs, imported)
            self.assertEqual(report['normalized_nodes'][0]['operator'], 'Reshape')
            if BINARY:
                self.compare_runtime(path, inputs, imported, cnn=True)

    def test_spatial_unsupported_attributes_are_explicit(self):
        model = self.spatial_model('Conv', [1, 1, 4, 4], [1, 1, 4, 4],
            [N.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), 'W')], inputs=['X', 'W'], auto_pad='SAME_UPPER')
        self.reject(model, 'AUTO_PAD')
        model = self.spatial_model('MaxPool', [1, 1, 4, 4], [1, 1, 2, 2], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)
        self.reject(model, 'POOL_PARAMETERS')
        model = self.spatial_model('Flatten', [1, 1, 4, 4], [4, 4], axis=3)
        self.reject(model, 'FLATTEN_AXIS')
        model = self.spatial_model('Conv', [1, 1, 4, 4], [1, 1, 4, 4], inputs=['X', 'W'], pads=[1, 1, 1, 1])
        model.graph.input.append(H.make_tensor_value_info('W', T.FLOAT, [1, 1, 3, 3]))
        self.reject(model, 'CONV_WEIGHT')

    def test_spatial_work_and_effective_kernel_bounds(self):
        model = self.spatial_model('MaxPool', [1, 1, 256, 256], [1, 1, 256, 256],
            kernel_shape=[256, 256], pads=[127, 127, 128, 128])
        self.reject(model, 'CNN_WORK')
        model = self.spatial_model('Conv', [1, 1, 1, 1], [1, 1, 1, 1],
            [N.from_array(np.ones((1, 1, 1025, 1), dtype=np.float32), 'W')],
            inputs=['X', 'W'], pads=[32768, 0, 32768, 0], dilations=[64, 1])
        self.reject(model, 'CNN_EFFECTIVE_KERNEL')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch dependency needed for explicit BatchNorm folding test')
    def test_eval_batchnorm_fold_preserves_source_and_numerics(self):
        import torch
        from export_torch_example import fold_inference_conv_bn
        class Source(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 2, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(2)
                with torch.no_grad():
                    self.conv.weight.fill_(.125)
                    self.conv.bias.fill_(.25)
                    self.bn.running_mean.copy_(torch.tensor([.5, -.25]))
                    self.bn.running_var.copy_(torch.tensor([.5, 1.5]))
            def forward(self, x):
                return self.bn(self.conv(x))
        source = Source()
        with self.assertRaisesRegex(ValueError, 'evaluation-mode'):
            fold_inference_conv_bn(source)
        source.eval()
        state = {name: value.clone() for name, value in source.state_dict().items()}
        folded = fold_inference_conv_bn(source)
        self.assertIsInstance(source.bn, torch.nn.BatchNorm2d)
        self.assertIsInstance(folded.bn, torch.nn.Identity)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 8 - 1
        with torch.inference_mode():
            torch.testing.assert_close(source(x), folded(x), rtol=1e-5, atol=1e-5)

    def test_sixty_four_nodes_import_without_exhaustive_claim(self):
        nodes = [H.make_node('Relu', ['X' if i == 0 else f'v{i-1}'], [f'v{i}']) for i in range(64)]
        model = H.make_model(H.make_graph(nodes, 'bounded_limit',
            [H.make_tensor_value_info('X', T.FLOAT, [1, 1])],
            [H.make_tensor_value_info('v63', T.FLOAT, [1, 1])]),
            opset_imports=[H.make_opsetid('', 13)], ir_version=9)
        path = self.save(model)
        inputs = self.root / 'bounded.npz'
        np.savez(inputs, X=np.array([[-1]], dtype=np.float32))
        report = import_model(path, inputs, self.root / 'bounded-import')
        self.assertEqual(report['generated_node_count'], 64)
        self.assertFalse(report['execution_validated'])
        self.assertFalse(report['hardware_compatibility_validated'])

    def test_node_limit_and_output_count(self):
        nodes = [H.make_node('Relu', ['X' if i == 0 else f'v{i-1}'], [f'v{i}']) for i in range(65)]
        model = H.make_model(H.make_graph(nodes, 'limit', [H.make_tensor_value_info('X', T.FLOAT, [1, 1])],
             [H.make_tensor_value_info('v64', T.FLOAT, [1, 1])]), opset_imports=[H.make_opsetid('', 13)], ir_version=9)
        self.reject(model, 'NODE_LIMIT')
        model = onnx.load(self.model)
        model.graph.output.append(H.make_tensor_value_info('linear', T.FLOAT, [2, 2]))
        self.reject(model, 'OUTPUT_COUNT')

    def test_corrupt_model_and_cli_inspect(self):
        bad = self.root / 'bad.onnx'
        bad.write_bytes(b'not protobuf')
        self.assertEqual(inspect_model(bad)['issues'][0]['code'], 'MODEL_READ')
        output = self.root / 'inspection.json'
        run = subprocess.run([sys.executable, str(ROOT / 'scripts/onnx_frontend.py'), 'inspect',
            '--model', str(self.root / 'examples/unsupported_broadcast.onnx'), '--out', str(output)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse(json.loads(output.read_text())['supported'])
        repeat = subprocess.run([sys.executable, str(ROOT / 'scripts/onnx_frontend.py'), 'inspect',
            '--model', str(self.model), '--out', str(output)], capture_output=True, text=True)
        self.assertEqual(repeat.returncode, 2)

    def test_runtime_matches_onnx_reference(self):
        if not BINARY:
            self.skipTest('supply --binary for real runtime integration')
        manifest = import_model(self.model, self.inputs, self.root / 'run-import')
        self.compare_runtime(self.model, self.inputs, self.root / 'run-import')
        self.assertEqual(manifest['outputs'][0]['source_name'], 'scores')

    def compare_runtime(self, model, inputs, imported, cnn=False):
        output = imported / 'runtime.json'
        run = subprocess.run([BINARY, 'verify', '--relu-supported', '--graph', str(imported / 'graph.plg'), '--json', str(output)] + (['--cnn-supported'] if cnn else []),
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        report = json.loads(output.read_text())
        with np.load(inputs, allow_pickle=False) as z:
            expected = ReferenceEvaluator(str(model)).run(None, dict(z))[0]
        self.assertIn('results', report)
        self.assertTrue(report['results'])
        for plan in report['results']:
            self.assertTrue(plan['correct'], plan)
            self.assertIn('output', plan)
            actual = np.asarray(plan['output'], dtype=np.float32).reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--binary')
    known, remaining = parser.parse_known_args()
    BINARY = str(Path(known.binary).resolve()) if known.binary else None
    unittest.main(argv=[sys.argv[0]] + remaining)
