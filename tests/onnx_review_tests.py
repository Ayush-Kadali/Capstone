#!/usr/bin/env python3
"""Independent importer review: hostile containers, strict semantics, ORT parity.

Optional dependencies: numpy, onnx, onnxruntime. Use --binary for full execution.
"""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
for dependency in ('numpy', 'onnx', 'onnxruntime'):
    if not importlib.util.find_spec(dependency):
        print('SKIP: independent ONNX review requires ' + dependency)
        sys.exit(77)
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as H, numpy_helper as N, TensorProto as T
sys.path.insert(0, str(ROOT / 'scripts'))
import onnx_frontend as F

BINARY = None


class ImportReview(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model.onnx'
        self.inputs = self.root / 'inputs.npz'
        self.out = self.root / 'imported'
        self.x = np.array([[1, -2], [3, 4]], dtype=np.float32)
        np.savez(self.inputs, x=self.x)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, nodes=None, initializers=(), inputs=None, output_shape=(2, 2), values=()):
        inputs = inputs if inputs is not None else [H.make_tensor_value_info('x', T.FLOAT, [2, 2])]
        graph = H.make_graph(nodes or [H.make_node('Relu', ['x'], ['y'])], 'review', inputs,
                             [H.make_tensor_value_info('y', T.FLOAT, output_shape)],
                             list(initializers), value_info=list(values))
        model = H.make_model(graph, opset_imports=[H.make_opsetid('', 13)], ir_version=10)
        onnx.save(model, self.model)
        return model

    def reject(self, code):
        report = F.inspect_model(self.model)
        self.assertFalse(report['supported'])
        self.assertIn(code, [i['code'] for i in report['issues']])
        with self.assertRaises(F.FrontendError):
            F.import_model(self.model, self.inputs, self.out)
        self.assertFalse(self.out.exists())

    def test_string_initializer_rejected_without_crash(self):
        self.save(initializers=[H.make_tensor('unused', T.STRING, [1, 1], ['text'])])
        self.reject('DTYPE')

    def test_overridable_initializer_rejected(self):
        self.save(initializers=[N.from_array(self.x, 'x')])
        self.reject('OVERRIDABLE_INITIALIZER')

    def test_unused_unsupported_node_is_not_silently_dropped(self):
        self.save([H.make_node('Relu', ['x'], ['y']), H.make_node('Sigmoid', ['x'], ['unused'])])
        self.reject('OPERATOR')

    def test_external_data_never_loaded(self):
        weight = T(name='weight', data_type=T.FLOAT, dims=[2, 2], data_location=T.EXTERNAL)
        weight.external_data.add(key='location', value='/definitely/private/unreadable-weight')
        self.save(initializers=[weight])
        with mock.patch.object(onnx.external_data_helper, 'load_external_data_for_tensor',
                               side_effect=AssertionError('external read attempted')):
            self.reject('EXTERNAL_DATA')

    def test_inconsistent_declared_output_shape(self):
        self.save(output_shape=(3, 2))
        self.reject('ONNX_VALIDATION')

    def test_runtime_broadcast_add_rejected(self):
        self.save([H.make_node('Add', ['x', 'b'], ['y'])], inputs=[
            H.make_tensor_value_info('x', T.FLOAT, [2, 2]),
            H.make_tensor_value_info('b', T.FLOAT, [1, 2])])
        self.reject('ADD_BROADCAST')

    def test_invalid_reduction_variants(self):
        for axis, keepdims, noop in [(0, 1, 0), (1, 0, 0), (1, 1, 1)]:
            with self.subTest(axis=axis, keepdims=keepdims, noop=noop):
                output_shape = [1, 2] if axis == 0 else ([2] if keepdims == 0 else [2, 1])
                self.save([H.make_node('ReduceSum', ['x', 'axes'], ['y'], keepdims=keepdims,
                                      noop_with_empty_axes=noop)],
                          [N.from_array(np.array([axis], np.int64), 'axes')], output_shape=output_shape)
                report = F.inspect_model(self.model)
                self.assertFalse(report['supported'])

    def test_nonfinite_and_wrong_input_storage(self):
        self.save()
        for arr in (self.x.astype(np.float64), self.x.reshape(1, 4),
                    np.full((2, 2), np.nan, np.float32), np.full((2, 2), np.inf, np.float32)):
            with self.subTest(dtype=str(arr.dtype), shape=arr.shape):
                np.savez(self.inputs, x=arr)
                with self.assertRaises(F.FrontendError):
                    F.import_model(self.model, self.inputs, self.out)
                self.assertFalse(self.out.exists())

    def test_oversized_npy_header_rejected_before_numpy_load(self):
        self.save()
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {'descr': '<f4', 'fortran_order': False,
                                                     'shape': (1000000000, 1)})
        with zipfile.ZipFile(self.inputs, 'w') as archive:
            archive.writestr('x.npy', header.getvalue())
        with mock.patch.object(np, 'load', side_effect=AssertionError('unsafe allocation path')) as load:
            with self.assertRaises(F.FrontendError):
                F.import_model(self.model, self.inputs, self.out)
            load.assert_not_called()

    def test_truncated_valid_shape_rejected_before_numpy_load(self):
        self.save()
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {'descr': '<f4', 'fortran_order': False, 'shape': (2, 2)})
        with zipfile.ZipFile(self.inputs, 'w') as archive:
            archive.writestr('x.npy', header.getvalue() + b'\x00' * 4)
        with mock.patch.object(np, 'load', side_effect=AssertionError('unsafe allocation path')) as load:
            with self.assertRaises(F.FrontendError):
                F.import_model(self.model, self.inputs, self.out)
            load.assert_not_called()

    def test_no_overwrite(self):
        self.save()
        self.out.mkdir()
        marker = self.out / 'graph.plg'
        marker.write_text('preserve me')
        with self.assertRaises(FileExistsError):
            F.import_model(self.model, self.inputs, self.out)
        self.assertEqual(marker.read_text(), 'preserve me')

    def test_boundaries_and_dynamic_dimensions(self):
        for shape in ([257, 256], ['batch', 2], [0, 2], [1, 2, 2]):
            with self.subTest(shape=shape):
                self.save(inputs=[H.make_tensor_value_info('x', T.FLOAT, shape)], output_shape=shape)
                self.assertFalse(F.inspect_model(self.model)['supported'])

    def test_original_onnx_runtime_vs_all_translated_placements(self):
        if BINARY is None:
            self.skipTest('pass --binary to exercise the actual runtime')
        for axis in (1, -1):
            with self.subTest(axis=axis):
                name = 'input / odd\n# text'
                nodes = [H.make_node('MatMul', [name, 'w'], ['m'], name='node\noutput injection'),
                         H.make_node('Relu', ['m'], ['r']),
                         H.make_node('Add', ['r', 'bias'], ['a']),
                         H.make_node('ReduceSum', ['a', 'axes'], ['y'])]
                weights = [N.from_array(np.array([[2, -1], [0, 3]], np.float32), 'w'),
                           N.from_array(np.array([[.5, 1], [2, -1]], np.float32), 'bias'),
                           N.from_array(np.array([axis], np.int64), 'axes')]
                self.save(nodes, weights, [H.make_tensor_value_info(name, T.FLOAT, [2, 2])], (2, 1))
                np.savez(self.inputs, **{name: np.asfortranarray(self.x)})
                destination = self.root / ('imported-' + str(axis))
                manifest = F.import_model(self.model, self.inputs, destination)
                self.assertFalse(manifest['execution_validated'])
                self.assertFalse(manifest['hardware_compatibility_validated'])
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                expected = ort.InferenceSession(str(self.model), sess_options=options,
                                                providers=['CPUExecutionProvider']).run(None, {name: self.x})[0]
                report = self.root / ('report-' + str(axis) + '.json')
                run = subprocess.run([str(BINARY), 'verify', '--graph', str(destination / 'graph.plg'),
                                      '--relu-supported', '--reuse-buffers', '--tile-size', '2',
                                      '--json', str(report)], text=True, capture_output=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                results = json.loads(report.read_text())['results']
                self.assertEqual(len(results), 16)
                for result in results:
                    self.assertTrue(result['correct'])
                    np.testing.assert_allclose(np.array(result['output']).reshape(2, 1), expected,
                                               rtol=1e-5, atol=1e-5)
                text = (destination / 'graph.plg').read_text()
                self.assertNotIn('injection', text)
                self.assertIn(name, manifest['tensor_mapping'])

    def differential(self, label, feeds, node_count, expected=None, cnn=False):
        if BINARY is None:
            self.skipTest('pass --binary for expanded-operator differential checks')
        np.savez(self.inputs, **feeds)
        destination = self.root / label
        manifest = F.import_model(self.model, self.inputs, destination)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        reference = ort.InferenceSession(str(self.model), sess_options=options,
                                        providers=['CPUExecutionProvider']).run(None, feeds)[0]
        if expected is not None:
            np.testing.assert_allclose(reference, expected, rtol=1e-5, atol=1e-5)
        report = destination / 'execution.json'
        run = subprocess.run([str(BINARY), 'verify', '--graph', str(destination / 'graph.plg'),
                              '--relu-supported', '--reuse-buffers', '--tile-size', '2',
                              *(['--cnn-supported'] if cnn else []),
                              '--json', str(report)], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        execution = json.loads(report.read_text())
        self.assertEqual(len(execution['results']), 2 ** node_count)
        normalized = {node['generated_node']: node for node in manifest['normalized_nodes']}
        self.assertEqual(len(normalized), node_count)
        mapping = {node['source_index']: node for node in manifest['node_mapping']}
        for generated, node in normalized.items():
            self.assertIn(generated, mapping[node['source_index']]['generated_nodes'])
            self.assertEqual(node['source_name'], mapping[node['source_index']]['source_name'])
        for result in execution['results']:
            self.assertTrue(result['correct'])
            np.testing.assert_allclose(np.asarray(result['output']).reshape(reference.shape), reference,
                                       rtol=1e-5, atol=1e-5)
            commands = [command for command in result['commands'] if command['kind'] == 'COMPUTE']
            self.assertEqual(len(commands), node_count)
            for command in commands:
                self.assertIn(command['node'], normalized)
                self.assertEqual(command['op'], normalized[command['node']]['operator'])
                self.assertEqual(command['origins'], [command['node']])
        return manifest, execution

    def test_gemm_orientations_and_all_supported_constant_bias_shapes(self):
        x = np.array([[1, -2, 3], [-4, 5, 6]], np.float32)
        w = np.array([[1, 2, -1, 3], [4, -2, 5, 1], [2, 3, -4, 6]], np.float32)
        biases = [None, np.array(.5, np.float32), np.array([.5, 1, -2, 3], np.float32),
                  np.array([[.5, 1, -2, 3]], np.float32), np.array([[1], [-2]], np.float32),
                  np.arange(8, dtype=np.float32).reshape(2, 4)]
        for transpose in (0, 1):
            for index, bias in enumerate(biases):
                with self.subTest(transpose=transpose, bias_shape=None if bias is None else bias.shape):
                    weights = [N.from_array(w.T.copy() if transpose else w, 'w')]
                    args = ['x', 'w']
                    if bias is not None:
                        weights.append(N.from_array(bias, 'b'))
                        args.append('b')
                    self.save([H.make_node('Gemm', args, ['y'], name='linear', transB=transpose)], weights,
                              [H.make_tensor_value_info('x', T.FLOAT, [2, 3])], (2, 4))
                    expected = x @ w + (0 if bias is None else bias)
                    self.differential(f'gemm-{transpose}-{index}', {'x': x}, 1 if bias is None else 2, expected)

    def test_constant_add_broadcast_both_sides(self):
        x = np.array([[1, -2, 3, 4], [5, 6, -7, 8]], np.float32)
        for index, bias in enumerate([np.array(2, np.float32), np.arange(4, dtype=np.float32),
                                       np.arange(4, dtype=np.float32).reshape(1, 4),
                                       np.array([[2], [3]], np.float32)]):
            for swap in (False, True):
                with self.subTest(shape=bias.shape, swap=swap):
                    args = ['b', 'x'] if swap else ['x', 'b']
                    self.save([H.make_node('Add', args, ['y'])], [N.from_array(bias, 'b')],
                              [H.make_tensor_value_info('x', T.FLOAT, [2, 4])], (2, 4))
                    self.differential(f'add-{index}-{swap}', {'x': x}, 1, x + bias)

    def test_transposed_weight_also_used_in_original_orientation(self):
        x = np.array([[1, -2, 3], [4, 5, -6]], np.float32)
        w = np.arange(12, dtype=np.float32).reshape(4, 3) / 4
        nodes = [H.make_node('Gemm', ['x', 'w'], ['g'], transB=1),
                 H.make_node('MatMul', ['g', 'w'], ['y'])]
        self.save(nodes, [N.from_array(w, 'w')], [H.make_tensor_value_info('x', T.FLOAT, [2, 3])], (2, 3))
        self.differential('shared-weight', {'x': x}, 2, (x @ w.T) @ w)

    def test_shared_bias_expanded_for_different_batch_shapes(self):
        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        z = np.arange(12, dtype=np.float32).reshape(3, 4)
        bias = np.array([1, -2, 3, -4], np.float32)
        self.save([H.make_node('Add', ['x', 'b'], ['y']), H.make_node('Add', ['z', 'b'], ['dead'])],
                  [N.from_array(bias, 'b')], [H.make_tensor_value_info('x', T.FLOAT, [2, 4]),
                                            H.make_tensor_value_info('z', T.FLOAT, [3, 4])], (2, 4))
        self.differential('shared-bias', {'x': x, 'z': z}, 2, x + bias)

    def test_gemm_nonunit_scaling_and_transposed_activation_rejected(self):
        for attrs in ({'alpha': 2.0}, {'beta': .5}, {'transA': 1}):
            with self.subTest(attrs=attrs):
                self.save([H.make_node('Gemm', ['x', 'w', 'b'], ['y'], **attrs)],
                          [N.from_array(np.eye(2, dtype=np.float32), 'w'), N.from_array(self.x, 'b')])
                self.assertFalse(F.inspect_model(self.model)['supported'])

    def test_gemm_runtime_weight_or_bias_not_baked_into_constants(self):
        for dynamic_name in ('w', 'b'):
            with self.subTest(dynamic_name=dynamic_name):
                weights = [N.from_array(np.eye(2, dtype=np.float32), n) for n in ('w', 'b') if n != dynamic_name]
                self.save([H.make_node('Gemm', ['x', 'w', 'b'], ['y'])], weights,
                          [H.make_tensor_value_info('x', T.FLOAT, [2, 2]),
                           H.make_tensor_value_info(dynamic_name, T.FLOAT, [2, 2])])
                self.assertFalse(F.inspect_model(self.model)['supported'])

    def test_gemm_expansion_respects_normalized_sixty_four_node_limit(self):
        for count in (32, 33):
            nodes = [H.make_node('Gemm', ['x' if i == 0 else f'h{i-1}', 'w', 'b'],
                                 ['y' if i == count - 1 else f'h{i}']) for i in range(count)]
            self.save(nodes, [N.from_array(np.eye(2, dtype=np.float32), 'w'),
                              N.from_array(np.zeros(2, np.float32), 'b')])
            report = F.inspect_model(self.model)
            self.assertEqual(report['supported'], count == 32, report['issues'])

    def test_serialized_graph_respects_executor_file_size_limit(self):
        names = [f'x{i}' for i in range(20)]
        self.save([H.make_node('Relu', [names[0]], ['y'])],
                  inputs=[H.make_tensor_value_info(name, T.FLOAT, [256, 256]) for name in names],
                  output_shape=(256, 256))
        value = np.full((256, 256), np.float32(1.23456789e30))
        np.savez_compressed(self.inputs, **{name: value for name in names})
        self.assertTrue(F.inspect_model(self.model)['supported'])
        with self.assertRaises(F.FrontendError):
            F.import_model(self.model, self.inputs, self.out)
        self.assertFalse(self.out.exists())

    def test_native_conv_groups_dilation_stride_padding_and_bias(self):
        cases = [([2, 2, 3, 4], [3, 2, 2, 2], 1, [1, 1], [1, 1], [0, 0, 0, 0], True),
                 ([1, 4, 3, 4], [4, 2, 2, 2], 2, [1, 1], [2, 1], [1, 0, 0, 1], False),
                 ([1, 2, 4, 5], [4, 1, 2, 2], 2, [2, 2], [1, 2], [1, 2, 0, 1], True)]
        for index, (shape, ws, groups, dilation, strides, pads, bias) in enumerate(cases):
            x = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 11 - 5) / 4
            w = (np.arange(np.prod(ws), dtype=np.float32).reshape(ws) % 7 - 3) / 4
            weights = [N.from_array(w, 'w')]
            args = ['x', 'w']
            if bias:
                weights.append(N.from_array(np.arange(ws[0], dtype=np.float32) + .25, 'b'))
                args.append('b')
            out = [shape[0], ws[0]] + [(shape[2+i] + pads[i] + pads[i+2] -
                                       ((ws[2+i]-1)*dilation[i]+1)) // strides[i]+1 for i in range(2)]
            self.save([H.make_node('Conv', args, ['y'], group=groups, dilations=dilation,
                                   strides=strides, pads=pads)], weights,
                      [H.make_tensor_value_info('x', T.FLOAT, shape)], out)
            manifest, _ = self.differential(f'cnn-conv-{index}', {'x': x}, 2 if bias else 1, cnn=True)
            self.assertEqual(manifest['normalized_nodes'][0]['logical_shape'], out)
            self.assertEqual(manifest['normalized_nodes'][0]['shape'], [out[0], int(np.prod(out[1:]))])

    def test_native_pool_padding_denominators_and_negative_values(self):
        shape = [2, 2, 3, 4]
        x = -(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1)
        for op, count in [('MaxPool', 0), ('AveragePool', 0), ('AveragePool', 1)]:
            attrs = {'kernel_shape': [2, 3], 'strides': [2, 1], 'pads': [1, 1, 0, 1]}
            if op == 'AveragePool':
                attrs['count_include_pad'] = count
            self.save([H.make_node(op, ['x'], ['y'], **attrs)],
                      inputs=[H.make_tensor_value_info('x', T.FLOAT, shape)], output_shape=[2, 2, 2, 4])
            self.differential(f'cnn-pool-{op}-{count}', {'x': x}, 1, cnn=True)

    def test_global_average_flatten_and_inferred_constant_reshape(self):
        shape = [2, 3, 2, 2]
        x = np.arange(24, dtype=np.float32).reshape(shape)
        self.save([H.make_node('GlobalAveragePool', ['x'], ['g']), H.make_node('Flatten', ['g'], ['y'])],
                  inputs=[H.make_tensor_value_info('x', T.FLOAT, shape)], output_shape=[2, 3])
        self.differential('cnn-global-flat', {'x': x}, 2, x.mean(axis=(2, 3)), cnn=True)
        self.save([H.make_node('Reshape', ['x', 'a'], ['r']), H.make_node('Reshape', ['r', 'b'], ['y'])],
                  [N.from_array(np.array([0, -1], np.int64), 'a'),
                   N.from_array(np.array([1, 3, 4, 2], np.int64), 'b')],
                  [H.make_tensor_value_info('x', T.FLOAT, shape)], [1, 3, 4, 2])
        self.differential('cnn-reshape', {'x': x}, 2, x.reshape(1, 3, 4, 2), cnn=True)

    def test_cnn_unsupported_attributes_and_batched_matmul_rejected(self):
        cases = [('Conv', ['x', 'w'], {'auto_pad': 'SAME_UPPER'}, [1, 1, 3, 3]),
                 ('MaxPool', ['x'], {'kernel_shape': [2, 2], 'ceil_mode': 1}, [1, 1, 2, 2]),
                 ('MaxPool', ['x'], {'kernel_shape': [2, 2], 'dilations': [2, 2]}, [1, 1, 1, 1]),
                 ('Flatten', ['x'], {'axis': 2}, [1, 9])]
        for op, args, attrs, out in cases:
            self.save([H.make_node(op, args, ['y'], **attrs)],
                      [N.from_array(np.ones((1, 1, 2, 2), np.float32), 'w')] if op == 'Conv' else [],
                      [H.make_tensor_value_info('x', T.FLOAT, [1, 1, 3, 3])], out)
            self.assertFalse(F.inspect_model(self.model)['supported'])
        self.save([H.make_node('MatMul', ['x', 'x'], ['y'])],
                  inputs=[H.make_tensor_value_info('x', T.FLOAT, [1, 1, 2, 2])], output_shape=[1, 1, 2, 2])
        self.reject('MATMUL_RANK')

    def test_pool_kernel_area_is_bounded_even_with_small_input_and_output(self):
        self.save([H.make_node('AveragePool', ['x'], ['y'], kernel_shape=[65536, 65536],
                               pads=[65535, 65535, 0, 0])],
                  inputs=[H.make_tensor_value_info('x', T.FLOAT, [1, 1, 1, 1])], output_shape=[1, 1, 1, 1])
        self.assertFalse(F.inspect_model(self.model)['supported'])

    def test_finite_subnormal_and_extreme_float32_inputs_round_trip_bitwise(self):
        if BINARY is None:
            self.skipTest('pass --binary for parser float32 round-trip regression')
        smallest = np.nextafter(np.float32(0), np.float32(1))
        tiny, maximum = np.finfo(np.float32).tiny, np.finfo(np.float32).max
        x = np.array([[smallest, -smallest, tiny], [-tiny, maximum, -maximum]], np.float32)
        self.save(inputs=[H.make_tensor_value_info('x', T.FLOAT, [2, 3])], output_shape=[2, 3])
        np.savez(self.inputs, x=x)
        F.import_model(self.model, self.inputs, self.out)
        report = self.root / 'finite-extremes.json'
        run = subprocess.run([str(BINARY), 'verify', '--graph', str(self.out / 'graph.plg'),
                              '--relu-supported', '--json', str(report)],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        raw = json.loads(report.read_text())
        recovered = np.asarray(raw['source_graph']['inputs'][0]['values'], np.float32).reshape(x.shape)
        np.testing.assert_array_equal(recovered.view(np.uint32), x.view(np.uint32))
        self.assertTrue(all(result['correct'] for result in raw['results']))
        nonrepresentable = self.root / 'underflow.plg'
        nonrepresentable.write_text('tensor X f32 1 1 1e-100\nrelu r X Y\noutput Y\n')
        failure = subprocess.run([str(BINARY), 'verify', '--graph', str(nonrepresentable)],
                                 capture_output=True, text=True, timeout=30)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn('E_VALUE', failure.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve() if args.binary else None
    unittest.main(argv=[sys.argv[0]] + remaining)
