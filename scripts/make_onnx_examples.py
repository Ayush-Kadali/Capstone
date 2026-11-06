#!/usr/bin/env python3
"""Generate real bounded ONNX models and named NPZ inputs (no downloads)."""
import argparse
import json
from pathlib import Path


def create_examples(out_dir):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    x = np.array([[1, -2, 3], [0, 4, -1]], dtype=np.float32)
    w = np.array([[2, 1], [1, -3], [-2, 2]], dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float32)
    inp = helper.make_tensor_value_info('features', TensorProto.FLOAT, [2, 3])
    result = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [2, 2])
    nodes = [helper.make_node('MatMul', ['features', 'weights'], ['linear'], name='encoder/matmul'),
             helper.make_node('Relu', ['linear'], ['active'], name='encoder/relu'),
             helper.make_node('Add', ['active', 'bias'], ['scores'], name='head/add')]
    graph = helper.make_graph(nodes, 'small_dense', [inp], [result],
                              [numpy_helper.from_array(w, 'weights'), numpy_helper.from_array(b, 'bias')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=9)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, out / 'supported_dense.onnx')
    np.savez(out / 'supported_dense.inputs.npz', features=x)
    # Legal runtime-input broadcasting remains outside the bounded importer subset.
    bad_graph = helper.make_graph([helper.make_node('Add', ['features', 'bias'], ['scores'])],
        'unsupported_broadcast', [helper.make_tensor_value_info('features', TensorProto.FLOAT, [2, 2]),
         helper.make_tensor_value_info('bias', TensorProto.FLOAT, [1, 2])],
        [result], [])
    bad = helper.make_model(bad_graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=9)
    onnx.checker.check_model(bad, full_check=True)
    onnx.save(bad, out / 'unsupported_broadcast.onnx')
    np.savez(out / 'unsupported_broadcast.inputs.npz', features=b, bias=np.ones((1, 2), dtype=np.float32))
    (out / 'README.json').write_text(json.dumps({'schema_version': 1,
        'supported': 'supported_dense.onnx', 'unsupported': 'unsupported_broadcast.onnx',
        'purpose': 'Real ONNX interchange and exact semantic rejection; synthetic small graph, not a trained model'}, indent=2) + '\n')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(create_examples(args.out))
