#!/usr/bin/env python3
"""Export a real, small PyTorch module through its ONNX exporter.

Weights are deterministic demonstration weights, not trained parameters; this
checks interoperability and numerical behavior, not classification accuracy.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def fold_inference_conv_bn(model):
    """Explicit eval-only fold; preserve the source model and its reference output."""
    import torch
    if model.training or model.conv.training or model.bn.training:
        raise ValueError('BatchNorm folding requires an evaluation-mode source model')
    if not isinstance(model.conv, torch.nn.Conv2d) or not isinstance(model.bn, torch.nn.BatchNorm2d):
        raise ValueError('Expected directly adjacent Conv2d and BatchNorm2d modules')
    folded = copy.deepcopy(model)
    folded.conv = torch.nn.utils.fuse_conv_bn_eval(folded.conv, folded.bn)
    folded.bn = torch.nn.Identity()
    return folded


def export(out, family="biasless-mlp"):
    import numpy as np
    import onnx
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if family not in ('biasless-mlp', 'biased-mlp', 'residual-mlp', 'deep-mlp', 'small-cnn', 'residual-cnn', 'small-cnn-bn-folded'):
        raise ValueError('Unknown demonstration family')
    biased = family != 'biasless-mlp'
    trunk = torch.nn.Sequential(torch.nn.Linear(3, 4, bias=biased), torch.nn.ReLU(),
                                torch.nn.Linear(4, 2, bias=biased)).eval()
    with torch.no_grad():
        trunk[0].weight.copy_(torch.tensor([[.5, -.25, .75], [-.5, .5, .25],
                                          [.25, 1., -.5], [1., .25, .5]]))
        trunk[2].weight.copy_(torch.tensor([[.5, -.25, 1., .25], [-.5, .75, .25, 1.]]))
        if biased:
            trunk[0].bias.copy_(torch.tensor([.25, -.5, .125, -.25]))
            trunk[2].bias.copy_(torch.tensor([.5, -.125]))
    model = trunk
    if family == 'residual-mlp':
        class ResidualMLP(torch.nn.Module):
            def __init__(self, trunk):
                super().__init__()
                self.trunk = trunk
                self.projection = torch.nn.Linear(3, 2, bias=False)
                with torch.no_grad():
                    self.projection.weight.copy_(torch.tensor([[.25, .5, -.25], [-.5, .25, .75]]))

            def forward(self, x):
                return self.trunk(x) + self.projection(x)
        model = ResidualMLP(trunk).eval()
    x = torch.tensor([[1., -2., 3.], [0., 4., -1.]])
    description = f'Linear(3,4,bias={biased}) -> ReLU -> Linear(4,2,bias={biased})' + (
        ' + Linear(3,2,bias=False) residual projection' if family == 'residual-mlp' else '')
    if family in ('small-cnn', 'residual-cnn', 'small-cnn-bn-folded'):
        class SmallCNN(torch.nn.Module):
            def __init__(self, residual):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 2, 3, padding=1, bias=True)
                self.relu = torch.nn.ReLU()
                self.bn = torch.nn.BatchNorm2d(2) if family == 'small-cnn-bn-folded' else torch.nn.Identity()
                self.residual = residual
                if residual:
                    self.projection = torch.nn.Conv2d(1, 2, 1, bias=False)
                    self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                    self.classifier = torch.nn.Linear(2, 3)
                else:
                    self.pool = torch.nn.MaxPool2d(2)
                    self.classifier = torch.nn.Linear(8, 3)
                with torch.no_grad():
                    self.conv.weight.copy_((torch.arange(18).reshape(2, 1, 3, 3) - 8) / 32)
                    self.conv.bias.copy_(torch.tensor([.125, -.25]))
                    if family == 'small-cnn-bn-folded':
                        self.bn.weight.copy_(torch.tensor([1.25, .75]))
                        self.bn.bias.copy_(torch.tensor([-.125, .25]))
                        self.bn.running_mean.copy_(torch.tensor([.125, -.25]))
                        self.bn.running_var.copy_(torch.tensor([.5, 1.5]))
                    self.classifier.weight.copy_((torch.arange(self.classifier.weight.numel()).reshape(self.classifier.weight.shape) % 7 - 3) / 16)
                    self.classifier.bias.copy_(torch.tensor([.25, -.125, .5]))
                    if residual:
                        self.projection.weight.copy_(torch.tensor([[[[.25]]], [[[-.5]]]]))

            def forward(self, x):
                y = self.relu(self.bn(self.conv(x)))
                if self.residual:
                    y = y + self.projection(x)
                return self.classifier(torch.flatten(self.pool(y), 1))
        model = SmallCNN(family == 'residual-cnn').eval()
        x = (torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) - 7) / 8
        description = 'Conv2d(1,2,3,padding1,bias) -> ReLU -> ' + (
            'Add(projection Conv2d1x1) -> GlobalAveragePool -> Flatten -> Linear(2,3,bias)'
            if family == 'residual-cnn' else 'MaxPool2d2 -> Flatten -> Linear(8,3,bias)')
    if family == 'deep-mlp':
        layers = []
        for index in range(10):
            linear = torch.nn.Linear(4, 4, bias=True)
            with torch.no_grad():
                linear.weight.copy_(torch.eye(4) * .5 +
                    (torch.arange(16).reshape(4, 4) % 7 - 3) / 64 + index / 1024)
                linear.bias.copy_(torch.tensor([.125, .25, .0625, -.0625]) + index / 128)
            layers.append(linear)
            if index < 9:
                layers.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*layers).eval()
        x = torch.tensor([[1., -2., 3., .5], [0., 4., -1., -.25]])
        description = '10 biased Linear(4,4) layers with 9 intervening ReLUs; 29 normalized nodes'
    folding = None
    source_model = model
    if family == 'small-cnn-bn-folded':
        model = fold_inference_conv_bn(source_model)
        description = 'Conv2d -> BatchNorm2d(eval) -> ReLU -> MaxPool -> Flatten -> Linear; explicit Conv+BN fold before import'
        with torch.inference_mode():
            source_expected = source_model(x).numpy()
            folded_expected = model(x).numpy()
            np.testing.assert_allclose(folded_expected, source_expected, rtol=1e-5, atol=1e-5)
            torch.onnx.export(source_model, (x,), str(out/'unfused.onnx'), dynamo=False,
                              opset_version=13, input_names=['features'], output_names=['scores'],
                              do_constant_folding=False, training=torch.onnx.TrainingMode.PRESERVE)
        original = onnx.load(out/'unfused.onnx')
        onnx.checker.check_model(original, full_check=True)
        if 'BatchNormalization' not in [n.op_type for n in original.graph.node]:
            raise RuntimeError('Unfused evidence must retain an actual BatchNormalization node')
        np.savez(out/'unfused-pytorch-reference.npz', scores=source_expected)
        folding = {'transformation': 'torch.nn.utils.fuse_conv_bn_eval on a deepcopy of the source eval model',
                   'source_modules': ['conv:Conv2d', 'bn:BatchNorm2d'],
                   'source_model_unchanged': isinstance(source_model.bn, torch.nn.BatchNorm2d),
                   'source_onnx': 'unfused.onnx', 'folded_onnx': 'mlp.onnx',
                   'source_training': source_model.training, 'batchnorm_epsilon': source_model.bn.eps,
                   'source_pytorch_vs_folded_pytorch_passed': True,
                   'max_abs_error': float(np.max(np.abs(source_expected - folded_expected))),
                   'batchnorm_native_import_supported': False,
                   'original_onnx_runtime_validation_performed': False}
    with torch.inference_mode():
        expected = model(x).numpy()
        torch.onnx.export(model, (x,), str(out/'mlp.onnx'), dynamo=False, opset_version=13,
                          input_names=['features'], output_names=['scores'],
                          do_constant_folding=True)
    np.savez(out/'inputs.npz', features=x.numpy())
    np.savez(out/'pytorch-reference.npz', scores=expected)
    graph = onnx.load(out/'mlp.onnx')
    onnx.checker.check_model(graph, full_check=True)
    if folding is not None and any(n.op_type == 'BatchNormalization' for n in graph.graph.node):
        raise RuntimeError('Folded graph unexpectedly retains BatchNormalization')
    manifest = {'schema_version': 1, 'producer': 'torch.onnx.export',
                'torch_version': torch.__version__, 'onnx_version': onnx.__version__,
                'family': family, 'folding': folding,
                'model': description,
                'opset': 13, 'operators': [n.op_type for n in graph.graph.node],
                'input_shape': list(x.shape), 'output': expected.tolist(),
                'trained': False, 'task_accuracy_evaluated': False,
                'sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in out.iterdir() if f.is_file()}}
    (out/'export.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--family', choices=['biasless-mlp', 'biased-mlp', 'residual-mlp', 'deep-mlp', 'small-cnn', 'residual-cnn', 'small-cnn-bn-folded'], default='biasless-mlp')
    args = parser.parse_args()
    export(args.out, args.family)
