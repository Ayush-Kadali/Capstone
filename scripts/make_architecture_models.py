#!/usr/bin/env python3
"""Export edge-CNN architecture variants for model/target comparison.

Weights are seeded random (untrained): latency, traffic and placement depend on
shapes, not weight values. Task accuracy is NOT evaluated for these files.
Resolution 160 exceeds the functional executor's 65,536-element limit (analysis
only); resolutions 16 and 64 stay executable so the same architectures can be run.
"""
import argparse
import hashlib
import json
from pathlib import Path

VARIANTS = ('standard', 'depthwise', 'residual')


def build(variant):
    import torch
    from torch import nn

    def separable(c_in, c_out, stride):
        # depthwise 3x3 (groups = channels) followed by pointwise 1x1
        return nn.Sequential(nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in), nn.ReLU(),
                             nn.Conv2d(c_in, c_out, 1), nn.ReLU())

    def dense(c_in, c_out, stride):
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1), nn.ReLU())

    class EdgeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            block = separable if variant == 'depthwise' else dense
            self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU())
            self.b1 = block(16, 32, 2)
            self.b2 = block(32, 32, 1)
            self.pool = nn.GlobalAvgPool2d() if hasattr(nn, 'GlobalAvgPool2d') else nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(32, 10)

        def forward(self, x):
            x = self.b1(self.stem(x))
            y = self.b2(x)
            if variant == 'residual':
                y = y + x
            return self.head(torch.flatten(self.pool(y), 1))
    return EdgeCNN().eval()


def export(out, resolutions=(16, 64, 160)):
    import numpy as np
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'purpose': 'architecture comparison; seeded untrained weights; task accuracy not evaluated',
                'torch': torch.__version__, 'opset': 13, 'exporter': 'torch.onnx.export(dynamo=False)', 'files': {}}
    for variant in VARIANTS:
        torch.manual_seed(20261106)
        model = build(variant)
        for res in resolutions:
            x = torch.from_numpy(np.random.default_rng(res).standard_normal((1, 3, res, res)).astype('float32'))
            name = f'edge-{variant}-{res}'
            torch.onnx.export(model, (x,), out / f'{name}.onnx', input_names=['image'], output_names=['scores'],
                              opset_version=13, dynamo=False, do_constant_folding=True)
            np.savez(out / f'{name}.inputs.npz', image=x.numpy())
            with torch.no_grad():
                np.save(out / f'{name}.torch-output.npy', model(x).numpy())
            from onnx_frontend import inspect_model
            check = inspect_model(out / f'{name}.onnx', scope='analysis')
            manifest['files'][name] = {'variant': variant, 'resolution': res, 'import_supported': check['supported'],
                                       'functionally_executable': check['functionally_executable'],
                                       'normalized_nodes': check.get('generated_node_count')}
    for p in sorted(out.iterdir()):
        if p.name != 'manifest.json':
            manifest.setdefault('sha256', {})[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='new directory')
    print(json.dumps(export(parser.parse_args().out)['files'], indent=1))
