#!/usr/bin/env python3
"""Train, export and evaluate the small trained digits example.

`train` fits an MLP and a small CNN on scikit-learn's bundled copy of the UCI
optical handwritten digits data (8x8, 1,797 images), selects the epoch by
validation accuracy, reports held-out test accuracy once and writes
examples/models/digits/. `evaluate` runs the retained test set through the
PlacementLens core for every feasible placement and scores the engine logits.

Accuracy is a property of these weights on this split, not of any accelerator.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261106
SPLIT = {'test': 360, 'validation': 360}      # train = 1,077
CNN_CHUNK = 120                                # 120x8x8x8 = 61,440 <= 65,536 elements
HIDDEN, CHANNELS = 32, 8
EPOCHS, BATCH, LR, WEIGHT_DECAY = 150, 64, 1e-2, 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def models():
    import torch
    mlp = torch.nn.Sequential(torch.nn.Linear(64, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 10))
    cnn = torch.nn.Sequential(torch.nn.Conv2d(1, CHANNELS, 3, padding=1), torch.nn.ReLU(),
                              torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Linear(CHANNELS*16, 10))
    return {'mlp': mlp, 'cnn': cnn}


def load_split():
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    x = (digits.data / 16.0).astype(np.float32)          # preprocessing: x/16 -> [0, 1]
    y = digits.target.astype(np.int64)
    index = np.arange(len(y))
    rest, test = train_test_split(index, test_size=SPLIT['test'], stratify=y, random_state=SEED)
    train, val = train_test_split(rest, test_size=SPLIT['validation'], stratify=y[rest], random_state=SEED)
    return x, y, {'train': np.sort(train), 'validation': np.sort(val), 'test': np.sort(test)}


def shaped(name, x):
    return x.reshape(-1, 1, 8, 8) if name == 'cnn' else x


def train(out):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import sklearn
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    x, y, split = load_split()
    data = {part: (torch.from_numpy(x[i]), torch.from_numpy(y[i])) for part, i in split.items()}
    np.savez(out/'split.npz', **{f'{k}_indices': v for k, v in split.items()}, test_labels=y[split['test']])
    accuracy = lambda m, n, part: int((m(shaped(n, data[part][0])).argmax(1) == data[part][1]).sum()) / len(data[part][1])
    results, files = {}, {}
    for name, model in models().items():
        torch.manual_seed(SEED)
        for layer in model:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        generator = torch.Generator().manual_seed(SEED)
        best = (-1.0, -1, None)
        for epoch in range(EPOCHS):
            model.train()
            order = torch.randperm(len(data['train'][1]), generator=generator)
            for start in range(0, len(order), BATCH):
                batch = order[start:start+BATCH]
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(shaped(name, data['train'][0][batch])), data['train'][1][batch])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val = accuracy(model, name, 'validation')
            if val > best[0]:                  # selection uses validation only
                best = (val, epoch, {k: v.clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best[2])
        model.eval()
        torch.save(model.state_dict(), out/f'{name}_state_dict.pt')
        with torch.no_grad():
            scores = {part: accuracy(model, name, part) for part in ('train', 'validation', 'test')}
            test_x = shaped(name, data['test'][0])
            torch_logits = model(test_x).numpy()
        # Fixed batch dims: whole test set for the MLP, 120-image chunks for the CNN.
        batch_size = len(test_x) if name == 'mlp' else CNN_CHUNK
        input_name = 'pixels' if name == 'mlp' else 'image'
        with torch.no_grad():
            torch.onnx.export(model, (test_x[:batch_size],), str(out/f'{name}.onnx'), dynamo=False,
                              opset_version=13, input_names=[input_name], output_names=['logits'],
                              do_constant_folding=True)
        onnx.checker.check_model(onnx.load(out/f'{name}.onnx'), full_check=True)
        inputs = []
        for chunk, start in enumerate(range(0, len(test_x), batch_size)):
            path = out/(f'{name}_test_inputs.npz' if name == 'mlp' else f'{name}_test_chunk{chunk}.npz')
            np.savez(path, **{input_name: test_x[start:start+batch_size].numpy()})
            inputs.append(path.name)
        ort_logits = ort_predict(out/f'{name}.onnx', [out/p for p in inputs])
        results[name] = {
            'architecture': ' -> '.join(type(layer).__name__ for layer in model),
            'onnx': f'{name}.onnx', 'state_dict': f'{name}_state_dict.pt', 'input_name': input_name,
            'onnx_batch': batch_size, 'test_inputs': inputs, 'demo_inputs': inputs[0],
            'operators': [n.op_type for n in onnx.load(out/f'{name}.onnx').graph.node],
            'parameters': int(sum(p.numel() for p in model.parameters())),
            'selected_epoch': best[1],
            'pytorch_accuracy': scores,
            'onnxruntime_test_accuracy': float((ort_logits.argmax(1) == y[split['test']]).mean()),
            'max_abs_logit_diff_pytorch_vs_ort': float(np.abs(ort_logits - torch_logits).max()),
            'argmax_agreement_pytorch_vs_ort': float((ort_logits.argmax(1) == torch_logits.argmax(1)).mean())}
    manifest = {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'trained': True,
        'dataset': {'name': 'Optical recognition of handwritten digits (sklearn.datasets.load_digits)',
                    'provenance': 'Copy of the test set of the UCI ML hand-written digits datasets '
                                  '(https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits); '
                                  'creator E. Alpaydin, July 1998, as stated in the bundled scikit-learn DESCR',
                    'licence': 'Not stated in the bundled scikit-learn DESCR; check the UCI repository page before redistribution',
                    'instances': int(len(y)), 'features': '8x8 integer pixels 0..16', 'classes': 10},
        'preprocessing': 'float32(x / 16.0); MLP input [N,64], CNN input [N,1,8,8] (row-major reshape)',
        'split': {'method': 'two stratified sklearn train_test_split calls (test, then validation)',
                  'random_state': SEED, 'sizes': {k: int(len(v)) for k, v in split.items()}, 'file': 'split.npz'},
        'training': {'seed': SEED, 'optimizer': 'Adam', 'learning_rate': LR, 'weight_decay': WEIGHT_DECAY,
                     'epochs': EPOCHS, 'batch_size': BATCH, 'loss': 'cross_entropy', 'threads': 1,
                     'deterministic_algorithms': True,
                     'selection': 'epoch with highest validation accuracy (first on ties); test evaluated once after selection'},
        'export': {'exporter': 'torch.onnx.export(dynamo=False)', 'opset': 13, 'dtype': 'float32', 'quantized': False},
        'models': results,
        'versions': {'python': sys.version.split()[0], 'platform': platform.platform(), 'torch': torch.__version__,
                     'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
                     'scikit_learn': sklearn.__version__},
        'command': 'python scripts/train_digits.py train --out examples/models/digits'}
    manifest['sha256'] = {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(results, indent=2))


def ort_predict(model, inputs):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    logits = []
    for path in inputs:
        with np.load(path, allow_pickle=False) as archive:
            logits.append(session.run(None, {name: archive[name] for name in archive.files})[0])
    return np.concatenate(logits)


def evaluate(model_dir, binary, profile, out):
    """Score every engine placement's logits against the held-out labels."""
    import numpy as np
    from explore_model import explore
    model_dir, out = Path(model_dir), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((model_dir/'manifest.json').read_text())
    labels = np.load(model_dir/'split.npz')['test_labels']
    summary = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'binary': str(binary), 'binary_sha256': sha(binary), 'profile': str(profile),
               'model_dir_manifest_sha256': sha(model_dir/'manifest.json'),
               'scope': 'host software engine with a hypothetical device profile; not physical accelerator accuracy',
               'command': ['python', *sys.argv], 'python': sys.version.split()[0], 'platform': platform.platform(),
               'numpy': np.__version__, 'onnxruntime': __import__('onnxruntime').__version__,
               'pruned_from_runs': 'core.json, import/graph.plg, model.onnx, inputs.npz (SHA-256 kept in each result.json)',
               'models': {}}
    for name, info in manifest['models'].items():
        ort_logits = ort_predict(model_dir/info['onnx'], [model_dir/p for p in info['test_inputs']])
        per_placement, statuses = {}, []
        for chunk, inputs in enumerate(info['test_inputs']):
            run = out/f'{name}-run{chunk}'
            receipt = explore(model_dir/info['onnx'], model_dir/inputs, profile, binary, run)
            statuses.append(receipt['status'])
            report = json.loads((run/'core.json').read_text())
            for result in report['results']:
                per_placement.setdefault(result['placement'], []).append(
                    np.asarray(result['output'], dtype=np.float32).reshape(-1, 10))
            # ponytail: bulky reproducible files are pruned; result.json keeps their SHA-256.
            for bulky in ('core.json', 'import/graph.plg', 'model.onnx', 'inputs.npz'):
                (run/bulky).unlink()
        names = sorted(p for p, parts in per_placement.items() if len(parts) == len(info['test_inputs']))
        if len(names) != len(per_placement):
            raise RuntimeError('Placement set differs between test chunks')
        stacked = np.stack([np.concatenate(per_placement[p]) for p in names])
        placements = {p: {'test_accuracy': float((logits.argmax(1) == labels).mean()),
                          'argmax_agreement_with_ort': float((logits.argmax(1) == ort_logits.argmax(1)).mean()),
                          'max_abs_logit_diff_vs_ort': float(np.abs(logits - ort_logits).max())}
                      for p, logits in zip(names, stacked)}
        np.savez_compressed(out/f'{name}-engine-logits.npz', placements=np.array(names),
                            engine_logits=stacked, ort_logits=ort_logits, labels=labels)
        cpu = 'C' * len(names[0])
        mixed = next(p for p in report['search']['ranked_feasible'] if {'C', 'A'} <= set(p['placement']))['placement']
        accuracies = [v['test_accuracy'] for v in placements.values()]
        summary['models'][name] = {
            'explorer_status_per_chunk': statuses, 'placements': len(placements),
            'ort_test_accuracy': float((ort_logits.argmax(1) == labels).mean()),
            'cpu_only': {'placement': cpu, **placements[cpu]},
            'lowest_modeled_cost_mixed': {'placement': mixed, **placements[mixed]},
            'engine_accuracy_min': min(accuracies), 'engine_accuracy_max': max(accuracies),
            'min_argmax_agreement_with_ort': min(v['argmax_agreement_with_ort'] for v in placements.values()),
            'max_abs_logit_diff_vs_ort': max(v['max_abs_logit_diff_vs_ort'] for v in placements.values()),
            'per_placement': placements}
    (out/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    t = sub.add_parser('train')
    t.add_argument('--out', type=Path, default=ROOT/'examples/models/digits')
    e = sub.add_parser('evaluate')
    e.add_argument('--models', type=Path, default=ROOT/'examples/models/digits')
    e.add_argument('--binary', type=Path, required=True)
    e.add_argument('--profile', type=Path, default=ROOT/'profiles/software-cnn-relu-on.json')
    e.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'train':
        train(args.out)
    else:
        sys.path.insert(0, str(ROOT/'scripts'))
        summary = evaluate(args.models, args.binary.resolve(), args.profile, args.out)
        print(json.dumps({k: {x: y for x, y in v.items() if x != 'per_placement'}
                          for k, v in summary['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
