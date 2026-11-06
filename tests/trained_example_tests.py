#!/usr/bin/env python3
"""Contracts for the trained digits example in examples/models/digits/."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

for dependency in ('numpy', 'onnx', 'onnxruntime'):
    if not importlib.util.find_spec(dependency):
        print('SKIP: trained example tests require ' + dependency)
        sys.exit(77)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT/'examples/models/digits'
EVIDENCE = ROOT/'artifacts/trained-digits-2026-11-06'
sys.path.insert(0, str(ROOT/'scripts'))
from train_digits import ort_predict
BINARY = ROOT/'build/placementlens'
MANIFEST = json.loads((MODELS/'manifest.json').read_text())
LABELS = np.load(MODELS/'split.npz')['test_labels']


class RetainedFiles(unittest.TestCase):
    def test_every_file_is_hashed_and_matches(self):
        files = {f.name for f in MODELS.iterdir() if f.is_file()} - {'manifest.json'}
        self.assertEqual(files, set(MANIFEST['sha256']))
        for name, digest in MANIFEST['sha256'].items():
            self.assertEqual(hashlib.sha256((MODELS/name).read_bytes()).hexdigest(), digest, name)

    def test_split_is_disjoint_and_complete(self):
        split = np.load(MODELS/'split.npz')
        parts = [split[f'{k}_indices'] for k in ('train', 'validation', 'test')]
        joined = np.concatenate(parts)
        self.assertEqual(len(joined), MANIFEST['dataset']['instances'])
        self.assertEqual(len(np.unique(joined)), len(joined))
        self.assertEqual(len(LABELS), len(split['test_indices']))

    def test_ort_test_accuracy_recomputed_from_retained_files(self):
        for name, info in MANIFEST['models'].items():
            with self.subTest(model=name):
                logits = ort_predict(MODELS/info['onnx'], [MODELS/p for p in info['test_inputs']])
                self.assertEqual(logits.shape, (len(LABELS), 10))
                self.assertAlmostEqual(float((logits.argmax(1) == LABELS).mean()),
                                       info['onnxruntime_test_accuracy'], places=12)
                self.assertGreater(info['onnxruntime_test_accuracy'], 0.9)   # genuinely trained

    def test_retained_engine_logits_match_summary(self):
        if not (EVIDENCE/'summary.json').is_file():
            self.skipTest('artifacts/ evidence is not present in this checkout')
        summary = json.loads((EVIDENCE/'summary.json').read_text())
        for name, result in summary['models'].items():
            with np.load(EVIDENCE/f'{name}-engine-logits.npz') as archive:
                np.testing.assert_array_equal(archive['labels'], LABELS)
                accuracy = (archive['engine_logits'].argmax(2) == LABELS).mean(axis=1)
                self.assertEqual(len(accuracy), result['placements'])
                self.assertAlmostEqual(float(accuracy.min()), result['engine_accuracy_min'], places=12)


class EngineDemo(unittest.TestCase):
    def test_demo_batches_verify_against_original_onnx(self):
        if not BINARY.is_file():
            self.skipTest('placementlens binary not built; pass --binary')
        for name, info in MANIFEST['models'].items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/'run'
                process = subprocess.run(
                    [sys.executable, str(ROOT/'scripts/capstone.py'), 'analyze', str(MODELS/info['onnx']),
                     '--inputs', str(MODELS/info['demo_inputs']), '--target', 'software-cnn-relu-on',
                     '--out', str(out), '--binary', str(BINARY), '--json'],
                    capture_output=True, text=True, timeout=240)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(json.loads(process.stdout)['simulator_verdict'], 'verified_on_supplied_inputs')
                result = json.loads((out/'result.json').read_text())
                self.assertTrue(all(c['original_onnx_match'] for c in result['comparisons']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path)
    args, remaining = parser.parse_known_args()
    if args.binary:
        BINARY = args.binary.resolve()
    unittest.main(argv=[sys.argv[0]] + remaining)
