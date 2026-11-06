"""Developer-facing verdicts must follow independent model evidence and plan scope."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

for dependency in ('onnx', 'onnxruntime', 'numpy'):
    if not importlib.util.find_spec(dependency):
        print('SKIP: developer CLI tests require '+dependency)
        sys.exit(77)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import developer_report as D
import explore_model as E
from make_onnx_examples import create_examples
import onnx
from onnx import helper as H, TensorProto as T
import numpy as np


class DeveloperContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        create_examples(cls.root/'models')
        cls.model = cls.root/'models/supported_dense.onnx'
        cls.inputs = cls.root/'models/supported_dense.inputs.npz'
        cls.profile = ROOT/'profiles/software-device-relu-off.json'
        cls.experiment = cls.root/'measured'
        E.explore(cls.model, cls.inputs, cls.profile, BINARY, cls.experiment, True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fresh(self, name, profile=None, model=None, inputs=None):
        out = self.root/name
        E.explore(model or self.model, inputs or self.inputs, profile or self.profile, BINARY, out)
        return out

    def test_different_device_plans_explain_boundary_savings(self):
        r = D.build_report(self.experiment)
        chosen, maximal = r['plans'][:2]
        self.assertEqual((chosen['placement'], maximal['placement']), ('ACC', 'ACA'))
        self.assertEqual((chosen['h2d_bytes']+chosen['d2h_bytes'], maximal['h2d_bytes']+maximal['d2h_bytes']), (64, 112))
        self.assertEqual((chosen['syncs'], maximal['syncs']), (1, 2))
        self.assertEqual(sum(x['bytes'] for x in maximal['copies']), 112)
        self.assertTrue(any('48' in finding for finding in r['findings']))
        self.assertEqual(r['physical_accelerator_verdict'], 'unknown')
        self.assertIsNone(r['physical_accelerator_latency_ms'])
        self.assertIn('CPU fallback', D.render(r))
        self.assertIn('encoder/relu', D.render(r))

    def test_measurements_scoped_and_host_baseline_opt_in(self):
        r = D.build_report(self.experiment)
        self.assertFalse(any(p['label'].startswith('CPU reference') for p in r['plans']))
        p = r['plans'][0]
        self.assertEqual(p['measured']['samples'], 5)
        self.assertIn('host', p['measured']['scope'])
        self.assertIn('not a confidence interval', D.render(r))
        self.assertTrue(D.build_report(self.experiment, True)['plans'][-1]['label'].startswith('CPU reference'))

    def test_no_single_run_presented_as_repeated_latency(self):
        r = D.build_report(self.fresh('unmeasured'))
        self.assertTrue(all(p['measured'] is None for p in r['plans']))
        self.assertNotIn('Host-only measurement:', D.render(r))
        self.assertIn('use analyze --measure', D.render(r))

    def test_per_node_support_does_not_imply_whole_plan_fit(self):
        profile = json.loads(self.profile.read_text())
        profile['memory']['device_bytes'] = 80
        path = self.root/'tight.json';path.write_text(json.dumps(profile))
        r = D.build_report(self.fresh('tight', profile=path))
        self.assertFalse(r['plans'][1]['feasible'])
        self.assertEqual(r['plans'][1]['placement'], 'ACA')
        self.assertIn('INFEASIBLE', D.render(r))

    def test_unsupported_importer_is_not_a_hardware_rejection(self):
        r = D.build_report(self.fresh('unsupported', model=self.root/'models/unsupported_broadcast.onnx',
                                     inputs=self.root/'models/unsupported_broadcast.inputs.npz'))
        self.assertEqual(r['simulator_verdict'], 'unsupported_by_importer')
        self.assertEqual(r['physical_accelerator_verdict'], 'unknown')
        self.assertEqual(r['plans'], [])
        self.assertNotIn('Illustrative score:', D.render(r))
        self.assertTrue(r['issues'])

    def test_bounded_search_never_becomes_an_exhaustive_claim(self):
        nodes = [H.make_node('Relu', ['x' if i==0 else 'r'+str(i-1)], ['r'+str(i)], name='relu'+str(i)) for i in range(13)]
        graph = H.make_graph(nodes, 'deep', [H.make_tensor_value_info('x', T.FLOAT, [1,2])], [H.make_tensor_value_info('r12', T.FLOAT, [1,2])])
        model = self.root/'deep.onnx';onnx.save(H.make_model(graph, opset_imports=[H.make_opsetid('',13)],ir_version=9),model)
        inputs = self.root/'deep.npz';np.savez(inputs,x=np.array([[-1,2]],np.float32))
        r = D.build_report(self.fresh('bounded', ROOT/'profiles/software-device-relu-on.json',model,inputs))
        self.assertFalse(r['search']['exhaustive'])
        self.assertIn('no global optimum claim',D.render(r))
        self.assertNotIn('Search: exhaustive', D.render(r))

    def test_modified_evidence_rejected(self):
        out = self.fresh('tamper')
        (out/'core.json').write_text((out/'core.json').read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Evidence changed'):D.build_report(out)

    def test_substituted_study_rejected(self):
        original = (self.experiment/'study/bundle.json').read_bytes()
        try:
            (self.experiment/'study/bundle.json').write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'Evidence changed'):D.build_report(self.experiment)
        finally:(self.experiment/'study/bundle.json').write_bytes(original)

    def test_missing_seals_cannot_inherit_validation(self):
        out = self.fresh('unsealed');p=out/'result.json';r=json.loads(p.read_text());r.pop('evidence_sha256');p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'fresh directory'):D.build_report(out)

    def test_failed_validation_has_no_performance_conclusions(self):
        out = self.fresh('failed');p=out/'result.json';r=json.loads(p.read_text());r['status']='numerical_mismatch';r['numerical_execution_validated']=False;p.write_text(json.dumps(r))
        report=D.build_report(out)
        self.assertEqual(report['simulator_verdict'],'validation_failed')
        self.assertEqual(report['plans'],[])

    def test_terminal_controls_are_escaped(self):
        self.assertNotIn('\x1b',D.clean('source\x1b[2J\nname'))
        self.assertNotIn('\n',D.clean('source\nname'))

    def test_cli_targets_explain_and_errors(self):
        cli=[sys.executable,str(ROOT/'scripts/capstone.py')]
        p=subprocess.run(cli+['targets'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0);self.assertIn('hypothetical',p.stdout)
        p=subprocess.run(cli+['explain',str(self.experiment),'--json'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['physical_accelerator_verdict'],'unknown')
        p=subprocess.run(cli+['inspect',str(self.root/'models/unsupported_broadcast.onnx')],capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertIn('UNSUPPORTED',p.stdout)
        p=subprocess.run(cli+['analyze',str(self.model),'--inputs',str(self.inputs),'--target','some-gpu','--out',str(self.root/'bad-target')],capture_output=True,text=True)
        self.assertEqual(p.returncode,1);self.assertIn('Unknown target',p.stderr)
        self.assertFalse((self.root/'bad-target').exists())


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binary',type=Path,required=True)
    args,rest=parser.parse_known_args();BINARY=args.binary.resolve()
    unittest.main(argv=[sys.argv[0],*rest])
