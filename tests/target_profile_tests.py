#!/usr/bin/env python3
"""Contract and rejection tests for evidence-limited target profiles."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('target_profile', ROOT / 'scripts/target_profile.py')
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

class Profiles(unittest.TestCase):
    def setUp(self):
        self.profile = tp.load_profile(ROOT / 'profiles/software-device-relu-on.json')

    def bad(self, section, key, value):
        p = copy.deepcopy(self.profile)
        (p if section is None else p[section])[key] = value
        with self.assertRaises(tp.ProfileError):
            tp.validate_profile(p)

    def test_flags_map_actual_units(self):
        flags = tp.engine_flags(self.profile)
        self.assertEqual(flags[flags.index('--bandwidth')+1], '1000000')
        self.assertEqual(flags[flags.index('--boundary-ms')+1], '1.3')
        self.assertIn('--relu-supported', flags)
        off = tp.load_profile(ROOT / 'profiles/software-device-relu-off.json')
        self.assertNotIn('--relu-supported', tp.engine_flags(off))

    def test_cnn_v2_toggle_and_authoritative_analysis(self):
        p = tp.load_profile(ROOT / 'profiles/software-cnn-relu-on.json')
        self.assertIn('--cnn-supported', tp.engine_flags(p))
        p['operators']['Conv2D'] = False
        with self.assertRaises(tp.ProfileError): tp.validate_profile(p)
        for op in tp.CNN_OPS: p['operators'][op] = False
        self.assertNotIn('--cnn-supported', tp.engine_flags(p))
        self.assertNotIn('--cnn-supported', tp.engine_flags(self.profile))
        self.assertFalse(tp.evaluate_node('Conv2D', [1, 4], [[1, 4], [1, 1]], p)['accepted'])

    def test_evidence_and_identity_cannot_upgrade(self):
        for level in ('documented', 'calibrated', 'hardware-validated'):
            self.bad('evidence', 'level', level)
        self.bad('evidence', 'hardware_validated', True)
        self.bad('target', 'backend', 'opencl')
        self.bad('target', 'kind', 'gpu')
        self.bad(None, 'id', 'beagley-ai')

    def test_limits_are_exact_and_unknown_keys_fail(self):
        for field, value in (('rank', 4), ('rank', 2.0), ('dtype', 'int8'), ('layout', 'NCHW'), ('max_elements', 999)):
            self.bad('tensor', field, value)
        self.bad(None, 'unknown', 0)
        self.bad(None, 'schema_version', True)
        self.bad('costs', 'bandwidth_gbps', 4)
        self.bad('operators', 'Conv', True)

    def test_masks_are_representable(self):
        self.bad('operators', 'Add', False)
        self.bad('operators', 'MatMulRelu', False)
        self.bad('operators', 'Add', 1)

    def test_cost_bounds_and_booleans(self):
        for value in (float('nan'), float('inf'), 10**1000, -1, True, '1'):
            self.bad('costs', 'launch_ms', value)
        self.bad('costs', 'bandwidth_bytes_per_ms', 0)
        self.bad('costs', 'sync_ms', 1e9+1)
        self.bad('memory', 'device_bytes', True)
        self.bad('memory', 'device_bytes', -1)
        self.bad('memory', 'tile_size', 65)

    def test_real_tile_scratch_bound(self):
        self.profile['memory'].update(tile_size=2, local_bytes=48)
        tp.validate_profile(self.profile)
        self.profile['memory']['local_bytes'] = 47
        with self.assertRaises(tp.ProfileError):
            tp.validate_profile(self.profile)

    def test_matmul_semantics_and_memory(self):
        result = tp.evaluate_node('MatMul', [2,4], [[2,3],[3,4]], self.profile)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['operand_output_bytes'], 104)
        self.profile['memory']['device_bytes'] = 103
        self.assertFalse(tp.evaluate_node('MatMul', [2,4], [[2,3],[3,4]], self.profile)['accepted'])
        self.assertFalse(tp.evaluate_node('MatMul', [2,4], [[2,5],[3,4]], self.profile)['accepted'])

    def test_relu_and_reduction_shapes(self):
        off = tp.load_profile(ROOT / 'profiles/software-device-relu-off.json')
        self.assertFalse(tp.evaluate_node('Relu', [2,3], [[2,3]], off)['accepted'])
        self.assertTrue(tp.evaluate_node('RowSum', [2,1], [[2,3]], off)['accepted'])
        self.assertFalse(tp.evaluate_node('RowSum', [1,3], [[2,3]], off)['accepted'])
        self.assertTrue(tp.evaluate_node('MatMulRelu', [2,4], [[2,3],[3,4]], self.profile)['accepted'])

    def test_broadcast_unknown_dynamic_rank_and_precision_rejected(self):
        nodes = [('Add',[2,3],[[2,3],[1,3]]),('Conv',[2,3],[[2,3],[2,3]]),
                 ('Relu',[None,3],[[None,3]]),('Relu',[1,2,3],[[1,2,3]]),
                 ('Relu',[257,256],[[257,256]]),('Relu',[2,3],[])]
        for op, shape, inputs in nodes:
            with self.subTest(op=op, shape=shape):
                self.assertFalse(tp.evaluate_node(op,shape,inputs,self.profile)['accepted'])
        self.assertFalse(tp.evaluate_node('Relu',[2,3],[[2,3]],self.profile,dtype='float16')['accepted'])

    def test_duplicate_operands_dedup_only_with_identity(self):
        self.profile['memory']['device_bytes'] = 32
        conservative = tp.evaluate_node('Add',[2,2],[[2,2],[2,2]],self.profile)
        exact = tp.evaluate_node('Add',[2,2],[[2,2],[2,2]],self.profile,input_names=['a','a'],output_name='b')
        self.assertFalse(conservative['accepted'])
        self.assertEqual(conservative['operand_output_bytes'],48)
        self.assertTrue(exact['accepted'])
        self.assertEqual(exact['operand_output_bytes'],32)
        self.assertFalse(tp.evaluate_node('Add',[2,2],[[2,2],[2,2]],self.profile,input_names=['a','b'],output_name='a')['accepted'])

    def test_json_duplicate_and_size_fail(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'p.json'
            path.write_text('{"id":1,"id":2}')
            with self.assertRaises(tp.ProfileError): tp.load_profile(path)
            path.write_text(' ' * 65537)
            with self.assertRaises(tp.ProfileError): tp.load_profile(path)

    def test_cli_is_json_and_reports_input_errors(self):
        command = [sys.executable, str(ROOT/'scripts/target_profile.py')]
        path = str(ROOT/'profiles/software-device-relu-on.json')
        for action in ('validate','flags'):
            result = subprocess.run(command+[action,path],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['profile_id'],self.profile['id'])
        result = subprocess.run(command+['analyze-node',path],capture_output=True,text=True)
        self.assertEqual(result.returncode,1)
        self.assertIn('requires --node',result.stderr)

if __name__ == '__main__':
    unittest.main()
