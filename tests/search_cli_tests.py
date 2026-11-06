#!/usr/bin/env python3
"""Independent bounded-search CLI provenance and large-graph regressions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BINARY = None


class SearchCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def graph(self, count):
        lines = ['tensor X f32 1 1 1', 'tensor B f32 1 1 1']
        for i in range(count):
            lines.append(f'add n{i} {"X" if i == 0 else "v" + str(i - 1)} B v{i}')
        lines.append(f'output v{count - 1}')
        path = self.root / f'graph-{count}.plg'
        path.write_text('\n'.join(lines) + '\n')
        return path

    def invoke(self, count=32, action='verify', flags=(), bounded=True, report=True):
        self.counter += 1
        output = self.root / f'report-{self.counter}.json'
        argv = [str(BINARY), action, '--graph', str(self.graph(count))]
        if bounded:
            argv += ['--search', 'bounded']
        argv += list(flags)
        if report:
            argv += ['--json', str(output)]
        run = subprocess.run(argv, text=True, capture_output=True, timeout=60)
        return run, json.loads(output.read_text()) if output.exists() else None

    def test_sixty_four_nodes_have_candidate_only_provenance_and_correct_results(self):
        run, report = self.invoke(64, flags=['--search-budget', '17', '--beam-width', '3', '--reuse-buffers'])
        self.assertEqual(run.returncode, 0, run.stderr)
        search = report['search']
        self.assertIs(search['exhaustive'], False)
        self.assertEqual(search['strategy'], 'bounded-beam')
        self.assertIsNone(search['infeasible'])
        self.assertEqual(search['evaluated'], 17)
        self.assertEqual(search['max_evaluations'], 17)
        self.assertTrue(search['budget_exhausted'])
        self.assertEqual(search['beam_width'], 3)
        self.assertLessEqual(len(report['results']), 5)
        placements = {r['placement'] for r in report['results']}
        self.assertIn('C' * 64, placements)
        self.assertIn('A' * 64, placements)
        self.assertTrue(search['maximal_feasible'])
        ranked = [(r['modeled_ms'], r['placement']) for r in search['ranked_feasible']]
        self.assertEqual(ranked, sorted(ranked))
        for result in report['results']:
            self.assertEqual(result['policy'], 'bounded-candidate')
            self.assertTrue(result['correct'])
            self.assertEqual(result['output'], [65])
            for command in result['commands']:
                self.assertEqual(command['version'], 3)
                if command['kind'] != 'COMPUTE':
                    self.assertEqual(command['parameters'], [])
        self.assertIn('no exhaustive coverage or global optimum claim', run.stdout)

    def test_exhaustive_remains_twelve_and_parser_sixty_four(self):
        run, report = self.invoke(13, bounded=False)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('E_LIMIT', run.stderr)
        self.assertIsNone(report)
        run, report = self.invoke(65)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('E_LIMIT', run.stderr)
        self.assertIsNone(report)

    def test_large_single_policy_execution_and_compile_do_not_enumerate(self):
        for policy, lane in [('cpu', 'C'), ('maximal', 'A')]:
            with self.subTest(policy=policy):
                run, report = self.invoke(32, action='run', bounded=False, flags=['--policy', policy])
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertFalse(report['search']['exhaustive'])
                self.assertEqual(report['search']['strategy'], 'single-policy')
                self.assertIsNone(report['search']['infeasible'])
                self.assertNotIn('maximal_feasible', report['search'])
                self.assertEqual(len(report['results']), 1)
                self.assertEqual(report['results'][0]['placement'], lane * 32)
                self.assertEqual(report['results'][0]['output'], [33])
                run, report = self.invoke(32, action='compile', bounded=False,
                                          flags=['--policy', policy], report=False)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn('No execution requested', run.stdout)

    def test_strict_maximal_infeasible_is_not_replaced(self):
        run, report = self.invoke(16, flags=['--device-bytes', '16', '--search-budget', '2', '--beam-width', '1'])
        self.assertEqual(run.returncode, 0, run.stderr)
        search = report['search']
        self.assertEqual(search['evaluated'], 2)
        self.assertEqual(search['rejected'], 1)
        self.assertEqual(search['maximal_placement'], 'A' * 16)
        self.assertFalse(search['maximal_feasible'])
        self.assertIn('E_MEMORY', search['maximal_rejection'])
        self.assertEqual([r['placement'] for r in report['results']], ['C' * 16])

    def test_unaffordable_initial_residency_cannot_claim_cpu_success(self):
        run, report = self.invoke(16, flags=['--device-bytes', '4', '--resident-input', 'X',
                                           '--resident-input', 'B', '--search-budget', '2'])
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('E_MEMORY', run.stderr)
        self.assertIsNone(report)

    def test_shuffled_execution_preserves_search_set_and_rankings(self):
        reports = []
        for seed in (1, 7):
            run, report = self.invoke(flags=['--search-budget', '20', '--beam-width', '4',
                                             '--order-seed', str(seed), '--async', '--reuse-buffers'])
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(all(r['correct'] and r['output'] == [33] for r in report['results']))
            reports.append(report)
        self.assertEqual(reports[0]['search'], reports[1]['search'])
        self.assertEqual({r['placement'] for r in reports[0]['results']},
                         {r['placement'] for r in reports[1]['results']})

    def test_invalid_search_options_fail_without_report(self):
        for flags in (['--search-budget', '1'], ['--search-budget', '2049'],
                      ['--beam-width', '0'], ['--beam-width', '33'], ['--search-budget', '-2']):
            with self.subTest(flags=flags):
                run, report = self.invoke(flags=flags)
                self.assertNotEqual(run.returncode, 0)
                self.assertIsNone(report)
        run, report = self.invoke(2, bounded=False, flags=['--beam-width', '2'])
        self.assertNotEqual(run.returncode, 0)
        self.assertIsNone(report)

    def test_small_exhaustive_report_keeps_exhaustive_identity(self):
        run, report = self.invoke(3, bounded=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(report['search']['exhaustive'])
        self.assertEqual(report['search']['strategy'], 'exhaustive')
        self.assertIsInstance(report['search']['infeasible'], int)
        self.assertEqual(len(report['results']), 8)

    def test_report_preserves_full_precision_of_nearly_equal_ranked_costs(self):
        path = self.graph(29)
        text = path.read_text().replace('tensor X f32 1 1 1', 'tensor X f32 2 2 1 -2 3 4')
        text = text.replace('tensor B f32 1 1 1', 'tensor B f32 2 2 1 1 1 1')
        path.write_text(text)
        report_path = self.root / 'precise-ranking.json'
        run = subprocess.run([str(BINARY), 'verify', '--graph', str(path), '--search', 'bounded',
                              '--search-budget', '128', '--beam-width', '8', '--json', str(report_path)],
                             text=True, capture_output=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(report_path.read_text())
        ranked = [(r['modeled_ms'], r['placement']) for r in report['search']['ranked_feasible']]
        self.assertEqual(ranked, sorted(ranked))
        scores = dict((r['placement'], r['modeled_ms']) for r in report['search']['ranked_feasible'])
        for result in report['results']:
            self.assertEqual(result['modeled_ms'], scores[result['placement']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve(strict=True)
    unittest.main(argv=[sys.argv[0]] + remaining)
