#!/usr/bin/env python3
"""Candidate-only measurement contracts and optional real larger-graph execution."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import bounded_study as bounded
import study
from study_tests import fixture_v3


def fixture():
    report = fixture_v3()
    report["search"].update(exhaustive=False, strategy="bounded-beam", infeasible=None,
        evaluated=2, rejected=0, budget_exhausted=False, beam_width=1,
        max_evaluations=2, maximal_placement="A", maximal_feasible=True, maximal_rejection="")
    report["search"]["ranked_feasible"] = [{"placement": "A", "modeled_ms": 99999}, {"placement": "C", "modeled_ms": 99999}]
    return report


class BoundedTests(unittest.TestCase):
    def bundle(self, calibration=None, evaluation=None, **kwargs):
        return bounded.build_bundle(calibration or [fixture(), fixture()], evaluation or [fixture(), fixture()], **kwargs)

    def test_nonexhaustive_artifact_is_distinct_and_never_claims_global_oracle(self):
        artifact = self.bundle()
        self.assertEqual(artifact["artifact_type"], "bounded-placement-study")
        self.assertFalse(artifact["oracle_available"])
        self.assertFalse(artifact["global_regret_available"])
        self.assertFalse(artifact["exhaustive"])
        self.assertNotIn("oracle", artifact["policies"])
        self.assertEqual(artifact["policies"]["best_observed_candidate"]["placement"], "A")
        self.assertEqual(artifact["policies"]["cpu"]["metrics"]["regret_vs_best_observed_ms"], 6.)
        self.assertIsNone(artifact["search"]["infeasible"])
        with self.assertRaises(ValueError):
            study.validate_bundle(artifact)
        with self.assertRaises(ValueError):
            study.validate_report(fixture())

    def test_calibration_and_heldout_evaluation_are_separate(self):
        evaluation = fixture()
        evaluation["results"][0]["host_execution_ms"] = 3.
        evaluation["results"][1]["host_execution_ms"] = 20.
        artifact = self.bundle(evaluation=[evaluation, deepcopy(evaluation)])
        self.assertEqual(artifact["policies"]["additive"]["placement"], "A")
        self.assertEqual(artifact["policies"]["best_observed_candidate"]["placement"], "C")
        self.assertEqual(artifact["policies"]["additive"]["metrics"]["regret_vs_best_observed_ms"], 17.)
        self.assertNotEqual(artifact["raw_runs"]["calibration_sha256"], artifact["raw_runs"]["evaluation_sha256"])

    def test_calibration_or_evaluation_failure_suppresses_all_conclusions(self):
        for phase in ("calibration", "evaluation"):
            bad = fixture()
            bad["results"][0]["correct"] = False
            artifact = self.bundle(**{phase: [bad]})
            self.assertFalse(artifact["conclusions_valid"])
            self.assertTrue(all(p["placement"] is None and p["metrics"] is None for p in artifact["policies"].values()))
            self.assertTrue(all(p["metrics"]["regret_vs_best_observed_ms"] is None for p in artifact["plans"]))
            self.assertTrue(all(p["metrics"]["p50_ms"] > 0 for p in artifact["plans"]))

    def test_missing_candidate_event_or_parameter_contract_is_rejected(self):
        for change in (lambda r: r["results"].pop(),
                       lambda r: r["results"][0]["events"].pop(),
                       lambda r: r["results"][0]["commands"][1].update(parameters=[1]),
                       lambda r: r["results"][0]["events"][0].update(name="COPY"),
                       lambda r: r["results"][0].update(placement="A")):
            report = fixture(); change(report)
            with self.assertRaises(ValueError):
                bounded.validate_report(report)

    def test_budget_counts_are_attempted_only_and_strict_maximal_is_not_substituted(self):
        report = fixture()
        report["results"] = report["results"][:1]
        report["search"].update(rejected=1, ranked_feasible=[{"placement": "C", "modeled_ms": 99999}], maximal_feasible=False,
                                maximal_rejection="E_MEMORY: maximal plan exceeds memory")
        artifact = self.bundle([report], [deepcopy(report)])
        maximal = artifact["policies"]["maximal"]
        self.assertEqual(maximal["status"], "infeasible")
        self.assertEqual(maximal["requested_placement"], "A")
        self.assertIsNone(maximal["placement"])
        self.assertEqual(artifact["search"]["rejected"], 1)
        self.assertIsNone(artifact["search"]["infeasible"])
        for field, value in (("exhaustive", True), ("infeasible", 1), ("evaluated", 3),
                             ("rejected", 3), ("beam_width", 33), ("max_evaluations", 2049), ("maximal_feasible", True)):
            bad = deepcopy(report); bad["search"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                bounded.validate_report(bad)

    def test_capability_and_ranked_cost_provenance_cannot_contradict_retained_execution(self):
        report = fixture()
        report["capabilities"][0]["device_supported"] = False
        report["search"]["maximal_placement"] = "C"
        with self.assertRaisesRegex(ValueError, "unsupported device"):
            bounded.validate_report(report)
        for change in (lambda r: r["search"]["ranked_feasible"][0].update(modeled_ms=-1),
                       lambda r: r["search"]["ranked_feasible"].reverse(),
                       lambda r: r["results"][0].update(modeled_ms=123)):
            report = fixture(); change(report)
            with self.assertRaises(ValueError):
                bounded.validate_report(report)
        report = fixture()
        report["results"] = report["results"][:1]
        report["search"].update(rejected=1, ranked_feasible=[{"placement":"C", "modeled_ms":99999}],
                                 maximal_feasible=False, maximal_rejection="")
        with self.assertRaisesRegex(ValueError, "rejection reason"):
            bounded.validate_report(report)

    def test_configuration_candidate_and_command_drift_rejected(self):
        for change in (lambda r: r["config"].update(device_bytes=99),
                       lambda r: r["search"].update(budget_exhausted=True),
                       lambda r: r["results"][0]["commands"][0].update(buffer_slot=7)):
            changed = fixture(); change(changed)
            with self.assertRaisesRegex(ValueError, "DRIFT"):
                self.bundle(evaluation=[changed])
        reordered = fixture(); reordered["results"].reverse()
        self.assertTrue(self.bundle(evaluation=[reordered])["conclusions_valid"])

    def test_saved_artifact_rejects_fabricated_oracle_regret_prediction_and_profiles(self):
        for change in (lambda b: b.update(oracle_available=True),
                       lambda b: b["policies"].update(oracle={}),
                       lambda b: b["plans"][0]["metrics"].update(regret_ms=0),
                       lambda b: b["plans"][0]["metrics"].update(regret_vs_best_observed_ms=999),
                       lambda b: b["plans"][0]["prediction"].update(boundary_ms=999),
                       lambda b: b["profiles"]["commands"].clear(),
                       lambda b: b["policies"]["best_observed_candidate"].update(placement="C")):
            artifact = self.bundle(); change(artifact)
            with self.assertRaises(ValueError):
                bounded.validate_bundle(artifact)

    def test_uncertainty_and_seed_are_reproducible_with_cpu_guard(self):
        left = self.bundle(profile_error=.4, seed=42)
        right = self.bundle(profile_error=.4, seed=42)
        self.assertEqual(left["injected_profiles"], right["injected_profiles"])
        self.assertEqual(left["policies"], right["policies"])
        for profile in left["injected_profiles"].values():
            self.assertGreaterEqual(profile["factor"], .6)
            self.assertLessEqual(profile["factor"], 1.4)
        self.assertIn(left["policies"]["robust"]["reason"], ("MINIMUM_UNCERTAINTY_SCORE", "CPU_GUARD_UNCERTAIN_ADVANTAGE"))

    def test_comparison_labels_distinct_candidate_sets_and_measured_policy_deltas(self):
        before = self.bundle()
        report = fixture(); report["config"]["device_bytes"] = 8
        report["results"] = report["results"][:1]
        report["search"].update(rejected=1, ranked_feasible=[{"placement":"C", "modeled_ms":99999}],
                                 maximal_feasible=False, maximal_rejection="E_MEMORY: too small")
        after = self.bundle([report], [deepcopy(report)])
        result = bounded.compare_bundles(before, after)
        self.assertEqual(result["artifact_type"], "bounded-placement-comparison")
        self.assertFalse(result["candidate_sets_equal"])
        self.assertFalse(result["oracle_available"])
        self.assertIn("No global optimum", result["comparison_scope"])
        self.assertIsNone(result["policies"]["maximal"]["metric_deltas"])
        self.assertTrue(any(c["entity_id"] == "device_bytes" for c in result["changes"]))
        self.assertEqual(result["policies"]["cpu"]["metric_deltas"]["p50_ms"], 0)
        failed = fixture(); failed["results"][0]["correct"] = False
        invalid = self.bundle(evaluation=[failed])
        result = bounded.compare_bundles(before, invalid)
        self.assertFalse(result["conclusions_valid"])
        self.assertTrue(all(p["metric_deltas"] is None for p in result["policies"].values()))

    def test_run_options_cannot_redirect_or_unbound_execution(self):
        for options in ({"flags": ["--json", "/tmp/replace"]}, {"search_budget": 1}, {"search_budget": 2049},
                        {"beam_width": 0}, {"beam_width": 33}, {"samples": 1}, {"calibration": 1},
                        {"seed": True}, {"profile_error": 1}, {"uncertainty_weight": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                bounded.run_study("missing", "missing", "missing", **options)


def integration(binary):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        graph = root / "chain.plg"
        lines = ["tensor X f32 1 2 -1 2"]
        for index in range(16):
            lines.append(f"relu n{index} {'X' if index == 0 else 't'+str(index-1)} t{index}")
        graph.write_text("\n".join(lines) + "\noutput t15\n")
        artifact = bounded.run_study(binary, graph, root / "study", flags=["--relu-supported"],
                                     search_budget=32, beam_width=4, warmup=1, calibration=2, samples=3, seed=17)
        assert artifact["conclusions_valid"] and not artifact["oracle_available"]
        assert len(artifact["graph"]["nodes"]) == 16
        assert 2 <= len(artifact["plans"]) <= 6
        assert artifact["search"]["evaluated"] <= 32
        assert len(artifact["raw_runs"]["invocations"]) == 6
        assert len({i["order_seed"] for i in artifact["raw_runs"]["invocations"]}) == 6
        for invocation in artifact["raw_runs"]["invocations"]:
            report = bounded.load_report(root / "study" / invocation["report"])
            assert all(r["output"] == [0, 2] for r in report["results"])
        bounded.replay(root / "study", root / "replay.json")
        assert json.loads((root / "replay.json").read_text())["analysis_matches"]
        with unittest.TestCase().assertRaisesRegex(ValueError, "output already exists"):
            bounded.replay(root / "study", root / "replay.json")
        with unittest.TestCase().assertRaisesRegex(ValueError, "must be empty"):
            bounded.run_study(binary, graph, root / "study")
        raw = root / "study/evaluation-000.json"
        changed = json.loads(raw.read_text()); changed["results"][0]["host_execution_ms"] += 1
        raw.write_text(json.dumps(changed))
        with unittest.TestCase().assertRaisesRegex(ValueError, "hash mismatch"):
            bounded.replay(root / "study", root / "replay-tampered.json")
        cli = subprocess.run([sys.executable, str(ROOT / "scripts/bounded_study.py"), "run", "--binary", str(binary),
             "--graph", str(graph), "--out", str(root / "cli-study"), "--search-budget", "16", "--beam-width", "2",
             "--warmup", "0", "--calibration", "2", "--samples", "2", "--", "--relu-supported"], capture_output=True, text=True)
        assert cli.returncode == 0, cli.stdout + cli.stderr
        assert json.loads(cli.stdout)["oracle_available"] is False
        print("PASS: real 16-node bounded study, separate invocations, candidate-only metrics, exact output, replay and tamper rejection")


if __name__ == "__main__":
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BoundedTests))
    if not result.wasSuccessful():
        sys.exit(1)
    if binary:
        integration(binary)
