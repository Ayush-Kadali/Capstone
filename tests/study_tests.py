#!/usr/bin/env python3
"""Independent timing fixtures exercise study math; optional real CLI smoke."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import study


def fixture(cpu_ms=12., device_ms=6., device_compute_ms=2.):
    graph = {"inputs": [{"name": "X", "shape": [1, 2], "dtype": "f32", "values": [-1, 2]}],
             "nodes": [{"id": "activation", "op": "ReLU", "inputs": ["X"], "output": "Y", "origins": ["activation"]}],
             "output": "Y"}
    def command(i, kind, backend, inputs=(), output="", op=None):
        return {"id": i, "version": 1, "kind": kind, "backend": backend, "inputs": list(inputs),
                "output": output, "shape": [1, 2] if kind != "SYNC" else [0, 0],
                "op": op, "node": "activation" if op else "", "origins": ["activation"] if op else [], "modeled_ms": 99999}
    cpu = [command(0, "ALLOC", "CPU", output="H:Y"),
           command(1, "COMPUTE", "CPU", ["H:X"], "H:Y", "ReLU")]
    device = [command(0, "ALLOC", "virtual-device", output="D:X"),
              command(1, "COPY", "virtual-device", ["H:X"], "D:X"),
              command(2, "ALLOC", "virtual-device", output="D:Y"),
              command(3, "COMPUTE", "virtual-device", ["D:X"], "D:Y", "ReLU"),
              command(4, "SYNC", "virtual-device"), command(5, "ALLOC", "CPU", output="H:Y"),
              command(6, "COPY", "CPU", ["D:Y"], "H:Y")]
    def result(placement, commands, durations, total):
        start = 0
        events = []
        for cmd, duration in zip(commands, durations):
            events.append({"command_id": cmd["id"], "node": cmd["node"], "name": cmd["kind"],
                           "lane": cmd["backend"], "start_us": start * 1000, "duration_us": duration * 1000,
                           "modeled_ms": 99999, "device_us": None})
            start += duration
        return {"placement": placement, "commands": commands, "events": events, "host_execution_ms": total,
                "correct": True, "modeled_ms": 99999, "h2d_bytes": 8 if placement == "A" else 0,
                "d2h_bytes": 8 if placement == "A" else 0, "peak_allocated_device_bytes": 16 if placement == "A" else 0,
                "synchronizations": 1 if placement == "A" else 0, "islands": 1 if placement == "A" else 0}
    return {"schema_version": 1, "source_graph": graph, "compiled_graph": graph,
            "capabilities": [{"node_id": "activation", "device_supported": True, "reason": "explicitly supported"}],
            "config": {"device_bytes": 100, "relu_supported": True}, "execution_backend": "software",
            "search": {"ranked_feasible": [{"placement": "C"}, {"placement": "A"}], "infeasible": 0},
            "results": [result("C", cpu, [1, 10], cpu_ms),
                        result("A", device, [.25, 1, .25, device_compute_ms, .5, .25, 1], device_ms)]}


def fixture_v3():
    report = fixture()
    report["schema_version"] = 3
    for graph in (report["source_graph"], report["compiled_graph"]):
        for node in graph["nodes"]:
            node["parameters"] = []
    for result in report["results"]:
        for command in result["commands"]:
            command.update(version=3, parameters=[])
    return report


class StudyMathTests(unittest.TestCase):
    def bundle(self, calibration=None, evaluation=None, **kwargs):
        return study.build_bundle(calibration or [fixture(), fixture()], evaluation or [fixture(), fixture()], **kwargs)

    def test_hardware_provenance_distinguishes_local_opencl_from_board(self):
        software = self.bundle()
        self.assertTrue(software["provenance"]["hardware_claim"].startswith("none;"))
        report = fixture()
        report["execution_backend"] = "OpenCL GPU: test device"
        bundle = self.bundle([report], [deepcopy(report)])
        claim = bundle["provenance"]["hardware_claim"]
        self.assertIn("local OpenCL backend execution was recorded", claim)
        self.assertIn("device-event timings are unavailable", claim)
        self.assertIn("not BeagleY-AI validation", claim)
        report["results"][1]["events"][3]["device_us"] = 12.5
        bundle = self.bundle([report], [deepcopy(report)])
        self.assertIn("device-event timings are present", bundle["provenance"]["hardware_claim"])

    def test_percentiles_interpolate_known_distribution(self):
        self.assertEqual(study.percentile([10, 2, 6, 4], .5), 5)
        self.assertAlmostEqual(study.percentile([10, 2, 6, 4], .95), 9.4)
        self.assertEqual(study.percentile([3], .95), 3)

    def test_oracle_and_regret_use_heldout_complete_plan_measurements(self):
        # Calibration likes A, independently held-out measurements like C.
        bundle = self.bundle(evaluation=[fixture(cpu_ms=8, device_ms=14), fixture(cpu_ms=10, device_ms=18)])
        self.assertEqual(bundle["policies"]["additive"]["placement"], "A")
        self.assertEqual(bundle["policies"]["oracle"]["placement"], "C")
        metrics = bundle["policies"]["additive"]["metrics"]
        self.assertEqual(metrics["p50_ms"], 16)
        self.assertEqual(metrics["regret_ms"], 7)
        self.assertAlmostEqual(metrics["regret_ratio"], 16 / 9 - 1)
        self.assertTrue(metrics["harmful_offload"])
        self.assertEqual(bundle["stability"]["additive"]["harmful_offload_rate"], 1)

    def test_predictions_never_use_illustrative_model(self):
        first = self.bundle()
        altered = fixture()
        for result in altered["results"]:
            result["modeled_ms"] = 0
            for command in result["commands"]:
                command["modeled_ms"] = 0
        second = self.bundle([altered, altered])
        self.assertEqual(first["injected_profiles"], second["injected_profiles"])
        self.assertEqual(first["policies"], second["policies"])

    def test_seeded_error_bounded_and_reproducible(self):
        profiles = study.calibrate([fixture()])
        a = study.perturb_profiles(profiles, .4, 25)
        self.assertEqual(a, study.perturb_profiles(profiles, .4, 25))
        self.assertNotEqual(a, study.perturb_profiles(profiles, .4, 26))
        for key, value in a.items():
            self.assertGreaterEqual(value["factor"], .6)
            self.assertLessEqual(value["factor"], 1.4)
            self.assertLessEqual(value["lower_ms"], profiles["commands"][key]["p50_ms"])
            self.assertGreaterEqual(value["upper_ms"], profiles["commands"][key]["p50_ms"])

    def test_uncertainty_guard_is_distinct_from_additive(self):
        # Device median remains attractive, but one calibration tail is large.
        calibration = [fixture(), fixture(), fixture(device_ms=105, device_compute_ms=100)]
        bundle = self.bundle(calibration)
        self.assertEqual(bundle["policies"]["additive"]["placement"], "A")
        self.assertEqual(bundle["policies"]["robust"]["placement"], "C")
        self.assertEqual(self.bundle(calibration, uncertainty_weight=0)["policies"]["robust"]["placement"], "A")

    def test_correctness_failure_suppresses_all_performance_conclusions(self):
        bad = fixture()
        bad["results"][1]["correct"] = False
        bundle = self.bundle(evaluation=[bad, fixture()])
        self.assertFalse(bundle["conclusions_valid"])
        self.assertIsNone(bundle["policies"]["oracle"]["plan_id"])
        for plan in bundle["plans"]:
            for key in ("regret_ms", "regret_ratio", "harmful_offload"):
                self.assertIsNone(plan["metrics"][key])
        for value in bundle["stability"].values():
            self.assertIsNone(value["harmful_offload_rate"])

    def test_bad_calibration_also_gates_conclusions(self):
        bad = fixture()
        bad["results"][0]["correct"] = False
        self.assertFalse(self.bundle([bad, fixture()])["conclusions_valid"])

    def test_missing_event_not_replaced_by_modeled_timing(self):
        bad = fixture()
        bad["results"][0]["events"].pop()
        with self.assertRaisesRegex(ValueError, "execution event"):
            self.bundle([bad])

    def test_duplicate_or_incomplete_enumeration_rejected(self):
        for alter in (lambda r: r["results"].append(deepcopy(r["results"][0])),
                      lambda r: r["results"].pop(),
                      lambda r: r["search"].update(infeasible=1)):
            bad = fixture()
            alter(bad)
            with self.assertRaises(ValueError):
                study.validate_report(bad)

    def test_schema_invalid_and_nan_rejected(self):
        for bad in ({}, {"schema_version": 99}, None):
            with self.assertRaises(ValueError):
                study.validate_report(bad)
        for field, value in (("host_execution_ms", float("nan")), ("correct", "yes")):
            bad = fixture()
            bad["results"][0][field] = value
            with self.assertRaises(ValueError):
                study.validate_report(bad)
        with self.assertRaises(ValueError):
            study.validate_bundle({"schema_version": 1, "artifact_type": "placement-study"})

    def test_future_or_mixed_command_versions_are_rejected(self):
        report = fixture()
        report["results"][0]["commands"][0]["version"] = 999
        with self.assertRaisesRegex(ValueError, "command version"):
            study.validate_report(report)
        bundle = self.bundle()
        bundle["plans"][0]["commands"][0]["version"] = 999
        with self.assertRaisesRegex(ValueError, "command version"):
            study.validate_bundle(bundle)
        modern = fixture()
        modern["schema_version"] = 2
        for result in modern["results"]:
            for command in result["commands"]:
                command["version"] = 2
        self.assertTrue(self.bundle([modern, modern], [modern])["conclusions_valid"])

    def test_abi3_parameters_preserve_old_identities_and_separate_new_profiles(self):
        old = fixture()["results"][0]["commands"][1]
        old_signature = {key: old.get(key) for key in ("kind", "backend", "node", "op", "inputs", "output", "origins")}
        self.assertEqual(study.stable_commands([old])[0]["stable_id"], "cmd-" + study.digest(old_signature)[:20] + "-0")
        self.assertNotIn("parameters", json.loads(study.command_key(old, {"X": [1, 2]}, "software")))
        v3 = fixture_v3()
        self.assertTrue(self.bundle([v3], [deepcopy(v3)])["conclusions_valid"])
        first = {**deepcopy(old), "version": 3, "op": "Conv2D", "parameters": [1,1,2,2,1,1,1,1,1,0,0,0,0,1,1,1]}
        second = deepcopy(first)
        second["parameters"][7] = 2  # distinct stride despite identical packed tensor shapes
        for key in (lambda c: study.command_key(c, {"X": [1, 2]}, "software"),
                    lambda c: study.stable_commands([c])[0]["stable_id"],
                    lambda c: study.plan_structure({"commands": [c]})):
            self.assertNotEqual(key(first), key(second))
        self.assertEqual(json.loads(study.command_key(first, {}, "software"))["parameters"], first["parameters"])
        self.assertNotEqual(study.stable_commands([old])[0]["stable_id"], study.stable_commands([{**old, "version": 3, "parameters": []}])[0]["stable_id"])

    def test_abi3_parameter_contract_rejects_malformed_arrays_counts_and_graph_mismatch(self):
        for parameters in (None, "1,2", [True], [1.0], [-1], [65537], [1]):
            report = fixture_v3()
            report["results"][0]["commands"][1]["parameters"] = parameters
            with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, "parameter"):
                study.validate_report(report)
        report = fixture_v3()
        del report["results"][0]["commands"][0]["parameters"]
        with self.assertRaisesRegex(ValueError, "parameter"):
            study.validate_report(report)
        for op, count in study.PARAMETER_COUNTS.items():
            descriptor = {"op": op, "kind": "COMPUTE", "parameters": [1] * count}
            self.assertEqual(study.validate_parameters(descriptor, 3), [1] * count)
            with self.subTest(op=op), self.assertRaisesRegex(ValueError, "count"):
                study.validate_parameters({**descriptor, "parameters": [1] * (count + 1)}, 3)
        for version in (1, 2):
            with self.assertRaisesRegex(ValueError, "version 3"):
                study.validate_parameters({"kind": "COMPUTE", "op": "Reshape", "parameters": [1, 2]}, version)
        with self.assertRaisesRegex(ValueError, "noncompute"):
            study.validate_parameters({"kind": "ALLOC", "parameters": [1, 2]}, 3)
        report = fixture_v3()
        report["compiled_graph"]["nodes"][0].update(op="Reshape", parameters=[1, 2])
        for result in report["results"]:
            next(c for c in result["commands"] if c["kind"] == "COMPUTE").update(op="Reshape", parameters=[1, 2])
        self.assertTrue(self.bundle([report], [deepcopy(report)])["conclusions_valid"])
        changed = deepcopy(report)
        changed["results"][0]["commands"][1]["parameters"] = [2, 1]
        with self.assertRaisesRegex(ValueError, "differ from graph"):
            study.validate_report(changed)
        bundle = self.bundle([report], [deepcopy(report)])
        bundle["graph"]["nodes"][0]["parameters"] = [2, 1]
        with self.assertRaisesRegex(ValueError, "differ from graph"):
            study.validate_bundle(bundle)
        report = fixture_v3()
        report["results"][0]["initial_buffers"] = [{"id":0,"version":3,"kind":"ALLOC","parameters":[1]}]
        with self.assertRaisesRegex(ValueError, "noncompute"):
            study.validate_report(report)

    def test_abi3_parameter_changes_detect_calibration_drift_and_bundle_tampering(self):
        original = fixture_v3()
        original["compiled_graph"]["nodes"][0].update(op="Reshape", parameters=[1, 2])
        for result in original["results"]:
            next(c for c in result["commands"] if c["kind"] == "COMPUTE").update(op="Reshape", parameters=[1, 2])
        changed = deepcopy(original)
        changed["compiled_graph"]["nodes"][0]["parameters"] = [2, 1]
        for result in changed["results"]:
            next(c for c in result["commands"] if c["kind"] == "COMPUTE")["parameters"] = [2, 1]
        with self.assertRaisesRegex(ValueError, "DRIFT"):
            self.bundle([original], [changed])
        bundle = self.bundle([original], [deepcopy(original)])
        bundle["plans"][0]["commands"][1]["parameters"] = [2, 1]
        with self.assertRaisesRegex(ValueError, "stable command identity"):
            study.validate_bundle(bundle)
        bundle = self.bundle([original], [deepcopy(original)])
        bundle["plans"][0]["prediction"]["command_estimates"][1]["profile_key"] = "{}"
        with self.assertRaisesRegex(ValueError, "parameter/profile identity"):
            study.validate_bundle(bundle)
        bundle = self.bundle([original], [deepcopy(original)])
        key = next(key for key in bundle["profiles"]["commands"] if json.loads(key)["kind"] == "COMPUTE")
        changed_key = json.loads(key)
        del changed_key["parameters"]
        bundle["profiles"]["commands"][json.dumps(changed_key)] = bundle["profiles"]["commands"].pop(key)
        with self.assertRaisesRegex(ValueError, "parameters"):
            study.validate_bundle(bundle)
        study.validate_options(["--cnn-supported", "--relu-supported"])

    def test_plan_drift_rejected_but_order_changes_allowed(self):
        reordered = fixture()
        reordered["results"].reverse()
        self.bundle(evaluation=[reordered])
        changed = fixture()
        changed["results"][0]["commands"][1]["inputs"] = ["H:other"]
        with self.assertRaisesRegex(ValueError, "DRIFT"):
            self.bundle(evaluation=[changed])
        changed = fixture()
        changed["execution_backend"] = "different hardware"
        with self.assertRaisesRegex(ValueError, "DRIFT"):
            self.bundle(evaluation=[changed])

    def test_shapes_direction_and_backend_separate_profile_keys(self):
        profiles = study.calibrate([fixture()])
        compute = [json.loads(k) for k in profiles["commands"] if json.loads(k)["kind"] == "COMPUTE"]
        self.assertEqual(len(compute), 2)
        self.assertEqual({x["backend"] for x in compute}, {"CPU", "virtual-device"})
        self.assertTrue(all(x["input_shapes"] == [[1, 2]] for x in compute))
        self.assertEqual(len(profiles["subgraphs"]), 2)

    def test_launch_bearing_compute_context_does_not_alias_continuing_kernel(self):
        command = deepcopy(fixture()["results"][1]["commands"][3])
        command["id"] = 2
        prior = deepcopy(command)
        prior["id"] = 0
        sync = deepcopy(fixture()["results"][1]["commands"][4])
        sync["id"] = 1
        shapes = {"X": [1, 2], "Y": [1, 2]}
        continuing = study.command_profile_keys({"commands": [prior, sync, command]}, shapes, "host software with injected waits")
        prior["backend"] = "CPU"
        starting = study.command_profile_keys({"commands": [prior, sync, command]}, shapes, "host software with injected waits")
        self.assertNotEqual(continuing[2], starting[2])
        self.assertFalse(json.loads(continuing[2])["device_island_start"])
        self.assertTrue(json.loads(starting[2])["device_island_start"])
        # An intervening SYNC alone does not restart the executor's launch model.
        self.assertTrue(json.loads(continuing[0])["device_island_start"])

    def test_replay_projection_covers_predictions_profiles_metrics_and_stability(self):
        original = self.bundle()
        for change in (lambda b: b["plans"][0]["prediction"].update(boundary_ms=999999),
                       lambda b: b["profiles"].update(provenance="altered"),
                       lambda b: b["plans"][0]["metrics"].update(tile_count=999),
                       lambda b: b["stability"]["cpu"].update(modal_fraction=.1)):
            modified = deepcopy(original)
            change(modified)
            self.assertNotEqual(study.analysis_projection(original), study.analysis_projection(modified))
        modified = deepcopy(original)
        modified["experiment"]["selection_planning_ms"] = 999
        self.assertEqual(study.analysis_projection(original), study.analysis_projection(modified))

    def test_zero_latency_does_not_divide_by_zero(self):
        bundle = self.bundle(evaluation=[fixture(cpu_ms=0, device_ms=0)])
        self.assertTrue(all(p["metrics"]["regret_ratio"] is None for p in bundle["plans"]))

    def test_structural_diff_ignores_command_ordinal_renumbering(self):
        commands = fixture()["results"][1]["commands"]
        before = study.stable_commands(commands)
        for command in commands:
            command["id"] += 3
        after = study.stable_commands(commands)
        self.assertEqual([c["stable_id"] for c in before], [c["stable_id"] for c in after])
        bundle = self.bundle()
        other = self.bundle(evaluation=[fixture(cpu_ms=15)])
        diff = study.compare_bundles(bundle, other)
        self.assertEqual(diff["changes"], [])
        self.assertEqual(diff["policies"]["cpu"]["metric_deltas"]["p50_ms"], 3)

    def test_configuration_change_is_exact_not_inferred_cause(self):
        before, after = self.bundle(), self.bundle()
        after["experiment"]["config"]["device_bytes"] = 200
        diff = study.compare_bundles(before, after)
        self.assertEqual(diff["changes"][0]["reason_code"], "CONFIGURATION_CHANGED")
        self.assertIn("TIMING_DELTA_OBSERVED_NOT_CAUSAL", diff["policies"]["cpu"]["reason_codes"])

    def test_options_cannot_redirect_core_outputs(self):
        for flags in (["--json", "/tmp/other"], ["--graph", "other"], ["--device-bytes"], ["--mystery"]):
            with self.assertRaises(ValueError):
                study.validate_options(flags)
        study.validate_options(["--relu-supported", "--backend", "software", "--device-bytes", "200"])

    def test_bundle_rejects_invalid_cross_references_and_statistics(self):
        changes = [lambda b: b["plans"][0]["events"][0].update(command_id=900),
                   lambda b: b["plans"][0]["commands"][3].update(node="missing"),
                   lambda b: b["plans"][0]["commands"][1].update(inputs=["H:missing"]),
                   lambda b: b["plans"][0].update(placement="AAAA"),
                   lambda b: b["policies"]["cpu"].update(placement="A"),
                   lambda b: b["plans"][0]["metrics"].update(p50_ms=888),
                   lambda b: b["plans"][0]["metrics"].update(command_count=True),
                   lambda b: b["graph"]["nodes"][0].update(inputs=["unknown"]),
                   lambda b: b["plans"][0]["buffers"][0].update(bytes=999)]
        for change in changes:
            bundle = self.bundle()
            change(bundle)
            with self.assertRaises(ValueError):
                study.validate_bundle(bundle)

    def test_event_labels_must_match_the_referenced_command(self):
        for field, value in (("name", "COPY"), ("lane", "virtual-device"), ("node", "other")):
            report = fixture()
            report["results"][0]["events"][1][field] = value
            with self.assertRaisesRegex(ValueError, "event metadata"):
                study.validate_report(report)
            bundle = self.bundle()
            cpu = next(p for p in bundle["plans"] if p["placement"] == "C")
            cpu["events"][1][field] = value
            with self.assertRaises(ValueError):
                study.validate_bundle(bundle)

    def test_buffer_lifetime_and_unavailable_capability_provenance(self):
        bundle = self.bundle()
        plan = next(p for p in bundle["plans"] if p["placement"] == "A")
        buffers = {b["handle"]: b for b in plan["buffers"]}
        self.assertEqual(buffers["D:X"]["allocation_id"], 0)
        self.assertEqual(buffers["D:X"]["last_use_id"], 3)
        self.assertEqual(buffers["D:X"]["bytes"], 8)
        self.assertIsNone(buffers["D:X"]["free_id"])
        self.assertIsNone(buffers["D:X"]["slot"])
        self.assertEqual(bundle["capabilities"]["status"], "available")

    def test_canonical_maximal_is_explicitly_infeasible_without_substitution(self):
        report = fixture()
        report["results"] = report["results"][:1]
        report["search"] = {"ranked_feasible": [{"placement": "C"}], "infeasible": 1}
        bundle = self.bundle([report, report], [report, report])
        maximal = bundle["policies"]["maximal"]
        self.assertEqual(maximal["status"], "infeasible")
        self.assertEqual(maximal["requested_placement"], "A")
        self.assertIsNone(maximal["placement"])
        self.assertIsNone(maximal["metrics"])
        self.assertEqual(bundle["policies"]["maximal_feasible"]["placement"], "C")
        self.assertIsNone(study.compare_bundles(self.bundle(), bundle)["policies"]["maximal"]["metric_deltas"])
        bundle["policies"]["maximal"] = deepcopy(bundle["policies"]["maximal_feasible"])
        with self.assertRaisesRegex(ValueError, "maximal"):
            study.validate_bundle(bundle)

    def test_missing_capabilities_make_maximal_unavailable(self):
        report = fixture()
        del report["capabilities"]
        bundle = self.bundle([report, report], [report, report])
        self.assertEqual(bundle["policies"]["maximal"]["status"], "unavailable")

    def test_mirrored_inputs_have_separate_preparation_namespace_and_cost(self):
        report = fixture()
        report["config"]["mirrored_inputs"] = ["X"]
        for result in report["results"]:
            result["initial_buffers"] = [{"id": 0, "version": 1, "kind": "ALLOC", "backend": "virtual-device",
                "output": "D:X", "shape": [1, 2], "buffer_slot": 0}]
            result["input_preparation_ms"] = .5
            result["preparation_h2d_bytes"] = 8
            if result["placement"] == "A":
                result["commands"] = result["commands"][2:]
                result["events"] = result["events"][2:]
                for index, command in enumerate(result["commands"]):
                    command["id"] = index
                for index, event in enumerate(result["events"]):
                    event["command_id"] = index
        bundle = self.bundle([report, report], [report, report])
        for plan in bundle["plans"]:
            initial = next(b for b in plan["buffers"] if b["handle"] == "D:X")
            self.assertIsNone(initial["allocation_id"])
            self.assertEqual(initial["initial_buffer_id"], 0)
            self.assertEqual(plan["metrics"]["input_preparation"]["p50_ms"], .5)
            self.assertEqual(plan["metrics"]["preparation_h2d_bytes"], 8)
        bundle["plans"][0]["initial_buffers"][0]["output"] = "D:unknown"
        with self.assertRaisesRegex(ValueError, "initial device buffer"):
            study.validate_bundle(bundle)

    def test_invalid_error_and_sample_bounds_fail_before_launch(self):
        for error in (-.1, 1, float("nan")):
            with self.assertRaises(ValueError):
                study.perturb_profiles({"commands": {}}, error, 1)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                study.run_study("nonexistent", "nonexistent", folder, samples=1)


def integration(binary):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        graph = root / "input.plg"
        graph.write_text("tensor X f32 1 2 -1 2\nrelu activation X Y\noutput Y\n")
        bundle = study.run_study(binary, graph, root / "evidence", flags=["--relu-supported"],
                                 warmup=0, calibration=2, samples=3, seed=72, profile_error=.2)
        assert bundle["conclusions_valid"] and len(bundle["plans"]) == 2
        assert len(bundle["raw_runs"]["invocations"]) == 5
        assert len(set(r["order_seed"] for r in bundle["raw_runs"]["invocations"])) == 5
        assert all(p["metrics"]["count"] == 3 for p in bundle["plans"])
        replay = root / "replay.json"
        subprocess.run([sys.executable, str(Path(study.__file__)), "replay", str(root / "evidence"), "--out", str(replay)], check=True)
        assert json.loads(replay.read_text())["policies_match"]
        bundle_path = root / "evidence" / "bundle.json"
        saved_bundle = bundle_path.read_text()
        changed_bundle = json.loads(saved_bundle)
        changed_bundle["plans"][0]["prediction"]["boundary_ms"] = 999999
        bundle_path.write_text(json.dumps(changed_bundle))
        forged = subprocess.run([sys.executable, str(Path(study.__file__)), "replay", str(root / "evidence"), "--out", str(root / "forged-analysis.json")], capture_output=True, text=True)
        assert forged.returncode != 0 and "analysis mismatch" in forged.stderr
        assert not json.loads((root / "forged-analysis.json").read_text())["analysis_matches"]
        bundle_path.write_text(saved_bundle)
        original = root / "evidence" / "calibration-000.json"
        original.write_text(original.read_text().replace('"correct":true', '"correct":false'))
        tampered = subprocess.run([sys.executable, str(Path(study.__file__)), "replay", str(root / "evidence"), "--out", str(root / "tampered.json")], capture_output=True, text=True)
        assert tampered.returncode != 0 and "hash mismatch" in tampered.stderr
        rejected = root / "rejected-local-budget"
        try:
            study.run_study(binary, graph, rejected, flags=["--tile-size", "2", "--local-bytes", "47"],
                            warmup=1, calibration=2, samples=2)
            raise AssertionError("invalid local budget unexpectedly accepted")
        except RuntimeError as error:
            assert "E_CONFIG: tiled scratch exceeds local memory budget" in str(error), str(error)
            assert "see warmup-000.log" in str(error)
            assert "E_CONFIG" in (rejected / "warmup-000.log").read_text()
        if bundle["experiment"]["command_version"] == 3:
            cnn = root / "cnn.plg"
            cnn.write_text("tensor X f32 1 4 -1 2 3 4\ntensor W f32 1 1 2\n"
                "conv2d conv X W C 1 1 2 2 1 1 1 1 1 0 0 0 0 1 1 1\n"
                "maxpool2d max C M 1 1 2 2 2 2 2 2 0 0 0 0 0\n"
                "averagepool2d avg M A 1 1 1 1 1 1 1 1 0 0 0 0 0\n"
                "globalaveragepool global A G 1 1 1 1\nreshape reshape G Y 1 1\noutput Y\n")
            cnn_bundle = study.run_study(binary, cnn, root / "cnn-study", flags=["--cnn-supported", "--resident-input", "X"],
                                        warmup=0, calibration=2, samples=2)
            assert cnn_bundle["conclusions_valid"] and len(cnn_bundle["plans"]) == 32
            assert {n["op"] for n in cnn_bundle["graph"]["nodes"]} == {"Conv2D", "MaxPool2D", "AveragePool2D", "GlobalAveragePool", "Reshape"}
            assert all("parameters" in json.loads(key) for key in cnn_bundle["profiles"]["commands"])
            assert all(r["output"] == [8] for r in json.loads((root / "cnn-study/evaluation-000.json").read_text())["results"])
            print("PASS: ABI3 CNN five operators, 32 placements, parameterized profiles and exact expected output")
        print("PASS: real CLI study, seeded execution order, saved raw artifacts, deterministic replay, tamper rejection, visible configuration failure")


if __name__ == "__main__":
    binary = sys.argv[1] if len(sys.argv) > 1 else None
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StudyMathTests))
    if not result.wasSuccessful():
        sys.exit(1)
    if binary:
        integration(str(Path(binary).resolve()))
