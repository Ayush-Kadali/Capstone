#!/usr/bin/env python3
"""Measure a deterministic bounded candidate search without claiming a global oracle.

Calibration and held-out evaluation are separate core invocations. Candidate
generation uses the core's illustrative costs; measured policies select only
among the retained candidates. Exhaustive studies remain a separate artifact.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import study

ARTIFACT_TYPE = "bounded-placement-study"
MAX_GRAPH_BYTES = 16_000_000
MAX_REPORT_BYTES = 64_000_000


def integer(value, name, minimum=0, maximum=2**32 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"E_BOUNDED_SCHEMA: {name} must be an integer in [{minimum},{maximum}]")
    return value


def shape(value, name, zero=False):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"E_BOUNDED_SCHEMA: {name} requires a packed rank-2 shape")
    for component in value:
        integer(component, name, 0 if zero else 1, 65536)
    if math.prod(value) > 65536:
        raise ValueError(f"E_BOUNDED_SCHEMA: {name} exceeds tensor element bound")
    return value


def validate_report(report):
    """Validate measured retained candidates; never synthesize unvisited counts."""
    try:
        if type(report["schema_version"]) is not int or report["schema_version"] != 3:
            raise ValueError("E_BOUNDED_SCHEMA: core report/command version 3 required")
        search = report["search"]
        if search["exhaustive"] is not False or search["strategy"] != "bounded-beam" or search["infeasible"] is not None:
            raise ValueError("E_BOUNDED_SCHEMA: explicit nonexhaustive search metadata required")
        evaluated = integer(search["evaluated"], "evaluated", 1)
        rejected = integer(search["rejected"], "rejected", 0, evaluated)
        budget = integer(search["max_evaluations"], "max_evaluations", 2, 2048)
        beam = integer(search["beam_width"], "beam_width", 1, 32)
        if evaluated > budget or type(search["budget_exhausted"]) is not bool:
            raise ValueError("E_BOUNDED_SCHEMA: invalid search budget accounting")
        if search["budget_exhausted"] and evaluated != budget:
            raise ValueError("E_BOUNDED_SCHEMA: exhausted search did not use its budget")
        graph = report["compiled_graph"]
        nodes = graph["nodes"]
        if not isinstance(nodes, list) or not 0 <= len(nodes) <= 64:
            raise ValueError("E_BOUNDED_SCHEMA: graph exceeds bounded node limit")
        node_ids = [node["id"] for node in nodes]
        if len(set(node_ids)) != len(node_ids) or any(not isinstance(n, str) or not n for n in node_ids):
            raise ValueError("E_BOUNDED_SCHEMA: invalid graph node identities")
        tensors = {}
        for tensor in graph["inputs"]:
            name = tensor["name"]
            if not isinstance(name, str) or not name or name in tensors or tensor["dtype"] != "f32":
                raise ValueError("E_BOUNDED_SCHEMA: invalid graph input")
            tensors[name] = shape(tensor["shape"], "input shape")
        known = set(tensors)
        for node in nodes:
            study.validate_parameters(node, 3, node=True)
            if any(name not in known for name in node["inputs"]) or node["output"] in known:
                raise ValueError("E_BOUNDED_SCHEMA: invalid topological graph")
            known.add(node["output"])
        if graph["output"] not in known:
            raise ValueError("E_BOUNDED_SCHEMA: unknown graph output")
        for node in report["source_graph"]["nodes"]:
            study.validate_parameters(node, 3, node=True)
        caps = report["capabilities"]
        if len(caps) != len(nodes) or {c["node_id"] for c in caps} != set(node_ids) or any(type(c["device_supported"]) is not bool for c in caps):
            raise ValueError("E_BOUNDED_SCHEMA: invalid capability identities")
        supported = {c["node_id"]: c["device_supported"] for c in caps}
        maximal = study.strict_maximal_placement(report)
        if search["maximal_placement"] != maximal or type(search["maximal_feasible"]) is not bool:
            raise ValueError("E_BOUNDED_SCHEMA: strict maximal metadata contradicts capabilities")
        results = report["results"]
        if not isinstance(results, list) or not 1 <= len(results) <= min(beam + 2, evaluated - rejected):
            raise ValueError("E_BOUNDED_SCHEMA: invalid retained candidate count")
        placements = [r["placement"] for r in results]
        ranked = [r["placement"] for r in search["ranked_feasible"]]
        ranked_costs = [(study.finite(r["modeled_ms"], "ranked modeled cost"), r["placement"]) for r in search["ranked_feasible"]]
        if ranked_costs != sorted(ranked_costs):
            raise ValueError("E_BOUNDED_SCHEMA: retained candidates must be cost/placement ranked")
        modeled_costs = dict((placement, cost) for cost, placement in ranked_costs)
        if len(set(placements)) != len(placements) or len(set(ranked)) != len(ranked) or set(placements) != set(ranked):
            raise ValueError("E_BOUNDED_SCHEMA: every retained candidate must be measured once")
        if "C" * len(nodes) not in placements:
            raise ValueError("E_BOUNDED_SCHEMA: CPU candidate missing")
        if search["maximal_feasible"] != (maximal in placements):
            raise ValueError("E_BOUNDED_SCHEMA: feasible strict maximal must be retained")
        if not search["maximal_feasible"] and (not isinstance(search["maximal_rejection"], str) or not search["maximal_rejection"].startswith("E_MEMORY") or rejected < 1):
            raise ValueError("E_BOUNDED_SCHEMA: strict maximal rejection reason missing")
        for result in results:
            placement = result["placement"]
            if not isinstance(placement, str) or len(placement) != len(nodes) or set(placement) - {"C", "A"}:
                raise ValueError("E_BOUNDED_SCHEMA: invalid placement")
            if any(lane == "A" and not supported[node_ids[index]] for index, lane in enumerate(placement)):
                raise ValueError("E_BOUNDED_SCHEMA: candidate executes unsupported device operation")
            if study.finite(result["modeled_ms"], "result modeled cost") != modeled_costs[placement]:
                raise ValueError("E_BOUNDED_SCHEMA: candidate modeled cost differs from search ranking")
            if type(result["correct"]) is not bool:
                raise ValueError("E_BOUNDED_SCHEMA: correctness must be boolean")
            study.finite(result["host_execution_ms"], "host_execution_ms")
            commands, events = result["commands"], result["events"]
            if any(type(c["id"]) is not int for c in commands) or [c["id"] for c in commands] != list(range(len(commands))):
                raise ValueError("E_BOUNDED_SCHEMA: command IDs must be contiguous integers")
            initial = result.get("initial_buffers", [])
            if any(type(c["id"]) is not int for c in initial) or [c["id"] for c in initial] != list(range(len(initial))):
                raise ValueError("E_BOUNDED_SCHEMA: initial buffer IDs invalid")
            handles = {"H:" + name: dims for name, dims in tensors.items()}
            for descriptor in initial:
                output = descriptor["output"]
                if descriptor["kind"] != "ALLOC" or descriptor["backend"] != "virtual-device" or output[:2] != "D:" or output[2:] not in tensors or output in handles or descriptor["shape"] != tensors[output[2:]]:
                    raise ValueError("E_BOUNDED_SCHEMA: invalid initial device buffer")
                handles[output] = descriptor["shape"]
            for command in commands + initial:
                if type(command["version"]) is not int or command["version"] != 3:
                    raise ValueError("E_BOUNDED_SCHEMA: mixed command version")
                study.validate_parameters(command, 3)
                shape(command["shape"], "command shape", zero=True)
            compute_nodes = []
            for command in commands:
                kind = command["kind"]
                if kind not in ("ALLOC", "COPY", "COMPUTE", "SYNC", "FREE") or command["backend"] not in ("CPU", "virtual-device"):
                    raise ValueError("E_BOUNDED_SCHEMA: invalid command kind/backend")
                if any(name not in handles for name in command["inputs"]):
                    raise ValueError("E_BOUNDED_SCHEMA: command reads unavailable buffer")
                output = command["output"]
                if output and (output[:2] not in ("H:", "D:") or output[2:] not in known):
                    raise ValueError("E_BOUNDED_SCHEMA: invalid output tensor handle")
                if kind == "ALLOC":
                    if not output or output in handles:
                        raise ValueError("E_BOUNDED_SCHEMA: invalid allocation")
                    handles[output] = command["shape"]
                elif output and (output not in handles or handles[output] != command["shape"]):
                    raise ValueError("E_BOUNDED_SCHEMA: command output storage mismatch")
                if kind == "COMPUTE":
                    index = len(compute_nodes)
                    if index >= len(nodes):
                        raise ValueError("E_BOUNDED_SCHEMA: extra compute command")
                    node = nodes[index]
                    if command["node"] != node["id"] or command["op"] != node["op"] or command["parameters"] != node["parameters"] or [name[2:] for name in command["inputs"]] != node["inputs"] or output[2:] != node["output"] or command["backend"] != ("CPU" if placement[index] == "C" else "virtual-device"):
                        raise ValueError("E_BOUNDED_SCHEMA: compute descriptor differs from graph/placement")
                    compute_nodes.append(node["id"])
                if kind == "FREE":
                    for name in ([output] if output else command["inputs"]):
                        handles.pop(name, None)
            if compute_nodes != node_ids:
                raise ValueError("E_BOUNDED_SCHEMA: incomplete compute schedule")
            if any(type(e["command_id"]) is not int for e in events) or sorted(e["command_id"] for e in events) != list(range(len(commands))):
                raise ValueError("E_BOUNDED_SCHEMA: each command requires one event")
            for event in events:
                command = commands[event["command_id"]]
                if (event["name"], event["lane"], event["node"]) != (command["kind"], command["backend"], command["node"]):
                    raise ValueError("E_BOUNDED_SCHEMA: event contradicts command")
                study.finite(event["start_us"], "event start")
                study.finite(event["duration_us"], "event duration")
                if event.get("device_us") is not None:
                    study.finite(event["device_us"], "device event duration")
        json.dumps(report, allow_nan=False)
        return report
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"E_BOUNDED_SCHEMA: malformed candidate report ({error})") from error


def structure(report):
    return {"graph": report["compiled_graph"], "config": report["config"],
            "capabilities": report["capabilities"], "search": report["search"],
            "plans": {r["placement"]: study.plan_structure(r) for r in report["results"]},
            "backends": {r["placement"]: r.get("execution_backend", report.get("execution_backend")) for r in report["results"]}}


def build_bundle(calibration_reports, evaluation_reports, *, seed=20261004, profile_error=0., uncertainty_weight=1., metadata=None):
    if not calibration_reports or not evaluation_reports:
        raise ValueError("E_BOUNDED_SAMPLE: separate calibration and evaluation reports required")
    reports = calibration_reports + evaluation_reports
    for report in reports:
        validate_report(report)
    reference = reports[0]
    identity = structure(reference)
    if any(structure(report) != identity for report in reports[1:]):
        raise ValueError("E_BOUNDED_DRIFT: retained candidate set, graph, configuration or command plans changed")
    profiles = study.calibrate(calibration_reports, validator=validate_report)
    predictions, selections, robust_reason, planning_ms, injected = study.predict(reference, profiles, profile_error, seed, uncertainty_weight)
    valid = all(r["correct"] for report in reports for r in report["results"])
    metrics = {}
    for result in reference["results"]:
        placement = result["placement"]
        measured = [next(r for r in report["results"] if r["placement"] == placement) for report in evaluation_reports]
        samples = [r["host_execution_ms"] for r in measured]
        metrics[placement] = {**study.distribution(samples), "samples_ms": samples,
                              "correct": all(r["correct"] for report in reports for r in report["results"] if r["placement"] == placement),
                              "h2d_bytes": result["h2d_bytes"], "d2h_bytes": result["d2h_bytes"],
                              "peak_allocated_device_bytes": result["peak_allocated_device_bytes"],
                              "command_count": len(result["commands"]),
                              "synchronizations": result["synchronizations"], "islands": result["islands"]}
        for field in ("end_to_end_ms", "queue_wait_ms", "input_preparation_ms"):
            values = [r.get(field) for r in measured]
            metrics[placement][field.removesuffix("_ms")] = study.distribution(values) if all(v is not None for v in values) else None
        for field in ("diagnostic_readback_bytes", "software_buffer_reuses", "runtime_peak_device_reserved_bytes", "tile_count", "tile_load_bytes", "peak_local_bytes", "device_buffer_slots", "buffer_reuses", "peak_live_device_bytes", "preparation_h2d_bytes", "injected_boundary_ms"):
            metrics[placement][field] = result.get(field)
    best = min(metrics, key=lambda p: (metrics[p]["p50_ms"], p)) if valid else None
    plans = []
    examples = {r["placement"]: r for r in evaluation_reports[0]["results"]}
    for candidate in reference["results"]:
        placement = candidate["placement"]
        result = examples[placement]
        metric = metrics[placement]
        metric["regret_vs_best_observed_ms"] = max(0., metric["p50_ms"] - metrics[best]["p50_ms"]) if valid else None
        partitions = study.partition_records(reference["compiled_graph"], placement)
        partition_by_node = {node: part["id"] for part in partitions for node in part["node_ids"]}
        commands = [{**command, "partition_id": partition_by_node.get(command.get("node"))} for command in study.stable_commands(result["commands"])]
        command_by_id = {c["id"]: c for c in commands}
        events = [{**event, "id": "event-" + command_by_id[event["command_id"]]["stable_id"],
                   "command_stable_id": command_by_id[event["command_id"]]["stable_id"],
                   "partition_id": command_by_id[event["command_id"]]["partition_id"]} for event in result["events"]]
        plans.append({"id": "candidate-" + (placement or "empty"), "placement": placement,
                      "metrics": metric, "prediction": predictions[placement],
                      "commands": commands, "events": events, "partitions": partitions,
                      "buffers": study.buffer_records(reference, result), "passes": result.get("passes", []),
                      "illustrative_modeled_ms": result["modeled_ms"],
                      "initial_buffers": result.get("initial_buffers", []),
                      "execution_backend": result.get("execution_backend", reference.get("execution_backend"))})
    policies = {}
    selections["best_observed_candidate"] = best
    for name, placement in selections.items():
        if not valid:
            policies[name] = {"status": "suppressed_correctness_failure", "placement": None, "metrics": None}
        elif placement is None:
            policies[name] = {"status": "infeasible", "placement": None, "metrics": None,
                              "requested_placement": reference["search"]["maximal_placement"],
                              "reason": reference["search"]["maximal_rejection"]}
        else:
            policies[name] = {"status": "available", "placement": placement, "metrics": metrics[placement]}
    policies["robust"]["reason"] = robust_reason if valid else "SUPPRESSED_CORRECTNESS_FAILURE"
    for name, policy in policies.items():
        policy["plan_id"] = "candidate-" + (policy["placement"] or "empty") if policy["placement"] is not None else None
        policy["reason_code"] = policy.get("reason", "BEST_OBSERVED_CANDIDATE_ONLY" if name == "best_observed_candidate" else "RETAINED_CANDIDATE_SELECTION")
    bundle = {"schema_version": 1, "artifact_type": ARTIFACT_TYPE, "exhaustive": False,
              "oracle_available": False, "global_regret_available": False, "conclusions_valid": valid,
              "policy_scope": "Retained measured candidates only; unvisited placements may be faster.",
              "best_observed_definition": "Minimum held-out evaluation p50 among retained candidates; not a global oracle.",
              "candidate_generation_scope": "Core deterministic bounded beam ranked by illustrative costs, not measured latency.",
              "displayed_event_scope": "Events from first held-out evaluation invocation; aggregate metrics cover all held-out evaluations.",
              "regret_definition": "Difference from best observed retained candidate only; unavailable when correctness fails.",
              "search": reference["search"], "graph": reference["compiled_graph"], "source_graph": reference["source_graph"],
              "capabilities": {"status": "available", "nodes": reference["capabilities"], "provenance": "exact core capability decisions"},
              "graph_id": "graph-" + study.digest(reference["compiled_graph"])[:20],
              "tensors": study.tensor_records(reference["compiled_graph"], profiles["shapes"]),
              "frontend_passes": reference.get("frontend_passes", []),
              "structure_sha256": study.digest(identity),
              "profiles": profiles, "injected_profiles": injected, "plans": plans, "policies": policies,
              "provenance": {"hardware_claim": study.hardware_provenance(reference), "measured": "Host executor time; separate calibration and evaluation invocations", "predicted": "Measured command profile sums with empirical uncertainty; candidate-only selection"},
              "experiment": {**(metadata or {}), "seed": seed, "profile_error": profile_error,
                  "uncertainty_weight": uncertainty_weight, "calibration_runs": len(calibration_reports),
                  "evaluation_runs": len(evaluation_reports), "command_version": 3,
                  "selection_planning_ms": planning_ms, "config": reference["config"]},
              "raw_runs": {"calibration_sha256": [study.digest(r) for r in calibration_reports],
                           "evaluation_sha256": [study.digest(r) for r in evaluation_reports]}}
    return validate_bundle(bundle)


def validate_bundle(bundle):
    """Check candidate-only claims, statistics, cross references and policy replay."""
    try:
        if bundle["schema_version"] != 1 or bundle["artifact_type"] != ARTIFACT_TYPE or any(bundle[name] is not False for name in ("exhaustive", "oracle_available", "global_regret_available")):
            raise ValueError("E_BOUNDED_SCHEMA: bounded artifact cannot assert global optimality")
        if type(bundle["conclusions_valid"]) is not bool:
            raise ValueError("E_BOUNDED_SCHEMA: correctness gate must be boolean")
        policies = bundle["policies"]
        if set(policies) != {"cpu", "maximal", "additive", "robust", "best_observed_candidate"}:
            raise ValueError("E_BOUNDED_SCHEMA: candidate policies must not contain a global oracle")
        plans = bundle["plans"]
        # Reuse the raw structural validator on stored command descriptors. These
        # summaries are not written out or represented as new raw measurements.
        report = {"schema_version": 3, "compiled_graph": bundle["graph"], "source_graph": bundle["source_graph"],
                  "capabilities": bundle["capabilities"]["nodes"], "config": bundle["experiment"]["config"], "search": bundle["search"],
                  "results": [{"placement": p["placement"], "commands": p["commands"], "events": p["events"],
                      "initial_buffers": p["initial_buffers"], "host_execution_ms": p["metrics"]["p50_ms"],
                      "correct": p["metrics"]["correct"], "execution_backend": p["execution_backend"],
                      "modeled_ms": p["illustrative_modeled_ms"]} for p in plans]}
        validate_report(report)
        if bundle["structure_sha256"] != study.digest(structure(report)):
            raise ValueError("E_BOUNDED_SCHEMA: candidate structure identity mismatch")
        if bundle["graph_id"] != "graph-" + study.digest(bundle["graph"])[:20] or bundle["tensors"] != study.tensor_records(bundle["graph"], bundle["profiles"]["shapes"]):
            raise ValueError("E_BOUNDED_SCHEMA: graph/tensor identity mismatch")
        by_placement = {p["placement"]: p for p in plans}
        best = min(by_placement, key=lambda p: (by_placement[p]["metrics"]["p50_ms"], p))
        valid = bundle["conclusions_valid"]
        for plan in plans:
            if plan["id"] != "candidate-" + (plan["placement"] or "empty"):
                raise ValueError("E_BOUNDED_SCHEMA: invalid candidate identity")
            if [c["stable_id"] for c in plan["commands"]] != [c["stable_id"] for c in study.stable_commands(plan["commands"])]:
                raise ValueError("E_BOUNDED_SCHEMA: invalid stable command identity")
            if plan["partitions"] != study.partition_records(bundle["graph"], plan["placement"]) or plan["buffers"] != study.buffer_records({"compiled_graph": bundle["graph"]}, plan):
                raise ValueError("E_BOUNDED_SCHEMA: partition/buffer references mismatch")
            partition_by_node = {node: part["id"] for part in plan["partitions"] for node in part["node_ids"]}
            for command in plan["commands"]:
                if command["partition_id"] != partition_by_node.get(command["node"]):
                    raise ValueError("E_BOUNDED_SCHEMA: command partition mismatch")
            for event in plan["events"]:
                command = plan["commands"][event["command_id"]]
                if event["command_stable_id"] != command["stable_id"] or event["id"] != "event-" + command["stable_id"] or event["partition_id"] != command["partition_id"]:
                    raise ValueError("E_BOUNDED_SCHEMA: event identity mismatch")
            metric = plan["metrics"]
            expected = study.distribution(metric["samples_ms"])
            if any(metric[name] != value for name, value in expected.items()):
                raise ValueError("E_BOUNDED_SCHEMA: candidate sample statistic mismatch")
            if metric["count"] != bundle["experiment"]["evaluation_runs"] or (valid and not metric["correct"]):
                raise ValueError("E_BOUNDED_SCHEMA: invalid observation count/correctness gate")
            regret = max(0., metric["p50_ms"] - by_placement[best]["metrics"]["p50_ms"]) if valid else None
            if metric["regret_vs_best_observed_ms"] != regret or "regret_ms" in metric or "regret_ratio" in metric:
                raise ValueError("E_BOUNDED_SCHEMA: invalid candidate-only regret")
            for name in ("h2d_bytes", "d2h_bytes", "peak_allocated_device_bytes", "command_count"):
                integer(metric[name], name)
            if metric["command_count"] != len(plan["commands"]):
                raise ValueError("E_BOUNDED_SCHEMA: command count mismatch")
        for key, record in bundle["profiles"]["commands"].items():
            study.validate_parameters(json.loads(key), 3)
            if any(record[name] != value for name, value in study.distribution(record["samples_ms"]).items()):
                raise ValueError("E_BOUNDED_SCHEMA: profile sample statistic mismatch")
        experiment = bundle["experiment"]
        predictions, selections, reason, _, injected = study.predict(report, bundle["profiles"], experiment["profile_error"], experiment["seed"], experiment["uncertainty_weight"])
        if injected != bundle["injected_profiles"] or any(plan["prediction"] != predictions[plan["placement"]] for plan in plans):
            raise ValueError("E_BOUNDED_SCHEMA: candidate predictions differ from saved profiles")
        selections["best_observed_candidate"] = best
        for name, placement in selections.items():
            policy = policies[name]
            expected_id = "candidate-" + (policy["placement"] or "empty") if policy["placement"] is not None else None
            if policy["plan_id"] != expected_id:
                raise ValueError("E_BOUNDED_SCHEMA: policy plan identity mismatch")
            if not valid:
                if policy["status"] != "suppressed_correctness_failure" or policy["placement"] is not None or policy["metrics"] is not None:
                    raise ValueError("E_BOUNDED_SCHEMA: failed correctness must suppress candidate conclusions")
            elif placement is None:
                if policy["status"] != "infeasible" or policy["placement"] is not None or policy["metrics"] is not None or policy["requested_placement"] != report["search"]["maximal_placement"]:
                    raise ValueError("E_BOUNDED_SCHEMA: strict maximal cannot be substituted")
            elif policy["status"] != "available" or policy["placement"] != placement or policy["metrics"] != by_placement[placement]["metrics"]:
                raise ValueError("E_BOUNDED_SCHEMA: candidate policy mismatch")
        if policies["robust"]["reason"] != (reason if valid else "SUPPRESSED_CORRECTNESS_FAILURE"):
            raise ValueError("E_BOUNDED_SCHEMA: robust selection reason mismatch")
        for phase in ("calibration", "evaluation"):
            if len(bundle["raw_runs"][phase + "_sha256"]) != experiment[phase + "_runs"]:
                raise ValueError("E_BOUNDED_SCHEMA: raw observation count mismatch")
        json.dumps(bundle, allow_nan=False)
        return bundle
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"E_BOUNDED_SCHEMA: malformed candidate bundle ({error})") from error


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare_bundles(left, right):
    """Compare retained-candidate evidence without implying shared/global optima."""
    validate_bundle(left)
    validate_bundle(right)
    changes = []
    def difference(kind, before, after):
        for key in sorted(before.keys() | after.keys()):
            if before.get(key) != after.get(key):
                changes.append({"entity_id": key, "kind": kind, "before": before.get(key),
                                "after": after.get(key), "reason_code": kind.upper() + "_CHANGED",
                                "provenance": "exact saved artifact"})
    old_nodes = {n["id"]: n for n in left["graph"]["nodes"]}
    new_nodes = {n["id"]: n for n in right["graph"]["nodes"]}
    difference("node", old_nodes, new_nodes)
    def inputs(bundle):
        return {t["name"]: {"shape": t["shape"], "dtype": t["dtype"],
                 "values_sha256": study.digest(t.get("values"))} for t in bundle["graph"]["inputs"]}
    difference("input_tensor", inputs(left), inputs(right))
    difference("graph_output", {"output": left["graph"]["output"]}, {"output": right["graph"]["output"]})
    difference("config", left["experiment"]["config"], right["experiment"]["config"])
    difference("source_target", {k: left["experiment"].get(k) for k in ("model_sha256", "workbench_target_profile")},
               {k: right["experiment"].get(k) for k in ("model_sha256", "workbench_target_profile")})
    difference("search", {k: left["search"][k] for k in ("beam_width", "max_evaluations", "evaluated", "rejected", "budget_exhausted")},
               {k: right["search"][k] for k in ("beam_width", "max_evaluations", "evaluated", "rejected", "budget_exhausted")})
    valid = left["conclusions_valid"] and right["conclusions_valid"]
    policies = {}
    for name in ("cpu", "maximal", "additive", "robust", "best_observed_candidate"):
        old, new = left["policies"][name], right["policies"][name]
        row = {"before_placement": old["placement"], "after_placement": new["placement"],
               "before_status": old["status"], "after_status": new["status"], "changed_node_ids": [],
               "added_commands": [], "removed_commands": [], "changed_commands": [], "metric_deltas": None,
               "reason_codes": ["RETAINED_CANDIDATES_ONLY", "TIMING_DELTA_OBSERVED_NOT_CAUSAL"],
               "provenance": {"commands": "exact", "latency": "held-out host observations, potentially different candidate sets"}}
        if old["plan_id"] is not None and new["plan_id"] is not None:
            before = next(p for p in left["plans"] if p["id"] == old["plan_id"])
            after = next(p for p in right["plans"] if p["id"] == new["plan_id"])
            old_lanes = dict(zip(old_nodes, before["placement"]))
            new_lanes = dict(zip(new_nodes, after["placement"]))
            row["changed_node_ids"] = [key for key in new_lanes if key in old_lanes and new_lanes[key] != old_lanes[key]]
            old_commands = {c["stable_id"]: c for c in before["commands"]}
            new_commands = {c["stable_id"]: c for c in after["commands"]}
            row["added_commands"] = [new_commands[k] for k in sorted(new_commands.keys() - old_commands.keys())]
            row["removed_commands"] = [old_commands[k] for k in sorted(old_commands.keys() - new_commands.keys())]
            row["changed_commands"] = [{"stable_id": k, "before": old_commands[k], "after": new_commands[k]}
                for k in sorted(old_commands.keys() & new_commands.keys())
                if {a: b for a, b in old_commands[k].items() if a not in ("id", "modeled_ms")} !=
                   {a: b for a, b in new_commands[k].items() if a not in ("id", "modeled_ms")}]
            if valid:
                row["metric_deltas"] = {k: after["metrics"][k] - before["metrics"][k]
                    for k in ("p50_ms", "p95_ms", "h2d_bytes", "d2h_bytes", "peak_allocated_device_bytes", "command_count")}
        else:
            row["reason_codes"].append("BASELINE_OR_CANDIDATE_UNAVAILABLE")
        policies[name] = row
    return {"schema_version": 1, "artifact_type": "bounded-placement-comparison", "exhaustive": False,
            "oracle_available": False, "conclusions_valid": valid, "changes": changes, "policies": policies,
            "comparison_scope": "Observed retained candidates in each study; candidate sets and workloads may differ. No global optimum, causal speedup or task-accuracy conclusion.",
            "candidate_sets_equal": {p["placement"] for p in left["plans"]} == {p["placement"] for p in right["plans"]},
            "source_hashes": [study.digest(left), study.digest(right)]}


def load_report(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("E_BOUNDED_LIMIT: raw report exceeds 64 MB")
    return validate_report(json.loads(raw))


def run_study(binary, graph, out, *, flags=(), search_budget=256, beam_width=16,
              warmup=1, calibration=3, samples=5, seed=20261004, profile_error=0.,
              uncertainty_weight=1., timeout=120):
    study.validate_options(flags)
    integer(search_budget, "search_budget", 2, 2048)
    integer(beam_width, "beam_width", 1, 32)
    integer(seed, "seed", 0)
    for name, value, minimum in (("warmup", warmup, 0), ("calibration", calibration, 2), ("samples", samples, 2)):
        integer(value, name, minimum, 30)
    study.finite(timeout, "timeout", 1)
    study.perturb_profiles({"commands": {}}, profile_error, seed)
    study.finite(uncertainty_weight, "uncertainty_weight")
    if uncertainty_weight > 1:
        raise ValueError("E_BOUNDED_OPTION: uncertainty_weight must be <=1")
    binary, graph, out = Path(binary).resolve(strict=True), Path(graph).resolve(strict=True), Path(out).resolve()
    with graph.open("rb") as stream:
        graph_bytes = stream.read(MAX_GRAPH_BYTES + 1)
    if len(graph_bytes) > MAX_GRAPH_BYTES:
        raise ValueError("E_BOUNDED_LIMIT: graph exceeds 16 MB")
    if out.exists() and any(out.iterdir()):
        raise ValueError("E_BOUNDED_OUTPUT: output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    copied = out / "input.plg"
    copied.write_bytes(graph_bytes)
    metadata = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "python": sys.version, "binary_sha256": study.file_hash(binary), "graph_sha256": study.file_hash(copied),
                "flags": list(flags), "search_budget": search_budget, "beam_width": beam_width,
                "warmup_runs": warmup, "timeout_seconds": timeout,
                "measurement_order": "Seeded shuffled retained candidates; deterministic candidate generation",
                "warmup_scope": "Fresh process each run; only OS/code caches may persist"}
    save(out / "configuration.json", metadata)
    collected = {"calibration": [], "evaluation": []}
    invocations = []
    warmup_identity = None
    for phase, count in (("warmup", warmup), ("calibration", calibration), ("evaluation", samples)):
        for index in range(count):
            stem = f"{phase}-{index:03d}"
            order_seed = (seed + len(invocations) * 104729) % 2**32
            command = [str(binary), "verify", "--graph", str(copied), *map(str, flags), "--search", "bounded",
                       "--search-budget", str(search_budget), "--beam-width", str(beam_width),
                       "--order-seed", str(order_seed), "--json", str(out / (stem + ".json"))]
            started = time.perf_counter_ns()
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                save(out / "failure.json", {"status": "timeout", "phase": phase, "index": index,
                     "timeout_seconds": timeout, "command": command, "conclusions_valid": False})
                raise RuntimeError(f"E_BOUNDED_RUN: {stem} timed out; see failure.json") from error
            (out / (stem + ".log")).write_text(process.stdout + process.stderr)
            invocations.append({"phase": phase, "index": index, "command": command, "returncode": process.returncode,
                "order_seed": order_seed, "report": stem + ".json", "log": stem + ".log",
                "subprocess_wall_ms": (time.perf_counter_ns() - started) / 1e6})
            save(out / "invocations.json", invocations)
            if process.returncode not in (0, 2) or not (out / (stem + ".json")).is_file():
                raise RuntimeError(f"E_BOUNDED_RUN: {stem} failed; see {stem}.log: " + process.stderr[:500])
            report = load_report(out / (stem + ".json"))
            if process.returncode == 2 and all(r["correct"] for r in report["results"]):
                raise RuntimeError("E_BOUNDED_RUN: nonzero exit contradicts correctness report")
            if report["search"]["max_evaluations"] != search_budget or report["search"]["beam_width"] != beam_width:
                raise ValueError("E_BOUNDED_RUN: core search settings differ from requested bounds")
            current = structure(report)
            if warmup_identity is None:
                warmup_identity = current
            if current != warmup_identity:
                raise ValueError("E_BOUNDED_DRIFT: warmup/calibration/evaluation candidate structure changed")
            if phase == "warmup":
                if not all(r["correct"] for r in report["results"]):
                    raise ValueError("E_BOUNDED_RUN: warmup correctness failed; no accepted study")
            else:
                collected[phase].append(report)
    bundle = build_bundle(collected["calibration"], collected["evaluation"], seed=seed,
                          profile_error=profile_error, uncertainty_weight=uncertainty_weight, metadata=metadata)
    bundle["raw_runs"]["invocations"] = invocations
    save(out / "bundle.json", bundle)
    return bundle


def replay(directory, output):
    directory = Path(directory)
    output = Path(output)
    if output.exists():
        raise ValueError("E_BOUNDED_REPLAY: output already exists; preserve previous evidence")
    original = validate_bundle(json.loads((directory / "bundle.json").read_text()))
    if original.get("artifact_type") != ARTIFACT_TYPE or original.get("schema_version") != 1:
        raise ValueError("E_BOUNDED_REPLAY: expected bounded study schema1")
    collected = {}
    for phase, count_name in (("calibration", "calibration_runs"), ("evaluation", "evaluation_runs")):
        collected[phase] = [load_report(directory / f"{phase}-{index:03d}.json") for index in range(original["experiment"][count_name])]
        if [study.digest(r) for r in collected[phase]] != original["raw_runs"][phase + "_sha256"]:
            raise ValueError("E_BOUNDED_REPLAY: raw report hash mismatch")
    experiment = original["experiment"]
    rebuilt = build_bundle(collected["calibration"], collected["evaluation"], seed=experiment["seed"],
                           profile_error=experiment["profile_error"], uncertainty_weight=experiment["uncertainty_weight"], metadata=experiment)
    def projection(bundle):
        result = deepcopy(bundle)
        result["experiment"].pop("selection_planning_ms", None)
        result["raw_runs"].pop("invocations", None)
        return result
    matches = projection(rebuilt) == projection(original)
    save(output, {"schema_version": 1, "artifact_type": "bounded-study-replay", "analysis_matches": matches,
                  "exhaustive": False, "oracle_available": False})
    if not matches:
        raise ValueError("E_BOUNDED_REPLAY: analysis mismatch")
    return rebuilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    for name in ("binary", "graph", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    for name, default in (("search-budget", 256), ("beam-width", 16), ("warmup", 1), ("calibration", 3), ("samples", 5), ("seed", 20261004)):
        run.add_argument("--" + name, type=int, default=default)
    for name, default in (("profile-error", 0.), ("uncertainty-weight", 1.)):
        run.add_argument("--" + name, type=float, default=default)
    run.add_argument("core_flags", nargs=argparse.REMAINDER)
    repeat = sub.add_parser("replay")
    repeat.add_argument("directory", type=Path)
    repeat.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "run":
            flags = args.core_flags[1:] if args.core_flags[:1] == ["--"] else args.core_flags
            result = run_study(args.binary, args.graph, args.out, flags=flags, search_budget=args.search_budget,
                beam_width=args.beam_width, warmup=args.warmup, calibration=args.calibration, samples=args.samples,
                seed=args.seed, profile_error=args.profile_error, uncertainty_weight=args.uncertainty_weight)
            print(json.dumps({"artifact": str(args.out / "bundle.json"), "candidates": len(result["plans"]),
                              "conclusions_valid": result["conclusions_valid"], "oracle_available": False}))
            return 0 if result["conclusions_valid"] else 2
        replay(args.directory, args.out)
        print("Bounded candidate study replay matched; no global oracle asserted.")
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
