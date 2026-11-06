#!/usr/bin/env python3
"""Measured bounded-placement studies; Python stdlib, authoritative C++ plans.

Calibration and evaluation are separate invocations. Nothing in this module
uses illustrative modeled_ms as a measured timing or a profile fallback.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time


SCHEMA_VERSION = 2
MAX_PLANS = 4096
PARAMETER_COUNTS = {"Conv2D": 16, "MaxPool2D": 13, "AveragePool2D": 13,
                    "GlobalAveragePool": 4, "Reshape": 2,
                    "MatMul": 0, "ReLU": 0, "Add": 0, "MatMulReLU": 0, "RowSum": 0}


def validate_parameters(descriptor, version, *, node=False):
    """ABI3 carries explicit normalized attributes; old reports cannot carry CNN semantics."""
    parameters = descriptor.get("parameters", [] if version < 3 else None)
    if not isinstance(parameters, list) or any(type(value) is not int or not 0 <= value <= 65536 for value in parameters):
        raise ValueError("E_STUDY_SCHEMA: parameters must be an integer array in [0,65536]")
    compute = node or descriptor.get("kind") == "COMPUTE"
    op = descriptor.get("op")
    if version < 3:
        if parameters or (compute and PARAMETER_COUNTS.get(op, 0)):
            raise ValueError("E_STUDY_SCHEMA: parameterized operators require command version 3")
    elif compute:
        if op not in PARAMETER_COUNTS or len(parameters) != PARAMETER_COUNTS[op]:
            raise ValueError("E_STUDY_SCHEMA: invalid operator parameter count")
    elif parameters:
        raise ValueError("E_STUDY_SCHEMA: noncompute parameters must be empty")
    return parameters


def parameter_identity(descriptor):
    # Preserve historical v1/v2 profile keys and stable identifiers exactly.
    return {"parameters": descriptor["parameters"]} if descriptor.get("version") == 3 else {}



def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"E_STUDY_SCHEMA: {name} must be finite and >= {minimum}")
    return value


def percentile(values, fraction):
    """Linear interpolation between ordered samples (not a confidence bound)."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("E_STUDY_SAMPLE: nonempty values and fraction in [0,1] required")
    ordered = sorted(finite(x, "sample") for x in values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def distribution(values):
    return {"count": len(values), "p50_ms": percentile(values, .5),
            "p95_ms": percentile(values, .95), "min_ms": min(values), "max_ms": max(values)}


def plan_structure(result):
    fields = ("id", "version", "kind", "backend", "node", "op", "inputs", "output", "shape", "origins", "buffer_slot")
    return {"initial_buffers": [{**{k: c.get(k) for k in fields}, **parameter_identity(c)} for c in result.get("initial_buffers", [])],
            "commands": [{**{k: c.get(k) for k in fields}, **parameter_identity(c)} for c in result["commands"]]}


def validate_report(report):
    """Reject incomplete enumerations, missing event measurements and drift."""
    try:
        if type(report["schema_version"]) is not int or report["schema_version"] not in (1, 2, 3):
            raise ValueError("E_STUDY_SCHEMA: unsupported core report schema")
        nodes = report["compiled_graph"]["nodes"]
        results = report["results"]
        if not isinstance(nodes, list) or len(nodes) > 12 or not isinstance(results, list) or not 1 <= len(results) <= MAX_PLANS:
            raise ValueError("E_STUDY_LIMIT: bounded graph/plan limit exceeded")
        for graph in (report["compiled_graph"], report["source_graph"]):
            for node in graph["nodes"]:
                validate_parameters(node, report["schema_version"], node=True)
        node_parameters = {node["id"]: node.get("parameters", []) for node in nodes}
        placements = [r["placement"] for r in results]
        expected = [r["placement"] for r in report["search"]["ranked_feasible"]]
        if len(set(placements)) != len(placements) or set(placements) != set(expected):
            raise ValueError("E_STUDY_SCHEMA: every feasible placement must be measured exactly once")
        rejected = report["search"]["infeasible"]
        if type(rejected) is not int or rejected < 0 or rejected + len(results) != 2 ** len(nodes):
            raise ValueError("E_STUDY_SCHEMA: incomplete feasible/infeasible enumeration")
        for result in results:
            if len(result["placement"]) != len(nodes) or set(result["placement"]) - {"A", "C"}:
                raise ValueError("E_STUDY_SCHEMA: invalid placement")
            finite(result["host_execution_ms"], "host_execution_ms")
            if type(result["correct"]) is not bool:
                raise ValueError("E_STUDY_SCHEMA: correctness must be boolean")
            commands, events = result["commands"], result["events"]
            if any(type(c.get("version")) is not int or c["version"] != report["schema_version"]
                   for c in commands + result.get("initial_buffers", [])):
                raise ValueError("E_STUDY_SCHEMA: unsupported or incompatible command version")
            for command in commands + result.get("initial_buffers", []):
                parameters = validate_parameters(command, report["schema_version"])
                if report["schema_version"] == 3 and command["kind"] == "COMPUTE" and parameters != node_parameters.get(command.get("node")):
                    raise ValueError("E_STUDY_SCHEMA: compute parameters differ from graph")
            ids = [c["id"] for c in commands]
            if ids != list(range(len(commands))):
                raise ValueError("E_STUDY_SCHEMA: command IDs must be contiguous")
            event_ids = [e["command_id"] for e in events]
            if len(event_ids) != len(set(event_ids)) or set(event_ids) != set(ids):
                raise ValueError("E_STUDY_SCHEMA: one measured execution event per command required")
            for event in events:
                finite(event["start_us"], "event start")
                finite(event["duration_us"], "event duration")
                command = commands[event["command_id"]]
                if (event.get("name"), event.get("lane"), event.get("node")) != (command["kind"], command["backend"], command.get("node")):
                    raise ValueError("E_STUDY_SCHEMA: event metadata contradicts referenced command")
            for command in commands:
                if not isinstance(command["kind"], str) or not isinstance(command["backend"], str):
                    raise ValueError("E_STUDY_SCHEMA: command kind/backend missing")
                shape = command["shape"]
                if len(shape) != 2 or any(type(x) is not int or x < 0 for x in shape):
                    raise ValueError("E_STUDY_SCHEMA: rank-2 nonnegative shapes required")
        if "C" * len(nodes) not in placements:
            raise ValueError("E_STUDY_SCHEMA: CPU baseline missing")
        return report
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"E_STUDY_SCHEMA: malformed core report ({exc})") from exc


def command_key(command, shape_map, execution_backend, device_island_start=False):
    """Profiles distinguish op, backend, all operand/output shapes and direction."""
    return json.dumps({**parameter_identity(command), "kind": command["kind"], "op": command.get("op"),
                       "backend": command["backend"], "execution_backend": execution_backend,
                       "device_island_start": device_island_start,
                       "dtype": "f32", "layout": "contiguous-rank2",
                       "input_shapes": [shape_map.get(x[2:], []) for x in command.get("inputs", [])],
                       "output_shape": command["shape"],
                       "direction": [x[:2] for x in command.get("inputs", [])] + [command.get("output", "")[:2]]},
                      sort_keys=True, separators=(",", ":"))


def command_profile_keys(result, shape_map, execution_backend):
    """Keep launch-bearing compute samples distinct from continuing kernels.

    The executor charges launch on a CPU-to-device COMPUTE transition. ALLOC,
    COPY, FREE and SYNC do not reset that transition, even in buffer-reuse mode.
    """
    previous_compute = "CPU"
    keys = {}
    for command in result["commands"]:
        start = (command["kind"] == "COMPUTE" and command["backend"] == "virtual-device" and previous_compute != "virtual-device")
        keys[command["id"]] = command_key(command, shape_map, execution_backend, start)
        if command["kind"] == "COMPUTE":
            previous_compute = command["backend"]
    return keys


def shapes_for(report):
    shapes = {t["name"]: t["shape"] for t in report["compiled_graph"]["inputs"]}
    for result in report["results"]:
        for command in result["commands"]:
            if command.get("output") and all(command["shape"]):
                shapes[command["output"][2:]] = command["shape"]
    return shapes


def calibrate(reports, validator=validate_report):
    """Persist command samples and observed consecutive compute-region spans.

    Region spans are descriptive calibration evidence. Placement policies use
    command profiles so each transfer/residency decision remains inspectable.
    """
    if not reports:
        raise ValueError("E_STUDY_SAMPLE: calibration requires observations")
    observed, regions, overhead = defaultdict(list), defaultdict(list), defaultdict(list)
    shape_map = shapes_for(reports[0])
    for report in reports:
        validator(report)
        for result in report["results"]:
            execution_backend = result.get("execution_backend", report.get("execution_backend", "unavailable"))
            profile_keys = command_profile_keys(result, shape_map, execution_backend)
            events = {e["command_id"]: e for e in result["events"]}
            for command in result["commands"]:
                observed[profile_keys[command["id"]]].append(events[command["id"]]["duration_us"] / 1000)
            # Runtime scaffolding outside timed command bodies remains separate.
            body = sum(e["duration_us"] for e in events.values()) / 1000
            overhead[len(result["commands"])].append(max(0, result["host_execution_ms"] - body))
            current = []
            def flush():
                if current:
                    first, last = events[current[0]["id"]], events[current[-1]["id"]]
                    key = json.dumps([json.loads(profile_keys[c["id"]]) for c in current], sort_keys=True)
                    regions[key].append(max(0, last["start_us"] + last["duration_us"] - first["start_us"]) / 1000)
                    current.clear()
            for command in result["commands"]:
                if command["kind"] == "COMPUTE":
                    if current and command["backend"] != current[-1]["backend"]:
                        flush()
                    current.append(command)
                elif command["kind"] not in ("ALLOC", "FREE"):
                    flush()
            flush()
    def records(samples):
        return {str(key): {"samples_ms": values, **distribution(values)} for key, values in sorted(samples.items())}
    return {"commands": records(observed), "subgraphs": records(regions),
            "runtime_overhead_by_command_count": records(overhead), "shapes": shape_map,
            "provenance": "measured host execution events from calibration only; subgraphs are observed region spans"}


def perturb_profiles(profiles, error, seed):
    finite(error, "profile_error")
    if error >= 1:
        raise ValueError("E_STUDY_OPTION: profile_error must be < 1")
    if type(seed) is not int:
        raise ValueError("E_STUDY_OPTION: seed must be an integer")
    rng = random.Random(seed)
    result = {}
    for key, record in sorted(profiles["commands"].items()):
        factor = rng.uniform(1 - error, 1 + error)
        center = record["p50_ms"] * factor
        # Envelope combines empirical sample spread and the known injected
        # multiplicative error bound. It is not a statistical confidence interval.
        result[key] = {"estimate_ms": center, "lower_ms": record["min_ms"] * factor / (1 + error),
                       "upper_ms": record["max_ms"] * factor / (1 - error), "factor": factor}
    return result


def predict(report, profiles, error=0.0, seed=20261004, uncertainty_weight=1.0):
    started = time.perf_counter_ns()
    finite(uncertainty_weight, "uncertainty_weight")
    if uncertainty_weight > 1:
        raise ValueError("E_STUDY_OPTION: uncertainty_weight must be <= 1")
    perturbed = perturb_profiles(profiles, error, seed)
    by_kind = defaultdict(list)
    for key, record in perturbed.items():
        by_kind[json.loads(key)["kind"]].append(record["estimate_ms"])
    uniform_copy = percentile(by_kind.get("COPY", [0]), .5)
    uniform_sync = percentile(by_kind.get("SYNC", [0]), .5)
    predictions = {}
    for result in report["results"]:
        execution_backend = result.get("execution_backend", report.get("execution_backend", "unavailable"))
        profile_keys = command_profile_keys(result, profiles["shapes"], execution_backend)
        total = lower = upper = compute = 0.0
        details = []
        for command in result["commands"]:
            key = profile_keys[command["id"]]
            if key not in perturbed:
                raise ValueError("E_STUDY_PROFILE: no calibrated command profile; no modeled fallback")
            cost = perturbed[key]
            total += cost["estimate_ms"]
            lower += cost["lower_ms"]
            upper += cost["upper_ms"]
            if command["kind"] == "COMPUTE":
                compute += cost["estimate_ms"]
            details.append({"command_id": command["id"], "profile_key": key, **cost})
        count = len(result["commands"])
        residual = profiles["runtime_overhead_by_command_count"].get(str(count))
        if residual is None:
            raise ValueError("E_STUDY_PROFILE: missing calibrated runtime overhead")
        total += residual["p50_ms"]
        lower += residual["min_ms"]
        upper += residual["max_ms"]
        kinds = Counter(c["kind"] for c in result["commands"])
        predictions[result["placement"]] = {
            "additive_ms": compute + kinds["COPY"] * uniform_copy + kinds["SYNC"] * uniform_sync,
            "boundary_ms": total, "lower_ms": lower, "upper_ms": upper,
            "robust_score_ms": total + uncertainty_weight * (upper - total),
            "runtime_overhead_ms": residual["p50_ms"], "command_estimates": details,
            "reason_codes": ["PROFILE_CALIBRATED", "RESIDENCY_COMMANDS_EXACT", "SHAPE_DEPENDENT_TRANSFER",
                             "SYNCHRONIZATION_INCLUDED", "MEMORY_FEASIBILITY_CORE", "EMPIRICAL_ERROR_ENVELOPE"]}
    cpu = "C" * len(report["compiled_graph"]["nodes"])
    additive = min(predictions, key=lambda p: (predictions[p]["additive_ms"], p))
    robust = min(predictions, key=lambda p: (predictions[p]["robust_score_ms"], p))
    reason = "MINIMUM_UNCERTAINTY_SCORE"
    # Demand conservative evidence before offloading. At weight=0 this reduces
    # to the point-estimate boundary-aware policy, an explicit ablation.
    cpu_threshold = predictions[cpu]["boundary_ms"] - uncertainty_weight * (
        predictions[cpu]["boundary_ms"] - predictions[cpu]["lower_ms"])
    if robust != cpu and predictions[robust]["robust_score_ms"] >= cpu_threshold:
        robust, reason = cpu, "CPU_GUARD_UNCERTAIN_ADVANTAGE"
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    maximal = strict_maximal_placement(report)
    return predictions, {"cpu": cpu, "maximal": maximal if maximal in predictions else None,
                         "additive": additive, "robust": robust}, reason, elapsed_ms, perturbed


def strict_maximal_placement(report):
    if "capabilities" not in report:
        return None
    supported = {c["node_id"]: c["device_supported"] for c in report["capabilities"]}
    if set(supported) != {n["id"] for n in report["compiled_graph"]["nodes"]}:
        raise ValueError("E_STUDY_SCHEMA: missing or invalid capability node IDs")
    return "".join("A" if supported[n["id"]] else "C" for n in report["compiled_graph"]["nodes"])


def stable_commands(commands):
    counts = Counter()
    rows = []
    for command in commands:
        signature = {k: command.get(k) for k in ("kind", "backend", "node", "op", "inputs", "output", "origins")}
        signature.update(parameter_identity(command))
        key = digest(signature)[:20]
        occurrence = counts[key]
        counts[key] += 1
        rows.append({**command, "stable_id": f"cmd-{key}-{occurrence}"})
    return rows


def buffer_records(report, result):
    """Logical buffer lifetimes derived from authoritative handle operations.

    A missing free or physical slot is unavailable, never guessed from liveness.
    Inputs exist before command zero; last use and actual allocation end differ.
    """
    records, active = [], {}
    stable_ids = {c["id"]: c["stable_id"] for c in stable_commands(result["commands"])}
    for tensor in report["compiled_graph"]["inputs"]:
        handle = "H:" + tensor["name"]
        row = {"id": "buffer-" + digest([handle, None])[:20], "tensor": tensor["name"], "handle": handle,
               "backend": "CPU", "shape": tensor["shape"], "bytes": math.prod(tensor["shape"]) * 4,
               "allocation_id": None, "first_use_id": None, "last_use_id": None, "free_id": None, "slot": None,
               "preexisting_input": True, "provenance": "exact input storage and derived command-use interval"}
        records.append(row)
        active[handle] = row
    for descriptor in result.get("initial_buffers", []):
        handle = descriptor["output"]
        row = {"id": "buffer-initial-" + digest(handle)[:20], "tensor": handle[2:], "handle": handle,
               "backend": descriptor["backend"], "shape": descriptor["shape"], "bytes": math.prod(descriptor["shape"]) * 4,
               "allocation_id": None, "initial_buffer_id": descriptor["id"], "first_use_id": None, "last_use_id": None,
               "free_id": None, "slot": descriptor.get("buffer_slot"), "preexisting_input": True,
               "provenance": "exact mirrored input preparation before executor interval; upload cost reported separately"}
        records.append(row)
        active[handle] = row
    for command in result["commands"]:
        handle = command.get("output", "")
        if command["kind"] == "ALLOC":
            row = {"id": "buffer-" + digest([handle, stable_ids[command["id"]]])[:20], "tensor": handle[2:], "handle": handle,
                   "backend": command["backend"], "shape": command["shape"], "bytes": math.prod(command["shape"]) * 4,
                   "allocation_id": command["id"], "first_use_id": None, "last_use_id": None, "free_id": None,
                   "slot": command.get("buffer_slot"), "preexisting_input": False,
                   "provenance": "exact allocation and derived command-use interval; slot unavailable unless core emits it"}
            records.append(row)
            active[handle] = row
        elif command["kind"] == "FREE":
            freed = [handle] if handle else command.get("inputs", [])
            for freed_handle in freed:
                if freed_handle in active:
                    active.pop(freed_handle)["free_id"] = command["id"]
        else:
            for used in command.get("inputs", []) + ([handle] if handle else []):
                if used in active:
                    if active[used]["first_use_id"] is None:
                        active[used]["first_use_id"] = command["id"]
                    active[used]["last_use_id"] = command["id"]
    return records


def tensor_records(graph, shapes):
    inputs = {t["name"] for t in graph["inputs"]}
    producers = {n["output"]: n["id"] for n in graph["nodes"]}
    return [{"id": name, "shape": shapes[name], "dtype": "f32", "layout": "contiguous-rank2",
             "bytes": math.prod(shapes[name]) * 4, "producer_node": producers.get(name),
             "consumer_nodes": [n["id"] for n in graph["nodes"] if name in n["inputs"]],
             "is_input": name in inputs, "is_output": name == graph["output"], "provenance": "exact-structural"}
            for name in sorted(inputs | producers.keys())]


def partition_records(graph, placement):
    """Contiguous backend segments of the authoritative topological schedule.

    These are schedule partitions, not a claim of fused vendor subgraphs or
    graph-theoretic maximal connected components in a branched DAG.
    """
    partitions = []
    for node, lane in zip(graph["nodes"], placement):
        backend = "CPU" if lane == "C" else "virtual-device"
        if not partitions or partitions[-1]["backend"] != backend:
            partitions.append({"backend": backend, "node_ids": []})
        partitions[-1]["node_ids"].append(node["id"])
    for partition in partitions:
        partition["id"] = "partition-" + digest(partition)[:20]
        partition["semantics"] = "contiguous backend segment of topological schedule"
        partition["provenance"] = "derived from exact placement"
    return partitions


def hardware_provenance(report):
    """Describe the observed backend without inferring target-board coverage."""
    backends = {r.get("execution_backend", report.get("execution_backend", "unavailable"))
                for r in report["results"]}
    if any(isinstance(name, str) and name.startswith("OpenCL GPU:") for name in backends):
        device_events = any(e.get("device_us") is not None
                            for r in report["results"] for e in r["events"])
        evidence = "device-event timings are present" if device_events else "device-event timings are unavailable"
        return ("local OpenCL backend execution was recorded; " + evidence +
                "; host executor intervals include dispatch/waits/diagnostic work; "
                "not BeagleY-AI validation or evidence of optimized/general accelerator speedup")
    return "none; host software results do not validate physical accelerator performance"


def build_bundle(calibration_reports, evaluation_reports, *, seed=20261004, profile_error=0.0,
                 uncertainty_weight=1.0, metadata=None):
    if not calibration_reports or not evaluation_reports:
        raise ValueError("E_STUDY_SAMPLE: separate calibration and evaluation observations required")
    all_reports = calibration_reports + evaluation_reports
    for report in all_reports:
        validate_report(report)
    reference = all_reports[0]
    structure = {r["placement"]: plan_structure(r) for r in reference["results"]}
    backend_by_plan = {r["placement"]: r.get("execution_backend", reference.get("execution_backend")) for r in reference["results"]}
    for report in all_reports[1:]:
        if (report["schema_version"] != reference["schema_version"] or report["compiled_graph"] != reference["compiled_graph"] or report["config"] != reference["config"] or
                report.get("capabilities") != reference.get("capabilities") or
                {r["placement"]: r.get("execution_backend", report.get("execution_backend")) for r in report["results"]} != backend_by_plan or
                {r["placement"]: plan_structure(r) for r in report["results"]} != structure):
            raise ValueError("E_STUDY_DRIFT: graph, configuration or feasible command plans changed within study")
    profiles = calibrate(calibration_reports)
    predictions, selected, robust_reason, planning_ms, perturbed = predict(
        reference, profiles, profile_error, seed, uncertainty_weight)
    measurements = defaultdict(list)
    for report in evaluation_reports:
        for result in report["results"]:
            measurements[result["placement"]].append(result["host_execution_ms"])
    correct = all(r["correct"] for report in all_reports for r in report["results"])
    stats = {p: distribution(samples) for p, samples in measurements.items()}
    oracle = min(stats, key=lambda p: (stats[p]["p50_ms"], p)) if correct else None
    cpu_time = stats[selected["cpu"]]["p50_ms"]
    oracle_time = stats[oracle]["p50_ms"] if oracle is not None else None
    plans = []
    example_results = {r["placement"]: r for r in evaluation_reports[0]["results"]}
    for placement in sorted(measurements):
        result = example_results[placement]
        median = stats[placement]["p50_ms"]
        repetitions = [r for report in evaluation_reports for r in report["results"] if r["placement"] == placement]
        optional_timings = {}
        for field in ("end_to_end_ms", "queue_wait_ms", "input_preparation_ms"):
            values = [r.get(field) for r in repetitions]
            optional_timings[field.removesuffix("_ms")] = distribution(values) if all(v is not None for v in values) else None
        cpu_compute = []
        for repetition in repetitions:
            cpu_ids = {c["id"] for c in repetition["commands"] if c["kind"] == "COMPUTE" and c["backend"] == "CPU"}
            cpu_compute.append(sum(e["duration_us"] for e in repetition["events"] if e["command_id"] in cpu_ids) / 1000)
        metric = {**stats[placement], "samples_ms": measurements[placement],
                  "regret_ms": max(0, median - oracle_time) if correct else None,
                  "regret_ratio": median / oracle_time - 1 if correct and oracle_time > 0 else None,
                  "harmful_offload": placement != selected["cpu"] and median > cpu_time if correct else None,
                  "h2d_bytes": result["h2d_bytes"], "d2h_bytes": result["d2h_bytes"],
                  "peak_allocated_device_bytes": result["peak_allocated_device_bytes"],
                  "command_count": len(result["commands"]), "synchronizations": result["synchronizations"],
                  "islands": result["islands"], "cpu_compute": distribution(cpu_compute), **optional_timings,
                  "correct": all(r["correct"] for report in all_reports
                    for r in report["results"] if r["placement"] == placement)}
        for field in ("diagnostic_readback_bytes", "software_buffer_reuses", "runtime_peak_device_reserved_bytes",
                      "tile_count", "tile_load_bytes", "peak_local_bytes", "device_buffer_slots", "buffer_reuses", "peak_live_device_bytes", "preparation_h2d_bytes", "injected_boundary_ms"):
            metric[field] = result.get(field)
        partitions = partition_records(reference["compiled_graph"], placement)
        node_partition = {node: p["id"] for p in partitions for node in p["node_ids"]}
        commands = [{**c, "partition_id": node_partition.get(c.get("node"))} for c in stable_commands(result["commands"])]
        command_map = {c["id"]: c for c in commands}
        events = [{**e, "id": "event-" + command_map[e["command_id"]]["stable_id"],
                   "command_stable_id": command_map[e["command_id"]]["stable_id"],
                   "partition_id": command_map[e["command_id"]]["partition_id"]} for e in result["events"]]
        plans.append({"id": f"plan-{placement or 'empty'}", "placement": placement, "metrics": metric,
                      "prediction": predictions[placement], "commands": commands, "partitions": partitions,
                      "initial_buffers": result.get("initial_buffers", []),
                      "buffers": buffer_records(reference, result),
                      "events": events, "passes": result.get("passes", []),
                      "reason_codes": predictions[placement]["reason_codes"],
                      "illustrative_modeled_ms": result.get("modeled_ms"),
                      "execution_backend": result.get("execution_backend", reference.get("execution_backend"))})
    indexed = {p["placement"]: p for p in plans}
    policies = {}
    for policy, placement in selected.items():
        if placement is None:
            requested = strict_maximal_placement(reference)
            policies[policy] = {"placement": None, "plan_id": None, "metrics": None,
                "requested_placement": requested, "status": "infeasible" if requested is not None else "unavailable",
                "display_name": "Maximal capability", "reason_code": "CORE_PLAN_INFEASIBLE" if requested is not None else "CAPABILITY_DECISIONS_UNAVAILABLE"}
            continue
        policies[policy] = {"placement": placement, "plan_id": indexed[placement]["id"],
                            "status": "available", "requested_placement": placement,
                            "display_name": {"cpu": "CPU only", "maximal": "Maximal capability",
                                             "additive": "Profile additive", "robust": "Boundary and uncertainty aware"}[policy],
                            "metrics": deepcopy(indexed[placement]["metrics"]),
                            "reason_code": robust_reason if policy == "robust" else {
                                "cpu": "CPU_REFERENCE", "maximal": "ALL_CAPABILITY_ACCEPTED_NODES",
                                "additive": "MINIMUM_COMPUTE_PLUS_UNIFORM_BOUNDARIES"}[policy]}
    policies["oracle"] = {"placement": oracle, "plan_id": indexed[oracle]["id"] if correct else None,
                           "reason_code": "MINIMUM_EVALUATION_P50" if correct else "SUPPRESSED_CORRECTNESS_FAILURE"}
    feasible_maximal = min(indexed, key=lambda p: (-p.count("A"), p))
    policies["maximal_feasible"] = {"display_name": "Optional maximum feasible node count",
        "status": "available", "placement": feasible_maximal, "plan_id": indexed[feasible_maximal]["id"],
        "metrics": deepcopy(indexed[feasible_maximal]["metrics"]), "reason_code": "MAXIMUM_SUPPORTED_NODE_COUNT_FEASIBLE"}
    stability = {name: [] for name in selected}
    for offset in range(8):
        _, choices, _, _, _ = predict(reference, profiles, profile_error, seed + offset, uncertainty_weight)
        for name, placement in choices.items():
            stability[name].append(placement)
    policy_stability = {name: {"seeds": list(range(seed, seed + 8)), "placements": values,
        "modal_fraction": max(Counter(values).values()) / len(values), "unique_plans": len(set(values)),
        "available_selections": sum(p is not None for p in values),
        "harmful_offload_rate": sum(indexed[p]["metrics"]["harmful_offload"] for p in values) / len(values)
            if correct and all(p is not None for p in values) else None}
        for name, values in stability.items()}
    bundle = {"schema_version": SCHEMA_VERSION, "artifact_type": "placement-study", "conclusions_valid": correct,
              "correctness": {"all_calibration_and_evaluation_correct": correct,
                              "reason_code": "REFERENCE_MATCH" if correct else "SUPPRESSED_CORRECTNESS_FAILURE"},
              "provenance": {"structural": "exact C++ graph/command plan", "measured": "host wall time; calibration/evaluation separated",
                 "controlled_emulation": "when config.emulate_boundaries is true, host timings include requested cancellable software waits; no physical transfer latency claim",
                 "predicted": "calibrated profile sums with seeded multiplicative error; empirical envelopes, not confidence intervals",
                 "modeled": "illustrative core costs retained separately; never substituted for measurements",
                 "hardware_claim": hardware_provenance(reference)},
              "experiment": {"seed": seed, "profile_error": profile_error, "uncertainty_weight": uncertainty_weight,
                 "core_report_schema_version": reference["schema_version"], "command_version": reference["schema_version"],
                 "calibration_runs": len(calibration_reports), "evaluation_runs": len(evaluation_reports),
                 "config": reference["config"], "selection_planning_ms": planning_ms,
                 "planning_scope": "Python profile aggregation/scoring/selection; excludes core enumeration and subprocess startup",
                 "oracle_definition": "minimum evaluation p50 of executor interval (commands/output collection); excludes validation/reference/teardown; same feasible set, noise not removed",
                 "maximal_definition": "all capability-accepted nodes assigned to device; explicit infeasible if complete placement exceeds constraints",
                 "config_sha256": digest(reference["config"]), **(metadata or {})},
              "graph": reference["compiled_graph"], "source_graph": reference["source_graph"],
              "graph_id": "graph-" + digest(reference["compiled_graph"])[:20],
              "tensors": tensor_records(reference["compiled_graph"], profiles["shapes"]),
              "capabilities": {"status": "available", "nodes": reference["capabilities"], "provenance": "exact core capability decisions"}
                  if "capabilities" in reference else {"status": "unavailable", "nodes": [], "provenance": "core did not export capability decisions; not inferred"},
              "frontend_passes": reference.get("frontend_passes", []), "profiles": profiles,
              "injected_profiles": perturbed, "plans": plans, "policies": policies, "stability": policy_stability,
              "rejections": {"count": reference["search"]["infeasible"], "reason_code": "CORE_CAPABILITY_OR_MEMORY_CONSTRAINT",
                             "details": reference["search"].get("rejected", [])},
              "raw_runs": {"calibration_sha256": [digest(r) for r in calibration_reports],
                           "evaluation_sha256": [digest(r) for r in evaluation_reports]}}
    bundle["structure_sha256"] = digest({"graph": bundle["graph"], "plans": structure})
    return validate_bundle(bundle)


def validate_bundle(bundle):
    try:
        if bundle["schema_version"] != 2 or bundle["artifact_type"] != "placement-study":
            raise ValueError("E_STUDY_SCHEMA: expected placement-study schema 2")
        plans = bundle["plans"]
        if not isinstance(plans, list) or not 1 <= len(plans) <= MAX_PLANS:
            raise ValueError("E_STUDY_SCHEMA: invalid plan count")
        ids = [p["id"] for p in plans]
        if len(set(ids)) != len(ids):
            raise ValueError("E_STUDY_SCHEMA: duplicate plan IDs")
        if type(bundle["conclusions_valid"]) is not bool:
            raise ValueError("E_STUDY_SCHEMA: conclusion gate must be boolean")
        graph = bundle["graph"]
        version = bundle["experiment"]["command_version"]
        if type(version) is not int or version not in (1, 2, 3) or bundle["experiment"]["core_report_schema_version"] != version:
            raise ValueError("E_STUDY_SCHEMA: unsupported command/core schema version")
        if not isinstance(graph["nodes"], list) or len(graph["nodes"]) > 12:
            raise ValueError("E_STUDY_LIMIT: bounded graph limit exceeded")
        for candidate_graph in (graph, bundle["source_graph"]):
            for node in candidate_graph["nodes"]:
                validate_parameters(node, version, node=True)
        if version == 3:
            for profile_key in bundle["profiles"]["commands"]:
                validate_parameters(json.loads(profile_key), version)
            for region_key in bundle["profiles"]["subgraphs"]:
                region = json.loads(region_key)
                if not isinstance(region, list):
                    raise ValueError("E_STUDY_SCHEMA: invalid subgraph profile key")
                for descriptor in region:
                    validate_parameters(descriptor, version)
        node_ids = [n["id"] for n in graph["nodes"]]
        if len(node_ids) != len(set(node_ids)) or any(not isinstance(n, str) or not n for n in node_ids):
            raise ValueError("E_STUDY_SCHEMA: invalid or duplicate node ID")
        tensors = {}
        for tensor in graph["inputs"]:
            name = tensor["name"]
            shape = tensor["shape"]
            if not isinstance(name, str) or not name or name in tensors or tensor["dtype"] != "f32":
                raise ValueError("E_STUDY_SCHEMA: invalid input tensor")
            if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
                raise ValueError("E_STUDY_SCHEMA: invalid input shape")
            tensors[name] = shape
        for node in graph["nodes"]:
            if any(name not in tensors for name in node["inputs"]) or node["output"] in tensors:
                raise ValueError("E_STUDY_SCHEMA: undefined operand, duplicate output or non-topological graph")
            tensors[node["output"]] = bundle["profiles"]["shapes"][node["output"]]
            if len(tensors[node["output"]]) != 2 or any(type(x) is not int or x <= 0 for x in tensors[node["output"]]):
                raise ValueError("E_STUDY_SCHEMA: invalid node output shape")
        if graph["output"] not in tensors:
            raise ValueError("E_STUDY_SCHEMA: undefined graph output")
        if bundle["tensors"] != tensor_records(graph, tensors):
            raise ValueError("E_STUDY_SCHEMA: invalid tensor records or references")
        placements = set()
        for plan in plans:
            placement = plan["placement"]
            if not isinstance(placement, str) or len(placement) != len(node_ids) or set(placement) - {"C", "A"} or placement in placements:
                raise ValueError("E_STUDY_SCHEMA: invalid or duplicate placement")
            placements.add(placement)
            metric = plan["metrics"]
            samples = metric["samples_ms"]
            if not isinstance(samples, list) or not samples or type(metric["count"]) is not int or metric["count"] != len(samples):
                raise ValueError("E_STUDY_SCHEMA: sample count mismatch")
            expected = distribution(samples)
            for key in ("p50_ms", "p95_ms", "min_ms", "max_ms"):
                if not math.isclose(finite(metric[key], key), expected[key], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("E_STUDY_SCHEMA: sample statistic mismatch")
            for key in ("h2d_bytes", "d2h_bytes", "peak_allocated_device_bytes", "command_count", "synchronizations", "islands"):
                if type(metric[key]) is not int or metric[key] < 0:
                    raise ValueError("E_STUDY_SCHEMA: integer metric required")
            if type(metric["correct"]) is not bool or (bundle["conclusions_valid"] and not metric["correct"]):
                raise ValueError("E_STUDY_SCHEMA: inconsistent correctness gate")
            commands = plan["commands"]
            if metric["command_count"] != len(commands) or any(type(c["id"]) is not int for c in commands) or [c["id"] for c in commands] != list(range(len(commands))):
                raise ValueError("E_STUDY_SCHEMA: invalid command IDs/count")
            if len({c["stable_id"] for c in plan["commands"]}) != len(plan["commands"]):
                raise ValueError("E_STUDY_SCHEMA: duplicate stable command IDs")
            if [c["stable_id"] for c in commands] != [c["stable_id"] for c in stable_commands(commands)]:
                raise ValueError("E_STUDY_SCHEMA: invalid stable command identity")
            if [c["node"] for c in commands if c["kind"] == "COMPUTE"] != node_ids:
                raise ValueError("E_STUDY_SCHEMA: compute schedule does not match graph")
            if plan["partitions"] != partition_records(graph, placement):
                raise ValueError("E_STUDY_SCHEMA: invalid partition membership")
            node_partition = {node: p["id"] for p in plan["partitions"] for node in p["node_ids"]}
            defined_handles = {"H:" + t["name"] for t in graph["inputs"]}
            initial = plan.get("initial_buffers", [])
            if [c["id"] for c in initial] != list(range(len(initial))):
                raise ValueError("E_STUDY_SCHEMA: invalid initial-buffer IDs")
            for descriptor in initial:
                validate_parameters(descriptor, version)
                handle = descriptor["output"]
                if type(descriptor["version"]) is not int or descriptor["version"] != version or descriptor["kind"] != "ALLOC" or descriptor["backend"] != "virtual-device" or handle[:2] != "D:" or handle[2:] not in {t["name"] for t in graph["inputs"]} or descriptor["shape"] != tensors[handle[2:]] or handle in defined_handles:
                    raise ValueError("E_STUDY_SCHEMA: invalid initial device buffer")
                defined_handles.add(handle)
            for command in commands:
                if type(command["version"]) is not int or command["version"] != version:
                    raise ValueError("E_STUDY_SCHEMA: unsupported command version")
                validate_parameters(command, version)
                shape = command["shape"]
                if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x < 0 for x in shape):
                    raise ValueError("E_STUDY_SCHEMA: invalid command shape")
                if command.get("node") and command["node"] not in node_ids:
                    raise ValueError("E_STUDY_SCHEMA: command refers to unknown node")
                if command["partition_id"] != node_partition.get(command.get("node")):
                    raise ValueError("E_STUDY_SCHEMA: command refers to wrong partition")
                if command["kind"] == "COMPUTE":
                    index = node_ids.index(command["node"])
                    node = graph["nodes"][index]
                    if command.get("parameters", []) != node.get("parameters", []):
                        raise ValueError("E_STUDY_SCHEMA: compute parameters differ from graph")
                    if command["backend"] != ("CPU" if placement[index] == "C" else "virtual-device") or command["op"] != node["op"] or [x[2:] for x in command["inputs"]] != node["inputs"] or command["output"][2:] != node["output"]:
                        raise ValueError("E_STUDY_SCHEMA: compute descriptor differs from graph/placement")
                if any(handle not in defined_handles for handle in command["inputs"]):
                    raise ValueError("E_STUDY_SCHEMA: command refers to undefined buffer")
                output = command.get("output", "")
                if output:
                    if output[:2] not in ("H:", "D:") or output[2:] not in tensors:
                        raise ValueError("E_STUDY_SCHEMA: command refers to unknown tensor")
                    if command["kind"] == "ALLOC":
                        defined_handles.add(output)
                    elif output not in defined_handles:
                        raise ValueError("E_STUDY_SCHEMA: command writes unallocated buffer")
                if command["kind"] == "FREE":
                    for handle in ([output] if output else command["inputs"]):
                        defined_handles.discard(handle)
            events = plan["events"]
            if any(type(e["command_id"]) is not int for e in events) or sorted(e["command_id"] for e in events) != list(range(len(commands))):
                raise ValueError("E_STUDY_SCHEMA: event references invalid command")
            for event in events:
                finite(event["start_us"], "event start")
                finite(event["duration_us"], "event duration")
                if event.get("device_us") is not None:
                    finite(event["device_us"], "device event duration")
                if event.get("node") and event["node"] not in node_ids:
                    raise ValueError("E_STUDY_SCHEMA: event references unknown node")
                command = commands[event["command_id"]]
                if (event.get("name"), event.get("lane"), event.get("node")) != (command["kind"], command["backend"], command.get("node")):
                    raise ValueError("E_STUDY_SCHEMA: event metadata contradicts referenced command")
                if event["command_stable_id"] != command["stable_id"] or event["id"] != "event-" + command["stable_id"] or event["partition_id"] != command["partition_id"]:
                    raise ValueError("E_STUDY_SCHEMA: inconsistent stable event reference")
            for buffer in plan["buffers"]:
                if buffer["tensor"] not in tensors or buffer["shape"] != tensors[buffer["tensor"]] or buffer["bytes"] != math.prod(buffer["shape"]) * 4:
                    raise ValueError("E_STUDY_SCHEMA: invalid buffer tensor/shape/size")
                for key in ("allocation_id", "first_use_id", "last_use_id", "free_id"):
                    if buffer[key] is not None and (type(buffer[key]) is not int or not 0 <= buffer[key] < len(commands)):
                        raise ValueError("E_STUDY_SCHEMA: buffer references invalid command")
            expected_buffers = buffer_records({"compiled_graph": graph}, {"commands": commands, "initial_buffers": initial})
            if plan["buffers"] != expected_buffers:
                raise ValueError("E_STUDY_SCHEMA: buffer lifetimes differ from authoritative commands")
            if version == 3:
                expected_profiles = command_profile_keys(plan, bundle["profiles"]["shapes"], plan.get("execution_backend", "unavailable"))
                estimates = plan["prediction"]["command_estimates"]
                if len(estimates) != len(commands) or any(detail["command_id"] != index or detail["profile_key"] != expected_profiles[index]
                        or detail["profile_key"] not in bundle["profiles"]["commands"]
                        or detail["profile_key"] not in bundle["injected_profiles"]
                        for index, detail in enumerate(estimates)):
                    raise ValueError("E_STUDY_SCHEMA: command parameter/profile identity mismatch")
            for key in ("additive_ms", "boundary_ms", "lower_ms", "upper_ms", "robust_score_ms", "runtime_overhead_ms"):
                finite(plan["prediction"][key], "prediction " + key)
            if plan["prediction"]["lower_ms"] > plan["prediction"]["upper_ms"]:
                raise ValueError("E_STUDY_SCHEMA: inverted prediction envelope")
            if not bundle["conclusions_valid"] and any(plan["metrics"][k] is not None for k in ("regret_ms", "regret_ratio", "harmful_offload")):
                raise ValueError("E_STUDY_SCHEMA: correctness failure must suppress conclusions")
        plans_by_id = {p["id"]: p for p in plans}
        for policy in ("cpu", "maximal", "additive", "robust", "oracle"):
            selection = bundle["policies"][policy]
            plan_id = selection["plan_id"]
            unavailable_maximal = (policy == "maximal" and plan_id is None and selection.get("status") in ("infeasible", "unavailable"))
            if plan_id not in ids and not unavailable_maximal and not (policy == "oracle" and plan_id is None and not bundle["conclusions_valid"]):
                raise ValueError("E_STUDY_SCHEMA: policy refers to unknown plan")
            if plan_id is not None:
                if selection["placement"] != plans_by_id[plan_id]["placement"]:
                    raise ValueError("E_STUDY_SCHEMA: policy placement/plan mismatch")
                if "metrics" in selection and selection["metrics"] != plans_by_id[plan_id]["metrics"]:
                    raise ValueError("E_STUDY_SCHEMA: policy metrics/plan mismatch")
        capabilities = bundle["capabilities"]
        if capabilities["status"] not in ("available", "unavailable"):
            raise ValueError("E_STUDY_SCHEMA: invalid capability status")
        if capabilities["status"] == "available":
            caps = capabilities["nodes"]
            if len(caps) != len(node_ids) or {c["node_id"] for c in caps} != set(node_ids):
                raise ValueError("E_STUDY_SCHEMA: invalid capability node references")
            if any(type(c["device_supported"]) is not bool or not isinstance(c["reason"], str) for c in caps):
                raise ValueError("E_STUDY_SCHEMA: invalid capability decision")
        maximal = bundle["policies"]["maximal"]
        if capabilities["status"] == "available":
            by_node = {c["node_id"]: c["device_supported"] for c in capabilities["nodes"]}
            requested = "".join("A" if by_node[n] else "C" for n in node_ids)
            if requested in placements:
                if maximal["status"] != "available" or maximal["placement"] != requested:
                    raise ValueError("E_STUDY_SCHEMA: maximal capability baseline substituted")
            elif maximal["status"] != "infeasible" or maximal.get("requested_placement") != requested or maximal["placement"] is not None or maximal["metrics"] is not None:
                raise ValueError("E_STUDY_SCHEMA: invalid strict maximal infeasibility")
        elif maximal["status"] != "unavailable" or maximal["placement"] is not None or maximal["metrics"] is not None:
            raise ValueError("E_STUDY_SCHEMA: maximal baseline cannot infer missing capability decisions")
        # Reject NaN/Infinity anywhere, including derived/provenance objects.
        json.dumps(bundle, allow_nan=False)
        return bundle
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"E_STUDY_SCHEMA: malformed bundle ({exc})") from exc


def compare_bundles(left, right):
    """Exact structural differences; timing deltas are observations, not causes."""
    validate_bundle(left)
    validate_bundle(right)
    changes = []
    a = {n["id"]: n for n in left["graph"]["nodes"]}
    b = {n["id"]: n for n in right["graph"]["nodes"]}
    for node in sorted(a.keys() | b.keys()):
        if a.get(node) != b.get(node):
            changes.append({"entity_id": node, "kind": "node", "before": a.get(node), "after": b.get(node),
                            "reason_code": "GRAPH_STRUCTURE_CHANGED", "provenance": "exact"})
    for key in sorted(left["experiment"]["config"].keys() | right["experiment"]["config"].keys()):
        before, after = left["experiment"]["config"].get(key), right["experiment"]["config"].get(key)
        if before != after:
            changes.append({"entity_id": key, "kind": "config", "before": before, "after": after,
                            "reason_code": "CONFIGURATION_CHANGED", "provenance": "exact"})
    by_policy = {}
    for policy in ("cpu", "maximal", "additive", "robust"):
        if left["policies"][policy]["plan_id"] is None or right["policies"][policy]["plan_id"] is None:
            by_policy[policy] = {"before_placement": left["policies"][policy]["placement"],
                "after_placement": right["policies"][policy]["placement"],
                "before_status": left["policies"][policy].get("status", "available"),
                "after_status": right["policies"][policy].get("status", "available"),
                "changed_node_ids": [], "added_commands": [], "removed_commands": [], "changed_commands": [],
                "metric_deltas": None, "reason_codes": ["BASELINE_OR_CANDIDATE_UNAVAILABLE"],
                "provenance": {"availability": "exact core feasible set", "latency": "unavailable"}}
            continue
        l = next(p for p in left["plans"] if p["id"] == left["policies"][policy]["plan_id"])
        r = next(p for p in right["plans"] if p["id"] == right["policies"][policy]["plan_id"])
        lc, rc = {c["stable_id"]: c for c in l["commands"]}, {c["stable_id"]: c for c in r["commands"]}
        by_policy[policy] = {"before_placement": l["placement"], "after_placement": r["placement"],
            "changed_node_ids": [node["id"] for index, node in enumerate(right["graph"]["nodes"])
                                 if node["id"] in a and l["placement"][list(a).index(node["id"])] != r["placement"][index]],
            "added_commands": [rc[k] for k in sorted(rc.keys() - lc.keys())],
            "removed_commands": [lc[k] for k in sorted(lc.keys() - rc.keys())],
            "changed_commands": [{"stable_id": k, "before": lc[k], "after": rc[k]} for k in sorted(lc.keys() & rc.keys())
                                 if {x: v for x, v in lc[k].items() if x not in ("id", "modeled_ms")} !=
                                    {x: v for x, v in rc[k].items() if x not in ("id", "modeled_ms")}],
            "metric_deltas": {key: r["metrics"][key] - l["metrics"][key] for key in
                              ("p50_ms", "p95_ms", "h2d_bytes", "d2h_bytes", "peak_allocated_device_bytes", "command_count")},
            "reason_codes": ["PLACEMENT_CHANGED" if l["placement"] != r["placement"] else "PLACEMENT_STABLE",
                             "TIMING_DELTA_OBSERVED_NOT_CAUSAL"], "provenance": {"commands": "exact", "latency": "measured"}}
    return {"schema_version": 2, "artifact_type": "placement-comparison", "changes": changes, "policies": by_policy,
            "conclusions_valid": left["conclusions_valid"] and right["conclusions_valid"],
            "source_hashes": [digest(left), digest(right)]}


def analysis_projection(bundle):
    """Every deterministic analysis field, excluding measured planner duration
    and invocation/environment metadata that is not regenerated by replay.
    """
    fields = ("structure_sha256", "graph_id", "graph", "source_graph", "tensors", "capabilities", "frontend_passes",
              "profiles", "injected_profiles", "plans", "policies", "stability", "correctness", "conclusions_valid", "rejections", "provenance")
    projection = {key: bundle[key] for key in fields}
    projection["experiment"] = {key: bundle["experiment"][key] for key in
        ("seed", "profile_error", "uncertainty_weight", "calibration_runs", "evaluation_runs", "core_report_schema_version",
         "command_version", "config", "config_sha256", "oracle_definition", "maximal_definition", "planning_scope")}
    return projection


def validate_options(flags):
    switches = {"--cnn-supported", "--relu-supported", "--fuse", "--reuse-buffers", "--async", "--emulate-boundaries"}
    values = {"--backend", "--tile-size", "--local-bytes", "--device-bytes", "--boundary-ms", "--sync-ms", "--launch-ms", "--bandwidth", "--resident-input"}
    index = 0
    while index < len(flags):
        flag = flags[index]
        if flag in switches:
            index += 1
        elif flag in values and index + 1 < len(flags) and not str(flags[index + 1]).startswith("--"):
            index += 2
        else:
            raise ValueError(f"E_STUDY_OPTION: unsupported or incomplete core option {flag}")


def run_study(binary, graph, out, *, flags=(), warmup=1, calibration=3, samples=5,
              seed=20261004, profile_error=0.0, uncertainty_weight=1.0, timeout=120):
    """Run complete feasible sets; save raw reports/logs/config before summary.

    Each verify invocation creates fresh runtime state. Warmups warm OS/code
    caches only; they do not establish persistent device allocation/cache state.
    """
    validate_options(flags)
    for name, count, minimum in (("warmup", warmup, 0), ("calibration", calibration, 2), ("samples", samples, 2)):
        if type(count) is not int or not minimum <= count <= 30:
            raise ValueError(f"E_STUDY_OPTION: {name} must be between {minimum} and 30")
    finite(timeout, "timeout", 1)
    perturb_profiles({"commands": {}}, profile_error, seed)
    finite(uncertainty_weight, "uncertainty_weight")
    if uncertainty_weight > 1:
        raise ValueError("E_STUDY_OPTION: uncertainty_weight must be <= 1")
    binary, graph, out = Path(binary).resolve(), Path(graph).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("E_STUDY_OUTPUT: output directory must be empty (preserve previous evidence)")
    out.mkdir(parents=True, exist_ok=True)
    graph_bytes = graph.read_bytes()
    if len(graph_bytes) > 16000000:
        raise ValueError("E_STUDY_LIMIT: graph exceeds 16 MB")
    copied_graph = out / "input.plg"
    copied_graph.write_bytes(graph_bytes)
    metadata = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "machine": platform.machine(), "python": sys.version, "binary": str(binary),
                "binary_sha256": file_hash(binary), "graph_sha256": file_hash(copied_graph),
                "flags": list(flags), "warmup_runs": warmup, "timeout_seconds": timeout,
                "measurement_order": "seeded shuffled complete feasible set each invocation; recorded order seeds",
                "warmup_scope": "fresh process per run, only OS/code caches can persist"}
    (out / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    collected = {"calibration": [], "evaluation": []}
    logs = []
    for phase, count in (("warmup", warmup), ("calibration", calibration), ("evaluation", samples)):
        for index in range(count):
            stem = f"{phase}-{index:03d}"
            path = out / f"{stem}.json"
            order_seed = (seed + len(logs) * 104729) % (2 ** 32)
            command = [str(binary), "verify", "--graph", str(copied_graph), *map(str, flags),
                       "--order-seed", str(order_seed), "--json", str(path)]
            started = time.perf_counter_ns()
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            (out / f"{stem}.log").write_text(completed.stdout + completed.stderr)
            logs.append({"phase": phase, "index": index, "command": command, "returncode": completed.returncode,
                         "order_seed": order_seed, "subprocess_wall_ms": elapsed_ms, "report": path.name, "log": f"{stem}.log"})
            (out / "invocations.json").write_text(json.dumps(logs, indent=2) + "\n")
            if not path.is_file():
                diagnostic = completed.stderr.strip().splitlines()
                detail = ": " + diagnostic[0][:500] if diagnostic else ""
                raise RuntimeError(f"E_STUDY_RUN: {stem} produced no report; see {stem}.log{detail}")
            report = validate_report(json.loads(path.read_text()))
            # A numerical mismatch report is retained and conclusions suppressed.
            # Other nonzero exits must not masquerade as an accepted experiment.
            if completed.returncode and all(r["correct"] for r in report["results"]):
                raise RuntimeError(f"E_STUDY_RUN: {stem} failed ({completed.returncode}); see {stem}.log")
            if phase == "warmup" and not all(r["correct"] for r in report["results"]):
                raise RuntimeError(f"E_STUDY_RUN: warmup correctness failure; see {stem}.json")
            if phase in collected:
                collected[phase].append(report)
    bundle = build_bundle(collected["calibration"], collected["evaluation"], seed=seed,
                          profile_error=profile_error, uncertainty_weight=uncertainty_weight, metadata=metadata)
    bundle["raw_runs"]["invocations"] = logs
    (out / "profiles.json").write_text(json.dumps(bundle["profiles"], indent=2, allow_nan=False) + "\n")
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    return bundle


def run_sweep(binary, out, *, widths=(2, 8), topologies=("chain", "shared"), errors=(0., .3),
              warmup=1, calibration=2, samples=3, seed=20261004, extra_flags=()):
    """Bounded one-factor/configuration sweep with held-out timing per case.

    Profile-error levels reuse exactly the same raw observations so timing
    noise is not mistaken for the effect of profile corruption.
    """
    from check_matrix import workload
    widths, topologies, errors = tuple(widths), tuple(topologies), tuple(errors)
    if (not widths or any(type(w) is not int or not 1 <= w <= 64 for w in widths) or
            not topologies or set(topologies) - {"chain", "shared", "residual"} or not errors or
            len(widths) * len(topologies) * 10 > 40 or len(errors) > 8):
        raise ValueError("E_STUDY_OPTION: sweep requires widths 1..64, known topologies, <=40 measured cases, <=8 errors")
    for error in errors:
        perturb_profiles({"commands": {}}, error, seed)
    validate_options(extra_flags)
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("E_STUDY_OUTPUT: sweep output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for width, topology in itertools.product(widths, topologies):
        graph = out / f"{topology}-{width}.plg"
        graph.write_text(workload(topology, width, seed))
        # Explicit software waits create controlled host-timing experiments.
        # They do not measure physical buses; retain model-only isolation too.
        configurations = {
            "fragmented": [],
            "supported": ["--relu-supported"],
            "tight-memory": ["--relu-supported", "--device-bytes", str(3 * width * width * 4)],
            "modeled-boundary-high": ["--relu-supported", "--boundary-ms", "5"],
            "tight-local": ["--relu-supported", "--tile-size", "4", "--local-bytes", "192"],
            "mirrored-inputs": ["--relu-supported", "--resident-input", "X", "--resident-input", "W"],
            "measured-boundary-delays": ["--relu-supported", "--emulate-boundaries", "--boundary-ms", ".15", "--sync-ms", ".05"],
            "measured-bandwidth-low": ["--relu-supported", "--emulate-boundaries", "--boundary-ms", "0", "--sync-ms", "0", "--bandwidth", "1024"],
            "measured-sync-high": ["--relu-supported", "--emulate-boundaries", "--boundary-ms", "0", "--sync-ms", ".2"],
            "measured-launch-high": ["--relu-supported", "--emulate-boundaries", "--boundary-ms", "0", "--sync-ms", "0", "--launch-ms", ".2"],
        }
        for config_name, flags in configurations.items():
            name = f"{topology}-{width}-{config_name}"
            folder = out / name
            bundle = run_study(binary, graph, folder, flags=[*extra_flags, *flags], warmup=warmup,
                               calibration=calibration, samples=samples, seed=seed, profile_error=errors[0])
            reports = {phase: [json.loads(p.read_text()) for p in sorted(folder.glob(f"{phase}-*.json"))]
                       for phase in ("calibration", "evaluation")}
            for index, error in enumerate(errors):
                variant = bundle if index == 0 else build_bundle(reports["calibration"], reports["evaluation"],
                    seed=seed, profile_error=error, metadata={"parent_measurements": name,
                        "graph_sha256": bundle["experiment"]["graph_sha256"], "binary_sha256": bundle["experiment"]["binary_sha256"]})
                destination = folder / f"error-{index:02d}.json"
                destination.write_text(json.dumps(variant, indent=2, allow_nan=False) + "\n")
                rows.append({"case": name, "width": width, "topology": topology, "configuration": config_name,
                             "profile_error": error, "bundle": str(destination.relative_to(out)),
                             "conclusions_valid": variant["conclusions_valid"],
                             "policies": {k: {"placement": v["placement"], "status": v.get("status", "available"),
                                "regret_ms": (v.get("metrics") or {}).get("regret_ms"),
                                "p50_ms": (v.get("metrics") or {}).get("p50_ms"),
                                "harmful_offload": (v.get("metrics") or {}).get("harmful_offload")}
                                for k, v in variant["policies"].items()}})
            print(f"Measured {name}: {len(bundle['plans'])} feasible placements", flush=True)
    summary = {"schema_version": 2, "artifact_type": "placement-sweep", "seed": seed, "cases": rows,
               "conclusions_valid": all(r["conclusions_valid"] for r in rows),
               "provenance": "host software executor measurements; measured-* boundary cases inject real software waits, not physical hardware latency; modeled-boundary-high changes estimates only",
               "aggregates": {}}
    if summary["conclusions_valid"]:
        for policy in ("cpu", "maximal", "additive", "robust"):
            values = [r["policies"][policy] for r in rows if r["policies"][policy]["status"] == "available"]
            summary["aggregates"][policy] = {"available_case_count": len(values), "unavailable_case_count": len(rows) - len(values),
                "mean_regret_ms": sum(v["regret_ms"] for v in values) / len(values) if values else None,
                "harmful_offload_rate": sum(v["harmful_offload"] for v in values) / len(values) if values else None}
    (out / "sweep.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run")
    run.add_argument("--binary", type=Path, default=Path("build/placementlens"))
    run.add_argument("--graph", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--calibration", type=int, default=3)
    run.add_argument("--samples", type=int, default=5)
    run.add_argument("--seed", type=int, default=20261004)
    run.add_argument("--profile-error", type=float, default=0.0)
    run.add_argument("--uncertainty-weight", type=float, default=1.0)
    run.add_argument("core_flags", nargs=argparse.REMAINDER, help="core options after --")
    diff = actions.add_parser("diff")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--out", type=Path, required=True)
    replay = actions.add_parser("replay", help="recompute selection from saved raw reports; timing is not reexecuted")
    replay.add_argument("directory", type=Path)
    replay.add_argument("--out", type=Path, required=True)
    sweep = actions.add_parser("sweep", help="bounded graph/capability/memory/profile-error matrix")
    sweep.add_argument("--binary", type=Path, default=Path("build/placementlens"))
    sweep.add_argument("--out", type=Path, required=True)
    sweep.add_argument("--widths", type=int, nargs="+", default=[2, 8])
    sweep.add_argument("--topologies", nargs="+", default=["chain", "shared"])
    sweep.add_argument("--errors", type=float, nargs="+", default=[0., .3])
    sweep.add_argument("--warmup", type=int, default=1)
    sweep.add_argument("--calibration", type=int, default=2)
    sweep.add_argument("--samples", type=int, default=3)
    sweep.add_argument("--seed", type=int, default=20261004)
    sweep.add_argument("core_flags", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "run":
        flags = args.core_flags[1:] if args.core_flags[:1] == ["--"] else args.core_flags
        result = run_study(args.binary, args.graph, args.out, flags=flags, warmup=args.warmup,
                           calibration=args.calibration, samples=args.samples, seed=args.seed,
                           profile_error=args.profile_error, uncertainty_weight=args.uncertainty_weight)
        print(json.dumps({"bundle": str(args.out / "bundle.json"), "conclusions_valid": result["conclusions_valid"],
                          "placements": {k: v["placement"] for k, v in result["policies"].items()}}, indent=2))
    elif args.action == "sweep":
        flags = args.core_flags[1:] if args.core_flags[:1] == ["--"] else args.core_flags
        result = run_sweep(args.binary, args.out, widths=args.widths, topologies=args.topologies,
                           errors=args.errors, warmup=args.warmup, calibration=args.calibration,
                           samples=args.samples, seed=args.seed, extra_flags=flags)
        print(json.dumps({"summary": str(args.out / "sweep.json"), "cases": len(result["cases"])}, indent=2))
    elif args.action == "diff":
        result = compare_bundles(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
        if args.out.exists():
            raise ValueError("E_STUDY_OUTPUT: refusing to replace existing output")
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    else:
        original = validate_bundle(json.loads((args.directory / "bundle.json").read_text()))
        reports = {phase: [json.loads(p.read_text()) for p in sorted(args.directory.glob(f"{phase}-*.json"))]
                   for phase in ("calibration", "evaluation")}
        for phase in reports:
            if [digest(r) for r in reports[phase]] != original["raw_runs"][f"{phase}_sha256"]:
                raise ValueError("E_STUDY_REPLAY: raw report hash mismatch")
        config = original["experiment"]
        result = build_bundle(reports["calibration"], reports["evaluation"], seed=config["seed"],
                              profile_error=config["profile_error"], uncertainty_weight=config["uncertainty_weight"])
        check = {"schema_version": 2, "artifact_type": "study-replay", "structure_matches": result["structure_sha256"] == original["structure_sha256"],
                 "policies_match": result["policies"] == original["policies"],
                 "injected_profiles_match": result["injected_profiles"] == original["injected_profiles"],
                 "analysis_matches": analysis_projection(result) == analysis_projection(original),
                 "provenance": "deterministic analysis replay of hashed recordings; no new timing measurements"}
        if args.out.exists():
            raise ValueError("E_STUDY_OUTPUT: refusing to replace existing output")
        args.out.write_text(json.dumps(check, indent=2) + "\n")
        if not all(check[k] for k in ("structure_matches", "policies_match", "injected_profiles_match", "analysis_matches")):
            raise ValueError("E_STUDY_REPLAY: analysis mismatch")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
