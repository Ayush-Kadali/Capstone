"""Measure the allocation/synchronization tradeoff of exact-size buffer reuse.

The output is local software evidence, not an accelerator speedup claim.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
import sys

from study import distribution, file_hash


def investigate(binary, out, *, width=64, samples=21, warmup=3, seed=20261004):
    if width not in (16, 32, 64, 128) or not 5 <= samples <= 101 or not 0 <= warmup <= 10:
        raise ValueError("bounded width, 5..101 samples, and 0..10 warmups required")
    binary, out = Path(binary).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    values = [rng.randint(-8, 8) / 16 for _ in range(width * width)]
    graph = out / "allocation-chain.plg"
    lines = [f"tensor X f32 {width} {width} " + " ".join(map(str, values))]
    previous = "X"
    for i in range(7):
        lines.append(f"add add{i} {previous} X T{i}")
        previous = f"T{i}"
    lines.append(f"output {previous}")
    graph.write_text("\n".join(lines) + "\n")
    expected = [8 * value for value in values]
    runs = []
    for iteration in range(warmup + samples):
        order = ["retain", "reuse"]
        rng.shuffle(order)
        for variant in order:
            report = out / f"{iteration:03d}-{variant}.json"
            argv = [str(binary), "run", "--graph", str(graph), "--policy", "maximal", "--json", str(report)]
            if variant == "reuse":
                argv.append("--reuse-buffers")
            result = subprocess.run(argv, text=True, capture_output=True, timeout=90)
            (out / f"{iteration:03d}-{variant}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"execution failed: {argv}: {result.stderr}")
            raw = json.loads(report.read_text())
            row = raw["results"][0]
            if not row["correct"] or row["output"] != expected or row["placement"] != "A" * 7:
                raise RuntimeError("independent exact chain-output check failed")
            event_map = {event["command_id"]: event for event in row["events"]}
            allocation_us = sum(event_map[c["id"]]["duration_us"] for c in row["commands"] if c["kind"] == "ALLOC")
            sync_us = sum(event_map[c["id"]]["duration_us"] for c in row["commands"] if c["kind"] == "SYNC")
            runs.append({"iteration": iteration, "warmup": iteration < warmup, "variant": variant,
                         "argv": argv, "report": report.name, "report_sha256": file_hash(report),
                         "executor_ms": row["host_execution_ms"], "end_to_end_ms": row["end_to_end_ms"],
                         "allocation_ms": allocation_us / 1000, "synchronization_ms": sync_us / 1000,
                         "reserved_device_bytes": row["peak_allocated_device_bytes"],
                         "actual_software_reuses": row["software_buffer_reuses"],
                         "synchronizations": row["synchronizations"],
                         "transfer_bytes": row["h2d_bytes"] + row["d2h_bytes"], "correct": True})
    summary = {}
    for variant in ("retain", "reuse"):
        rows = [r for r in runs if r["variant"] == variant and not r["warmup"]]
        summary[variant] = {key: distribution([r[key] for r in rows]) for key in
                            ("executor_ms", "end_to_end_ms", "allocation_ms", "synchronization_ms")}
        for key in ("reserved_device_bytes", "actual_software_reuses", "synchronizations", "transfer_bytes"):
            if len({r[key] for r in rows}) != 1:
                raise RuntimeError(f"structural drift in {key}")
            summary[variant][key] = rows[0][key]
    pairs = []
    for iteration in range(warmup, warmup + samples):
        pair = {r["variant"]: r for r in runs if r["iteration"] == iteration}
        pairs.append(pair["reuse"]["executor_ms"] - pair["retain"]["executor_ms"])
    root = Path(__file__).resolve().parents[1]
    sources = [root / "CMakeLists.txt", Path(__file__), root / "scripts/study.py"] + list((root / "src").glob("*.cpp")) + list((root / "include/placementlens").glob("*.hpp"))
    evidence = {"schema_version": 1, "artifact_type": "memory-performance-investigation",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "environment": platform.platform(),
        "python": sys.version, "binary_sha256": file_hash(binary), "graph_sha256": file_hash(graph),
        "source_sha256": {str(p.relative_to(root)): file_hash(p) for p in sources},
        "hypothesis": "Exact-size reuse reduces device allocation storage; allocation savings may be offset by lifetime synchronization and bookkeeping.",
        "change": "Enable --reuse-buffers; keep graph, placement, numerical kernels and configured transfer costs fixed.",
        "method": {"seed": seed, "width": width, "warmup_pairs": warmup, "measurement_pairs": samples,
            "order": "seeded shuffled order within each pair; one fresh CLI process per sample",
            "executor_interval": "command interpretation and output copy, excluding process startup, planning, initial input preparation, reference and teardown",
            "end_to_end_interval": "runtime runner including preparation, reference and teardown, excluding CLI planning/process startup",
            "clock": "C++ steady_clock; host software backend"},
        "results": summary, "paired_reuse_minus_retain_ms": pairs,
        "claims": {"correctness": "independent exact output and original-graph reference passed for every sample",
            "memory_reduction_bytes": summary["retain"]["reserved_device_bytes"] - summary["reuse"]["reserved_device_bytes"],
            "executor_median_difference_ms": summary["reuse"]["executor_ms"]["p50_ms"] - summary["retain"]["executor_ms"]["p50_ms"],
            "timing_causality": "Observed difference of the whole reuse mechanism; individual noise and bookkeeping costs are not independently isolated."},
        "limitations": ["Host software execution only; no BeagleY-AI or GPU measurements.",
            "Logical device pool bytes exclude host retained tensors, allocator metadata, OS memory and scratch storage.",
            "Synchronization validates sequential software state; its measured cost is not real device synchronization latency.",
            "Small timings, process scheduling, cache state and clock noise limit speed claims; no confidence interval is asserted."],
        "runs": runs}
    (out / "investigation.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"results": summary, "claims": evidence["claims"]}, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/placementlens"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    investigate(args.binary, args.out, width=args.width, samples=args.samples, warmup=args.warmup, seed=args.seed)
