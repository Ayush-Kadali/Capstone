#!/usr/bin/env python3
"""Compare all bounded capstone placements with independent PyTorch CPU ops.

This is numerical external validation with separately scoped host timings,
not a speedup benchmark or BeagleY-AI validation. No dependency is installed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


CONFIGS = {
    "fragmented": [],
    "pooled-tiled": ["--relu-supported", "--reuse-buffers", "--tile-size", "2"],
    "fused-prepared-async": ["--relu-supported", "--reuse-buffers", "--tile-size", "3",
                             "--fuse", "--resident-input", "X", "--async"],
}
FAMILIES = ("chain", "shared", "residual", "reduction")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def distribution(samples):
    if not samples or any(not math.isfinite(v) or v < 0 for v in samples):
        raise ValueError("invalid measured duration")
    return {"count": len(samples), "median_ms": statistics.median(samples),
            "minimum_ms": min(samples), "maximum_ms": max(samples), "samples_ms": samples}


def generate(family, width, seed):
    if family not in FAMILIES or not 1 <= width <= 16:
        raise ValueError("unsupported bounded graph family or width")
    rng = random.Random(seed)
    m, k, n = (width, width, width) if family == "residual" else (width, width + 1, width + 2)
    lines = [f"# production comparison family={family} width={width} seed={seed}"]
    for name, rows, cols in (("X", m, k), ("W", k, n), ("B", m, n)):
        lines.append(f"tensor {name} f32 {rows} {cols} " + " ".join(str(rng.randint(-8, 8)/16) for _ in range(rows*cols)))
    lines += ["matmul mm X W M", "relu activation M R"]
    if family == "chain":
        lines += ["add bias R B Y"]
    elif family == "shared":
        lines += ["add branch M R S", "add bias S B Y"]
    elif family == "residual":
        lines += ["add skip R X S", "matmul projection S W P", "relu activation2 P T", "add bias T B Y"]
    else:
        lines += ["rowsum reduce R Y"]
    return "\n".join(lines + ["output Y"]) + "\n"


def prepare_torch(torch, graph):
    """Consume exported original graph, never compiled/fused capstone plans."""
    values = {}
    for tensor in graph["inputs"]:
        if tensor["dtype"] != "f32" or tensor["layout"] != "contiguous-row-major" or len(tensor["shape"]) != 2:
            raise ValueError("production adapter requires contiguous rank-2 f32 inputs")
        values[tensor["name"]] = torch.tensor(tensor["values"], dtype=torch.float32, device="cpu").reshape(tensor["shape"])
    return values


def execute_torch(torch, graph, inputs):
    values = dict(inputs)
    for node in graph["nodes"]:
        operands = [values[name] for name in node["inputs"]]
        op = node["op"]
        if op == "MatMul":
            output = torch.mm(*operands)
        elif op == "ReLU":
            output = torch.relu(*operands)
        elif op == "Add":
            output = torch.add(*operands)
        elif op == "RowSum":
            output = torch.sum(operands[0], dim=1, keepdim=True, dtype=torch.float32)
        elif op == "MatMulReLU":
            output = torch.relu(torch.mm(*operands))
        else:
            raise ValueError(f"unsupported production adapter opcode: {op}")
        if list(output.shape) != node["shape"]:
            raise ValueError("production adapter output shape mismatch")
        values[node["output"]] = output
    return values[graph["output"]]


def compare_values(actual, expected, atol=1e-5, rtol=1e-5):
    if len(actual) != len(expected) or not actual:
        return {"correct": False, "max_abs_error": None, "reason": "output length mismatch"}
    if any(not math.isfinite(x) for x in actual + expected):
        return {"correct": False, "max_abs_error": None, "reason": "nonfinite output"}
    errors = [abs(a-b) for a, b in zip(actual, expected)]
    return {"correct": all(e <= atol + rtol*abs(b) for e, b in zip(errors, expected)),
            "max_abs_error": max(errors), "reason": "finite elementwise atol+rtol check"}


def validate_coverage(report):
    if report["schema_version"] not in (1, 2, 3):
        raise ValueError("unsupported capstone report schema")
    rows = report["results"]
    maps = [row["placement"] for row in rows]
    feasible = {row["placement"] for row in report["search"]["ranked_feasible"]}
    if not rows or len(maps) != len(set(maps)) or set(maps) != feasible:
        raise ValueError("missing or duplicate executed feasible placement")
    if len(rows) + report["search"]["infeasible"] != 2**len(report["compiled_graph"]["nodes"]):
        raise ValueError("incomplete placement enumeration")


def run_case(torch, binary, out, family, width, seed, config_name, repeats, warmups):
    name = f"{family}-{width}-{config_name}"
    case = out / name
    case.mkdir()
    graph_path = case / "graph.plg"
    graph_path.write_text(generate(family, width, seed))
    raw, batches = [], []
    original = None
    for trial in range(warmups + repeats):
        path = case / f"capstone-{trial:02d}.json"
        command = [str(binary), "verify", "--graph", str(graph_path), *CONFIGS[config_name],
                   "--order-seed", str(seed + trial), "--json", str(path)]
        start = time.perf_counter_ns()
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        wall_ms = (time.perf_counter_ns()-start)/1e6
        (case / f"capstone-{trial:02d}.txt").write_text(result.stdout + result.stderr)
        if result.returncode not in (0, 2):
            raise RuntimeError(f"capstone execution failed for {name}: {result.stderr}")
        report = json.loads(path.read_text())
        validate_coverage(report)
        if original is not None and original != report["source_graph"]:
            raise ValueError("original graph changed across repeats")
        original = report["source_graph"]
        raw.append(report)
        batches.append({"trial": trial, "warmup": trial < warmups, "command": command,
                        "returncode": result.returncode, "process_wall_ms": wall_ms, "report": path.name})
    inputs = prepare_torch(torch, original)
    torch_runs = []
    with torch.inference_mode():
        for trial in range(warmups + repeats):
            start = time.perf_counter_ns()
            output = execute_torch(torch, original, inputs)
            elapsed = (time.perf_counter_ns()-start)/1e6
            torch_runs.append({"trial": trial, "warmup": trial < warmups,
                               "execution_ms": elapsed, "output": output.flatten().tolist(),
                               "shape": list(output.shape)})
    expected = torch_runs[warmups]["output"]
    if not all(compare_values(row["output"], expected)["correct"] for row in torch_runs):
        raise ValueError("PyTorch output was nonfinite or changed across repeats")
    comparisons = []
    placements = sorted(row["placement"] for row in raw[0]["results"])
    for placement in placements:
        rows = [next(row for row in report["results"] if row["placement"] == placement) for report in raw]
        checks = [compare_values(row["output"], expected) for row in rows]
        comparisons.append({"placement": placement,
                            "correct": all(row["correct"] and item["correct"] for row, item in zip(rows, checks)),
                            "max_abs_error_vs_pytorch": max((item["max_abs_error"] for item in checks if item["max_abs_error"] is not None), default=None),
                            "checks_by_trial": checks,
                            "host_execution": distribution([row["host_execution_ms"] for row in rows[warmups:]]),
                            "end_to_end": distribution([row["end_to_end_ms"] for row in rows[warmups:]])})
    evidence = {"case": name, "family": family, "width": width, "seed": seed,
                "graph_sha256": sha(graph_path), "config": config_name, "flags": CONFIGS[config_name],
                "capstone_batches": batches, "pytorch_runs": torch_runs,
                "pytorch_execution": distribution([row["execution_ms"] for row in torch_runs[warmups:]]),
                "placements": comparisons, "correct": all(row["correct"] for row in comparisons),
                "performance_conclusion": None,
                "performance_reason": "timing scopes differ; no cross-runtime speedup inference"}
    save(case / "comparison.json", evidence)
    return evidence, original, inputs


def profile_graph(torch, graph, inputs, out):
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        with torch.profiler.record_function("capstone_original_graph"):
            execute_torch(torch, graph, inputs)
    profile.export_chrome_trace(str(out / "pytorch-cpu-trace.json"))
    rows = [{"operator": item.key, "calls": item.count,
             "self_cpu_time_total_us": item.self_cpu_time_total,
             "cpu_time_total_us": item.cpu_time_total, "input_shapes": item.input_shapes}
            for item in profile.key_averages(group_by_input_shape=True)]
    save(out / "pytorch-key-averages.json", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/placementlens"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 7])
    parser.add_argument("--families", choices=FAMILIES, nargs="+", default=list(FAMILIES))
    parser.add_argument("--configs", choices=list(CONFIGS), nargs="+", default=list(CONFIGS))
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 100 or not 0 <= args.warmups <= 10 or not 1 <= args.threads <= 64:
        parser.error("bounded repeats 1..100, warmups 0..10 and threads 1..64 required")
    if not 0 <= args.seed <= 2**32-256 or any(not 1 <= width <= 16 for width in args.widths):
        parser.error("seed or width outside bounded range")
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name == "torch":
            print("SKIP: PyTorch unavailable; no dependency was installed")
            return 77
        raise
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    binary = args.binary.resolve()
    before_hash = sha(binary)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("output directory must be empty to preserve existing evidence")
    root = Path(__file__).resolve().parents[1]
    source_hashes = {str(path.relative_to(root)): sha(path) for pattern in ("src/*.cpp", "include/placementlens/*.hpp") for path in root.glob(pattern)}
    source_hashes["scripts/compare_production.py"] = sha(__file__)
    torch_root = Path(torch.__file__).resolve().parent
    installed_sources = {name: sha(torch_root / name) for name in (
        "_ops.py", "include/ATen/ops/mm_ops.h", "include/ATen/core/dispatch/Dispatcher.h",
        "profiler/profiler.py") if (torch_root / name).is_file()}
    save(out / "environment.json", {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
         "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
         "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
         "torch_extension_sha256": sha(torch._C.__file__), "torch_build": torch.__config__.show(),
         "inspected_torch_sources_sha256": installed_sources,
         "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
         "device": "cpu", "dtype": "float32", "inference_mode": True, "deterministic_algorithms": True,
         "binary_path": str(binary), "binary_sha256": before_hash, "source_snapshot_sha256": source_hashes,
         "source_snapshot_note": "hashes of observed workspace; binary hash is authoritative, no clean-commit claim",
         "command": sys.argv, "seed": args.seed, "repeats": args.repeats, "warmups": args.warmups})
    dispatch = {}
    for operator in ("aten::mm", "aten::relu", "aten::add.Tensor", "aten::sum.dim_IntList"):
        dispatch[operator] = torch._C._dispatch_dump_table(operator)
    save(out / "pytorch-dispatch-tables.json", dispatch)
    cases = []
    profile_source = None
    for index, family in enumerate(args.families):
        for width in args.widths:
            for config in args.configs:
                evidence, graph, inputs = run_case(torch, binary, out, family, width,
                                                   args.seed + index*17 + width, config, args.repeats, args.warmups)
                cases.append({"case": evidence["case"], "correct": evidence["correct"],
                              "placements": len(evidence["placements"]), "comparison": evidence["case"] + "/comparison.json"})
                if family == "reduction" or profile_source is None:
                    profile_source = (graph, inputs, evidence["case"])
                print(f"{evidence['case']}: {len(evidence['placements'])} placements, {'PASS' if evidence['correct'] else 'FAIL'}")
    profiler = profile_graph(torch, profile_source[0], profile_source[1], out)
    if sha(binary) != before_hash:
        raise RuntimeError("binary changed during comparison; discard this incomplete run")
    summary = {"schema_version": 1, "purpose": "external production operator runtime comparison",
               "production_runtime": "PyTorch CPU eager operators", "capstone_backend": "software",
               "board_validation": False, "cases": cases, "total_cases": len(cases),
               "total_placements": sum(case["placements"] for case in cases),
               "total_measured_placement_runs": args.repeats*sum(case["placements"] for case in cases),
               "correct": all(case["correct"] for case in cases), "performance_conclusion": None,
               "atol": 1e-5, "rtol": 1e-5, "profiled_case": profile_source[2],
               "profile_operators": [row["operator"] for row in profiler],
               "timing_scope": {"pytorch_execution": "Python graph loop and CPU eager ops; precreated inputs; inference mode; output serialization excluded",
                                "capstone_host_execution": "command interpreter and strict kernel validation; source preparation and independent reference excluded",
                                "capstone_end_to_end": "runner lifetime including source prep and reference; async queue wait included when selected",
                                "process_wall": "one capstone verify subprocess for all placements, parsing, planning and JSON output included",
                                "warmup": "initial discarded PyTorch calls and capstone subprocess runs; capstone repeats use fresh processes",
                                "comparability": "different scopes; no cross-runtime speedup ratio is justified"}}
    save(out / "summary.json", summary)
    return 0 if summary["correct"] else 2


if __name__ == "__main__":
    sys.exit(main())
