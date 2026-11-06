#!/usr/bin/env python3
"""Optional PyTorch numerical integration; exit 77 only when torch is absent."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import torch
except ModuleNotFoundError as error:
    if error.name == "torch":
        print("SKIP: PyTorch unavailable")
        sys.exit(77)
    raise

binary, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location("production_comparison", root / "scripts/compare_production.py")
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)
checks = 0


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(label)


def rejects(action):
    try:
        action()
    except ValueError:
        check(True, "expected rejection")
        return
    check(False, "expected ValueError")


graph = {"output": "Y", "inputs": [
    {"name": "A", "dtype": "f32", "layout": "contiguous-row-major", "shape": [2,3], "values": [1,-2,3,0,4,-1]},
    {"name": "B", "dtype": "f32", "layout": "contiguous-row-major", "shape": [3,2], "values": [2,1,1,-3,-2,2]}],
    "nodes": [{"id": "mm", "op": "MatMul", "inputs": ["A","B"], "output": "M", "shape": [2,2]},
              {"id": "relu", "op": "ReLU", "inputs": ["M"], "output": "R", "shape": [2,2]},
              {"id": "sum", "op": "RowSum", "inputs": ["R"], "output": "Y", "shape": [2,1]}]}
with torch.inference_mode():
    inputs = production.prepare_torch(torch, graph)
    check(production.execute_torch(torch, graph, inputs).flatten().tolist() == [13,6], "independent hand-computed production adapter output")
    check(inputs["A"].flatten().tolist() == [1,-2,3,0,4,-1], "production adapter preserves input tensors")
    malformed = copy.deepcopy(graph)
    malformed["nodes"][0]["op"] = "Unsupported"
    rejects(lambda: production.execute_torch(torch, malformed, inputs))
    malformed = copy.deepcopy(graph)
    malformed["nodes"][0]["shape"] = [4,1]
    rejects(lambda: production.execute_torch(torch, malformed, inputs))
    malformed = copy.deepcopy(graph)
    malformed["inputs"][0]["dtype"] = "f16"
    rejects(lambda: production.prepare_torch(torch, malformed))
check(production.compare_values([1.0], [1.0])["correct"], "equal output accepted")
check(not production.compare_values([0.0], [1.0])["correct"], "mismatch detected")
check(not production.compare_values([float("nan")], [1.0])["correct"], "nonfinite result rejected")
check(not production.compare_values([], [1.0])["correct"], "missing output rejected")
check(production.generate("chain", 2, 17) == production.generate("chain", 2, 17), "seeded graph generator reproducible")
rejects(lambda: production.generate("unknown", 2, 17))
rejects(lambda: production.generate("chain", 0, 17))

with tempfile.TemporaryDirectory(prefix="placementlens-production-test-") as directory:
    command = [sys.executable, str(root / "scripts/compare_production.py"), "--binary", str(binary),
               "--out", directory, "--families", "chain", "reduction", "--widths", "2",
               "--configs", "pooled-tiled", "fused-prepared-async", "--repeats", "2", "--warmups", "1"]
    run = subprocess.run(command, text=True, capture_output=True, timeout=180)
    check(run.returncode == 0, run.stdout + run.stderr)
    out = Path(directory)
    summary = json.loads((out / "summary.json").read_text())
    check(summary["correct"] and summary["total_cases"] == 4, "production comparison completes all bounded cases")
    check(summary["total_placements"] == 24 and summary["total_measured_placement_runs"] == 48, "every feasible placement compared repeatedly")
    check(summary["performance_conclusion"] is None and not summary["board_validation"], "comparison makes no unsupported speedup or board claim")
    check("aten::mm" in summary["profile_operators"] and "aten::sum" in summary["profile_operators"], "real profiler observes production MatMul and RowSum")
    trace = json.loads((out / "pytorch-cpu-trace.json").read_text())
    check(bool(trace["traceEvents"]), "production profiler exported actual events")
    environment = json.loads((out / "environment.json").read_text())
    check(environment["torch_threads"] == 1 and environment["device"] == "cpu", "runtime configuration retained")
    check(environment["binary_sha256"] == production.sha(binary), "binary identity matches tested executable")
    for case in summary["cases"]:
        evidence = json.loads((out / case["comparison"]).read_text())
        check(evidence["correct"] and len(evidence["pytorch_runs"]) == 3, "raw warmup and measurement outputs retained")
        check(evidence["pytorch_execution"]["count"] == 2, "warmup excluded from timing summaries")
        for row in evidence["placements"]:
            check(row["correct"] and row["host_execution"]["count"] == 2, "per-placement measured distribution and correctness retained")
        raw = json.loads((out / case["case"] / "capstone-00.json").read_text())
        production.validate_coverage(raw)
        missing = copy.deepcopy(raw)
        missing["results"].pop()
        rejects(lambda: production.validate_coverage(missing))
    rerun = subprocess.run(command, text=True, capture_output=True, timeout=30)
    check(rerun.returncode != 0 and "must be empty" in rerun.stderr, "existing experiment evidence is not overwritten")

print(f"PASS {checks} production comparison assertions")
