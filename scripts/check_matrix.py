"""Run deterministic graph/configuration cases through every feasible placement.

This is a functional verification matrix, not a hardware performance benchmark.
Generated inputs, full reports, and a summary stay in the output directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess


def workload(topology, width, seed):
    rng = random.Random(seed)
    lines = [f"# generated {topology}, width={width}, seed={seed}"]
    for name in ("X", "W", "B"):
        # Binary fractions keep round-off modest. This suite does not claim to
        # cover ill-conditioned matrices or large cancellation-heavy reductions.
        values = [str(rng.randint(-8, 8) / 16) for _ in range(width * width)]
        lines.append(f"tensor {name} f32 {width} {width} " + " ".join(values))
    lines.extend(["matmul mm X W M", "relu activation M R"])
    if topology == "chain":
        lines.append("add bias R B Y")
    elif topology == "shared":
        lines.extend(["add reuse M R S", "add bias S B Y"])
    else:
        lines.extend(["add skip R X S", "matmul projection S W P",
                      "relu final_activation P T", "add output_bias T B Y"])
    lines.append("output Y")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/placementlens"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/verification"))
    args = parser.parse_args()
    binary = args.binary.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for topology in ("chain", "shared", "residual"):
        for width in (2, 8, 16):
            seed = 20260918 + width
            graph = args.out / f"{topology}-{width}.plg"
            graph.write_text(workload(topology, width, seed))
            tensor_bytes = width * width * 4
            configs = {
                "fragmented": [],
                "supported": ["--relu-supported"],
                "fused": ["--relu-supported", "--fuse"],
                "tight-memory": ["--relu-supported", "--device-bytes", str(3 * tensor_bytes)],
                "expensive-copies": ["--boundary-ms", "5"],
            }
            for config_name, flags in configs.items():
                name = f"{topology}-{width}-{config_name}"
                report = args.out / f"{name}.json"
                command = [str(binary), "verify", "--graph", str(graph), *flags, "--json", str(report)]
                run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                (args.out / f"{name}.txt").write_text(run.stdout + run.stderr)
                if run.returncode:
                    raise RuntimeError(f"{name} failed: {run.stdout}\n{run.stderr}")
                data = json.loads(report.read_text())
                results = data["results"]
                placements = [r["placement"] for r in results]
                count = len(data["compiled_graph"]["nodes"])
                expected = {p["placement"] for p in data["search"]["ranked_feasible"]}
                if len(set(placements)) != len(placements) or set(placements) != expected:
                    raise RuntimeError(f"{name}: duplicate or unexecuted feasible placement")
                if len(results) + data["search"]["infeasible"] != 2 ** count:
                    raise RuntimeError(f"{name}: assignments missing from enumeration")
                if not results or not all(r["correct"] for r in results):
                    raise RuntimeError(f"{name}: numerical mismatch")
                rows.append({
                    "case": name, "seed": seed, "source_nodes": len(data["source_graph"]["nodes"]),
                    "compiled_nodes": count, "executed_placements": len(results),
                    "rejected_assignments": data["search"]["infeasible"],
                    "executed_commands": sum(len(r["commands"]) for r in results),
                    "max_abs_error": max(r["max_abs_error"] for r in results),
                    "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
                    "report": report.name, "command": command,
                })
                print(f"{name:<32} {len(results):>3} placements checked")
    summary = {
        "purpose": "functional verification, no real accelerator performance claims",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "cases": rows,
        "total_cases": len(rows),
        "total_executed_placements": sum(r["executed_placements"] for r in rows),
        "total_executed_commands": sum(r["executed_commands"] for r in rows),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nPASS: {len(rows)} cases, {summary['total_executed_placements']} executed placements, "
          f"{summary['total_executed_commands']} commands. Reports: {args.out}")


if __name__ == "__main__":
    main()
