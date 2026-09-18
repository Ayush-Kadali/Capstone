"""Explain a compare/demo report using its recorded commands and cost profile."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def breakdown(result, config):
    commands = result["commands"]
    launch = result["islands"] * config["launch_ms"]
    compute = sum(c["modeled_ms"] for c in commands if c["kind"] == "COMPUTE") - launch
    copies = sum(c["modeled_ms"] for c in commands if c["kind"] == "COPY")
    sync = sum(c["modeled_ms"] for c in commands if c["kind"] == "SYNC")
    if not math.isclose(compute + copies + launch + sync, result["modeled_ms"], abs_tol=1e-8):
        raise ValueError("Cost breakdown does not match recorded plan total")
    return compute, copies, launch, sync


def copy_counts(result):
    return Counter((c["inputs"][0], c["output"]) for c in result["commands"] if c["kind"] == "COPY")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    results = {r["policy"]: r for r in data["results"]}
    if "cpu" not in results or "cost" not in results:
        parser.error("Use a report produced by compare or demo")

    print("Placement comparison (all times below are modeled milliseconds)")
    print(f"{'Policy':<10} {'Map':<14} {'Compute':>9} {'Copies':>9} {'Launch':>9} {'Sync':>9} {'Total':>9}")
    for result in results.values():
        parts = breakdown(result, data["config"])
        print(f"{result['policy']:<10} {result['placement']:<14}" + "".join(f" {p:9.4f}" for p in (*parts, result["modeled_ms"])))

    selected = results["cost"]
    baseline = results.get("maximal", results["cpu"])
    print(f"\nComparing {baseline['policy']} with cost-selected placement:")
    changes = 0
    for node, old, new in zip(data["compiled_graph"]["nodes"], baseline["placement"], selected["placement"]):
        if old != new:
            changes += 1
            print(f"  {node['id']} ({node['op']}): {'device' if old == 'A' else 'CPU'} -> {'device' if new == 'A' else 'CPU'}")
    if not changes:
        print("  Both policies selected the same assignment.")
    for label, counts in (("Removed", copy_counts(baseline) - copy_counts(selected)),
                          ("Added", copy_counts(selected) - copy_counts(baseline))):
        for (source, destination), count in sorted(counts.items()):
            print(f"  {label} copy: {source} -> {destination} (x{count})")
    print(f"  Retained device bytes: {baseline['peak_allocated_device_bytes']} -> {selected['peak_allocated_device_bytes']}")
    print(f"  Synchronization commands: {baseline['synchronizations']} -> {selected['synchronizations']}")
    print(f"  Model cost difference: {baseline['modeled_ms'] - selected['modeled_ms']:.4f} ms")
    print(f"  All recorded outputs match the original graph: {all(r['correct'] for r in results.values())}")
    print("\nThis explains the declared model. Shape-independent compute constants and a")
    print("launch charge per device run are simplifications; this is not GPU speedup evidence.")


if __name__ == "__main__":
    main()
