"""Seeded small DAGs checked with exact rational arithmetic and independent residency.

No third-party packages. Binary-fraction inputs keep these small cases exactly
representable in FP32; this is an execution/lowering oracle, not a timing oracle.
"""
import itertools
import json
from fractions import Fraction
from pathlib import Path
import random
import subprocess
import sys

binary = str(Path(sys.argv[1]).resolve())
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
seed = 20261004
rng = random.Random(seed)
executions = 0


def evaluate(op, a, b, width):
    if op == "relu":
        return [max(Fraction(0), x) for x in a]
    if op == "add":
        return [x + y for x, y in zip(a, b)]
    return [sum((a[i * width + k] * b[k * width + j] for k in range(width)), Fraction(0))
            for i in range(width) for j in range(width)]


def structural_cost(nodes, placement, output, tensor_bytes):
    # Count distinct resident tensors directly from graph edges, independent of
    # emitted command descriptors. Immutable tensors never need a second copy.
    host = {"X", "W"}
    device = set()
    h2d = d2h = 0
    for (_, inputs, result), backend in zip(nodes, placement):
        resident = host if backend == "C" else device
        missing = set(inputs) - resident
        if backend == "C":
            d2h += len(missing) * tensor_bytes
        else:
            h2d += len(missing) * tensor_bytes
        resident.update(missing)
        resident.add(result)
    if output not in host:
        d2h += tensor_bytes
    return h2d, d2h, len(device) * tensor_bytes


for case in range(18):
    width = 1 + case % 3
    values = {name: [Fraction(rng.randint(-4, 4), 4) for _ in range(width * width)] for name in ("X", "W")}
    lines = [f"# differential seed={seed} case={case}"]
    for name, data in values.items():
        lines.append(f"tensor {name} f32 {width} {width} " + " ".join(str(float(x)) for x in data))
    nodes = []
    for index in range(3 + case % 3):
        op = ("matmul", "relu", "add")[index % 3]
        available = list(values)
        first = available[-1] if index % 2 else rng.choice(available)
        inputs = [first] if op == "relu" else [first, rng.choice(available)]
        result = f"T{index}"
        values[result] = evaluate(op, values[first], values[inputs[-1]], width)
        nodes.append((op, inputs, result))
        lines.append(f"{op} n{index} " + " ".join(inputs) + f" {result}")
    # Also exercise observable intermediates and input-valued graph outputs.
    output = "X" if case % 6 == 0 else ("T0" if case % 6 == 1 else nodes[-1][2])
    lines.append(f"output {output}")
    graph = out / f"case-{case}.plg"
    graph.write_text("\n".join(lines) + "\n")
    expected_output = [float(x) for x in values[output]]
    for supported, budget in ((False, 1048576), (True, 1048576), (True, width * width * 4 * 3)):
        report = out / f"case-{case}-{int(supported)}-{budget}.json"
        command = [binary, "verify", "--graph", str(graph), "--device-bytes", str(budget), "--json", str(report)]
        if supported:
            command.append("--relu-supported")
        process = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert process.returncode == 0, (command, process.stderr)
        data = json.loads(report.read_text())
        expected = {}
        tensor_bytes = width * width * 4
        for placement in itertools.product("CA", repeat=len(nodes)):
            if any(backend == "A" and op == "relu" and not supported for (op, _, _), backend in zip(nodes, placement)):
                continue
            counts = structural_cost(nodes, placement, output, tensor_bytes)
            if counts[2] <= budget:
                expected["".join(placement)] = counts
        actual = {row["placement"]: row for row in data["results"]}
        assert len(actual) == len(data["results"]), "duplicate placement result"
        assert len(actual) + data["search"]["infeasible"] == 2 ** len(nodes)
        assert set(actual) == set(expected), (case, supported, budget, set(actual), set(expected))
        for placement, row in actual.items():
            assert row["correct"] and row["output"] == expected_output, (case, placement, row["output"], expected_output)
            assert (row["h2d_bytes"], row["d2h_bytes"], row["peak_allocated_device_bytes"]) == expected[placement]
            commands = row["commands"]
            assert [cmd["id"] for cmd in commands] == list(range(len(commands)))
            assert sum(cmd["shape"][0] * cmd["shape"][1] * 4 for cmd in commands
                       if cmd["kind"] == "ALLOC" and cmd["backend"] == "virtual-device") == expected[placement][2]
            executions += 1
        # A second identical run must preserve every structural/numerical report
        # field. Host measurements are explicitly allowed to vary.
        again = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert again.returncode == 0, again.stderr
        repeated = json.loads(report.read_text())
        for payload in (data, repeated):
            for row in payload["results"]:
                del row["host_execution_ms"]
                del row["end_to_end_ms"]
                for event in row["events"]:
                    del event["start_us"], event["duration_us"]
        assert data == repeated, f"nondeterministic structural report in case {case}"
summary = {"seed": seed, "graphs": 18, "configurations": 54, "placements_checked": executions,
           "repeat_runs": 54, "oracle": "exact Fraction arithmetic; graph-edge residency sets; exhaustive independent assignments"}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"PASS: {executions} placements across 54 cases; exact rational output, residency and deterministic reports")
