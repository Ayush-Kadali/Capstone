"""End-to-end contracts: real executable, exported plans, errors and traces."""
import json
import pathlib
import subprocess
import sys
import tempfile

binary, root = sys.argv[1], pathlib.Path(sys.argv[2])
checks = 0

def invoke(*args, expected=0):
    global checks
    result = subprocess.run([binary, *map(str, args)], capture_output=True, text=True, timeout=30)
    assert result.returncode == expected, (args, result.stdout, result.stderr)
    checks += 1
    return result

with tempfile.TemporaryDirectory() as td:
    report, trace = pathlib.Path(td)/"report.json", pathlib.Path(td)/"trace.json"
    base = ["--graph", root/"examples/fragmented.plg"]
    invoke("demo", *base, "--json", report, "--trace", trace)
    data = json.loads(report.read_text())
    assert [r["placement"] for r in data["results"]] == ["CCC", "ACA", "ACC"]
    assert all(r["correct"] and r["output"] == [1,14,7,1] for r in data["results"])
    assert data["search"]["infeasible"] == 4
    assert len(data["search"]["ranked_feasible"]) == 4
    assert data["results"][1]["modeled_ms"] > data["results"][0]["modeled_ms"]
    assert "uncalibrated" in data["cost_provenance"]
    events = json.loads(trace.read_text())["traceEvents"]
    assert len(events) == len(data["results"][-1]["commands"])
    assert all(e["dur"] >= 0 and e["ts"] >= 0 and e["tid"] == 1 for e in events)
    for result in data["results"]:
        assert abs(sum(c["modeled_ms"] for c in result["commands"]) - result["modeled_ms"]) < 1e-8
        assert [p["name"] for p in result["passes"]] == ["CapabilityAnalysisPass", "PlacementPass", "InsertTransfersPass", "LowerToExecutionPlanPass"]
        assert all(p["input_ir"] and p["output_ir"] and p["reasons"] and p["consequences"] for p in result["passes"])
        assert "ALLOC" not in result["passes"][2]["output_ir"]
        assert "ALLOC" in result["passes"][3]["output_ir"]
    assert data["frontend_passes"][0]["name"] == "ValidateGraphPass"
    assert "tensor A f32" in data["frontend_passes"][0]["input_ir"]
    explanation = subprocess.run([sys.executable, str(root/"scripts/explain_run.py"), str(report)],
                                 capture_output=True, text=True, timeout=30)
    assert explanation.returncode == 0, explanation.stderr
    assert "bias (Add): device -> CPU" in explanation.stdout
    assert "Removed copy: H:R -> D:R" in explanation.stdout
    assert "Model cost difference: 2.5500 ms" in explanation.stdout
    invoke("demo", *base, "--relu-supported", "--fuse", "--json", report)
    fused = json.loads(report.read_text())
    assert len(fused["compiled_graph"]["nodes"]) == 2
    assert fused["compiled_graph"]["nodes"][0]["origins"] == ["mm", "activation"]
    invoke("compare", *base, "--device-bytes", "80", "--json", report)
    assert [r["policy"] for r in json.loads(report.read_text())["results"]] == ["cpu", "cost"]
    invoke("run", *base, "--policy", "maximal", "--device-bytes", "80", expected=1)
    invoke("run", "--graph", root/"examples/invalid_shape.plg", expected=1)
    invoke("inspect", *base)
    invoke("compile", *base)
    invoke("run", *base, "--boundary-ms", "nan", expected=1)
    invoke("run", *base, "--boundary-ms", "-1", expected=1)
    invoke("run", *base, "--device-bytes", "-1", expected=1)
    invoke("run", *base, "--policy", "unknown", expected=1)
    invoke("run", *base, "--boundary-ms", "1oops", expected=1)
    invoke("run", *base, "--oops", expected=1)
    invoke("run", *base, "--json", root/"examples/fragmented.plg", expected=1)
    invoke("run", *base, "--json", root/"examples/../examples/fragmented.plg", expected=1)
    invoke("--help")
    for options in ([], ["--relu-supported"], ["--relu-supported", "--fuse"], ["--device-bytes", "80"]):
        result = invoke("verify", *base, *options, "--json", report)
        verified = json.loads(report.read_text())
        assert "numerical failures" in result.stdout
        assert len(verified["results"]) == len(verified["search"]["ranked_feasible"])
        assert len(verified["results"]) + verified["search"]["infeasible"] == 2 ** len(verified["compiled_graph"]["nodes"])
        assert {r["placement"] for r in verified["results"]} == {r["placement"] for r in verified["search"]["ranked_feasible"]}
        assert all(r["correct"] and r["output"] == [1,14,7,1] for r in verified["results"])
    invoke("verify", *base, "--trace", trace, expected=1)
    markdown = pathlib.Path(td)/"run.md"
    for options in ([], ["--relu-supported", "--fuse"], ["--device-bytes", "80"], ["--boundary-ms", "5"]):
        result = invoke("explain", *base, *options, "--report", markdown, "--json", report, "--trace", trace)
        explanation = markdown.read_text()
        evidence = json.loads(report.read_text())
        assert result.stdout.startswith(explanation)
        assert "## Compiler decisions" in explanation and "## Selected executable plan" in explanation
        assert "shape" in explanation and "hardware-calibrated" in explanation
        assert all(r["correct"] for r in evidence["results"])
        assert "| cost | " + evidence["results"][-1]["placement"] in explanation
        if options == []:
            assert "bias (Add): virtual-device -> CPU" in explanation
            assert "Copy bytes: 112 -> 64" in explanation
            assert "Total modeled cost reduction: 2.550048 ms" in explanation
        if "--fuse" in options:
            assert "FUSE mm + activation" in explanation
            assert "mm activation" in explanation
        if "--device-bytes" in options:
            assert "Maximal placement is infeasible" in explanation
    invoke("explain", *base, "--report", root/"examples/fragmented.plg", expected=1)
    invoke("explain", *base, "--report", report, "--json", report, expected=1)
    invoke("run", *base, "--report", markdown, expected=1)
    invoke("explain", "--help")
    # Inputs and output paths fail explicitly, without damaging the input graph.
    invoke("unknown", *base, expected=1)
    invoke("run", expected=1)
    invoke("run", *base, "--json", expected=1)
    invoke("run", "--graph", pathlib.Path(td)/"missing.plg", expected=1)
    invoke("run", "--graph", td, expected=1)
    invoke("run", *base, "--json", pathlib.Path(td)/"missing"/"out.json", expected=1)
    fixture = pathlib.Path(td)/"input.plg"
    original = (root/"examples/fragmented.plg").read_text()
    fixture.write_text(original)
    alias = pathlib.Path(td)/"alias.plg"
    alias.symlink_to(fixture)
    invoke("run", "--graph", fixture, "--json", alias, expected=1)
    assert fixture.read_text() == original
    for invalid in (b"\xff", b"\xc0\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2\x82", b"\xe2X\xa0"):
        fixture.write_bytes(b"#" + invalid + b"\n" + original.encode())
        result = invoke("run", "--graph", fixture, "--json", report, expected=1)
        assert "UTF-8" in result.stderr
    fixture.write_text("# café → 🙂\n" + original, encoding="utf-8")
    invoke("run", "--graph", fixture, "--json", report)
    assert "café → 🙂" in json.loads(report.read_text(encoding="utf-8"))["frontend_passes"][0]["input_ir"]
    oversized = pathlib.Path(td)/"oversized.plg"
    oversized.write_bytes(b"#" * 16000001)
    result = invoke("inspect", "--graph", oversized, expected=1)
    assert "E_LIMIT: graph file exceeds 16 MB" in result.stderr
    for dimension in ("-18446744073709551615", "18446744073709551616", "-1", "0", "1oops"):
        fixture.write_text(f"tensor A f32 {dimension} 1 1\nrelu r A Y\noutput Y\n")
        invoke("run", "--graph", fixture, expected=1)
    fixture.write_text("tensor A f32 1 1 3e38\ntensor B f32 1 1 -2\nmatmul m A B M\nrelu r M Y\noutput Y\n")
    for fusion_flags in ([], ["--fuse"]):
        result = invoke("run", "--graph", fixture, "--relu-supported", *fusion_flags, expected=1)
        assert "E_NUMERIC" in result.stderr
    # A finite runtime sum can have an unusable, overflowing reference. Reject
    # explicitly, before writing a report with an invalid `inf` numeric token.
    fixture.write_text("tensor A f32 1 12 " + " ".join(["2.83568648e37"] * 12) +
                       "\ntensor B f32 12 1 " + " ".join(["1"] * 12) + "\nmatmul m A B Y\noutput Y\n")
    overflow_report = pathlib.Path(td)/"overflow.json"
    result = invoke("run", "--graph", fixture, "--policy", "cpu", "--json", overflow_report, expected=1)
    assert "E_NUMERIC: reference" in result.stderr and not overflow_report.exists()
    fixture.write_text("tensor A f32 1 1 1\nsoftmax r A Y\noutput Y\n")
    result = invoke("run", "--graph", fixture, expected=1)
    assert "unknown statement" in result.stderr
    fixture.write_text("tensor A f32 1 3 100000000 1 -100000000\n"
                       "tensor B f32 3 1 1 1 1\nmatmul m A B Y\noutput Y\n")
    invoke("run", "--graph", fixture, "--policy", "cpu", "--json", report, expected=2)
    mismatch = json.loads(report.read_text())["results"][0]
    assert not mismatch["correct"] and mismatch["max_abs_error"] == 1 and mismatch["output"] == [0]
    invoke("compare", *base, "--async", "--json", report, "--trace", trace)
    asynchronous = json.loads(report.read_text())
    assert [r["placement"] for r in asynchronous["results"]] == ["CCC", "ACA", "ACC"]
    for row in asynchronous["results"]:
        assert row["correct"] and row["output"] == [1,14,7,1]
        assert row["execution_backend"].startswith("async worker:")
        assert row["queue_wait_ms"] >= 0 and row["end_to_end_ms"] >= row["host_execution_ms"]
        assert len(row["events"]) == len(row["commands"])
        for command, event in zip(row["commands"], row["events"]):
            assert event["command_id"] == command["id"] and event["name"] == command["kind"]
            assert event["start_us"] >= 0 and event["duration_us"] >= 0 and event["device_us"] is None
    invoke("run", *base, "--backend", "unknown", expected=1)
    invoke("compile", *base, "--async", expected=1)
    invoke("verify", *base, "--async", "--json", report)
    assert all(r["correct"] for r in json.loads(report.read_text())["results"])
    invoke("verify", *base, "--reuse-buffers", "--tile-size", "2", "--local-bytes", "48",
           "--resident-input", "B", "--resident-input", "A", "--order-seed", "19", "--json", report)
    resident = json.loads(report.read_text())
    assert resident["config"]["mirrored_inputs"] == ["A", "B"]
    assert all(r["correct"] and r["preparation_h2d_bytes"] == 48 and len(r["initial_buffers"]) == 2
               and r["end_to_end_ms"] >= r["host_execution_ms"] + r["input_preparation_ms"] for r in resident["results"])
    assert any(r["tile_count"] > 0 and r["software_buffer_reuses"] > 0 for r in resident["results"])
    invoke("run", *base, "--tile-size", "65", expected=1)
    invoke("run", *base, "--tile-size", "2", "--local-bytes", "47", expected=1)
    invoke("run", *base, "--resident-input", "missing", expected=1)
    invoke("run", *base, "--resident-input", "A", "--resident-input", "A", expected=1)
    invoke("run", *base, "--resident-input", "A", "--device-bytes", "23", expected=1)
    invoke("run", *base, "--sync-ms", "-1", expected=1)
    invoke("run", *base, "--bandwidth", "0", expected=1)
    invoke("run", *base, "--order-seed", "1", expected=1)
    # Command-level queue runtime: same results, per-command engine lanes and dependency waits.
    queued = invoke("run", *base, "--policy", "maximal", "--relu-supported", "--runtime", "command-queue",
                    "--emulate-boundaries", "--boundary-ms", "20", "--launch-ms", "20", "--json", report, "--trace", trace)
    assert "copy/compute engine overlap" in queued.stdout and "synthetic waits" in queued.stdout
    result = json.loads(report.read_text())["results"][0]
    assert result["correct"] and result["output"] == [1,14,7,1] and result["engine_overlap_us"] > 0
    assert result["execution_backend"].startswith("command-queue simulator")
    by_id = {e["command_id"]: e for e in result["events"]}
    for e in result["events"]:
        command = result["commands"][e["command_id"]]
        assert e["engine"] == ("copy-engine" if command["kind"] == "COPY" else
                               "compute-engine" if command["kind"] == "COMPUTE" and command["backend"] == "virtual-device" else "host")
        assert all(by_id[w]["start_us"] + by_id[w]["duration_us"] <= e["start_us"] for w in e["waits"])
    assert {e["tid"] for e in json.loads(trace.read_text())["traceEvents"]} == {1, 2, 3}
    invoke("verify", *base, "--reuse-buffers", "--resident-input", "A", "--runtime", "command-queue", "--json", report)
    assert all(r["correct"] for r in json.loads(report.read_text())["results"])
    invoke("run", *base, "--runtime", "parallel", expected=1)
    invoke("compile", *base, "--runtime", "command-queue", expected=1)
print(f"PASS: {checks} CLI invocations plus report/trace contracts")
