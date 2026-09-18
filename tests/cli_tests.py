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
print(f"PASS: {checks} CLI invocations plus report/trace contracts")
