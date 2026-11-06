"""Verify the integrated software candidate in a fresh source/build copy.

Run from the repository root: python3 scripts/verify_release.py --out artifacts/release-check
The destination must not exist. No original build or presentation artifacts change.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
workspace = out / "workspace"
workspace.mkdir()
# Copy only release sources and runnable documentation, excluding private material,
# academic submissions and pre-existing generated artifacts.
files = [root / "CMakeLists.txt", root / "README.md", root / "requirements-onnx.txt"] + list((root / "profiles").rglob("*.json"))
for directory in ("src", "include", "tests", "examples", "scripts", "showcase", "workbench", "docs/runtime", "planning"):
    files.extend(p for p in (root / directory).rglob("*") if p.is_file()
                 and p.suffix in (".cpp", ".hpp", ".py", ".sh", ".plg", ".md", ".html", ".js", ".css",
                                  ".onnx", ".npz", ".npy", ".pt", ".json")
                 and "__pycache__" not in p.parts and "html-guide" not in p.parts)
hashes = {}
for source in sorted(files):
    relative = source.relative_to(root)
    target = workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
manifest = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "machine": platform.machine(), "python": sys.version, "source_sha256": hashes,
            "commands": [], "status": "running"}


def save_manifest():
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run(command, expected=0, label=None, timeout=900):
    index = len(manifest["commands"])
    log = f"{index:03d}-{label or 'command'}.log"
    print(shlex.join(command), flush=True)
    with (out / log).open("w") as stream:
        result = subprocess.run(command, cwd=workspace, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, env={**os.environ, "CAPSTONE_TEST_PYTHON": sys.executable})
    manifest["commands"].append({"argv": command, "cwd": str(workspace), "expected_exit": expected,
                                 "exit": result.returncode, "log": log})
    save_manifest()
    if result.returncode != expected:
        raise RuntimeError(f"Expected exit {expected}, got {result.returncode}: {command}; see {log}")


try:
    for command in (["cmake", "--version"], ["c++", "--version"], [sys.executable, "--version"]):
        run(command, label="environment")
    # Includes clean Release configure/build, all CTest suites and five demo scenes.
    run(["bash", "scripts/run_demo.sh"], label="release-demo", timeout=3600)
    # Exercise each unique documented runnable command. Build/test/demo commands
    # are covered above, sanitizer commands below, so duplicate docs do not
    # manufacture repeated verification. Commands are argv, never shell code.
    docs = ["README.md", "showcase/README.md", "showcase/CLI_WALKTHROUGH.md",
            "showcase/DEMO_SCRIPT.md", "showcase/SUBMISSION.md", "showcase/ENGINEERING_NOTES.md",
            "docs/runtime/ANALYTICAL_MODEL.md", "docs/runtime/TRAINED_EXAMPLE.md", "docs/runtime/COMMAND_QUEUE.md"]
    interpreters = ("python3", "$PY", ".venv/bin/python", "/path/to/venv/bin/python")
    # Environment setup and build lines are exercised by the explicit build steps.
    setup_prefixes = ("PY=", "python3 -m venv", ".venv/bin/pip", "cmake", "ctest", "bash", "(cd")
    manifest["skipped_documented_setup_lines"] = []
    seen = set()
    documented = []
    for doc in docs:
        content = (workspace / doc).read_text()
        for block in re.findall(r"```sh\n(.*?)```", content, flags=re.S):
            for line in block.replace("\\\n", " ").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(setup_prefixes) or "&&" in line:
                    manifest["skipped_documented_setup_lines"].append({"document": doc, "line": line})
                    continue
                argv = shlex.split(line, comments=True)
                if argv[0] in interpreters:
                    argv[0] = "python3"
                if argv[0] not in ("build/placementlens", "python3", "cat", "mkdir"):
                    raise RuntimeError(f"Unreviewed documented command in {doc}: {line}")
                if argv[0] == "python3":
                    argv[0] = sys.executable  # documented python3 = the release environment
                key = tuple(argv)
                if key in seen:
                    continue
                seen.add(key)
                expected = 1 if "examples/invalid_shape.plg" in argv else 0
                run(argv, expected, "documented")
                documented.append({"document": doc, "argv": argv, "expected_exit": expected})
    # Also check the standalone matrix command documented inline.
    run([sys.executable, "scripts/check_matrix.py"], label="standalone-matrix")
    manifest["documented_commands"] = documented
    run(["cmake", "-S", ".", "-B", "build-asan", "-DCMAKE_BUILD_TYPE=Debug", "-DPLACEMENTLENS_SANITIZERS=ON", "-DPython3_EXECUTABLE=" + sys.executable], label="asan-configure")
    run(["cmake", "--build", "build-asan", "--parallel", "2"], label="asan-build")
    run(["ctest", "--test-dir", "build-asan", "-V"], label="asan-tests", timeout=3600)
    run(["cmake", "-S", ".", "-B", "build-tsan", "-DCMAKE_BUILD_TYPE=Debug", "-DPLACEMENTLENS_TSAN=ON", "-DPython3_EXECUTABLE=" + sys.executable], label="tsan-configure")
    run(["cmake", "--build", "build-tsan", "--parallel", "2"], label="tsan-build")
    run(["ctest", "--test-dir", "build-tsan", "-V", "-R", "async_contracts|queue_contracts|residency_contracts|runtime_review_contracts"], label="tsan-tests", timeout=1800)
    run([sys.executable, "scripts/capstone.py", "demo", "--out", "demo-evidence", "--binary", "build/placementlens"], label="six-step-demo")
    run([sys.executable, "scripts/cross_check_engine.py", "--binary", "build/placementlens", "--out", "cross-check.json"], label="engine-cross-check", timeout=1800)
    for build in ("build", "build-asan", "build-tsan"):
        executable = workspace / build / "placementlens"
        manifest.setdefault("binary_sha256", {})[build] = hashlib.sha256(executable.read_bytes()).hexdigest()
    manifest["status"] = "passed"
except Exception as error:
    manifest["status"] = "failed"
    manifest["error"] = str(error)
    raise
finally:
    save_manifest()
print(f"PASS: release evidence saved in {out}")
