"""Read-only BeagleY-AI readiness record. Does not install, flash or run inference."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


def read_optional(path, limit=16384):
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            return {"status": "unavailable", "reason": "read limit exceeded"}
        return {"status": "observed", "text": data.decode("utf-8", errors="replace").replace("\0", "\n").strip()}
    except OSError as error:
        return {"status": "unavailable", "reason": str(error)}


def is_beagley(model):
    normalized = "".join(c for c in model.lower() if c.isalnum())
    return "beagleyai" in normalized


def collect():
    model = read_optional("/proc/device-tree/model")
    target = model["status"] == "observed" and is_beagley(model["text"])
    processors = []
    for entry in sorted(Path("/sys/class/remoteproc").glob("remoteproc*")):
        processors.append({"id": entry.name, **{key: read_optional(entry / key) for key in ("name", "state", "firmware")}})
    runtime = {"status": "unavailable", "reason": "onnxruntime module is not installed"}
    # Probe an installed module in a bounded child: a broken vendor library must
    # not discard the platform observations or hang this readiness command.
    if importlib.util.find_spec("onnxruntime"):
        command = [sys.executable, "-c", "import json,onnxruntime as r; print(json.dumps({'version':r.__version__,'providers':r.get_available_providers()}))"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                runtime = {"status": "observed", **json.loads(result.stdout)}
            else:
                runtime = {"status": "probe_failed", "exit": result.returncode, "stderr": result.stderr[-4096:]}
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as error:
            runtime = {"status": "probe_failed", "reason": str(error)}
    return {"schema_version": 1, "artifact_type": "board-readiness", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_identity_observed": target, "platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version, "model": model, "os_release": read_optional("/etc/os-release"),
            "kernel": platform.release(), "remote_processors": processors, "onnx_runtime": runtime,
            "accelerator_execution_validated": False,
            "unverified": ["exact SD image/checksum and SDK/firmware tool compatibility", "actual TI offloaded partitions",
                "project compiled-subgraph adapter", "same-input numerical correctness", "repeated board measurements and profiler evidence"],
            "interpretation": "Read-only inventory. Provider or running firmware presence is not evidence of accelerator workload execution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; preserve earlier readiness records")
    record = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also protects an output that appeared during collection.
    with args.out.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved read-only readiness record: {args.out}")
    print("Board identity observed; accelerator execution still unverified." if record["target_identity_observed"] else
          "Target unavailable: this host has not been identified as BeagleY-AI.")
    return 0 if record["target_identity_observed"] else 77


if __name__ == "__main__":
    sys.exit(main())
