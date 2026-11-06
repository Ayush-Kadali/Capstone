#!/usr/bin/env python3
"""Staged, fail-closed TI ONNX Runtime adapter. See docs/runtime/TI_ADAPTER.md.

This invokes an already installed TI runtime only after explicit compile/run.
It never installs software, downloads models, or treats a provider list as offload.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile

TAG = "11_00_06_00"
SDK = "11.00.00.08"
FIRMWARE = "11.00.00.06"
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL_INPUT = 64 * 1024 * 1024
MAX_ARTIFACT_TOTAL = 512 * 1024 * 1024
COMPILE_EP = "TIDLCompilationProvider"
RUN_EP = "TIDLExecutionProvider"
CPU_EP = "CPUExecutionProvider"


class AdapterError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AdapterError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, limit=MAX_FILE):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), f"Expected a regular, non-symlink file: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, f"File exceeds {limit}-byte limit: {path}")
    return data


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_record(path):
    data = read(path, MAX_ARTIFACT_TOTAL)
    return {"size": len(data), "sha256": sha(data)}


def tree_records(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Artifact root must be a non-symlink directory")
    result, total = {}, 0
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            require(not (Path(current) / name).is_symlink(), "Artifact symlinks are unsupported")
        for name in sorted(files):
            path = Path(current) / name
            record = file_record(path)
            total += record["size"]
            require(total <= MAX_ARTIFACT_TOTAL and len(result) < 4096, "Artifact tree exceeds bounds")
            result[path.relative_to(root).as_posix()] = record
    return dict(sorted(result.items()))


def load_inputs(data, inputs):
    """Inspect NPY headers before NumPy allocates; never extract ZIP members."""
    import numpy as np
    require(len(data) <= MAX_FILE, "NPZ exceeds 16 MiB")
    shapes = {item["source_name"]: item["shape"] for item in inputs}
    require(1 <= len(shapes) <= 64, "Expected 1..64 named model inputs")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            require(len(members) == len(shapes), "NPZ array count differs from model")
            require(len({m.filename for m in members}) == len(members), "Duplicate NPZ member")
            require({m.filename for m in members} == {name + ".npy" for name in shapes}, "NPZ names must exactly match model inputs")
            require(sum(m.file_size for m in members) <= MAX_FILE, "Uncompressed NPZ exceeds 16 MiB")
            for member in members:
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    require(version in ((1, 0), (2, 0)), "Unsupported NPY version")
                    parser = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                    shape, _, dtype = parser(stream)
                    require(list(shape) == shapes[member.filename[:-4]] and dtype == np.dtype("float32"), "NPZ shape/dtype differs from static f32 model")
                    require(1 <= math.prod(shape) <= 65536, "Input tensor exceeds element bound")
                    require(member.file_size - stream.tell() == math.prod(shape) * 4, "NPY payload length differs from header")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        require(all(np.isfinite(a).all() for a in arrays.values()), "Nonfinite model input")
        return arrays
    except AdapterError:
        raise
    except Exception as error:
        raise AdapterError(f"Invalid non-pickle NPZ: {error}") from error


def input_identity(arrays):
    """Hash tensors, not ZIP metadata: repacking cannot hide calibration reuse."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(json.dumps([name, list(arrays[name].shape)], separators=(",", ":")).encode())
        digest.update(arrays[name].tobytes(order="C"))
    return digest.hexdigest()


def benchmark_proof(raw):
    """Documented TI get_TI_benchmark_data keys; no inference from EP presence."""
    require(isinstance(raw, dict), "TI benchmark data must be a dictionary")
    cleaned = {}
    for key, value in raw.items():
        require(isinstance(key, str) and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), "Invalid TI benchmark entry")
        cleaned[key] = value
    ids = sorted({match.group(1) for key in cleaned for match in [re.fullmatch(r"ts:subgraph_(\d+)_proc_start", key)] if match})
    partitions = []
    for subgraph in ids:
        prefix = "ts:subgraph_" + subgraph
        start, end = cleaned[prefix + "_proc_start"], cleaned.get(prefix + "_proc_end")
        require(isinstance(end, (int, float)) and end > start >= 0, "Invalid/nonpositive TI subgraph processing interval")
        copies = {}
        for direction in ("in", "out"):
            a, b = cleaned.get(prefix + "_copy_" + direction + "_start"), cleaned.get(prefix + "_copy_" + direction + "_end")
            if a is not None or b is not None:
                require(a is not None and b is not None and b >= a >= 0, "Malformed TI copy interval")
                copies[direction + "_ns"] = b - a
        partitions.append({"subgraph_id": subgraph, "processing_ns": end - start, "copy_intervals": copies})
    require(partitions, "No positive TI subgraph execution evidence; CPU-only fallback cannot pass")
    # TI's pinned example divides these timestamp deltas by 1e6 for milliseconds.
    return {"status": "vendor_subgraph_timestamps_observed", "partition_count": len(partitions),
            "partitions": partitions, "raw": cleaned,
            "source_operator_mapping": "unknown; subgraph timing does not identify source nodes",
            "sum_processing_ns": sum(p["processing_ns"] for p in partitions)}


def compare_outputs(actual, expected, output_specs, atol, rtol):
    import numpy as np
    require(len(actual) == len(expected) == len(output_specs), "Output count differs")
    comparisons = []
    for observed, reference, spec in zip(actual, expected, output_specs):
        a, b = np.asarray(observed), np.asarray(reference)
        require(list(a.shape) == list(b.shape) == spec["shape"], "Output shape differs from original ONNX")
        require(a.dtype == b.dtype == np.dtype("float32"), "Expected f32 boundary tensors; internal quantization is separately recorded")
        require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite inference output")
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        passed = bool(np.all(delta <= atol + rtol * np.abs(b.astype(np.float64))))
        comparisons.append({"name": spec["source_name"], "passed": passed, "max_absolute_error": float(delta.max()), "atol": atol, "rtol": rtol})
    return comparisons


def environment_record(path):
    record = json.loads(read(path, 65536))
    require(isinstance(record, dict) and set(record) == {"schema_version", "tools_tag", "sdk_linux", "firmware_release", "onnxruntime_version", "image_identity", "pairing_evidence"}, "Environment record keys differ; see TI_ADAPTER.md")
    require(type(record["schema_version"]) is int and record["schema_version"] == 1, "Unsupported environment record")
    require(record["tools_tag"] == TAG and record["sdk_linux"] == SDK and record["firmware_release"] == FIRMWARE, "Only the reviewed 11.0 release pairing is supported; inspect the installed image before choosing tools")
    require(all(isinstance(record[k], str) and 1 <= len(record[k]) <= 4096 for k in record if k != "schema_version"), "Empty or malformed environment declaration")
    return record


def probe(tools_checkout=None):
    from board_probe import collect
    result = collect()
    result["artifact_type"] = "ti-adapter-prerequisites"
    result["adapter_source_sha256"] = sha(read(__file__))
    result["system"] = platform.system()
    result["python_version"] = list(sys.version_info[:3])
    result["soc_environment"] = os.environ.get("SOC")
    result["tools_checkout"] = {"status": "not_selected"}
    if tools_checkout:
        root = Path(tools_checkout).resolve()
        def git(*args):
            command = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, timeout=10)
            require(command.returncode == 0, "Cannot inspect selected TI tools checkout")
            return command.stdout.strip()
        require(root.is_dir(), "TI tools checkout not found")
        commit = git("rev-parse", "HEAD")
        tag_commit = git("rev-parse", TAG + "^{commit}")
        result["tools_checkout"] = {"status": "observed", "path": str(root), "commit": commit,
            "required_tag": TAG, "tag_commit": tag_commit,
            "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}
        for relative in ("examples/osrt_python/ort/onnxrt_ep.py", "examples/osrt_python/common_utils.py", "docs/version_compatibility_table.md"):
            result["tools_checkout"].setdefault("source_files", {})[relative] = file_record(root / relative)
    return result


def check_environment(observed, declared, stage):
    require(type(declared.get("schema_version")) is int and declared["schema_version"] == 1 and declared.get("tools_tag") == TAG and declared.get("sdk_linux") == SDK and declared.get("firmware_release") == FIRMWARE, "Unsupported declared release pairing")
    require(observed.get("system") == "Linux", "TI compilation/execution requires the documented Linux host; native macOS is unsupported")
    require(observed.get("soc_environment") == "am67a", "Set SOC=am67a explicitly for BeagleY-AI / J722S")
    checkout = observed["tools_checkout"]
    require(checkout.get("status") == "observed" and checkout.get("commit") == checkout.get("tag_commit") and not checkout.get("tracked_changes"), "An unmodified checkout at reviewed TI tools tag 11_00_06_00 is required")
    runtime = observed["onnx_runtime"]
    require(runtime.get("status") == "observed", "Installed TI ONNX Runtime could not be observed")
    require(runtime.get("version") == declared["onnxruntime_version"], "Observed ONNX Runtime version differs from environment record")
    provider = COMPILE_EP if stage == "compile" else RUN_EP
    require(provider in runtime.get("providers", []) and CPU_EP in runtime.get("providers", []), "Required TI/CPU execution provider is unavailable; no software substitution")
    if stage == "compile":
        require(observed["machine"] == "x86_64" and observed["python_version"][:2] == [3, 10], "Pinned TI compilation path requires x86_64 Python 3.10")
        release = observed["os_release"].get("text", "")
        require(re.search(r'^ID="?ubuntu"?$', release, re.M) and re.search(r'^VERSION_ID="?22\.04"?$', release, re.M), "Pinned compilation host requires Ubuntu 22.04")
        tools = Path(os.environ.get("TIDL_TOOLS_PATH", "/nonexistent"))
        require(tools.is_dir() and tools.resolve().is_relative_to(Path(checkout["path"]).resolve()), "TIDL_TOOLS_PATH must select existing tools inside the reviewed checkout")
    else:
        require(observed["machine"] == "aarch64" and observed["target_identity_observed"], "Hardware inference requires observed BeagleY-AI identity on aarch64; host emulation is excluded")
        require(any(p.get("state", {}).get("text") == "running" and any(label in p.get("name", {}).get("text", "").lower() for label in ("c7", ".dsp")) for p in observed["remote_processors"]), "No running C7x/DSP remote processor was observed")


def new_directory(path):
    path = Path(path).absolute()
    require(not path.exists(), "Destination already exists; preserve previous evidence")
    require(not any(p.is_symlink() for p in [path] + list(path.parents)), "Output symlink paths are unsupported")
    path.mkdir(parents=True)
    return path


def validated_compilation(root):
    """Bind the receipt to saved calibration tensors, not editable summary lists."""
    root = Path(root)
    receipt = json.loads(read(root / "receipt.json"))
    require(receipt.get("action") == "compile" and receipt.get("status") == "compiled_unverified_on_hardware", "Expected successful compilation receipt")
    records = tree_records(root)
    records.pop("receipt.json", None)
    require(records == receipt.get("evidence_files"), "Compilation evidence file hashes differ")
    data = read(root / "request.json")
    require(sha(data) == receipt.get("request_sha256"), "Compilation request hash differs")
    request = json.loads(data)
    require(request.get("action") == "compile" and sha(read(root / "model.onnx")) == request.get("model_sha256") == receipt.get("model_sha256"), "Compilation model identity differs")
    require(request.get("environment_declared") == receipt.get("environment_declared") and request.get("tensor_bits_requested") == receipt.get("tensor_bits_requested"), "Compilation policy differs from receipt")
    identities = []
    for index, item in enumerate(request["inputs"]):
        require(item["path"] == f"inputs/{index:03d}.npz", "Unsafe calibration snapshot path")
        data = read(root / item["path"])
        identity = input_identity(load_inputs(data, request["model"]["inputs"]))
        require(sha(data) == item["sha256"] and identity == item["tensor_sha256"], "Calibration snapshot differs")
        identities.append(identity)
    require(identities and identities == receipt.get("calibration_tensor_sha256"), "Calibration identities differ from recorded tensor evidence")
    return receipt, identities


def prepare(args, observed, declared):
    """Snapshot bounded inputs before native vendor code sees them."""
    import onnx_frontend
    check_environment(observed, declared, args.action)
    model_bytes = read(args.model)
    require(1 <= len(args.inputs) <= 32, "Provide 1..32 input NPZ files")
    payloads = [read(path) for path in args.inputs]
    require(sum(map(len, payloads)) <= MAX_TOTAL_INPUT, "Input corpus exceeds 64 MiB")
    out = new_directory(args.out)
    args._destination_created = True
    (out / "model.onnx").write_bytes(model_bytes)
    model = onnx_frontend.inspect_model(out / "model.onnx")
    require(model["supported"], "Model is outside the capstone's bounded ONNX frontend: " + json.dumps(model["issues"]))
    identities, inputs = [], []
    (out / "inputs").mkdir()
    for index, data in enumerate(payloads):
        arrays = load_inputs(data, model["inputs"])
        identity = input_identity(arrays)
        require(identity not in identities, "Duplicate input tensor sample")
        identities.append(identity)
        relative = f"inputs/{index:03d}.npz"
        (out / relative).write_bytes(data)
        inputs.append({"path": relative, "sha256": sha(data), "tensor_sha256": identity})
    request = {"schema_version": 1, "action": args.action, "model_sha256": sha(model_bytes), "model": model,
        "inputs": inputs, "environment_observed": observed, "environment_declared": declared,
        "sdk_pairing_status": "operator_recorded_not_independently_verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "atol": args.atol, "rtol": args.rtol, "warmup": args.warmup, "repeats": args.repeats}
    artifacts = out / "vendor-artifacts"
    artifacts.mkdir()
    if args.action == "compile":
        request["tensor_bits_requested"] = args.tensor_bits
        request["calibration_iterations"] = args.calibration_iterations
        request["tidl_tools_path"] = str(Path(os.environ["TIDL_TOOLS_PATH"]).resolve())
    else:
        compiled = Path(args.compiled).resolve()
        receipt, calibration = validated_compilation(compiled)
        require(receipt.get("model_sha256") == request["model_sha256"], "Compiled model hash differs")
        require(receipt.get("environment_declared") == declared, "Compilation/inference environment declarations differ")
        require(not set(identities) & set(calibration), "Held-out inputs overlap calibration tensors")
        recorded = receipt["artifact_files"]
        require(recorded and tree_records(compiled / "vendor-artifacts") == recorded, "Compilation artifact hashes differ")
        # copytree follows no links because the entire source tree was checked.
        shutil.copytree(compiled / "vendor-artifacts", artifacts, dirs_exist_ok=True)
        require(tree_records(artifacts) == recorded, "Copied artifact snapshot differs")
        request["compiled_receipt_sha256"] = sha(read(compiled / "receipt.json"))
        request["tensor_bits_requested"] = receipt["tensor_bits_requested"]
        request["compiled_artifact_files"] = recorded
    write_json(out / "request.json", request)
    return out, request


def session(rt, model_path, provider, options):
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    result = rt.InferenceSession(str(model_path), providers=[provider, CPU_EP] if provider != CPU_EP else [CPU_EP],
        provider_options=[options, {}] if provider != CPU_EP else [{}], sess_options=so)
    require(provider in result.get_providers(), "Requested provider disappeared while creating session")
    result.disable_fallback()
    return result


def worker(out, rt=None):
    """CLI runs this in a child to capture native stdout/stderr and crashes."""
    import numpy as np
    injected_runtime = rt is not None
    out = Path(out)
    request = json.loads(read(out / "request.json"))
    require(request.get("action") in ("compile", "run") and request.get("schema_version") == 1, "Invalid worker request")
    require(request.get("tensor_bits_requested") in (8, 16), "Invalid requested precision")
    require(0 <= request["warmup"] <= 10 and 1 <= request["repeats"] <= 20, "Invalid worker repetition bounds")
    require(all(isinstance(request[k], (int, float)) and math.isfinite(request[k]) and request[k] >= 0 for k in ("atol", "rtol")), "Invalid worker numerical policy")
    if rt is None:
        # The CLI's internal child entry point cannot bypass platform checks.
        observed = probe(request["environment_observed"]["tools_checkout"]["path"])
        check_environment(observed, request["environment_declared"], request["action"])
        import onnxruntime as rt
    require(rt.__version__ == request["environment_declared"]["onnxruntime_version"], "Runtime version changed between parent and child")
    require(sha(read(out / "model.onnx")) == request["model_sha256"], "Model snapshot changed")
    if request["action"] == "run":
        require(request.get("compiled_artifact_files") and tree_records(out / "vendor-artifacts") == request["compiled_artifact_files"], "Copied compilation artifact snapshot changed before native execution")
    corpus = []
    require(1 <= len(request["inputs"]) <= 32, "Invalid worker corpus size")
    for index, item in enumerate(request["inputs"]):
        require(item["path"] == f"inputs/{index:03d}.npz", "Unsafe worker input path")
        data = read(out / item["path"])
        require(sha(data) == item["sha256"], "Input snapshot changed")
        corpus.append(load_inputs(data, request["model"]["inputs"]))
        require(input_identity(corpus[-1]) == item["tensor_sha256"], "Input tensor identity changed")
    options = {"artifacts_folder": str((out / "vendor-artifacts").resolve()), "tensor_bits": request["tensor_bits_requested"], "debug_level": 1}
    if request["action"] == "compile":
        options.update({"tidl_tools_path": request["tidl_tools_path"], "accuracy_level": 1,
            "advanced_options:calibration_frames": len(corpus),
            "advanced_options:calibration_iterations": request["calibration_iterations"]})
    write_json(out / "provider-options.json", options)
    provider = COMPILE_EP if request["action"] == "compile" else RUN_EP
    start = time.perf_counter_ns()
    device = session(rt, out / "model.onnx", provider, options)
    load_ns = time.perf_counter_ns() - start
    runtime_file = Path(rt.__file__)
    result = {"session_load_ns": load_ns, "runtime_version": rt.__version__, "runtime_module": str(runtime_file),
        "runtime_module_file": file_record(runtime_file), "providers": device.get_providers(), "provider_options": options,
        "evidence_origin": "injected_runtime_contract_test" if injected_runtime else "installed_runtime",
        "hardware_runtime_invoked": not injected_runtime and request["action"] == "run"}
    if request["action"] == "compile":
        for inputs in corpus:
            device.run(None, inputs)
        del device
        result["artifact_files"] = tree_records(out / "vendor-artifacts")
        require(result["artifact_files"], "Compilation produced no artifacts; offload is unproven")
        result["status"] = "compiled_unverified_on_hardware"
        result["accelerator_execution_validated"] = False
    else:
        cpu = session(rt, out / "model.onnx", CPU_EP, {})
        output_specs = request["model"]["outputs"]
        samples, accuracy_ok, offload_ok, observed_intervals = [], True, True, set()
        (out / "outputs").mkdir()
        for case, inputs in enumerate(corpus):
            reference = cpu.run(None, inputs)
            np.savez(out / "outputs" / f"{case:03d}-reference.npz", **{f"output_{i}": a for i, a in enumerate(reference)})
            for repeat in range(request["warmup"] + request["repeats"]):
                start = time.perf_counter_ns()
                cpu_output = cpu.run(None, inputs)
                cpu_wall_ns = time.perf_counter_ns() - start
                require(all(c["passed"] for c in compare_outputs(cpu_output, reference, output_specs, 0, 0)), "CPU reference was not repeatable")
                start = time.perf_counter_ns()
                actual = device.run(None, inputs)
                wall_ns = time.perf_counter_ns() - start
                sample = {"case": case, "repeat": repeat, "warmup": repeat < request["warmup"], "host_wall_ns": wall_ns, "cpu_host_wall_ns": cpu_wall_ns}
                sample["comparison"] = compare_outputs(actual, reference, output_specs, request["atol"], request["rtol"])
                accuracy_ok &= all(c["passed"] for c in sample["comparison"])
                raw = device.get_TI_benchmark_data() if hasattr(device, "get_TI_benchmark_data") else None
                try:
                    sample["offload"] = benchmark_proof(raw)
                    fingerprint = sha(json.dumps(sample["offload"]["raw"], sort_keys=True).encode())
                    require(fingerprint not in observed_intervals, "TI benchmark dictionary did not advance; per-invocation execution is unproven")
                    observed_intervals.add(fingerprint)
                except AdapterError as error:
                    offload_ok = False
                    sample["offload"] = {"status": "unverified", "reason": str(error), "raw": raw}
                np.savez(out / "outputs" / f"{case:03d}-{repeat:03d}.npz", **{f"output_{i}": a for i, a in enumerate(actual)})
                samples.append(sample)
                write_json(out / f"sample-{case:03d}-{repeat:03d}.json", sample)
                require(input_identity(inputs) == request["inputs"][case]["tensor_sha256"], "Runtime modified an input tensor")
        result.update({"samples": samples, "numerical_validation_passed": accuracy_ok,
            "vendor_subgraph_execution_observed": offload_ok,
            "accelerator_execution_validated": False,
            "status": "board_run_with_vendor_evidence" if accuracy_ok and offload_ok else "validation_failed",
            "claim_limit": "Positive TI subgraph timestamps and output comparisons are evidence for this run. SDK/firmware pairing, source-node partition mapping, quantized internal precision, and complete board acceptance still need review."})
        # A successful subprocess alone cannot close the hardware acceptance gate.
        result["host_wall_median_ns"] = float(np.median([s["host_wall_ns"] for s in samples if not s["warmup"]]))
        result["host_wall_p95_ns"] = float(np.percentile([s["host_wall_ns"] for s in samples if not s["warmup"]], 95))
        result["cpu_host_wall_median_ns"] = float(np.median([s["cpu_host_wall_ns"] for s in samples if not s["warmup"]]))
        result["measurement_order"] = "CPU then TI for every sample; no randomized ordering or thermal correction"
    write_json(out / "worker-result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "compile", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tools-checkout", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--compiled", type=Path)
    parser.add_argument("--tensor-bits", type=int, choices=(8, 16))
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--calibration-iterations", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    out = None
    try:
        observed = probe(args.tools_checkout)
        if args.action == "probe":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out, observed)
            print("Read-only TI prerequisite observations saved; hardware execution is unverified.")
            return 0 if observed["target_identity_observed"] else 77
        require(all((args.environment, args.model, args.inputs, args.tools_checkout)), "compile/run require --environment --model --inputs --tools-checkout")
        require(args.action != "compile" or args.tensor_bits is not None, "Compilation requires explicit --tensor-bits 8 or 16")
        require(args.action != "run" or (args.compiled is not None and args.tensor_bits is None), "Run requires --compiled; precision comes from that compilation")
        require(args.atol is not None and args.rtol is not None and all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)), "Specify finite nonnegative --atol and --rtol before seeing outputs")
        require(0 <= args.warmup <= 10 and 1 <= args.repeats <= 20 and 1 <= args.calibration_iterations <= 100 and 1 <= args.timeout <= 3600, "Repeat/calibration/timeout limits exceeded")
        try:
            out, request = prepare(args, observed, environment_record(args.environment))
        except Exception as error:
            # prepare owns a fresh destination only after its preflight. Never
            # add diagnostics to an unrelated pre-existing output directory.
            candidate = args.out.absolute()
            if getattr(args, "_destination_created", False):
                write_json(candidate / "failure.json", {"status": "preparation_failed", "message": str(error), "accelerator_execution_validated": False})
            raise
        with (out / "vendor.log").open("xb") as log:
            try:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_worker", str(out)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                exit_code = child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                exit_code = 124
        receipt = {"schema_version": 1, "artifact_type": "ti-onnx-adapter", "action": args.action,
            "model_sha256": request["model_sha256"], "request_sha256": sha(read(out / "request.json")),
            "environment_declared": request["environment_declared"], "sdk_pairing_status": request["sdk_pairing_status"],
            "tensor_bits_requested": request["tensor_bits_requested"], "native_process_exit": exit_code,
            "numerical_policy": {"atol": request["atol"], "rtol": request["rtol"]},
            "accelerator_execution_validated": False, "status": "native_execution_failed"}
        if exit_code == 0 and (out / "worker-result.json").exists():
            result = json.loads(read(out / "worker-result.json"))
            receipt["status"] = result["status"]
            if args.action == "compile":
                receipt["artifact_files"] = result["artifact_files"]
                receipt["calibration_tensor_sha256"] = [i["tensor_sha256"] for i in request["inputs"]]
            else:
                receipt["numerical_validation_passed"] = result["numerical_validation_passed"]
                receipt["vendor_subgraph_execution_observed"] = result["vendor_subgraph_execution_observed"]
        receipt["evidence_files"] = tree_records(out)
        write_json(out / "receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(out / "receipt.json"), "accelerator_execution_validated": False}))
        return 0 if receipt["status"] in ("compiled_unverified_on_hardware", "board_run_with_vendor_evidence") else 1
    except (AdapterError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"TI adapter rejected request: {error}", file=sys.stderr)
        if out is not None and not (out / "failure.json").exists():
            write_json(out / "failure.json", {"status": "error", "message": str(error), "accelerator_execution_validated": False})
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_worker":
        try:
            # Linux-only native child. A runaway vendor logger/compiler cannot
            # grow any individual output (including vendor.log) past 64 MiB.
            import resource
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
            worker(Path(sys.argv[2]))
        except Exception as exc:
            print(f"Native worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(main())
