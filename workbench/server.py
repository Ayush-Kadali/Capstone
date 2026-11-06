"""Local capstone workbench: bounded requests to the authoritative study engine.

No external dependencies. Bind only to loopback; never exposes arbitrary paths,
commands or source execution. ONNX/NPZ uploads are strictly bounded data. Start with --binary build/placementlens.
"""
import argparse
import base64
import binascii
import hashlib
import importlib
from concurrent.futures import ThreadPoolExecutor
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
import tempfile
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_matrix import workload
from study import run_study, validate_bundle as validate_exhaustive, compare_bundles as compare_exhaustive
from bounded_study import validate_bundle as validate_bounded, compare_bundles as compare_bounded
import architecture_model

def validate_bundle(bundle):
    if isinstance(bundle, dict) and bundle.get("artifact_type") == "bounded-placement-study":
        return validate_bounded(bundle)
    return validate_exhaustive(bundle)

def compare_bundles(left, right):
    if left.get("artifact_type") != right.get("artifact_type"):
        raise ValueError("Cannot compare exhaustive and bounded-candidate studies. Choose two studies with the same search scope.")
    return compare_bounded(left, right) if left.get("artifact_type") == "bounded-placement-study" else compare_exhaustive(left, right)


MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_MODEL_REQUEST = 45 * 1024 * 1024
PROFILE_IDS = ("software-device-relu-off", "software-device-relu-on", "software-cnn-relu-on")

def model_capabilities():
    missing = []
    for name in ("numpy", "onnx", "onnxruntime"):
        try:
            importlib.import_module(name)
        except (ImportError, OSError):
            missing.append(name)
    from target_profile import load_profile
    return {"available": not missing, "missing_dependencies": missing,
            "max_file_bytes": MAX_UPLOAD_BYTES,
            "profiles": [load_profile(ROOT / "profiles" / (key + ".json")) for key in PROFILE_IDS],
            "scope": "Bounded ONNX import; host software execution with hypothetical targets. No hardware prediction or validation."}

ESTIMATE_TARGETS = (2, 4)  # bounded comparison width; the backend itself allows up to 8

def architecture_profiles():
    return {"profiles": architecture_model.registered(), "compare_min": ESTIMATE_TARGETS[0],
            "compare_max": ESTIMATE_TARGETS[1], "estimator": architecture_model.VERSION,
            "scope": "Analytical estimates from declared architecture assumptions. No execution, measurement, vendor compilation or hardware validation."}

def decode_upload(name, encoded):
    if not isinstance(name, str) or not 1 <= len(name) <= 255 or any(ord(c) < 32 for c in name):
        raise ValueError("Upload display names must be 1..255 printable characters")
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
        raise ValueError("Each file must be 1 byte..16 MiB")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 upload") from exc
    if not 1 <= len(data) <= MAX_UPLOAD_BYTES:
        raise ValueError("Each file must be 1 byte..16 MiB")
    return data

def bounded_json(path, limit=MAX_UPLOAD_BYTES):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Saved model evidence exceeds size limit")
    return json.loads(raw)



class StudyService:
    def __init__(self, binary, directory, runner=run_study):
        self.binary = Path(binary).resolve()
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capstone-study")
        self.lock = threading.Lock()
        self.estimating = threading.Lock()
        self.jobs = {}
        self.active = False
        self.closed = False
        # Only previously completed, valid study directories are exposed.
        for path in sorted(self.directory.glob("*/bundle.json"))[:100]:
            if re.fullmatch(r"[0-9a-f]{32}", path.parent.name):
                try:
                    bundle = self.read_bundle(path.parent.name)
                    self.jobs[path.parent.name] = {"id": path.parent.name, "state": "complete", "label": bundle["experiment"].get("workbench_label", "Saved study")}
                    if "workbench_model_id" in bundle["experiment"]:
                        self.jobs[path.parent.name]["kind"] = "onnx"
                except (ValueError, OSError):
                    pass

    @staticmethod
    def config(payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        allowed = {"topology", "width", "relu_supported", "fusion", "reuse_buffers", "asynchronous", "mirrored_inputs", "emulate_boundaries", "launch_ms", "boundary_ms", "sync_ms", "local_bytes", "device_bytes", "samples", "seed", "profile_error", "tile_size"}
        if payload.keys() - allowed:
            raise ValueError("Unknown configuration fields")
        c = {"topology": "chain", "width": 8, "relu_supported": False, "fusion": False,
             "reuse_buffers": False, "asynchronous": False, "mirrored_inputs": False, "emulate_boundaries": False, "launch_ms": .2, "boundary_ms": 1.3,
             "device_bytes": 1048576, "samples": 3, "seed": 20261004,
             "profile_error": 0., "tile_size": 0, "sync_ms": .05, "local_bytes": 32768, **payload}
        if c["topology"] not in ("chain", "shared", "residual"):
            raise ValueError("Choose chain, shared or residual")
        for key, options in (("width", (2, 4, 8, 16)), ("samples", (3, 5, 7)), ("tile_size", (0, 2, 4, 8, 16))):
            if type(c[key]) is not int or c[key] not in options:
                raise ValueError(f"Unsupported {key}")
        for key in ("relu_supported", "fusion", "reuse_buffers", "asynchronous", "mirrored_inputs", "emulate_boundaries"):
            if type(c[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key, low, high in (("boundary_ms", 0, 100), ("sync_ms", 0, 100), ("launch_ms", 0, 100), ("profile_error", 0, 1)):
            if type(c[key]) not in (int, float) or not low <= c[key] <= high:
                raise ValueError(f"{key} must be in [{low}, {high}]")
        if c["profile_error"] >= 1:
            raise ValueError("profile_error must be less than 1")
        for key, low, high in (("device_bytes", 0, 16777216), ("local_bytes", 0, 1048576), ("seed", 0, 2**32-1)):
            if type(c[key]) is not int or not low <= c[key] <= high:
                raise ValueError(f"{key} must be an integer in [{low}, {high}]")
        return c

    def submit(self, payload):
        config = self.config(payload)
        with self.lock:
            if self.closed:
                raise RuntimeError("Workbench is shutting down")
            if self.active:
                raise RuntimeError("An experiment is already running; wait for it to finish")
            if len(self.jobs) >= 100:
                raise RuntimeError("This evidence directory has 100 studies; start with a new --out directory")
            key = uuid.uuid4().hex
            self.jobs[key] = {"id": key, "state": "running", "config": config,
                              "label": f"{config['topology']} {config['width']}×{config['width']}"}
            self.active = True
            self.pool.submit(self._run, key, config)
            return dict(self.jobs[key])

    def _run(self, key, config):
        try:
            inputs = self.directory / "inputs"
            inputs.mkdir(exist_ok=True)
            graph = inputs / f"{key}.plg"
            graph.write_text(workload(config["topology"], config["width"], config["seed"]))
            flags = ["--boundary-ms", str(config["boundary_ms"]), "--sync-ms", str(config["sync_ms"]),
                     "--device-bytes", str(config["device_bytes"]), "--local-bytes", str(config["local_bytes"]), "--launch-ms", str(config["launch_ms"])]
            for field, flag in (("relu_supported", "--relu-supported"), ("fusion", "--fuse"), ("reuse_buffers", "--reuse-buffers"), ("asynchronous", "--async"), ("emulate_boundaries", "--emulate-boundaries")):
                if config[field]:
                    flags.append(flag)
            if config["mirrored_inputs"]:
                for name in ("X", "W", "B"):
                    flags += ["--resident-input", name]
            if config["tile_size"]:
                flags += ["--tile-size", str(config["tile_size"])]
            bundle = self.runner(self.binary, graph, self.directory / key, flags=flags,
                                 warmup=1, calibration=3, samples=config["samples"], seed=config["seed"], profile_error=config["profile_error"])
            validate_bundle(bundle)
            # Core/study data remain authoritative; this metadata is presentation identity.
            bundle["experiment"]["workbench_label"] = self.jobs[key]["label"]
            bundle["experiment"]["workbench_configuration"] = config
            temporary = self.directory / key / "bundle.workbench.tmp"
            temporary.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.directory / key / "bundle.json")
            with self.lock:
                self.jobs[key]["state"] = "complete"
        except Exception as error:
            with self.lock:
                self.jobs[key].update(state="failed", error=str(error))
        finally:
            with self.lock:
                self.active = False

    def submit_model(self, payload):
        fields = {"model_name", "model_base64", "inputs_name", "inputs_base64", "profile_id"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("ONNX request requires model, named NPZ inputs and a registered target profile")
        if payload["profile_id"] not in PROFILE_IDS:
            raise ValueError("Unknown target profile")
        data = {kind: decode_upload(payload[kind + "_name"], payload[kind + "_base64"]) for kind in ("model", "inputs")}
        capabilities = model_capabilities()
        if not capabilities["available"]:
            raise ValueError("ONNX exploration unavailable in this server Python environment: install " + ", ".join(capabilities["missing_dependencies"]))
        with self.lock:
            if self.closed or self.active:
                raise RuntimeError("Workbench is shutting down or an experiment is already running")
            if len(self.jobs) >= 100:
                raise RuntimeError("This evidence directory has 100 studies; start with a new --out directory")
            key = uuid.uuid4().hex
            upload = self.directory / key / "upload"
            upload.mkdir(parents=True, exist_ok=False)
            for kind, filename in (("model", "model.onnx"), ("inputs", "inputs.npz")):
                (upload / filename).write_bytes(data[kind])
            metadata = {"schema_version": 1, "id": key, "model_name": payload["model_name"],
                        "inputs_name": payload["inputs_name"], "profile_id": payload["profile_id"],
                        "model_sha256": hashlib.sha256(data["model"]).hexdigest(),
                        "inputs_sha256": hashlib.sha256(data["inputs"]).hexdigest()}
            (upload / "metadata.json").write_text(json.dumps(metadata, allow_nan=False))
            self.jobs[key] = {"id": key, "state": "running", "kind": "onnx", "label": "ONNX: " + payload["model_name"]}
            self.active = True
            self.pool.submit(self._run_model, key, metadata)
            return dict(self.jobs[key])

    def _run_model(self, key, metadata):
        try:
            from explore_model import explore
            folder = self.directory / key
            result = explore(folder / "upload/model.onnx", folder / "upload/inputs.npz",
                             ROOT / "profiles" / (metadata["profile_id"] + ".json"),
                             self.binary, folder / "exploration", measured=True)
            self._seal_model_result(key)
            if result.get("status") != "verified" or result.get("performance_conclusions_valid") is not True:
                reasons = "; ".join(issue.get("message", str(issue)) for issue in result.get("issues", []))
                raise ValueError("Model cannot produce a validated study: " + result.get("status", "unknown") + ". " + reasons)
            self.read_model_result(key)
            bundle = validate_bundle(bounded_json(folder / "exploration/study/bundle.json", 32_000_000))
            if not bundle["conclusions_valid"]:
                raise ValueError("Model study failed correctness validation")
            bundle["experiment"]["workbench_label"] = self.jobs[key]["label"]
            bundle["experiment"]["workbench_model_id"] = key
            bundle["experiment"]["workbench_target_profile"] = metadata["profile_id"]
            bundle["experiment"]["model_sha256"] = metadata["model_sha256"]
            # Publish only after the original ONNX reference gate succeeds.
            (folder / "bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
            with self.lock:
                self.jobs[key]["state"] = "complete"
        except Exception as error:
            if (self.directory / key / "exploration/result.json").is_file():
                try:
                    self._seal_model_result(key)
                except (ValueError, OSError) as seal_error:
                    error = ValueError(str(error) + "; evidence sealing failed: " + str(seal_error))
            with self.lock:
                self.jobs[key].update(state="failed", error=str(error))
        finally:
            with self.lock:
                self.active = False

    def estimate(self, payload, compare=False):
        """Analytical target estimate. scripts/architecture_model.py is authoritative;
        the returned JSON is its output with only `model` replaced by the display name
        (never a server path). The browser renders it and recomputes nothing."""
        target = "profile_ids" if compare else "profile_id"
        if not isinstance(payload, dict) or not any(set(payload) == source | {target, "resident_weights"}
                                                    for source in ({"model_id"}, {"model_name", "model_base64"})):
            raise ValueError(f"Estimate requires model_id or model_name + model_base64, {target} and resident_weights")
        if type(payload["resident_weights"]) is not bool:
            raise ValueError("resident_weights must be boolean")
        known = {p["id"]: p for p in architecture_model.registered()}
        ids = payload[target] if compare else [payload[target]]
        if not isinstance(ids, list) or not all(isinstance(x, str) and x in known for x in ids):
            raise ValueError("Unknown architecture profile")
        low, high = ESTIMATE_TARGETS
        if compare and not (low <= len(ids) <= high and len(set(ids)) == len(ids)):
            raise ValueError(f"Compare {low}..{high} distinct architecture profiles")
        if not self.estimating.acquire(blocking=False):
            raise RuntimeError("An estimate is already running; wait for it to finish")
        try:
            with tempfile.TemporaryDirectory(prefix="capstone-estimate-") as scratch:
                if "model_id" in payload:
                    key = payload["model_id"]
                    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}", key):
                        raise ValueError("Invalid saved model identifier")
                    path = self.directory / key / "upload/model.onnx"
                    try:
                        metadata = bounded_json(self.directory / key / "upload/metadata.json", 4096)
                        with path.open("rb") as stream:
                            raw = stream.read(MAX_UPLOAD_BYTES + 1)
                    except OSError:
                        raise ValueError("Unknown saved model") from None
                    if metadata.get("id") != key or len(raw) > MAX_UPLOAD_BYTES or hashlib.sha256(raw).hexdigest() != metadata.get("model_sha256"):
                        raise ValueError("Saved model provenance mismatch")
                    name = metadata["model_name"]
                else:
                    name = payload["model_name"]
                    path = Path(scratch) / "model.onnx"
                    path.write_bytes(decode_upload(name, payload["model_base64"]))
                profiles = [known[x] for x in ids]
                if compare:
                    result = architecture_model.compare_targets(path, profiles, resident_weights=payload["resident_weights"])
                else:
                    result = architecture_model.analyze(path, profiles[0], resident_weights=payload["resident_weights"], with_sensitivity=True)
        finally:
            self.estimating.release()
        for report in (result, *result.get("reports", [])):
            if "model" in report:
                report["model"] = name
        return result

    @staticmethod
    def _model_evidence_files():
        return ("result.json", "profile.json", "compatibility.json", "target-analysis.json", "import/import.json", "import/graph.plg", "study/bundle.json")

    def _seal_model_result(self, key):
        folder = self.directory / key / "exploration"
        hashes = {}
        for name in self._model_evidence_files():
            if not (folder / name).is_file():
                continue
            limit = 32_000_000 if name == "study/bundle.json" else MAX_UPLOAD_BYTES
            with (folder / name).open("rb") as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ValueError("Model evidence exceeds size limit: " + name)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        path = self.directory / key / "evidence-hashes.json"
        if not path.exists():
            path.write_text(json.dumps({"schema_version": 1, "id": key, "sha256": hashes}))

    def read_model_result(self, key):
        if not re.fullmatch(r"[0-9a-f]{32}", key):
            raise ValueError("Invalid model study identifier")
        folder = self.directory / key
        metadata = bounded_json(folder / "upload/metadata.json", 4096)
        result = bounded_json(folder / "exploration/result.json")
        sealed = bounded_json(folder / "evidence-hashes.json", 4096)
        hashes = sealed.get("sha256", {})
        if sealed.get("schema_version") != 1 or sealed.get("id") != key or not isinstance(hashes, dict) or not {"result.json", "profile.json"} <= set(hashes) or set(hashes) - set(self._model_evidence_files()):
            raise ValueError("Invalid model evidence manifest")
        present = {name for name in self._model_evidence_files() if (folder / "exploration" / name).is_file()}
        if set(hashes) != present:
            raise ValueError("Model evidence manifest does not cover all saved files")
        for name, expected in hashes.items():
            limit = 32_000_000 if name == "study/bundle.json" else MAX_UPLOAD_BYTES
            with (folder / "exploration" / name).open("rb") as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit or hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Model evidence provenance mismatch: " + name)
        if metadata.get("schema_version") != 1 or metadata.get("id") != key or metadata.get("profile_id") not in PROFILE_IDS:
            raise ValueError("Invalid model upload manifest")
        if result.get("schema_version") != 1 or result.get("hardware_validated") is not False:
            raise ValueError("Invalid model exploration receipt")
        from target_profile import load_profile
        profile = load_profile(folder / "exploration/profile.json")
        if profile != result.get("profile") or profile["id"] != metadata["profile_id"]:
            raise ValueError("Target profile provenance mismatch")
        for kind, filename in (("model", "model.onnx"), ("inputs", "inputs.npz")):
            for file in (folder / "upload" / filename, folder / "exploration" / filename):
                with file.open("rb") as stream:
                    raw = stream.read(MAX_UPLOAD_BYTES + 1)
                if len(raw) > MAX_UPLOAD_BYTES or hashlib.sha256(raw).hexdigest() != metadata.get(kind + "_sha256"):
                    raise ValueError("Uploaded model/input provenance mismatch")
            if result.get(kind + "_sha256") != metadata[kind + "_sha256"]:
                raise ValueError("Model receipt provenance mismatch")
        detail = {"upload": metadata, "result": result}
        for name, relative in (("compatibility", "compatibility.json"), ("target_analysis", "target-analysis.json"), ("import_manifest", "import/import.json")):
            path = folder / "exploration" / relative
            if path.exists():
                detail[name] = bounded_json(path)
        if "compatibility" in detail and detail["compatibility"].get("model_sha256") != metadata["model_sha256"]:
            raise ValueError("Compatibility source provenance mismatch")
        if "target_analysis" in detail and (detail["target_analysis"].get("profile_id") != metadata["profile_id"] or detail["target_analysis"].get("hardware_validated") is not False):
            raise ValueError("Target analysis provenance mismatch")
        if result.get("status") == "verified":
            if result.get("numerical_execution_validated") is not True or not result.get("comparisons") or not all(x.get("original_onnx_match") is True and x.get("core_reference_match") is True for x in result["comparisons"]):
                raise ValueError("Model receipt has no valid original-ONNX comparison")
            if any(detail.get("import_manifest", {}).get(kind + "_sha256") != metadata[kind + "_sha256"] for kind in ("model", "inputs")):
                raise ValueError("Imported source provenance mismatch")
            graph = folder / "exploration/import/graph.plg"
            if hashlib.sha256(graph.read_bytes()).hexdigest() != result.get("graph_sha256"):
                raise ValueError("Imported graph provenance mismatch")
            if detail.get("import_manifest", {}).get("graph_sha256") != result.get("graph_sha256"):
                raise ValueError("Import manifest provenance mismatch")
        return detail

    def _read_bundle(self, key):
        if not re.fullmatch(r"[0-9a-f]{32}", key):
            raise ValueError("Invalid study identifier")
        path = self.directory / key / "bundle.json"
        with path.open("rb") as source:
            raw = source.read(32_000_001)
        if len(raw) > 32_000_000:
            raise ValueError("Study exceeds the 32 MB workbench limit")
        bundle = validate_bundle(json.loads(raw))
        if "workbench_model_id" in bundle["experiment"] or (self.directory / key / "upload/metadata.json").exists():
            if bundle["experiment"].get("workbench_model_id") != key:
                raise ValueError("Model study identity mismatch")
            evidence = self.read_model_result(key)
            if evidence["result"].get("status") != "verified" or evidence["result"].get("performance_conclusions_valid") is not True:
                raise ValueError("Model study has no validated external reference")
            published = bounded_json(self.directory / key / "exploration/study/bundle.json", 32_000_000)
            upload = evidence["upload"]
            published["experiment"].update(workbench_label="ONNX: " + upload["model_name"],
                workbench_model_id=key, workbench_target_profile=upload["profile_id"], model_sha256=upload["model_sha256"])
            if published != bundle:
                raise ValueError("Published model study differs from the original-ONNX-validated evidence")
        return bundle, raw

    def read_bundle(self, key):
        return self._read_bundle(key)[0]

    def export(self, key, kind, query):
        """Export only validated saved evidence; never accept a filesystem path.

        Chrome Trace is a representation of the recorded host events. Keep
        timestamps/durations verbatim and retain hardware duration as metadata.
        """
        bundle, raw = self._read_bundle(key)
        if kind == "bundle":
            if query:
                raise ValueError("Bundle export does not accept query parameters")
            return raw, f"capstone-study-{key}.json"
        if kind != "trace" or set(query) != {"plan"} or len(query["plan"]) != 1:
            raise ValueError("Trace export requires exactly one saved plan identifier")
        plan_id = query["plan"][0]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", plan_id):
            raise ValueError("Invalid plan identifier")
        plan = next((record for record in bundle["plans"] if record["id"] == plan_id), None)
        if plan is None:
            raise ValueError("Selected plan does not exist in this saved study")
        trace = {
            "displayTimeUnit": "ms",
            "provenance": "Recorded host command durations; logical lanes do not imply concurrent hardware execution",
            "study_id": key,
            "plan_id": plan_id,
            "placement": plan["placement"],
            "conclusions_valid": bundle["conclusions_valid"],
            "study_artifact_type": bundle["artifact_type"],
            "exhaustive": bundle["artifact_type"] != "bounded-placement-study",
            "oracle_available": bundle["artifact_type"] != "bounded-placement-study",
            "correct": plan["metrics"]["correct"],
            "execution_backend": plan.get("execution_backend"),
            "traceEvents": [
                {"name": f"{event['name']} {event.get('node') or ''}",
                 "cat": "host execution", "ph": "X", "pid": 1, "tid": 1,
                 "ts": event["start_us"], "dur": event["duration_us"],
                 "args": {"command_id": event["command_id"],
                          "logical_backend": event["lane"],
                          "device_us": event.get("device_us"),
                          "event_id": event.get("id"),
                          "command_stable_id": event.get("command_stable_id"),
                          "partition_id": event.get("partition_id")}}
                for event in plan["events"]
            ],
        }
        return trace, f"capstone-trace-{key}-{plan_id}.json"

    def snapshot(self):
        with self.lock:
            return [dict(job) for job in self.jobs.values()]

    def close(self):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=True)


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, value, mime="application/json", attachment=None):
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode() if mime == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if attachment is not None:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def local_request(self):
            port = self.server.server_port
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed:
                self.send(403, {"error": "Local workbench host required"}); return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {f"http://{h}" for h in allowed}:
                self.send(403, {"error": "Same-origin request required"}); return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlsplit(self.path)
            try:
                assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
                if path.path in assets:
                    name, mime = assets[path.path]
                    return self.send(200, (ROOT / "workbench" / name).read_bytes(), mime)
                if path.path == "/api/architectures":
                    return self.send(200, architecture_profiles())
                if path.path == "/api/model-capabilities":
                    return self.send(200, model_capabilities())
                if path.path.startswith("/api/model-results/"):
                    return self.send(200, service.read_model_result(path.path.removeprefix("/api/model-results/")))
                if path.path == "/api/studies":
                    return self.send(200, service.snapshot())
                if path.path.startswith("/api/bundles/"):
                    return self.send(200, service.read_bundle(path.path.removeprefix("/api/bundles/")))
                if path.path.startswith("/api/export/"):
                    match = re.fullmatch(r"/api/export/([0-9a-f]{32})/(bundle|trace)", path.path)
                    if match is None:
                        raise ValueError("Invalid saved-study export route")
                    payload, filename = service.export(match[1], match[2], parse_qs(path.query, keep_blank_values=True))
                    return self.send(200, payload, attachment=filename)
                if path.path == "/api/compare":
                    query = parse_qs(path.query)
                    return self.send(200, compare_bundles(service.read_bundle(query["before"][0]), service.read_bundle(query["after"][0])))
                self.send(404, {"error": "Unknown resource"})
            except (ValueError, OSError, KeyError, IndexError) as error:
                self.send(400, {"error": str(error)})

        def do_HEAD(self):
            # Browser export preflight validates the same route without
            # transferring the JSON body. GET remains a normal attachment.
            self.do_GET()

        def do_POST(self):
            if not self.local_request():
                return
            routes = {"/api/run": (16384, service.submit), "/api/models/run": (MAX_MODEL_REQUEST, service.submit_model),
                      "/api/architectures/estimate": (MAX_MODEL_REQUEST, service.estimate),
                      "/api/architectures/compare": (MAX_MODEL_REQUEST, lambda value: service.estimate(value, compare=True))}
            if self.path not in routes:
                return self.send(404, {"error": "Unknown action"})
            if self.headers.get("Content-Type") != "application/json" or self.headers.get("X-Capstone-Request") != "1":
                return self.send(415, {"error": "Use a JSON workbench request"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                limit, action = routes[self.path]
                if not 0 < length <= limit:
                    raise ValueError(f"Request size must be 1..{limit} bytes")
                payload = json.loads(self.rfile.read(length))
                self.send(200 if self.path.startswith("/api/architectures/") else 202, action(payload))
            except (ValueError, UnicodeDecodeError, OSError) as error:
                self.send(400, {"error": str(error)})
            except RuntimeError as error:
                self.send(409, {"error": str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/placementlens")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/workbench")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error("Build the C++ executable before starting the workbench")
    service = StudyService(args.binary, args.out)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(service))
    print(f"Capstone workbench: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
