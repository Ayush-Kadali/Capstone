#!/usr/bin/env python3
"""Adapter contract tests with injected fake providers. No TI hardware evidence."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ti_onnx_adapter as adapter

try:
    import numpy as np
    import onnx
except ImportError:
    print("SKIP: TI adapter contract tests require numpy and onnx; no vendor runtime required")
    sys.exit(77)


def npz(value=1, dtype="float32", key="x", shape=(1, 2)):
    out = io.BytesIO()
    np.savez(out, **{key: np.full(shape, value, dtype=dtype)})
    return out.getvalue()


def declaration():
    return {"schema_version": 1, "tools_tag": adapter.TAG, "sdk_linux": adapter.SDK,
        "firmware_release": adapter.FIRMWARE, "onnxruntime_version": "fake-contract-runtime",
        "image_identity": "contract fixture, not a board image", "pairing_evidence": "synthetic contract test"}


def observations(root, stage="run"):
    return {"system": "Linux", "machine": "aarch64" if stage == "run" else "x86_64",
        "python_version": [3, 10, 12], "soc_environment": "am67a", "target_identity_observed": True,
        "tools_checkout": {"status": "observed", "commit": "123", "tag_commit": "123", "tracked_changes": False, "path": str(root)},
        "onnx_runtime": {"status": "observed", "version": "fake-contract-runtime", "providers": [adapter.COMPILE_EP, adapter.RUN_EP, adapter.CPU_EP]},
        "remote_processors": [{"state": {"text": "running"}, "name": {"text": "c7x"}}],
        "os_release": {"text": 'ID=ubuntu\nVERSION_ID="22.04"'}}


def timing(step=1):
    return {"ts:subgraph_0_proc_start": step * 100, "ts:subgraph_0_proc_end": step * 100 + 30,
        "ts:subgraph_0_copy_in_start": step * 100 - 10, "ts:subgraph_0_copy_in_end": step * 100,
        "ts:subgraph_0_copy_out_start": step * 100 + 30, "ts:subgraph_0_copy_out_end": step * 100 + 40}


class FakeRuntime:
    """Deliberately fake: tests prove adapter control flow, never offload."""
    __version__ = "fake-contract-runtime"
    __file__ = __file__
    SessionOptions = types.SimpleNamespace
    GraphOptimizationLevel = types.SimpleNamespace(ORT_DISABLE_ALL=0)

    def __init__(self, evidence=True, error=0, stale=False, shape=False, drop=False):
        self.evidence, self.error, self.stale, self.shape, self.drop = evidence, error, stale, shape, drop
        self.calls, self.counter = [], 0

    def InferenceSession(self, model, providers, provider_options, sess_options):
        self.calls.append({"providers": providers, "options": provider_options})
        runtime = self
        provider = providers[0]
        class Session:
            def get_providers(self):
                return [adapter.CPU_EP] if runtime.drop else providers
            def disable_fallback(self):
                pass
            def run(self, output_names, inputs):
                if provider == adapter.COMPILE_EP:
                    folder = Path(provider_options[0]["artifacts_folder"])
                    (folder / "fake-net.bin").write_bytes(b"contract-test-not-a-vendor-artifact")
                result = np.maximum(inputs["x"], 0).astype("float32")
                if provider == adapter.RUN_EP:
                    runtime.counter += 1
                    result += runtime.error
                    if runtime.shape:
                        result = result.reshape(2, 1)
                return [result]
            def get_TI_benchmark_data(self):
                return timing(1 if runtime.stale else runtime.counter) if runtime.evidence else {}
        return Session()


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.spec = [{"source_name": "x", "shape": [1, 2]}]

    def tearDown(self):
        self.temp.cleanup()

    def model(self):
        graph = onnx.helper.make_graph([onnx.helper.make_node("Relu", ["x"], ["y"])], "fixture",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8)
        path = self.root / "model.onnx"
        path.write_bytes(model.SerializeToString())
        return path

    def args(self, action="compile", value=1):
        model = self.model()
        inputs = self.root / f"{action}.npz"
        inputs.write_bytes(npz(value))
        return argparse.Namespace(action=action, model=model, inputs=[inputs], out=self.root / action,
            atol=0.01, rtol=0.01, warmup=1, repeats=2, tensor_bits=8, calibration_iterations=3, compiled=None)

    def make_worker(self, action="run"):
        out = self.root / "worker"
        out.mkdir()
        (out / "inputs").mkdir()
        (out / "vendor-artifacts").mkdir()
        (out / "vendor-artifacts/network.bin").write_bytes(b"synthetic compiled fixture")
        data = npz()
        (out / "inputs/000.npz").write_bytes(data)
        model = self.model().read_bytes()
        (out / "model.onnx").write_bytes(model)
        request = {"schema_version": 1, "action": action, "model_sha256": adapter.sha(model), "tensor_bits_requested": 8,
            "tidl_tools_path": str(self.root), "calibration_iterations": 3,
            "environment_declared": declaration(), "environment_observed": observations(self.root),
            "atol": 0.01, "rtol": 0.01, "warmup": 1, "repeats": 2,
            "inputs": [{"path": "inputs/000.npz", "sha256": adapter.sha(data), "tensor_sha256": adapter.input_identity(adapter.load_inputs(data, self.spec))}],
            "model": {"inputs": self.spec, "outputs": [{"source_name": "y", "shape": [1, 2]}]}}
        request["compiled_artifact_files"] = adapter.tree_records(out / "vendor-artifacts")
        adapter.write_json(out / "request.json", request)
        return out

    def test_exact_named_f32(self):
        self.assertEqual(adapter.load_inputs(npz(), self.spec)["x"].shape, (1, 2))

    def test_input_rejects_wrong_name_dtype_shape_nan(self):
        for data in [npz(key="z"), npz(dtype="float64"), npz(shape=(2, 1)), npz(value=float("nan"))]:
            with self.subTest(), self.assertRaises(adapter.AdapterError):
                adapter.load_inputs(data, self.spec)

    def test_rejects_duplicate_zip_member(self):
        content = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(npz())) as original, zipfile.ZipFile(content, "w") as duplicate:
            payload = original.read("x.npy")
            duplicate.writestr("x.npy", payload)
            duplicate.writestr("x.npy", payload)
        with self.assertRaises(adapter.AdapterError):
            adapter.load_inputs(content.getvalue(), self.spec)

    def test_forged_npy_shape_rejected_before_allocation(self):
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False, "shape": (2**40,)})
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("x.npy", header.getvalue())
        with self.assertRaises(adapter.AdapterError):
            adapter.load_inputs(payload.getvalue(), self.spec)

    def test_input_hash_ignores_zip_container(self):
        data = npz()
        packed = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data)) as original, zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED) as recompressed:
            recompressed.writestr("x.npy", original.read("x.npy"))
        self.assertNotEqual(adapter.sha(data), adapter.sha(packed.getvalue()))
        self.assertEqual(adapter.input_identity(adapter.load_inputs(data, self.spec)), adapter.input_identity(adapter.load_inputs(packed.getvalue(), self.spec)))

    def test_provider_presence_is_not_offload(self):
        for raw in [{}, {"provider": "TIDLExecutionProvider"}, {"ts:run_start": 1, "ts:run_end": 10}, timing(0)]:
            with self.subTest(raw=raw), self.assertRaises(adapter.AdapterError):
                adapter.benchmark_proof(raw)

    def test_actual_interval_extraction(self):
        proof = adapter.benchmark_proof(timing())
        self.assertEqual(proof["partition_count"], 1)
        self.assertEqual(proof["sum_processing_ns"], 30)
        self.assertIn("unknown", proof["source_operator_mapping"])
        self.assertEqual(proof["raw"], timing())

    def test_bad_vendor_interval(self):
        for end in [99, 100, float("nan")]:
            raw = timing()
            raw["ts:subgraph_0_proc_end"] = end
            with self.subTest(end=end), self.assertRaises(adapter.AdapterError):
                adapter.benchmark_proof(raw)

    def test_compile_mac_rejected_before_output(self):
        args = self.args()
        observed = observations(self.root, "compile")
        observed["system"] = "Darwin"
        with self.assertRaisesRegex(adapter.AdapterError, "macOS"):
            adapter.prepare(args, observed, declaration())
        self.assertFalse(args.out.exists())

    def test_host_emulation_cannot_be_board(self):
        observed = observations(self.root)
        observed["machine"] = "x86_64"
        with self.assertRaisesRegex(adapter.AdapterError, "host emulation"):
            adapter.check_environment(observed, declaration(), "run")

    def test_missing_provider_fails(self):
        observed = observations(self.root)
        observed["onnx_runtime"]["providers"] = [adapter.CPU_EP]
        with self.assertRaisesRegex(adapter.AdapterError, "provider"):
            adapter.check_environment(observed, declaration(), "run")

    def test_wrong_version_or_checkout_or_soc_fails(self):
        for category in ("version", "checkout", "soc", "firmware"):
            observed = observations(self.root)
            if category == "version": observed["onnx_runtime"]["version"] = "different"
            if category == "checkout": observed["tools_checkout"]["tracked_changes"] = True
            if category == "soc": observed["soc_environment"] = "am68pa"
            if category == "firmware": observed["remote_processors"] = []
            with self.subTest(category=category), self.assertRaises(adapter.AdapterError):
                adapter.check_environment(observed, declaration(), "run")

    def test_environment_record_does_not_accept_unknown_pair(self):
        path = self.root / "env.json"
        record = declaration()
        record["sdk_linux"] = "12.0"
        adapter.write_json(path, record)
        with self.assertRaisesRegex(adapter.AdapterError, "pairing"):
            adapter.environment_record(path)

    def test_non_overwrite_and_symlink_artifacts(self):
        with self.assertRaises(adapter.AdapterError):
            adapter.new_directory(self.root)
        (self.root / "link").symlink_to(__file__)
        with self.assertRaises(adapter.AdapterError):
            adapter.tree_records(self.root)

    def test_compare_shape_and_numerical_policy(self):
        spec = [{"source_name": "y", "shape": [1, 2]}]
        reference = np.ones((1, 2), dtype="float32")
        self.assertFalse(adapter.compare_outputs([reference + 1], [reference], spec, 0.01, 0.01)[0]["passed"])
        with self.assertRaises(adapter.AdapterError):
            adapter.compare_outputs([reference.reshape(2, 1)], [reference], spec, 0.01, 0.01)

    def test_compile_options_are_explicit_and_no_hardware_claim(self):
        out = self.make_worker("compile")
        runtime = FakeRuntime()
        result = adapter.worker(out, runtime)
        self.assertEqual(result["status"], "compiled_unverified_on_hardware")
        self.assertFalse(result["accelerator_execution_validated"])
        options = runtime.calls[0]["options"][0]
        self.assertEqual(options["tensor_bits"], 8)
        self.assertEqual(options["advanced_options:calibration_frames"], 1)
        self.assertEqual(runtime.calls[0]["providers"], [adapter.COMPILE_EP, adapter.CPU_EP])

    def test_run_contract_saves_every_output_and_raw_proof(self):
        out = self.make_worker()
        result = adapter.worker(out, FakeRuntime())
        self.assertEqual(result["status"], "board_run_with_vendor_evidence")
        self.assertFalse(result["accelerator_execution_validated"])
        self.assertEqual(len(result["samples"]), 3)
        self.assertEqual(len(list((out / "outputs").glob("*.npz"))), 4)
        self.assertTrue(all(s["offload"]["raw"] for s in result["samples"]))

    def test_no_evidence_or_stale_evidence_fails_closed(self):
        for options in [{"evidence": False}, {"stale": True}]:
            with self.subTest(options=options):
                out = self.make_worker()
                result = adapter.worker(out, FakeRuntime(**options))
                self.assertEqual(result["status"], "validation_failed")
                self.assertFalse(result["vendor_subgraph_execution_observed"])
                import shutil
                shutil.rmtree(out)

    def test_numerical_failure_cannot_be_success(self):
        result = adapter.worker(self.make_worker(), FakeRuntime(error=1))
        self.assertEqual(result["status"], "validation_failed")
        self.assertFalse(result["numerical_validation_passed"])

    def test_provider_dropped_at_session_creation_fails(self):
        with self.assertRaisesRegex(adapter.AdapterError, "disappeared"):
            adapter.worker(self.make_worker(), FakeRuntime(drop=True))

    def test_shape_changed_by_vendor_fails(self):
        with self.assertRaisesRegex(adapter.AdapterError, "shape"):
            adapter.worker(self.make_worker(), FakeRuntime(shape=True))

    def test_snapshot_change_rejected(self):
        out = self.make_worker()
        (out / "inputs/000.npz").write_bytes(npz(3))
        with self.assertRaisesRegex(adapter.AdapterError, "changed"):
            adapter.worker(out, FakeRuntime())

    def test_worker_path_traversal_rejected(self):
        out = self.make_worker()
        request = json.loads((out / "request.json").read_text())
        request["inputs"][0]["path"] = "../../other.npz"
        (out / "request.json").write_text(json.dumps(request))
        with self.assertRaisesRegex(adapter.AdapterError, "Unsafe"):
            adapter.worker(out, FakeRuntime())

    def test_calibration_overlap_rejected_after_repacking(self):
        args = self.args("run")
        args.compiled = self.root / "compiled"
        args.compiled.mkdir()
        identity = adapter.input_identity(adapter.load_inputs(args.inputs[0].read_bytes(), self.spec))
        adapter.write_json(args.compiled / "receipt.json", {"action": "compile", "status": "compiled_unverified_on_hardware",
            "model_sha256": adapter.sha(args.model.read_bytes()), "environment_declared": declaration(), "calibration_tensor_sha256": [identity]})
        with patch.object(adapter, "validated_compilation", return_value=(json.loads((args.compiled / "receipt.json").read_text()), [identity])):
            with self.assertRaisesRegex(adapter.AdapterError, "overlap"):
                adapter.prepare(args, observations(self.root), declaration())

    def test_compile_snapshot_has_source_mapping(self):
        args = self.args()
        with patch.dict("os.environ", {"TIDL_TOOLS_PATH": str(self.root)}):
            out, request = adapter.prepare(args, observations(self.root, "compile"), declaration())
        self.assertEqual((out / "model.onnx").read_bytes(), args.model.read_bytes())
        self.assertEqual(request["sdk_pairing_status"], "operator_recorded_not_independently_verified")
        self.assertEqual(request["model"]["nodes"][0]["operator"], "Relu")

    def test_compile_artifact_tamper_rejected(self):
        args = self.args("run", value=2)
        args.compiled = self.root / "compiled"
        artifacts = args.compiled / "vendor-artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "network.bin").write_bytes(b"a")
        recorded = adapter.tree_records(artifacts)
        adapter.write_json(args.compiled / "receipt.json", {"action": "compile", "status": "compiled_unverified_on_hardware",
            "model_sha256": adapter.sha(args.model.read_bytes()), "environment_declared": declaration(),
            "calibration_tensor_sha256": ["another-sample"], "artifact_files": recorded, "tensor_bits_requested": 8})
        (artifacts / "network.bin").write_bytes(b"b")
        with patch.object(adapter, "validated_compilation", return_value=(json.loads((args.compiled / "receipt.json").read_text()), ["another-sample"])):
            with self.assertRaisesRegex(adapter.AdapterError, "hashes"):
                adapter.prepare(args, observations(self.root), declaration())

    def test_worker_artifact_mutation_rejected_before_provider(self):
        out = self.make_worker()
        (out / "vendor-artifacts/network.bin").write_bytes(b"changed")
        runtime = FakeRuntime()
        with self.assertRaisesRegex(adapter.AdapterError, "artifact snapshot"):
            adapter.worker(out, runtime)
        self.assertEqual(runtime.calls, [])

    def test_calibration_summary_tamper_rejected(self):
        out = self.make_worker("compile")
        request = json.loads((out / "request.json").read_text())
        receipt = {"action": "compile", "status": "compiled_unverified_on_hardware",
            "evidence_files": adapter.tree_records(out), "request_sha256": adapter.sha((out / "request.json").read_bytes()),
            "model_sha256": request["model_sha256"], "environment_declared": declaration(),
            "tensor_bits_requested": 8, "calibration_tensor_sha256": ["tampered summary"]}
        adapter.write_json(out / "receipt.json", receipt)
        with self.assertRaisesRegex(adapter.AdapterError, "Calibration identities"):
            adapter.validated_compilation(out)

    def test_boolean_schema_version_rejected(self):
        path = self.root / "env.json"
        record = declaration()
        record["schema_version"] = True
        adapter.write_json(path, record)
        with self.assertRaisesRegex(adapter.AdapterError, "environment record"):
            adapter.environment_record(path)

    def test_valid_compilation_snapshot_to_disjoint_run_preparation(self):
        compiled = self.make_worker("compile")
        result = adapter.worker(compiled, FakeRuntime())
        request = json.loads((compiled / "request.json").read_text())
        receipt = {"action": "compile", "status": "compiled_unverified_on_hardware",
            "evidence_files": adapter.tree_records(compiled), "request_sha256": adapter.sha((compiled / "request.json").read_bytes()),
            "model_sha256": request["model_sha256"], "environment_declared": declaration(),
            "tensor_bits_requested": 8, "calibration_tensor_sha256": [item["tensor_sha256"] for item in request["inputs"]],
            "artifact_files": result["artifact_files"]}
        adapter.write_json(compiled / "receipt.json", receipt)
        self.assertEqual(adapter.validated_compilation(compiled)[1], receipt["calibration_tensor_sha256"])
        args = self.args("run", value=2)
        args.compiled = compiled
        out, request = adapter.prepare(args, observations(self.root), declaration())
        self.assertEqual(adapter.tree_records(out / "vendor-artifacts"), receipt["artifact_files"])
        self.assertEqual(request["tensor_bits_requested"], 8)

    def test_declared_board_dsp_name_is_accepted_as_inventory_only(self):
        observed = observations(self.root)
        observed["remote_processors"][0]["name"]["text"] = "79000000.dsp"
        adapter.check_environment(observed, declaration(), "run")

    def test_cli_prepare_failure_writes_only_fresh_diagnostic(self):
        model = self.model()
        bad_input = self.root / "bad.npz"
        bad_input.write_bytes(npz(dtype="float64"))
        env = self.root / "env.json"
        adapter.write_json(env, declaration())
        out = self.root / "cli-failure"
        argv = ["compile", "--tools-checkout", str(self.root), "--environment", str(env), "--model", str(model),
            "--inputs", str(bad_input), "--tensor-bits", "8", "--atol", "0.1", "--rtol", "0.1", "--out", str(out)]
        with patch.object(adapter, "probe", return_value=observations(self.root, "compile")), patch.dict("os.environ", {"TIDL_TOOLS_PATH": str(self.root)}):
            self.assertEqual(adapter.main(argv), 1)
        self.assertEqual(json.loads((out / "failure.json").read_text())["status"], "preparation_failed")
        self.assertFalse((out / "receipt.json").exists())
        (out / "failure.json").unlink()
        with patch.object(adapter, "probe", return_value=observations(self.root, "compile")), patch.dict("os.environ", {"TIDL_TOOLS_PATH": str(self.root)}):
            self.assertEqual(adapter.main(argv), 1)
        self.assertFalse((out / "failure.json").exists())


if __name__ == "__main__":
    print("TI adapter contract tests: injected fake providers; no SDK or hardware executed.")
    unittest.main()
