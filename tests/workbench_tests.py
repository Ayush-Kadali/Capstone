"""Local API/runner contract tests, plus a real C++ capability-toggle study."""
from copy import deepcopy
import base64
import io
from unittest.mock import patch
from types import SimpleNamespace
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from study_tests import fixture
spec = importlib.util.spec_from_file_location("workbench_server", ROOT / "workbench/server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
from study import build_bundle
BINARY = Path(sys.argv.pop(1)).resolve() if len(sys.argv)>1 and not sys.argv[1].startswith("-") else ROOT / "build/placementlens"


def example():
    return build_bundle([fixture(),fixture()], [fixture(),fixture()])


def wait(service, key):
    deadline = time.monotonic()+45
    while time.monotonic()<deadline:
        value = next(j for j in service.snapshot() if j["id"]==key)
        if value["state"]!="running":
            return value
        threading.Event().wait(.01)
    raise AssertionError("study did not finish within 45 seconds")


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.calls=[]
        def runner(binary,graph,out,**options):
            self.calls.append((binary, graph.read_text(), options))
            out.mkdir(parents=True)
            return example()
        self.service=server.StudyService(BINARY, self.temp.name,runner)

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def test_bounded_configuration_rejects_paths_unknown_fields_and_bad_types(self):
        for config in ({"binary":"/bin/sh"},{"width":True},{"width":128},{"samples":30},
                       {"seed":-1},{"device_bytes":2**30},{"relu_supported":1},
                       {"topology":"$(anything)"},{"profile_error":float("nan")},
                       {"profile_error":1},{"boundary_ms":float("inf")},{"tile_size":64},
                       {"sync_ms":-1},{"launch_ms":101},{"emulate_boundaries":1},{"local_bytes":-1},{"mirrored_inputs":"yes"},[]):
            with self.subTest(config=config),self.assertRaises(ValueError):
                self.service.config(config)

    def test_request_maps_exactly_to_authoritative_runner_and_bundle_filename(self):
        config={"topology":"shared","width":4,"samples":5,"seed":99,"profile_error":.3,
                "relu_supported":True,"fusion":True,"reuse_buffers":True,"asynchronous":True,
                "tile_size":2,"device_bytes":1024,"boundary_ms":2,"sync_ms":.2,"local_bytes":48,"mirrored_inputs":True,"emulate_boundaries":True,"launch_ms":.3}
        job=self.service.submit(config)
        self.assertEqual(wait(self.service,job["id"])["state"],"complete")
        binary,text,options=self.calls[0]
        self.assertEqual(binary,BINARY)
        self.assertIn("tensor X f32 4 4",text)
        self.assertIn("add reuse M R S",text)
        self.assertEqual(options["warmup"],1)
        self.assertEqual(options["calibration"],3)
        self.assertEqual(options["samples"],5)
        self.assertEqual(options["seed"],99)
        self.assertEqual(options["profile_error"],.3)
        self.assertEqual(options["flags"],["--boundary-ms","2","--sync-ms","0.2","--device-bytes","1024","--local-bytes","48","--launch-ms","0.3",
            "--relu-supported","--fuse","--reuse-buffers","--async","--emulate-boundaries","--resident-input","X","--resident-input","W","--resident-input","B","--tile-size","2"])
        bundle=self.service.read_bundle(job["id"])
        self.assertEqual(bundle["experiment"]["workbench_configuration"],config)
        self.assertTrue((Path(self.temp.name)/job["id"]/"bundle.json").is_file())
        self.assertFalse((Path(self.temp.name)/job["id"]/"bundle.workbench.tmp").exists())

    def test_runner_failure_is_visible_and_preserves_partial_evidence(self):
        def failure(binary,graph,out,**options):
            out.mkdir();(out/"failure.log").write_text("injected failure evidence")
            raise RuntimeError("engine rejected memory budget")
        self.service.runner=failure
        job=self.service.submit({})
        result=wait(self.service,job["id"])
        self.assertEqual(result["state"],"failed")
        self.assertIn("engine rejected",result["error"])
        self.assertTrue((Path(self.temp.name)/job["id"]/"failure.log").is_file())
        self.assertFalse(self.service.active)

    def test_only_one_study_can_run_and_shutdown_rejects_work(self):
        started=threading.Event(); release=threading.Event()
        def blocked(binary,graph,out,**options):
            started.set();release.wait(5);out.mkdir();return example()
        self.service.runner=blocked
        job=self.service.submit({})
        try:
            self.assertTrue(started.wait(2))
            with self.assertRaisesRegex(RuntimeError,"already running"):
                self.service.submit({})
        finally:
            release.set()
        self.assertEqual(wait(self.service,job["id"])["state"],"complete")
        self.service.close()
        with self.assertRaisesRegex(RuntimeError,"shutting down"):
            self.service.submit({})

    def test_identifiers_and_invalid_schema_cannot_escape_evidence_directory(self):
        for key in ("../secret", "a/../b", "/etc/passwd", "a"*31, "g"*32):
            with self.assertRaises(ValueError):self.service.read_bundle(key)
        key="a"*32;folder=Path(self.temp.name)/key;folder.mkdir()
        (folder/"bundle.json").write_text('{"schema_version": 99}')
        with self.assertRaises(ValueError):self.service.read_bundle(key)

    def test_reopen_restores_completed_valid_bundles(self):
        job=self.service.submit({});wait(self.service,job["id"])
        reopened=server.StudyService(BINARY,self.temp.name)
        try:self.assertEqual(reopened.snapshot()[0]["id"],job["id"])
        finally:reopened.close()

    def test_incorrect_bundle_keeps_conclusion_gate(self):
        bad=fixture();bad["results"][0]["correct"]=False
        def runner(binary,graph,out,**options):
            out.mkdir();return build_bundle([fixture(),fixture()],[bad,fixture()])
        self.service.runner=runner
        job=self.service.submit({});wait(self.service,job["id"])
        bundle=self.service.read_bundle(job["id"])
        self.assertFalse(bundle["conclusions_valid"])
        self.assertIsNone(bundle["policies"]["oracle"]["plan_id"])
        self.assertTrue(all(p["metrics"]["regret_ms"] is None for p in bundle["plans"]))
        failed=next(plan for plan in bundle["plans"] if not plan["metrics"]["correct"])
        trace,_=self.service.export(job["id"],"trace",{"plan":[failed["id"]]})
        self.assertFalse(trace["conclusions_valid"])
        self.assertFalse(trace["correct"])
        self.assertEqual(trace["placement"],failed["placement"])
        self.assertEqual(len(trace["traceEvents"]),len(failed["events"]))

    @staticmethod
    def model_payload(model=b"model", inputs=b"inputs"):
        return {"model_name": "../../display-only.onnx", "inputs_name": "inputs.npz",
                "model_base64": base64.b64encode(model).decode(), "inputs_base64": base64.b64encode(inputs).decode(),
                "profile_id": "software-device-relu-off"}

    def test_model_upload_bounds_names_profiles_and_dependency_diagnostics(self):
        payload = self.model_payload()
        for change in ({"profile_id": "../../custom"}, {"model_base64": "%%%"}, {"inputs_base64": ""},
                       {"model_name": "bad\nname"}, {"inputs_name": "x" * 256}, {"binary": "/bin/sh"},
                       {"model_base64": True}, {"profile_id": ["software-device-relu-off"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.service.submit_model({**payload, **change})
        with patch.object(server, "MAX_UPLOAD_BYTES", 3), self.assertRaises(ValueError):
            self.service.submit_model(payload)
        with patch.object(server, "model_capabilities", return_value={"available": False, "missing_dependencies": ["onnxruntime"]}), self.assertRaisesRegex(ValueError, "server Python environment.*onnxruntime"):
            self.service.submit_model(payload)
        self.assertEqual(self.service.snapshot(), [])
        with patch.object(server.importlib, "import_module", side_effect=ImportError("test")):
            self.assertEqual(server.model_capabilities()["missing_dependencies"], ["numpy", "onnx", "onnxruntime"])

    def test_model_sealing_failure_always_finishes_job_and_size_is_bounded(self):
        def fake_explore(model, inputs, profile, binary, out, **kwargs):
            out.mkdir()
            (out / "result.json").write_text("{}")
            return {"status": "verified", "performance_conclusions_valid": True}
        with patch.object(server, "model_capabilities", return_value={"available": True}), \
             patch.dict(sys.modules, {"explore_model": SimpleNamespace(explore=fake_explore)}), \
             patch.object(self.service, "_seal_model_result", side_effect=ValueError("evidence exceeds size limit")):
            job = self.service.submit_model(self.model_payload())
            done = wait(self.service, job["id"])
            self.assertEqual(done["state"], "failed")
            self.assertIn("evidence sealing failed", done["error"])
            self.assertFalse(self.service.active)
        path = Path(self.temp.name) / job["id"] / "exploration/result.json"
        path.write_bytes(b"x" * 9)
        with patch.object(server, "MAX_UPLOAD_BYTES", 8), self.assertRaisesRegex(ValueError, "size limit"):
            self.service._seal_model_result(job["id"])

    def make_onnx(self, unsupported=False, dense=False):
        try:
            import numpy as np
            import onnx
            import onnxruntime
        except ImportError:
            self.skipTest("Optional ONNX dependencies unavailable in this Python environment")
        h, t = onnx.helper, onnx.TensorProto
        nodes = [h.make_node("Gemm" if dense else "MatMul", ["X", "W", "B"] if dense else ["X", "W"], ["M"], name="dense"),
                 h.make_node("Sigmoid" if unsupported else "Relu", ["M"], ["Y"], name="activation")]
        initializers = [onnx.numpy_helper.from_array(np.array([[1, -1], [2, 1]], dtype=np.float32), "W")]
        if dense:
            initializers.append(onnx.numpy_helper.from_array(np.array([.1, -.2], dtype=np.float32), "B"))
        graph = h.make_graph(nodes, "workbench", [h.make_tensor_value_info("X", t.FLOAT, [2, 2])],
                             [h.make_tensor_value_info("Y", t.FLOAT, [2, 2])], initializers)
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=9)
        stream = io.BytesIO(); np.savez(stream, X=np.array([[1, 2], [-1, 3]], dtype=np.float32))
        return model.SerializeToString(), stream.getvalue()

    def test_model_real_execution_original_reference_and_saved_provenance(self):
        model, inputs = self.make_onnx(dense=True)
        job = self.service.submit_model(self.model_payload(model, inputs))
        done = wait(self.service, job["id"])
        self.assertEqual(done["state"], "complete", done)
        result = self.service.read_model_result(job["id"])
        self.assertEqual(result["result"]["status"], "verified")
        self.assertTrue(result["result"]["performance_conclusions_valid"])
        self.assertFalse(result["result"]["hardware_validated"])
        self.assertEqual(result["upload"]["model_name"], "../../display-only.onnx")
        bundle = self.service.read_bundle(job["id"])
        self.assertTrue(bundle["conclusions_valid"])
        self.assertEqual(bundle["experiment"]["workbench_model_id"], job["id"])
        self.assertTrue(all(x["original_onnx_match"] for x in result["result"]["comparisons"]))
        # Restart only restores models whose external-reference provenance remains valid.
        reopened = server.StudyService(BINARY, self.temp.name)
        self.assertEqual(reopened.snapshot()[0]["id"], job["id"])
        reopened.close()
        published_path = Path(self.temp.name) / job["id"] / "bundle.json"
        published_bytes = published_path.read_bytes()
        unrelated = example()
        unrelated["experiment"].update(bundle["experiment"])
        # A valid but unrelated graph must never inherit this ONNX receipt.
        unrelated["experiment"]["command_version"] = 1
        unrelated["experiment"]["core_report_schema_version"] = 1
        published_path.write_text(json.dumps(unrelated))
        with self.assertRaisesRegex(ValueError, "differs from the original-ONNX"):
            self.service.read_bundle(job["id"])
        stripped = deepcopy(bundle); del stripped["experiment"]["workbench_model_id"]
        published_path.write_text(json.dumps(stripped))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.service.read_bundle(job["id"])
        published_path.write_bytes(published_bytes)
        evidence = Path(self.temp.name) / job["id"] / "exploration"
        for filename in ("profile.json", "target-analysis.json", "import/import.json", "result.json", "study/bundle.json"):
            path = evidence / filename; original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "provenance mismatch"):
                self.service.read_bundle(job["id"])
            path.write_bytes(original)
        (evidence / "inputs.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.service.read_model_result(job["id"])

    def test_large_onnx_uses_bounded_candidates_with_validated_exports_and_comparison(self):
        self.make_onnx()  # Require optional dependencies in this interpreter.
        import numpy as np
        import onnx
        h, t = onnx.helper, onnx.TensorProto
        nodes = [h.make_node("Add", ["X" if i == 0 else "v" + str(i-1), "W"], ["v" + str(i)], name="add" + str(i)) for i in range(29)]
        graph = h.make_graph(nodes, "large-workbench", [h.make_tensor_value_info("X", t.FLOAT, [1, 2])],
              [h.make_tensor_value_info("v28", t.FLOAT, [1, 2])],
              [onnx.numpy_helper.from_array(np.full((1, 2), .125, dtype=np.float32), "W")])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=9)
        stream = io.BytesIO(); np.savez(stream, X=np.array([[1, 2]], dtype=np.float32))
        job = self.service.submit_model(self.model_payload(model.SerializeToString(), stream.getvalue()))
        done = wait(self.service, job["id"])
        self.assertEqual(done["state"], "complete", done)
        bundle = self.service.read_bundle(job["id"])
        self.assertEqual(bundle["artifact_type"], "bounded-placement-study")
        self.assertEqual(len(bundle["graph"]["nodes"]), 29)
        self.assertLessEqual(len(bundle["plans"]), 10)
        self.assertFalse(bundle["oracle_available"])
        self.assertNotIn("oracle", bundle["policies"])
        self.assertIn("best_observed_candidate", bundle["policies"])
        detail = self.service.read_model_result(job["id"])
        self.assertTrue(detail["result"]["performance_conclusions_valid"])
        self.assertIn("bounded", detail["result"]["search_scope"])
        raw, _ = self.service.export(job["id"], "bundle", {})
        self.assertEqual(json.loads(raw), bundle)
        trace, _ = self.service.export(job["id"], "trace", {"plan": [bundle["plans"][0]["id"]]})
        self.assertFalse(trace["exhaustive"])
        self.assertFalse(trace["oracle_available"])
        self.assertEqual(len(trace["traceEvents"]), len(bundle["plans"][0]["commands"]))
        comparison = server.compare_bundles(bundle, bundle)
        self.assertEqual(comparison["artifact_type"], "bounded-placement-comparison")
        self.assertFalse(comparison["oracle_available"])
        self.assertEqual(comparison["changes"], [])
        self.assertTrue(all(value == 0 for value in comparison["policies"]["best_observed_candidate"]["metric_deltas"].values()))
        with self.assertRaisesRegex(ValueError, "same search scope"):
            server.compare_bundles(bundle, example())
        reopened = server.StudyService(BINARY, self.temp.name)
        self.assertEqual(reopened.read_bundle(job["id"])["artifact_type"], "bounded-placement-study")
        reopened.close()

    def test_model_unsupported_and_invalid_inputs_preserve_diagnostics_without_bundle(self):
        model, inputs = self.make_onnx(unsupported=True)
        job = self.service.submit_model(self.model_payload(model, inputs))
        done = wait(self.service, job["id"])
        self.assertEqual(done["state"], "failed")
        detail = self.service.read_model_result(job["id"])
        self.assertEqual(detail["result"]["status"], "unsupported_by_importer")
        self.assertTrue(detail["compatibility"]["issues"])
        self.assertFalse(detail["result"]["performance_conclusions_valid"])
        self.assertFalse((Path(self.temp.name) / job["id"] / "bundle.json").exists())
        model, _ = self.make_onnx()
        import numpy as np
        objects = io.BytesIO(); np.savez(objects, X=np.array([[object()]], dtype=object))
        wrong_names = io.BytesIO(); np.savez(wrong_names, wrong=np.zeros((2, 2), dtype=np.float32))
        for bad in (b"not an NPZ archive", objects.getvalue(), wrong_names.getvalue()):
            job = self.service.submit_model(self.model_payload(model, bad))
            done = wait(self.service, job["id"])
            self.assertEqual(done["state"], "failed")
            self.assertFalse((Path(self.temp.name) / job["id"] / "bundle.json").exists())
            self.assertFalse(self.service.read_model_result(job["id"])["result"]["performance_conclusions_valid"])

    def estimate_payload(self, model, **change):
        return {"model_name": "display.onnx", "model_base64": base64.b64encode(model).decode(),
                "profile_id": "wide-array", "resident_weights": False, **change}

    def direct(self, model, call, *args, **kwargs):
        # The server must return architecture_model's own output; only `model` becomes the display name.
        path = Path(self.temp.name) / "direct.onnx"
        path.write_bytes(model)
        result = call(path, *args, **kwargs)
        for report in (result, *result.get("reports", [])):
            if "model" in report:
                report["model"] = "display.onnx"
        return result

    def test_architecture_estimate_equals_backend_and_validates_inputs(self):
        model, _ = self.make_onnx(dense=True)
        am = server.architecture_model
        result = self.service.estimate(self.estimate_payload(model, resident_weights=True))
        self.assertEqual(result, self.direct(model, am.analyze, am.load("wide-array"), resident_weights=True, with_sensitivity=True))
        self.assertGreater(result["analysis"]["selected"]["estimated_ms"], 0)
        self.assertTrue(result["sensitivity"])
        self.assertFalse(result["hardware_execution_validated"])
        self.assertNotIn(self.temp.name, json.dumps(result))
        for change in ({"profile_id": "software-device-relu-off"}, {"profile_id": "../../profiles/architectures/wide-array"},
                       {"profile_id": ["wide-array"]}, {"resident_weights": 1}, {"binary": "/bin/sh"},
                       {"model_base64": "%%%"}, {"model_name": "bad\nname"}, {"model_id": "0" * 32}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.service.estimate(self.estimate_payload(model, **change))
        with self.service.estimating, self.assertRaisesRegex(RuntimeError, "already running"):
            self.service.estimate(self.estimate_payload(model))

    def test_architecture_compare_bounds_equality_and_unsupported_model(self):
        model, _ = self.make_onnx(dense=True)
        am = server.architecture_model
        ids = ["compact-vector", "wide-array"]
        payload = {"model_name": "display.onnx", "model_base64": base64.b64encode(model).decode(), "resident_weights": False}
        result = self.service.estimate({**payload, "profile_ids": ids}, compare=True)
        self.assertEqual(result, self.direct(model, am.compare_targets, [am.load(x) for x in ids]))
        self.assertIn("conclusion", result)
        self.assertEqual(server.architecture_profiles()["compare_max"], 4)
        for bad in (["wide-array"], ["wide-array", "wide-array"], ["wide-array", "nope"], "wide-array",
                    ["compact-vector", "dense-conv-array", "fragmented-array", "wide-array", "compact-vector"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.service.estimate({**payload, "profile_ids": bad}, compare=True)
        with self.assertRaises(ValueError):  # single-target field on the compare route
            self.service.estimate({**payload, "profile_id": "wide-array"}, compare=True)
        unsupported, _ = self.make_onnx(unsupported=True)
        single = self.service.estimate(self.estimate_payload(unsupported))
        self.assertFalse(single["import_supported"])
        self.assertIsNone(single["analysis"])
        self.assertEqual(single["sensitivity"], [])
        compared = self.service.estimate({**payload, "model_base64": base64.b64encode(unsupported).decode(), "profile_ids": ids}, compare=True)
        self.assertEqual(compared["nominal_order"], [])
        self.assertIsNone(compared["robust_winner"])
        self.assertTrue(compared["conclusion"].startswith("No ranking"))

    def test_architecture_estimate_reads_saved_upload_only_with_matching_hash(self):
        model, _ = self.make_onnx(dense=True)
        key = "a" * 32
        upload = Path(self.temp.name) / key / "upload"
        upload.mkdir(parents=True)
        (upload / "model.onnx").write_bytes(model)
        (upload / "metadata.json").write_text(json.dumps({"id": key, "model_name": "saved.onnx",
            "model_sha256": server.hashlib.sha256(model).hexdigest()}))
        result = self.service.estimate({"model_id": key, "profile_id": "dense-conv-array", "resident_weights": False})
        self.assertEqual(result["model"], "saved.onnx")
        self.assertEqual(result["target"], "dense-conv-array")
        (upload / "model.onnx").write_bytes(model + b"x")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.service.estimate({"model_id": key, "profile_id": "dense-conv-array", "resident_weights": False})
        with self.assertRaises(ValueError):
            self.service.estimate({"model_id": "../" + key, "profile_id": "dense-conv-array", "resident_weights": False})

    def test_architecture_http_routes(self):
        model, _ = self.make_onnx(dense=True)
        http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler_for(self.service))
        thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
        def post(path, body):
            connection = http_client(http.server_port)
            connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json", "X-Capstone-Request": "1"})
            response = connection.getresponse(); value = json.loads(response.read()); connection.close()
            return response.status, value
        try:
            connection = http_client(http.server_port); connection.request("GET", "/api/architectures")
            listing = json.loads(connection.getresponse().read()); connection.close()
            self.assertEqual([p["id"] for p in listing["profiles"]], [p["id"] for p in server.architecture_model.registered()])
            status, value = post("/api/architectures/estimate", self.estimate_payload(model))
            self.assertEqual(status, 200)
            self.assertEqual(value, json.loads(json.dumps(self.direct(model, server.architecture_model.analyze, server.architecture_model.load("wide-array"),
                                                resident_weights=False, with_sensitivity=True))))  # JSON turns tuples into lists
            self.assertEqual(post("/api/architectures/estimate", self.estimate_payload(model, profile_id="custom"))[0], 400)
            self.assertEqual(post("/api/architectures/compare", {**self.estimate_payload(model), "profile_ids": ["wide-array"]})[0], 400)
        finally:
            http.shutdown(); http.server_close()

    def test_model_upload_shares_single_job_slot_and_rejects_closed_service(self):
        model, inputs = self.make_onnx()
        with self.service.lock:
            self.service.active = True
        with self.assertRaises(RuntimeError):
            self.service.submit_model(self.model_payload(model, inputs))
        self.service.active = False
        self.service.close()
        with self.assertRaises(RuntimeError):
            self.service.submit_model(self.model_payload(model, inputs))

    def test_loopback_http_assets_headers_and_api(self):
        http=server.ThreadingHTTPServer(("127.0.0.1",0),server.handler_for(self.service))
        thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
        def request(method,path,body=None,headers=None):
            connection=http_client(http.server_port)
            connection.request(method,path,body,headers or {})
            response=connection.getresponse();value=response.read();result=response.status,dict(response.getheaders()),value
            connection.close();return result
        try:
            for path in ("/","/app.js","/style.css"):
                status,headers,body=request("GET",path)
                self.assertEqual(status,200);self.assertTrue(body)
                self.assertIn("frame-ancestors 'none'",headers["Content-Security-Policy"])
            self.assertEqual(request("GET","/api/studies",headers={"Host":"evil.invalid"})[0],403)
            self.assertEqual(request("GET","/api/studies",headers={"Origin":"https://evil.invalid"})[0],403)
            self.assertEqual(request("POST","/api/run","{}")[0],415)
            self.assertEqual(request("GET","/api/bundles/../secret")[0],400)
            self.assertEqual(request("GET","/api/compare")[0],400)
            self.assertEqual(request("GET","/missing")[0],404)
            headers={"Content-Type":"application/json","X-Capstone-Request":"1"}
            self.assertEqual(request("POST","/api/models/run","{}")[0],415)
            self.assertEqual(request("POST","/api/models/run","{}",{**headers,"Origin":"https://evil.invalid"})[0],403)
            self.assertEqual(request("POST","/api/models/run","{}",{**headers,"Host":"evil.invalid"})[0],403)
            self.assertEqual(request("POST","/api/models/run","{}",{**headers,"Content-Length":str(server.MAX_MODEL_REQUEST+1)})[0],400)
            self.assertEqual(request("POST","/api/models/run",json.dumps(self.model_payload()).replace("bW9kZWw=", "%%%%"),headers)[0],400)
            self.assertEqual(request("POST","/api/models/run","{}",headers)[0],400)
            self.assertEqual(request("GET","/api/model-results/../../escape")[0],400)
            status,_,raw=request("GET","/api/model-capabilities")
            self.assertEqual(status,200)
            self.assertEqual([p["id"] for p in json.loads(raw)["profiles"]],list(server.PROFILE_IDS))
            self.assertEqual(request("POST","/api/run","[]",headers)[0],400)
            self.assertEqual(request("POST","/api/run","x"*16385,headers)[0],400)
            status,_,body=request("POST","/api/run","{}",headers)
            self.assertEqual(status,202);key=json.loads(body)["id"];wait(self.service,key)
            self.assertEqual(request("GET",f"/api/bundles/{key}")[0],200)
            status,_,body=request("GET",f"/api/compare?before={key}&after={key}")
            self.assertEqual(status,200);self.assertEqual(json.loads(body)["changes"],[])
            status,headers,body=request("GET",f"/api/export/{key}/bundle")
            self.assertEqual(status,200)
            self.assertEqual(body,(Path(self.temp.name)/key/"bundle.json").read_bytes())
            self.assertEqual(headers["Content-Disposition"],f'attachment; filename="capstone-study-{key}.json"')
            saved=self.service.read_bundle(key)
            plan=saved["plans"][0]
            route=f"/api/export/{key}/trace?plan={plan['id']}"
            status,headers,body=request("GET",route)
            self.assertEqual(status,200)
            self.assertEqual(headers["Content-Disposition"],f'attachment; filename="capstone-trace-{key}-{plan["id"]}.json"')
            trace=json.loads(body)
            self.assertEqual(trace["plan_id"],plan["id"])
            self.assertEqual(trace["placement"],plan["placement"])
            self.assertEqual(trace["conclusions_valid"],saved["conclusions_valid"])
            self.assertEqual(trace["correct"],plan["metrics"]["correct"])
            self.assertEqual(trace["execution_backend"],plan["execution_backend"])
            self.assertEqual(len(trace["traceEvents"]),len(plan["events"]))
            for exported,original in zip(trace["traceEvents"],plan["events"]):
                self.assertEqual(exported["ts"],original["start_us"])
                self.assertEqual(exported["dur"],original["duration_us"])
                self.assertEqual(exported["name"],f"{original['name']} {original.get('node') or ''}")
                self.assertEqual(exported["args"]["command_id"],original["command_id"])
                self.assertEqual(exported["args"]["logical_backend"],original["lane"])
                self.assertEqual(exported["args"]["device_us"],original.get("device_us"))
                self.assertEqual(exported["args"]["command_stable_id"],original.get("command_stable_id"))
                self.assertEqual(exported["tid"],1)
            status,head,head_body=request("HEAD",route)
            self.assertEqual(status,200);self.assertEqual(head_body,b"")
            self.assertEqual(int(head["Content-Length"]),len(body))
            for invalid in (f"/api/export/{key}/trace",f"/api/export/{key}/trace?plan=",
                f"/api/export/{key}/trace?plan=plan-missing",f"/api/export/{key}/trace?plan=../../secret",
                f"/api/export/{key}/trace?plan={plan['id']}&plan={plan['id']}",
                f"/api/export/{key}/trace?plan={plan['id']}&file=/etc/passwd",
                f"/api/export/{key}/bundle?file=/etc/passwd","/api/export/../bundle",
                "/api/export/"+"0"*32+"/bundle"):
                with self.subTest(route=invalid):
                    status,headers,body=request("GET",invalid)
                    self.assertEqual(status,400)
                    self.assertNotIn("Content-Disposition",headers)
                    self.assertIn("error",json.loads(body))
        finally:
            http.shutdown();http.server_close();thread.join()

    def test_real_core_capability_toggle_has_structural_diff_and_raw_logs(self):
        self.assertTrue(BINARY.is_file(),"current C++ binary required")
        self.service.runner=server.run_study
        before=self.service.submit({"width":2});before_job=wait(self.service,before["id"])
        self.assertEqual(before_job["state"],"complete",before_job.get("error"))
        after=self.service.submit({"width":2,"relu_supported":True});after_job=wait(self.service,after["id"])
        self.assertEqual(after_job["state"],"complete",after_job.get("error"))
        left=self.service.read_bundle(before["id"]);right=self.service.read_bundle(after["id"])
        self.assertTrue(left["conclusions_valid"] and right["conclusions_valid"])
        self.assertEqual(left["policies"]["maximal"]["placement"],"ACA")
        self.assertEqual(right["policies"]["maximal"]["placement"],"AAA")
        diff=server.compare_bundles(left,right)
        policy=diff["policies"]["maximal"]
        self.assertIn("activation",policy["changed_node_ids"])
        self.assertTrue(policy["removed_commands"])
        self.assertLess(policy["metric_deltas"]["command_count"],0)
        self.assertIn("TIMING_DELTA_OBSERVED_NOT_CAUSAL",policy["reason_codes"])
        self.assertEqual(len(list((Path(self.temp.name)/before["id"]).glob("evaluation-*.json"))),3)
        self.assertTrue((Path(self.temp.name)/before["id"]/"invocations.json").is_file())

    def test_rendering_preparation_costs_prediction_objectives_and_provenance(self):
        """Run the real render functions in a minimal DOM stub, not a browser.

        Browser layout/interaction checks are separate. This regression prevents
        preparation costs or the selected objective disappearing from display.
        """
        node = shutil.which("node")
        self.assertIsNotNone(node,"Node required for render contract test")
        program = r'''
const vm = require("node:vm"), fs = require("node:fs");
const bundle = JSON.parse(fs.readFileSync(0,"utf8"));
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      value: id === "policy" ? "additive" : "", selectedOptions:[{text:"policy"}],
      innerHTML:"", textContent:"", hidden:false, attributes:{}, addEventListener() {},
      setAttribute(name,value){this.attributes[name]=value},
      getAttribute(name){return this.attributes[name]??null},
      removeAttribute(name){delete this.attributes[name]}
    });
    return elements.get(id);
  }, addEventListener() {}
};
const context = vm.createContext({document, console, __bundle:bundle,
  localStorage:{getItem(){return null}}, fetch:()=>new Promise(()=>{}), setTimeout() {}});
vm.runInContext(fs.readFileSync(process.argv[1],"utf8"),context);
vm.runInContext('state.bundle=__bundle;state.id="a".repeat(32);render()',context);
const additive = elements.get("metrics").innerHTML;
const bundleHref=elements.get("download").getAttribute("href");
const traceHref=elements.get("trace-export").getAttribute("href");
elements.get("policy").value="robust";
vm.runInContext('render()',context);
const robust=elements.get("metrics").innerHTML;
const unavailable=elements.get("hardware").innerHTML;
vm.runInContext('state.plan.execution_backend="OpenCL GPU: Apple M4";state.plan.events[0].device_us=2;renderTimeline()',context);
const measured=elements.get("hardware").innerHTML;
vm.runInContext('state.bundle.conclusions_valid=false;render()',context);
const failed=elements.get("metrics").innerHTML;
const gate=elements.get("correctness").textContent;
vm.runInContext('state.bundle.policies.maximal={status:"infeasible",plan_id:null,requested_placement:"A",reason_code:"CORE_PLAN_INFEASIBLE"}',context);
elements.get("policy").value="maximal";
vm.runInContext('render()',context);
console.log(JSON.stringify({additive,robust,unavailable,measured,failed,gate,bundleHref,traceHref,
  disabledTrace:elements.get("trace-export").getAttribute("aria-disabled"),
  missingTraceHref:elements.get("trace-export").getAttribute("href"),
  missing:elements.get("policy-reason").textContent,emptyMetrics:elements.get("metrics").innerHTML}));
'''
        bundle=example()
        for plan in bundle["plans"]:
            plan["metrics"].update(preparation_h2d_bytes=48,input_preparation={"p50_ms":.125},injected_boundary_ms=.25)
            plan["prediction"].update(additive_ms=1.234,boundary_ms=2.345,robust_score_ms=3.456)
        completed=subprocess.run([node,"-e",program,str(ROOT/"workbench/app.js")],input=json.dumps(bundle),text=True,capture_output=True,check=True)
        result=json.loads(completed.stdout)
        self.assertIn("Input setup H→D bytes",result["additive"])
        self.assertIn("48 B",result["additive"])
        self.assertIn("0.125 ms",result["additive"])
        self.assertIn("Setup + executor H→D",result["additive"])
        self.assertIn("Additive objective",result["additive"])
        self.assertIn("1.234 ms",result["additive"])
        self.assertIn("Robust objective",result["robust"])
        self.assertIn("3.456 ms",result["robust"])
        self.assertIn("External overlay unavailable",result["unavailable"])
        self.assertIn("Measured device events",result["measured"])
        self.assertIn("Apple M4",result["measured"])
        self.assertIn("Suppressed",result["failed"])
        self.assertIn("Correctness failed",result["gate"])
        self.assertIn("infeasible",result["missing"])
        self.assertIn("No substitute plan",result["missing"])
        self.assertEqual(result["emptyMetrics"],"")
        self.assertEqual(result["bundleHref"],"/api/export/"+"a"*32+"/bundle")
        self.assertRegex(result["traceHref"],r"^/api/export/a{32}/trace\?plan=plan-[CA]+$")
        self.assertEqual(result["disabledTrace"],"true")
        self.assertIsNone(result["missingTraceHref"])

    def test_bounded_renderer_labels_observed_candidates_and_suppresses_global_oracle(self):
        from bounded_study_tests import fixture as candidate_fixture
        import bounded_study
        report = candidate_fixture()
        bundle = bounded_study.build_bundle([report, deepcopy(report)], [deepcopy(report)])
        program = r'''
const vm=require("node:vm"),fs=require("node:fs"),elements=new Map();
const document={getElementById(id){
 if(!elements.has(id))elements.set(id,{value:id==="policy"?"oracle":"",selectedOptions:[{text:"policy"}],innerHTML:"",textContent:"",hidden:false,attributes:{},addEventListener(){},setAttribute(n,v){this.attributes[n]=v},getAttribute(n){return this.attributes[n]??null},removeAttribute(n){delete this.attributes[n]}});
 return elements.get(id);
},addEventListener(){}};
const context=vm.createContext({document,console,__bundle:JSON.parse(fs.readFileSync(0,"utf8")),localStorage:{getItem(){return null}},fetch:()=>new Promise(()=>{}),setTimeout(){}});
vm.runInContext(fs.readFileSync(process.argv[1],"utf8"),context);
vm.runInContext('state.bundle=__bundle;state.id="a".repeat(32);render()',context);
const normal={metrics:elements.get("metrics").innerHTML,options:elements.get("policy").innerHTML,policy:elements.get("policy").value,scope:elements.get("search-scope").textContent,identity:elements.get("identity-summary").textContent,provenance:elements.get("provenance").innerHTML};
vm.runInContext('state.bundle.conclusions_valid=false;render()',context);
console.log(JSON.stringify({normal,failed:elements.get("metrics").innerHTML}));
'''
        node=shutil.which("node"); self.assertIsNotNone(node)
        result=subprocess.run([node,"-e",program,str(ROOT/"workbench/app.js")],input=json.dumps(bundle),capture_output=True,text=True,check=True)
        value=json.loads(result.stdout); normal=value["normal"]
        self.assertEqual(normal["policy"],"best_observed_candidate")
        self.assertIn("Best observed candidate",normal["options"])
        self.assertNotIn("Measured oracle",normal["options"])
        self.assertNotIn("Oracle regret",normal["metrics"])
        self.assertIn("Gap to best observed candidate",normal["metrics"])
        self.assertIn("no global oracle or global regret",normal["scope"])
        self.assertIn("retained candidates",normal["identity"])
        self.assertIn("Rejected attempted candidates",normal["provenance"])
        self.assertIn("Suppressed",value["failed"])

    def test_export_read_errors_remain_visible_without_new_windows_or_blob_urls(self):
        program=r'''
const vm=require("node:vm"),fs=require("node:fs"),elements=new Map();
const document={getElementById(id){
  if(!elements.has(id))elements.set(id,{textContent:"",hidden:false,addEventListener(){}});
  return elements.get(id);
},addEventListener(){}};
const context=vm.createContext({document,console,fetch:()=>new Promise(()=>{}),setTimeout(){}});
vm.runInContext(fs.readFileSync(process.argv[1],"utf8"),context);
let requests=0, prevented=0;
context.fetch=async()=>{requests++;return {ok:false,status:400,json:async()=>({error:"Selected plan does not exist"})}};
context.event={currentTarget:{download:"file.json",getAttribute(name){return name==="aria-disabled"?"false":"/api/export/"+"a".repeat(32)+"/trace?plan=missing"}},preventDefault(){prevented++}};
(async()=>{
 await vm.runInContext("requestExport(event)",context);
 const diagnostic=elements.get("error").textContent;
 context.event.currentTarget.getAttribute=()=>"true";
 await vm.runInContext("requestExport(event)",context);
 console.log(JSON.stringify({diagnostic,requests,prevented}));
})().catch(error=>{console.error(error);process.exitCode=1});
'''
        node=shutil.which("node")
        self.assertIsNotNone(node)
        result=subprocess.run([node,"-e",program,str(ROOT/"workbench/app.js")],capture_output=True,text=True,check=True)
        value=json.loads(result.stdout)
        self.assertIn("Export unavailable: Selected plan does not exist",value["diagnostic"])
        self.assertEqual(value["requests"],2) # HEAD then bounded JSON diagnostic
        self.assertEqual(value["prevented"],1) # only disabled anchors are stopped


def http_client(port):return http.client.HTTPConnection("127.0.0.1",port,timeout=10)


if __name__=="__main__":unittest.main()
