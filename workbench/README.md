# Local execution workbench

This is an artifact explorer and bounded experiment client for the capstone. PlacementLens remains the compiler hackathon component name. The browser does not implement placement, cost prediction, correctness checking or experiment comparison.

## Start

From `Capstone_Project`:

```sh
python3 workbench/server.py --binary build/placementlens --out artifacts/workbench --port 8765
```

Open `http://127.0.0.1:8765`. The server binds only to loopback. It needs permission to bind a local socket; a sandbox may prevent that. The interface loads no external fonts, scripts or services. Stop the server with Ctrl+C. An active bounded study is allowed to finish so its evidence can be retained.

The server accepts a fixed configuration vocabulary, generated graph families, and bounded ONNX plus named-NPZ uploads. It never accepts executable source, shell commands or arbitrary file paths from the browser. One worker runs at most one study; a directory is limited to 100 studies. Each job preserves raw reports, logs, configuration, profiles and the authoritative `bundle.json` written by `scripts/study.py`. Failed studies expose their error instead of showing partial output as successful evidence.

## Follow one mechanism

1. Run a chain with device ReLU support disabled. Select **Strict maximal supported** and pin the completed study with **Use as baseline**.
2. Enable ReLU support and run again with the same graph width and seed.
3. Select the activation operation. Its capability record, logical lane, tensor storage, command rows and host events are linked by exported identities.
4. Read the comparison's configuration change, changed placement, added/removed commands and observed metric deltas. Structural changes are exact; the UI does not label timing differences as causal proof.

The strict maximal baseline remains unavailable when its requested placement violates constraints. It is never replaced by the optional **Maximal feasible node count** comparator. Incorrect outputs suppress oracle/regret and comparison performance conclusions; diagnostic observations remain visible.

## Implemented views

- W1: compiled/source graphs, capability reasons, source lineage, schedule-segment identity, linked tensor and command inspection, lowering-pass records.
- W2: graph width/topology, ReLU capability, fusion, software reuse, host worker, mirrored inputs, local scratch/tile size, modeled copy/sync/launch cost, injected profile error, and saved baseline/candidate comparison. All controls execute through the same core study runner.
- W3: exact plan memory/transfer records, measured host-command timeline, separate policy objectives, empirical prediction envelope, p50/p95, regret, correctness gate and explicit unavailable data.
- W4: optional COPY-count prediction captured before running and revealed against the saved policy's exact command count. It is stored only in this browser. No learning proficiency claim is made.
- W5: available `device_us` event durations appear alongside host durations only when present in a saved artifact. A saved Apple M4 OpenCL study identifies that backend explicitly; it is not BeagleY-AI evidence. Vendor internals remain unavailable unless exported. Browser controls run software studies only.

The timeline groups recorded host commands by kind and logical lane. It does not invent asynchronous overlap or hardware scheduling. Aggregate queue delay is not rendered as fabricated individual queue events. Chrome Trace export preserves the original host timestamps on a single host thread.

## Measurement boundaries

The metrics panel distinguishes executor H→D/D→H bytes, input-preparation H→D bytes and input-preparation p50. Mirrored input setup occurs outside the timed executor interval. The displayed setup-plus-executor H→D total is a labeled derived sum. Numerical diagnostic readback bytes remain a separate counter; they are not planner COPY traffic. Reported host end-to-end timing is not interchangeable with executor-only timing.

With **Inject boundary waits (host simulation)** disabled, copy/sync/launch controls change only illustrative model costs. When enabled, the core injects bounded software waits into host execution; requested wait duration is displayed separately. These observations still do not measure physical accelerator transfers. The Additive objective and Robust objective show their actual distinct selection values; the empirical envelope belongs to the boundary estimate and is not a statistical confidence interval.

## Reproducible saved evidence

The service restores validated schema-2 bundles from `<out>/<32 lowercase hexadecimal characters>/bundle.json`. To expose existing local studies, copy their complete artifact directory to a new identifier directory while the server is stopped. Preserve the original evidence; do not rewrite raw reports. Invalid or incompatible bundles are rejected, not automatically migrated.

The canonical studies and the OpenCL study produced elsewhere in this repository are the input evidence, not UI fixtures with invented timings. An exported UI bundle contains the exact saved study plus presentation identity. The JSON/raw links remain usable without the UI.

Downloads use stable, same-origin attachment routes: `/api/export/<study-id>/bundle` returns the saved bundle byte-for-byte, and `/api/export/<study-id>/trace?plan=<plan-id>` returns Chrome Trace built from that saved plan's events. Both study and plan references are validated; paths, filenames and extra query fields cannot be supplied by the browser. A bundle must pass schema validation and the 32 MB limit before export. Filenames contain only fixed labels and validated identifiers.

Export controls are ordinary download anchors with no new window or temporary blob URL. Unavailable policies have no trace link. A parallel HEAD check reports server read/validation errors in the page without replacing the user's current study. GET independently validates the same evidence. Standalone traces preserve source timestamps, host durations, event/command/partition identities, available device durations, backend identity, placement and correctness/conclusion gates. Incorrect traces remain exportable for diagnosis with `correct: false` and `conclusions_valid: false`; they are not presented as successful performance evidence.

## Source map and checks

`server.py` validates bounded requests and calls `study.run_study`. `app.js` renders artifacts and handles linked selection; its small helpers escape all dynamic HTML. `style.css` supplies the responsive layout. `index.html` defines labeled controls and accessible document structure. The Node render regression test executes these actual rendering functions with a minimal DOM stub; it is not a substitute for browser layout testing.

```sh
python3 tests/workbench_tests.py build/placementlens
node --check workbench/app.js
```

API tests include an injected runner, single-job backpressure, failure preservation, invalid identifiers/schema/options, same-origin/Host checks, asset delivery, correctness gating, exact control-to-CLI mapping and a real C++ unsupported-ReLU comparison. Browser QA should separately check keyboard navigation, narrow viewport layout, linked selection, failure display and an actual exported device-event artifact.

## Design decisions

The layout follows a compiler bench: bounded controls on the left, graph/plan/evidence on the right, then a linked operation inspector. The graph's distinct CPU/device rails carry the visual emphasis; ordinary tables carry the evidence.

Tokens are slate ink `#162d46`, cool paper `#f1f5f8`, white surface `#ffffff`, CPU blue `#245bac`, device teal `#0d706f`, and model amber `#8c5409`. Locally available Avenir Next/Segoe UI carries text; Menlo/Consolas is reserved for handles and command source. Provenance has both textual labels and color, keyboard focus remains visible, and motion is unnecessary. The initial generic landing-page recommendation from the design search was rejected in favor of its data-dense dashboard guidance, which fits graph and command inspection.

## Explore your ONNX model

Start the server with a Python environment containing the optional `numpy`, `onnx`, and `onnxruntime` packages. The form reports missing dependencies from that same interpreter; generated teaching graphs remain available without those packages.

1. Under **Bring an ONNX model**, choose the `.onnx` file and an `.npz` containing arrays keyed by the exact ONNX input names. Each file must be nonempty and no larger than 16 MiB. NPZ object arrays/pickle and external ONNX tensors are rejected by the authoritative importer.
2. Select a registered software target profile: `software-device-relu-off`, `software-device-relu-on`, or `software-cnn-relu-on`. Their evidence level is **hypothetical**: these runs do not predict a named physical accelerator or perform vendor compilation.
3. Choose **Explore ONNX model**. The original ONNX model executes through ONNX Runtime CPU, and every core placement and measured observation is compared with that independent reference before a study is published.
4. Inspect the source hash, original input names, target evidence level and source-to-generated-node mappings, then use the existing policy, graph, memory, command, timeline and comparison views.

The current supported subset is defined in `scripts/onnx_frontend.py` and documented in `docs/runtime/ONNX_FRONTEND.md`. Unsupported semantics produce structured import diagnostics with no accepted bundle. An unsuccessful model hides the previously selected study's performance panels to avoid presenting old measurements as this model's results; choosing a saved successful study restores those views.

`POST /api/models/run` accepts exactly `model_name`, `model_base64`, `inputs_name`, `inputs_base64`, and `profile_id`. Display names are never filesystem paths. The JSON request is capped at 45 MiB, encoded lengths are checked before decoding, decoded files are each capped at 16 MiB, and profiles are an exact registry allowlist. Host/Origin and request-marker checks apply. It shares the same single worker and 100-job limit as generated experiments; the browser cannot provide arbitrary target fields, commands or code.

Artifacts use a fresh 32-character identifier with fixed paths: `upload/` preserves source bytes and names; `exploration/` contains the authoritative `explore_model.explore` output; `evidence-hashes.json` binds the source/target manifests; top-level `bundle.json` appears only after the external reference gate. Reads and exports check hashes, source identities, target identity and correctness receipts. Local JSON hashes detect accidental artifact changes; they are not signatures or protection against an attacker rewriting the entire evidence directory.

Optional integration tests create a real dense ONNX graph, validate every placement against original ONNX Runtime, check restart restoration and reject modified provenance, unsupported operations, invalid NPZ data and malformed uploads. They skip only those new real-model tests if optional ONNX dependencies are absent. Existing generated-graph tests continue to run.

## Larger models: measured bounded candidates

Models whose normalized graph exceeds 12 operations use the core's bounded beam search (up to 64 normalized nodes), with 128 unique whole-plan attempts and a beam width of 8 in the model explorer. CPU and feasible strict-maximal baselines are retained. The UI explicitly shows attempts, budget, retained candidates and whether the budget was exhausted. Candidate generation uses illustrative costs; calibration and held-out execution measure the retained candidates afterward.

These results use `bounded-placement-study` schema 1, distinct from exhaustive `placement-study` schema 2. The policy menu replaces **Measured oracle** with **Best observed candidate**. **Gap to best observed candidate** compares only measured retained candidates. Unvisited placements may be faster, and their feasibility is unknown. No global oracle or global regret is asserted.

Graph, ONNX mappings, parameters, buffers, commands, measured timelines and exports remain available. Comparison between two bounded studies shows structural/input/profile/configuration changes and observed selected-policy metric deltas, including explicit candidate-set differences. Those observations establish neither causal speedup nor equivalent task accuracy. Cross-comparison between exhaustive and bounded studies is rejected with a readable scope explanation.

For ONNX studies, the source/target evidence seal now includes the original reference-validated study bundle. The published workbench bundle must match it exactly except for the four documented presentation annotations: label, model ID, target-profile ID and model hash. Removing the model marker or substituting another valid study cannot inherit the original model's numerical receipt. Older ONNX workbench artifacts without this stronger study seal are not automatically migrated or accepted; rerun the source model into a new evidence directory. Generated studies are unaffected.

## Architecture target estimate

The **Architecture target estimate** panel (below the study evidence) runs `scripts/architecture_model.py` on the server for an ONNX model and one or more declared architecture profiles from `profiles/architectures/`. It is an analytical estimate: nothing executes, no NPZ is needed, and no hardware, vendor compiler or measurement is involved. The browser renders the returned JSON and recomputes no placement, cost or ranking.

1. Choose a model: a saved ONNX upload from this evidence directory (its stored bytes must still match the upload hash), or the file chosen under **Bring an ONNX model** (sent as a bounded upload and deleted after the estimate).
2. Tick one target to estimate, or 2–4 targets to compare. Optionally assume weights are already resident on the accelerator.
3. A single estimate shows the selected placement for each normalized node, with its source ONNX node, operator, exact operation/traffic counts, modeled kernel time and any declared-rule fallback reason. It also shows modeled kernels / transfers / syncs, the dominant cost, an **assumption range, not a confidence interval** from the profile's efficiency scenarios, exact moved bytes, synchronizations and peak device memory, the search scope, CPU-only and maximal-supported baselines, and a one-parameter sensitivity table (×0.5, ×2, with the placement reselected each time).
4. A comparison shows the backend's conclusion first. A robust winner is reported only when the efficiency ranges separate and no one-parameter perturbation changes the winner. Otherwise the result reads **Tie** or **no reliable ranking**, and the winner flips are listed. Each target's per-node detail can be expanded.

Labels: **Modeled** marks milliseconds, ranges, dominant cost and sensitivity. **Exact** marks placement letters, bytes, synchronizations and peak memory, which are counted exactly from shapes and declared rules. **Unavailable** marks a model the importer rejected (no placement and no latency) and an infeasible placement. **Not hardware-validated** applies to every estimate. When the open saved study comes from the same ONNX upload, node names in the table select the matching node in the graph view.

API: `GET /api/architectures` lists the registered profiles and the compare bounds. `POST /api/architectures/estimate` accepts exactly `profile_id`, `resident_weights` and either `model_id` (a saved 32-hex upload) or `model_name` + `model_base64`. It returns `architecture_model.analyze(..., with_sensitivity=True)`. `POST /api/architectures/compare` takes `profile_ids` (2–4 distinct registered IDs) instead and returns `architecture_model.compare_targets(...)`. Both reject unknown profile IDs and fields. Both reuse the 16 MiB upload limit, the 45 MiB request cap and the Host/Origin/request-marker checks. Only one estimate runs at a time; another request gets HTTP 409. The only change to the backend output is that `model` holds the display name rather than a server path. The tests check that the endpoint output equals a direct `architecture_model` call.
