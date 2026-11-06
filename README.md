# Hardware-Aware Runtime for Controlled CPU/Accelerator Placement Analysis

Working title: *Transparent Execution-Provider Runtime and Placement-Robustness Testbed*. B.Tech CSE (AI & Data Science) capstone, MIT World Peace University, Group 7.

An explainable model-to-accelerator exploration system. You give it a small ONNX model and a declared accelerator profile. It shows which operations the accelerator may run, which fall back to the CPU, what data must move and when the host must wait, whether the plan fits in memory, and a modeled latency with a best/worst-case range under stated assumptions (a scenario range, not a confidence interval). It then executes the plan on a functional simulated device and checks the result against the original model in ONNX Runtime.

**PlacementLens** is the name of the C++ compiler/runtime executable only (`build/placementlens`), not of the whole project.

```text
ONNX (opset 13, static f32) → importer → internal IR (.plg) → capability analysis (rules from a profile)
  → placement (CPU-only / maximal / cost; exhaustive ≤ 12 nodes, bounded beam to 64)
  → explicit commands (ALLOC / COPY / COMPUTE / SYNC / FREE)
  → runtime (serial, graph-job async queue, or command-level device queue)
  → functional simulated device → fences → check against ONNX Runtime → reports and traces
```

**Hardware status.** Physical BeagleY-AI setup, deployment and measurement were deliberately deferred for this release. No board measurement exists, and nothing here is a hardware validation. The architecture profiles are hypothetical. Analytical milliseconds are assumption-based scenarios. See [planning/HARDWARE_ROADMAP.md](planning/HARDWARE_ROADMAP.md).

## Install

Requirements: a C++17 compiler, CMake, Python 3 (tested with 3.12+). Node.js is used only by the workbench rendering tests.

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements-onnx.txt      # onnx, onnxruntime, numpy (pinned)
.venv/bin/python scripts/capstone.py build          # cmake configure + build into build/
```

PyTorch and scikit-learn are optional. They are needed only to retrain the digit models (`scripts/train_digits.py`) or re-export examples; the trained ONNX files are already in `examples/models/digits/`.

## One CLI: `scripts/capstone.py`

| Command | What it does |
|---|---|
| `build` | Configure and build the C++ engine into `build/` |
| `targets` | List architecture profiles and older execution profiles, with evidence levels |
| `architecture-targets` | List architecture profiles only |
| `inspect MODEL [--analysis-scope]` | Check ONNX import semantics; analysis scope accepts models too large to execute |
| `estimate MODEL --target ID` | Analytical work, traffic, memory, placement search and sensitivity; no execution |
| `compare-targets MODEL --targets A B ...` | Same model on 2–8 profiles; says whether the ranking is robust |
| `compare-models M1 M2 ... --target ID` | 2–8 model variants on one profile; accuracy is never invented |
| `execute MODEL --inputs NPZ --target ID --out DIR [--runtime command-queue]` | Run the real engine under a profile's rules; check every placement against ONNX Runtime |
| `analyze MODEL --inputs NPZ --target ID --out DIR [--measure]` | Execution-profile study with optional held-out host measurements |
| `explain RUN_DIR` | Read a saved `analyze` result without re-running |
| `demo --out DIR` | The six-step demonstration with a saved transcript |

`--out` directories must not already exist; nothing is overwritten.

## Runnable examples

```sh
PY=.venv/bin/python
# Trained digit classifiers (real data; accuracy evaluated)
$PY scripts/capstone.py estimate examples/models/digits/mlp.onnx --target wide-array
$PY scripts/capstone.py execute examples/models/digits/cnn.onnx \
    --inputs examples/models/digits/cnn_test_chunk0.npz --target fragmented-array --out artifacts/run-cnn
# Untrained edge-CNN architecture variants (standard / depthwise / residual at 16, 64, 160 px)
$PY scripts/capstone.py compare-models examples/models/edge-cnn/edge-standard-160.onnx \
    examples/models/edge-cnn/edge-depthwise-160.onnx --target compact-vector
$PY scripts/capstone.py compare-targets examples/models/edge-cnn/edge-depthwise-64.onnx \
    --targets compact-vector wide-array dense-conv-array
# Full demonstration
$PY scripts/capstone.py demo --out artifacts/demo-NEW
```

The digit models: MLP 64-32-10 (test accuracy 0.9694) and CNN (0.9750) on the 360-image held-out split of scikit-learn's bundled UCI digits. The engine's logits equal ONNX Runtime's on every placement checked. See [docs/runtime/TRAINED_EXAMPLE.md](docs/runtime/TRAINED_EXAMPLE.md). The edge-CNN variants are untrained; their accuracy is not evaluated.

## Tests

```sh
ctest --test-dir build --output-on-failure
.venv/bin/python tests/architecture_model_tests.py --binary build/placementlens -v
```

ONNX-dependent suites report a skip (exit 77) when `onnx` is not installed.

## Documentation

| Topic | Where |
|---|---|
| Acceptance record for this release (authoritative requirement status) | [docs/runtime/COLLEGE_RELEASE_ACCEPTANCE.md](docs/runtime/COLLEGE_RELEASE_ACCEPTANCE.md) |
| Learning guide (code-linked modules, exercises, defense prep, self-assessment) | [showcase/ENGINEERING_NOTES.md](showcase/ENGINEERING_NOTES.md); HTML version in [showcase/html-guide/](showcase/html-guide/index.html) |
| Analytical model: formulas, worked examples, limits | [docs/runtime/ANALYTICAL_MODEL.md](docs/runtime/ANALYTICAL_MODEL.md) |
| Demonstration | [docs/runtime/DEMONSTRATION.md](docs/runtime/DEMONSTRATION.md) |
| ONNX importer contract | [docs/runtime/ONNX_FRONTEND.md](docs/runtime/ONNX_FRONTEND.md) |
| Command-level queue | [docs/runtime/COMMAND_QUEUE.md](docs/runtime/COMMAND_QUEUE.md) |
| October completion matrix (historical; open hardware rows) | [docs/runtime/COMPLETION_MATRIX.md](docs/runtime/COMPLETION_MATRIX.md) |
| Scope changes | [planning/SCOPE_CHANGELOG.md](planning/SCOPE_CHANGELOG.md) |
| Future hardware work | [planning/HARDWARE_ROADMAP.md](planning/HARDWARE_ROADMAP.md), [docs/runtime/BOARD_VALIDATION.md](docs/runtime/BOARD_VALIDATION.md) |
| Earlier material (October study, workbench, OpenCL run, hackathon demo) | [docs/runtime/](docs/runtime/), [showcase/README.md](showcase/README.md), [workbench/README.md](workbench/README.md) |

## Source and local evidence

- **Source:** `src/`, `include/`, `scripts/`, `tests/`, `workbench/`, `examples/` (including `examples/models/`), `docs/`, `planning/`, `showcase/`, `CMakeLists.txt`, `requirements-onnx.txt`.
- **Local evidence, not source:** `artifacts/` is git-ignored. Runs, transcripts and measurement bundles written there stay on the machine that produced them unless added explicitly. Documents that cite `artifacts/...` refer to that local evidence.
- **Build outputs:** `build/` and `.venv/` are git-ignored.
- **Check before packaging:** `profiles/` (including `profiles/architectures/*.json`, which the engine and estimator need) is not ignored but is not yet tracked; add it before a source package is made.

## Engineering rules

- Correctness before performance claims.
- Every number is labelled: exact, modeled, measured, evaluated or unavailable.
- Interfaces state ownership, lifetime, completion and failure behaviour.
- Simulator agreement is not hardware evidence.

## License

No license chosen yet — owner decision.
