# PlacementLens

**An explainable compiler pipeline for heterogeneous AI graph placement.**

SegFault 2026 · Boundary Conditions · Explainable compilers

The runnable prototype shows how a backend capability decision changes graph placement, transfers, buffer allocations, and execution commands. It executes those commands, checks their output, and exports the compiler artifacts that explain the result.

PlacementLens names this compiler hackathon component. The broader capstone has
a wider scope and is not named PlacementLens; this guide covers the component's
bounded first release.

## Run the whole demo

Requirements: C++17 compiler, CMake 3.16+, and Python 3 for the CLI integration tests and report inspector. No third-party C++ libraries. Tested locally on macOS arm64 with AppleClang 17; Linux/Windows have not been verified yet.

From the repository root:

```sh
bash scripts/run_demo.sh
```

This builds and tests the executable, runs five scenes, and saves reports under `artifacts/segfault/`. It does not download a model or require a GPU. The local functional device uses separate buffers and runs its arithmetic on the host.

For a report directly from the executable:

```sh
build/placementlens explain --graph examples/fragmented.plg --report artifacts/segfault/run-report.md --json artifacts/segfault/explained.json
```

The same explanation appears on stdout and in the Markdown file. It includes compiler decisions, alternatives, cost components, the selected commands and actual correctness results. JSON preserves full pass snapshots and tensor output. See [the CLI walkthrough](CLI_WALKTHROUGH.md) for a short interactive demo.

To inspect one compiler transformation:

```sh
python3 scripts/show_passes.py artifacts/segfault/fragmented.json --policy maximal --pass-name InsertTransfersPass
```

To build manually:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
build/placementlens demo --graph examples/fragmented.plg
```

## What the demo proves

The graph is `MatMul → ReLU → Add`. A capability switch controls whether the virtual device supports ReLU. Input tensors are explicit, so the result can be checked by hand: `[[1,14],[7,1]]`.

| Scenario | Selected modeled-optimal placement | What to explain |
|---|---|---|
| All operators supported | `AAA` | One consecutive device region |
| ReLU unsupported | `ACC` | Keeping Add on CPU avoids another device region |
| ReLU unsupported, maximal policy | `ACA` | More offloading introduces extra copies and synchronization |
| ReLU unsupported, fixed copy cost = 5 ms | `CCC` | Transfer overhead outweighs modeled compute savings |
| Full support and optional fusion | `AA` | Guarded MatMul/ReLU fusion removes an intermediate allocation |

`C` means CPU, `A` means virtual accelerator. Each letter corresponds to a node in the compiled graph. An “island” here means a consecutive device run in the fixed sequential schedule; it is not a general graph-connected-component algorithm.

**These are modeled decisions, not measured accelerator speedups.** Compute costs are deliberately simple per-op constants, independent of tensor shape. The model also adds fixed copy cost, bytes/bandwidth, device-run launch cost, and explicit synchronization cost. It is useful for controlled demonstrations and is not calibrated to a real chip.

Actual command durations are measured on the host and exported separately. The runtime performs the matrix arithmetic; it does not sleep to manufacture the model's predicted timings.

## Where the compiler is

```text
.plg source
    ↓ ValidateGraphPass
typed graph IR, shapes, topological order
    ↓ optional MatMulReluFusionPass
    ↓ CapabilityAnalysisPass
legal backend assignments and rejection reasons
    ↓ PlacementPass
node → backend assignments
    ↓ InsertTransfersPass
logical copy/compute/synchronization IR
    ↓ LowerToExecutionPlanPass
explicit allocations + versioned commands
    ↓ checked executor
output + original-graph reference comparison + host trace
```

The small pass driver orchestrates these stages directly; it is not an LLVM/MLIR pass-plugin system. Each selected-policy report contains input/output snapshots, changes, reasons, and consequences for the frontend and selected plans. `verify` records frontend snapshots and transfer/lowering history for each enumerated plan, without the selected-policy capability/placement narratives. Original node IDs survive fusion. The same lowered commands are exported and executed.

The cost policy enumerates at most `2^12` assignments. It rejects capability-invalid or memory-infeasible plans, then chooses the lowest complete-plan modeled cost. This is a bounded modeled oracle, not an empirical measured oracle and not a new search algorithm. CPU-only and maximal-capability policies are comparison baselines. Equal modeled costs keep enumeration order deterministically.

## Commands

```sh
build/placementlens inspect --graph examples/fragmented.plg
build/placementlens compile --graph examples/fragmented.plg --policy maximal
build/placementlens compare --graph examples/fragmented.plg --relu-supported
build/placementlens run --graph examples/branched.plg --relu-supported --fuse
build/placementlens verify --graph examples/branched.plg --relu-supported
build/placementlens explain --graph examples/fragmented.plg
build/placementlens demo --graph examples/fragmented.plg --json artifacts/report.json --trace artifacts/trace.json
build/placementlens --help
```

Create the output directory before saving reports. An infeasible maximal policy is reported and omitted from comparison; the CPU and cost policies still run. Requesting that infeasible policy directly fails with an error.

## Input format

```text
tensor A f32 2 3 1 -2 3 0 4 -1
tensor B f32 3 2 2 1 1 -3 -2 2
tensor D f32 2 2 1 1 1 1
matmul mm A B C
relu activation C R
add bias R D Y
output Y
```

Tensor values are row-major. Node syntax is `operation node_id operands... output`. Comments begin with `#`. Names use letters, digits, and underscores. The prototype supports a single declared output, at most 64 operations (exhaustive search up to 12), at most 64 inputs, and at most 65,536 elements per tensor. Core storage is contiguous rank-two FP32; native CNN parameters describe logical NCHW tensors. Core Add requires equal shapes. The bounded ONNX frontend explicitly materializes constant bias broadcasting and handles Gemm, Conv, pooling and shape operations; see `docs/runtime/ONNX_FRONTEND.md`. Dynamic shapes, implicit casting and source-code execution are rejected. For more than 12 nodes use `verify --search bounded`, which reports retained candidates without claiming exhaustive coverage.

CLI graph files must be valid UTF-8 and at most 16,000,000 bytes. Dimensions are
positive unsigned decimal tokens; signed and fractional tokens are rejected.

## Runtime contract

- `H:name` and `D:name` are distinct owned storage. Residency never silently crosses backends.
- Allocation precedes writes; operands must be initialized; writes cannot overwrite initialized storage.
- Command version, IDs, opcode, shapes, backend, capacity, and device capability are checked.
- Copies into host storage require completion of pending device compute.
- All allocations are retained until the execution ends. Reported peak device allocation is exact for this retention policy; it is not optimized liveness.
- Failures raise an error; the runtime does not change datatypes, automatically replan, or claim successful execution.
- Arithmetic uses FP32 kernels and an independent original-graph evaluator with double accumulation and FP32 results. Numerical checks use absolute and relative tolerance `1e-5`; cancellation-heavy or large reductions may need a more developed error policy later.
- Nonfinite runtime or reference results fail with `E_NUMERIC`, including MatMul
  overflow before fused ReLU. Finite numerical disagreement exits 2; parsing,
  execution and I/O errors exit 1. Failed invocations do not refresh reports, so
  check the exit status before using an older output file.
- Measured host execution covers command processing and final output extraction;
  it excludes graph validation, input-buffer setup, planning and reference checking.

## Evidence and next work

[Validation](VALIDATION.md) · [Demo narration](DEMO_SCRIPT.md) · [Submission document](SUBMISSION.md) · [Engineering notes](ENGINEERING_NOTES.md) · [Design contract](DESIGN.md)

This milestone is a runnable slice of the capstone. Calibrated profiles, a measured oracle, robust placement experiments, concurrency, a real backend, and the full paper are subsequent work.
