# PlacementLens: first-release contract

Release boundary: 4 October 2026 hardening of the existing September prototype.
See `RELEASE_CHECKLIST.md`. This is a bounded milestone within the existing
capstone, not a claim that the full roadmap is complete.

## Working path

Small text graph → FP32 rank-two IR → stable topological ordering and shape validation → optional single-consumer MatMul/ReLU fusion → capability-valid placement → explicit allocation/copy/compute/synchronization commands → synchronous CPU/functional-device execution → reference check → report and trace.

No third-party runtime dependencies. C++17 and CMake. A small text format keeps the frontend visible. ONNX, production compiler integration, real-device performance, concurrency, calibrated profiles, and the empirical timing oracle remain later milestones.

## Compiler mechanisms

- Names are unique; every input is defined; cycles and incompatible shapes are rejected.
- Topological ordering scans nodes in source order, scheduling each ready node,
  then repeats for earlier blocked nodes. It is deterministic; it does not restart
  the scan after every scheduled node or optimize the execution schedule.
- Fusion is opt-in, requires one use of the MatMul output, must not eliminate the graph output, and requires the backend's ReLU support. It does not fuse through a disabled capability.
- Lowering materializes distinct host/device buffers and transfers using valid residency. The executor consumes this same plan.
- The pass driver records ValidateGraphPass, optional MatMulReluFusionPass, CapabilityAnalysisPass, PlacementPass, InsertTransfersPass, and LowerToExecutionPlanPass. Transfer insertion produces a distinct logical transfer IR; lowering then adds concrete allocations and versioned command IDs. JSON includes each pass's input/output IR, changes, reasons, and consequences.

## Placement comparison

- CPU-only and maximal supported-node assignment are baselines.
- Bounded modeled-optimal placement enumerates at most 12 nodes; it minimizes a declared illustrative complete-plan cost over the legal lowered assignments.
- It is explicitly a modeled oracle, not the measured empirical oracle from the full research plan and not a new placement algorithm.
- All selected comparison plans actually execute and pass a CPU reference check. Wall-clock functional execution time is recorded separately and does not rank projected accelerator costs.

## Runtime invariants

- Host and simulated device buffers are distinct owned vectors; a command cannot access the wrong address space implicitly.
- Every allocation is live before access, handles are unique, ranges/shapes agree, and device allocation stays within configured capacity.
- Device compute validates descriptor version, capability, shapes and storage before execution.
- Synchronous completion is explicit. Failed execution does not produce successful output.
- The public executor validates source shape/storage before kernel access. Analysis,
  lowering and reference functions require an unmodified validated graph (see the
  header). Runtime or reference nonfinite arithmetic raises `E_NUMERIC`; fusion
  cannot mask a nonfinite MatMul result with ReLU.
- This prototype retains all buffers until run end; peak allocated device bytes are exact for that policy, not an optimized memory plan.

## Evidence

Unit tests challenge validation, topology, fusion uses/output preservation, lowering residency, memory feasibility, numerics, malformed descriptors and unsupported device work. CLI integration tests validate JSON reports, trace provenance, option errors and deterministic plans. Sanitizers check the executed paths.

The `verify` command executes every feasible placement. The generated verification matrix covers chains, shared intermediates and residual-style graphs under five configurations. The `explain` command emits a report from the same plans and execution results used by the CLI, with Markdown export and JSON evidence.

`verify` records frontend plus transfer/lowering pass snapshots per enumerated
plan; selected-policy reports additionally include capability and placement pass
records. The search ranking records every feasible assignment's cost and a total
count of rejected assignments; it does not retain a rejection reason per mask.
Independent Python differential checks use exact rational arithmetic and
graph-edge residency accounting, and compare repeated reports after removing
only measured host timing. Timestamped traces and host durations are not deterministic.
