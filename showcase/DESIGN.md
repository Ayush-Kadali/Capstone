# PlacementLens: Friday prototype contract

Target: SegFault submission on Friday 18 September 2026 (user-supplied date; exact cutoff pending). This is a bounded demo milestone within the existing capstone, not a claim that the full roadmap is complete.

## Working path

Small text graph → FP32 rank-two IR → stable topological ordering and shape validation → optional single-consumer MatMul/ReLU fusion → capability-valid placement → explicit allocation/copy/compute/synchronization commands → synchronous CPU/functional-device execution → reference check → report and trace.

No third-party runtime dependencies. C++17 and CMake. A small text format keeps the frontend visible. ONNX, production compiler integration, real-device performance, concurrency, calibrated profiles, and the empirical timing oracle remain later milestones.

## Compiler mechanisms

- Names are unique; every input is defined; cycles and incompatible shapes are rejected.
- Topological ordering preserves input node order when resolving ties.
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
- This prototype retains all buffers until run end; peak allocated device bytes are exact for that policy, not an optimized memory plan.

## Evidence

Unit tests challenge validation, topology, fusion uses/output preservation, lowering residency, memory feasibility, numerics, malformed descriptors and unsupported device work. CLI integration tests validate JSON reports, trace provenance, option errors and deterministic plans. Sanitizers check the executed paths.

The `verify` command executes every feasible placement. The generated verification matrix covers chains, shared intermediates and residual-style graphs under five configurations. The `explain` command emits a report from the same plans and execution results used by the CLI, with Markdown export and JSON evidence.
