# Hardware-Aware Edge AI Runtime

## Runnable PlacementLens prototype

PlacementLens compiles small matrix graphs into explicit CPU/device command plans, executes them using a functional software device, and explains its decisions. It is the current runnable part of the capstone.

```sh
bash scripts/run_demo.sh
build/placementlens explain --graph examples/fragmented.plg --report artifacts/segfault/run-report.md
```

The report shows validation, fusion decisions, capability checks, placement alternatives, transfer insertion, buffer allocations, cost components, and correctness results. `verify` executes every feasible placement. Tests include a 45-case matrix with 591 placement executions across chain, shared-intermediate, and residual-style graphs.

See [the demo guide](showcase/README.md), [submission document](showcase/SUBMISSION.md), and [validation record](showcase/VALIDATION.md). Accelerator costs in this version are illustrative; the virtual device executes in software on the host.

For a terminal walkthrough, see [CLI examples](showcase/CLI_WALKTHROUGH.md). [Engineering notes](showcase/ENGINEERING_NOTES.md) connect the implementation to real runtime concerns and explain its limits relative to existing tools.

## Broader capstone

This repository contains the implementation workspace for a capstone project focused on the systems software between an AI computation graph and heterogeneous edge hardware.

The broader roadmap includes the following work; this list is not a claim that every item is implemented in the prototype:

- a limited graph frontend and typed internal representation;
- graph-to-command lowering;
- a C++ host runtime with explicit buffers, commands, queues, events, errors, and shutdown behavior;
- a CPU reference backend for numerical validation;
- a functional accelerator simulator with explicit device and local memory;
- deterministic execution traces, profiling, and performance analysis;
- an interactive execution workbench that links graph placement, tensors, memory,
  commands, timelines, and causal performance changes;
- concurrency, synchronization, ownership, and failure handling;
- a bounded comparison with a real GPU, NPU, or accelerator backend after the core runtime is stable.

The mandatory implementation is laptop-first so development and testing do not depend on proprietary hardware or SDK access. Real hardware is treated as a validation backend, not as a prerequisite for the core system.

## Repository Structure

```text
src/         Runtime, graph, backend, simulator, and tooling source code
experiments/ Reproducible benchmarks and validation scripts
assets/      Public diagrams, screenshots, and figures
```

The workbench is built only after the headless graph, plan, trace, and measurement
contracts are stable.

## Engineering Principles

- Correctness is required before performance claims.
- Interfaces must define ownership, lifetime, completion, and failure behavior.
- Measured, modeled, and unsupported hardware claims must remain clearly separated.
- Results must retain configurations, seeds, raw measurements, and traces.
- Core components must include unit, integration, negative, concurrency, and regression tests.
- Every displayed metric must identify whether it is exact, measured, modeled,
  derived, or unavailable.
