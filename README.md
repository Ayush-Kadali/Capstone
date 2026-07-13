# Hardware-Aware Edge AI Runtime

This repository contains the implementation workspace for a capstone project focused on the systems software between an AI computation graph and heterogeneous edge hardware.

The project explores:

- a limited graph frontend and typed internal representation;
- graph-to-command lowering;
- a C++ host runtime with explicit buffers, commands, queues, events, errors, and shutdown behavior;
- a CPU reference backend for numerical validation;
- a functional accelerator simulator with explicit device and local memory;
- deterministic execution traces, profiling, and performance analysis;
- concurrency, synchronization, ownership, and failure handling;
- a bounded comparison with a real GPU, NPU, or accelerator backend after the core runtime is stable.

The mandatory implementation is laptop-first so development and testing do not depend on proprietary hardware or SDK access. Real hardware is treated as a validation backend, not as a prerequisite for the core system.

## Repository Structure

```text
src/         Runtime, graph, backend, simulator, and tooling source code
experiments/ Reproducible benchmarks and validation scripts
assets/      Public diagrams, screenshots, and figures
```

## Engineering Principles

- Correctness is required before performance claims.
- Interfaces must define ownership, lifetime, completion, and failure behavior.
- Measured, modeled, and unsupported hardware claims must remain clearly separated.
- Results must retain configurations, seeds, raw measurements, and traces.
- Core components must include unit, integration, negative, concurrency, and regression tests.
