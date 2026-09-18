# What this prototype is useful for

PlacementLens lets us inspect a small AI graph all the way from operator placement to executed commands. Change an accelerator capability, transfer cost, or memory budget, then see which assignments remain legal, which copies are inserted, and whether the output still agrees with the original graph.

The immediate user is someone learning or testing graph placement and lowering. A backend developer could use this kind of controlled environment to isolate a residency or capability mistake before implementing a larger integration. We have not demonstrated adoption by such developers, or integration with a production runtime.

## What connects it to real systems

| Mechanism | Implementation here | What it does not model |
|---|---|---|
| Backend support | Per-node opcode and memory checks | Vendor kernel catalogues, layouts, quantization constraints |
| Separate address spaces | Distinct `H:` and `D:` buffers with explicit copies | PCIe, DMA, physical device memory |
| Completion | Pending-device flag; host read rejected before `SYNC` | Asynchronous queues, fences, overlapping work |
| Memory capacity | Actual allocated tensor bytes, retained until run end | Cache, SRAM tiling, buffer reclamation, allocator overhead |
| Graph optimization | Guarded MatMul/ReLU fusion and original node IDs | Arbitrary fusion, native kernel generation, LLVM/MLIR integration |
| Placement | Enumerate legal assignments and score lowered commands | A scalable production partitioner or trained predictor |
| Measurement | Actual command durations on the host | GPU/NPU execution time or hardware speedup |

The virtual device executes arithmetic. It is a functional backend with deliberately restricted capabilities, not a timing-accurate model of a named accelerator.

## Results worth inspecting

These are reproducible observations from the current implementation, not a reconstruction of how it was developed.

1. **Individual support does not guarantee whole-plan feasibility.** With the small demo and an 80-byte device budget, each selected operator can fit locally, but maximal placement retains 112 bytes across the full plan and is rejected. The cost policy can select a feasible plan. Buffers are not freed yet, so this result depends on the retention policy.
2. **Shared inputs need not be copied repeatedly.** `insert_transfers` keeps a set of valid resident handles. The test with two Add operations sharing an input verifies a single host-to-device input copy.
3. **Fusion needs a use check.** In the shared-intermediate graph, both ReLU and Add consume the MatMul output. Replacing the MatMul output with its ReLU result would discard negative values needed by the other consumer. Fusion is refused.
4. **Completion is part of correctness.** A negative test deletes synchronization commands from a valid device plan. The executor rejects the device-to-host read. That tests a protocol rule even though execution is synchronous.
5. **A legal command sequence can still be numerically wrong.** Another test replaces ReLU with an Add using the same input twice. Shape checks pass, but the independent reference comparison fails.
6. **More device assignments can cost more under a declared model.** Keeping the final Add on the CPU raises modeled compute cost from 2.4 to 4.0 ms, but removes three copies, one launch charge, and one synchronization charge. Total modeled cost falls from 10.7001 to 8.1501 ms.

Run the last example with:

```sh
bash scripts/run_demo.sh
python3 scripts/explain_run.py artifacts/segfault/fragmented.json
```

The cost policy wins its own modeled objective by construction: it exhaustively searches it. This is not independent evidence that the model predicts hardware well. Hardware measurements on held-out workloads would be needed for that claim.

## Existing software and our scope

The problem has substantial existing tooling. These sources were checked on 18 September 2026.

| Existing system | Relevant documented feature | Relationship to this prototype |
|---|---|---|
| [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/) | `GetCapability()` identifies supported nodes or subgraphs; providers execute assigned work | Our capability and placement stages illustrate similar responsibilities, without implementing an execution provider |
| [ONNX Runtime I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) | Input/output copies can consume a large part of run time; binding lets users control device residency | This supports the motivation for making copies visible. It does not validate our numerical cost constants |
| [MLIR pass infrastructure](https://mlir.llvm.org/docs/PassManagement/#ir-printing) | IR printing before and after passes, filtering, and change-sensitive output | Before/after snapshots are established practice. Our scope connects snapshots to placements, copies, execution, and correctness in one small inspectable implementation |
| [IREE](https://iree.dev/) | Compiler and runtime encompassing computation, scheduling, hardware backends, debugging and profiling | A much broader implementation to study. We do not claim that our project replaces IREE or supplies capabilities it lacks |

Our contribution at this stage is the working integration and controlled experiments. We have not established a novel placement algorithm or a previously unknown research result. The environment can later support experiments on profile error, boundary costs, and policy quality once timing is grounded in hardware.

## Why the explainable-compilers theme fits

The source is a graph language with typed tensors. Validation and shape inference establish a legal IR. Fusion changes graph structure. Placement attaches a backend to each node. Transfer insertion creates explicit movement and synchronization operations. Lowering materializes allocations and command descriptors. The runtime consumes those descriptors.

The explanation records the representations and rules that produced a plan. It is deterministic compiler evidence, rather than generated prose about an unseen computation. `show_passes.py` displays the recorded differences. `explain_run.py` accounts for the cost change from the actual commands.

This is not the official GPU profitability predictor implementation: it has no source-loop analysis, OpenMP/OpenACC offloading, learned model, or measured GPU labels. It is appropriate to present under the open explainable-compilers theme, subject to the organisers' evaluation.

## Verification beyond the opening example

`python3 scripts/check_matrix.py` builds nine deterministic input graphs: three topologies at matrix widths 2, 8, and 16. Each is tested with missing ReLU support, full support, fusion, constrained memory, and expensive copies. `verify` executes every feasible placement rather than only the cheapest one.

The recorded run checks 45 configurations, 591 executed placements, and 11,553 commands. These counts describe repeated executions under different configurations; they are not 591 independent application benchmarks. The inputs use bounded binary fractions. This tests lowering and functional correctness, not trained-model accuracy, extreme floating-point stability, or real-device throughput.

Raw graphs, reports, output logs, seeds, command lines, platform information, and the binary hash are written under `artifacts/verification/`. Rebuild and rerun to produce evidence for the submitted revision.
