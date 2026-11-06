# Capstone engineering notes and learning guide

This guide explains an explainable model-to-accelerator exploration system (faculty title: *Hardware-Aware Runtime for Controlled CPU/Accelerator Placement Analysis*; working title: *Transparent Execution-Provider Runtime and Placement-Robustness Testbed*). It imports a small ONNX model, decides CPU/accelerator placement under a declared architecture profile, lowers the plan to explicit commands, executes them on a functional simulated device, checks the result against ONNX Runtime and estimates latency analytically. **PlacementLens** is only the C++ compiler/runtime executable (`build/placementlens`), not the whole project.

**For the viva, start here:** [the 7-day route](#a-7-day-study-and-oral-defense-route), [the 30-second introduction](#a-30-second-introduction), [the viva questions](#likely-viva-and-engineering-questions) and [the self-assessment checklist](#self-assessment-checklist-do-this-yourself). Modules 20–24 cover the November 2026 release (analytical model, profiles as engine rules, analysis scope, trained example, command-level queue); modules 1–19 build up the engine from a small `.plg` example. Physical hardware is deferred for this release ([HARDWARE_ROADMAP.md](../planning/HARDWARE_ROADMAP.md)). Reading the guide prepares you to explain and change the code; it is not evidence that you understand it.

Background material from earlier stages: [ONNX import](../docs/runtime/ONNX_FRONTEND.md), [simulator target profiles](../docs/runtime/TARGET_PROFILES.md), the [code walkthrough](../docs/segfault_2026/ONE_DAY_CODE_WALKTHROUGH.md), the [CLI walkthrough](CLI_WALKTHROUGH.md) and the [validation record](VALIDATION.md).

## Begin with the problem

An AI computation can be written as operations that consume and produce arrays of numbers. A processor may support only some operations. Even when an operation is supported and its arithmetic is cheap, moving its inputs and outputs can make that assignment expensive. The software must decide where to run work and arrange the storage and ordering required to produce the right answer.

For a firmware reader, a command plan resembles a list of descriptors with explicit storage, operand, and completion requirements. Here the executor is an ordinary host function. There is no physical device driver, interrupt handler, DMA engine, or register interface hidden behind it.

A **tensor** is an array plus its dimensions. The storage descriptors use two-dimensional contiguous single-precision floating-point arrays (`f32`). The ONNX frontend also accepts logical NCHW rank-4 tensors, packs their storage as `[N,C*H*W]`, and passes spatial dimensions to native CNN commands; it does not replace convolution with matrix multiplication. A **graph** records which operation needs which tensor. A **backend** is a place that can execute an operation. **Placement** chooses a backend for each graph node. An **intermediate representation (IR)** is a structured form of the program used between compiler stages. A **pass** analyses or changes that form. **Lowering** makes previously implicit work explicit enough for the executor to consume.

The distinction between what is executed and what is estimated is essential:

| Quantity or mechanism | Implemented meaning | Limit |
|---|---|---|
| Arithmetic | Host scalar/tiled kernels; optional OpenCL kernels | BeagleY-AI adapter and validation remain open |
| `H:` / `D:` storage | Separate host vectors in software; real buffers with OpenCL | Backend identity must accompany measurements |
| Completion | Sequential command visibility checks; optional graph-job worker and fences | Asynchronous submission does not imply overlapping commands |
| Capacity | Retained storage by default; optional exact-size pool and separate tile scratch limit | No cache, cycle or driver-memory model |
| `modeled_ms` | Sum of declared compute/copy/launch/sync assumptions | Uncalibrated; not a hardware measurement |
| `host_execution_ms`, trace durations | Measured host command execution and protocol checks | Not device throughput; tiny-run overhead matters |
| Accelerator measurements | OpenCL event durations when the optional backend is available | Local GPU measurements do not establish board behavior |

The compiler path is `ONNX + named input arrays → checked semantic lowering → internal graph text → validated graph → optional fused graph → placement → transfers → allocated commands`. Small handwritten internal graphs remain supported for mechanism experiments. The runtime path is `commands → buffers and arithmetic → output → reference comparison → report`. The source graph remains available for checking the compiled result.

## A short orientation route

Allow roughly 90 minutes for one bounded tour; use the longer route below for practice. Build instructions and prerequisites live in [the demo guide](README.md).

1. Read `examples/fragmented.plg`, calculate the four output values, then run `inspect` (15 minutes).
2. Read the structs in `include/placementlens/core.hpp`, then `validate`, `capability`, and `choose` in `src/core.cpp` (20 minutes).
3. Trace `ACC` through `insert_transfers`, `lower`, and `execute`. Explain each copy, allocation, and completion check (25 minutes).
4. Compare the report with the commands. Distinguish cost-model arithmetic, numerical checks, and host timing (15 minutes).
5. Predict a one-option change before running it. Use one exercise below to test your explanation (15 minutes).

Run these from the repository root after building. The output directory must exist before writing reports.

```sh
mkdir -p artifacts/learning-guide
build/placementlens inspect --graph examples/fragmented.plg
build/placementlens compile --graph examples/fragmented.plg --policy cost
build/placementlens explain --graph examples/fragmented.plg --report artifacts/learning-guide/run.md --json artifacts/learning-guide/run.json --trace artifacts/learning-guide/trace.json
python3 scripts/show_passes.py artifacts/learning-guide/run.json --policy cost --pass-name InsertTransfersPass
python3 scripts/explain_run.py artifacts/learning-guide/run.json
```

`compile` prints a plan without executing it. `explain` executes the compared plans, checks their outputs, and reports the recorded decisions. These are different actions even when they select the same placement.

## One example through every representation

Read [examples/fragmented.plg](../examples/fragmented.plg). Its statements are data, not code evaluated by a shell or interpreter:

```text
tensor A f32 2 3 1 -2 3 0 4 -1
tensor B f32 3 2 2 1 1 -3 -2 2
tensor D f32 2 2 1 1 1 1
matmul mm A B C
relu activation C R
add bias R D Y
output Y
```

The graph has three operations named `mm`, `activation`, and `bias`. Tensor names identify values; node names identify operations. `A`, `B`, and `D` are supplied inputs. `C`, `R`, and `Y` are computed once each.

```text
A = [[ 1,-2, 3], [0, 4,-1]]       shape 2 x 3
B = [[ 2, 1], [1,-3], [-2,2]]     shape 3 x 2
D = [[ 1, 1], [1, 1]]            shape 2 x 2

C = A @ B   = [[-6,13], [6,-14]]
R = ReLU(C) = [[ 0,13], [6,  0]]
Y = R + D   = [[ 1,14], [7,  1]]
```

For example, `C[0,0] = 1*2 + (-2)*1 + 3*(-2) = -6`. Matrix multiplication sums a row of A against a column of B, so A's column count must equal B's row count. ReLU replaces negative entries with zero. Add combines corresponding entries and requires identical shapes; there is no broadcasting.

Storage is **row-major**: complete rows appear consecutively in a flat vector. Position `(r,c)` is `r * columns + c`. A and B each contain six FP32 values, or 24 bytes on the supported four-byte-float platform. C, R, D, and Y each occupy 16 bytes. Shape checks precede loops over this storage.

`parse` creates `Graph` inputs and `Node` records. `validate` fills `Graph::shapes` and orders nodes so producers precede consumers. This is a **topological schedule**. Validation scans nodes in source order, completing available nodes and repeating the scan for blocked earlier nodes. The resulting schedule is deterministic and fixed; placement search does not reorder independent work. With fusion disabled, the three original operations survive unchanged.

By default ReLU is unsupported on the software device. `capability` accepts MatMul and Add but rejects ReLU. Placement names follow compiled node order: `C` means CPU and `A` means virtual accelerator. CPU policy chooses `CCC`; maximal locally supported placement chooses `ACA`; cost policy selects `ACC` under the default assumptions.

The transfer stage for `ACA` makes these movements explicit. Each compute/copy destination also needs an allocation before its first write in the subsequent lowering stage.

```text
COPY H:A → D:A     24 bytes
COPY H:B → D:B     24 bytes
COMPUTE MatMul on device → D:C
SYNC
COPY D:C → H:C     16 bytes
COMPUTE ReLU on CPU → H:R
COPY H:R → D:R     16 bytes
COPY H:D → D:D     16 bytes
COMPUTE Add on device → D:Y
SYNC
COPY D:Y → H:Y     16 bytes
```

That is 80 H2D bytes, 32 D2H bytes, six copies, and two consecutive device runs. H2D means host-to-device and D2H means the reverse. Under `ACC`, Add stays on the CPU. The copies of R, D, and Y disappear: 48 H2D bytes, 16 D2H bytes, three copies, and one device run remain.

For `ACC`, `lower` emits the complete 13-command sequence below. Graph inputs already exist on the host, so there are no input-host ALLOC commands. IDs start at zero. The current command contract is version 3, which carries native CNN/reshape parameters. Historical first-release reports used version 1; the prior runtime extension used version 2. This source-level contract is not a binary wire ABI.

| ID | Command | State after the command |
|---:|---|---|
| 0 | ALLOC `D:A` | Storage exists, value is not initialized |
| 1 | COPY `H:A → D:A` | Device A is initialized |
| 2 | ALLOC `D:B` | Device B storage exists |
| 3 | COPY `H:B → D:B` | Device B is initialized |
| 4 | ALLOC `D:C` | MatMul output storage exists |
| 5 | COMPUTE MatMul `D:A D:B → D:C` | C has values; device completion is pending |
| 6 | SYNC | Device completion is visible by the protocol |
| 7 | ALLOC `H:C` | Host C storage exists |
| 8 | COPY `D:C → H:C` | Host C is initialized |
| 9 | ALLOC `H:R` | Host ReLU output storage exists |
| 10 | COMPUTE ReLU `H:C → H:R` | R is initialized |
| 11 | ALLOC `H:Y` | Host final-output storage exists |
| 12 | COMPUTE Add `H:R H:D → H:Y` | Y is initialized |

An operation, a tensor, and a command are different units: three source operations become 13 commands here.

`execute` consumes those descriptors sequentially. Allocated and initialized are distinct states; zero-filled allocated storage is not permission to read an unwritten tensor. The result is `[1,14,7,1]`. `reference` separately evaluates the original graph, without using the placement or transfer plan. `execute` compares output values with an absolute-plus-relative tolerance.

The report retains the input and compiled graphs, configuration, placement alternatives, commands, source-node IDs, pass snapshots, numerical result, modeled totals, and measured host timings. The Markdown explanation is generated from those objects; it does not run another placement algorithm. The trace records host command durations on a single execution thread.

If `--relu-supported --fuse` is enabled, `fuse_matmul_relu` replaces MatMul followed by ReLU with `MatMulRelu`. Its output is R and its `origins` contain both `mm` and `activation`. There are then two compiled nodes, so a placement name has two letters. The original three-node source remains the reference. This fusion removes intermediate C from the compiled computation; it cannot remove C if another consumer or the declared output still needs it.

## Read the modules as contracts

An invariant is a condition that must remain true across a stage. The following contracts describe the normal validated pipeline; direct C++ callers must also respect function preconditions. Each exercise is a proposed learner change, not a claim that it has already been implemented. Make one change at a time in a scratch working copy, predict the consequence, and run the relevant existing tests. Answers to the numbered questions are in a separate section below.

### 1. Shapes, tensors, and graph records

**Code:** [include/placementlens/core.hpp](../include/placementlens/core.hpp), `Shape`, `Tensor`, `Node`, `Graph`; [src/core.cpp](../src/core.cpp), `Shape::elements`, `Shape::bytes`, `result_shape`.

**Purpose and contract:** represent mathematical values before backend decisions. Inputs are positive dimensions and finite input values. Outputs are bounded element/byte counts and inferred result shapes. Tensor storage length must match its shape; definitions are unique. A shape has at most 65,536 elements. `Node::origins` links transformed work to original node IDs.

**Example and failure:** A 2×3 times B 3×2 produces C 2×2. A 2×3 times B 4×2 fails `E_SHAPE`. Two equal element counts alone do not make Add legal: 1×4 and 2×2 are different shapes.

**Design and limits:** flat vectors and rank-two shapes make indexing inspectable. The division-based bound in `elements` avoids multiplying unchecked dimensions. Logical NCHW dimensions live in native CNN parameters, while the buffer storage stays rank two. The ONNX importer explicitly expands accepted constant biases; this does not add general runtime broadcasting. Strided views, dynamic shapes and mixed precision remain unsupported.

**Try yourself:** add a small test that distinguishes same element count from same shape for Add. **Predict Q1:** Why would simply comparing vector lengths accept an invalid Add? **Debug Q2:** Where should an oversized MatMul result fail before allocating its output?

### 2. Input language and validation

**Code:** [src/core.cpp](../src/core.cpp), `load`, `parse`, `validate`, `identifier`, `result_shape`.

**Purpose and contract:** turn a text stream into a validated `Graph`, or throw an error. Parsing checks statement syntax, unsigned decimal dimension tokens, finite f32 input data, arity and trailing tokens. Validation checks names, unique definitions, references, shapes and acyclic dependencies; it returns a graph with shapes and a stable topological order. Limits include 1–64 operations; exhaustive search is separately capped at 12, and there are at most 64 inputs. The CLI additionally validates UTF-8 and bounds graph-file reading to 16 MB before appending excess data.

**Example and failure:** `relu activation C R` can be written before its producer in source and still be scheduled after it. A missing tensor produces `E_REFERENCE`; a cycle produces `E_CYCLE`; redefining A as an output produces `E_SSA`.

**Design and limits:** each tensor is defined once, a simplified single-assignment property. It allows retained copies without invalidation. A tiny grammar keeps malformed-input behavior inspectable; a general importer or a mature parser library would handle a much larger language but add dependencies and contracts. Node count is bounded, not unrestricted.

**Try yourself:** write a reordered version of the example and compare its validated dependency order. **Predict Q3:** Must textual statement order equal execution order? **Debug Q4:** Why is a missing tensor diagnosed differently from a dependency cycle?

### 3. Guarded fusion

**Code:** [src/core.cpp](../src/core.cpp), `fuse_matmul_relu`; [examples/branched.plg](../examples/branched.plg).

**Purpose and contract:** accept a validated graph plus configuration, return a `PassResult` containing a revalidated graph and remarks. Fuse only a MatMul/ReLU pair with a single-use, non-output intermediate and configured ReLU support. Preserve the final value and original IDs.

**Example and failure:** `mm → activation` becomes one `MatMulRelu` node producing R. In the branched example, another Add reads C before clamping, so fusion is skipped rather than destroying those negative values.

**Design and limits:** conservative guards are easy to audit. Fusion happens before placement and its target-support guard applies even if a later plan uses the CPU. A richer compiler could retain alternatives per backend or duplicate/recompute intermediates. This implementation does not search such choices or assign an extra compute saving to fusion.

**Try yourself:** create a graph whose declared output is C even though ReLU also consumes C; inspect the refusal reason. **Predict Q5:** Does `--fuse` alone fuse the default example? **Debug Q6:** How can one compiled command still identify two source operations?

### 4. Device capability and configuration

**Code:** [src/core.cpp](../src/core.cpp), `validate_config`, `capability`; [include/placementlens/core.hpp](../include/placementlens/core.hpp), `Config`, `Capability`.

**Purpose and contract:** configuration validation checks finite, admissible cost parameters. Capability consumes a node, validated shapes, and configuration; it returns acceptance plus a reason. Acceptance requires an implemented opcode and enough capacity for distinct operands plus output. ReLU and MatMulRelu require the ReLU flag. CPU execution is the fallback within the supported graph language.

**Example and failure:** MatMul needs 24+24+16 = 64 local bytes. With an 80-byte device budget it is accepted, as is the 48-byte Add. Yet their combined retained plan can exceed 80 bytes. Unsupported source syntax is a parser error; absent device ReLU is a capability decision on a valid graph.

**Design and limits:** a local footprint is cheap to explain but does not prove plan-wide feasibility. Real capability checks may depend on dtype, layout, dimensions, quantization and vendor kernels. Native CNN descriptors now validate relevant static shape/attribute rules, but physical dtype/layout restrictions, quantization and vendor-kernel acceptance remain unmodeled.

**Try yourself:** add a capability test for `add n X X Y`; count X once in its local footprint. **Predict Q7:** Can all individually accepted nodes form an infeasible plan? **Debug Q8:** Why must zero bandwidth be rejected before costing a copy?

### 5. Placement search and its objective

**Code:** [src/core.cpp](../src/core.cpp), `compute_cost`, `enumerate`, `choose`, `placement_name`; [src/search.cpp](../src/search.cpp), `bounded_search`.

**Purpose and contract:** choose a backend vector and fully lowered `Plan` from a validated graph, configuration and policy. `cpu` assigns all CPU; `maximal` assigns all locally supported nodes to the device; `cost` enumerates assignments, rejects illegal or over-budget plans, and chooses the smallest modeled total. Stable sorting preserves enumeration order on exact ties.

**Example and failure:** three nodes give eight assignments; default missing ReLU support rejects four. `ACC` costs less than `ACA` with the declared constants. Maximal placement at 80 bytes fails `E_MEMORY`; cost placement can still choose a feasible plan.

**Design and limits:** exhaustive enumeration makes the bounded optimum inspectable. It grows as `2^n`, capped at 12 nodes, and searches only placement for the already transformed graph and fixed schedule. The core also supports graphs through 64 nodes using explicit `--search bounded`: a budgeted beam search retains competitive plans and baselines. Its report marks `exhaustive=false`; candidate quality and optimality are separate questions. Automatic CLI explorer/measured-study integration now passes the 29-node deep MLP with nine retained candidates; browser validation of that larger-model artifact remains pending. The study uses a candidate-set comparator, not an exhaustive global oracle. Winning the same model used for selection is not independent validation of hardware profitability.

**Try yourself:** in a standalone Python calculation, derive the copy-cost threshold where default `ACC` and `CCC` tie, then test nearby CLI values. **Predict Q9:** What does increasing `--boundary-ms` to 5 favor? **Debug Q10:** Why can a bigger MatMul still have the same modeled compute cost?

### 6. Transfer insertion and residency

**Code:** [src/core.cpp](../src/core.cpp), `insert_transfers`, its `ensure` and `sync` lambdas, `handle`; [include/placementlens/core.hpp](../include/placementlens/core.hpp), `TransferIR`, `TransferStep`.

**Purpose and contract:** convert a validated graph and placement into logical copy/compute/sync steps plus byte and device-run counts. Initially host input handles are valid; configured mirrored inputs also have prepared device copies. An operand must exist on its compute backend; copy it only when absent there. Device work must be synchronized before host reads, and the declared output must become host-resident.

**Example and failure:** `ACA` transfers 112 bytes; `ACC` transfers 64. The shared-input test copies X to the device once even when multiple operations read it. A placement-length mismatch fails before walking the graph. A direct invalid graph can have no valid producer copy; ordinary validation prevents that upstream.

**Design and limits:** immutable values allow a set of valid handles to replace a cache-coherence protocol. There is one sequential schedule and one pending flag. In-place mutation or concurrent queues would require versioning and finer dependency/completion tracking. The current pass conservatively synchronizes before every CPU compute following pending device work, even if that CPU node is independent.

**Try yourself:** build two Add nodes sharing X; predict H2D bytes before running maximal placement. **Predict Q11:** Does each operand occurrence require another copy? **Debug Q12:** What invariant would in-place writes invalidate?

### 7. Allocation and command lowering

**Code:** [src/core.cpp](../src/core.cpp), `lower`, `plan_text`; [include/placementlens/core.hpp](../include/placementlens/core.hpp), `Command`, `Plan`, `command_version`.

**Purpose and contract:** convert transfer steps into versioned commands with allocations, contiguous IDs and a summed modeled cost. Inputs already exist on the host. Every other destination receives one ALLOC before its first write. Device allocations must stay within the configured retained-byte budget.

**Example and failure:** `ACC` retains D:A, D:B and D:C: 64 bytes. `ACA` additionally retains D:R, D:D and D:Y: 112 bytes. These happen to equal the corresponding transfer totals in this example; retained storage and transferred bytes are different quantities in general. An 80-byte `ACA` plan is rejected even though local capability checks passed.

**Design and limits:** explicit ALLOC commands expose the storage contract. Default buffers live until the run ends. With `--reuse-buffers`, last-use analysis emits FREE and assigns exact-size physical slots; `peak_device_bytes` counts the reserved pool, including idle fragments. Initial mirrored buffers have a separate preparation descriptor list. Allocations have no cost in the illustrative model, but their actual command durations enter measured profiles.

**Try yourself:** add a negative test deleting an ALLOC and renumbering commands; identify the first missing-storage error. **Predict Q13:** Does allocating a buffer make its data valid? **Debug Q14:** Why renumber after deleting a command in an allocation test?

### 8. Runtime execution and reference checking

**Code:** [src/core.cpp](../src/core.cpp), `execute`, `reference`; [include/placementlens/core.hpp](../include/placementlens/core.hpp), `Run`, `Event`.

**Purpose and contract:** `execute` takes the original source, a command plan and configuration, then produces output, correctness/error metrics and events. It first validates a defensive copy of the public source graph, then checks command version/order, backend handles, allocation, initialization, single writes, shapes, device support, capacity and completion. A malformed plan throws; a structurally executable numerical mismatch produces `Run::correct == false`. Final output must be initialized on the host.

**Example and failure:** deleting SYNC rejects a device result read with `E_SYNC`. Replacing the CPU ReLU with `Add(C,C)` can satisfy shape/storage checks yet compute the wrong values; the reference detects the mismatch. Both failures matter: a safe descriptor sequence need not implement the right mathematics.

**Design and limits:** separate maps/vectors make ownership simple. The runtime uses float accumulation, while `reference` accumulates MatMul in double and rounds outputs to float at each graph node. Those matrix paths do not share a numerical kernel, but do share graph/shape infrastructure. The newer CNN float and double paths **do** share indexing helpers in `src/cnn.cpp`; original-ONNX Runtime checks supply the independent semantic comparison for that path. The comparison accepts `abs(actual-expected) <= 1e-5 + 1e-5*abs(expected)` and rejects nonfinite node results with `E_NUMERIC`, including a MatMul preactivation before fused ReLU can mask it. A nonfinite f32 reference result is also rejected; finite cancellation-sensitive inputs can still return `correct == false`. This is evidence for tested inputs, not an all-input floating-point proof. An external oracle and difficult numerical cases strengthen the evidence.

**Try yourself:** write a one-element graph with a hand-calculated negative MatMul result followed by ReLU; verify CPU, maximal, cost and fused outputs. **Predict Q15:** Does removing SYNC make arithmetic asynchronous? **Debug Q16:** Why can a float computation disagree with a double-accumulating reference without a placement bug?

### 9. CLI, reports, and traces

**Code:** [src/main.cpp](../src/main.cpp), `main`, `number`, `save`; [src/core.cpp](../src/core.cpp), `report_json`, `trace_json`; [src/report.cpp](../src/report.cpp), `explain_report`, `costs`; [scripts/show_passes.py](../scripts/show_passes.py), top-level report reader; [scripts/explain_run.py](../scripts/explain_run.py), `breakdown`, `copy_counts`, `main`.

**Purpose and contract:** parse options, retain source/compiled graphs, run the selected action, then serialize actual plan/result objects. JSON includes provenance and schema version; Markdown explains recorded decisions. `breakdown` checks component sums against totals. Output paths must differ from the source and each other. Normal success exits 0, exceptions exit 1, numerical disagreement exits 2.

**Example and failure:** default `explain` records CCC/ACA/ACC, `[1,14,7,1]`, and different model/host timing fields. `--json` pointing at the input graph is rejected. The trace uses a single host thread and reports logical backend as metadata; separate logical backends do not imply simultaneous hardware execution.

**Design and limits:** explicit flags and ordinary JSON make experiments reproducible. Pass records retain before/after text and reasons; these are direct function calls, not an LLVM/MLIR pass manager or a general semantic-diff engine. Python helpers consume reports and do not replan. Timings vary between runs, so compare deterministic plan/output fields separately from clocks. A schema library could provide stronger interchange validation if external consumers grow.

**Try yourself:** add a report-inspector check that recomputes total copied bytes from COPY shapes and verifies H2D+D2H. **Predict Q17:** Which repeated-run fields should match, and which should vary? **Debug Q18:** Why does the cost breakdown subtract launch cost from COMPUTE totals?

### 10. Tests and reproducible experiment tools

**Code:** [tests/core_tests.cpp](../tests/core_tests.cpp), `check`, `rejects`, `parse_bad`, `renumber`, `main`; [tests/cli_tests.py](../tests/cli_tests.py), `invoke`; [tests/differential_tests.py](../tests/differential_tests.py), `evaluate`, `structural_cost`; [scripts/check_matrix.py](../scripts/check_matrix.py), `workload`, `main`; [CMakeLists.txt](../CMakeLists.txt), test registration and `PLACEMENTLENS_SANITIZERS`.

**Purpose and contract:** challenge module contracts and the real CLI, compare execution with numerical expectations, and preserve evidence. Core tests include a hand calculation, plan mutations and seeded varied dimensions/values. CLI tests inspect exported plans, diagnostics and reports. The matrix creates three topologies at widths 2, 8 and 16 and five configurations each, then executes every feasible placement. The differential suite uses exact Python `Fraction` arithmetic and independent graph-edge residency sets on 18 small seeded DAGs across three configurations. It compares feasible assignments, numerical output, copy bytes and retained bytes, then repeats runs to compare deterministic report fields. These DAGs are unfused; the core suite separately checks fusion against the original graph.

**Example and failure:** a shared MatMul intermediate must block fusion. Matrix tests require unique executed placements, equality with the reported feasible set, and feasible-plus-rejected count equal to `2^compiled_nodes`. These checks catch omissions that checking only the winning policy misses. Negative tests must distinguish expected rejection from process crashes or unrelated diagnostics.

**Design and limits:** generated micro-workloads isolate behaviors. They are not trained models, industry benchmarks or proof of all legal inputs. The matrix uses bounded binary fractions and does not stress every cancellation/overflow case. Sanitizers find some memory/undefined-behavior bugs in executed paths; passing them does not establish algorithmic correctness. Random tests need seeds and concrete failing inputs for reproduction.

**Try yourself:** choose a small rectangular graph with values you can evaluate without the implementation; test all feasible placements and retain the seed/input if you randomize it. **Predict Q19:** Why is executing only the model winner insufficient? **Debug Q20:** Does agreement with the cheapest modeled cost establish numerical correctness or hardware speedup?

### 11. Lifetimes, physical reuse and initial residency

**Code:** [src/core.cpp](../src/core.cpp), `lower`, `execute`, `DeviceAllocations`; [MEMORY.md](../docs/runtime/MEMORY.md) and [RESIDENCY.md](../docs/runtime/RESIDENCY.md) describe the supporting contracts.

**Purpose and invariants:** a tensor names an immutable value, while a physical slot can hold several non-overlapping values over time. Lowering computes final device uses, synchronizes before release, emits FREE and assigns idle exact-size slots. Runtime validation independently checks slot ownership, shape, initialization, capacity and use after free. An idle slot still consumes the reserved pool budget. The external backend releases/reallocates buffers and does not claim physical software-vector reuse.

**Example and failure:** the measured seven-Add chain uses 131,072 retained bytes or 49,152 pooled bytes. Changing a FREE slot to another live slot must fail. Mirroring inputs A and B in the worked example prepares 48 bytes before execution and removes their later H2D commands; preparation still costs time and storage.

**Rationale and alternatives:** exact-size pooling is deterministic and easy to audit but can strand capacity in idle size classes. A coalescing arena could reduce fragmentation while adding alignment and aliasing complexity. Keeping inputs on both host and device is a bounded residency experiment; device-only initial state and persistent cross-run storage need additional contracts.

**Try yourself:** create a chain with alternating tensor sizes and predict reserved versus simultaneously live bytes. Add a test that attempts to free a shared input before its final consumer. **Predict Q21:** Can peak live bytes fit while the pool is infeasible? **Debug Q22:** Why is reduced executor time with mirrored inputs insufficient to claim faster one-shot execution?

### 12. Tiled compute and reduction

**Code:** [src/kernels.cpp](../src/kernels.cpp), `compute_software`, `tiled_product`, `scalar_product`; [KERNELS.md](../docs/runtime/KERNELS.md); `tests/kernel_tests.cpp`.

**Purpose and invariants:** numerical kernels fill already allocated output storage. MatMul loads bounded A/B tiles, accumulates into a C tile, and handles edge tiles without reading padding. A square tile of edge t needs `3*t*t*sizeof(float)` scratch bytes. The reference uses a separate double-accumulating implementation over the original graph. RowSum maps rows×columns to rows×1.

**Example and failure:** a tile edge of 2 requires 48 bytes; a 47-byte local budget is rejected. A 3×5 by 5×4 multiplication exercises partial row/inner tiles. A fused negative overflow is rejected before ReLU can hide it. `tile_load_bytes` counts actual scratch loads, not all processor cache traffic or board DMA.

**Rationale and alternatives:** fixed square scratch arrays make movement explicit. They may be slower than scalar host arithmetic because copies and bookkeeping cost time; no cache optimization is assumed. Rectangular tiling and vectorized kernels are future mechanisms to evaluate after a baseline.

**Try yourself:** calculate tile counts and loaded bytes for a rectangular case before changing tile size. **Predict Q23:** Does doubling the tile edge double scratch storage? **Debug Q24:** Why must kernels preserve the preallocated output vector?

### 13. Asynchronous submission and ownership

**Code:** [include/placementlens/async.hpp](../include/placementlens/async.hpp), `AsyncRuntime`, `Fence`, `Completion`; [src/async.cpp](../src/async.cpp); [ASYNC.md](../docs/runtime/ASYNC.md).

**Purpose and invariants:** producers submit owned graph/plan/config values into a bounded queue; a worker executes jobs and publishes results or exceptions. Capacity includes queued and running jobs. A fence retains completion state after the runtime is destroyed. Mutex publication and future readiness make inputs/results visible; cancellation uses an atomic token while execution runs outside the queue mutex.

**Example and failure:** hold the first runner at a condition-variable gate, then submit two jobs to capacity two. While both remain unfinished, a third submission must reject, and a wait timeout must not cancel either job. Without the gate, a fast job may finish and free capacity before the third submission. Accepted running cancellation wins over a racing normal return, but storage stays alive until the runner exits. Shutdown joins the worker; a blocking vendor call can delay cancellation.

**Rationale and alternatives:** one worker provides a clear admission/lifetime problem without pretending to model parallel device engines. The queue holds graph jobs, and commands inside a job are sequential. A per-command transport or multi-worker backend needs finer dependency and device ownership rules. CLI `--async` currently submits/collects plans sequentially; the C++ API and tests exercise multiple outstanding jobs.

**Try yourself:** gate an injected runner with a condition variable, fill the queue, cancel one queued job, then prove that admission recovers. Avoid sleep-based race tests. **Predict Q25:** Is a ready successful fence sufficient proof of numerical correctness? **Debug Q26:** Why is detaching the worker unsafe for buffers captured by a runner?

### 14. Measured policies and the complete-plan oracle

**Code:** [scripts/study.py](../scripts/study.py), `calibrate`, `predict`, `build_bundle`, `run_study`, `run_sweep`, `validate_bundle`; [STUDY.md](../docs/runtime/STUDY.md).

**Purpose and invariants:** calibration measures command/subgraph costs; later evaluation runs measure every feasible complete plan. Additive and uncertainty-aware selection use calibration, while the oracle and regret use held-out complete-plan measurements. Backend/shape/direction keys separate unlike operations. Seeded execution order and profile perturbations are recorded. An infeasible maximal-supported placement stays infeasible; a different feasible maximum-node policy has a different name.

**Example and failure:** increase profile error with the same saved measurements to study decision stability without introducing new timing noise. A missing command event, mismatched graph/configuration or invalid policy reference fails validation. Incorrect outputs disable performance conclusions. A robust policy may perform worse; empirical intervals are not statistical confidence guarantees.

**Rationale and alternatives:** exhaustive enumeration gives an inspectable oracle for at most 12 nodes, not a scalable model compiler. Command sums are an approximation; measured complete-plan time includes overhead that an additive model can miss. The optional `--emulate-boundaries` performs bounded host waits for configured transfer/launch/sync assumptions. Those are controlled simulator delays, not observed device latency; default model-only knobs do not slow execution.

**Try yourself:** independently recompute one policy's regret from its held-out median and the measured best median. **Predict Q27:** Why must calibration and evaluation be separate? **Debug Q28:** Why can a higher modeled boundary cost leave measured host latency unchanged?

### 15. Device integration and the external reference

**Code:** [include/placementlens/device.hpp](../include/placementlens/device.hpp), `DeviceBackend`; [src/opencl_device.cpp](../src/opencl_device.cpp), `make_opencl_device`; [scripts/compare_production.py](../scripts/compare_production.py); [BOARD_VALIDATION.md](../docs/runtime/BOARD_VALIDATION.md).

**Purpose and invariants:** allocation, upload, compute, download, synchronization and release cross an explicit backend boundary. The runtime scopes ownership of each allocation and cleans up after success, cancellation or failure. OpenCL commands block and have real profiling events where exposed. Diagnostic readbacks validate intermediate finiteness and are separately counted because they add transfer overhead outside planned COPY commands.

**Example and failure:** reuse one backend for two graph runs and prove the first run left no owned allocations. Inject a read failure and preserve the original exception while attempting cleanup. PyTorch executes original-graph operators independently and supplies a production numerical comparison; its eager CPU timer covers different work and is not a direct speedup benchmark.

**Rationale and alternatives:** an abstract per-command backend is suitable for these kernels. A TI compiled-subgraph runtime may need a coarser adapter and quantization metadata; forwarding arbitrary f32 descriptors to an NPU is not valid. The local OpenCL GPU is supplementary evidence. Board access, compatible tools, actual offload and project integration remain required.

**Try yourself:** implement a fake backend that fails the second allocation and assert cleanup of the first. **Predict Q29:** Can a reported execution provider prove accelerator offload? **Debug Q30:** Why does GPU kernel-event time differ from host command time?

### 16. Artifact-backed workbench

**Code:** [workbench/server.py](../workbench/server.py), `StudyService`, `handler_for`; [workbench/app.js](../workbench/app.js); `scripts/study.py`, `validate_bundle`, `compare_bundles`. Consult the completion matrix for current browser-validation status.

**Purpose and invariants:** the local service validates bounded configuration and invokes the same CLI/study path. Saved graph, tensor, partition, command, buffer and event IDs link the views. The browser chooses what to inspect; it does not solve placement or compute correctness. Structural differences and measured differences remain separate evidence.

**Example and failure:** save a supported run, disable middle-op support, and follow the changed partition, transfers and commands. A corrupt event reference must reject before rendering a trustworthy explanation. Failure or unavailable profiler data must remain visible. A lower median by itself cannot prove that one particular copy caused the change.

**Rationale and alternatives:** a plain local web client keeps the engine headless and avoids a second implementation. A larger notebook or model editor would add unrelated state and execution contracts. Prediction prompts exercise reasoning but do not assess competence automatically.

**Try yourself:** pick a graph node and verify every highlighted command/tensor against its saved JSON. **Predict Q31:** What component must run when capability changes? **Debug Q32:** What should the UI do when output correctness fails?

### 17. Real model import and one-to-many lowering

**Code:** [scripts/onnx_frontend.py](../scripts/onnx_frontend.py), `_analyze`, `inspect_model`, `import_model`; [scripts/export_torch_example.py](../scripts/export_torch_example.py), `export`; [ONNX contract](../docs/runtime/ONNX_FRONTEND.md).

**Purpose and invariants:** accept a standard interchange model only when its domain, opset, tensor metadata, attributes and numerical inputs fit the implemented semantics. Named NPZ arrays must match the declared runtime inputs. Constant transformations may transpose a Gemm weight or expand a bias; runtime inputs may not be silently converted into weights. Both source and generated node limits are checked. NPZ headers and payload lengths are validated before allocation, and generated graph text has its own 16,000,000-byte bound.

**Example and failure:** PyTorch's biased `Linear(3,4) → ReLU → Linear(4,2)` exports as Gemm→Relu→Gemm. Each Gemm lowers to MatMul+Add, so three source nodes become five executable nodes and 32 possible placements with all operations enabled. A broadcasting runtime bias is rejected even though the same values as a constant initializer can be expanded legally. `alpha=.5` on Gemm is rejected rather than silently ignored.

**Larger example:** `--family deep-mlp` exports ten biased width-4 Linear layers and nine ReLUs, producing 29 normalized nodes. The tested bounded search checked nine retained candidates against original ONNX Runtime and PyTorch, with an explicit 128-evaluation budget. It did not enumerate all placements.

**Follow the data:** `model.onnx` preserves original names. `import.json.node_mapping.generated_nodes` connects each source index to one or more generated IDs. `normalized_nodes` carries exact generated operands and shapes; `tensor_lowering` records constant transposes/expansions. Follow those IDs into `core.json` capabilities, compute commands and events. A source name alone need not be unique, so the source index also matters.

**Try yourself:** export `--family biased-mlp`, count the source/normalized nodes, and trace one bias to its materialized tensor. **Predict Q33:** Why can a 33-Gemm model exceed the 64-node executable bound? **Debug Q34:** Why can checking only the generated graph miss an importer bug?

### 18. Native spatial kernels and packed tensor storage

**Code:** [src/cnn.cpp](../src/cnn.cpp), `cnn_shape`, `compute_cnn`, `reference_cnn`; [include/placementlens/cnn.hpp](../include/placementlens/cnn.hpp); `Node::parameters` and command parameters in [core.hpp](../include/placementlens/core.hpp); the CNN cases in `onnx_frontend._analyze`.

**Purpose and invariants:** execute static Conv2D, MaxPool2D, AveragePool2D, GlobalAveragePool and contiguous Reshape with actual spatial indexing. A logical `[N,C,H,W]` tensor uses packed storage `[N,C*H*W]`; native command parameters retain the logical dimensions, group, kernel, stride, padding and dilation. Conv bias is a separate constant-channel Add. Parameter counts, packed shapes, output extents, memory bounds and total kernel visits are validated before the loops run. ABI version 3 makes these parameters part of the explicit command contract.

**Example and failure:** a `[1,1,4,4]` input with two 3×3 filters and one-cell padding produces logical `[1,2,4,4]`, packed `[1,32]`. The Conv command still names a 3×3 kernel; there is no hidden im2col MatMul replacement. A 256×256 pool window applied over 65,536 outputs requests billions of visits despite bounded tensors, so the work guard rejects it. Ceil-mode pooling and automatic SAME padding are currently rejected rather than approximated.

**Reference boundary:** float execution and double accumulation share CNN indexing helpers. A wrong index can therefore contaminate both. The original, untranslated ONNX must be evaluated independently. The retained small-CNN and residual-CNN exports match original ONNX Runtime and PyTorch outputs across 128 and 512 software placements respectively. Those counts describe tested placements, not independent models or physical accelerators.

**Try yourself:** calculate one padded convolution output and one AveragePool boundary output with `count_include_pad=0`, then with `1`. Locate all parameters in both import and command JSON. **Predict Q35:** Why does a packed `[1,32]` output not imply that the operation is a dense matrix multiplication? **Debug Q36:** How could the internal float and double CNN checks both pass an indexing defect?

### 19. Target assumptions, original-model checks, and explicit folding

**Code:** [scripts/target_profile.py](../scripts/target_profile.py), `load_profile`, `engine_flags`; [scripts/explore_model.py](../scripts/explore_model.py), `explore`, `compare_outputs`; [workbench/server.py](../workbench/server.py), `_run_model`; [scripts/export_torch_example.py](../scripts/export_torch_example.py), `fold_inference_conv_bn`.

**Purpose and invariants:** join a checked model to a named, reproducible software target, execute it, and preserve the difference between assumptions and evidence. Schema-2 `software-cnn-relu-on` enables the shared native CNN/Reshape capability group. The explorer gets per-node acceptance and reasons from the actual C++ core; it does not repeat CNN legality in the Python `evaluate_node` teaching helper. Model upload uses this same path and withholds a performance study until original-model checks pass.

**Example and failure:** upload a small CNN under a matrix-only profile, then the CNN-enabled profile. Inspect which operations require CPU fallback and which COPY commands change before discussing time. A capability flag states software eligibility, not vendor compilation or NPU execution. A finite but incorrect output still invalidates performance conclusions. Original ONNX evaluation and native execution are compared with an explicit tolerance and declared logical output shape.

**Explicit BatchNorm folding:** the `small-cnn-bn-folded` family retains an eval Conv+BatchNorm source model and `unfused.onnx`, folds a deep copy with `fuse_conv_bn_eval`, and exports the folded graph. Original/folded PyTorch and original/folded ONNX Runtime outputs are compared; the retained example also checks all 128 native placements. The original ONNX keeps its actual BatchNormalization node, and the importer still rejects that unfused operator. The transformation is tested evidence for a bounded eval case, not a native BatchNorm kernel or training-time transformation.

**Try yourself:** run the folded family and compare `unfused.onnx`, `mlp.onnx`, `export.json` and `validation.json`. Explain which tool performed each transformation and which numerical checks are independent. **Predict Q37:** Does `hardware_validated=false` become true after every software placement agrees? **Debug Q38:** Why must an eval-only BatchNorm fold reject a training-mode module?

A concrete route through the retained evidence is `artifacts/onnx-cnn-2026-10-04/biased-mlp-export`, then `small-cnn-export`, then `bn-folded-cnn-export`, followed by `small-cnn-study`. Use [the ONNX guide](../docs/runtime/ONNX_FRONTEND.md) to reproduce them in new directories. The source modules have deterministic untrained weights; task accuracy and physical board performance have not been established.

### 20. Analytical performance model

**Code:** [scripts/architecture_model.py](../scripts/architecture_model.py), `work`, `tile_scratch_bytes`, `tiled_traffic`, `kernel_cost`, `schedule`, `choose`, `sensitivity`, `compare_targets`, `compare_models`; full derivation in [ANALYTICAL_MODEL.md](../docs/runtime/ANALYTICAL_MODEL.md).

**Purpose:** estimate one request's latency, data movement and device memory for a placement on a declared accelerator, from shapes alone, and say whether the answer survives changes to the assumptions. It replaces nothing in the engine; the engine's older `compute_cost` constants (module 5) are a separate illustrative model.

**Inputs/outputs:** importer report (analysis scope) + validated profile + placement string (`C`/`A` per node) → per-node ops, traffic, tile utilization, bottleneck; H2D/D2H bytes, syncs, peak device bytes; `estimated_ms` or `None` if infeasible; scenario range; 14 sensitivity rows.

**Ownership and invariants:** the module owns no runtime state; every function is pure and deterministic (same input → byte-identical JSON). One multiply-add = 2 ops. `kernel_ms = max(compute, memory) + dispatch`. An infeasible placement never receives a latency. A target is a "robust winner" only if efficiency ranges are separated **and** no ×0.5/×2 one-parameter change flips the winner. The model is labelled *modeled*, never *measured*.

**Worked example:** `y = x@w`, x [4,8], w [8,2]: 2·4·2·8 = 128 ops, ideal traffic 4·(32+16+8) = 224 B. On an 8×8×8 tile, utilization = 64/512 = 12.5 %; at 1 GOPS, compute = 128/(10⁶·0.125) = 0.001024 ms. Placement `AC` on wide-array then moves x (128 B) and w (64 B) H2D, y (32 B) D2H with one sync; peak device = 224 + 4,096 workspace = 4,320 B; total 0.016439 ms, almost all of it the 12 µs dispatch. The digit-MLP `ACCCC` example is worked step by step in ANALYTICAL_MODEL.md §10 (0.112570 ms).

**Failure behaviour:** schema violation → `ValueError` (CLI exit 1); unsupported model → `analysis: null` and exit 2; memory or rule violation → `feasible_under_assumptions: false`; target comparison with different CPU blocks → refused.

**Why this design:** a formula a student can recompute is worth more here than a precise number about an invented chip. It is fast enough to score 1,024 placements and repeat the search for 14 perturbations. Its movement half is checked exactly against the engine (module 21).

**Alternative and tradeoff:** a Timeloop-style mapping search would choose loop orders and tiles per layer and give much better estimates for a *real* array, at the cost of a large, mostly unvalidated architecture specification. A cycle simulator would capture overlap and pipelining but needs a real microarchitecture and is far slower. Neither would make hypothetical profiles real.

**Limits to say out loud:** no copy/compute overlap, no cache model on CPU, no fusion, no quantization, no contention or thermal effects; scenario ranges are assumption ranges, not confidence intervals. Memory follows one of two declared policies that mirror the engine (`retain`, the default, or `release` = `--reuse-buffers`), so peak retained payload agrees with the engine for the same policy. Feasibility verdicts agree exactly: the model checks payload + the declared 4 KiB `workspace_bytes`, and the engine receives `--device-bytes = memory_bytes − workspace_bytes`. For `AAAAA` on the digit MLP (payload 329,152 B, 333,248 B with workspace) both accept at 333,248 B and both reject at 333,247 B ([ANALYTICAL_MODEL.md](../docs/runtime/ANALYTICAL_MODEL.md) §4.7).

**Try yourself:** hand-compute the `[16,16]×[16,16]` utilization on wide-array, then run `capstone.py estimate` on a model and check one row. **Predict Q39:** On compact-vector, why do `accelerator.peak_gops ×0.5` and `×2` give the same digit-MLP estimate? **Debug Q40:** An estimate prints `infeasible` for `AAAAA` but CPU-only works. Which two fields of `failures` do you inspect first, and what are the two possible causes?

### 21. Architecture profiles become engine rules

**Code:** [profiles/architectures/*.json](../profiles/architectures/); [scripts/architecture_model.py](../scripts/architecture_model.py), `validate`, `load`, `execution_flags`, `support`; [src/core.cpp](../src/core.cpp), `target_rule_rejection`, `capability`, and the device-COMPUTE check in `execute`; [src/main.cpp](../src/main.cpp) flags `--device-unsupported`, `--device-max-tensor-bytes`, `--device-no-grouped-conv`; `EngineCrossCheck` in [tests/architecture_model_tests.py](../tests/architecture_model_tests.py).

**Purpose:** one declared profile drives both the estimate and the real executed plan, so a fallback shown in an estimate is a fallback the engine actually executes.

**Inputs/outputs:** profile → list of engine flags. Example: `fragmented-array` → `--device-bytes 4194304 --local-bytes 65536 --device-max-tensor-bytes 262144 --cnn-supported` (ReLU flag absent). `dense-conv-array` adds `--relu-supported` and `--device-no-grouped-conv`.

**Ownership and invariants:** the profile file owns the declaration; `validate` rejects extra/missing fields, duplicate keys, `hardware_validated: true`, unordered efficiencies, and tile scratch larger than local memory. Five layers stay separate: hardware assumptions (estimate only), support rules (both), functional flags (engine), estimates (modeled), measurements (host software). Timing fields are **not** passed to the engine. In C++, `capability` consults `target_rule_rejection`, and `execute` re-applies the same rule to every device COMPUTE command, so a hand-edited plan fails with `E_CAPABILITY: device rejects command`.

**Worked example:** the trained digit CNN on fragmented-array: ReLU `n001` is rejected, so the maximal plan is `AACAAAA`. The ReLU input `t000` (245,760 B) comes back D2H and its output `t001` (245,760 B) goes out H2D: +491,520 B and +1 sync versus wide-array's `AAAAAAA` (286,688 → 532,448 B H2D, 4,800 → 250,560 B D2H). The engine's counts and the analytical counts are equal.

**Failure behaviour:** unknown excluded operator → `E_CONFIG`; more than 16 exclusions → `E_CONFIG`; profile errors stop before any engine call.

**Why this design:** if the estimator and the executor each had their own legality rules, they would drift and the estimate would describe a plan nobody runs. Translating to flags keeps one source of truth for legality, while keeping timing assumptions out of the functional engine.

**Alternative and tradeoff:** a shared rules library compiled into both C++ and Python would avoid the translation step but adds a binding layer. Embedding the analytical model in C++ would avoid Python, but makes the formulas harder to read and change.

**A defect the cross-check found and fixed:** the first model counted a sync per CPU node that pulled device data and freed device copies only at a tensor's overall last use. A skip-connection graph `Y = relu(relu(A)) + A` exposed both (placement `AACC`: engine 1 sync, model 2; under buffer reuse `CAAC`: engine peak 64 B, model 96 B). The model now copies the engine's `pending` rule and per-domain lifetimes. Release check: 1,034 executed placements across 6 models, 3 profiles and both memory policies, zero mismatches in bytes, syncs or peak memory. See ANALYTICAL_MODEL.md §11.

**Try yourself:** add `"Add"` removal to a copy of wide-array, run `execution_flags`, and predict the `--device-unsupported` flags before printing. **Predict Q41:** Why does `execution_flags` not emit `--device-unsupported ReLU` for fragmented-array? **Debug Q42:** A plan JSON was edited by hand to put a grouped Conv on the device under dense-conv-array rules. Where does it fail, and with what code?

### 22. Analysis-scope importer

**Code:** [scripts/onnx_frontend.py](../scripts/onnx_frontend.py), `_analyze(path, scope)`, `issue`, `inspect_model`, constants `MAX_ELEMENTS`, `ANALYSIS_MAX_ELEMENTS`, `ANALYSIS_MAX_NODES`, `EXECUTION_LIMIT_CODES`; CLI `capstone.py inspect --analysis-scope`.

**Purpose:** let a model author estimate a realistically sized model even when the bounded executor cannot run it, without pretending it was executed.

**Inputs/outputs:** ONNX path + `scope` (`execution` default, or `analysis`) → report with `issues` (semantic rejections), `execution_limits` (size limits only), `supported`, `functionally_executable`.

**Ownership and invariants:** semantics are checked identically in both scopes. Only codes in `EXECUTION_LIMIT_CODES` (such as `TENSOR_LIMIT`) are diverted to `execution_limits` in analysis scope. `import_model` always uses execution scope, so an analysis-only model can never reach the engine. `functionally_executable = supported and not execution_limits`.

**Worked example:** `edge-standard-160.onnx` has a 3×160×160 = 76,800-element input, above the 65,536-element executor limit. Execution scope: unsupported. Analysis scope: supported, `functionally_executable: false`, limit `TENSOR_LIMIT`, and an estimate is produced. Its stem Conv input is 307,200 B, above wide-array's 262,144 B per-tensor rule, so that node falls back to CPU in the estimate.

**Failure behaviour:** a Softmax or wrong opset is still an `issue` in analysis scope, so there is no estimate. Above 1,024 source nodes or 128 computational (generated) nodes the analysis itself refuses.

**Why this design:** one importer, one set of semantics, two size envelopes. A second "lenient" importer would drift from the executed semantics.

**Alternative and tradeoff:** raising the executor's limits would let larger models run but costs memory/time on every test and the bounded guarantees the executor relies on. Estimating directly from ONNX shape inference would skip normalization, so estimates would not match the nodes the engine places.

**Try yourself:** run `inspect` with and without `--analysis-scope` on `edge-depthwise-160.onnx` and list the codes. **Predict Q43:** Could `capstone.py execute` run an analysis-only model if you pass its NPZ? **Debug Q44:** An estimate exists but `execute` refuses the same model. Which field explains it?

### 23. Trained example: accuracy versus numerical agreement

**Code:** [scripts/train_digits.py](../scripts/train_digits.py); `examples/models/digits/manifest.json`; `tests/trained_example_tests.py`; [TRAINED_EXAMPLE.md](../docs/runtime/TRAINED_EXAMPLE.md).

**Purpose:** answer two different questions separately. *Numerical agreement*: does the engine compute the same function as the original ONNX model on every placement? *Task accuracy*: is that function any good at recognizing digits?

**Inputs/outputs:** scikit-learn's bundled UCI digits (1,797 8×8 images), `x/16` preprocessing, stratified 1,077 / 360 / 360 split, seed 20261106 → `mlp.onnx` (64-32-10, 2,410 parameters) and `cnn.onnx` (Conv 1→8 3×3 → ReLU → MaxPool → Gemm, 1,370 parameters), test inputs, labels, hashes.

**Ownership and invariants:** the test set is scored once, after selecting weights on validation. Accuracy belongs to the trained model and split; agreement belongs to the compiler/runtime. A perfect engine reproduces the model's accuracy; it cannot raise it.

**Worked example:** MLP test accuracy 0.9694 (349/360), CNN 0.9750 (351/360). PyTorch vs ONNX Runtime max logit difference 7.63e-6. Engine logits were identical to ONNX Runtime on all 32 MLP placements and all 86 feasible CNN placements, so engine accuracy equals ONNX Runtime accuracy on every placement. One image is 0.28 percentage points, so the two-image gap between MLP and CNN means nothing.

**Failure behaviour:** if a placement's logits differ beyond tolerance, that placement is incorrect and its timing is discarded, whatever its accuracy happens to be. An accuracy that matched by luck would not rescue an incorrect placement.

**Why this design:** untrained examples can show agreement but not usefulness. A tiny, bundled dataset keeps the example offline and reproducible (byte-identical retraining).

**Alternative and tradeoff:** a standard benchmark such as MNIST or CIFAR at real resolution would be more convincing about accuracy, but needs downloads and exceeds the executor's 65,536-element tensor limit at useful batch sizes.

**Try yourself:** compute how many test images a 0.5-point difference corresponds to. **Predict Q45:** Could an engine bug leave accuracy unchanged? **Debug Q46:** Engine accuracy on one placement drops to 0.96 while ONNX Runtime stays at 0.9694. What do you compare first?

### 24. Command-level device queue

Full contract, state machine and happens-before rules: [COMMAND_QUEUE.md](../docs/runtime/COMMAND_QUEUE.md).

**Code:** [include/placementlens/queue.hpp](../include/placementlens/queue.hpp), `CommandQueue`, `CommandEvent`, `QueueRecord`, `Lane`, `execute_command_queue`; [src/queue.cpp](../src/queue.cpp), `run`, `engine`, the admission code that records each buffer's `allocation`, `writer` and `readers`; CLI `placementlens ... --runtime command-queue` and `capstone.py execute --runtime command-queue`; [COMMAND_QUEUE.md](../docs/runtime/COMMAND_QUEUE.md).

**Purpose:** submit individual lowered commands, each returning an event, instead of one whole graph job (module 13). Copies run on a copy-engine thread, device computes on a compute-engine thread, and ALLOC/FREE/SYNC plus CPU computes on the submitting host thread after their dependencies.

**Invariants visible in the code:** handles are single-assignment (one writer, then readers). A command waits for the events of its operands' writers. FREE waits for the allocation, the writer and every reader. A command whose dependency failed or was cancelled fails with `E_DEPENDENCY` without running. Each engine is in-order; because submission order is topological, the oldest unfinished command is always runnable, so there is no deadlock. Engine capacity is bounded; `try_submit` returns empty when full. A timeout on `wait_for` observes only; it does not cancel.

**Worked example:** in the demo, all 16 placements of the trained digit MLP under `fragmented-array` run through the command queue and match ONNX Runtime; `queue_contracts` also shows the queue is bit-identical to the serial executor on every placement of the example graphs. The synthetic overlap run injects 20 ms waits so copy and compute intervals visibly overlap; the overlap is injected host waiting, not hardware time.

**Failure behaviour:** invalid commands are rejected at submission (`E_VERSION`, `E_HANDLE`, `E_SHAPE`, `E_FREED`, `E_SSA`, `E_MEMORY`). A failed or cancelled command marks every dependent `Failed` with `E_DEPENDENCY: command N failed|cancelled: <root cause>`; dependents never run on bad data. A full engine refuses `try_submit`; a timed `submit` gives up after its timeout. Shutdown either drains or cancels, and always joins both engine threads. Cancelling a *running* command is a simulator convenience that real GPUs generally do not offer.

**Why and alternative:** per-command events let copies and compute overlap when no dependency forbids it, which is the CUDA-stream / OpenCL-queue model. The alternative, one serial interpreter, is simpler to reason about but cannot express overlap.

**Predict Q47:** COPY H→D of tensor B and COMPUTE that reads only tensor A are both submitted. May they overlap? **Debug Q48:** Name the happens-before edge that prevents a FREE from releasing a buffer while a COMPUTE still reads it.

## Account for the cost model

The model adds compute, copies, one launch charge per consecutive device run, and explicit synchronization. `compute_cost` returns constants per opcode/backend, independent of matrix dimensions. Dimensions affect copy sizes and capacity checks.

| Placement | Compute ms | Copies ms | Launch ms | Sync ms | Total modeled ms |
|---|---:|---:|---:|---:|---:|
| CCC | 9.0 | 0 | 0 | 0 | 9.000000 |
| ACA | 2.4 | 7.800112 | 0.4 | 0.1 | 10.700112 |
| ACC | 4.0 | 3.900064 | 0.2 | 0.05 | 8.150064 |

Each copy costs `1.3 ms + bytes / 1,000,000 bytes per ms` by default. From ACA to ACC, compute rises by 1.6 ms, but copies fall by 3.900048 ms, launch by 0.2 ms and sync by 0.05 ms. The declared total falls by 2.550048 ms. Keeping the supported Add on the CPU is therefore preferred by this model.

A device run is a consecutive group of device-assigned nodes, not one generated hardware kernel. The optional OpenCL backend compiles and dispatches kernels individually. The illustrative model's fixed launch charge does not prove real work needs just one launch. The measured study below profiles actual commands separately; it still does not model overlapping execution.

Measured host executor time excludes parsing, fusion, placement search, lowering, defensive source validation, input preparation and the subsequent reference evaluator. It includes command execution/checks, event bookkeeping and output collection. Trace durations measure individual command bodies, so their sum need not equal total host time. Runner end-to-end includes preparation/reference/teardown; async end-to-end also includes queue wait. See each field's definition before comparing timers.

## What the evidence can support

The following observations remain useful controlled experiments:

1. Local operator support does not guarantee full-plan capacity: the 80-byte case distinguishes the checks.
2. Shared immutable operands need not be copied repeatedly: residency tests inspect byte counts and values.
3. Fusion must preserve visible intermediate values: shared consumers and declared outputs prevent removal.
4. Completion belongs to the command contract: a missing SYNC is rejected despite synchronous host arithmetic.
5. Structurally legal commands can be numerically wrong: the substituted-ReLU test exercises the reference check.
6. More device assignments can have a larger declared total: ACA versus ACC explains where the extra cost comes from.

The original matrix record described 45 configurations, 591 executed placements, and 11,553 executed commands. These are historical counts for the retained workload, not an assertion that the current binary has passed. Use the current [runtime validation record](../docs/runtime/VALIDATION.md), [completion matrix](../docs/runtime/COMPLETION_MATRIX.md) and generated receipts for status; the showcase validation file describes historical evidence. Counts represent repeated executions under different configurations, not independent application benchmarks. [scripts/verify_release.py](../scripts/verify_release.py) reproduces release builds, documented CLI commands and sanitizer checks in a fresh copy; [scripts/run_demo.sh](../scripts/run_demo.sh) builds/tests the normal demo. The optional pre-existing `scripts/build_demo_replay.py` assembles a silent video from captured CLI output; it is outside this headless release and its embedded historical test counts are not current acceptance evidence. The matrix summary and release manifest preserve seeds, invocation arguments, platform information and binary hashes alongside reports and graph inputs, connecting results to what actually ran.

The intended user is a model author exploring hardware constraints or a compiler/runtime engineer explaining placement and lowering. The current bounded ONNX examples make that interaction concrete, while target assumptions still describe software profiles. A backend developer could use this controlled environment to isolate a residency or capability mistake; adoption or production integration has not been demonstrated. The contribution so far is a working, inspectable integration and controlled experiments. There is no established novel placement algorithm or previously unknown research result.

This is also a concrete compiler example: validation establishes typed IR, fusion changes the graph, placement attaches a backend, and lowering exposes copies and allocations consumed by a runtime. It does not implement source-loop analysis, OpenMP/OpenACC offloading, a learned GPU profitability predictor, native code generation, or LLVM/MLIR integration.

## A deeper one-to-two-week route

Use ten sessions of about 60–90 minutes, spread over one or two weeks. An unfinished exercise is a reason to revisit the relevant contract, not a reason to assert mastery. No new roadmap features are required to follow this route.

| Session | Read and do | Concrete check to attempt |
|---|---|---|
| 1 | Foundations and the fragmented input; compute all values on paper | Explain one MatMul entry and all tensor byte sizes |
| 2 | Modules 1–2; reorder a graph and create a malformed shape | Locate the first validation rule that accepts/rejects each |
| 3 | Module 3; compare shared-intermediate and observable-output cases | Explain preserved values and trace `origins` |
| 4 | Modules 4–5; change support, then copy cost, then capacity separately | Predict the selected map and explain any disagreement |
| 5 | Module 6; draw the valid-residency set after every ACC step | Count copies for a shared input without running it first |
| 6 | Module 7; inspect allocations and mutate one plan in a test | Distinguish retained bytes, transfer bytes and initialized data |
| 7 | Module 8; read the float and double loops side by side | Explain a numerical mismatch and a completion error separately |
| 8 | Module 9; compare JSON, Markdown and host trace | Recompute totals and identify nondeterministic timing fields |
| 9 | Module 10; implement one small exercise and run focused checks | State the input, prediction, observation and residual uncertainty |
| 10 | Rerun one complete example and read the targeted references below | Explain current limits and propose one measurable hardware experiment |

If using a week, combine two short sessions on some days. If using two weeks, use gaps to retry prediction questions without the answers visible. These are suggested activities, not a record of work performed.

## Answers to prediction and debugging questions

Read this section after making your own prediction.

| Question | Answer |
|---|---|
| Q1 | Equal vector lengths do not encode row/column meaning; 1×4 and 2×2 contain four values but Add requires equal shapes. |
| Q2 | `result_shape` constructs the MatMul shape and calls `Shape::elements`; the size bound can fail during validation before runtime allocation. |
| Q3 | No. Defined operands can refer to nodes appearing later in the text; validation establishes a dependency-respecting schedule. Reordering independent nodes may change the fixed schedule and placement/cost behavior. |
| Q4 | Undefined names have no producer at all. A cycle has definitions but no remaining node whose inputs can become available. `validate` checks references before scheduling. |
| Q5 | No. The conservative fusion rule also requires configured ReLU support. Use both flags to enable this example's transformation. |
| Q6 | The fused `Node::origins` combines original IDs; transfer and command records copy those IDs. One command can correspond to two source nodes. |
| Q7 | Yes. Local operand/output footprint and total retained allocations are different constraints; ACA at 80 bytes illustrates it. |
| Q8 | Copy time divides bytes by bandwidth. Zero would make the objective nonfinite/invalid, preventing meaningful ordering of plans. |
| Q9 | CPU-only CCC for this example. Increasing fixed cost per copy penalizes boundary crossings; supported does not imply selected. |
| Q10 | Compute is a declared per-op/backend constant. Matrix dimensions affect bytes but no operation-count term is implemented in `compute_cost`. |
| Q11 | No. Once `D:X` is valid it can serve repeated operands and later consumers, because X is immutable. |
| Q12 | Both H:X and D:X could be marked valid while containing different versions. In-place updates would need invalidation/version tracking. |
| Q13 | No. ALLOC establishes storage; COPY or COMPUTE initializes its value. `execute` checks initialization separately. |
| Q14 | IDs must equal command positions. Without renumbering, an `E_SEQUENCE` failure can hide the missing-allocation behavior being tested. |
| Q15 | No. Arithmetic already ran synchronously. Removing SYNC violates the explicit visibility contract and is rejected. |
| Q16 | Finite precision and accumulation order change rounding and cancellation. Diagnose values and numerical conditioning before assuming transfers or placement are wrong. |
| Q17 | Inputs, configuration, ordering, placements, commands, origins and modeled totals should be reproducible for the same build/options. Measured durations and evidence timestamps generally vary. |
| Q18 | Lowering puts the launch charge into the first device COMPUTE of each run. Subtracting it when presenting compute separately avoids counting the same launch twice. |
| Q19 | Nonwinning plans exercise other transfer, completion and storage paths. A cost change could later select one of them. |
| Q20 | Neither. The minimum proves only an optimum of the enumerated declared objective; numerical results need an independent calculation, and hardware profitability needs measured validation. |

For the module-5 threshold exercise, default ACC costs `4 + 3b + 64/1,000,000 + 0.2 + 0.05 = 4.250064 + 3b`, where `b` is fixed cost per copy. Equality with CCC's 9 ms occurs at `b ≈ 1.583312 ms`. This is a two-plan comparison; always check the other feasible assignments before calling it a global policy threshold.

### Answers for the integrated-system modules

| Question | Answer |
|---|---|
| Q21 | Yes. Idle slots of the wrong size remain reserved. Live tensor bytes can be lower than the physical pool reservation. |
| Q22 | Initial uploads moved outside that timer. Examine preparation bytes/time and the defined end-to-end interval before comparing one-shot execution. |
| Q23 | No. Three square tiles require four times the scratch storage when their edge doubles. |
| Q24 | Replacing the output vector can discard the pooled allocation and invalidate the mechanism and its reuse accounting. |
| Q25 | No. Successful execution can return `run.correct == false`; exception-free completion and numerical correctness are separate. |
| Q26 | Detached work can access destroyed runtime/captured storage. Join establishes an ownership boundary, even when cancellation takes time. |
| Q27 | Selecting and evaluating on the same observations rewards fitting timing noise. Held-out evaluation tests the decision independently of calibration. |
| Q28 | Model-only constants affect predicted scores. They do not add work to the executor unless explicit boundary emulation is enabled. |
| Q29 | No. Provider enumeration proves availability only. Retain actual partition/offload logs and measured execution evidence. |
| Q30 | Host time includes dispatch, waits, validation and diagnostic reads. A GPU event measures its specific device operation, excluding much of that work. |
| Q31 | The authoritative core planner/executor, invoked through the bounded experiment service. Browser state is not a placement algorithm. |
| Q32 | Display the failure and suppress performance conclusions while keeping raw artifacts available for diagnosis. |

### Answers for the model-to-hardware modules

| Question | Answer |
|---|---|
| Q33 | Thirty-three biased Gemms become 66 computational nodes because each needs MatMul plus Add. The executable bound is 64 after lowering; the separate exhaustive-search bound is 12. |
| Q34 | The generated graph and its own reference can agree on the same incorrectly translated computation. Original ONNX evaluation tests the translation against an external semantic path. |
| Q35 | Packing describes contiguous storage. Conv parameters still specify NCHW dimensions, kernel, groups, strides, pads and dilation, and native loops execute those spatial semantics. |
| Q36 | Both paths share indexing helpers. Different accumulation precision does not make identical indexing logic independent; the original ONNX comparison addresses that gap. |
| Q37 | No. Those checks establish numerical agreement for the tested software paths/input. Vendor compilation, actual offload and board measurement require their own evidence. |
| Q38 | Training BatchNorm depends on batch statistics and updates running state. An eval fold uses fixed stored statistics and cannot substitute for those training semantics. |

### Answers for the college-release modules

| Question | Answer |
|---|---|
| Q39 | The selected `AAAAC` plan's accelerator kernels are memory- or dispatch-bound. `kernel_ms = max(compute, memory) + dispatch`, so a faster or slower compute peak leaves the max unchanged. Bandwidth changes do move it (×0.5 even flips the plan to CPU-only). |
| Q40 | Look at `failures[].node` and `failures[].reasons`. A named node means a support rule rejected it (operator, per-tensor size, grouped Conv); `node: null` means peak device bytes plus workspace exceeded `memory_bytes`. |
| Q41 | ReLU is controlled by the separate `--relu-supported` flag. `execution_flags` omits that flag and skips the redundant exclusion (`relu_gated`). The engine's `capability` then rejects ReLU/MatMulReLU with "ReLU is not supported on this device target". |
| Q42 | Lowering calls `capability` → `target_rule_rejection` and fails with `E_CAPABILITY: ...grouped convolution excluded by the declared target support rules`. If a plan bypassed lowering, `execute` re-checks every device COMPUTE and fails with `E_CAPABILITY: device rejects command: ...`. |
| Q43 | No. `import_model` always uses execution scope and raises `FrontendError` for the size limit, and `execute` stops at `unsupported_by_importer` before the engine runs. |
| Q44 | `functionally_executable: false` with the reason in `execution_limits` (for example `TENSOR_LIMIT`). |
| Q45 | Yes. A small numerical error that never changes an argmax leaves accuracy identical. That is why logits are compared directly, not just accuracy. |
| Q46 | That placement's logits against the ONNX Runtime logits (max absolute difference and the first differing node), not the accuracy. A difference means the placement is incorrect; find which command or kernel diverges. |
| Q47 | Yes. They run on different engines (copy vs compute) and neither depends on the other: the COPY writes B, the COMPUTE reads only A. |
| Q48 | At admission each buffer records its readers; FREE waits for the allocation, the writer and every recorded reader's event. The reader's completion happens-before the FREE runs. |

## Full-system study and defense route

The earlier ten-session route establishes the default pipeline. For a full-system tour, use the following ten sessions instead, splitting long exercises across two weeks. Completion of a session means an attempted exercise and recorded observation; it is not automatic evidence of mastery.

| Session | Investigation | Evidence to produce yourself |
|---|---|---|
| 1 | Calculate and trace the original example | Tensor values, bytes, dependencies and source lineage |
| 2 | Validate, fuse and place two shared/branched graphs | A legal and an illegal transformation with reasons |
| 3 | Lower commands and mutate invalid descriptors | Explain the exact rejection layer |
| 4 | Trace last-use slots and mirrored preparation | Separate live, reserved and transferred bytes |
| 5 | Work a rectangular tile and reduction by hand | Scratch size, tile loads, numerical tolerance |
| 6 | Gate two queued jobs, then cancel/shut down | State transitions and happens-before explanation |
| 7 | Recompute held-out regret and perturb profiles | Raw samples, formulas and a policy failure case |
| 8 | Reproduce the memory performance investigation | Mechanism change, measured uncertainty and limits |
| 9 | Toggle a workbench capability and inspect the diff | A causal chain cross-checked against raw records |
| 10 | Compare production/backend contracts; rehearse defense | Two-minute overview and fifteen-minute worked trace |

For the two-minute overview, explain the placement question, show one supported operation whose offload introduces transfers, distinguish simulator assumptions from observations, and state the strongest verified result plus the open board gate. For fifteen minutes, spend roughly three minutes on graph/placement, four on lowering/memory, three on queue/ownership, three on measured evidence and two on limitations. Be prepared to reconstruct one function, debug an injected fault, and explain one rejected alternative without copying the guide.

Future work should reuse these interfaces: a compiled-subgraph TI adapter behind the backend/correctness boundary; a per-command queue with explicit dependency fences; or a new memory allocator compared through unchanged workloads and measurement tools. None becomes a completed feature by appearing in this list. (Update, November 2026: a per-command software queue now exists, module 24. The TI adapter and allocator items remain future work; see [HARDWARE_ROADMAP.md](../planning/HARDWARE_ROADMAP.md).)

## A 7-day study and oral-defense route

About two hours a day. Each day ends with something you produce yourself, on paper or in a scratch copy. Reading the guide does not count.

| Day | Read | Do | You should be able to |
|---|---|---|---|
| 1 | Begin with the problem; modules 1–2, 17 | Import `examples/models/digits/mlp.onnx` with `capstone.py inspect`; write the 5 normalized nodes and every tensor shape by hand | Explain why one Gemm becomes MatMul + Add, and what 65,536 elements limits |
| 2 | Modules 4–7, 21 | Draw the residency sets after every node for `AACAAAA` on the digit CNN; count the transfer bytes | Predict the engine's H2D/D2H/sync counts before running `execute` |
| 3 | Module 20 and [ANALYTICAL_MODEL.md](../docs/runtime/ANALYTICAL_MODEL.md) §4–10 | Hand-compute the 4×8×2 MatMul and two rows of the digit-MLP `ACCCC` table | Derive every number in one `estimate` row from the formulas |
| 4 | ANALYTICAL_MODEL.md §5–7, 11; module 22 | Run `compare-targets` on `edge-depthwise-64`; explain each winner flip | Explain why "no reliable ranking" is a correct result and how a range differs from a confidence interval |
| 5 | Modules 8, 13, 24; [ASYNC.md](../docs/runtime/ASYNC.md) | Run one placement with `--runtime command-queue`; delete a SYNC in a scratch test and read the error | State two happens-before rules and where the code enforces them |
| 6 | Module 23; [TRAINED_EXAMPLE.md](../docs/runtime/TRAINED_EXAMPLE.md); [HARDWARE_ROADMAP.md](../planning/HARDWARE_ROADMAP.md) | Run `capstone.py demo`; label each line exact/modeled/measured/evaluated | Explain accuracy vs agreement, and what the hardware step would add |
| 7 | This section's scripts and questions | Give the 2-minute and 15-minute talks aloud, timed; answer 10 random viva questions without notes; do the self-assessment checklist | Defend one design choice and one limitation without reading |

## A 30-second introduction

"We built an explainable model-to-accelerator exploration system. You give it an ONNX model and a description of an accelerator. It shows which operations the accelerator can run, which fall back to the CPU, exactly how much data has to move and when the CPU must wait, whether the plan fits in memory, and an estimated latency with a best/worst-case range under stated assumptions. Then it actually executes the plan on a simulated device and checks the answer against ONNX Runtime. The accelerators are hypothetical, and we did not measure real hardware in this release."

## A two-minute explanation

1. **Problem (20 s).** An accelerator may support only some operations. Every unsupported one sends data back to the CPU and forces a wait. Whether offloading helps depends on that movement, not only on raw speed.
2. **Pipeline (40 s).** ONNX → importer → internal graph → capability rules from an architecture profile → placement → explicit commands (ALLOC, COPY, COMPUTE, SYNC, FREE) → simulated device → check against ONNX Runtime. One CLI, `scripts/capstone.py`, runs all of it.
3. **Key result (30 s).** On the trained digit CNN, removing ReLU support from the target (fragmented-array) turns `AAAAAAA` into `AACAAAA`: 491,520 more bytes and one more sync. The engine executes that plan, and the analytical model's byte and sync counts equal the engine's exactly. All 192 executed placements match ONNX Runtime.
4. **Honesty (20 s).** Latency numbers are modeled from declared assumptions, shown with ranges and a sensitivity test; the tool refuses to rank targets when a single parameter change flips the winner. Accuracy (0.9694 MLP, 0.9750 CNN) is a property of the trained models, separate from numerical agreement.
5. **Limits (10 s).** Functional simulator, not cycle-accurate; f32 only; hypothetical profiles; BeagleY-AI work deferred.

## A 15-minute technical walkthrough

| Minutes | Topic | Show |
|---|---|---|
| 0–2 | Problem and what is exact / modeled / measured / evaluated | The table at the top of this guide |
| 2–4 | Import | `capstone.py inspect examples/models/digits/cnn.onnx`; source-node lineage; analysis scope on a 160 px model |
| 4–6 | Profiles and rules | One profile JSON; `execution_flags`; `target_rule_rejection` in `src/core.cpp` |
| 6–8 | Placement and lowering | `capstone.py execute ... --target fragmented-array`; the fallback reason; the 32 commands |
| 8–10 | Runtime and completion | SYNC rule; graph-job queue vs command-level queue; one happens-before rule; `E_SYNC` on a missing SYNC |
| 10–12 | Analytical model | Hand-check one `estimate` row; tile utilization; weight- vs output-stationary; scenario range; sensitivity |
| 12–13 | Robustness | `compare-targets` on `edge-depthwise-64`: no reliable ranking, and why |
| 13–14 | Correctness and accuracy | Placements vs ONNX Runtime; accuracy vs agreement |
| 14–15 | Limits and next step | The defect the cross-check found on a skip graph and how it was fixed; hardware roadmap |

## Likely viva and engineering questions

| # | Question | Concise answer |
|---|---|---|
| 1 | What exactly does the system do? | Imports a bounded ONNX model, decides CPU/accelerator placement under declared rules, lowers to explicit commands, executes them on a functional simulated device, checks against ONNX Runtime, and estimates latency analytically. |
| 2 | What is PlacementLens? | The C++ compiler/runtime executable only, not the whole project. |
| 3 | Why a simulator instead of the real board? | Hardware was deferred for this release. The simulator lets every command, buffer and ordering rule be checked deterministically; it checks every executed plan against ONNX Runtime on the supplied inputs; it says nothing about chip latency. |
| 4 | What does the simulator *not* prove? | Vendor compiler acceptance, quantized accuracy, real latency, DMA/cache/thermal behaviour. |
| 5 | Is the simulator cycle-accurate? | No. It is functional: it computes correct values in the declared order. Timing is either modeled (formula) or host software time. |
| 6 | How is the analytical latency computed? | Per kernel `max(ops/(peak·eff·util), bytes/(bw·eff)) + dispatch`, plus transfers `fixed + bytes/link_bw`, plus syncs, kernels serial. |
| 7 | Why count a multiply-add as 2 operations? | It is the common GOPS convention (one multiply, one add), so peak figures and op counts use the same unit. |
| 8 | What is tile utilization? | Useful work over padded work: `M·N·K / Π ceil(d/t)·t`. Small matrices on big tiles waste the array. |
| 9 | Weight-stationary vs output-stationary here? | If the whole weight operand plus one tile set fits local memory, weights are read once; otherwise they are re-read once per row tile. Activations are re-read once per column tile. |
| 10 | Why does a fallback cost so much? | The fallen-back op's input must come to the host (D2H + sync) and its output must go back (H2D). For the CNN ReLU that is 491,520 B and one sync. |
| 11 | How do you know the analytical movement is right? | `EngineCrossCheck` compares H2D/D2H bytes, sync count and peak device memory with the engine's emitted command streams for every executed placement, under both memory policies: equal (1,034 placement checks in the release run). It also found and fixed a real sync/lifetime defect. Agreement covers the executed cases, not every possible graph. |
| 12 | Is it right for every graph? | Not proven. Bytes, syncs and peak payload matched in all 1,034 executed release checks. The skip graph `Y=relu(relu(A))+A` first exposed a model defect (2 syncs vs the engine's 1; 96 B vs 64 B peak), fixed by copying the engine's pending-work sync rule and per-domain lifetimes. This is agreement on the tested cases, not a proof. |
| 13 | Is the scenario range a confidence interval? | No. It is the estimate at the declared low and high efficiencies. No probability is involved. |
| 14 | When do you call a target the winner? | Only if its range is separated from the others and no ×0.5/×2 change of any of 7 parameters flips the winner. |
| 15 | Why report "no reliable ranking" as a result? | Because the assumptions genuinely do not support a ranking. Hiding that would be a false claim. |
| 16 | Why identical CPU assumptions in target comparisons? | Otherwise a host change would confound the accelerator comparison. |
| 17 | What is the search? | Exhaustive over legal placements when ≤ 10 legal nodes (analytical) / ≤ 12 nodes (engine); deterministic beam otherwise, labelled as candidates not optimum. |
| 18 | What is analysis scope? | Same import semantics with larger size limits; the result is marked not functionally executable. |
| 19 | Accuracy vs numerical agreement? | Agreement: engine logits equal ONNX Runtime's (a runtime property). Accuracy: correct labels on held-out data (a model property). |
| 20 | Is 0.9750 vs 0.9694 a meaningful difference? | No. That is two images out of 360; one image is 0.28 points. |
| 21 | What does SYNC guarantee? | The host does not read device results until device work has completed. Removing it is rejected with `E_SYNC`. |
| 22 | What does the command-level queue add? | Per-command events and separate copy/compute engines, so independent copies and computes may overlap; FREE waits for every reader. |
| 23 | Is the overlap in the demo real? | It is real overlap of software threads, created by injected waits. It is not hardware time. |
| 24 | How would this run on a real GPU? | Commands become stream/queue operations, SYNC becomes an event wait, COPY becomes an async memcpy. The rules transfer; the performance does not. |
| 25 | How would it run on BeagleY-AI? | Through TI's TIDL compiler and ONNX Runtime provider, which choose their own partitions and quantize. The adapter is prepared but untested; see the hardware roadmap. |
| 26 | Why not Timeloop or a cycle simulator? | They need a real architecture description to pay off. With hypothetical profiles, a transparent formula is easier to check and explain. |
| 27 | How do the profile rules reach the C++ engine? | `execution_flags` emits `--device-unsupported`, `--device-max-tensor-bytes`, `--device-no-grouped-conv`, `--relu-supported`, `--cnn-supported`, memory limits; `target_rule_rejection` applies them. |
| 28 | Do the model and the engine agree on memory feasibility? | They agree on peak retained payload, because both use the same declared policy: `retain` keeps every device allocation (engine default); `release` frees a device copy after its last device reader and waits for the device before freeing (engine `--reuse-buffers`). Feasibility verdicts agree too: the model checks payload + the declared 4 KiB workspace, and the engine receives `--device-bytes = memory_bytes − workspace_bytes`. The digit MLP's maximal plan fits at 333,248 B and is rejected at 333,247 B by both (`test_memory_feasibility_boundary_matches_engine`). |
| 29 | What is novel? | No novel algorithm is claimed. The contribution is a working, inspectable integration with exact cross-checks and honest labelling. |
| 30 | What would you do next? | Write a documentation-based profile for the real target, predict, then measure on the board and report the prediction error. |
| 31 | What was your own contribution? | Answer personally and specifically. This guide cannot answer it. |
| 32 | How does this map to an NPU (e.g. TI C7x/MMA via TIDL)? | An NPU compiler takes a whole subgraph, quantizes it (typically int8) and emits a compiled artifact driven through descriptor/DMA rings; there is no per-op dispatch from the host. Placement becomes the vendor compiler's partitioning decision; our capability rules, transfers and fences correspond to its supported-op list, partition boundaries and completion signals. See [HARDWARE_ROADMAP.md](../planning/HARDWARE_ROADMAP.md) §5. |
| 33 | And a TPU? | XLA compiles the whole program ahead of time, so host fallback is rare and placement is mostly a compiler decision. Performance hinges on systolic-array tile utilization, which is the same padding effect our tile-utilization formula shows, at a much larger scale. |
| 34 | Your result is about transfer bytes, but the M4 and BeagleY-AI share DRAM between CPU and accelerator. Does it still mean anything? | On shared-DRAM SoCs a "copy" may become a zero-copy mapping (map/unmap) or cache flush/invalidate rather than a byte copy, so our H2D/D2H byte counts are an upper bound on data movement there. The ordering rules still hold: the host must wait for device completion, and buffer ownership must pass cleanly between CPU and device. The sync count and the happens-before rules are the part that transfers. |
| 35 | Where is the AI and data-science content? | ONNX models from PyTorch; a trained digit MLP and CNN with a stratified held-out split and evaluated accuracy; the distinction between accuracy (model) and numerical agreement (runtime); and ML-systems deployment: where operators run and what moving tensors costs. |
| 36 | Why build this instead of using ONNX Runtime execution providers, TVM or IREE? | Those systems hide partition, copy and sync decisions inside their runtimes. This project makes each decision explicit and checkable on small models. It claims no superiority and replaces none of them. |
| 37 | Where do the profile numbers (300 / 3000 GOPS, 4 GB/s link, 1 / 12 µs dispatch) come from? Are they realistic? | They are teaching scenarios, chosen to contrast a small-tile, low-dispatch design with a large-tile, high-dispatch one; each profile says `"level": "hypothetical"` and names no board. They are not from a datasheet, and no latency claim about a real chip follows from them. |
| 38 | Which code existed before the capstone? | The first bounded PlacementLens release (`.plg` graph, capability, placement, lowering, serial executor, CLI; built for the SegFault 2026 hackathon) was imported on 18 September 2026 (commit `db05bc9`, about 1,300 lines). The ONNX importer, studies, async and command-level queues, kernels, OpenCL backend, analytical model, architecture profiles, trained example and workbench were added afterwards. Say who did which part personally (Q31). |

## Self-assessment checklist (do this yourself)

Generated material, including this guide, the answers above and any AI-written explanation, **does not establish your understanding or your contribution**. Only work you do without notes does. Tick an item only after doing it yourself, and keep the paper or scratch file.

- [ ] Draw the full pipeline on paper from memory: ONNX → importer → IR → capability → placement → commands → runtime → simulated device → fences → ONNX Runtime check. Name the file and function for each stage.
- [ ] Hand-compute the 4×8×2 MatMul: operations (128), ideal traffic (224 B), utilization on an 8×8×8 tile (12.5 %), compute time at 1 GOPS (0.001024 ms). Then the `AC` plan on wide-array: H2D, D2H, syncs, peak device bytes.
- [ ] Before running anything, predict the `AACAAAA` transfer bytes for the digit CNN on fragmented-array (each tensor, each direction). Then run `capstone.py execute` and compare.
- [ ] Recompute two rows of the digit-MLP `ACCCC` estimate on wide-array from the formulas.
- [ ] Explain one happens-before rule (e.g. "a COMPUTE's completion happens-before the host reads its output") and show where the code enforces it.
- [ ] In a scratch copy, inject one bug (delete a SYNC, swap a COPY direction, change `2 * m * n * k` to `m * n * k`, or remove a rule in `target_rule_rejection`). Predict which test fails, run the tests, and find the bug from the failure message alone.
- [ ] Explain why scenario ranges are not confidence intervals, in two sentences.
- [ ] Explain one winner flip from `compare-targets` on `edge-depthwise-64`.
- [ ] Explain the defect the cross-check found on the skip-connection graph and how it was fixed.
- [ ] State accuracy vs agreement for the trained example with the actual numbers.
- [ ] Explain what the hardware roadmap would add and why simulator agreement is not hardware evidence.
- [ ] Give the 2-minute explanation aloud, timed, without notes.
- [ ] Write down, in your own words, which parts of the code and documents you personally wrote, changed or reviewed. Be ready to defend that list.

## Targeted references and relation to existing systems

For module 13, read the C++ draft's [shared future state, paragraphs 6–9](https://eel.is/c++draft/futures.state) for readiness and publication, then [condition-variable wait, paragraphs 8–13](https://eel.is/c++draft/thread.condition.condvar) for unlocking, spurious wakeups and predicate loops. Match those rules to the mutex, promise and wait predicate in `src/async.cpp`; do not infer race freedom solely from using these library types. For module 15, the backend-specific official sources and profiler limitations are linked in `BOARD_VALIDATION.md` and `PRODUCTION_COMPARISON.md`. For module 14, read the equations and caveats in `STUDY.md` before relating this bounded implementation to published placement systems.

The following primary documentation was checked on 4 October 2026. Read the named sections after the corresponding code; installing these systems is not part of this guide.

| Reference and section | Read beside | Connection and limit |
|---|---|---|
| [LLVM Language Reference: Introduction and Identifiers](https://llvm.org/docs/LangRef.html#identifiers) | Modules 1–2 | Introduces typed, single-assignment IR. Our own structs use unique tensor definitions; they are not LLVM IR and do not implement LLVM's full SSA/control-flow model. |
| [MLIR Pass Infrastructure: Operation Pass, Analysis Management, IR Printing](https://mlir.llvm.org/docs/PassManagement/#ir-printing) | Modules 3 and 9 | Distinguishes analyses/transformations and documents before/after IR inspection. Our snapshots follow that established practice with a much smaller direct-call pipeline. |
| [ONNX Runtime Execution Providers: opening `GetCapability()` explanation and APIs for Execution Provider](https://onnxruntime.ai/docs/execution-providers/) | Modules 4–5 | Real execution providers identify supported nodes/subgraphs. Our capability/placement functions illustrate related responsibilities; we do not implement an ONNX provider or its kernel ecosystem. |
| [ONNX Runtime I/O Binding: opening explanation and C++ example](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) | Modules 6–7 | Explains how device residency and input/output copies affect a run. It motivates making copies visible; it does not validate our cost constants. |
| [Goldberg, What Every Computer Scientist Should Know About Floating-Point Arithmetic: Rounding Error, Relative Error and Ulps, Cancellation](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) | Modules 8 and 10 | Explains finite representation and numerical error. Apply that reasoning to float accumulation and tolerance checks, rather than treating every small difference as a compiler defect. |

[IREE](https://iree.dev/) documents a broader compiler/runtime system spanning scheduling and execution. It is another implementation to study, rather than a capability this prototype replaces. Existing production compiler/runtime systems are much broader. This project is useful when the entire small experiment needs to remain visible, but has not been shown superior to their diagnostics or execution tooling.

The remaining board experiment uses a fixed MatMul→ReLU→Add family through a verified TI runtime path, separating compile/load/setup, transfers, compute and synchronization where the vendor exposes them. Keep calibration and evaluation separate, retain raw repetitions and actual offload evidence, and compare simulator assumptions with observations. The Mac OpenCL study is supplementary; it does not complete that board experiment.
