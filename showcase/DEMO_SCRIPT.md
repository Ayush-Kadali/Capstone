# Five-minute recording guide

Goal: explain one cause-and-effect chain with real terminal output. Rehearse once before recording. You can record the terminal with your normal screen recorder, speak over it, and upload the result as an unlisted YouTube video. The automatic demo script is also suitable for a silent rehearsal; it does not create a video or upload anything.

## Before recording

Run `bash scripts/run_demo.sh` from the repository root. Confirm all checks pass. Increase terminal font size and hide unrelated windows. Keep the graph file and this script nearby. You only need the commands below; avoid reading every command descriptor aloud.

## 0:00–0:35 — Problem and scope

“We are Boundary Conditions. Our capstone studies the software between AI graphs and hardware. PlacementLens is the part we are presenting here: it explains how a graph becomes a CPU and accelerator execution plan. Moving tensors between backends can outweigh compute savings. The prototype exposes those decisions and executes the commands on a functional software device.”

“The costs in this demo are illustrative. We measure host execution separately and are not claiming a real accelerator speedup.”

## 0:35–1:10 — Show the input

```sh
cat examples/fragmented.plg
build/placementlens --help
build/placementlens inspect --graph examples/fragmented.plg --relu-supported
```

“The input is a typed matrix graph: matrix multiplication, ReLU, then Add. Each tensor has an explicit FP32 shape and values. Validation checks shapes, dependencies, and unique definitions. With ReLU enabled, all three operations are supported on the device.”

## 1:10–2:15 — Change one capability, show the cascade

```sh
build/placementlens compare --graph examples/fragmented.plg --relu-supported
build/placementlens compare --graph examples/fragmented.plg
```

“Now I remove ReLU from the device capability profile. The maximal policy puts MatMul and Add on the accelerator, with ReLU on the CPU. That creates two device regions. Its modeled cost is 10.7001 milliseconds, compared with 9 for CPU-only. The cost-based plan instead keeps Add on the CPU, reduces transferred bytes from 112 to 64, and has a modeled cost of 8.1501 milliseconds.”

“The decision comes from enumerating the legal bounded placements and adding compute, transfer, launch, and synchronization costs. It is exact for our declared model and search space; it is not a measured hardware optimum.”

## 2:15–3:10 — Show the actual compiler transformation

```sh
build/placementlens explain --graph examples/fragmented.plg --report artifacts/segfault/run-report.md
```

“This report is generated from the selected plans and actual results. It shows the rules, placements and cost components, then the command stream. Transfer insertion adds copies where the selected backend lacks an operand. Lowering adds allocations and ordered commands. H and D are separate host and device buffers. These are the commands the runtime executes. The JSON also retains before-and-after pass snapshots.”

## 3:10–3:45 — Correctness and defensive errors

```sh
build/placementlens run --graph examples/fragmented.plg
build/placementlens run --graph examples/invalid_shape.plg
```

“The output is one, fourteen, seven, one, in row-major order. We compare it against an independent evaluator of the original graph. Invalid matrix dimensions are rejected before execution. The executor also checks storage, version, shape, synchronization, memory capacity, and device capability.”

The second command is expected to fail. Say why, then move on.

## 3:45–4:20 — Show the decision responds to cost

```sh
build/placementlens compare --graph examples/fragmented.plg --boundary-ms 5
```

“With a higher boundary cost, CPU-only wins the modeled comparison. The system exposes the assumption that changed the decision.”

If time remains, optional:

```sh
build/placementlens compile --graph examples/fragmented.plg --relu-supported --fuse
```

“A guarded MatMul–ReLU fusion removes a single-use intermediate while preserving the original node identities. It refuses shared or externally visible intermediates.”

## 4:20–5:00 — Evidence and next steps

“The verification matrix runs 45 configurations and executes all 591 feasible placements across chain, shared-intermediate and residual-style graphs. Those are functional tests on small generated workloads, not GPU benchmarks. The current runtime is synchronous and its costs are uncalibrated. The next experiment is to collect profiles from a real backend and test whether these placement decisions hold up on different workloads.”

For a longer live walkthrough, use [CLI examples](CLI_WALKTHROUGH.md). For comparisons with existing compiler/runtime tools, use [Engineering notes](ENGINEERING_NOTES.md).
