# Explore one decision at a time

Build and check the project with `bash scripts/run_demo.sh`. Then use the commands below from the repository root. The executable uses explicit options so runs can be repeated; there is no hidden interactive configuration.

## Start with the help and input

```sh
build/placementlens --help
cat examples/fragmented.plg
build/placementlens inspect --graph examples/fragmented.plg
```

Point out the 2x3 and 3x2 input matrices. MatMul produces a 2x2 tensor. ReLU removes negative values; Add adds an equal-shaped matrix. Device support is checked separately from graph validity. ReLU is deliberately absent from the default device profile.

## Ask for the complete explanation

```sh
build/placementlens explain --graph examples/fragmented.plg --report artifacts/segfault/run-report.md --json artifacts/segfault/explained.json
```

Walk through these sections, pausing between them:

1. Input and rules: the actual capability flag, memory budget and cost assumptions.
2. Compiler decisions: validation, optional fusion, capability analysis, placement, transfer insertion and allocation lowering.
3. Placement comparison: CPU, maximal support and cost-selected plans, with separate cost components.
4. Search evidence: all eight assignments considered, four feasible in this case.
5. Selected executable plan: the allocations, copies, compute commands and SYNC actually executed.
6. Execution checks: source-node lineage, numerical agreement and measured host execution time.

The selected map is `ACC`. The final Add stays on the CPU even though the device supports it. Relative to `ACA`, this removes the second trip to the device and reduces copy bytes from 112 to 64. The cost reduction belongs to the illustrative model.

## Change one rule

```sh
build/placementlens compare --graph examples/fragmented.plg --relu-supported
build/placementlens compare --graph examples/fragmented.plg --boundary-ms 5
build/placementlens explain --graph examples/fragmented.plg --device-bytes 80
```

Full ReLU support permits `AAA`, the all-device modeled optimum. Expensive copies make `CCC` preferable under the model. An 80-byte device budget makes maximal placement infeasible because it would retain 112 device bytes, even though individual operators pass the local check. These are three separate experiments, each starting from the default configuration.

## Show a real graph transformation

```sh
build/placementlens explain --graph examples/fragmented.plg --relu-supported --fuse
build/placementlens explain --graph examples/branched.plg --relu-supported --fuse
```

The first graph allows MatMul/ReLU fusion; the shared-intermediate graph refuses it. Show the recorded reason. The fused command retains both original node IDs. For the all-device small chain, fusion removes an intermediate and reduces retained device storage from 112 to 96 bytes. This model does not assign an extra compute-time saving to fusion.

For a literal before/after diff from the recorded report:

```sh
python3 scripts/show_passes.py artifacts/segfault/fragmented.json --policy maximal --pass-name InsertTransfersPass
```

## Challenge the implementation

```sh
build/placementlens verify --graph examples/branched.plg --relu-supported
build/placementlens run --graph examples/invalid_shape.plg
ctest --test-dir build --output-on-failure
```

`verify` executes all feasible placements against the original graph reference. The invalid-shape command is expected to exit with an error. CTest checks the contracts, the CLI and the 45-case generated matrix.

Close by stating the boundary: the prototype executes real arithmetic and checks compiler/runtime contracts on a software device. It needs calibrated profiles and a real backend before making hardware profitability claims.
