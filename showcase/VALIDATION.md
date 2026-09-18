# Verification record

Verified locally on 18 September 2026 with AppleClang 17, macOS arm64, C++17, CMake, and Python 3.14. Linux and Windows have not been exercised.

## Completed checks

- Release build completes with the enabled warning flags and no compiler warnings.
- `core_contracts`: **337 assertions pass**.
- `cli_contracts`: **29 real CLI invocations pass**, plus a report-inspector invocation, with parsed JSON, trace, model-sum, pass-snapshot, exhaustive-verification, and Markdown-report assertions.
- `placement_matrix`: **45 cases, 591 executed placements, 11,553 commands**, all numerical comparisons passing. Nine generated graphs span three topologies and widths 2, 8 and 16. Five configurations change support, fusion, memory and copy cost.
- AddressSanitizer and UndefinedBehaviorSanitizer Debug build: all three test suites pass on the exercised paths.
- `bash scripts/run_demo.sh`: all five scenes complete, including the expected rejection of invalid matrix dimensions.
- Native `explain` reports are checked for fragmented support, fusion, infeasible maximal placement, expensive copies, and output-file collisions. Saved Markdown agrees with stdout and recorded JSON results.

## What was challenged

Validation rejects unsupported f16 inputs, invalid dimensions, trailing/missing values, undefined references, cycles, duplicate tensor definitions, and incompatible MatMul shapes. Planner checks cover capability constraints, memory feasibility, and decision reversals when copy cost changes.

Runtime negative tests include wrong protocol versions and command IDs, missing synchronization, missing buffers, unsupported device work, wrong output shapes, and duplicate allocation. A validly shaped but numerically wrong command plan is detected by the independent reference comparison.

Numerical tests exercise 24 seeded matrix cases with varying dimensions and values across every feasible placement of the original and fused graphs. Additional tests check shared input residency and fusion refusal for multi-use or graph-output intermediates.

Report tests confirm that every selected result contains named compiler passes with before/after IR, reasons, and consequences. Transfer IR has no allocation commands; execution-plan IR contains concrete allocations. Every measured trace event corresponds to one executed command, all events use the same host thread, and their logical backends are annotated separately.

## Verified demo results

| Configuration | CPU | Maximal | Cost policy |
|---|---|---|---|
| Default, ReLU unsupported | CCC, 9.0000 ms | ACA, 10.7001 ms | ACC, 8.1501 ms |
| ReLU supported | CCC, 9.0000 ms | AAA, 6.9501 ms | AAA, 6.9501 ms |
| Fixed copy cost 5 ms | CCC, 9.0000 ms | ACA, 32.9001 ms | CCC, 9.0000 ms |
| ReLU supported and fusion | CC, 9.0000 ms | AA, 6.9501 ms | AA, 6.9501 ms |

All costs in this table are **modeled**. Every selected plan actually computes the expected output `[1,14,7,1]`. Host times are recorded separately and are not benchmarked GPU/NPU performance.

The optional fused full-device plan has 13 commands and 96 retained device bytes versus 15 commands and 112 bytes for its unfused counterpart. This model assigns no additional compute-time saving to fusion; the verified effect is the removed operation/intermediate allocation.

## Reproduce sanitizer checks

```sh
cmake -S . -B /tmp/placementlens-asan -DCMAKE_BUILD_TYPE=Debug -DPLACEMENTLENS_SANITIZERS=ON
cmake --build /tmp/placementlens-asan --parallel 2
ctest --test-dir /tmp/placementlens-asan --output-on-failure
```

## Limits of this evidence

These checks establish behavior for the tested bounded inputs and error paths. They do not prove general correctness, security hardening, portable builds, concurrency safety, hardware profitability, or paper completion. The functional runtime is synchronous and retains buffers until run end. Performance constants are uncalibrated. Larger or cancellation-heavy reductions may require a more developed numerical tolerance policy.

The generated matrix is functional verification on controlled inputs, not a real-model performance benchmark. Full per-case evidence is available by running `python3 scripts/check_matrix.py`; the script saves input graphs, reports, logs, seeds, executed command lines, platform information and a binary hash under `artifacts/verification/`. CTest runs the same matrix under its build directory.
