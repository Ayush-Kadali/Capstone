# First-release verification record

Verified on **4 October 2026 (Asia/Kolkata)** for the existing compiler hackathon
component, whose executable is named PlacementLens. PlacementLens is not the name
of the broader capstone. This record covers the bounded synchronous workflow in
[the release checklist](RELEASE_CHECKLIST.md), not the historical full roadmap.

## Reproduce from current sources

From the repository root, run
`python3 scripts/verify_release.py --out artifacts/release-check`.
The destination must not exist. The script copies release sources into a fresh
workspace, builds Release, runs all tests and the five-scene demo, executes every
unique shell command in the runnable README/showcase/learning guides, and builds
and tests Debug with AddressSanitizer and UndefinedBehaviorSanitizer. It preserves
the original build and presentation artifacts.

Individual sanitizer checks remain available:

```sh
cmake -S . -B /tmp/placementlens-asan -DCMAKE_BUILD_TYPE=Debug -DPLACEMENTLENS_SANITIZERS=ON
cmake --build /tmp/placementlens-asan --parallel 2
ctest --test-dir /tmp/placementlens-asan --output-on-failure
```

For a clean standalone repetition use a new build directory. The release harness
uses its own fresh `build-asan` directory with these same settings.

## Environment and evidence

The recorded run used macOS 15.7.3 arm64, AppleClang 17.0.0.17000604, C++17,
CMake 4.2.3 and Python 3.14.6. Raw tool-version output is authoritative.

Local evidence is under `artifacts/release-2026-10-04/`:

- `initial-status.txt`, `pre-existing.patch`: intake state; existing version-label
  removals and unrelated untracked files were preserved.
- `baseline-configure.log`, `baseline-build.log`, `baseline-tests.log`: all three
  original suites passed before repairs (337 core assertions, 29 CLI invocations,
  45 matrix cases). The baseline build was moved under this evidence directory
  after testing; its original CMake paths in logs are historical, not rerun paths.
- `defect-baseline/`: concrete input files, original binary output and recorded
  exit statuses for signed-dimension, fused-overflow and invalid-JSON failures.
- `final/manifest.json`: complete source snapshot hashes, environment, every
  verification command, working directory, expected/actual exit status, log path,
  and Release/ASan binary hashes; overall status `passed`.
- `final/workspace/build/Testing/Temporary/LastTest.log`: detailed Release results.
  The corresponding `build-asan` path holds sanitizer results.
- `final/workspace/build/verification/`: 45-case matrix inputs, per-case reports,
  logs, seeds, configurations, command lines, platform and binary hash summary.
- `final/workspace/build/differential/`: independent generated inputs, reports,
  seed and summary. The generator/commands are preserved in the source snapshot.
- `final/workspace/artifacts/segfault/`: actual five-scene output, JSON, Markdown
  and measured host trace; learning-guide commands also save their own reports.

These artifacts are intentionally ignored by Git and are regenerable. No commit
or publication was made. Later documentation-only corrections are recorded in
`final-documentation-sha256.json`; code hashes must still match the tested snapshot.

## Acceptance evidence

| Gate | Observed result |
|---|---|
| Reproducible build | Fresh source/build copy configured and built in Release and Debug ASan/UBSan; no compiler warnings in build logs |
| Unit and negative contracts | **365 core assertions** pass in each build |
| CLI and report integration | **54 CLI invocations**, plus report/trace assertions and inspector invocation, pass in each build |
| Existing exhaustive matrix | **45 configurations, 591 placements, 11,553 commands** pass in each build |
| Independent differential/property checks | **18 seeded DAGs, 54 configurations, 580 placements** checked with exact Python rational arithmetic, independently derived feasible sets, transfer bytes and retained capacity; another 54 invocations verify deterministic report fields |
| Documented workflow | Five-scene demo plus **27 unique documented commands**; intended invalid-shape command exits 1; all other expected commands succeed |
| Report consistency | Commands sum to recorded model totals, trace events correspond to executed commands, source lineage survives fusion, Markdown agrees with JSON, outputs agree with reference where success is reported |
| Learning guide | Existing `ENGINEERING_NOTES.md` expanded and checked against final functions and the emitted 13-command ACC plan; includes exercises, separate answers and study routes |
| Final review | Independent core and CLI reviews, source/diff review and whitespace checks; no unresolved reproduced core defect |

Enabled warnings: `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`.
Both sanitizer and Release runs execute all four CTest suites. No sanitizer
finding occurred on those paths. The original 24-seed C++ numerical suite checks
all placements before and after fusion. The additional Python suite uses seed
`20261004` and unfused small DAGs; matrix seeds are `20260918 + width`.

## Reproduced defects and repairs

These are observations made in this release pass, not invented development history.

| Symptom before repair | Cause and repair | Regression evidence |
|---|---|---|
| A dimension `-18446744073709551615` became 1 and execution passed on the 64-bit host | Unsigned extraction accepted a negative value modulo the unsigned range; parse digit-only bounded decimal dimensions | Core/CLI signed, overflowing, fractional and zero dimension rejection |
| Fused ReLU turned nonfinite MatMul into finite zero and could report PASS | Finiteness was checked only after clamping; check MatMul result before activation | Negative-infinity and NaN cases across original/fused placements |
| A finite runtime result with an overflowing reference wrote bare `inf` into JSON | Unusable reference reached error serialization; reject nonfinite FP32 reference intermediates with `E_NUMERIC` | Core oracle-overflow fixture; CLI confirms exit 1 and no new report |
| Mutating public Graph input storage could overrun an output vector | Executor trusted shape/storage agreement; validate a defensive source copy before kernel access | Oversized/empty storage, zero shape and nonfinite input regressions under ASan/UBSan |
| An invalid byte in a comment yielded undecodable JSON despite exit 0 | Raw source is retained in pass snapshots; CLI now rejects invalid UTF-8 | Invalid leading/continuation bytes, overlongs, surrogates, truncation and out-of-range code points; valid Unicode preserved |
| The CLI could allocate the entire input before enforcing its size cap | Read into bounded chunks and reject before appending excess bytes; check stream errors | 16,000,001-byte input and unreadable graph regressions |

Finite numerical disagreement remains distinct from execution failure: a
cancellation-heavy `[1e8,1,-1e8]` dot `[1,1,1]` produces runtime 0 versus reference
1, returns exit 2, and writes valid JSON with `correct:false`. Rejected execution
returns exit 1. A failed invocation does not refresh prior report files; use exit
status to avoid mistaking stale output for current evidence.

## Demonstrated results

| Configuration | CPU | Maximal | Cost policy |
|---|---|---|---|
| Default, ReLU unsupported | CCC, 9.000000 ms | ACA, 10.700112 ms | ACC, 8.150064 ms |
| ReLU supported | CCC, 9.000000 ms | AAA, 6.950080 ms | AAA, 6.950080 ms |
| Fixed copy cost 5 ms | CCC, 9.000000 ms | ACA, 32.900112 ms | CCC, 9.000000 ms |
| ReLU supported and fusion | CC, 9.000000 ms | AA, 6.950080 ms | AA, 6.950080 ms |

All times in this table are **illustrative modeled costs**. All listed executions
produce `[1,14,7,1]`. Actual command durations are separately measured host times;
there are no accelerator measurements. The all-device fused plan retains 96
bytes in 13 commands, versus 112 bytes in 15 commands without fusion. This model
assigns no extra compute-time saving to fusion.

## Limits and deferred checks

This release is synchronous, retains buffers until run end, accepts a bounded
rank-two FP32 graph language and uses shape-independent compute constants.
Enumeration is exhaustive only for the compiled graph and fixed schedule.
`verify` has frontend and transfer/lowering snapshots; selected-policy reports
also have capability/placement narratives. Timing fields are nondeterministic.
The executor's timing excludes validation, initial input setup and reference
checking. It includes command validation, allocations, arithmetic, copies,
event bookkeeping and output extraction.

Linux/Windows builds, ThreadSanitizer/concurrency, external accelerator execution,
calibrated costs and an empirical measured oracle were **not verified**. There is
no asynchronous path to claim race safety for. Sanitizers and bounded property
checks are not a proof of all inputs or general security hardening. Existing
optional video/presentation helpers and historical academic submissions are
outside this headless acceptance scope and were preserved. Learner understanding
and team authorship have not been evaluated by these automated checks.

## One hardware follow-up

On one already available real accelerator backend, run the same MatMul→ReLU→Add
chain at a few small matrix sizes. Compare CPU-only against full offload, retain
outputs and raw repetitions, and separately measure input copy, compute, output
copy and required synchronization. Use separate calibration/evaluation sizes and
compare measured end-to-end ordering with the model. Do not add a second backend
or begin paid compute as part of this release.
