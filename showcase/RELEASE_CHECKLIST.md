# First-release checklist

Scope agreed by the 4 October 2026 release request: finish the existing synchronous
graph → placement → command plan → execution → report workflow. PlacementLens
names this compiler hackathon component and its executable, not the whole capstone.
This boundary takes precedence over unfinished ambitions in the historical roadmap;
it does not redefine the broader research project.

## Initial assessment and final disposition

| Initial state | Work | Final disposition |
|---|---|---|
| Implemented | Bounded graph validation, fusion, capability, placement, transfers, allocations, execution, reference and reporting | Retained and tested together |
| Broken (reproduced during review) | Wrapped signed dimensions; fusion masking overflow; nonfinite reference breaking JSON; malformed public input storage; invalid UTF-8 report source; size check after unrestricted reading | Fixed with focused regressions; see defect table in `VALIDATION.md` |
| Missing | Current clean-build and independent verification evidence | Fresh Release and ASan/UBSan proof pack, 54 CLI cases, exact-rational/residency oracle, deterministic report checks |
| Missing | Complete code-linked teaching route | Expanded existing `ENGINEERING_NOTES.md`; foundations, ten module contracts, exercises, separate answers, short and ten-session routes |
| Deferred | Real accelerator measurements, calibrated costs, measured oracle, profile-error study, concurrency, buffer reuse, tiling, GUI, general importer, extra operators | Outside this bounded first release; not represented as implemented |

## Completed acceptance gates

- [x] Establish baseline: all three original suites passed; preserve intake changes.
- [x] Fix reproduced defects with parser, numerical, runtime and JSON regressions.
- [x] Pass all four suites in clean Release and Debug ASan/UBSan builds.
- [x] Check compiler warnings and sanitizers; disclose unavailable platform/concurrency checks.
- [x] Exercise five-scene demo and 27 distinct documented commands in a clean copy.
- [x] Save commands, configuration, seeds, environment, source/binary hashes and logs.
- [x] Verify reports against executed commands and distinguish model/host/hardware provenance.
- [x] Check the learning guide against final code and the actual 13-command ACC plan.
- [x] Review the final diff; record limits and one bounded hardware follow-up.

[Validation and defect evidence](VALIDATION.md) supplies the counts, reproduction
command and limitations. [Engineering guide](ENGINEERING_NOTES.md) supplies the
code-linked explanation. Local raw evidence is in `artifacts/release-2026-10-04/`.

The release is complete within this boundary. Real hardware, whole-roadmap
completion, research novelty, learner understanding and authorship are not claims
of these tests. Existing optional presentation/video helpers and historical
academic submissions were preserved and excluded from headless acceptance.

At intake, tracked changes removed version labels from `CMakeLists.txt` and
`src/main.cpp`; those remain. Other pre-existing untracked files remain intact.
No private career/networking/IP material or academic submission was edited.
No commit, push, publication, purchase or paid compute was performed.
