#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
mkdir -p artifacts/segfault
printf '\nSCENE 1: All three operators supported; one device region\n'
build/placementlens compare --graph examples/fragmented.plg --relu-supported --json artifacts/segfault/full-support.json | tee artifacts/segfault/full-support.txt
printf '\nSCENE 2: ReLU unsupported; inspect placement and lowering\n'
build/placementlens demo --graph examples/fragmented.plg --json artifacts/segfault/fragmented.json --trace artifacts/segfault/trace.json | tee artifacts/segfault/fragmented.txt
printf '\nSCENE 3: Expensive boundaries make CPU-only preferable\n'
build/placementlens compare --graph examples/fragmented.plg --boundary-ms 5 --json artifacts/segfault/expensive-boundaries.json | tee artifacts/segfault/expensive-boundaries.txt
printf '\nSCENE 4: Optional safe fusion with full capability\n'
build/placementlens demo --graph examples/fragmented.plg --relu-supported --fuse --json artifacts/segfault/fused.json | tee artifacts/segfault/fused.txt
printf '\nSCENE 5: Invalid graph rejected before execution (expected error)\n'
if build/placementlens run --graph examples/invalid_shape.plg > artifacts/segfault/rejection.txt 2>&1; then
    printf 'ERROR: invalid graph unexpectedly succeeded\n'
    exit 1
fi
cat artifacts/segfault/rejection.txt
printf '\nWhy the fragmented placements differ:\n'
python3 scripts/explain_run.py artifacts/segfault/fragmented.json
printf '\nSaving the full explanation report:\n'
build/placementlens explain --graph examples/fragmented.plg --report artifacts/segfault/run-report.md --json artifacts/segfault/explained.json > artifacts/segfault/explained.txt
printf '\nDemo complete. Reports, real CLI output and host trace: artifacts/segfault/\n'
