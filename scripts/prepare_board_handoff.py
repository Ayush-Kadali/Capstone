"""Create or verify a bounded source-only board handoff. Never installs or runs it."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import stat
import sys
import tarfile


REQUIRED = (
    "CMakeLists.txt", "README.md",
    "include/placementlens/async.hpp", "include/placementlens/core.hpp",
    "include/placementlens/search.hpp", "include/placementlens/cnn.hpp", "include/placementlens/device.hpp", "include/placementlens/kernels.hpp",
    "src/search.cpp", "src/cnn.cpp", "src/async.cpp", "src/core.cpp", "src/kernels.cpp", "src/main.cpp",
    "src/opencl_device.cpp", "src/report.cpp",
    "scripts/bounded_study.py", "scripts/board_probe.py", "scripts/check_matrix.py", "scripts/study.py",
    "scripts/compare_production.py", "scripts/investigate_memory.py",
    "scripts/explain_run.py", "scripts/show_passes.py", "scripts/run_demo.sh",
    "scripts/prepare_board_handoff.py", "scripts/ti_onnx_adapter.py",
    "scripts/onnx_frontend.py", "scripts/explore_model.py", "scripts/target_profile.py",
    "scripts/capstone.py", "scripts/developer_report.py", "tests/developer_cli_tests.py",
    "scripts/make_onnx_examples.py", "scripts/export_torch_example.py", "requirements-onnx.txt",
    "profiles/software-cnn-relu-on.json", "profiles/software-device-relu-off.json", "profiles/software-device-relu-on.json",
    "tests/async_tests.cpp", "tests/board_probe_tests.py", "tests/core_tests.cpp",
    "tests/search_cli_tests.py", "tests/bounded_study_tests.py", "tests/cli_tests.py", "tests/device_dispatch_tests.cpp", "tests/differential_tests.py",
    "tests/search_tests.cpp", "tests/cnn_tests.cpp", "tests/kernel_tests.cpp", "tests/memory_tests.cpp", "tests/opencl_tests.cpp",
    "tests/production_tests.py", "tests/residency_tests.cpp", "tests/runtime_review_tests.cpp",
    "tests/study_tests.py", "tests/workbench_tests.py", "tests/board_handoff_tests.py",
    "tests/onnx_frontend_tests.py", "tests/onnx_review_tests.py",
    "tests/model_explorer_tests.py", "tests/target_profile_tests.py", "tests/ti_adapter_tests.py",
    "examples/add.plg", "examples/branched.plg", "examples/fragmented.plg",
    "examples/invalid_shape.plg", "examples/unsupported_dtype.plg",
    "workbench/app.js", "workbench/index.html", "workbench/server.py", "workbench/style.css",
    "showcase/ENGINEERING_NOTES.md",
    "docs/runtime/ARCHITECTURE.md", "docs/runtime/ASYNC.md",
    "docs/runtime/BOARD_VALIDATION.md", "docs/runtime/BOARD_HANDOFF.md",
    "docs/runtime/COMPLETION_MATRIX.md", "docs/runtime/DEBUGGING_DOSSIER.md",
    "docs/runtime/DEMONSTRATION.md", "docs/runtime/DEVICE_CONTRACT.md",
    "docs/runtime/INTEGRATION_REVIEW.md", "docs/runtime/KERNELS.md",
    "docs/runtime/MEMORY.md", "docs/runtime/PERFORMANCE_INVESTIGATION.md",
    "docs/runtime/PRODUCTION_COMPARISON.md", "docs/runtime/RESIDENCY.md",
    "docs/runtime/BOUNDED_SEARCH.md", "docs/runtime/STUDY.md",
    "docs/runtime/ONNX_FRONTEND.md", "docs/runtime/TARGET_PROFILES.md", "docs/runtime/TI_ADAPTER.md",
    "docs/runtime/DEVELOPER_CLI.md", "docs/runtime/PRODUCT_READINESS.md",
    "planning/MODEL_TO_HARDWARE_GOAL.md",
)
OPTIONAL = ("workbench/README.md", "LICENSE", "LICENSE.txt")
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
GENERATED = "HANDOFF_README.md"
MANIFEST = "HANDOFF_MANIFEST.json"


README = """# Portable board source candidate

This archive contains source, tests, documentation and a prepared TI ONNX adapter,
not a board image or validated TI accelerator integration. No installation, flashing, build or hardware validation was
performed by packaging it. PlacementLens names the compiler executable only.
The whole capstone's BeagleY-AI/TI offload and measurement gates remain open.

The manifest records captured source bytes and their SHA-256 hashes. Its source
identity is a digest of the sorted source file records, not a clean Git commit
or a signature. Documentation links to historical artifacts/planning material
may be unavailable here: those files are intentionally outside this archive.

After transferring and unpacking the reviewed archive into a fresh directory,
run from that directory. Required dependencies: CMake >=3.16, a C++17 compiler,
native build tool and thread support. Python >=3.9 is needed for the Python
tests/tools. Packaging does not install these dependencies.

```sh
python3 scripts/board_probe.py --out board-readiness.json
cmake -S . -B build-board -DCMAKE_BUILD_TYPE=Release -DPLACEMENTLENS_OPENCL=OFF
cmake --build build-board --parallel 2
ctest --test-dir build-board --output-on-failure -R '^(core_contracts|async_contracts|device_dispatch_contracts|kernel_contracts|memory_contracts|residency_contracts|runtime_review_contracts|cli_contracts|placement_matrix|differential_contracts|measured_study_contracts|board_probe_contracts)$'
python3 tests/board_handoff_tests.py
build-board/placementlens verify --graph examples/fragmented.plg --reuse-buffers --tile-size 2 --async --json board-software-results.json
```

The probe is read-only inventory; exit 77 means target identity was not observed.
Even exit 0 does not prove inference/offload. Preserve its output and record the
actual SD image checksum and matching SDK/runtime separately. The software
verification above runs CPU host arithmetic, including its logical device lane.

The complete registered test set is `ctest --test-dir build-board
--output-on-failure`. It additionally requires Node.js for workbench JavaScript
checks. PyTorch is optional: production_reference_contracts skips with code 77
only when torch is absent; a broken installation or test failure is not a skip.
ONNX import/reference suites are optional and skip with code 77 when their
dependencies are missing. See requirements-onnx.txt and ONNX_FRONTEND.md; package
compatibility on the board must be checked before installing. Profiles initially
describe hypothetical software devices, not TI hardware capability.
OpenCL is disabled by default, so opencl_contracts skips with 77. Enabling OpenCL
requires its actual headers/runtime and is not a TI C7x/MMA implementation.

The optional local workbench needs Python, and a browser for interaction:

```sh
python3 workbench/server.py --binary build-board/placementlens --out artifacts/board-workbench --port 8765
```

It binds to loopback. On another computer use an explicitly configured SSH
tunnel; this package does not expose a network service or configure access.
See docs/runtime/BOARD_VALIDATION.md for the unresolved TI adapter/offload,
numerical reference and repeated-measurement gates, and docs/runtime/BOARD_HANDOFF.md for
archive verification. Broader BSP scope and personal technical defense remain
separate acceptance items in the completion matrix.
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def safe_name(name):
    path = PurePosixPath(name)
    return (isinstance(name, str) and bool(name) and not path.is_absolute()
            and str(path) == name and "\\" not in name
            and all(part not in (".", "..") for part in path.parts))


def file_record(name, data):
    return {"path": name, "bytes": len(data), "sha256": digest(data)}


def capture(root):
    root = Path(root).resolve(strict=True)
    captured, missing = {}, []
    for name in sorted(REQUIRED + OPTIONAL):
        path = root / name
        for parent in [path, *path.parents]:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError(f"E_HANDOFF_SOURCE: symlink is not allowed: {name}")
        try:
            info = path.stat()
        except FileNotFoundError:
            if name in OPTIONAL:
                missing.append(name)
                continue
            raise ValueError(f"E_HANDOFF_SOURCE: required file missing: {name}") from None
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError(f"E_HANDOFF_SOURCE: not a bounded regular file: {name}")
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"E_HANDOFF_SOURCE: file exceeds limit: {name}")
        captured[name] = data
    if sum(map(len, captured.values())) > MAX_TOTAL_BYTES:
        raise ValueError("E_HANDOFF_SOURCE: source total exceeds limit")
    return captured, missing


def verify_archive(path):
    """Validate names/types/bounds and every hash in memory. Never extract files."""
    entries, total = {}, 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = member.name
            if (not safe_name(name) or name in entries or not member.isfile()
                    or name not in set(REQUIRED + OPTIONAL + (GENERATED, MANIFEST))):
                raise ValueError("E_HANDOFF_ARCHIVE: unexpected, duplicate or unsafe member")
            total += member.size
            if member.size < 0 or member.size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("E_HANDOFF_ARCHIVE: size limit exceeded")
            entries[name] = archive.extractfile(member).read()
    if MANIFEST not in entries or GENERATED not in entries:
        raise ValueError("E_HANDOFF_ARCHIVE: generated metadata missing")
    manifest = json.loads(entries.pop(MANIFEST))
    if not isinstance(manifest, dict):
        raise ValueError("E_HANDOFF_ARCHIVE: manifest must be an object")
    expected = [file_record(name, data) for name, data in sorted(entries.items())]
    source = [record for record in expected if record["path"] != GENERATED]
    missing = sorted(set(OPTIONAL) - entries.keys())
    if (manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "portable-board-source"
            or manifest.get("hardware_validation_performed") is not False
            or manifest.get("files") != expected
            or manifest.get("source_identity_sha256") != digest(canonical(source))
            or manifest.get("missing_optional") != missing
            or not set(REQUIRED).issubset(entries)):
        raise ValueError("E_HANDOFF_ARCHIVE: manifest or content hash mismatch")
    return manifest


def prepare(root, out):
    out = Path(out)
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError("E_HANDOFF_OUTPUT: destination must be new or empty")
    captured, missing = capture(root)
    source = [file_record(name, data) for name, data in sorted(captured.items())]
    captured[GENERATED] = README.encode()
    manifest = {"schema_version": 1, "artifact_type": "portable-board-source",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_identity_sha256": digest(canonical(source)),
                "identity_scope": "captured allowlisted source bytes; not a clean commit or signature",
                "packaging_host": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
                "hardware_validation_performed": False,
                "missing_optional": missing,
                "files": [file_record(name, data) for name, data in sorted(captured.items())]}
    metadata = json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    captured[MANIFEST] = metadata
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "capstone-board-source.tar.gz"
    # Exclusive creation prevents replacing evidence even if another writer won.
    with archive_path.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", format=tarfile.USTAR_FORMAT) as archive:
            for name, data in sorted(captured.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))
    verify_archive(archive_path)
    with (out / "manifest.json").open("xb") as stream:
        stream.write(metadata)
    receipt = {"archive": archive_path.name, "sha256": digest(archive_path.read_bytes()),
               "source_identity_sha256": manifest["source_identity_sha256"], "verified": True,
               "hardware_validation_performed": False}
    with (out / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create")
    create.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    create.add_argument("--out", type=Path, required=True)
    verify = actions.add_parser("verify")
    verify.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.action == "create":
        print(json.dumps(prepare(args.source_root, args.out), indent=2))
    else:
        result = verify_archive(args.archive)
        print(json.dumps({"verified": True, "files": len(result["files"]),
                          "source_identity_sha256": result["source_identity_sha256"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, tarfile.TarError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
