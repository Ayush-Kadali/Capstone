"""Source handoff checks use temporary synthetic trees; no board or SDK needed."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_board_handoff as handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.expected = {}
        for name in handoff.REQUIRED:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = ("synthetic source: " + name + "\n").encode()
            path.write_bytes(data)
            self.expected[name] = data

    def package(self, name="output"):
        destination = self.base / name
        handoff.prepare(self.source, destination)
        return destination / "capstone-board-source.tar.gz"

    def rewrite(self, original, transform):
        with tarfile.open(original, "r:gz") as archive:
            entries = [(copy.copy(member), archive.extractfile(member).read()) for member in archive]
        changed = self.base / "changed.tar.gz"
        with tarfile.open(changed, "w:gz") as archive:
            for member, data in transform(entries):
                member.size = len(data) if member.isfile() else 0
                archive.addfile(member, io.BytesIO(data) if member.isfile() else None)
        return changed

    def test_roundtrip_content_and_independent_hashes(self):
        archive_path = self.package()
        verified = handoff.verify_archive(archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            self.assertEqual(len(members), len(handoff.REQUIRED) + 2)
            self.assertTrue(all(m.isfile() and m.uid == m.gid == 0 and m.mtime == 0 for m in members))
            for name, data in self.expected.items():
                self.assertEqual(archive.extractfile(name).read(), data)
            for row in verified["files"]:
                data = archive.extractfile(row["path"]).read()
                self.assertEqual(row["bytes"], len(data))
                self.assertEqual(row["sha256"], hashlib.sha256(data).hexdigest())
        source_records = [r for r in verified["files"] if r["path"] in handoff.REQUIRED]
        encoded = json.dumps(source_records, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(verified["source_identity_sha256"], hashlib.sha256(encoded).hexdigest())
        receipt = json.loads((archive_path.parent / "receipt.json").read_text())
        self.assertEqual(receipt["sha256"], hashlib.sha256(archive_path.read_bytes()).hexdigest())
        self.assertFalse(receipt["hardware_validation_performed"])
        self.assertEqual(verified, json.loads((archive_path.parent / "manifest.json").read_text()))

    def test_allowlist_excludes_private_binaries_and_nested_output(self):
        for name in (".git/config", "build/placementlens", "artifacts/private.json", "research/paper.pdf", "src/private.hpp"):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("MUST NEVER SHIP")
        out = self.source / "artifacts/handoff"
        handoff.prepare(self.source, out)
        with tarfile.open(out / "capstone-board-source.tar.gz", "r:gz") as archive:
            self.assertEqual(set(archive.getnames()), set(handoff.REQUIRED) | {handoff.GENERATED, handoff.MANIFEST})
            self.assertTrue(all(b"MUST NEVER SHIP" not in archive.extractfile(m).read() for m in archive))

    def test_optional_missing_is_recorded_and_existing_is_hashed(self):
        first = handoff.verify_archive(self.package())
        self.assertEqual(first["missing_optional"], sorted(handoff.OPTIONAL))
        (self.source / "workbench/README.md").write_text("optional instructions")
        second = handoff.verify_archive(self.package("second"))
        self.assertNotIn("workbench/README.md", second["missing_optional"])
        self.assertIn("workbench/README.md", [r["path"] for r in second["files"]])

    def test_source_identity_changes_only_with_captured_source(self):
        first = handoff.verify_archive(self.package())
        second = handoff.verify_archive(self.package("second"))
        self.assertEqual(first["source_identity_sha256"], second["source_identity_sha256"])
        (self.source / "src/core.cpp").write_text("changed source")
        third = handoff.verify_archive(self.package("third"))
        self.assertNotEqual(first["source_identity_sha256"], third["source_identity_sha256"])

    def test_required_missing_fails_before_output(self):
        (self.source / "src/core.cpp").unlink()
        with self.assertRaisesRegex(ValueError, "required file missing"):
            self.package()
        self.assertFalse((self.base / "output").exists())

    def test_nonempty_output_preserved(self):
        out = self.base / "output"
        out.mkdir()
        sentinel = out / "keep"
        sentinel.write_text("original")
        with self.assertRaisesRegex(ValueError, "destination must be new or empty"):
            self.package()
        self.assertEqual(sentinel.read_text(), "original")
        self.assertEqual(list(out.iterdir()), [sentinel])

    def test_symlink_file_and_parent_rejected(self):
        core = self.source / "src/core.cpp"
        private = self.base / "private"
        private.write_text("private")
        core.unlink()
        core.symlink_to(private)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.package()
        core.unlink()
        core.write_bytes(self.expected["src/core.cpp"])
        original = self.source / "include"
        moved = self.base / "moved_include"
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.package()

    def test_source_and_archive_size_bounds(self):
        with patch.object(handoff, "MAX_FILE_BYTES", 8):
            with self.assertRaisesRegex(ValueError, "bounded regular file"):
                self.package()
        archive = self.package()
        with patch.object(handoff, "MAX_TOTAL_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "size limit"):
                handoff.verify_archive(archive)

    def test_content_tamper_rejected(self):
        archive = self.package()
        def change(entries):
            return [(m, b"forged" if m.name == "src/core.cpp" else data) for m, data in entries]
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            handoff.verify_archive(self.rewrite(archive, change))

    def test_unsafe_duplicate_and_nonregular_members_rejected_without_extraction(self):
        original = self.package()
        for name, kind in (("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                           ("src/../escape", tarfile.REGTYPE), ("src\\escape", tarfile.REGTYPE),
                           ("src/core.cpp", tarfile.REGTYPE), ("LICENSE", tarfile.SYMTYPE),
                           ("LICENSE", tarfile.LNKTYPE), ("LICENSE", tarfile.DIRTYPE)):
            with self.subTest(name=name, kind=kind):
                def add(entries):
                    member = tarfile.TarInfo(name)
                    member.type, member.linkname = kind, "../../outside"
                    return entries + [(member, b"hostile")]
                with self.assertRaisesRegex(ValueError, "unsafe member"):
                    handoff.verify_archive(self.rewrite(original, add))
        self.assertFalse((self.base / "escape").exists())

    def test_missing_member_and_forged_source_identity_rejected(self):
        original = self.package()
        missing = self.rewrite(original, lambda entries: [(m, d) for m, d in entries if m.name != "src/core.cpp"])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            handoff.verify_archive(missing)
        def forge(entries):
            result = []
            for member, data in entries:
                if member.name == handoff.MANIFEST:
                    manifest = json.loads(data)
                    manifest["source_identity_sha256"] = "0" * 64
                    data = json.dumps(manifest).encode()
                result.append((member, data))
            return result
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            handoff.verify_archive(self.rewrite(original, forge))

    def test_cli_create_verify_and_refuse_overwrite(self):
        script = str(Path(handoff.__file__))
        out = self.base / "cli"
        command = [sys.executable, script, "create", "--source-root", str(self.source), "--out", str(out)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        verified = subprocess.run([sys.executable, script, "verify", str(out / "capstone-board-source.tar.gz")], capture_output=True, text=True)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertTrue(json.loads(verified.stdout)["verified"])
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("E_HANDOFF_OUTPUT", repeated.stderr)

    def test_contradictory_hardware_claim_rejected(self):
        original = self.package()
        def forge(entries):
            result = []
            for member, data in entries:
                if member.name == handoff.MANIFEST:
                    manifest = json.loads(data)
                    manifest["hardware_validation_performed"] = True
                    data = json.dumps(manifest).encode()
                result.append((member, data))
            return result
        with self.assertRaisesRegex(ValueError, "manifest or content hash mismatch"):
            handoff.verify_archive(self.rewrite(original, forge))


if __name__ == "__main__":
    unittest.main()
