"""Inventory does not imply board validation; no hardware is needed for these tests."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from board_probe import is_beagley, read_optional


class ProbeTests(unittest.TestCase):
    def test_identity_distinguishes_older_board(self):
        self.assertTrue(is_beagley("BeagleBoard.org BeagleY-AI"))
        self.assertFalse(is_beagley("BeagleBone AI-64"))
        self.assertFalse(is_beagley("Apple M4"))

    def test_missing_and_bounded_reads_remain_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            self.assertEqual(read_optional(path)["status"], "unavailable")
            path.write_bytes(b"BeagleY-AI\0")
            self.assertEqual(read_optional(path)["text"], "BeagleY-AI")
            self.assertEqual(read_optional(path, 3)["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
