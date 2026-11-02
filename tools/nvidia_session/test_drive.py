"""Checks that the NVIDIA kit requires evidence of a cut move, beyond matching generated tokens."""
import tempfile
import unittest
from pathlib import Path

from drive import get_cut_move_problem


class CutMove(unittest.TestCase):
    def test_requires_a_move_and_successful_resizes(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "after-1.out"
            settings = Path(str(out) + ".settings")
            stderr = Path(str(out) + ".stderr")
            stderr.write_text("", encoding="utf-8")
            self.assertIn("no settings", get_cut_move_problem(out))
            settings.write_text("END_MEMORY\n", encoding="utf-8")
            self.assertIn("no layer cut", get_cut_move_problem(out))
            stderr.write_text("strata-ds41: the layer cut moved: K=20 -> K=20\n", encoding="utf-8")
            self.assertIn("no layer cut", get_cut_move_problem(out))
            stderr.write_text("strata-ds41: the layer cut moved: K=20 -> K=12 (pool resize)\n", encoding="utf-8")
            self.assertEqual("", get_cut_move_problem(out))
            settings.write_text("ERR gpu0 cannot shrink\n", encoding="utf-8")
            self.assertIn("resize failed", get_cut_move_problem(out))
            settings.write_text("SHORTFALL gpu0\nEND_MEMORY\n", encoding="utf-8")
            self.assertIn("resize failed", get_cut_move_problem(out))


if __name__ == "__main__":
    unittest.main()
