"""pack_gguf must never take an Engram file's metadata for the model's: ISTA-DASLab's final Engram says general.architecture 'deepseek41-engram'
(the first release's said 'engram').  The fixture is the real header of DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf (its 416 bytes: metadata and the tensor
directory, no data); engram-6shard-header.gguf is the header of the same file in ISTA-DASLab's 6-shard layout (2026-10-28), where it is shard 6
of every set: no architecture, the split's keys and one tensor (per_layer_token_embd)."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import pack_gguf as PG  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "dsv41" / "engram-final-header.gguf"
FIXTURE_6 = ROOT / "tests" / "fixtures" / "dsv41" / "engram-6shard-header.gguf"


def tiny_gguf(path: Path, arch: str, extra: dict | None = None) -> Path:
    """A GGUF with only metadata (string and uint32 values), no tensors."""
    kv = {"general.architecture": arch, **(extra or {})}
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)))
    for k, v in kv.items():
        out += struct.pack("<Q", len(k)) + k.encode()
        if isinstance(v, str):
            out += struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v.encode()
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", v)
    out += b"\0" * (-len(out) % 32)
    path.write_bytes(bytes(out))
    return path


class EngramHeader(unittest.TestCase):
    def test_the_real_final_engram_header_is_an_engram_file(self):
        g = PG.Gguf(FIXTURE)
        self.assertEqual(g.kv["general.architecture"], "deepseek41-engram")
        self.assertTrue(PG.is_engram_file(g))
        self.assertEqual(len(g.tensors), 2)
        self.assertEqual(g.data_start, 416)

    def test_it_is_never_the_group_the_model_metadata_comes_from(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            shard = PG.Gguf(tiny_gguf(d / "model-00001-of-00004.gguf", "deepseek41", {"split.count": 4}))
            engram = PG.Gguf(FIXTURE)
            for order in ([engram, shard], [shard, engram]):
                kv = PG.model_kv(order)
                self.assertIsNotNone(kv)
                self.assertEqual(kv["general.architecture"], "deepseek41")
            self.assertIsNone(PG.model_kv([engram]))                    # an Engram alone is not a model

    def test_the_first_releases_engram_is_one_too(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(PG.is_engram_file(PG.Gguf(tiny_gguf(Path(d) / "e.gguf", "engram"))))
            self.assertFalse(PG.is_engram_file(PG.Gguf(tiny_gguf(Path(d) / "m.gguf", "deepseek41"))))


class SixShardLayout(unittest.TestCase):
    """ISTA-DASLab's 6-shard layout: shards 1-5 of the split are the set's files, shard 6 is the shared Engram file (no ...-00006-of-00006.gguf)."""

    def test_shard_6_is_the_engram_and_holds_the_same_tables_as_the_4_shard_one(self):
        g, old = PG.Gguf(FIXTURE_6), PG.Gguf(FIXTURE)
        self.assertTrue(PG.is_engram_file(g))
        self.assertEqual((g.kv["split.no"], g.kv["split.count"]), (5, 6))
        (name, typ, dims, off), = g.tensors
        self.assertEqual((name, off, g.data_start), ("per_layer_token_embd.weight", 0, old.data_start))
        self.assertEqual(dims[1], sum(t[2][1] for t in old.tensors))       # one table of both layers' rows: the same bytes from the same offset
        self.assertEqual(old.tensors[1][3], old.tensors[0][2][1] * 256 // 32 * 18)

    def pack(self, d: Path, *extra) -> str:
        names = [tiny_gguf(d / f"m-{i:05d}-of-00006.gguf", "deepseek41", {"split.no": i - 1, "split.count": 6}).name for i in range(1, 6)]
        argv = ["pack_gguf.py", "--model", str(d), "--shards", *names, "--out", str(d / "pack"), *extra]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as cm:
            PG.main()
        return str(cm.exception)

    def test_five_shards_and_the_engram_are_the_whole_split(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("5 of the 6 shards are here", self.pack(Path(d)))
            self.assertNotIn("shards are here", self.pack(Path(d), "--engram", str(FIXTURE_6)))     # (it stops later: the tiny shards hold no tensors)
            self.assertNotIn("shards are here", self.pack(Path(d), "--engram", str(FIXTURE)))       # any version of the Engram


if __name__ == "__main__":
    unittest.main()
