"""Tests of report.py, the NVIDIA session's judge: masking, request-by-request comparison, ctest logs, the table."""
import tempfile
import unittest
from pathlib import Path

import report as R

RUN = """>>> start: --pack x
INFO vram_free_mib=9000 expert_slots=512
READY
>>> GEN 24 <40 ids>
PP 40 40 120.5 331.9
T 11
T 12
DONE 2 40 120.5 33.1 length 7 3 0
>>> BGEN 0 12 <41 ids>
BDONE 0 12 length 812.3
>>> SAVE C:/r/before-1.out.ses
SAVED 300 1 41.0
exit 0
"""


class Mask(unittest.TestCase):
    def test_measured_parts_and_file_names_are_masked(self):
        a = R.mask(RUN)
        b = R.mask(RUN.replace("120.5 331.9", "99.0 inf").replace("33.1 length 7 3", "40.2 length 9 9")
                   .replace("812.3", "700.0").replace("9000", "8000").replace("before-1", "after-2")
                   .replace("41.0", "12.5"))
        self.assertEqual(a, b)
        self.assertIn("DONE 2 40 _ _ length\n", a)
        self.assertIn("T 12\n", a)

    def test_durations_and_rates_are_masked_but_counts_are_not(self):
        self.assertEqual(R.mask("filled in 3.2 s, 512 slots, 1.5 GB/s, 40 ms"),
                         "filled in _ s, 512 slots, _ GB/s, _ ms")


class CompareRuns(unittest.TestCase):
    def test_same_tokens_pass(self):
        status, note = R.compare_runs([RUN, RUN], [RUN, RUN])
        self.assertEqual(status, "PASS")
        self.assertIn("4 of 4 parts the same", note)

    def test_a_token_that_differs_after_is_a_diff(self):
        status, note = R.compare_runs([RUN, RUN], [RUN, RUN.replace("T 12", "T 13")])
        self.assertEqual(status, "DIFF")
        self.assertIn("differ: >>> GEN 24 <40 ids>", note)

    def test_a_request_the_build_before_does_not_repeat_is_not_compared(self):
        noisy = RUN.replace("T 12", "T 14")
        status, note = R.compare_runs([RUN, noisy], [RUN.replace("T 12", "T 15")])
        self.assertEqual(status, "PASS")
        self.assertIn("1 not compared", note)

    def test_a_run_that_did_not_finish_fails(self):
        status, note = R.compare_runs([RUN], [RUN.replace("exit 0", "STOPPED: the engine ended early\nexit 1")])
        self.assertEqual((status, note), ("FAIL", "did not finish: after-1"))

    def test_a_slower_decode_beyond_the_tolerance_is_a_diff(self):
        fast = RUN.replace("DONE 2 40 120.5 33.1", "DONE 128 40 120.5 10000")      # 12.8 tok/s
        slow = RUN.replace("DONE 2 40 120.5 33.1", "DONE 128 40 120.5 12000")      # 10.7 tok/s
        self.assertEqual(R.compare_runs([fast], [slow])[0], "PASS")
        status, note = R.compare_runs([fast], [slow], tolerance=5)
        self.assertEqual(status, "DIFF")
        self.assertIn("decode 12.8 -> 10.7 tok/s (-16.7%)", note)

    def test_server_speed_lines(self):
        self.assertEqual(R.speeds("SPEED long decode=20.5 prompt=300.0\n"), {"decode": [20.5], "prompt": [300.0]})


class Ctest(unittest.TestCase):
    LOG = """      Start  1: conversation_cache_test
 1/5 Test  #1: conversation_cache_test ..........   Passed    0.05 sec
 2/5 Test  #2: vmm_test .........................***Failed    1.10 sec
 3/5 Test  #3: expert_cache_segmented_test ......***Skipped   0.02 sec
 4/5 Test  #4: decode_cluster_parity ............   Passed    0.40 sec
 5/5 Test  #5: verify_parity ....................***Exception: SegFault  0.12 sec
"""

    def test_parse(self):
        self.assertEqual(R.parse_ctest(self.LOG), {
            "conversation_cache_test": "Passed", "vmm_test": "Failed", "expert_cache_segmented_test": "Skipped",
            "decode_cluster_parity": "Passed", "verify_parity": "Exception: SegFault"})

    def test_compare(self):
        before = R.parse_ctest(self.LOG)
        status, note = R.compare_ctest(before, before)
        self.assertEqual(status, "FAIL")
        self.assertIn("fail in both: verify_parity, vmm_test", note)
        self.assertIn("below sm_90", note)
        after = dict(before, conversation_cache_test="Failed")
        del after["vmm_test"]
        status, note = R.compare_ctest(before, after)
        self.assertEqual(status, "DIFF")
        self.assertIn("conversation_cache_test (Passed -> Failed), vmm_test (Failed -> missing)", note)
        self.assertEqual(R.compare_ctest(before, {}), ("FAIL", "no test results after the refactor"))


class Summary(unittest.TestCase):
    def test_rows_are_replaced_and_counted(self):
        with tempfile.TemporaryDirectory() as d:
            R.add_row(Path(d), "unit-86", "FAIL", "x")
            R.add_row(Path(d), "smoke-q38", "SKIP", "no model")
            R.add_row(Path(d), "unit-86", "PASS", "204 tests")
            text = R.summary(Path(d))
        self.assertIn("unit-86    PASS    204 tests", text)
        self.assertIn("totals: 1 PASS, 0 DIFF, 0 FAIL, 1 SKIP", text)

    def test_startup_lines(self):
        err = ("strata generate: GPU 0: NVIDIA GeForce RTX 3060\n"
               "strata generate: layer split: CUDA1 runs layers 12-47, expert cache 4096 slots\n")
        self.assertEqual(R.startup_lines(err), err.splitlines()[1] + "\n")


if __name__ == "__main__":
    unittest.main()
