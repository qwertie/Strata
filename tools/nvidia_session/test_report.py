"""Tests of report.py, the NVIDIA session's judge: masking, request-by-request verdicts, speeds, builds, ctest logs,
the samples and the table."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import report as R

RUN = """>>> start: --pack x
INFO vram_free_mib=9000 expert_slots=512
READY
>>> GEN 24 <40 ids 0badf00d>
PP 40 40 120.5 331.9
T 11
T 12
DONE 2 40 120.5 33.1 length 7 3 0
>>> BGEN 0 12 1000,1001
>>> BGEN 1 12 1000,1002
BADM 0 1
BT 0 5
BT 1 6
BT 0 7
BDONE 0 12 length 812.3
BT 1 8
BDONE 1 12 length 800.0
>>> LIST_CONVERSATIONS
CONVERSATION slot0 vram holding 60 32768 1539309568 65617920
END_CONVERSATIONS
>>> SAVE C:/r/before-1.out.ses
SAVED 300 1 41.0
exit 0
"""


def fast(run: str, ms: float) -> str:
    """Gets `run` with a 128-token answer decoded in `ms`."""
    return run.replace("DONE 2 40 120.5 33.1", f"DONE 128 40 120.5 {ms}")


class Mask(unittest.TestCase):
    def test_measured_parts_and_file_names_are_masked(self):
        a = R.mask(RUN)
        b = R.mask(RUN.replace("120.5 331.9", "99.0 inf").replace("33.1 length 7 3", "40.2 length 9 9")
                   .replace("812.3", "700.0").replace("9000", "8000").replace("before-1", "after-2")
                   .replace("41.0", "12.5"))
        self.assertEqual(a, b)
        self.assertIn("DONE 2 40 _ _ length\n", a)
        self.assertIn("T 12\n", a)

    def test_durations_rates_and_free_memory_are_masked_but_counts_are_not(self):
        self.assertEqual(R.mask("filled in 3.2 s, 512 slots, 1.5 GB/s, 40 ms, 7.31 GiB free, 408 MiB free"),
                         "filled in _ s, 512 slots, _ GB/s, _ ms, _ GiB free, _ MiB free")

    def test_the_q36_conversation_sizes_9f3ac8a3_changed_are_masked_for_q36_only(self):
        grown = RUN.replace("1539309568 65617920", "1542266880 68575232")
        self.assertEqual(R.compare_runs([RUN, RUN], [grown, grown], "smoke-q36-86")[0], "PASS")
        self.assertEqual(R.compare_runs([RUN, RUN], [grown, grown], "smoke-q38-86")[0], "DIFF")


class CompareRuns(unittest.TestCase):
    def test_same_tokens_pass(self):
        status, note = R.compare_runs([RUN, RUN], [RUN, RUN])
        self.assertEqual(status, "PASS")
        self.assertIn("6 of 6 parts the same", note)

    def test_two_slots_lines_are_compared_slot_by_slot(self):
        other_order = RUN.replace("BT 1 6\nBT 0 7\n", "BT 0 7\nBT 1 6\n")
        self.assertEqual(R.compare_runs([RUN, RUN], [other_order, other_order])[0], "PASS")
        swapped = RUN.replace("BT 1 6\n", "BT 1 9\n")
        self.assertEqual(R.compare_runs([RUN, RUN], [swapped, swapped])[0], "DIFF")

    def test_an_answer_no_run_before_printed_is_a_diff_when_the_build_before_repeats(self):
        status, note = R.compare_runs([RUN, RUN], [RUN, RUN.replace("T 12", "T 13")])
        self.assertEqual(status, "VARIES")
        self.assertIn("GEN 24 <40 ids 0badf00d>: varies in the build after: after-2 from token 1 (new)", note)
        status, note = R.compare_runs([RUN, RUN], [RUN.replace("T 12", "T 13")] * 2)
        self.assertEqual(status, "DIFF")
        self.assertIn("differs: after-1 from token 1 (new), after-2 from token 1 (new)", note)

    def test_an_answer_the_other_build_also_printed_is_chance(self):
        other = RUN.replace("T 11", "T 14")
        status, note = R.compare_runs([RUN, other, RUN], [other, RUN, RUN])
        self.assertEqual(status, "VARIES")
        self.assertIn("varies in both builds: before-2 from token 0, after-1 from token 0", note)
        self.assertNotIn("(new)", note)

    def test_both_builds_varying_without_an_answer_in_common_is_not_a_diff(self):
        a, b, c, d = (RUN.replace("T 12", f"T {t}") for t in (20, 21, 22, 23))
        status, note = R.compare_runs([a, b], [c, d])
        self.assertEqual(status, "VARIES")
        self.assertIn("after-1 from token 1 (new), after-2 from token 1 (new)", note)

    def test_a_continuation_of_another_answer_is_not_compared(self):
        later = RUN.replace(">>> LIST", ">>> GEN 8 <45 ids 11111111>\nT 1\n>>> LIST")
        moved = later.replace("11111111>\nT 1", "22222222>\nT 2")
        status, note = R.compare_runs([later, later], [moved, moved])
        self.assertEqual(status, "PASS")
        self.assertIn("1 not compared", note)
        self.assertIn("GEN 8 <45 ids 11111111>: no run after sent this prompt; after-1, after-2 not compared", note)

    def test_a_run_that_did_not_finish_or_did_not_run_fails(self):
        status, note = R.compare_runs([RUN], [RUN.replace("exit 0", "STOPPED: the engine ended early\nexit 1")])
        self.assertEqual((status, note), ("FAIL", "did not finish: after-1"))
        self.assertEqual(R.compare_runs([RUN], [], "multi-x")[0], "FAIL")

    def test_a_slower_decode_is_a_diff_only_when_every_run_after_is_slower(self):
        before = [fast(RUN, 10000), fast(RUN, 10400)]                  # 12.8 and 12.3 tok/s
        status, note = R.compare_runs(before, [fast(RUN, 12000), fast(RUN, 12500)])
        self.assertEqual(status, "DIFF")
        self.assertIn("decode tok/s 12.6 (12.3-12.8, n=2) -> 10.5 (10.2-10.7, n=2) (-16.7%)", note)
        self.assertEqual(R.compare_runs(before, [fast(RUN, 12000), fast(RUN, 10100)])[0], "PASS")
        self.assertEqual(R.compare_runs(before[:1], [fast(RUN, 12000)])[0], "PASS")   # one run: not judged

    def test_server_speed_lines(self):
        self.assertEqual(R.speeds("SPEED long decode=20.5 prompt=300.0\n"), {"decode": [20.5], "prompt": [300.0]})


class Explain(unittest.TestCase):
    def test_a_slow_deepseek_check_says_where_the_time_went(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "after-1.out"
            out.write_text(">>> GEN 64 <16 ids 00000000>\nSTAT tok_s=2.0 ms_tok=500 vram_hit=0.5 cpu=100.0 ram=0.000"
                           "\nDONE 64 16 100.0 32000.0 length 0 0\nexit 0\n", encoding="utf-8")
            (Path(d) / "after-1.out.stderr").write_text(
                "ds4: 1000 routed experts on the CPU lane (10.0 GB, in place in the model's files), 6 threads\n"
                "ds4: the page file is small\n", encoding="utf-8")
            (Path(d) / "after-1.out.sys.csv").write_text(
                "seconds,cpu_pct,ram_free_gb,pagefile_pct,disk_read_mbs\n1,90,1.5,40,800\n2,80,1.0,50,1000\n",
                encoding="utf-8")
            self.assertEqual(R.explain_slow_decode(Path(d)), (
                "slow decode: 100 experts a token on the CPU lane, ~1.00 GB a token read in place from the model's "
                "files; VRAM hit rate 0.50, RAM store 0.0 experts a token; ds4: the page file is small; during the "
                "run: CPU 85%, RAM free 1.0 GB, page file 50%, disk reads 900 MB/s"))
            out.write_text(out.read_text(encoding="utf-8").replace("32000.0", "3200.0"), encoding="utf-8")
            self.assertEqual(R.explain_slow_decode(Path(d)), "")


class Telemetry(unittest.TestCase):
    #: A peer run's engine stderr with run.ps1's $Telemetry: the startup and hit rate lines are the first session's
    #: (multi-peer-BS), the timing lines follow the engine's printf formats
    LOG = """strata trace: 13310 MiB free after the CUDA context and the weight arena
strata generate: expert cache 6303 slots, 11.95 GiB of VRAM; policy is
strata trace: 1205 MiB free after opening the expert cache
strata generate: peer GPU 1: 5434 experts, 10.34 GiB (filled in 43.9 s); with the primary's 6303 that is 11737 of \
24576 on the GPUs
strata trace: read 1966 tokens (batched) in 70012.4 ms
strata decode timing: 40 windows, avg T 4.00, 3.20 tokens/window, 50.10 ms/window = verify 40.00 (GPU-reach wait 12.50 \
+ per-layer host 20.00 [plan 1.00 actq 2.00 jobs 3.00 CPU 14.00] + stage 7.50) + commit/emit 5.00 + draft 5.10; per \
layer-window: CPU experts 1.00 (1.50 entries), VRAM hits 6.00, PCIe 2.00; adapt wait 0.00 join 0.00, 0.0 swaps/window
strata serve: decode expert cache hit rate: 87.5% (44407 hits / 50752 lookups); 28928 more read by the GPU over PCIe \
or from another GPU (36.3% of all 79680 routed)
strata serve: stage 0: 80 windows; per window: wait for the GPU 9.000 ms, pool + plan 20.000 ms, host staging \
1.000 ms, commit 0.500 ms
strata decode timing: 40 windows, avg T 4.00, 3.10 tokens/window, 52.10 ms/window = verify 41.00 (GPU-reach wait 13.50 \
+ per-layer host 20.00 [plan 1.00 actq 2.00 jobs 3.00 CPU 14.00] + stage 7.50) + commit/emit 5.00 + draft 6.10; per \
layer-window: CPU experts 1.00 (1.50 entries), VRAM hits 6.00, PCIe 2.00; adapt wait 0.00 join 0.00, 0.0 swaps/window
strata serve: decode expert cache hit rate: 91.3% (45879 hits / 50264 lookups); 30376 more read by the GPU over PCIe \
or from another GPU (37.7% of all 80640 routed)
strata serve: stage 0: 160 windows; per window: wait for the GPU 10.000 ms, pool + plan 21.000 ms, host staging \
1.000 ms, commit 0.500 ms
strata serve: stage 1: 160 windows; per window: wait for the GPU 4.100 ms, pool + plan 2.000 ms, host staging \
1.000 ms, commit 0.500 ms
"""

    def test_a_run_is_read(self):
        self.assertEqual(R.read_telemetry(self.LOG), {
            "ms/window": 51.1, "GPU wait": 13.0, "CUDA0 wait": 10.0, "CUDA0 pool": 21.0, "CUDA1 wait": 4.1,
            "CUDA1 pool": 2.0, "CUDA0 cache": 100 * 90286 / 160320, "other GPU or PCIe": 100 * 59304 / 160320,
            "CPU": 100 * 10730 / 160320, "CUDA0 holds": 6303, "CUDA1 holds": 5434})
        self.assertEqual(R.read_telemetry(self.LOG.replace("strata decode", "x").replace("strata serve: stage", "x")),
                         {})

    def test_helper_and_split_placements(self):
        log = ("strata serve: stage 0: 1 windows; per window: wait for the GPU 1.000 ms, pool + plan 1.000 ms\n"
               "strata generate: CUDA1: 5495 additional experts, 10.43 GiB; results return through pinned host rows\n"
               "strata generate: layer split: CUDA1 runs layers 36-47, expert cache 2501 slots (5.06 GiB), 2501 of "
               "its 6144 profiled pairs; slot 0 verified\nstrata generate: layer split: CUDA0 runs layers 0-35\n")
        self.assertEqual(R.read_telemetry(log)["CUDA1 holds"], 2501)
        self.assertEqual(R.read_telemetry(log.replace("CUDA1 runs", "CUDA2 runs"))["CUDA1 holds"], 5495)

    def test_each_build_gets_a_line_and_the_lines_are_not_compared(self):
        with tempfile.TemporaryDirectory() as d:
            for side, log in (("before", self.LOG), ("after", self.LOG.replace("4.100 ms", "6.100 ms"))):
                for r in (1, 2):
                    (Path(d) / f"{side}-{r}.out").write_text(RUN, encoding="utf-8")
                    (Path(d) / f"{side}-{r}.out.stderr").write_text(log, encoding="utf-8")
            self.assertEqual(R.describe_telemetry(Path(d)), "".join(
                f"\n    {side}: 51.10 ms/window, 13.00 of it waiting for the GPUs; CUDA0 waits 10.00 ms/window for its "
                f"GPU, pool 21.00; CUDA1 waits {wait} ms/window for its GPU, pool 2.00; routed experts: CUDA0 cache "
                f"56%, other GPU or PCIe 37%, CPU 7%; experts held: CUDA0 6,303, CUDA1 5,434"
                for side, wait in (("before", "4.10"), ("after", "6.10"))))
        plain = "".join(x for x in self.LOG.splitlines(keepends=True)
                        if not x.startswith(("strata trace", "strata decode", "strata serve: stage")))
        self.assertIn("strata trace", R.startup_lines(self.LOG))      # drive.py copies some into the output
        self.assertEqual(R.mask(R.startup_lines(self.LOG)), R.mask(R.startup_lines(plain)))


class Build(unittest.TestCase):
    def test_only_the_last_attempts_failures_count_and_only_a_missing_engine_fails(self):
        log = ("FAILED: [code=2] CMakeFiles/old_test.dir/x.cpp.obj\nninja: build stopped: subcommand failed.\n"
               f"{R.RETRY_MARK}\nFAILED: [code=2] CMakeFiles/native_hadamard_test.dir/src/a.cpp.obj\n"
               "FAILED: [code=1] gdn_chunk_bench.exe\nninja: build stopped: subcommand failed.\n")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "build.log").write_text(log, encoding="utf-8")
            for e in R.ENGINES[1:]:
                (Path(d) / e).touch()
            self.assertEqual(R.judge_build(Path(d) / "build.log", Path(d)), (
                "FAIL", "engines missing: strata.exe; not built: gdn_chunk_bench, native_hadamard_test"))
            (Path(d) / "strata.exe").touch()
            self.assertEqual(R.judge_build(Path(d) / "build.log", Path(d))[0], "PASS")


class RowMeasurements(unittest.TestCase):
    def test_missing_corrupt_and_incomplete_measurements_each_get_a_failure_row(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / f"rows-{case}.json" for case in ("card", "ram", "ssd")]
            paths[1].write_text("not json", encoding="utf-8")
            paths[2].write_text('{"rounds": []}', encoding="utf-8")
            status, note = R.aggregate_rows(root, paths)
            self.assertEqual(status, "FAIL")
            rows = json.loads((root / "summary.json").read_text())
            self.assertEqual(set(rows), {"memory-rows-card", "memory-rows-ram", "memory-rows-ssd"})
            self.assertTrue(all(row[0] == "FAIL" for row in rows.values()))
            self.assertIn("0/3 complete", note)

    def test_partial_results_are_aggregated_and_a_report_error_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            good, missing = root / "rows-card.json", root / "rows-ram.json"
            good.write_text('{"rounds": [{"sample": 1}], "memory_at_end": {}}', encoding="utf-8")
            with patch("report.subprocess.run", return_value=SimpleNamespace(
                    returncode=0, stdout="table", stderr="")) as run:
                status, note = R.aggregate_rows(root, [good, missing])
            self.assertEqual(status, "FAIL")
            self.assertIn("1/2 complete", note)
            self.assertEqual(run.call_args.args[0][-1], str(good))
            self.assertNotIn(str(missing), run.call_args.args[0])
            self.assertEqual((root / "rows-report.txt").read_text(), "table")
            with patch("report.subprocess.run", return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr="bad data")):
                self.assertIn("aggregation exited 1", R.aggregate_rows(root, [good])[1])
            with patch("report.subprocess.run", side_effect=OSError("unavailable")):
                self.assertEqual(R.aggregate_rows(root, [good])[0], "FAIL")
            with patch("report.subprocess.run", return_value=SimpleNamespace(
                    returncode=0, stdout="table", stderr="")):
                self.assertEqual(R.aggregate_rows(root, [good])[0], "PASS")


class ExerciseEvidence(unittest.TestCase):
    def test_hop_checks_require_computed_experts_in_every_hop_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for side in ("before", "after"):
                (root / f"{side}-1.out").write_text(RUN, encoding="utf-8")
                (root / f"{side}-1.out.stderr").write_text(
                    "placed 100 experts for one hop\ncomputed 0 experts of other cards' layers\n", encoding="utf-8")
            check = "memory-hop-SB"
            self.assertIn("no hops", R.get_exercise_problem(root, check))
            (root / "after-1.out.stderr").write_text(
                "strata-q38: so far gpu1 computed 27 experts of other cards' layers (40 rows)\n", encoding="utf-8")
            self.assertEqual(R.get_exercise_problem(root, check), "")
            self.assertIn("before-1", R.get_exercise_problem(root, "memory-ds41-hop-default-queues-SB"))
            (root / "after-2.out").write_text(RUN, encoding="utf-8")
            self.assertIn("after-2", R.get_exercise_problem(root, check))
            R.main(["runs", str(root), check])
            self.assertEqual(json.loads((root / "summary.json").read_text())[check][0], "FAIL")

    def test_report_rejects_a_saved_cut_run_with_a_failed_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            check = "memory-cut-move-SB"
            run_dir = root / check
            run_dir.mkdir()
            for side in ("before", "after"):
                (run_dir / f"{side}-1.out").write_text(RUN, encoding="utf-8")
            (run_dir / "after-1.out.settings").write_text("ERR restoring settings\n", encoding="utf-8")
            R.main(["runs", str(root), check])
            self.assertEqual(json.loads((root / "summary.json").read_text())[check][0], "FAIL")

    def test_a_failed_engine_target_cannot_pass_with_an_old_executable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for exe in R.ENGINES:
                (root / exe).touch()
            log = root / "build.log"
            log.write_text("FAILED: strata-q38.exe\n", encoding="utf-8")
            self.assertEqual(R.judge_build(log, root)[0], "FAIL")
            log.unlink()
            self.assertEqual(R.judge_build(log, root)[0], "FAIL")


class Ctest(unittest.TestCase):
    LOG = """      Start  1: conversation_cache_test
 1/6 Test  #1: conversation_cache_test ..........   Passed    0.05 sec
 2/6 Test  #2: vmm_test .........................***Failed    1.10 sec
 3/6 Test  #3: expert_cache_segmented_test ......***Skipped   0.02 sec
 4/6 Test  #4: decode_cluster_parity ............   Passed    0.40 sec
 5/6 Test  #5: verify_parity ....................***Exception: SegFault  0.12 sec
 6/6 Test  #6: iq_parity ........................***Not Run   0.00 sec
"""

    def test_parse(self):
        self.assertEqual(R.parse_ctest(self.LOG), {
            "conversation_cache_test": "Passed", "vmm_test": "Failed", "expert_cache_segmented_test": "Skipped",
            "decode_cluster_parity": "Passed", "verify_parity": "Exception: SegFault", "iq_parity": "Not Run"})

    def test_compare(self):
        before = R.parse_ctest(self.LOG)
        status, note = R.compare_ctest(before, before)
        self.assertEqual(status, "FAIL")
        self.assertIn("fail in both: verify_parity, vmm_test; not built before: iq_parity; not built after: "
                      "iq_parity", note)
        self.assertIn("below sm_90", note)
        after = dict(before, conversation_cache_test="Failed", iq_parity="Passed")
        del after["vmm_test"]
        status, note = R.compare_ctest(before, after)
        self.assertEqual(status, "DIFF")
        self.assertIn("differ: conversation_cache_test (Passed -> Failed), vmm_test (Failed -> missing);", note)
        self.assertEqual(R.compare_ctest(before, dict(before, iq_parity="Passed"))[0], "FAIL")   # fail in both
        self.assertEqual(R.compare_ctest(before, {}), ("FAIL", "no test results after the refactor"))


class Summary(unittest.TestCase):
    def test_rows_are_replaced_and_counted(self):
        with tempfile.TemporaryDirectory() as d:
            R.add_row(Path(d), "unit-86", "FAIL", "x")
            R.add_row(Path(d), "smoke-q38", "SKIP", "no model")
            R.add_row(Path(d), "unit-86", "PASS", "204 tests")
            text = R.summary(Path(d))
        self.assertIn("unit-86    PASS    204 tests", text)
        self.assertIn("totals: 1 PASS, 0 VARIES, 0 DIFF, 0 FAIL, 1 SKIP", text)

    def test_each_run_gets_a_line_on_its_samples(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "multi-x").mkdir()
            (Path(d) / "multi-x" / "before-1.out.gpu.csv").write_text(
                "0, NVIDIA GeForce RTX 3080, 210, 405, 40, 30.1, 0, 900, 1, 16, 0x0000000000000001\n"
                "0, NVIDIA GeForce RTX 3080, 1900, 9251, 70, 300.5, 98, 19000, 4, 16, 0x0000000000000004\n"
                "0, NVIDIA GeForce RTX 3080, 1800, 9251, 72, 290.0, 95, 19000, 4, 16, 0x0000000000000000\n"
                "1, NVIDIA GeForce RTX 3060, 1700, 7500, 60, 150.0, 50, 9000, [N/A], [N/A], [N/A]\n"
                "1, NVIDIA GeForce RTX 3060, 1777, 7500\n", encoding="utf-8")
            text = R.summary(Path(d))
        self.assertIn("multi-x before-1: GPU0 1850 MHz 72 C 96% load gen4 x16 power cap 50%, "
                      "GPU1 1700 MHz 60 C 50% load\n", text)

    def test_startup_lines(self):
        err = ("strata generate: GPU 0: NVIDIA GeForce RTX 3060\n"
               "strata generate: layer split: CUDA1 runs layers 12-47, expert cache 4096 slots\n")
        self.assertEqual(R.startup_lines(err), err.splitlines()[1] + "\n")

class PipelineGroups(unittest.TestCase):
    def test_group_rates_exclude_standalone_and_admission_times(self):
        text = "DONE 96 12 20 1000 length\nSPEED groups decode=12.5 prompt=0\n"
        self.assertEqual(R.speeds(text), {"decode": [12.5], "prompt": []})

    def test_group_tokens_ignore_cross_admission_interleaving(self):
        tokens = "GROUP_TOKENS 0 [[1],[2],[3],[4]]\nGROUP_TOKENS 1 [[1],[2],[3],[4]]\nexit 0\n"
        before = ">>> BGEN 0 x\nBT 0 1\n>>> BGEN 1 y\nBT 1 2\n" + tokens
        after = ">>> BGEN 0 x\n>>> BGEN 1 y\nBT 0 1\nBT 1 2\n" + tokens
        self.assertEqual(R.compare_runs([before], [after], "memory-groups-q38-SB")[0], "PASS")
        self.assertEqual(R.compare_runs([before], [after.replace('[[1]', '[[9]')],
                                        "memory-groups-q38-SB")[0], "DIFF")

    def test_group_check_requires_identity_speed_and_occupied_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for side in ('before', 'after'):
                (root / f'{side}-1.out').write_text(
                    ('GROUP_IDENTITY passed slots=4\nSPEED groups decode=10 prompt=0\n' * 2) +
                    'GROUP_TOKENS 0 [[1],[2],[3],[4]]\nGROUP_TOKENS 1 [[1],[2],[3],[4]]\n', encoding='utf-8')
                (root / f'{side}-1.out.stderr').write_text('', encoding='utf-8')
            self.assertIn('no evidence', R.get_exercise_problem(root, 'memory-groups-q38-SB'))
            (root / 'after-1.out.stderr').write_text(
                'strata-q38: pipeline: groups=2 stages=2 peak_in_flight=2\n', encoding='utf-8')
            self.assertEqual(R.get_exercise_problem(root, 'memory-groups-q38-SB'), '')
            path = root / 'after-1.out'
            path.write_text(path.read_text().replace('passed slots=4', 'differs slots=2'), encoding='utf-8')
            self.assertIn('answers differ', R.get_exercise_problem(root, 'memory-groups-q38-SB'))
            (root / 'before-1.out').write_text('GROUP_IDENTITY passed slots=4\n', encoding='utf-8')
            self.assertIn('missing', R.get_exercise_problem(root, 'memory-groups-q38-SB'))


class DeepSeekGroups(unittest.TestCase):
    def test_live_concurrency_requires_overlap_only_for_divisible_counts(self):
        for count in range(1, 5):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                tokens = json.dumps([[1]] * count)
                for side in ('before', 'after'):
                    (root / f'{side}-1.out').write_text(
                        (f'GROUP_IDENTITY passed slots={count}\nSPEED groups decode=10 prompt=0\n' * 2) +
                        f'GROUP_TOKENS 0 {tokens}\nGROUP_TOKENS 1 {tokens}\n', encoding='utf-8')
                check = f'memory-groups-ds41-{count}c-dspark-on-SB'
                if count % 2 == 0:
                    self.assertIn('no evidence', R.get_exercise_problem(root, check))
                    (root / 'after-1.out.stderr').write_text(
                        'pipeline: groups=2 stages=2 peak_in_flight=2\n', encoding='utf-8')
                self.assertEqual(R.get_exercise_problem(root, check), '')


if __name__ == "__main__":
    unittest.main()
