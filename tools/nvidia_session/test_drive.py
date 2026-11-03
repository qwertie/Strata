"""Checks that the NVIDIA kit requires evidence of a cut move, beyond matching generated tokens."""
import io
import queue
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from drive import Engine, get_cut_move_problem


class CutMove(unittest.TestCase):
    def test_requires_a_move_and_successful_resizes(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "after-1.out"
            settings = Path(str(out) + ".settings")
            stderr = Path(str(out) + ".stderr")
            stderr.write_text("", encoding="utf-8")
            self.assertIn("no settings", get_cut_move_problem(out))
            settings.write_text("END_MEMORY\n", encoding="utf-8")
            self.assertIn("missing or incomplete", get_cut_move_problem(out))
            answers = (">>> after 24 tokens: CHANGE_SETTINGS memory_mib.gpu0=512\nEND_MEMORY\n"
                       ">>> after 48 tokens: CHANGE_SETTINGS memory_mib.gpu0=1024\nEND_MEMORY\n")
            settings.write_text(answers, encoding="utf-8")
            self.assertIn("no layer cut", get_cut_move_problem(out))
            stderr.write_text("strata-ds41: the layer cut moved: K=20 -> K=20\n", encoding="utf-8")
            self.assertIn("no layer cut", get_cut_move_problem(out))
            stderr.write_text("strata-ds41: the layer cut moved: K=20 -> K=12 (pool resize)\n", encoding="utf-8")
            self.assertEqual("", get_cut_move_problem(out))
            settings.write_text(answers.replace("1024\nEND_MEMORY", "1024\nERR restore failed") + "\n",
                                encoding="utf-8")
            self.assertIn("resize failed", get_cut_move_problem(out))
            settings.write_text("ERR gpu0 cannot shrink\n", encoding="utf-8")
            self.assertIn("resize failed", get_cut_move_problem(out))
            settings.write_text("SHORTFALL gpu0\nEND_MEMORY\n", encoding="utf-8")
            self.assertIn("resize failed", get_cut_move_problem(out))


class Telemetry(unittest.TestCase):
    def test_keeps_all_memory_fields_and_process_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Engine.__new__(Engine)
            engine.err_path = str(Path(folder) / "run.out.stderr")
            engine.p = SimpleNamespace(pid=123)
            fields = ["GPU gpu0 total=12000 free=1000 weights=2000 scratch=40\n", "END_MEMORY\n"]
            with patch.object(engine, "ask_quietly", return_value=fields), patch(
                    "drive.subprocess.run", return_value=SimpleNamespace(
                        stdout="GPU-uuid,123,strata-q38.exe,11000 MiB\n", stderr="", returncode=0)) as smi:
                engine.record_memory("ready")
            text = Path(engine.err_path.removesuffix(".stderr") + ".memory").read_text(encoding="utf-8")
            self.assertIn("".join(fields), text)
            self.assertIn("engine PID 123", text)
            self.assertIn("11000 MiB", text)
            self.assertIn("--query-compute-apps=", smi.call_args.args[0][1])
            with patch.object(engine, "ask_quietly", return_value=fields), patch(
                    "drive.subprocess.run", side_effect=FileNotFoundError("no smi")):
                engine.record_memory("after scenario")
            self.assertIn("process VRAM unavailable", Path(
                engine.err_path.removesuffix(".stderr") + ".memory").read_text(encoding="utf-8"))

    def test_batch_end_does_not_hide_a_late_settings_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Engine.__new__(Engine)
            engine.err_path = str(Path(folder) / "run.out.stderr")
            engine.p = SimpleNamespace(stdin=io.StringIO())
            engine.out = io.StringIO()
            engine.lines = queue.Queue()
            for line in ("BT 0 1\n", "BDONE 0 1 length 1.0\n", "ERR restore failed\n"):
                engine.lines.put(line)
            self.assertEqual(engine.bgen(0, 96, [1000], {1: "memory_mib.gpu0=512"}), [1000, 1])
            self.assertIn("ERR restore failed", Path(engine.err_path.removesuffix(
                ".stderr") + ".settings").read_text(encoding="utf-8"))


@unittest.skipUnless(shutil.which("powershell"), "PowerShell is required")
class BuildGates(unittest.TestCase):
    def run_functions(self, body):
        """Exercises script functions with fake commands so tests never build or query a GPU."""
        script = Path(__file__).with_name("run.ps1").read_text(encoding="utf-8")
        functions = script[script.index("function Main"):script.rindex("\nMain")]
        command = "Import-Module Microsoft.PowerShell.Utility\n" + functions + "\n" + body
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_failed_preparation_and_failed_revision_queries_block_dependents(self):
        for prepare in ("$false", "$true"):
            output = self.run_functions(r'''
$Root='C:\kit'; $Work='C:\fake'; $Results='C:\results'; $Steps='C:\steps'
$Blocked=@{}; $Before='before-rev'; $After='after-rev'; $DryRun=$false
function Initialize-Worktree { return PREPARE }
function git { $global:LASTEXITCODE=128; return $null }
function Invoke-Step { throw 'A build or engine must not start' }
function Add-Row { Write-Output "ROW $args" }
Invoke-Builds 86-real 86
Invoke-Pairs check multi strata.exe 86 ''
Invoke-Unit 86
'''.replace("PREPARE", prepare))
            self.assertIn("ROW build-before-86 FAIL", output)
            self.assertIn("ROW build-after-86 FAIL", output)
            self.assertIn("ROW check FAIL", output)
            self.assertIn("ROW unit-86 FAIL", output)

    def test_checkout_failure_is_returned_and_cached(self):
        output = self.run_functions(r'''
$Prepared=@{}; $Steps='C:\steps'; $Root='C:\kit'
function Invoke-Step { return 128 }
if (Initialize-Worktree before C:\fake rev) { throw 'Failed checkout accepted' }
if (Initialize-Worktree before C:\fake rev) { throw 'Failed checkout cached as success' }
Write-Output 'blocked'
''')
        self.assertIn("blocked", output)

    def test_missing_deepseek_argument_is_a_visible_skip(self):
        output = self.run_functions(r'''
$DeepSeekDir=''; $DryRun=$false
function Add-Row { Write-Output "ROW $args" }
Invoke-DeepSeekMemory
Invoke-DeepSeek
''')
        self.assertIn("ROW memory-ds41 SKIP D9 needs -DeepSeekDir", output)
        self.assertIn("ROW deepseek SKIP needs -DeepSeekDir", output)

    def test_config_failure_records_all_three_rows_without_starting_measurements(self):
        output = self.run_functions(r'''
$Work='C:\fake'; $Results='C:\results'; $Small=$null; $DryRun=$false
function Invoke-DeepSeekMemory {}
function Test-Need { return $true }
function Invoke-Pairs {}
function Invoke-Multi {}
function Invoke-Step {
    param($Name)
    if ($Name -ne 'memory-rows-config') { throw 'Must not measure with a failed config' }
    return 1
}
function Add-Row { Write-Output "ROW $args" }
function Invoke-Report { Write-Output "REPORT $args" }
Invoke-Memory
''')
        for case in ("card", "ram", "ssd"):
            self.assertIn(f"ROW memory-rows-{case} FAIL", output)
        self.assertIn("REPORT rows C:\\results memory-rows-report", output)

    def test_provenance_checks_revision_and_executable_hash(self):
        output = self.run_functions(r'''
$Work='C:\fake'; $Root='C:\kit'; $After='revision'; $Blocked=@{}; $DryRun=$false
function git { $global:LASTEXITCODE=0; return 'revision' }
function Get-Content { return '{"revision":"revision","hashes":{"strata-q38.exe":"hash"}}' }
function Get-FileHash { return @{ Hash='hash' } }
if (-not (Test-BuildReady after 86 strata-q38.exe)) { throw 'Valid provenance refused' }
function Get-FileHash { return @{ Hash='changed' } }
if (Test-BuildReady after 86 strata-q38.exe) { throw 'Changed binary accepted' }
function git { $global:LASTEXITCODE=128; return $null }
if (Test-BuildReady after 86 strata-q38.exe) { throw 'Unknown revision accepted' }
Write-Output 'verified'
''')
        self.assertIn("verified", output)


if __name__ == "__main__":
    unittest.main()
