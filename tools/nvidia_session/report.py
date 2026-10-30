"""Judges the checks of the NVIDIA test session (run.ps1) and keeps its summary table.

    report.py runs <results> <check> [--tolerance PCT]    <results>/<check>/before-N.out against after-N.out
    report.py ctest <results> <check> <before log> <after log>
    report.py row <results> <check> <status> [note]
    report.py summary <results>

A check's status is PASS, FAIL (a run did not finish, or a test failed in both builds), DIFF (the build after the
refactor printed something else than the build before) or SKIP. The rows are kept in <results>/summary.json; a check
run again replaces its row.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

#: Measured times, and what depends on the clock or on the free VRAM, masked before two outputs are compared
MASKS = [(re.compile(p, re.M), r) for p, r in (
    (r"^(PP \d+ \d+) [\d.]+ [\d.inf]+", r"\1 _ _"),
    (r"^(B?DONE \d+ \d+)(?: [\d.]+ [\d.]+)? ([a-z]+)\b.*", r"\1 _ _ \2"),   # and the counters after the reason
    (r"^((?:SAVED|RESTORED) \d+ \d+) [\d.]+$", r"\1 _"),
    (r"^(INFO|STAT|SPEED|SWAIT)\b.*", r"\1 _"),
    (r"\d+ MiB available", "_ MiB available"),
    (r"\S+\.ses\b", "X.ses"),
    (r"\b\d+(?:\.\d+)? ?(s|ms|GB/s|MB/s|MiB/s)\b", r"_ \1"),
)]
#: The engine's startup lines (stderr) that multi-GPU checks compare: the split plan, each card's slots and GiB, the
#: peer and helper caches, the page-locked part of the RAM copy
STARTUP = re.compile(r"layer split|peer GPU|context ready|prompt buffers|registered for the GPU|expert cache|helper|"
                     r"remote|slots")
#: What a reader of the summary must know about a test's result on the RTX 30 cards
NOTES = {"decode_cluster_parity": "exits 0 without comparing anything below sm_90 (no cluster kernels on sm_86)"}
CTEST = re.compile(r"^\s*\d+/\d+ Test\s+#\d+: (\S+) \.*\s*(?:\*\*\*)?(.+?)\s+[\d.]+ sec", re.M)


def main(argv: list[str]) -> int:
    cmd, results = argv[0], Path(argv[1])
    if cmd == "summary":
        text = summary(results)
        (results / "summary.txt").write_text(text, encoding="utf-8", newline="\n")
        print(text)
        return 0
    check = argv[2]
    if cmd == "runs":
        tol = float(argv[argv.index("--tolerance") + 1]) if "--tolerance" in argv else None
        files = {s: [p.read_text(encoding="utf-8", errors="replace")
                     for p in sorted((results / check).glob(f"{s}-*.out"))] for s in ("before", "after")}
        status, note = compare_runs(files["before"], files["after"], tol) if files["after"] else ("SKIP", "no runs")
    elif cmd == "ctest":
        logs = [parse_ctest(Path(p).read_text(encoding="utf-8", errors="replace")) if Path(p).exists() else {}
                for p in argv[3:5]]
        status, note = compare_ctest(*logs)
    else:
        status, note = argv[3], " ".join(argv[4:])
    add_row(results, check, status, note)
    print(f"{check}: {status} {note}")
    return 0


def mask(text: str) -> str:
    """Gets `text` with the measured parts of each line replaced by `_` (MASKS)."""
    for pattern, repl in MASKS:
        text = pattern.sub(repl, text)
    return text


def split_blocks(text: str) -> list[str]:
    """Splits an output at the lines the driver writes for each request and for the start (">>> ...")."""
    return [b for b in re.split(r"^(?=>>> )", text, flags=re.M) if b]


def is_finished(text: str) -> bool:
    """Says whether a run got to its end and its engine exited with 0 (drive.py writes "exit <code>" last)."""
    return text.rstrip().endswith("\nexit 0")


def compare_runs(before: list[str], after: list[str], tolerance: float | None = None) -> tuple[str, str]:
    """Compares the runs of the two builds request by request, masked. A request counts only where every run of the
    build before printed the same, since some answers differ between two runs of one build (DeepSeek, see the plan's
    open issues). With `tolerance`, a decode speed after more than that many percent below before's is a DIFF too.
    Returns (status, note)."""
    failed = [f"{side}-{i + 1}" for side, runs in (("before", before), ("after", after))
              for i, text in enumerate(runs) if not is_finished(text)]
    if failed:
        return "FAIL", "did not finish: " + ", ".join(failed)
    b, a = [split_blocks(mask(t)) for t in before], [split_blocks(mask(t)) for t in after]
    if len({len(x) for x in b + a}) > 1:
        return "DIFF", "the runs printed different numbers of requests"
    noisy, diff = 0, []
    for i, ref in enumerate(b[0]):
        if any(x[i] != ref for x in b[1:]):
            noisy += 1
        elif any(x[i] != ref for x in a):
            diff.append(ref.splitlines()[0])
    notes = [f"{len(b[0]) - noisy - len(diff)} of {len(b[0])} parts the same"]
    if noisy:
        notes.append(f"{noisy} not compared (the build before printed them differently in two runs)")
    if len(before) < 2:
        notes.append("one run a build: repeatability not checked")
    if diff:
        notes.append("differ: " + "; ".join(diff[:5]))
    slow = False
    sb, sa = speeds("".join(before)), speeds("".join(after))
    for kind in ("decode", "prompt"):
        if sb[kind] and sa[kind]:
            mb, ma = statistics.median(sb[kind]), statistics.median(sa[kind])
            notes.append(f"{kind} {mb:.1f} -> {ma:.1f} tok/s ({100 * (ma / mb - 1):+.1f}%)")
            slow |= kind == "decode" and tolerance is not None and ma < mb * (1 - tolerance / 100)
    return ("DIFF" if diff or slow else "PASS"), "; ".join(notes)


def speeds(text: str) -> dict[str, list[float]]:
    """Gets the tokens/s of the answers in an output: decode from each DONE line of 64 tokens or more (its fields:
    tokens, prompt tokens, prompt ms, decode ms), decode and prompt from the server client's SPEED lines."""
    out: dict[str, list[float]] = {"decode": [], "prompt": []}
    for m in re.finditer(r"^DONE (\d+) \d+ [\d.]+ ([\d.]+)", text, re.M):
        if int(m[1]) >= 64 and float(m[2]) > 0:
            out["decode"].append(int(m[1]) * 1000 / float(m[2]))
    for m in re.finditer(r"^SPEED .*decode=([\d.]+) prompt=([\d.]+)", text, re.M):
        out["decode"].append(float(m[1]))
        out["prompt"].append(float(m[2]))
    return out


def startup_lines(stderr: str) -> str:
    """Gets the engine's startup lines a multi-GPU check compares (STARTUP)."""
    return "".join(line + "\n" for line in stderr.splitlines() if STARTUP.search(line))


def parse_ctest(log: str) -> dict[str, str]:
    """Reads ctest's console output: test name -> its result as ctest words it ("Passed", "Failed", "Skipped",
    "Not Run", "Timeout", "Exception: SegFault", ...)."""
    return {m[1]: m[2] for m in CTEST.finditer(log)}


def compare_ctest(before: dict[str, str], after: dict[str, str]) -> tuple[str, str]:
    """Compares the per-test results of the two builds; a test missing after the refactor is a difference too (a
    moved test must still be built and registered). Returns (status, note)."""
    if not after:
        return "FAIL", "no test results after the refactor"
    differ = [f"{t} ({before.get(t, 'missing')} -> {after.get(t, 'missing')})"
              for t in sorted(set(before) | set(after)) if t not in after or before.get(t, after[t]) != after[t]]
    both = sorted(t for t, r in after.items() if r not in ("Passed", "Skipped") and before.get(t) == r)
    counts = {r: list(after.values()).count(r) for r in sorted(set(after.values()))}
    notes = [f"{len(after)} tests after: " + ", ".join(f"{n} {r}" for r, n in counts.items())]
    if differ:
        notes.append("differ: " + ", ".join(differ))
    if both:
        notes.append("fail in both: " + ", ".join(both))
    notes += [f"{t} {n}" for t, n in NOTES.items() if t in after]
    return ("DIFF" if differ else "FAIL" if both else "PASS"), "; ".join(notes)


def add_row(results: Path, check: str, status: str, note: str) -> None:
    """Puts a check's result into <results>/summary.json, replacing an earlier row of the same check."""
    f = results / "summary.json"
    rows = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    rows[check] = [status, note]
    f.write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")


def summary(results: Path) -> str:
    """Gets the summary table: one line per check, its status and its note."""
    f = results / "summary.json"
    rows = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    width = max([len(c) for c in rows] + [5])
    lines = [f"{'check':<{width}}  status  note"] + [f"{c:<{width}}  {s:<6}  {n}" for c, (s, n) in rows.items()]
    counts = [s for s, _ in rows.values()]
    lines.append("totals: " + ", ".join(f"{counts.count(s)} {s}" for s in ("PASS", "DIFF", "FAIL", "SKIP")))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
