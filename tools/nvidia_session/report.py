"""Judges the checks of the NVIDIA test session (run.ps1) and keeps its summary table.

    report.py runs <results> <check>                       <results>/<check>/before-N.out against after-N.out
    report.py build <results> <check> <build log> <build folder> [note]
    report.py rows <results> <check> <measurement JSON> ...
    report.py ctest <results> <check> <before log> <after log>
    report.py row <results> <check> <status> [note]
    report.py summary <results>

A check's status is PASS, VARIES (some runs printed something else, but not in a way that tells the builds apart),
DIFF (the build after the refactor printed something else than the build before, or decoded slower), FAIL (a run did
not finish, an engine or a test was not built, or a test failed in both builds) or SKIP. The rows are kept in
<results>/summary.json; a check run again replaces its row.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path

#: Measured times, and what depends on the clock or on the free memory, masked before two outputs are compared
MASKS = [(re.compile(p, re.M), r) for p, r in (
    (r"^(PP \d+ \d+) [\d.]+ [\d.inf]+", r"\1 _ _"),
    (r"^(B?DONE \d+ \d+)(?: [\d.]+ [\d.]+)? ([a-z]+)\b.*", r"\1 _ _ \2"),   # and the counters after the reason
    (r"^((?:SAVED|RESTORED) \d+ \d+) [\d.]+$", r"\1 _"),
    (r"^(INFO|STAT|SPEED|SWAIT)\b.*", r"\1 _"),
    (r"^strata serve: decode expert cache hit rate.*\n", ""),
    (r"^strata (?:trace|decode|pipeline)\b.*\n|^strata serve: stage \d+: .*\n", ""),   # run.ps1's $Telemetry
    (r"\d+ MiB (available|free)", r"_ MiB \1"),
    (r"\b\d+\.\d+ GiB\b", "_ GiB"),
    (r"miss cost x[\d.]+|pcie_frac [\d.]+|best of [\d. ]+", "_"),
    (r"\S+\.ses\b", "X.ses"),
    (r"\b\d+(?:\.\d+)? ?(s|ms|GB/s|MB/s|MiB/s)\b", r"_ \1"),
)]
#: Differences a commit made on purpose, masked in the checks whose names match: (check, pattern, replacement)
INTENDED = [(re.compile(c), re.compile(p, re.M), r) for c, p, r in (
    # 9f3ac8a3 (A5 state): Qwen3.6's LIST_CONVERSATIONS sizes count every region of a session (~2.8 MiB more)
    (r"q36", r"^(CONVERSATION \S+ \S+ \S+ \d+ \d+) \d+ \d+$", r"\1 _ _"),
)]
#: The lines of one batch slot. They are compared slot by slot, since the order in which two slots' lines interleave
#: is timing (57a2e1e0 made Qwen3.6 read two queued BGEN lines together, as Qwen3.8 does, which changed it)
SLOT_LINE = re.compile(r"^(?:BT|BDONE|BADM) (\d+)\b")
#: The lines describe_telemetry reads from an engine's log: a stage's host time per window since the start
#: (STRATA_SPLIT_TIMING; stage k runs on CUDAk), a request's time per window (STRATA_DECODE_TIMING), a request's routed
#: experts, and how many experts each card holds once started
STAGE_TIMING = re.compile(r"^strata serve: stage (\d+): \d+ windows; per window: wait for the GPU ([\d.]+) ms, "
                          r"pool \+ plan ([\d.]+) ms", re.M)
DECODE_TIMING = re.compile(r"^strata decode timing: .*? ([\d.]+) ms/window = verify [\d.]+ \(GPU-reach wait ([\d.]+)",
                           re.M)
HIT_RATE = re.compile(r"decode expert cache hit rate: [\d.]+% \((\d+) hits / (\d+) lookups\)(?:; (\d+) more read)?")
PLACEMENT = re.compile(r"^strata generate: (?:expert cache (\d+) slots, |layer split: CUDA(\d+) runs layers \S+, "
                       r"expert cache (\d+) slots|peer GPU (\d+): (\d+) experts|CUDA(\d+): (\d+) additional experts)",
                       re.M)
#: The engine's startup lines (stderr) that multi-GPU checks compare: the split plan, each card's slots and GiB, the
#: peer and helper caches, the page-locked part of the RAM copy
STARTUP = re.compile(r"layer split|peer GPU|context ready|prompt buffers|registered for the GPU|expert cache|helper|"
                     r"remote|slots")
#: What a reader of the summary must know about a test's result on the RTX 30 cards
NOTES = {"decode_cluster_parity": "exits 0 without comparing anything below sm_90 (no cluster kernels on sm_86)"}
CTEST = re.compile(r"^\s*\d+/\d+ Test\s+#\d+: (\S+) \.*\s*(?:\*\*\*)?(.+?)\s+[\d.]+ sec", re.M)
#: A decode this many percent slower in every run after than in every run before is a DIFF
SPEED_TOLERANCE = 5
#: Below this many tokens/s a DeepSeek check's note says where the time went
SLOW_DECODE = 3
#: The engines every build must have
ENGINES = ("strata.exe", "strata-q36.exe", "strata-dsv41.exe")
#: The line run.ps1 writes into a build's log before it runs the build a second time
RETRY_MARK = "retrying the build"
#: The columns drive.py asks nvidia-smi for; the last one is called clocks_event_reasons.active by newer drivers
GPU_FIELDS = ("index", "name", "clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "utilization.gpu",
              "memory.used", "pcie.link.gen.current", "pcie.link.width.current", "clocks_throttle_reasons.active")
#: The bits of the throttle reasons that slow a busy card (idle, application clocks and sync boost do not)
THROTTLE = {0x4: "power cap", 0x8: "hardware slowdown", 0x40: "software thermal", 0x80: "hardware thermal",
            0x100: "power brake"}


def main(argv: list[str]) -> int:
    cmd, results = argv[0], Path(argv[1])
    if cmd == "summary":
        text = summary(results)
        (results / "summary.txt").write_text(text, encoding="utf-8", newline="\n")
        print(text)
        return 0
    check = argv[2]
    if cmd == "runs":
        files = {s: [p.read_text(encoding="utf-8", errors="replace")
                     for p in sorted((results / check).glob(f"{s}-*.out"))] for s in ("before", "after")}
        status, note = compare_runs(files["before"], files["after"], check)
        if problem := get_exercise_problem(results / check, check):
            status, note = "FAIL", problem
            if "memory-groups-" in check and all(
                    len(re.findall(r"^SPEED groups ", run, re.M)) == 2
                    for runs in files.values() for run in runs):
                note = "; ".join([note, *compare_speeds(files["before"], files["after"])[1]])
        if "memory-ds41-remote-" in check:
            counts = []
            for path in sorted((results / check).glob("after-*.out.stderr")):
                cards = {}
                for gpu, count in re.findall(r"so far (gpu\d+) computed (\d+) experts of other cards' layers",
                                             path.read_text(encoding="utf-8", errors="replace")):
                    cards[gpu] = max(cards.get(gpu, 0), int(count))
                counts.append(sum(cards.values()))
            note += f"; remote experts per run: {counts}"
        note = "; ".join(filter(None, [note, explain_slow_decode(results / check)])) + describe_telemetry(
            results / check)
    elif cmd == "rows":
        status, note = aggregate_rows(results, [Path(p) for p in argv[3:]])
    elif cmd == "build":
        status, note = judge_build(Path(argv[3]), Path(argv[4]))
        note = "; ".join(filter(None, [" ".join(argv[5:]), note]))
    elif cmd == "ctest":
        logs = [parse_ctest(Path(p).read_text(encoding="utf-8", errors="replace")) if Path(p).exists() else {}
                for p in argv[3:5]]
        status, note = compare_ctest(*logs)
    else:
        status, note = argv[3], " ".join(argv[4:])
    add_row(results, check, status, note)
    print(f"{check}: {status} {note}")
    return 0


def aggregate_rows(results: Path, paths: list[Path]) -> tuple[str, str]:
    """Produces the rows table from complete measurements while retaining a failure for each unusable case."""
    valid, problems = [], []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data.get("rounds") or "memory_at_end" not in data:
                raise ValueError("incomplete measurement")
            valid.append(path)
        except (OSError, ValueError) as error:
            note = f"{path.name}: {error}; see steps/memory-{path.stem}.log"
            problems.append(note)
            add_row(results, "memory-" + path.stem, "FAIL", note)
    if valid:
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[1] / "measure_rows.py"),
                 "--report", *map(str, valid)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120, check=False)
            (results / "rows-report.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                problems.append(f"aggregation exited {result.returncode}; see rows-report.txt")
        except (OSError, subprocess.TimeoutExpired) as error:
            problems.append(f"aggregation unavailable: {error}")
    else:
        problems.append("no complete row measurements to aggregate")
    note = f"{len(valid)}/{len(paths)} complete measurements; " + "; ".join(problems or ["see rows-report.txt"])
    return ("FAIL" if problems else "PASS"), note


def get_cut_move_problem(out: Path) -> str:
    """Gets why a cut-move check did not exercise a successful move, even if generation completed."""
    settings = out.with_suffix(out.suffix + ".settings")
    if not settings.exists():
        return "no settings answers recorded during decoding"
    text = settings.read_text(encoding="utf-8")
    if re.search(r"\b(ERR|SHORTFALL)\b", text):
        return "a memory resize failed; see the settings answers"
    changes = re.findall(r"^>>> after (\d+) tokens: CHANGE_SETTINGS .*?\n(.*?)(?=^>>>|\Z)", text, re.M | re.S)
    if [int(token) for token, _ in changes] != [24, 48] or any(
            not re.search(r"^END_MEMORY\b", answer, re.M) for _, answer in changes):
        return "shrink or restore settings answer missing or incomplete"
    stderr_path = out.with_suffix(out.suffix + ".stderr")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    moves = re.findall(r"layer cut moved: K=(\d+) -> K=(\d+)", stderr)
    if not any(before != after for before, after in moves):
        return "no layer cut moved; see stderr and the settings answers"
    return ""


def get_exercise_problem(folder: Path, check: str) -> str:
    """Requires workload-specific evidence before comparing memory checks."""
    if "memory-groups-" in check:
        for side in ("before", "after"):
            runs = list(folder.glob(f"{side}-*.out"))
            if not runs:
                return f"no {side} group runs"
            for run in runs:
                if Path(str(run) + ".watchdog").exists():
                    return f"{run.name}: memory watchdog tripped"
                text = run.read_text(encoding="utf-8", errors="replace")
                tokens = re.findall(r"^GROUP_TOKENS \d+ (.*)$", text, re.M)
                try:
                    valid = len(tokens) == 2 and all(len(json.loads(row)) == 4 for row in tokens)
                except (ValueError, TypeError):
                    valid = False
                identity = re.findall(r"^GROUP_IDENTITY (?:passed slots=4|differs slots=\S+)$", text, re.M)
                if not valid or len(identity) != 2 or len(speeds(text)["decode"]) != 2:
                    return f"{run.name}: missing concurrent identity or aggregate speed evidence"
                if text.count("GROUP_IDENTITY passed slots=4") != 2:
                    return f"{run.name}: concurrent answers differ from standalone"
                if "q38" in check and side == "after":
                    path = Path(str(run) + ".stderr")
                    stderr = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                    if not re.search(r"pipeline: groups=2 stages=2 peak_in_flight=2", stderr):
                        return f"{run.name}: no evidence of two occupied stages"
    if "hop" in check or "ds41-remote-on-" in check:
        sides = ("before", "after") if "default-queues" in check else ("after",)
        for side in sides:
            runs = list(folder.glob(f"{side}-*.out"))
            if not runs:
                return f"no hops: no {side} runs"
            for run in runs:
                stderr = Path(str(run) + ".stderr")
                text = stderr.read_text(encoding="utf-8", errors="replace") if stderr.exists() else ""
                if "expert hop failed" in text:
                    return f"{run.name}: expert hop failed; see stderr"
                if not any(int(n) > 0 for n in re.findall(
                        r"computed (\d+) experts of other cards' layers", text)):
                    return f"no hops: {run.name} has no nonzero computed-expert counter in stderr"
    if "cut-move" in check:
        for run in folder.glob("after-*.out"):
            if problem := get_cut_move_problem(run):
                return f"{run.name}: {problem}"
            if "ds41" in check:
                path = Path(str(run) + ".stderr")
                text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                migrations = re.findall(r"dense layer (\d+) migrated: gpu(\d+) -> gpu(\d+) \((\d+) bytes\)", text)
                moved = {int(layer) for layer, src, dst, size in migrations if src != dst and int(size) > 0}
                expected = set()
                for before, after in re.findall(r"layer cut moved: K=(\d+) -> K=(\d+)", text):
                    expected.update(range(min(int(before), int(after)), max(int(before), int(after))))
                if not expected or not expected.issubset(moved):
                    return f"{run.name}: missing cross-card dense-weight migration evidence for moved layers"
    return ""


def compare_runs(before: list[str], after: list[str], check: str = "") -> tuple[str, str]:
    """Compares the runs of the two builds request by request, masked (see judge_part), and their decode speeds.
    Returns (status, note)."""
    failed = [f"{side}-{i + 1}" for side, runs in (("before", before), ("after", after))
              for i, text in enumerate(runs) if not is_finished(text)]
    if failed:
        return "FAIL", "did not finish: " + ", ".join(failed)
    if not before or not after:
        return "FAIL", f"no output of the build {'before' if not before else 'after'} (see steps\\{check}-*.log)"
    b, a = [split_parts(t, check) for t in before], [split_parts(t, check) for t in after]
    if len({len(x) for x in b + a}) > 1:
        return "DIFF", "the runs printed different numbers of requests"
    judged = [judge_part([x[i] for x in b], [x[i] for x in a]) for i in range(len(b[0]))]
    statuses = [s for s, _ in judged]
    odd = [n for s, n in judged if s == "DIFF"] + [n for s, n in judged if s in ("VARIES", "SKIP")]
    notes = [f"{statuses.count('PASS')} of {len(judged)} parts the same"] + (
        [f"{statuses.count('SKIP')} not compared"] if "SKIP" in statuses else [])
    if len(before) < 2:
        notes.append("one run a build: repeatability not checked")
    notes += odd[:6] + ([f"{len(odd) - 6} more parts differ"] if len(odd) > 6 else [])
    slow, speed_notes = compare_speeds(before, after)
    status = "DIFF" if slow or "DIFF" in statuses else "VARIES" if "VARIES" in statuses else "PASS"
    return status, "; ".join(notes + speed_notes)


def judge_part(before: list[str], after: list[str]) -> tuple[str, str]:
    """Judges one request's output in the runs that sent the build before's most common request line (drive.py puts
    a digest of the prompt into it, so a conversation that went on from another answer is not compared). Where a
    build printed it in more than one way, an output of the other build that equals one of them is chance, not
    change: the part VARIES. It is a DIFF where no run of one build printed what a run of the other did, and at least
    one build printed it one way only. SKIP: no run after sent that request. Returns (status, note); the note names
    each run that differs from the build before's most common output, from which token, and "(new)" where no run of
    the other build printed the same."""
    head = Counter(first_line(p) for p in before).most_common(1)[0][0]
    runs = {side: {f"{side}-{i + 1}": p for i, p in enumerate(outs) if first_line(p) == head}
            for side, outs in (("before", before), ("after", after))}
    bset, aset = set(runs["before"].values()), set(runs["after"].values())
    other = [f"{side}-{i + 1}" for side, outs in (("before", before), ("after", after))
             for i, p in enumerate(outs) if first_line(p) != head]
    unlike = f"; {', '.join(other)} not compared (another prompt: an earlier answer differed)" if other else ""
    if not aset:
        return "SKIP", f"{title(head)}: no run after sent this prompt{unlike}"
    if len(bset | aset) == 1:
        return "PASS", ""
    varies = [s for s, v in (("before", bset), ("after", aset)) if len(v) > 1]
    if bset & aset or len(varies) == 2:
        status, how = "VARIES", "varies in " + ("both builds" if len(varies) == 2 else f"the build {varies[0]}")
    else:
        status, how = "DIFF", "differs"
    ref = Counter(runs["before"].values()).most_common(1)[0][0]
    seen = {"before": aset, "after": bset}
    named = [f"{name} from {first_difference(ref, p)}" + ("" if p in seen[side] else " (new)")
             for side, outs in runs.items() for name, p in outs.items() if p != ref]
    return status, f"{title(ref)}: {how}: {', '.join(named)}{unlike}"


def first_line(part: str) -> str:
    """Gets the request line a part starts with."""
    return part.split("\n", 1)[0]


def first_difference(ref: str, other: str) -> str:
    """Says where `other` first differs from `ref`: the index of the first answer token (T and BT lines) that differs,
    or, with the same tokens, the first other line that does."""
    tokens = [[x for x in s.splitlines() if x.startswith(("T ", "BT "))] for s in (ref, other)]
    if tokens[0] != tokens[1]:
        return f"token {next((i for i, (x, y) in enumerate(zip(*tokens)) if x != y), min(map(len, tokens)))}"
    line = next((y for x, y in zip_longest(ref.splitlines(), other.splitlines(), fillvalue="(end)") if x != y))
    return f"line '{line[:40]}'"


def title(part: str) -> str:
    """Gets a part's request line without the driver's ">>> " and with long token lists shortened."""
    return re.sub(r"\d+(?:,\d+){7,}", lambda m: f"<{m[0].count(',') + 1} ids>", first_line(part)[4:])


def compare_speeds(before: list[str], after: list[str]) -> tuple[bool, list[str]]:
    """Compares the decode and prompt speeds per run (each run's median). Returns whether the build after decoded
    slower (by more than SPEED_TOLERANCE percent, every run after slower than every run before, two runs a build at
    least) and a note per kind."""
    slow, notes = False, []
    for kind in ("decode", "prompt"):
        sb, sa = [[statistics.median(v) for t in runs if (v := speeds(t)[kind])] for runs in (before, after)]
        if sb and sa:
            mb, ma = statistics.median(sb), statistics.median(sa)
            notes.append(f"{kind} tok/s {spread(sb)} -> {spread(sa)} ({100 * (ma / mb - 1):+.1f}%)")
            slow |= (kind == "decode" and min(len(sb), len(sa)) > 1 and max(sa) < min(sb)
                     and ma < mb * (1 - SPEED_TOLERANCE / 100))
    return slow, notes


def spread(v: list[float]) -> str:
    """Formats measurements as their median, range and count."""
    return f"{v[0]:.1f} (n=1)" if len(v) == 1 else f"{statistics.median(v):.1f} ({min(v):.1f}-{max(v):.1f}, n={len(v)})"


def speeds(text: str) -> dict[str, list[float]]:
    """Gets the tokens/s of the answers in an output: decode from each DONE line of 64 tokens or more (its fields:
    tokens, prompt tokens, prompt ms, decode ms), decode and prompt from the server client's SPEED lines."""
    out: dict[str, list[float]] = {"decode": [], "prompt": []}
    group_rates = re.findall(r"^SPEED groups decode=([\d.]+) prompt=0$", text, re.M)
    if group_rates:
        out["decode"] = [float(rate) for rate in group_rates]
        return out
    for m in re.finditer(r"^DONE (\d+) \d+ [\d.]+ ([\d.]+)", text, re.M):
        if int(m[1]) >= 64 and float(m[2]) > 0:
            out["decode"].append(int(m[1]) * 1000 / float(m[2]))
    for m in re.finditer(r"^SPEED .*decode=([\d.]+) prompt=([\d.]+)", text, re.M):
        out["decode"].append(float(m[1]))
        out["prompt"].append(float(m[2]))
    return out


def mask(text: str, check: str = "") -> str:
    """Gets `text` with the measured parts of each line replaced by `_` (MASKS), and the differences INTENDED for
    this check."""
    for pattern, repl in MASKS + [(p, r) for c, p, r in INTENDED if c.search(check)]:
        text = pattern.sub(repl, text)
    return text


def split_parts(text: str, check: str = "") -> list[str]:
    """Splits a masked output at the lines the driver writes for each request and for the start (">>> ..."), each
    part with its batch slots' lines put after the others, slot by slot (SLOT_LINE)."""
    if "memory-groups-" in check:
        return [f">>> GROUP {index}\n{tokens}\n" for index, tokens in
                re.findall(r"^GROUP_TOKENS (\d+) (.*)$", text, re.M)]
    parts = [b for b in re.split(r"^(?=>>> )", mask(text, check), flags=re.M) if b]
    return [group_slot_lines(p) for p in parts]


def group_slot_lines(part: str) -> str:
    """Gets a part with its batch slots' lines moved after the others, slot by slot, each slot's in their order."""
    lines = part.splitlines(keepends=True)
    slots = [(int(m[1]), i) for i, line in enumerate(lines) if (m := SLOT_LINE.match(line))]
    kept = [line for line in lines if not SLOT_LINE.match(line)]
    return "".join(kept + [lines[i] for _, i in sorted(slots)])


def is_finished(text: str) -> bool:
    """Says whether a run got to its end and its engine exited with 0 (drive.py writes "exit <code>" last)."""
    return text.rstrip().endswith("\nexit 0")


def explain_slow_decode(folder: Path) -> str:
    """Says where a DeepSeek check's time went when the build after decoded below SLOW_DECODE tokens/s: its experts
    a token on the CPU lane (read in place from the model's files), the cache hit rates and what the PC was doing,
    from its first run after. Empty for other checks and for faster ones."""
    out = folder / "after-1.out"
    err = folder / "after-1.out.stderr"
    if not out.exists() or not err.exists():
        return ""
    text, log = out.read_text(encoding="utf-8", errors="replace"), err.read_text(encoding="utf-8", errors="replace")
    lane = re.search(r"(\d+) routed experts on the CPU lane \(([\d.]+) GB", log)
    decode = speeds(text)["decode"]
    if not lane or not decode or statistics.median(decode) >= SLOW_DECODE:
        return ""
    stats, last = [], ""
    for line in text.splitlines():
        if line.startswith("STAT "):
            last = line
        elif line.startswith("DONE ") and int(line.split()[1]) >= 64 and last:
            stats.append(dict(kv.split("=", 1) for kv in last.split()[1:]))
    mean = lambda key: statistics.mean(float(s[key]) for s in stats)  # noqa: E731
    expert_gb = float(lane[2]) / int(lane[1])
    why = [f"slow decode: {mean('cpu'):.0f} experts a token on the CPU lane, ~{mean('cpu') * expert_gb:.2f} GB a "
           f"token read in place from the model's files; VRAM hit rate {mean('vram_hit'):.2f}, RAM store "
           f"{mean('ram'):.1f} experts a token"] if stats else []
    why += [line.strip() for line in log.splitlines() if re.search(r"page ?file|commit|paging", line, re.I)][:2]
    pc = summarize_pc(out.with_name(out.name + ".sys.csv"))
    return "; ".join(why + ([f"during the run: {pc}"] if pc else []))


def describe_telemetry(folder: Path) -> str:
    """Says per build where a check's decode time went, from the engine logs of its runs (the server's engine log,
    else the engine's stderr), each number the median of the runs': the ms of a decode window and how much of it
    waited for the GPUs, each stage's host wait for its GPU and its CPU pool time per window, the shares of the routed
    experts that CUDA0's cache, another GPU or PCIe, and the CPU computed, and how many experts each card holds. One
    indented line a build; empty without run.ps1's $Telemetry lines."""
    lines = []
    for side in ("before", "after"):
        runs = [read_telemetry(log.read_text(encoding="utf-8", errors="replace"))
                for out in sorted(folder.glob(f"{side}-*.out"))
                if (log := next((x for x in (Path(f"{out}.engine.log"), Path(f"{out}.stderr")) if x.exists()), None))]
        if runs := [r for r in runs if r]:
            lines.append(f"\n    {side}: " + "; ".join(summarize_telemetry(runs)))
    return "".join(lines)


def read_telemetry(log: str) -> dict[str, float]:
    """Gets one run's telemetry (see describe_telemetry) as name -> value; empty without the timing lines."""
    out: dict[str, float] = {}
    if windows := DECODE_TIMING.findall(log):
        out["ms/window"] = statistics.median(float(ms) for ms, _ in windows)
        out["GPU wait"] = statistics.median(float(wait) for _, wait in windows)
    for m in STAGE_TIMING.finditer(log):              # cumulative: a stage's last line holds the whole run
        out[f"CUDA{m[1]} wait"], out[f"CUDA{m[1]} pool"] = float(m[2]), float(m[3])
    if not out:                                       # the hit rates and startup lines are there without $Telemetry
        return out
    hits = [[int(x or 0) for x in m.groups()] for m in HIT_RATE.finditer(log)]
    if routed := sum(look + off for _, look, off in hits):
        out["CUDA0 cache"] = 100 * sum(h for h, _, _ in hits) / routed
        out["other GPU or PCIe"] = 100 * sum(off for _, _, off in hits) / routed
        out["CPU"] = 100 * sum(look - h for h, look, _ in hits) / routed
    for m in PLACEMENT.finditer(log):
        card, n = next((c, n) for c, n in (("0", m[1]), (m[2], m[3]), (m[4], m[5]), (m[6], m[7])) if n)
        out[f"CUDA{card} holds"] = int(n)
    return out


def summarize_telemetry(runs: list[dict[str, float]]) -> list[str]:
    """Formats the medians of runs' telemetry (read_telemetry) in describe_telemetry's order."""
    med = {k: statistics.median(r[k] for r in runs if k in r) for k in dict.fromkeys(k for r in runs for k in r)}
    parts = [f"{med['ms/window']:.2f} ms/window, {med['GPU wait']:.2f} of it waiting for the GPUs"] if (
        "ms/window" in med) else []
    parts += [f"{k[:-5]} waits {v:.2f} ms/window for its GPU, pool {med[k[:-4] + 'pool']:.2f}"
              for k, v in med.items() if k.endswith(" wait") and k != "GPU wait"]
    if "CPU" in med:
        parts.append("routed experts: " + ", ".join(f"{k} {med[k]:.0f}%" for k in ("CUDA0 cache", "other GPU or PCIe",
                                                                                     "CPU")))
    held = [f"{k[:-6]} {v:,.0f}" for k, v in med.items() if k.endswith(" holds")]
    return parts + (["experts held: " + ", ".join(held)] if held else [])


def startup_lines(stderr: str) -> str:
    """Gets the engine's startup lines a multi-GPU check compares (STARTUP)."""
    return "".join(line + "\n" for line in stderr.splitlines() if STARTUP.search(line))


def judge_build(log: Path, build: Path) -> tuple[str, str]:
    """Reads a build's log for the targets that failed in its last attempt (the build keeps going past a failure)
    and says which engines are missing or failed. Returns (status, note)."""
    text = log.read_text(encoding="utf-8", errors="replace").rsplit(RETRY_MARK, 1)[-1] if log.exists() else ""
    failed = sorted({target_of(p) for p in re.findall(r"^FAILED: (?:\[code=\d+\] )?(\S+)", text, re.M)})
    missing = [e for e in ENGINES if not (build / e).exists()]
    notes = ([f"engines missing: {', '.join(missing)}"] if missing else []) + (
        [f"not built: {', '.join(failed)}"] if failed else [])
    failed_engines = sorted(set(failed) & {Path(e).stem for e in ENGINES + ("strata-q38.exe", "strata-ds41.exe")})
    if not log.exists():
        notes.append("no build log")
    return ("FAIL" if missing or failed_engines or not log.exists() else "PASS"), "; ".join(notes)


def target_of(output: str) -> str:
    """Gets the CMake target that a file ninja failed to make belongs to."""
    m = re.search(r"CMakeFiles/([^/]+)\.dir/", output)
    return m[1] if m else Path(output).stem


def parse_ctest(log: str) -> dict[str, str]:
    """Reads ctest's console output: test name -> its result as ctest words it ("Passed", "Failed", "Skipped",
    "Not Run" (its program was not built), "Timeout", "Exception: SegFault", ...)."""
    return {m[1]: m[2] for m in CTEST.finditer(log)}


def compare_ctest(before: dict[str, str], after: dict[str, str]) -> tuple[str, str]:
    """Compares the per-test results of the two builds. A test the build after did not register or build where the
    build before ran it, or that ended otherwise, is a difference (a moved test must still be built and registered).
    Returns (status, note)."""
    if not after:
        return "FAIL", "no test results after the refactor"
    differ = [f"{t} ({before[t]} -> {after.get(t, 'missing')})"
              for t in sorted(before) if before[t] != "Not Run" and before[t] != after.get(t)]
    both = sorted(t for t, r in after.items() if r not in ("Passed", "Skipped", "Not Run") and before.get(t) == r)
    unbuilt = {s: sorted(t for t, r in runs.items() if r == "Not Run") for s, runs in (("before", before),
                                                                                       ("after", after))}
    counts = Counter(after.values())
    notes = [f"{len(after)} tests after: " + ", ".join(f"{n} {r}" for r, n in sorted(counts.items()))]
    if differ:
        notes.append("differ: " + ", ".join(differ))
    if both:
        notes.append("fail in both: " + ", ".join(both))
    notes += [f"not built {s}: " + ", ".join(t) for s, t in unbuilt.items() if t]
    notes += [f"{t} {n}" for t, n in NOTES.items() if t in after]
    return ("DIFF" if differ else "FAIL" if both or unbuilt["after"] else "PASS"), "; ".join(notes)


def add_row(results: Path, check: str, status: str, note: str) -> None:
    """Puts a check's result into <results>/summary.json, replacing an earlier row of the same check."""
    f = results / "summary.json"
    rows = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    rows[check] = [status, note]
    f.write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")


def summary(results: Path) -> str:
    """Gets the summary table, one line per check with its status and note, then a line per engine run on what the
    cards and the PC were doing (drive.py's samples)."""
    f = results / "summary.json"
    rows = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    width = max([len(c) for c in rows] + [5])
    lines = [f"{'check':<{width}}  status  note"] + [f"{c:<{width}}  {s:<6}  {n}" for c, (s, n) in rows.items()]
    counts = [s for s, _ in rows.values()]
    lines.append("totals: " + ", ".join(f"{counts.count(s)} {s}" for s in ("PASS", "VARIES", "DIFF", "FAIL", "SKIP")))
    runs = sorted(results.glob("*/*.out.gpu.csv"))
    if runs:
        lines.append("\nWhile each run went (busy samples: mean SM clock, max temperature, mean load, PCIe link; "
                     "share of samples throttled; the PC: mean CPU, least free RAM, most page file, mean disk reads):")
    for gpu in runs:
        run = gpu.name.removesuffix(".out.gpu.csv")
        pc = summarize_pc(gpu.with_name(f"{run}.out.sys.csv"))
        lines.append(f"{gpu.parent.name} {run}: " + "; ".join(filter(None, [summarize_gpus(gpu), pc])))
    return "\n".join(lines) + "\n"


def summarize_gpus(path: Path) -> str:
    """Summarizes nvidia-smi's samples (GPU_FIELDS) per card over the samples where it was busy (all if never)."""
    cards: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(path, GPU_FIELDS):
        cards.setdefault(row["index"], []).append(row)
    out = []
    for index, rows in sorted(cards.items()):
        busy = [r for r in rows if (number(r["utilization.gpu"]) or 0) >= 10] or rows
        values = lambda key: [v for r in busy if (v := number(r[key])) is not None]  # noqa: E731
        reasons = Counter(name for r in busy for bit, name in THROTTLE.items()
                          if parse_hex(r["clocks_throttle_reasons.active"]) & bit)
        out.append(f"GPU{index} " + " ".join(filter(None, [
            f"{statistics.mean(v):.0f} MHz" if (v := values("clocks.sm")) else "",
            f"{max(v):.0f} C" if (v := values("temperature.gpu")) else "",
            f"{statistics.mean(v):.0f}% load" if (v := values("utilization.gpu")) else "",
            f"gen{max(g):.0f} x{max(w):.0f}" if (g := values("pcie.link.gen.current")) and
            (w := values("pcie.link.width.current")) else "",
            ", ".join(f"{n} {100 * c / len(busy):.0f}%" for n, c in reasons.items())])))
    return ", ".join(out)


def summarize_pc(path: Path) -> str:
    """Summarizes drive.py's samples of the PC (CPU, free RAM, page file, disk reads); empty without them."""
    rows = list(read_csv(path))
    if not rows:
        return ""
    col = lambda key: [float(r[key]) for r in rows]  # noqa: E731
    return (f"CPU {statistics.mean(col('cpu_pct')):.0f}%, RAM free {min(col('ram_free_gb')):.1f} GB, page file "
            f"{max(col('pagefile_pct')):.0f}%, disk reads {statistics.mean(col('disk_read_mbs')):.0f} MB/s")


def read_csv(path: Path, fields: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    """Reads a samples file (its own header row unless `fields` are given); rows cut short by the end of a run are
    left out."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = csv.DictReader(f, fieldnames=fields, skipinitialspace=True)
        return [r for r in rows if None not in r.values() and r.get("index") != "index"]


def number(s: str) -> float | None:
    """Gets nvidia-smi's value as a number; None for "[N/A]" and the like."""
    try:
        return float(s)
    except ValueError:
        return None



def parse_hex(text: str) -> int:
    """Reads nvidia-smi's hex bit mask; an empty or unknown value ("[N/A]", "0x" from a sample cut short) is 0."""
    try:
        return int(text, 16)
    except ValueError:
        return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
