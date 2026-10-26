"""Counts the lines of Strata's own code at a git revision, so the code-sharing refactor
(docs/REFACTOR_PLAN.md) can show how much smaller each step made the code base.

    python tools/loc_report.py [REV]                       totals by top folder and language
    python tools/loc_report.py --detail [REV]              by area (src/core, src/common/state, ...)
    python tools/loc_report.py --compare BASE [REV]        BASE vs REV (default HEAD), with the change

A "line" is a non-blank line; comment lines count. Files come from git, not the working tree, so the
result does not depend on what is checked out. Copied third-party code, community benchmark results and
the desktop app (apps/) are left out: see EXCLUDED.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Path prefixes that are not Strata's own code (is_counted also leaves out every */vendor/ folder)
EXCLUDED = ("third_party/", "sycl/third_party/", "sycl/include/dpct/", "bench/results/", "apps/")
LANGUAGES = {
    ".cpp": "C++", ".cc": "C++", ".c": "C++", ".hpp": "C++", ".h": "C++", ".inc": "C++",
    ".cu": "CUDA", ".cuh": "CUDA", ".metal": "Metal", ".mm": "Obj-C++", ".py": "Python",
    ".cmake": "CMake", ".sh": "Scripts", ".bat": "Scripts", ".cmd": "Scripts", ".ps1": "Scripts",
    ".js": "Web", ".mjs": "Web", ".css": "Web", ".html": "Web",
}
# Folders whose subfolders are separate areas in --detail
DEEP_FOLDERS = ("src/base/", "src/common/", "src/models/", "sycl/src/", "metal/src/", "tests/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rev", nargs="?", default="HEAD")
    ap.add_argument("--compare", metavar="BASE", help="show BASE, REV and the change between them")
    ap.add_argument("--detail", action="store_true", help="one row per area instead of per top folder")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # the report has a non-ASCII arrow; Windows consoles default to cp1252
    key = area_of if args.detail else group_of
    if args.compare:
        print(format_comparison(count_lines(args.compare), count_lines(args.rev), key, args.compare, args.rev))
    else:
        print(format_totals(count_lines(args.rev), key, args.rev))
    return 0


def count_lines(rev: str) -> dict[str, int]:
    """Gets the non-blank line count of every counted file at `rev`, keyed by path."""
    blobs = {}
    # -z: paths are NUL-separated and never quoted
    for entry in git("ls-tree", "-r", "-z", rev).decode("utf-8").split("\0")[:-1]:
        meta, path = entry.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind == "blob" and mode != "120000" and is_counted(path):  # not a submodule or a symlink
            blobs[path] = sha
    out = git("cat-file", "--batch", input="\n".join(blobs.values()).encode() + b"\n")
    counts, pos = {}, 0
    for path in blobs:
        header_end = out.index(b"\n", pos)
        size = int(out[pos:header_end].split()[2])
        body = out[header_end + 1:header_end + 1 + size]
        counts[path] = sum(1 for line in body.splitlines() if line.strip())
        pos = header_end + 1 + size + 1
    return counts


def is_counted(path: str) -> bool:
    """Checks whether `path` is Strata's own code in a language that the report counts."""
    return language_of(path) is not None and not path.startswith(EXCLUDED) and "/vendor/" not in path


def language_of(path: str) -> str | None:
    """Gets the language column that `path` is reported under, or None for a file that is not code."""
    name = path.rsplit("/", 1)[-1]
    if name == "CMakeLists.txt":
        return "CMake"
    if name.endswith(".cu.in"):
        return "CUDA"
    return LANGUAGES.get(Path(name).suffix)


def group_of(path: str) -> str:
    """Gets the top folder that `path` is reported under by default ("(root)" for files at the top)."""
    path = get_source_path(path)
    return path.split("/", 1)[0] if "/" in path else "(root)"


def area_of(path: str) -> str:
    """Gets the folder that `path` is reported under by --detail."""
    path = get_source_path(path)
    parts = path.split("/")
    depth = 3 if path.startswith(DEEP_FOLDERS) else 2
    return "/".join(parts[:min(depth, len(parts) - 1)]) or "(root)"


def get_source_path(path: str) -> str:
    """Maps a header in include/strata/X to src/X (sycl/include/strata/X to sycl/src/X), so that
    headers are counted with their sources."""
    for headers, sources in (("include/strata/", "src/"), ("sycl/include/strata/", "sycl/src/")):
        if path.startswith(headers):
            return sources + path[len(headers):]
    return path


def format_totals(counts: dict[str, int], key, rev: str) -> str:
    """Formats `counts` as two Markdown tables: lines per `key(path)` and lines per language."""
    total = f"| **total** | **{sum(counts.values()):,}** |"
    lines = [f"Non-blank lines at `{rev}`"]
    for title, sums in (("Folder", tally(counts, key)), ("Language", tally(counts, language_of))):
        rows = (f"| {k} | {v:,} |" for k, v in sorted(sums.items(), key=lambda kv: -kv[1]))
        lines += ["", f"| {title} | Lines |", "|---|---:|", *rows, total]
    return "\n".join(lines)


def format_comparison(base: dict[str, int], new: dict[str, int], key, base_rev: str, new_rev: str) -> str:
    """Formats a Markdown table of lines per `key(path)` in `base` and `new`, and the change."""
    before, after = tally(base, key), tally(new, key)
    rows = []
    for k in sorted(before.keys() | after.keys(), key=lambda k: -max(before.get(k, 0), after.get(k, 0))):
        b, a = before.get(k, 0), after.get(k, 0)
        rows.append(f"| {k} | {b:,} | {a:,} | {a - b:+,} |")
    tb, ta = sum(base.values()), sum(new.values())
    percent = f" ({(ta - tb) / tb:+.1%})" if tb else ""
    total = f"| **total** | **{tb:,}** | **{ta:,}** | **{ta - tb:+,}**{percent} |"
    return "\n".join([f"Non-blank lines, `{base_rev}` → `{new_rev}`\n",
                      "| Folder | Before | After | Change |", "|---|---:|---:|---:|", *rows, total])


def tally(counts: dict[str, int], key) -> dict[str, int]:
    """Sums the line counts of the paths that `key` maps to the same name."""
    sums: dict[str, int] = defaultdict(int)
    for path, n in counts.items():
        sums[key(path)] += n
    return sums


def git(*args: str, input: bytes | None = None) -> bytes:
    """Runs git in the repository that holds this script and returns its standard output."""
    return subprocess.run(["git", "-C", str(ROOT), *args], input=input, capture_output=True, check=True).stdout


if __name__ == "__main__":
    sys.exit(main())
