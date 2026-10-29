"""The official DeepSeek-V4.1-Flash GGUF sets (IST-DAS 1.75 / 2.00 / 2.25 bpw), recognised by their pinned file names and sizes.

The Terminal's setup downloads them (catalog.MULTI_FILE_FAMILIES, DSV41_ISTA_FILES) and accepts the files of their earlier
layouts (DSV41_OLDER_FILES: ...-0000N-of-00004.gguf, model-2.00bpw-0000N-of-00004.gguf, engram-IQ4_NL.gguf). The app's
"I already have the model" preview used to know only the Qwen fingerprints, so a folder with such a set ended in "the folder holds more than one model" (the Engram is a second file) or, for one shard,
"this model type is not supported by the Strata engine". This module is how Control says what a folder of these files is, from the names and
sizes alone: no tensor is read and no 40 GB file is hashed in a preview (setup checks the SHA-256 once, when the set is adopted).

The tables are the catalog's (strata_common/catalog.py, never copied), so a new size added there is recognised here.
"""
from __future__ import annotations

import functools

from strata_common import catalog

FAMILY = "deepseek41"


@functools.cache
def tables() -> dict:
    """{"sets": {model: {"label", "layouts": [{name: bytes}, ...]}}, "engram": {name: bytes}} for the official sets; their layouts
    newest first (catalog.make_ista_set)."""
    family = catalog.MULTI_FILE_FAMILIES[FAMILY]
    sizes = {name: pin[0] for name, pin in catalog.DSV41_ALL_FILES.items()}
    result: dict = {"sets": {}, "engram": {}}
    for model, entry in family["models"].items():
        if entry.get("recipe") != "ista":
            continue                                # Unsloth's 508 GB set is not an "already have it" case of the app
        result["sets"][model] = {"label": entry.get("label", model),
                                 "layouts": [{name: sizes[name] for name in layout} for layout in entry["layouts"]]}
    for name in (catalog.DSV41_ISTA_ENGRAM, catalog.DSV41_FIRST_NAMES[catalog.DSV41_ISTA_ENGRAM]):
        result["engram"][name] = sizes[name]
    return result


def match(files: dict[str, int]) -> list[dict]:
    """The official sets the files {name: bytes} of one folder belong to, best first: complete sets before partial ones.
    Each: {"model", "label", "complete", "present": [names], "missing": [names], "wrong_size": [names], "legacy": bool (an earlier
    layout), "engram": name|None}, one for each layout of a set at least one of whose shards is there; names and sizes are the pinned
    ones.  A folder that mixes layouts is no complete set."""
    known = tables()
    engram = next((name for name in known["engram"] if files.get(name) == known["engram"][name]), None)
    found = []
    for model, entry in known["sets"].items():
        for layout, pinned in enumerate(entry["layouts"]):
            present = [n for n in pinned if n in files]
            if not present:
                continue
            wrong = [n for n in present if files[n] != pinned[n]]
            missing = [n for n in pinned if n not in files]
            found.append({"model": model, "label": entry["label"], "legacy": layout > 0, "present": sorted(present), "missing": missing,
                          "wrong_size": wrong, "complete": not missing and not wrong, "engram": engram,
                          "bytes": sum(pinned[n] for n in present if n not in wrong) + (known["engram"][engram] if engram else 0)})
    found.sort(key=lambda row: (not row["complete"], len(row["missing"]) + len(row["wrong_size"]), row["model"]))
    return found
