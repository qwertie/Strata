"""Checks that code in the new layout (docs/ARCHITECTURE.md) uses only the layers below it.

    python -m pytest tests/architecture

C++ is checked by its `#include "strata/..."` lines (not by includes relative to the file), Python by its absolute
imports (not in tests, which may set up any part of the system). Includes of the old folders that the refactor is still
emptying (OLD_CPP_FOLDERS) are allowed until they are gone.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CPP_SUFFIXES = {".cpp", ".cc", ".c", ".hpp", ".h", ".inc", ".inl", ".cu", ".cuh"}
CPP_INCLUDE = re.compile(r'^\s*#\s*include\s*"strata/([^"]+)"', re.MULTILINE)
# Folders under src/ and include/strata/ that the refactor empties and deletes (docs/ARCHITECTURE.md#old-folders)
OLD_CPP_FOLDERS = {"artifact", "core", "dsv41", "hip_compat", "kernels", "ngram", "plan", "platform", "prefill",
                   "qwen35", "spec"}
# The new C++ folders (relative to src/ or include/strata/) by layer. Files in one folder, including its subfolders,
# may include each other; folders in one layer may not, except as CPP_SAME_LAYER_USES allows. A new model family
# is added here.
CPP_LAYERS = {"base": 0, "base/gpu": 0, "base/platform": 0, "common": 1, "models/common": 2,
              "models/deepseekv41": 3, "models/qwen36": 3, "models/qwen38": 3, "program": 4}
CPP_SAME_LAYER_USES = {"base/gpu": {"base/platform"}}
# Python packages and scripts by layer; packages in one layer must not import each other
PY_LAYERS = {"strata_base": 0, "strata_common": 1, "strata_server": 2, "strata_control": 2, "strata_cli": 3,
             "setup": 4, "tools": 4}
PY_OLD_NAMES = {"serve": "strata_server"}
# The packages whose imports are checked
CHECKED_PY_PACKAGES = ("strata_base", "strata_common", "strata_control", "serve", "strata_cli")


def test_cpp_includes_follow_layers():
    problems = [f"{path.relative_to(ROOT).as_posix()}: {problem}"
                for path, own in find_cpp_files()
                for target in CPP_INCLUDE.findall(path.read_text(encoding="utf-8", errors="replace"))
                if (problem := find_cpp_problem(own, target))]
    assert not problems, "\n".join(problems)


def test_python_imports_follow_layers():
    problems = []
    for package in CHECKED_PY_PACKAGES:
        for path in sorted(p for p in (ROOT / package).rglob("*.py") if not p.name.startswith("test_")):
            for target in find_imported_packages(path):
                if problem := find_python_problem(package, target):
                    problems.append(f"{path.relative_to(ROOT).as_posix()}: {problem}")
    assert not problems, "\n".join(problems)


def test_rules_catch_violations():
    assert find_cpp_problem("models/qwen38/session.cpp", "models/common/geometry.hpp") is None
    assert find_cpp_problem("models/qwen38/session.cpp", "core/layer.hpp") is None
    assert find_cpp_problem("program/generate.cpp", "models/qwen36/engine.hpp") is None
    assert find_cpp_problem("common/state/pool.cpp", "models/qwen38/session.hpp")
    assert find_cpp_problem("base/gpu/device.cu", "common/state/pool.hpp")
    assert find_cpp_problem("models/qwen36/engine.cu", "models/qwen38/session.hpp")
    assert find_cpp_problem("base/gpu/device.cu", "base/platform/numa.hpp") is None
    assert find_cpp_problem("base/platform/numa.cpp", "base/gpu/device.hpp")
    assert find_cpp_problem("common/state/pool.cpp", "common/spec/draft.hpp") is None
    assert find_cpp_problem("common/state/pool.cpp", "nonexistent/x.hpp")
    assert find_cpp_problem("models/newfamily/engine.cu", "common/state/pool.hpp")
    assert find_python_problem("strata_common", "strata_base") is None
    assert find_python_problem("strata_server", "strata_common") is None
    assert find_python_problem("strata_base", "strata_common")
    assert find_python_problem("strata_server", "strata_control")
    assert find_python_problem("strata_control", "serve")
    assert find_python_problem("strata_control", "tools")


def find_cpp_files():
    """Yields each C++ file in the new folders with its path relative to src/ or include/strata/, which is how
    an include names it."""
    for base in (ROOT / "src", ROOT / "include" / "strata"):
        for folder in sorted({folder.split("/")[0] for folder in CPP_LAYERS}):
            for path in sorted((base / folder).rglob("*")):
                if path.suffix in CPP_SUFFIXES:
                    yield path, path.relative_to(base).as_posix()


def find_cpp_problem(own: str, target: str) -> str | None:
    """Gets what is wrong with file `own` including `target` (both relative to src/ or include/strata/), or
    None if the layers allow it."""
    own_folder, target_folder = get_cpp_folder(own), get_cpp_folder(target)
    if own_folder is None:
        return "is in no folder of CPP_LAYERS"
    if target_folder is None:
        if target.split("/", 1)[0] in OLD_CPP_FOLDERS:
            return None
        return f"includes strata/{target}, which is in no folder of CPP_LAYERS"
    if CPP_LAYERS[target_folder] > CPP_LAYERS[own_folder]:
        return f"includes strata/{target} from a higher layer"
    if (CPP_LAYERS[target_folder] == CPP_LAYERS[own_folder] and target_folder != own_folder
            and target_folder not in CPP_SAME_LAYER_USES.get(own_folder, ())):
        return f"includes strata/{target}, which {own_folder}/ may not use"
    return None


def get_cpp_folder(path: str) -> str | None:
    """Gets the longest CPP_LAYERS key that contains `path` (relative to src/ or include/strata/), or None
    for an old folder."""
    return max((folder for folder in CPP_LAYERS if path.startswith(folder + "/")), key=len, default=None)


def find_imported_packages(path: Path):
    """Yields the top-level package of each import in a Python file."""
    for node in ast.walk(ast.parse(path.read_bytes(), str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def find_python_problem(package: str, target: str) -> str | None:
    """Gets what is wrong with `package` importing `target`, or None if the layers allow it."""
    package, target = PY_OLD_NAMES.get(package, package), PY_OLD_NAMES.get(target, target)
    if target == package or target not in PY_LAYERS:
        return None
    if PY_LAYERS[target] >= PY_LAYERS[package]:
        return f"imports {target}, which is not in a lower layer"
    return None
