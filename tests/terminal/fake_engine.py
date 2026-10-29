"""A stand-in engine package and tokenizer, for tests that start a Strata server without a GPU or a model.

Used by test_component_start.py, test_runtime.py, test_updater.py and lifecycle_matrix.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from strata_control import components
from strata_common import engine_component as ec


def install_fake_engine(components_dir: Path, engine_main: str, backend: str) -> Path:
    """Installs `engine_main` (Python source) as engine 0.1.45 for `backend` in the component folder, behind the
    manifest-declared entry point a real engine package has; returns the package's folder."""
    folder = components_dir / "engine" / f"0.1.45-{components.host_platform()}-{backend}"
    folder.mkdir(parents=True)
    (folder / "engine_main.py").write_text(engine_main, encoding="utf-8")
    if os.name == "nt":
        entry = "engine.cmd"
        (folder / entry).write_bytes(f'@echo off\r\n"{sys.executable}" "%~dp0engine_main.py" %*\r\n'.encode())
    else:
        entry = "engine"
        (folder / entry).write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/engine_main.py" "$@"\n')
        (folder / entry).chmod(0o755)
    files = [{"path": name, "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
              "bytes": (folder / name).stat().st_size} for name in (entry, "engine_main.py")]
    (folder / ec.MANIFEST).write_text(json.dumps({
        "schema_version": 1, "id": "strata-engine", "version": "0.1.45", "platform": components.host_platform(),
        "backend": backend, "entrypoint": entry, "files": files}), encoding="utf-8")
    return folder


def write_byte_tokenizer(folder: Path) -> Path:
    """Creates `folder` with a tokenizer of the 256 byte tokens, one merge ("ab") and Qwen's three special tokens;
    returns the folder."""
    from tools.strata_tokenizer import BYTE_TO_UNICODE
    folder.mkdir()
    tokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    (folder / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}))
    (folder / "merges.txt").write_text("a b")
    (folder / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]))
    return folder
