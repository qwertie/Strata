"""SHA-256 digests of files, read in pieces so that a file of any size needs little memory.

This module has no network code, so the start path can verify an engine without loading the download machinery.
"""

from __future__ import annotations

import hashlib
import os

CHUNK = 1 << 20


def sha256_file(path: str | os.PathLike) -> str:
    """Gets the SHA-256 of the file's bytes as lower-case hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()
