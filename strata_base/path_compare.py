"""Whether two paths name one folder, or lie on one volume (so that a move between them is a rename)."""

from __future__ import annotations

import os
from pathlib import Path


def same_folder(a: Path, b: Path) -> bool:
    """Tells whether `a` and `b` are one folder however they are spelled (another case, a short 8.3 name, a link).
    Paths that do not exist are compared by their normalized spelling."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def same_drive(a: Path, b: Path) -> bool:
    """Tells whether `a` and `b` are on one volume. A path that does not exist yet counts as being on the volume of its
    nearest existing parent folder."""
    try:
        return os.stat(nearest_existing(Path(a))).st_dev == os.stat(nearest_existing(Path(b))).st_dev
    except OSError:
        return False


def nearest_existing(path: Path) -> Path:
    """Gets `path` itself when it exists, else its nearest parent folder that does (the root at the latest)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return path
