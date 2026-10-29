"""Replace a file that another program may be reading, on Windows as well as everywhere else.

A job record (a Time-skip `run.json`, a tuning `job.json`) is rewritten at every phase and READ by the app's host every couple of seconds.
On Windows, `os.replace` onto a file that is open in another process fails with PermissionError ("Access is denied") for as long as the
reader holds it. That is a split second, but a job saves many times, so a Time-skip or a tune died at random with that error. The fix is
to wait that split second and try again; any other error, and a PermissionError that does not go away, is raised as before.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Union

PathLike = Union[str, "os.PathLike[str]"]

#: how long a held file is waited for, in total (seconds) and between tries
PATIENCE_S = 5.0
STEP_S = 0.02


def replace(src: PathLike, dst: PathLike, *, patience: float = PATIENCE_S, sleep=time.sleep, clock=time.monotonic) -> None:
    deadline = clock() + patience
    step = STEP_S
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or clock() >= deadline:
                raise
            sleep(step)
            step = min(step * 1.5, 0.25)


def write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Write `text` to `path` through a neighbouring temporary file, so a reader never sees half a record."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp") if target.suffix else target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    replace(tmp, target)


def write_bytes(path: PathLike, data: bytes, *, private: bool = False) -> None:
    """Replaces the file's contents in one step: `data` goes to a temporary file in the same folder, reaches the disk
    and is then renamed over `path` (see replace), so neither a crash nor a reader ever sees half a file. The folder
    must exist. `private`: on Windows the file also gets an ACL that shuts out other accounts (best effort; elsewhere
    the temporary file's mode 0600 is kept)."""
    target = Path(path)
    descriptor, temp = tempfile.mkstemp(prefix=f".{target.name}-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        replace(temp, target)
        if os.name != "nt":
            try:
                folder = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(folder)
                finally:
                    os.close(folder)
            except OSError:
                pass                           # the file is already replaced: a failed folder sync is no failed save
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    if private and os.name == "nt":
        try:
            from strata_base import private_files
            private_files.restrict_to_current_user(target, directory=False)
        except Exception:                              # noqa: BLE001 - a file the user can read is still theirs
            pass


def write_new(path: Path, data: bytes) -> None:
    """Creates the file (mode 0600 on POSIX) and writes `data` through to the disk. Raises FileExistsError when the
    file is already there, and removes a half-written file when the write fails."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
