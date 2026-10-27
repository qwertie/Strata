"""Exclusive locks that one process holds on an open file so that a second process can tell the work is taken.

On Windows the lock covers one byte (`msvcrt.locking`); elsewhere it covers the whole file (`flock`). The OS drops
the lock when the file is closed or the process ends.
"""

from __future__ import annotations

import os


def try_lock_file(descriptor: int, windows_offset: int = 0) -> None:
    """Locks the open file without waiting; raises OSError when another process holds the lock.

    `windows_offset` is the byte that is locked on Windows; a byte beyond the data leaves the data readable by others.
    The file position is moved there on Windows.
    """
    if os.name == "nt":
        import msvcrt
        os.lseek(descriptor, windows_offset, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock_file(descriptor: int, windows_offset: int = 0) -> None:
    """Releases a lock taken by try_lock_file with the same `windows_offset`."""
    if os.name == "nt":
        import msvcrt
        os.lseek(descriptor, windows_offset, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_UN)
