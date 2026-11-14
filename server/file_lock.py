"""Cross-platform advisory file locking (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows)."""

from __future__ import annotations

import os
from typing import Any

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows does not ship POSIX fcntl.
    _fcntl = None  # type: ignore[assignment]

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX does not ship msvcrt.
    _msvcrt = None  # type: ignore[assignment]


def _resolve_fileno(fd_or_file: Any) -> int:
    if isinstance(fd_or_file, int):
        return fd_or_file
    fileno = getattr(fd_or_file, "fileno", None)
    if callable(fileno):
        return int(fileno())
    raise TypeError(f"Expected int file descriptor or file object with fileno(), got {type(fd_or_file).__name__}")


def flock(fd_or_file: Any, operation: int) -> None:
    """Acquire or release an advisory lock on `fd_or_file` across POSIX and Windows."""
    fd = _resolve_fileno(fd_or_file)
    if _fcntl is not None:
        posix_op = 0
        if operation & LOCK_UN:
            posix_op |= _fcntl.LOCK_UN
        elif operation & LOCK_EX:
            posix_op |= _fcntl.LOCK_EX
        elif operation & LOCK_SH:
            posix_op |= _fcntl.LOCK_SH
        if operation & LOCK_NB:
            posix_op |= _fcntl.LOCK_NB
        _fcntl.flock(fd, posix_op)
        return

    if _msvcrt is not None:
        try:
            orig_pos = os.lseek(fd, 0, os.SEEK_CUR)
        except OSError:
            orig_pos = None
        try:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
            except OSError:
                pass
            if operation & LOCK_UN:
                try:
                    _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                mode = _msvcrt.LK_NBLCK if (operation & LOCK_NB) else _msvcrt.LK_LOCK
                _msvcrt.locking(fd, mode, 1)
        finally:
            if orig_pos is not None:
                try:
                    os.lseek(fd, orig_pos, os.SEEK_SET)
                except OSError:
                    pass
