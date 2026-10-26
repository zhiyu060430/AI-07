"""Cross-platform exclusive file locks.

POSIX uses ``fcntl.flock``. Windows uses ``msvcrt.locking`` on the first byte.
The Windows call is ``LK_NBLCK`` plus a short sleep, because ``LK_LOCK`` gives
up after about ten seconds and a bake can take much longer.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_lock(path: Path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("a+b")
    try:
        _acquire(handle)
        yield handle
    finally:
        try:
            _release(handle)
        finally:
            handle.close()


def _acquire(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                # A conflicting lock is retryable. Anything else is not.
                winerror = getattr(exc, "winerror", None)
                if winerror not in {None, 33} and exc.errno not in {None, 11, 13, 16, 35}:
                    raise
                time.sleep(0.05)
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
