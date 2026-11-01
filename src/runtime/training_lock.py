"""OS-released writer lock: works on Windows and POSIX, including process crashes."""
from contextlib import contextmanager
import os
from pathlib import Path


def _acquire(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def training_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".training.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        try:
            _acquire(handle)
        except OSError as exc:
            raise RuntimeError("Training is already active or the writer lock is unavailable.") from exc
        try:
            yield
        finally:
            _release(handle)


def training_in_progress(root):
    path = Path(root) / ".training.lock"
    if not path.exists():
        return False
    try:
        with path.open("r+b") as handle:
            _acquire(handle)
            _release(handle)
    except OSError:
        return True
    return False
