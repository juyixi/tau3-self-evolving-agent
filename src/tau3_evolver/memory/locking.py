from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time
from types import TracebackType
from typing import BinaryIO


_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[tuple[str, str], ReentrantFileLock] = {}


class ReentrantFileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._handle: BinaryIO | None = None

    def __enter__(self) -> ReentrantFileLock:
        self._thread_lock.acquire()
        try:
            if self._depth == 0:
                handle = _open_lock_file(self.path)
                try:
                    _acquire_os_lock(handle)
                except BaseException:
                    handle.close()
                    raise
                self._handle = handle
            self._depth += 1
            return self
        except BaseException:
            self._thread_lock.release()
            raise

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        try:
            if self._depth < 1:
                raise RuntimeError("file lock released without a matching acquisition")
            self._depth -= 1
            if self._depth == 0:
                handle = self._handle
                self._handle = None
                if handle is None:
                    raise RuntimeError("file lock handle is missing")
                try:
                    _release_os_lock(handle)
                finally:
                    handle.close()
        finally:
            self._thread_lock.release()


def reentrant_process_lock(resource: Path, *, namespace: str) -> ReentrantFileLock:
    resolved = os.path.normcase(str(resource.resolve()))
    key = (namespace, resolved)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            digest = hashlib.sha256(f"{namespace}\0{resolved}".encode("utf-8")).hexdigest()
            lock = ReentrantFileLock(
                Path(tempfile.gettempdir()) / "tau3-evolver-locks" / f"{digest}.lock"
            )
            _LOCKS[key] = lock
        return lock


def _open_lock_file(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)
    return handle


if os.name == "nt":

    def _acquire_os_lock(handle: BinaryIO) -> None:
        import msvcrt

        retryable = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as error:
                if error.errno not in retryable:
                    raise
                time.sleep(0.05)

    def _release_os_lock(handle: BinaryIO) -> None:
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:

    def _acquire_os_lock(handle: BinaryIO) -> None:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    def _release_os_lock(handle: BinaryIO) -> None:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
