"""Picking up config file edits without a restart.

Two ways in: the file changing (inotify on Linux, a cheap stat poll elsewhere) and
SIGHUP. Both end in the same place -- read, validate, apply -- and a file that
fails validation leaves the running config exactly as it was.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import dataclasses
import hashlib
import logging
import os
import signal
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config, ConfigError, parse_config

if TYPE_CHECKING:
    from .proxy import Router

log = logging.getLogger("llm_router.reload")

# Events arrive in bursts -- an editor's save can be a create, a rename and a
# close-write in quick succession. Wait this long after the last one, then read once.
SETTLE_S = 0.2

# From <sys/inotify.h>. IN_CLOSE_WRITE fires when a writer *closes* the file, so
# it never sees a half-written one; IN_MOVED_TO catches editors (and Kubernetes
# ConfigMaps) that write a temporary file and rename it into place.
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
IN_ONLYDIR = 0x01000000
WATCH_MASK = IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE | IN_ONLYDIR

_EVENT_HEADER = struct.Struct("iIII")  # wd, mask, cookie, len; then `len` name bytes


class Inotify:
    """Just enough of inotify(7), through ctypes, to watch a few directories.

    Directories rather than the file itself: a watch follows an inode, and the
    common ways of saving -- write-and-rename, or a ConfigMap's symlink swap --
    replace the inode, which would silently end a watch on the file.
    """

    def __init__(self) -> None:
        # The libc already loaded into this process, so this works on glibc and
        # musl alike without having to find the library by name.
        self._libc = ctypes.CDLL(None, use_errno=True)
        fd = self._libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"inotify_init1: {os.strerror(err)}")
        self.fd = fd

    def watch(self, directory: Path) -> int:
        """Add a watch (idempotent: the same directory gets the same descriptor)."""
        wd = self._libc.inotify_add_watch(self.fd, os.fsencode(directory), WATCH_MASK)
        if wd < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"inotify_add_watch {directory}: {os.strerror(err)}")
        return wd

    def read(self) -> list[tuple[int, int, str]]:
        """Drain pending events as (watch descriptor, mask, name)."""
        events = []
        while True:
            try:
                buf = os.read(self.fd, 64 * 1024)
            except BlockingIOError:
                break
            if not buf:
                break
            offset = 0
            while offset + _EVENT_HEADER.size <= len(buf):
                wd, mask, _cookie, length = _EVENT_HEADER.unpack_from(buf, offset)
                start = offset + _EVENT_HEADER.size
                name = buf[start:start + length].split(b"\0", 1)[0]
                events.append((wd, mask, os.fsdecode(name)))
                offset = start + length
        return events

    def close(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self.fd)


class ConfigReloader:
    """Watches the config file and applies valid changes to a running Router."""

    def __init__(
        self,
        path: str | Path,
        router: Router,
        file_config: Config,
        *,
        watch: bool = True,
        poll_interval_s: float = 2.0,
        use_inotify: bool = True,
        handle_sighup: bool = False,
    ) -> None:
        self.path = Path(path)
        self.router = router
        # The config as the file last described it, before any CLI overrides. Only
        # used to notice edits to settings that need a restart to take effect.
        self._file_config = file_config
        # False: no file watching at all, only explicit requests (SIGHUP).
        self._watch = watch
        self.poll_interval_s = poll_interval_s
        self._use_inotify = use_inotify and sys.platform.startswith("linux")
        self._handle_sighup = handle_sighup and hasattr(signal, "SIGHUP")

        # Digest of the file as last applied, and of the last one rejected, so an
        # unchanged or already-reported file is not re-parsed and re-logged.
        self._applied: str | None = self._digest_file()
        self._rejected: str | None = None

        self._pending = asyncio.Event()
        self._force = False
        self._inotify: Inotify | None = None
        self._primary_wd: int | None = None
        self._tasks: list[asyncio.Task] = []

    # --------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._tasks.append(asyncio.create_task(self._worker()))

        if self._watch and self._use_inotify:
            try:
                self._start_inotify(loop)
            except (OSError, AttributeError) as exc:
                # AttributeError: a libc without inotify. OSError: most often the
                # per-user watch limit (fs.inotify.max_user_watches).
                log.warning("inotify unavailable (%s); polling the config instead", exc)
                self._close_inotify(loop)
        if self._watch and self._inotify is None:
            self._start_polling()

        if self._handle_sighup:
            loop.add_signal_handler(signal.SIGHUP, self.request_reload)
            if not self._watch:
                self.router.config_watch = "SIGHUP only"

        if self._watch:
            log.info(
                "watching %s for changes (%s)%s",
                self.path,
                self.router.config_watch,
                "; SIGHUP also reloads" if self._handle_sighup else "",
            )
        elif self._handle_sighup:
            log.info("file watching off; SIGHUP reloads %s", self.path)

    async def stop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._handle_sighup:
            loop.remove_signal_handler(signal.SIGHUP)
        self._close_inotify(loop)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def request_reload(self) -> None:
        """Reload now, even if the file looks unchanged. What SIGHUP does."""
        log.info("reload requested")
        self._force = True
        self._pending.set()

    # ----------------------------------------------------------------- inotify

    def _start_inotify(self, loop: asyncio.AbstractEventLoop) -> None:
        self._inotify = Inotify()
        self._primary_wd = self._inotify.watch(self.path.parent.absolute())
        self._watch_link_target()
        loop.add_reader(self._inotify.fd, self._on_inotify)
        self.router.config_watch = "inotify"

    def _watch_link_target(self) -> None:
        """If the config is a symlink, also watch the directory it points into.

        An edit to the target changes nothing in the link's own directory. The
        target can move (a ConfigMap update swaps it for a fresh directory), so
        this is re-run after every event; re-adding an existing watch is free.
        """
        if self._inotify is None or not self.path.is_symlink():
            return
        with contextlib.suppress(OSError):
            self._inotify.watch(self.path.resolve().parent)

    def _on_inotify(self) -> None:
        if self._inotify is None:
            return
        relevant = False
        # With a symlinked config, the event that matters can have any name --
        # a ConfigMap update renames `..data`, not the file -- so look at all of
        # them and let the content digest decide.
        follow_all = self.path.is_symlink()
        for wd, mask, name in self._inotify.read():
            if mask & IN_IGNORED and wd == self._primary_wd:
                # The watched directory itself went away (deleted, unmounted).
                log.warning(
                    "lost the inotify watch on %s; polling the config instead",
                    self.path.parent,
                )
                loop = asyncio.get_running_loop()
                self._close_inotify(loop)
                self._start_polling()
                self._pending.set()
                return
            if mask & IN_Q_OVERFLOW or follow_all or name == self.path.name:
                relevant = True
        if relevant:
            self._watch_link_target()
            self._pending.set()

    def _close_inotify(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._inotify is not None:
            with contextlib.suppress(Exception):
                loop.remove_reader(self._inotify.fd)
            self._inotify.close()
            self._inotify = None
            self._primary_wd = None

    # ----------------------------------------------------------------- polling

    def _start_polling(self) -> None:
        self._tasks.append(asyncio.create_task(self._poll_loop()))
        self.router.config_watch = f"polling every {self.poll_interval_s:g}s"

    def _stat(self) -> tuple[int, int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    async def _poll_loop(self) -> None:
        """Fallback watcher: one stat() per interval; the file is only read once
        it has changed *and then held still* for an interval, so a slow writer is
        never caught halfway."""
        last = self._stat()
        changed = False
        while True:
            await asyncio.sleep(self.poll_interval_s)
            current = self._stat()
            if current != last:
                last = current
                changed = True
            elif changed:
                changed = False
                self._pending.set()

    # ------------------------------------------------------------------- apply

    async def _worker(self) -> None:
        while True:
            await self._pending.wait()
            await asyncio.sleep(SETTLE_S)
            self._pending.clear()
            force, self._force = self._force, False
            try:
                self.check(force=force)
            except Exception:  # a bug here must not end the watcher
                log.exception("config reload failed unexpectedly")

    def _digest_file(self) -> str | None:
        try:
            return _digest(self.path.read_bytes())
        except OSError:
            return None

    def check(self, force: bool = False) -> bool:
        """Read the file and apply it if it changed. True if a config was applied."""
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            # Often just the gap between an editor's unlink and its rename.
            # Nothing to load; the next event will bring the new file.
            if force:
                self._reject(f"config file not found: {self.path}")
            return False
        except OSError as exc:
            self._reject(f"cannot read {self.path}: {exc}")
            return False

        digest = _digest(data)
        if not force and digest == self._applied:
            if self._rejected is not None:
                # A bad edit was reverted: the file matches what is running again.
                self._rejected = None
                self.router.config_error = None
                log.info("config file is back to the running version")
            return False
        if not force and digest == self._rejected:
            return False

        try:
            new = parse_config(data.decode())
        except (ConfigError, UnicodeDecodeError) as exc:
            self._rejected = digest
            self._reject(str(exc))
            return False

        self._warn_restart_only(new)
        running = self.router.config
        try:
            # Listening address and log file are fixed for the life of the process,
            # and may also have been overridden on the command line.
            self.router.apply_config(
                dataclasses.replace(
                    new,
                    host=running.host,
                    port=running.port,
                    log_file=running.log_file,
                )
            )
        except Exception as exc:
            log.exception("applying the new config failed")
            self._rejected = digest
            self.router.config_failed(f"internal error applying config: {exc}")
            return False

        self._applied = digest
        self._rejected = None
        self._file_config = new
        return True

    def _reject(self, message: str) -> None:
        log.error(
            "config reload rejected, still running generation %d: %s",
            self.router.config_generation,
            message,
        )
        self.router.config_failed(message)

    def _warn_restart_only(self, new: Config) -> None:
        old = self._file_config
        for label, before, after in (
            ("listen.host", old.host, new.host),
            ("listen.port", old.port, new.port),
            ("log_file", old.log_file, new.log_file),
        ):
            if before != after:
                log.warning(
                    "%s changed (%s -> %s) but only takes effect after a restart",
                    label,
                    before,
                    after,
                )


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()
