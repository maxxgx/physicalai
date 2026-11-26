# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Background-thread callback wrapper."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from physicalai.capture.frame import Frame
from physicalai.config import export_config

if TYPE_CHECKING:
    from physicalai.runtime.events import InferenceEvent, LifecycleEvent, MetricsEvent, TickEvent

logger = logging.getLogger(__name__)

_DEFAULT_FLUSH_TIMEOUT_S = 5.0
# Session boundaries are never dropped from a full queue; only telemetry is.
_KEPT_LIFECYCLE_EVENTS = frozenset({"start", "stop_requested", "shutdown"})
# At most this many kept events wait at once; the oldest gives way to a new one.
_MAX_KEPT_EVENTS = 8
_QUEUE_POLL_INTERVAL_S = 0.001


@export_config(class_path="physicalai.runtime.AsyncCallback")
class AsyncCallback:
    """Wraps a callback so all hooks run on a dedicated background thread.

    The control loop takes two short locks and appends to a deque per event.

    Overflow policy, with the worker stalled: session ``start``,
    ``stop_requested`` and ``shutdown`` events and flush requests are kept
    events; everything else is telemetry. At most ``max_queue`` telemetry
    events wait; a new one evicts the oldest telemetry event. At most 8 kept
    events wait; a new session event evicts the oldest kept event, so older
    sessions give way to the latest ones, and a flush request waits for room
    instead. The queue therefore never holds more than ``max_queue + 8``
    events, and finding what to evict looks at no more than 9 entries for
    telemetry. Evicted events are released after the locks.

    ``flush()`` and ``close()`` each wait at most ``flush_timeout_s`` for the
    queue to drain. If the inner callback is still busy when ``close()`` gives
    up, the inner ``close()`` runs on the worker thread once that event is done.
    """

    _ACTION_HOOKS = ("on_action_ready", "on_action_sent")

    def __init__(  # noqa: D107
        self,
        inner: Any,  # noqa: ANN401
        max_queue: int = 1024,
        flush_timeout_s: float = _DEFAULT_FLUSH_TIMEOUT_S,
    ) -> None:
        if max_queue <= 0:
            msg = f"max_queue must be positive, got {max_queue}"
            raise ValueError(msg)
        dropped = [h for h in self._ACTION_HOOKS if hasattr(inner, h)]
        if dropped:
            msg = (
                f"{type(inner).__name__} defines action hooks {dropped} which "
                "AsyncCallback does not forward (use synchronous attachment instead)"
            )
            raise TypeError(msg)
        self._inner = inner
        self._max_queue = max_queue
        self._flush_timeout_s = flush_timeout_s
        self._queue: deque[tuple[str, Any]] = deque()
        self._queue_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._kept = 0  # kept events in the queue, under _queue_lock
        self._exit_lock = threading.Lock()
        self._worker_done = False
        self._close_inner_on_exit = False
        self._stop = threading.Event()
        self._has_work = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="AsyncCallbackWorker", daemon=True)
        self._thread.start()

    def on_tick(self, event: TickEvent) -> None:
        """Enqueue tick event, copying borrowed frame buffers to prevent dangling SHM refs.

        Zero-copy SharedCamera frames are views into iceoryx2 shared memory that become
        invalid on the next read_latest() call. Since the background worker may process
        this event after the next tick, borrowed frames are replaced with owned copies
        before enqueuing.
        """
        camera_frames = event.camera_frames
        if any(not f.data.flags.owndata for f in camera_frames.values()):
            copied = {
                name: Frame(data=f.data.copy(), timestamp=f.timestamp, sequence=f.sequence)
                if not f.data.flags.owndata
                else f
                for name, f in camera_frames.items()
            }
            event = dataclasses.replace(event, camera_frames=copied)
        self._enqueue("on_tick", event)

    def on_inference(self, event: InferenceEvent) -> None:
        """Enqueue inference event for background processing."""
        self._enqueue("on_inference", event)

    def on_lifecycle(self, event: LifecycleEvent) -> None:
        """Enqueue lifecycle event for background processing."""
        self._enqueue("on_lifecycle", event)

    def on_metrics(self, event: MetricsEvent) -> None:
        """Enqueue metrics event for background processing."""
        self._enqueue("on_metrics", event)

    def close(self) -> None:
        """Drain queued events, stop the worker, and close the inner callback.

        Events still queued after ``flush_timeout_s`` are dropped, so a worker
        that is stuck in the inner callback finishes at most the event it is in.
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        deadline = time.monotonic() + self._flush_timeout_s
        try:
            self._flush_until(deadline)
        finally:
            with self._queue_lock:
                leftover = list(self._queue)
                self._queue.clear()
                self._kept = 0
            del leftover  # released outside the lock
            self._stop.set()
            self._has_work.set()
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            with self._exit_lock:
                deferred = not self._worker_done
                self._close_inner_on_exit = deferred
            if not deferred:
                close_fn = getattr(self._inner, "close", None)
                if close_fn is not None:
                    close_fn()

    def _stop_worker(self) -> None:
        """Stop the worker without closing the inner callback (rollback of a failed setup)."""
        with self._close_lock:
            self._closed = True
        self._stop.set()
        self._has_work.set()

    def flush(self) -> None:
        """Wait until all events queued before this call are delivered."""
        self._flush_until(time.monotonic() + self._flush_timeout_s)

    def _flush_until(self, deadline: float) -> None:
        if not self._thread.is_alive():
            return
        done = threading.Event()
        while self._thread.is_alive():
            with self._queue_lock:
                if self._kept < _MAX_KEPT_EVENTS:  # a flush never evicts a session event
                    self._queue.append(("_flush", done))
                    self._kept += 1
                    break
            if time.monotonic() >= deadline:
                logger.warning("Async callback did not flush within %.1fs", self._flush_timeout_s)
                return
            time.sleep(_QUEUE_POLL_INTERVAL_S)
        else:
            return
        self._has_work.set()
        if not done.wait(timeout=max(0.0, deadline - time.monotonic())):
            logger.warning("Async callback did not flush within %.1fs", self._flush_timeout_s)

    def _enqueue(self, method: str, event: Any) -> None:  # noqa: ANN401
        evicted = None
        with self._close_lock:
            if self._closed:
                return
            with self._queue_lock:
                item = (method, event)
                kept = _is_kept(item)
                if kept and self._kept >= _MAX_KEPT_EVENTS:
                    evicted = self._evict_locked(kept=True)
                elif not kept and len(self._queue) - self._kept >= self._max_queue:
                    evicted = self._evict_locked(kept=False)
                self._queue.append(item)
                self._kept += kept
        del evicted  # an evicted event (and its frames) is released outside the locks
        self._has_work.set()

    def _evict_locked(self, *, kept: bool) -> tuple[str, Any] | None:
        """Remove the oldest kept (or telemetry) event; the caller holds both locks.

        Returns:
            The removed event, for the caller to release after the locks.
        """
        for i, item in enumerate(self._queue):
            if _is_kept(item) is kept:
                del self._queue[i]
                self._kept -= kept
                return item
        return None

    def _worker(self) -> None:
        try:
            self._work()
        finally:
            with self._exit_lock:
                self._worker_done = True
                close_inner = self._close_inner_on_exit
            if close_inner:
                close_fn = getattr(self._inner, "close", None)
                if close_fn is not None:
                    try:
                        close_fn()
                    except Exception:
                        logger.exception("AsyncCallback inner %r.close failed", self._inner)

    def _work(self) -> None:
        while not self._stop.is_set():
            self._has_work.wait()
            self._has_work.clear()
            while True:
                with self._queue_lock:
                    if not self._queue:
                        break
                    method, event = self._queue.popleft()
                    self._kept -= _is_kept((method, event))
                if method == "_flush":
                    flush_fn = getattr(self._inner, "flush", None)
                    if flush_fn is not None:
                        try:
                            flush_fn()
                        except Exception:
                            logger.exception("AsyncCallback inner %r.flush failed", self._inner)
                    event.set()
                    continue
                fn = getattr(self._inner, method, None)
                if fn is not None:
                    try:
                        fn(event)
                    except Exception:
                        logger.exception("AsyncCallback inner %r.%s failed", self._inner, method)


def _is_kept(item: tuple[str, Any]) -> bool:
    method, event = item
    return method == "_flush" or (method == "on_lifecycle" and event.event in _KEPT_LIFECYCLE_EVENTS)
