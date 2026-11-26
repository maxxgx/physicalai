# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Any

from physicalai.runtime.callbacks.async_callback import AsyncCallback

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from physicalai.runtime.events import InferenceEvent, LifecycleEvent, MetricsEvent, TickEvent

logger = logging.getLogger(__name__)

_INFERENCE_QUEUE_MAXLEN = 64
_BACKGROUND_QUEUE = 16
_BACKGROUND_FLUSH_TIMEOUT_S = 0.5
_ACTION_HOOKS = ("on_action_ready", "on_action_sent")  # AsyncCallback cannot forward these


class _CallbackBus:
    """Internal dispatch bus for runtime callbacks.

    Two dispatch modes:
    - Fire-and-forget (emit_*, ``invoke_on_action_sent``): notification
      hooks, exceptions isolated and logged — a broken callback can't stop
      the run.
    - Request-response (``invoke_on_action_ready``): chains each callback's
      transform of the outgoing action. Exceptions propagate instead of
      being isolated — a callback that fails partway through the chain
      means the action can no longer be trusted, so it must not be sent to
      the robot silently un-transformed.

    Thread safety: ``emit_inference`` may be called from either the control
    thread (SyncExecution) or the inference thread (AsyncExecution). All other
    methods run on the control thread only.
    """

    def __init__(self, callbacks: Sequence[Any]) -> None:
        self._callbacks = list(callbacks)
        self._background_started = False
        self._inference_queue: deque[InferenceEvent] = deque(maxlen=_INFERENCE_QUEUE_MAXLEN)
        self._close_lock = threading.Lock()
        self._closed = False

    def emit_tick(self, event: TickEvent) -> None:
        self._drain_inference()
        for cb in self._callbacks:
            fn = getattr(cb, "on_tick", None)
            if fn is None:
                continue
            try:
                fn(event)
            except Exception:
                logger.exception("Callback %r failed in on_tick", cb)

    def emit_inference(self, event: InferenceEvent) -> None:
        """Enqueue inference event from background thread for control-thread delivery."""
        self._inference_queue.append(event)

    def start_background(self) -> None:
        """Put callbacks marked ``run_in_background`` behind :class:`AsyncCallback`, once.

        Callbacks that may block (e.g. RerunCallback) then run behind a small
        drop-oldest queue on their own thread and cannot stall the loop. Called
        at the start of a run, so a runtime that never runs starts no thread.
        A marked callback with action hooks stays synchronous, with a warning.
        """
        if self._background_started:
            return
        self._background_started = True
        wrapped: list[Any] = []
        started: list[AsyncCallback] = []
        try:
            for cb in self._callbacks:
                if getattr(cb, "run_in_background", False) is not True:
                    wrapped.append(cb)
                elif any(hasattr(cb, hook) for hook in _ACTION_HOOKS):
                    logger.warning(
                        "%s asks to run in the background but defines action hooks; running it synchronously",
                        type(cb).__name__,
                    )
                    wrapped.append(cb)
                else:
                    started.append(
                        AsyncCallback(cb, max_queue=_BACKGROUND_QUEUE, flush_timeout_s=_BACKGROUND_FLUSH_TIMEOUT_S)
                    )
                    wrapped.append(started[-1])
        except BaseException:
            for worker in started:
                worker._stop_worker()  # noqa: SLF001  # the user's callbacks stay the bus's to close
            raise
        self._callbacks = wrapped

    def emit_lifecycle(self, event: LifecycleEvent) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_lifecycle", None)
            if fn is None:
                continue
            try:
                fn(event)
            except Exception:
                logger.exception("Callback %r failed in on_lifecycle", cb)

    def invoke_on_action_ready(self, *, action: np.ndarray, step: int) -> np.ndarray:
        """Chain each callback's ``on_action_ready``, threading the return value.

        Every callback must return a valid action (no ``None`` sentinel) — a
        callback that doesn't want to change anything returns its input
        unchanged.

        Unlike the ``emit_*`` hooks, exceptions here are not isolated: a
        callback (e.g. a safety filter) that fails partway through the chain
        means the action can no longer be trusted, so the failure is logged
        (identifying which callback raised) and then re-raised instead of
        silently sending a partially-transformed action to the robot.

        Returns:
            The action after every callback has had a chance to transform it.
        """
        result = action
        for cb in self._callbacks:
            fn = getattr(cb, "on_action_ready", None)
            if fn is None:
                continue
            try:
                modified_action = fn(action=result, step=step)
                if modified_action is None:
                    logger.warning("Callback %r returned None from on_action_ready, ignoring", cb)
                else:
                    result = modified_action
            except Exception:
                logger.exception("Callback %r failed in on_action_ready", cb)
                raise
        return result

    def invoke_on_action_sent(self, *, action: np.ndarray, step: int) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_action_sent", None)
            if fn is None:
                continue
            try:
                fn(action=action, step=step)
            except Exception:
                logger.exception("Callback %r failed in on_action_sent", cb)

    def emit_metrics(self, event: MetricsEvent) -> None:
        """Fire-and-forget dispatch to ``on_metrics``, exceptions isolated.

        Called synchronously on the control thread (e.g. from within
        ``PolicySource.update()``), so unlike ``emit_inference`` no queue is
        needed here.
        """
        for cb in self._callbacks:
            fn = getattr(cb, "on_metrics", None)
            if fn is None:
                continue
            try:
                fn(event)
            except Exception:
                logger.exception("Callback %r failed in on_metrics", cb)

    def close(self) -> None:
        """Flush and close every callback exactly once."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.flush()
        finally:
            first_base_exception: BaseException | None = None
            for cb in self._callbacks:
                close_fn = getattr(cb, "close", None)
                if close_fn is not None:
                    try:
                        close_fn()
                    except Exception:
                        logger.exception("Callback %r failed in close", cb)
                    except BaseException as exc:  # noqa: BLE001
                        if first_base_exception is None:
                            first_base_exception = exc
            if first_base_exception is not None:
                raise first_base_exception

    def flush(self) -> None:
        """Deliver queued inference events and flush callback-owned buffers."""
        self._drain_inference()
        for cb in self._callbacks:
            flush_fn = getattr(cb, "flush", None)
            if flush_fn is not None:
                try:
                    flush_fn()
                except Exception:
                    logger.exception("Callback %r failed in flush", cb)

    def _drain_inference(self) -> None:
        while self._inference_queue:
            event = self._inference_queue.popleft()
            for cb in self._callbacks:
                fn = getattr(cb, "on_inference", None)
                if fn is None:
                    continue
                try:
                    fn(event)
                except Exception:
                    logger.exception("Callback %r failed in on_inference", cb)
