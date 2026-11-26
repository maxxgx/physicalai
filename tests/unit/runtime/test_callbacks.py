# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ARG005, D100, D101, D102, E741, F401, I001, PLC2701, PLR2004, PLR6301, S101, SLF001, TC002, TC003

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from physicalai.runtime._callback_bus import _CallbackBus
from physicalai.runtime import AsyncCallback, ConsoleCallback, JsonlCallback
from physicalai.runtime.events import InferenceEvent, LifecycleEvent, TickEvent
from tests.unit.runtime.conftest import FakeRobotObservation

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame


class TestCallbackBus:
    def _make_tick_event(self, step: int = 0) -> TickEvent:
        return TickEvent(
            session_id="test",
            step=step,
            timestamp=0.0,
            robot_state=FakeRobotObservation(joint_positions=np.zeros(3)),
            camera_frames={},
            action_sent=np.zeros(3),
            loop_duration_s=0.03,
            sleep_time_s=0.003,
            stale_obs=False,
        )

    def _make_inference_event(self) -> InferenceEvent:
        return InferenceEvent(
            session_id="test",
            timestamp=0.0,
            latency_s=0.1,
            offset=3,
            chunk=np.zeros((10, 3)),
        )

    def _make_lifecycle_event(self, event: str = "start") -> LifecycleEvent:
        return LifecycleEvent(
            session_id="test",
            timestamp=0.0,
            event=event,
            metadata={"fps": 30},
        )

    def test_emit_tick_dispatches_to_callback(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])
        event = self._make_tick_event()
        bus.emit_tick(event)
        cb.on_tick.assert_called_once_with(event)

    def test_emit_lifecycle_dispatches(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])
        event = self._make_lifecycle_event()
        bus.emit_lifecycle(event)
        cb.on_lifecycle.assert_called_once_with(event)

    def test_emit_inference_queues_for_drain(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])
        event = self._make_inference_event()
        bus.emit_inference(event)
        cb.on_inference.assert_not_called()
        bus.emit_tick(self._make_tick_event())
        cb.on_inference.assert_called_once_with(event)

    def test_invoke_on_action_ready_chains(self) -> None:
        cb1 = MagicMock()
        cb1.on_action_ready.return_value = np.ones(3)
        cb2 = MagicMock()
        cb2.on_action_ready.side_effect = lambda *, action, step: action

        bus = _CallbackBus([cb1, cb2])
        original = np.zeros(3)
        result = bus.invoke_on_action_ready(action=original, step=0)

        np.testing.assert_array_equal(result, np.ones(3))
        cb2.on_action_ready.assert_called_once()
        passed = cb2.on_action_ready.call_args[1]["action"]
        np.testing.assert_array_equal(passed, np.ones(3))

    def test_callback_exception_isolated(self) -> None:
        bad_cb = MagicMock()
        bad_cb.on_tick.side_effect = RuntimeError("oops")
        good_cb = MagicMock()
        bus = _CallbackBus([bad_cb, good_cb])
        bus.emit_tick(self._make_tick_event())
        good_cb.on_tick.assert_called_once()

    def test_invoke_on_action_ready_propagates_exception(self) -> None:
        bad_cb = MagicMock()
        bad_cb.on_action_ready.side_effect = RuntimeError("oops")
        good_cb = MagicMock()

        bus = _CallbackBus([bad_cb, good_cb])
        with pytest.raises(RuntimeError, match="oops"):
            bus.invoke_on_action_ready(action=np.zeros(3), step=0)

        good_cb.on_action_ready.assert_not_called()

    def test_close_calls_close_on_callbacks(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])
        bus.close()
        cb.close.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])

        bus.close()
        bus.close()

        cb.close.assert_called_once()

    def test_close_attempts_all_callbacks_before_propagating_base_exception(self) -> None:
        interrupted = MagicMock()
        interrupted.close.side_effect = KeyboardInterrupt
        remaining = MagicMock()
        bus = _CallbackBus([interrupted, remaining])

        with pytest.raises(KeyboardInterrupt):
            bus.close()

        remaining.close.assert_called_once()

    def test_concurrent_close_calls_callbacks_once(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        cb = MagicMock()

        def close() -> None:
            entered.set()
            assert release.wait(timeout=5.0)

        cb.close.side_effect = close
        bus = _CallbackBus([cb])
        threads = [threading.Thread(target=bus.close) for _ in range(2)]
        for thread in threads:
            thread.start()
        assert entered.wait(timeout=5.0)
        release.set()
        for thread in threads:
            thread.join(timeout=5.0)

        cb.close.assert_called_once()

    def test_missing_methods_skipped(self) -> None:
        class MinimalCallback:
            pass

        bus = _CallbackBus([MinimalCallback()])
        bus.emit_tick(self._make_tick_event())
        bus.emit_lifecycle(self._make_lifecycle_event())


class TestConsoleCallback:
    def test_throttles_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        cb = ConsoleCallback(throttle_steps=5)
        for i in range(10):
            cb.on_tick(
                TickEvent(
                    session_id="t",
                    step=i,
                    timestamp=0.0,
                    robot_state=FakeRobotObservation(joint_positions=np.zeros(3)),
                    camera_frames={},
                    action_sent=np.zeros(3),
                    loop_duration_s=0.03,
                    sleep_time_s=0.003,
                    stale_obs=False,
                )
            )
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().split("\n") if l]
        assert len(lines) == 2


class TestJsonlCallback:
    def test_writes_events(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        cb = JsonlCallback(path)

        cb.on_tick(
            TickEvent(
                session_id="s1",
                step=0,
                timestamp=1.0,
                robot_state=FakeRobotObservation(joint_positions=np.array([0.1, 0.2])),
                camera_frames={},
                action_sent=np.array([0.3, 0.4]),
                loop_duration_s=0.03,
                sleep_time_s=0.003,
                stale_obs=False,
            )
        )
        cb.on_lifecycle(
            LifecycleEvent(
                session_id="s1",
                timestamp=1.0,
                event="start",
                metadata={"fps": 30},
            )
        )
        cb.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        tick_record = json.loads(lines[0])
        assert tick_record["type"] == "tick"
        assert tick_record["step"] == 0
        lifecycle_record = json.loads(lines[1])
        assert lifecycle_record["type"] == "lifecycle"
        assert lifecycle_record["event"] == "start"


class TestAsyncCallback:
    def test_dispatches_events_asynchronously(self) -> None:
        inner = MagicMock(spec=["on_tick", "on_inference", "on_lifecycle", "close"])
        called = threading.Event()
        inner.on_lifecycle.side_effect = lambda e: called.set()
        cb = AsyncCallback(inner, max_queue=64)
        event = LifecycleEvent(session_id="t", timestamp=0.0, event="start", metadata={})
        cb.on_lifecycle(event)
        assert called.wait(timeout=2.0), "on_lifecycle not called within timeout"
        inner.on_lifecycle.assert_called_once_with(event)
        cb.close()

    def test_close_joins_thread(self) -> None:
        inner = MagicMock(spec=["on_tick", "on_inference", "on_lifecycle", "close"])
        cb = AsyncCallback(inner)
        cb.close()
        assert not cb._thread.is_alive()
        inner.close.assert_called_once()

    def test_concurrent_close_closes_inner_once(self) -> None:
        inner = MagicMock(spec=["close"])
        cb = AsyncCallback(inner)
        threads = [threading.Thread(target=cb.close) for _ in range(2)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        inner.close.assert_called_once()


class _BlockingCallback:
    """A callback whose hooks block until released (bounded, so a failing test cannot hang)."""

    run_in_background = True

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.ticks: list[int] = []
        self.lifecycle: list[str] = []
        self.closed_on: str | None = None

    def on_tick(self, event: TickEvent) -> None:
        self.entered.set()
        self.release.wait(5.0)
        self.ticks.append(event.step)

    def on_lifecycle(self, event: LifecycleEvent) -> None:
        self.lifecycle.append(event.event)

    def close(self) -> None:
        self.closed_on = threading.current_thread().name


def _lifecycle(event: str) -> LifecycleEvent:
    return LifecycleEvent(session_id="test", timestamp=0.0, event=event, metadata={})


def _workers() -> int:
    return sum(t.name == "AsyncCallbackWorker" and t.is_alive() for t in threading.enumerate())


class TestBackgroundCallbacks:
    def _tick(self, step: int) -> TickEvent:
        return TestCallbackBus()._make_tick_event(step)

    def test_a_blocking_background_callback_never_blocks_emit(self) -> None:
        import time

        inner = _BlockingCallback()
        bus = _CallbackBus([inner])
        bus.start_background()
        wrapped = bus._callbacks[0]
        assert isinstance(wrapped, AsyncCallback)
        worst = 0.0
        for step in range(500):
            started = time.perf_counter()
            bus.emit_tick(self._tick(step))
            worst = max(worst, time.perf_counter() - started)
            assert len(wrapped._queue) <= 16

        assert worst < 0.005, f"worst emit_tick took {worst * 1000:.2f} ms"
        inner.release.set()
        bus.close()
        wrapped._thread.join(2.0)
        assert inner.ticks[-1] == 499  # the newest events were kept, the oldest dropped
        assert len(inner.ticks) <= 18  # in flight + 16 queued + a flush race at most

    def test_shutdown_with_the_callback_still_blocked_is_bounded(self) -> None:
        import time

        inner = _BlockingCallback()
        bus = _CallbackBus([inner])
        bus.start_background()
        for step in range(50):
            bus.emit_tick(self._tick(step))
        assert inner.entered.wait(1.0)
        started = time.perf_counter()
        bus.flush()  # what RobotRuntime.run() does at the end
        bus.close()  # what RobotRuntime.disconnect() does after disconnecting the robot
        took = time.perf_counter() - started

        assert took < 2.0, f"flush and close took {took:.2f}s"
        assert inner.closed_on is None  # still busy: its close waits for the worker
        inner.release.set()
        bus._callbacks[0]._thread.join(2.0)
        assert inner.closed_on == "AsyncCallbackWorker"

    def test_plain_callbacks_stay_synchronous(self) -> None:
        plain = MagicMock(spec=["on_tick"])
        marked_false = MagicMock(spec=["on_tick", "run_in_background"])
        marked_false.run_in_background = False
        mock_with_everything = MagicMock()  # its auto-created attribute is not the marker
        bus = _CallbackBus([plain, marked_false, mock_with_everything])
        bus.start_background()

        assert bus._callbacks == [plain, marked_false, mock_with_everything]
        bus.emit_tick(self._tick(0))
        plain.on_tick.assert_called_once()

    def test_a_marked_callback_with_action_hooks_stays_synchronous(self, caplog: pytest.LogCaptureFixture) -> None:
        class _WithActionHook(_BlockingCallback):
            def on_action_ready(self, *, action: np.ndarray, step: int) -> np.ndarray:  # noqa: ARG002
                return action

        cb = _WithActionHook()
        bus = _CallbackBus([cb])
        with caplog.at_level("WARNING"):
            bus.start_background()

        assert bus._callbacks == [cb]
        assert "defines action hooks; running it synchronously" in caplog.text
        assert bus.invoke_on_action_ready(action=np.ones(2), step=0).tolist() == [1.0, 1.0]
        bus.close()

    def test_no_worker_starts_before_a_run(self) -> None:
        from physicalai.runtime import RobotRuntime
        from tests.unit.runtime.conftest import FakeActionSource

        before = _workers()
        robot = MagicMock()
        robot.connect.side_effect = ConnectionError("robot unplugged")
        runtime = RobotRuntime(robot=robot, action_source=FakeActionSource(), fps=30.0, callbacks=[_BlockingCallback()])
        with pytest.raises(ConnectionError):
            runtime.connect()

        assert _workers() == before  # a failed connect leaves no thread behind

    def test_a_failed_wrap_stops_the_workers_already_started(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import physicalai.runtime._callback_bus as bus_module

        real = bus_module.AsyncCallback
        made: list[AsyncCallback] = []

        def flaky(inner: object, **kwargs: Any) -> AsyncCallback:
            if made:
                raise RuntimeError("second wrap fails")
            made.append(real(inner, **kwargs))
            return made[-1]

        monkeypatch.setattr(bus_module, "AsyncCallback", flaky)
        first, second = _BlockingCallback(), _BlockingCallback()
        bus = _CallbackBus([first, second])
        with pytest.raises(RuntimeError, match="second wrap fails"):
            bus.start_background()
        made[0]._thread.join(2.0)

        assert not made[0]._thread.is_alive()
        assert first.closed_on is None  # the user's callback is not closed by the rollback
        bus.close()
        assert first.closed_on == threading.current_thread().name

    def test_session_boundaries_are_never_dropped(self) -> None:
        inner = _BlockingCallback()
        cb = AsyncCallback(inner, max_queue=4, flush_timeout_s=0.5)
        cb.on_tick(self._tick(0))
        assert inner.entered.wait(1.0)  # the worker is stuck in tick 0
        cb.on_lifecycle(_lifecycle("shutdown"))
        cb.on_lifecycle(_lifecycle("start"))
        for step in range(1, 50):
            cb.on_tick(self._tick(step))
            cb.on_lifecycle(_lifecycle("send_error"))  # telemetry, may be dropped
        inner.release.set()
        cb.flush()
        cb.close()

        assert inner.lifecycle[:2] == ["shutdown", "start"]
        assert inner.ticks[-1] == 49

    def test_an_evicted_event_is_released_outside_the_locks(self) -> None:
        held: list[bool] = []
        inner = _BlockingCallback()
        cb = AsyncCallback(inner, max_queue=1, flush_timeout_s=0.5)

        class _Frame:
            data = np.zeros(1)

            def __del__(self) -> None:
                held.append(cb._queue_lock.locked() or cb._close_lock.locked())

        cb.on_tick(self._tick(0))
        assert inner.entered.wait(1.0)
        event = dataclasses.replace(self._tick(1), camera_frames={"top": cast("Frame", _Frame())})
        cb.on_tick(event)
        del event
        cb.on_tick(self._tick(2))  # evicts tick 1 and its frame

        assert held == [False]
        inner.release.set()
        cb.close()

    def test_async_callback_flush_timeout_is_configurable(self) -> None:
        import time

        inner = _BlockingCallback()
        cb = AsyncCallback(inner, max_queue=4, flush_timeout_s=0.2)
        cb.on_tick(self._tick(0))
        started = time.perf_counter()
        cb.flush()
        took = time.perf_counter() - started
        inner.release.set()
        cb.close()

        assert 0.15 < took < 0.6, f"flush waited {took:.2f}s"

    def test_close_drops_what_a_stuck_inner_has_not_reached(self) -> None:
        inner = _BlockingCallback()
        cb = AsyncCallback(inner, max_queue=16, flush_timeout_s=0.2)
        for step in range(10):
            cb.on_tick(self._tick(step))
        cb.close()  # the inner is stuck in tick 0
        assert inner.closed_on is None  # not closed under a hook that is still running
        inner.release.set()
        cb._thread.join(2.0)

        assert not cb._thread.is_alive()
        assert inner.ticks == [0]  # it finished the event it was in, nothing queued after close
        assert inner.closed_on == "AsyncCallbackWorker"
