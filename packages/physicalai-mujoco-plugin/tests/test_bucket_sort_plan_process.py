# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Attempt planning in a worker process gives the same plan (and generator state) as planning in-process."""

from __future__ import annotations

import numpy as np
import pytest

from physicalai_mujoco_plugin.bucket_sort_arms import bucket_sort_arms
from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator, DemoParams, SceneSnapshot


@pytest.fixture(scope="module")
def desk():
    from physicalai_mujoco_plugin.robot import MuJoCoRobot

    robot = MuJoCoRobot("rebot_b601", scene="bucket_sort_desk_v2", cameras=[], http_port=0, seed=0)
    robot.connect()
    robot._automation.set_autopilot("off")  # noqa: SLF001
    yield robot
    robot.disconnect()


def _scattered_reset(robot) -> None:
    task = robot._sim.on_reset  # noqa: SLF001
    for seed in range(40, 200):
        robot._set_seed(seed)  # noqa: SLF001
        robot._reset()  # noqa: SLF001
        if not task.layout.pile and len(task.layout.objects) <= 3:
            return
    pytest.skip("no small scattered table drawn")


def test_worker_plans_the_same_attempt_and_draws(desk) -> None:
    _scattered_reset(desk)
    model, data, task = desk._sim.model, desk._sim.data, desk._sim.on_reset  # noqa: SLF001
    arms = bucket_sort_arms("rebot_b601")
    local = BucketSortDemonstrator(model, task, arms=arms, plan_process=False)
    remote = BucketSortDemonstrator(model, task, arms=arms, plan_process=True)
    try:
        snapshot = SceneSnapshot.take(data)
        q = np.array(arms[0].home); start = local._start_state() if local._q is not None else None  # noqa: SLF001
        local._q = q.copy(); remote._q = q.copy()  # noqa: SLF001
        start = local._start_state()  # noqa: SLF001
        params = DemoParams.for_heights(arms[0].heights)
        rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
        a = local.plan_attempt(snapshot, q, start, task.instruction, params, 0.02, rng_a, report=False)
        b = remote.plan_attempt(snapshot, q, start, task.instruction, params, 0.02, rng_b, report=False)
        assert remote._plan_proc is not None and remote._plan_proc.alive  # noqa: SLF001 - it really planned there
        np.testing.assert_array_equal(a.q, b.q)
        np.testing.assert_array_equal(a.grip, b.grip)
        assert (a.ok, a.kind, a.yaw_grasp, a.problems) == (b.ok, b.kind, b.yaw_grasp, b.problems)
        assert rng_a.bit_generator.state == rng_b.bit_generator.state
    finally:
        remote.close()
        local.close()
    assert remote._plan_proc is None  # noqa: SLF001 - closed


def _gone(pid: int, within_s: float) -> bool:
    import os
    import time

    end = time.monotonic() + within_s
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_a_closed_scene_leaves_no_planning_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai_mujoco_plugin.robot import MuJoCoRobot

    monkeypatch.setenv("PHYSICALAI_PLAN_PROCESS", "1")   # as the plugin runs it (the test session plans in-process)
    robot = MuJoCoRobot("rebot_b601", scene="bucket_sort_desk_v2", cameras=[], http_port=0, seed=0)
    robot.connect()
    try:
        demo = robot._automation._bucket_demo  # noqa: SLF001
        assert demo is not None
        worker = demo._plan_process_for_call()  # noqa: SLF001 - started on first use
        assert worker is not None and worker.alive
        pid = worker.pid
    finally:
        robot.disconnect()
    assert _gone(pid, 5.0), "the planning worker outlived its scene"


def test_the_planning_worker_ends_when_its_process_is_terminated(tmp_path) -> None:
    import os
    import signal
    import subprocess
    import sys

    script = tmp_path / "owner.py"
    # spawn re-imports the parent's __main__ in the worker: the guard keeps it from running the script again
    script.write_text(
        "import os, sys, time\n"
        "if __name__ == '__main__':\n"
        "    os.environ['PHYSICALAI_PLAN_PROCESS'] = '1'\n"
        "    from physicalai_mujoco_plugin.robot import MuJoCoRobot\n"
        "    r = MuJoCoRobot('rebot_b601', scene='bucket_sort_desk_v2', cameras=[], http_port=0, seed=0); r.connect()\n"
        "    w = r._automation._bucket_demo._plan_process_for_call()\n"
        "    print(w.pid, flush=True)\n"
        "    time.sleep(120)\n"
    )
    owner = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        pid = int(owner.stdout.readline())
        os.kill(owner.pid, signal.SIGTERM)   # no atexit, no close(): only the worker's own watch can end it
        owner.wait(10)
        assert _gone(pid, 8.0), "the planning worker outlived its terminated owner"
    finally:
        if owner.poll() is None:
            owner.kill()
