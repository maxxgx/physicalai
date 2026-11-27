# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RerunCallback against the real rerun-sdk and a viewer that accepts but never reads.

Runs in a subprocess with a hard timeout, so a native call that never returns fails the
test instead of hanging the suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("rerun")

_SCRIPT = r"""
import json, os, socket, sys, threading, time
import numpy as np
from physicalai.capture.frame import Frame
from physicalai.runtime import RerunCallback
from physicalai.runtime.events import LifecycleEvent, TickEvent

# A "viewer" that accepts TCP connections and never reads: the gRPC stream stalls
# with data in flight, so the SDK's queues fill (the SPS failure).
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen(16)
port = listener.getsockname()[1]
held = []
threading.Thread(target=lambda: [held.append(listener.accept()) for _ in iter(int, 1)], daemon=True).start()

class State:
    joint_positions = np.zeros(6)

# Three 1080p cameras: about 18 MiB per tick, so without the guard the SDK's 50 MiB
# input budget is full within three ticks.
rng = np.random.default_rng(0)
frames = {f"cam{i}": Frame(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8), 0.0, 0) for i in range(3)}
cb = RerunCallback(mode="connect", connect_addr=f"127.0.0.1:{port}", image_decimation=1)
cb.on_lifecycle(LifecycleEvent("s", time.time(), "start", {"fps": 30, "joint_names": [f"j{i}" for i in range(6)]}))
costs = []
for step in range(int(sys.argv[1])):
    t = time.perf_counter()
    cb.on_tick(TickEvent(
        session_id="s", step=step, timestamp=time.time(), robot_state=State(), camera_frames=frames,
        action_sent=np.zeros(6), loop_duration_s=0.0, sleep_time_s=0.0, stale_obs=False, source_updated=True,
    ))
    costs.append(time.perf_counter() - t)
    time.sleep(max(0.0, 1 / 30 - costs[-1]))
print(json.dumps({"ticks": len(costs), "max_tick_s": max(costs), "stalled": cb._sink_stalled}), flush=True)
os._exit(0)  # rerun-sdk's own exit flush waits forever on a stalled stream (not this callback)
"""


@pytest.mark.integration
@pytest.mark.slow
def test_a_viewer_that_never_reads_cannot_stall_on_tick() -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT, "300"], env=env, capture_output=True, text=True, timeout=20, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result["ticks"] == 300
    assert result["stalled"], "the sink never backed up, so this run did not test the guard"
    assert result["max_tick_s"] < 0.2
