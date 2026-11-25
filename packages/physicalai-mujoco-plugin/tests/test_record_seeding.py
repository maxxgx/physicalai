# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""r14 (r13 review): a recording must not repeat one scene. The plugin's seed is fixed by design (every reset repeats
it); the recorder (studio_record_v2.py) therefore sets the seed before every reset: S + k - 1 before try k with --seed S
(varied, and the same run again), none without. This checks, on the bucket-v2 desk, the plugin side of that contract
(a fixed seed repeats the scene; the recorder's next seed changes it; its sequence repeats), and the recorder's
sequence itself. Needs the bucket-v2 kit."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get("REAL2SIM_ROOT", Path.home() / "real2sim-data/bucket-v2")).expanduser()
KIT = ROOT / "kit/real2sim.json"
V2 = Path(__file__).resolve().parents[1] / "scripts/real2sim/v2"
pytestmark = pytest.mark.skipif(not KIT.exists(), reason="needs the bucket-v2 kit")


def _scene(robot) -> tuple:
    task = robot._sim.on_reset
    d = robot._sim.data
    objects = tuple(sorted(task.layout.objects))
    return task.instruction.text, objects, np.round(d.qpos[: min(len(d.qpos), 400)], 4).tobytes()


def test_seeds_per_episode_vary_and_repeat() -> None:
    os.environ.setdefault("REAL2SIM_ROOT", str(ROOT)); os.environ.setdefault("KIT", str(KIT)); os.environ.setdefault("PHYSICALAI_PLAN_PROCESS", "0")
    sys.path.insert(0, str(V2))
    import bucketsim_v2 as b

    robot = b.make_robot(0)
    seq = []
    for k in range(1, 4):   # the recorder's seeds for tries 1-3 with --seed 500
        robot._set_seed(500 + k - 1); robot._reset(); seq.append(_scene(robot))
    assert len({s[2] for s in seq}) == 3, "consecutive episodes must differ"
    robot._set_seed(500); robot._reset(); again = _scene(robot)
    assert again == seq[0], "the same seed repeats its scene (a run repeats exactly)"
    robot._reset(); assert _scene(robot)[2] == seq[0][2], "a fixed seed repeats on every reset (why the recorder sets it)"
    robot._set_seed(None)


def test_recorder_sets_a_seed_per_episode() -> None:
    src = (V2 / "studio_record_v2.py").read_text()
    assert re.search(r"def episode_seed\(k: int\)", src)
    assert 'http("POST", f"{SIM}/seed", {"seed": episode_seed(1)})' in src   # always: clears a seed fixed at start
    assert 'http("POST", f"{SIM}/seed", {"seed": episode_seed(tries + 1)})' in src   # before every next reset
