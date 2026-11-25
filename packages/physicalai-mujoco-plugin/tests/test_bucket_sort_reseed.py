# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""r16: a reset with a fixed seed repeats the demonstrator's episode, not only the scene. Before, the demonstrator drew
each episode's parameters from its own stream, made once per scene from the robot's constructor seed and never
reseeded, so an episode's DemoParams depended on how many episodes ran before it in the process (real2sim r15: seed
9107 alone and as the 7th of a batch got different speeds and grasp offsets). Needs the bucket-v2 kit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("REAL2SIM_ROOT", Path.home() / "real2sim-data/bucket-v2")).expanduser()
KIT = ROOT / "kit/real2sim.json"
V2 = Path(__file__).resolve().parents[1] / "scripts/real2sim/v2"
pytestmark = pytest.mark.skipif(not KIT.exists(), reason="needs the bucket-v2 kit")


@pytest.fixture(scope="module")
def bucketsim():
    os.environ.setdefault("REAL2SIM_ROOT", str(ROOT)); os.environ.setdefault("KIT", str(KIT)); os.environ.setdefault("PHYSICALAI_PLAN_PROCESS", "0")
    sys.path.insert(0, str(V2))
    import bucketsim_v2 as b

    return b


def _episode_params(b, robot, seed: int):
    """Reset to *seed* (the cpu into the red bucket), let the demonstrator start its episode, return its parameters."""
    m = robot._sim.model
    robot._automation.set_autopilot("off"); b.reset(robot, seed, b.type_body(m, "cpu"), "red"); robot._automation.set_autopilot("drive")
    demo = robot._automation.autopilot.demonstrator
    for _ in range(3):
        b.tick(robot, demo)
    return demo._episode.params, robot._sim.on_reset.instruction.text


def test_episode_params_repeat_alone_and_in_a_batch(bucketsim) -> None:
    b = bucketsim
    alone = _episode_params(b, b.make_robot(9107), 9107)
    batch = b.make_robot(9101)
    for seed in range(9101, 9107):   # draws #1-#6 of a batch, then 9107 as draw #7
        _episode_params(b, batch, seed)
    seventh = _episode_params(b, batch, 9107)
    again = _episode_params(b, batch, 9107)
    assert seventh == alone
    assert again == alone


def test_different_seeds_give_different_params(bucketsim) -> None:
    b = bucketsim
    robot = b.make_robot(9101)
    assert _episode_params(b, robot, 9101)[0] != _episode_params(b, robot, 9102)[0]
