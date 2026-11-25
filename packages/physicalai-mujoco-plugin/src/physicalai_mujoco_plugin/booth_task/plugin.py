# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""What the plugin needs to run the booth task as a registered scene with an autopilot.

- :func:`booth_reset`: the scene reset (``POST /reset``, connect): both arms at SLEEP, the box
  standing in its slot and the cover at its start, the free objects drawn within RUBRIC D7's start
  ranges from the simulation's generator;
- :class:`BoothAutopilot`: the demonstrator behind the plugin's autopilot (``drive`` writes the
  targets, ``leader`` only publishes them for Studio to send back as actions);
- :func:`is_booth_model`: whether a loaded model is this scene.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math

import mujoco
import numpy as np

from physicalai_mujoco_plugin.booth_task.demonstrator import BoothDemonstrator
from physicalai_mujoco_plugin.booth_task.scene import place_start


def is_booth_model(model: object) -> bool:
    """Whether *model* is the booth task scene: it has the bin, the bracket, the box and the cover.

    Returns:
        True when every body in ``BOOTH_BODIES`` exists.
    """
    try:
        return all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0 for name in BOOTH_BODIES)  # type: ignore[arg-type]
    except TypeError:
        return False


BOOTH_BODIES = ("bin", "bracket", "box", "cover")


def booth_reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """Start state: arms asleep, box upright in its slot, cover at its spot, both drawn within D7's ranges."""
    for name in ("box", "cover"):
        body = model.body(name).id
        adr = int(model.jnt_qposadr[model.body_jntadr[body]])
        dof = int(model.jnt_dofadr[model.body_jntadr[body]])
        data.qpos[adr : adr + 7] = model.qpos0[adr : adr + 7]
        data.qvel[dof : dof + 6] = 0.0
    _draw_objects(model, data, rng)
    place_start(model, data)


def _draw_objects(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """Cover +-20 mm and +-10 deg; the box +-2 mm and +-2 deg within its slot's play."""
    from physicalai_mujoco_plugin.booth_task.scene import BOX_LENGTH, slot_play  # noqa: PLC0415

    cover = int(model.jnt_qposadr[model.body_jntadr[model.body("cover").id]])
    data.qpos[cover : cover + 2] += rng.uniform(-0.02, 0.02, 2)
    _turn(data, cover, float(rng.uniform(-math.radians(10.0), math.radians(10.0))))
    box = int(model.jnt_qposadr[model.body_jntadr[model.body("box").id]])
    bin_quat = model.body_quat[model.body("bin").id]
    yaw = math.atan2(
        2 * (bin_quat[0] * bin_quat[3] + bin_quat[1] * bin_quat[2]), 1 - 2 * (bin_quat[2] ** 2 + bin_quat[3] ** 2)
    )
    play = slot_play(model)
    along, across = float(rng.uniform(-0.002, 0.002)), float(np.clip(rng.uniform(-0.002, 0.002), -play, play))
    data.qpos[box : box + 2] += [
        math.cos(yaw) * along - math.sin(yaw) * across,
        math.sin(yaw) * along + math.cos(yaw) * across,
    ]
    turn_play = max(play - abs(across), 0.0) / (BOX_LENGTH / 2)
    _turn(data, box, float(np.clip(rng.uniform(-math.radians(2.0), math.radians(2.0)), -turn_play, turn_play)))


def _turn(data: mujoco.MjData, adr: int, yaw: float) -> None:
    quat = np.zeros(4)
    mujoco.mju_mulQuat(
        quat, np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]), data.qpos[adr + 3 : adr + 7].copy()
    )
    data.qpos[adr + 3 : adr + 7] = quat


class BoothAutopilot:
    """The booth demonstrator as the plugin's autopilot: one ``step`` per control tick."""

    def __init__(self, model: mujoco.MjModel) -> None:
        """Bind to the loaded booth model."""
        self._model = model
        self._demo: BoothDemonstrator | None = None
        self._pending_reset = True

    def reset(self) -> None:
        """Restart the script from the current state at the next tick (call after a scene reset)."""
        self._pending_reset = True

    @property
    def phase(self) -> str:
        """The TASK_SPEC step running now."""
        return self._demo.phase if self._demo is not None else "idle"

    @property
    def target(self) -> None:
        """No per-item target (the conveyor autopilot reports the item it picks)."""
        return

    def step(self, data: mujoco.MjData, dt: float, *, apply: bool = True) -> np.ndarray:
        """Advance one tick; write the targets to ``ctrl`` when *apply*.

        Returns:
            The 14 targets in model units, left arm then right arm.
        """
        if self._pending_reset or self._demo is None:
            self._demo = BoothDemonstrator(self._model, period=dt)
            self._demo.reset(data)
            self._pending_reset = False
        targets = self._demo.step(data, dt)
        if apply:
            self._demo.apply(data, targets)
        return targets


__all__ = ["BoothAutopilot", "booth_reset", "is_booth_model"]
