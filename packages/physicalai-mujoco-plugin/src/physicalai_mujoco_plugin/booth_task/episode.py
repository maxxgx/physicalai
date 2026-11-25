# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The reviewer's entry point (RUBRIC D6): ``make_episode(seed, perturb) -> (model, data, controller)``.

    from physicalai_mujoco_plugin.booth_task import make_episode, CONTROL_PERIOD
    model, data, controller = make_episode(7, perturb=True)
    steps = round(CONTROL_PERIOD / model.opt.timestep)    # 20
    while data.time < EPISODE_END:
        controller(model, data)                          # writes data.ctrl only
        for _ in range(steps):
            mujoco.mj_step(model, data)

``controller.period`` is the control period it expects (1/30 s, 20 physics steps of 1/600 s).
With ``perturb=True`` the start state and model get RUBRIC D7's draws from ``numpy`` seeded with
*seed* (:class:`~physicalai_mujoco_plugin.booth_task.scene.Perturbation`); with ``perturb=False``
the nominal scene is returned (the seed is unused).
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import mujoco
import numpy as np

from physicalai_mujoco_plugin.booth_task.demonstrator import EPISODE_END, BoothDemonstrator
from physicalai_mujoco_plugin.booth_task.scene import CONTROL_PERIOD, Perturbation, load_model, place_start
from physicalai_mujoco_plugin.booth_task.scene import perturb as apply_perturbation

STEPS_PER_TICK = 20


def make_episode(seed: int, perturb: bool) -> tuple[mujoco.MjModel, mujoco.MjData, BoothDemonstrator]:  # noqa: FBT001 - the rubric's signature
    """A fresh booth episode: model, start state and the demonstrator controller.

    Args:
        seed: Seeds the perturbation draws.
        perturb: Apply RUBRIC D7's perturbations.

    Returns:
        The model (not to be modified), its data at the start state, and the controller, a callable
        ``controller(model, data)`` that only writes ``data.ctrl``; ``controller.period`` is its
        control period in seconds and ``controller.perturbation`` the draw (``None`` unperturbed).
    """
    model = load_model()
    data = mujoco.MjData(model)
    draw = None
    if perturb:
        draw = Perturbation.draw(np.random.default_rng(seed))
        apply_perturbation(model, data, draw)
    else:
        place_start(model, data)
    _settle(model, data)
    controller = BoothDemonstrator(model, period=CONTROL_PERIOD)
    controller.perturbation = draw  # type: ignore[attr-defined]
    controller.reset(data)
    return model, data, controller


SETTLE_S = 0.5


def _settle(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Let the objects come to rest on their supports, then start the clock at zero.

    The objects start a fraction of a millimetre above their supports; the arms hold SLEEP
    (gravity-compensated targets, as the real controller holds it) meanwhile, so the episode
    starts at rest.
    """
    from physicalai_mujoco_plugin.booth_task.scene import SIDES, arm_ids  # noqa: PLC0415

    gravity = np.zeros(model.nv)
    mujoco.mj_forward(model, data)
    mujoco.mj_rne(model, data, 0, gravity)
    for side in SIDES:
        ids = arm_ids(model, side)
        data.ctrl[ids.actuators] = data.qpos[ids.qpos] + gravity[ids.dof] / ids.kp
    while data.time < SETTLE_S:
        mujoco.mj_step(model, data)
    data.time = 0.0


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: BoothDemonstrator,
    *,
    until: float = EPISODE_END,
    on_tick: object | None = None,
) -> None:
    """Run the loop the reviewer runs: controller, then ``STEPS_PER_TICK`` physics steps."""
    steps = round(controller.period / model.opt.timestep)
    while data.time < until - 1e-9:
        controller(model, data)
        for _ in range(steps):
            mujoco.mj_step(model, data)
        if on_tick is not None:
            on_tick(model, data, controller)  # type: ignore[operator]


__all__ = ["CONTROL_PERIOD", "EPISODE_END", "STEPS_PER_TICK", "make_episode", "run_episode"]
