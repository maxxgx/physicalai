# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scripted, IK-driven, contact-only bimanual demonstrator for the booth task (two WidowX AI arms).

``make_episode(seed, perturb)`` is the reviewer's entry point (RUBRIC D6); the plugin registers the
scene as ``booth_task`` and runs the demonstrator as its autopilot.
"""

from physicalai_mujoco_plugin.booth_task.demonstrator import EPISODE_END, PHASES, BoothDemonstrator
from physicalai_mujoco_plugin.booth_task.episode import CONTROL_PERIOD, make_episode, run_episode

__all__ = ["CONTROL_PERIOD", "EPISODE_END", "PHASES", "BoothDemonstrator", "make_episode", "run_episode"]
