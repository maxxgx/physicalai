# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One WidowX AI arm driven from task-space targets: IK every tick, then position targets in ``ctrl``.

The real WidowX AI controller tracks joint position targets with its own model-based feed-forward
(gravity and dynamics). In the simulation the Menagerie position servos have none, so the target
written to ``ctrl`` is the IK solution plus the offset that makes the servo follow it: the arm's
inverse dynamics along the commanded motion (gravity, inertia with the motors' armature, Coriolis,
joint damping) divided by the servo gain, plus a slow integral of the remaining joint error (the
held object's weight). That offset is part of the action: the recorded action is exactly what the
servo was told.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math

import mujoco
import numpy as np

from physicalai_mujoco_plugin.booth_task.frames import Pose, site_pose
from physicalai_mujoco_plugin.booth_task.scene import (
    ARM_JOINTS,
    GRIPPER_MAX_TRAVEL,
    GRIPPER_SPEED,
    JOINT_SPEED_LIMIT,
    TCP_SITE,
    arm_ids,
)
from physicalai_mujoco_plugin.pose_ik import IKResult, PoseIK

SPEED_MARGIN = 0.9
"""Commanded joint speeds stay below this fraction of the real limits."""
INTEGRAL_GAIN = 1.5
"""1/s: how fast the integral removes a steady joint error."""
INTEGRAL_LIMIT = 0.12
"""rad: the integral's bound, so contact cannot wind it up."""
IK_ITERS = 25


class ArmServo:
    """IK and the position-target feed-forward for one arm."""

    def __init__(self, model: mujoco.MjModel, side: str, rest: np.ndarray) -> None:
        """Resolve the arm and build its IK on a private ``MjData``."""
        self.model = model
        self.side = side
        self.ids = arm_ids(model, side)
        self.ik = PoseIK(model, [f"{side}_{j}" for j in ARM_JOINTS], f"{side}_{TCP_SITE}", rest=rest)
        self.q_cmd = np.zeros(6)
        self.qd_cmd = np.zeros(6)
        self.qdd_cmd = np.zeros(6)
        self.integral = np.zeros(6)
        self.grip_cmd = 0.0
        self.grip_goal = 0.0
        self.last_ik: IKResult | None = None
        self.worst_ik_error = 0.0
        """Largest IK position error (m) of a task-space target so far."""
        self.speed_clipped = 0
        """Ticks where the IK step exceeded the speed limit and was cut."""

    # --- state -------------------------------------------------------------------------------------

    def sync(self, data: mujoco.MjData) -> None:
        """Take the arm's measured state as the command (start of a script)."""
        self.q_cmd = data.qpos[self.ids.qpos].copy()
        self.qd_cmd[:] = 0.0
        self.qdd_cmd[:] = 0.0
        self.integral[:] = 0.0
        self.grip_cmd = float(data.qpos[self.ids.gripper_qpos[0]])
        self.grip_goal = self.grip_cmd

    def tool_command(self) -> Pose:
        """The tool pose of the commanded joints (forward kinematics, private data).

        Returns:
            The tool pose in the booth frame.
        """
        pos, rot = self.ik.forward(self.q_cmd)
        return Pose(pos, rot)

    def tool_actual(self, data: mujoco.MjData) -> Pose:
        """The tool's measured pose.

        Returns:
            The tool pose in the booth frame.
        """
        return site_pose(data, self.ids.site)

    # --- commands ----------------------------------------------------------------------------------

    def track_pose(self, target: Pose, dt: float, predicted_step: np.ndarray | None = None) -> IKResult:
        """Solve IK for *target* and move the command toward the solution.

        The solve starts from the last command advanced by *predicted_step* (the segment's
        joint-space increment between its start and goal solutions), which keeps the solution on
        the branch the segment's goal was planned on.

        Returns:
            The IK result.
        """
        start = (
            self.q_cmd if predicted_step is None else np.clip(self.q_cmd + predicted_step, self.ik.lower, self.ik.upper)
        )
        result = self.ik.solve(start, target.pos, target.rot, iters=IK_ITERS)
        self.last_ik = result
        self.worst_ik_error = max(self.worst_ik_error, result.position_error)
        self._move_to(result.q, dt)
        return result

    def plan(self, target: Pose) -> IKResult:
        """The best IK solution for a segment's goal: reached, close to the current command, clear of limits.

        Tries the current command and a set of elbow-up postures with different wrist yaws and
        rolls, each with the base turned toward the target.

        Returns:
            The best IK result.
        """
        base = self.ik.forward(np.zeros(6))[0]
        heading = math.atan2(target.pos[1] - base[1], target.pos[0] - base[0])
        seeds = [self.q_cmd.copy()]
        seeds.extend(
            np.array([heading, 1.2, 1.0, 0.3, wrist_yaw, roll])
            for wrist_yaw in (-1.0, 0.0, 1.0)
            for roll in (-2.0, -1.0, 0.0, 1.0, 2.0)
        )
        best, best_score = None, math.inf
        for seed in seeds:
            result = self.ik.solve(np.clip(seed, self.ik.lower, self.ik.upper), target.pos, target.rot, iters=150)
            margin = float(np.min(np.minimum(result.q - self.ik.lower, self.ik.upper - result.q)))
            score = (
                1000.0 * result.position_error
                + 50.0 * result.rotation_error
                + 0.3 * float(np.sum(np.abs(result.q - self.q_cmd)))
                - 2.0 * min(margin, 0.3)
            )
            if score < best_score:
                best, best_score = result, score
        return best  # type: ignore[return-value]

    def track_joints(self, q: np.ndarray, dt: float) -> None:
        """Move the command toward a joint vector (used only for SLEEP <-> READY)."""
        self._move_to(np.clip(q, self.ik.lower, self.ik.upper), dt)

    def hold(self, dt: float) -> None:
        """Keep the command where it is."""
        self.qdd_cmd = -self.qd_cmd / dt
        self.qd_cmd = np.zeros(6)

    def _move_to(self, q: np.ndarray, dt: float) -> None:
        step = q - self.q_cmd
        limit = SPEED_MARGIN * JOINT_SPEED_LIMIT * dt
        if np.any(np.abs(step) > limit):
            self.speed_clipped += 1
            step = np.clip(step, -limit, limit)
        qd = step / dt
        self.qdd_cmd = (qd - self.qd_cmd) / dt
        self.qd_cmd = qd
        self.q_cmd = self.q_cmd + step  # noqa: PLR6104 - a new array: earlier commands may still be referenced

    def set_grip(self, travel: float) -> None:
        """Finger travel goal (m per finger): 0 is fully closed, the real gripper's limit fully open."""
        self.grip_goal = float(np.clip(travel, 0.0, GRIPPER_MAX_TRAVEL))

    # --- output ------------------------------------------------------------------------------------

    def targets(self, data: mujoco.MjData, torque: np.ndarray, dt: float) -> np.ndarray:
        """The seven position targets for this tick: six joints (rad) then the gripper (m).

        Args:
            data: The simulation (read only).
            torque: Inverse-dynamics torque on every dof (:class:`DynamicsModel`).
            dt: The control period.

        Returns:
            The seven targets.
        """
        ids = self.ids
        q = data.qpos[ids.qpos]
        error = self.q_cmd - q
        self.integral = np.clip(self.integral + INTEGRAL_GAIN * error * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        feed_forward = (torque[ids.dof] + ids.damping * self.qd_cmd) / ids.kp
        arm = self.q_cmd + feed_forward + self.integral
        opening = float(data.qpos[ids.gripper_qpos[0]])
        if self.grip_goal > self.grip_cmd and opening > self.grip_cmd:
            # Releasing a grasp: the command starts from where the fingers are, as the real
            # gripper's does, instead of ramping up through the squeeze first.
            self.grip_cmd = opening
        step = GRIPPER_SPEED * dt
        self.grip_cmd += float(np.clip(self.grip_goal - self.grip_cmd, -step, step))
        return np.concatenate([arm, [self.grip_cmd]])


class DynamicsModel:
    """Inverse dynamics of the arms along their commanded motion, on a private ``MjData``."""

    def __init__(self, model: mujoco.MjModel) -> None:
        """Allocate the private data."""
        self.model = model
        self.data = mujoco.MjData(model)
        self.torque = np.zeros(model.nv)

    def update(self, data: mujoco.MjData, arms: list[ArmServo]) -> np.ndarray:
        """Torques that move the arms along their commands from *data*'s measured positions.

        Gravity, inertia (with armature) and Coriolis of the arm bodies; held objects are not
        included (the servo's integral takes their weight).

        Returns:
            The torque on every dof (the instance's buffer, overwritten on each call).
        """
        d = self.data
        d.qpos[:] = data.qpos
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        for arm in arms:
            d.qvel[arm.ids.dof] = arm.qd_cmd
            d.qacc[arm.ids.dof] = arm.qdd_cmd
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        mujoco.mj_comVel(self.model, d)
        mujoco.mj_rne(self.model, d, 1, self.torque)
        self.torque += self.model.dof_armature * d.qacc
        return self.torque
