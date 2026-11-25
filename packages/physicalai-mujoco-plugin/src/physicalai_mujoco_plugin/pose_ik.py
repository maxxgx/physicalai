# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Full-pose inverse kinematics for one arm of a composed scene.

Damped least squares on a site (the tool centre point): position and orientation together,
weighted, with joint limits as an active set and a light pull toward a rest posture in the
null space. Runs on a private ``MjData``, so the simulation is never touched.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

MAX_STEP = 0.2
"""Largest joint change (rad) per iteration: big jumps can cross into another IK branch."""


@dataclass(frozen=True)
class IKResult:
    """A solution and how far it is from the target."""

    q: np.ndarray
    position_error: float
    """Metres."""
    rotation_error: float
    """Radians."""

    def reached(self, position_tol: float = 2e-3, rotation_tol: float = 0.03) -> bool:
        """Whether both errors are within tolerance.

        Returns:
            True when the target pose is reached.
        """
        return self.position_error <= position_tol and self.rotation_error <= rotation_tol


class PoseIK:
    """Damped least-squares IK for a site's full pose over a named set of joints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joints: list[str],
        site: str,
        *,
        rest: np.ndarray | None = None,
        rotation_weight: float = 0.3,
        damping: float = 1e-3,
    ) -> None:
        """Resolve the joints and the site.

        Args:
            model: The compiled scene.
            joints: The arm's hinge joints, base to wrist.
            site: The tool centre site.
            rest: Rest posture the null space leans toward (defaults to the joint range middle).
            rotation_weight: Metres per radian when position and rotation errors are traded.
            damping: Levenberg-Marquardt damping.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joints]
        if min(ids) < 0:
            msg = f"unknown joint in {joints}"
            raise ValueError(msg)
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in ids])
        self.lower = np.array([model.jnt_range[j][0] for j in ids])
        self.upper = np.array([model.jnt_range[j][1] for j in ids])
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site < 0:
            msg = f"unknown site {site!r}"
            raise ValueError(msg)
        self.rest = (self.lower + self.upper) / 2 if rest is None else np.asarray(rest, dtype=float)
        self.rotation_weight = rotation_weight
        self.damping = damping
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def forward(self, q: np.ndarray, qpos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The site pose for joint angles ``q`` (the rest of the state from ``qpos``).

        Returns:
            The site position and its 3x3 rotation, in the world frame.
        """
        if qpos is not None:
            self.data.qpos[:] = qpos
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def solve(  # noqa: PLR0914 - one numerical routine
        self,
        q0: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray | None,
        *,
        qpos: np.ndarray | None = None,
        iters: int = 60,
        posture_gain: float = 0.02,
    ) -> IKResult:
        """Joint angles that put the site at ``position`` with orientation ``rotation``.

        Args:
            q0: Start joint angles.
            position: Target site position (world).
            rotation: Target site rotation (world, 3x3); ``None`` solves for position only.
            qpos: Full state for the bodies the arm does not own (a moved base, say).
            iters: Iteration cap.
            posture_gain: Null-space pull toward :attr:`rest`.

        Returns:
            The solution and its remaining errors.
        """
        model, data = self.model, self.data
        if qpos is not None:
            data.qpos[:] = qpos
        q = np.clip(np.asarray(q0, dtype=float).copy(), self.lower, self.upper)
        quat, e_rot = np.zeros(4), np.zeros(3)
        e_pos = np.zeros(3)
        rot_err = 0.0
        n = len(q)
        for _ in range(iters):
            data.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            p = data.site_xpos[self.site]
            r = data.site_xmat[self.site].reshape(3, 3)
            e_pos = position - p
            if rotation is not None:
                mujoco.mju_mat2Quat(quat, (rotation @ r.T).ravel())
                mujoco.mju_quat2Vel(e_rot, quat, 1.0)
                rot_err = float(np.linalg.norm(e_rot))
            if np.linalg.norm(e_pos) < 1e-4 and rot_err < 1e-3:
                break
            mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.site)
            jp = self._jacp[:, self.dof_adr]
            jr = self._jacr[:, self.dof_adr] * self.rotation_weight
            if rotation is None:
                jac, err = jp, e_pos
            else:
                jac, err = np.vstack([jp, jr]), np.concatenate([e_pos, e_rot * self.rotation_weight])
            free = np.ones(n, dtype=bool)
            dq = np.zeros(n)
            for _ in range(n):
                j = jac * free
                inv = j.T @ np.linalg.inv(j @ j.T + self.damping * np.eye(len(err)))
                dq = inv @ err
                null = (np.eye(n) - inv @ j) * free
                dq += null @ (posture_gain * (self.rest - q))
                dq *= free
                blocked = free & (((q >= self.upper - 1e-9) & (dq > 0)) | ((q <= self.lower + 1e-9) & (dq < 0)))
                if not blocked.any():
                    break
                free &= ~blocked
            biggest = float(np.max(np.abs(dq)))
            if biggest > MAX_STEP:
                dq *= MAX_STEP / biggest
            q = np.clip(q + dq, self.lower, self.upper)
        data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        e_pos = position - data.site_xpos[self.site]
        if rotation is not None:
            mujoco.mju_mat2Quat(quat, (rotation @ data.site_xmat[self.site].reshape(3, 3).T).ravel())
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            rot_err = float(np.linalg.norm(e_rot))
        return IKResult(q=q, position_error=float(np.linalg.norm(e_pos)), rotation_error=rot_err)
