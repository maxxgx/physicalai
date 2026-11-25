# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rigid transforms for task-space targets: a pose is a position and a 3x3 rotation, in the booth frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

MIN_ROTATION_ANGLE = 1e-12
"""Radians: below this a rotation vector counts as no rotation."""


@dataclass(frozen=True)
class Pose:
    """A rigid transform: ``world = rot @ local + pos``."""

    pos: np.ndarray
    rot: np.ndarray

    def __matmul__(self, other: Pose) -> Pose:
        """Compose: ``(self @ other)`` maps *other*'s local frame through *self*.

        Returns:
            The composed transform.
        """
        return Pose(self.pos + self.rot @ other.pos, self.rot @ other.rot)

    def inv(self) -> Pose:
        """The inverse transform.

        Returns:
            The inverse pose.
        """
        rt = self.rot.T
        return Pose(-rt @ self.pos, rt)

    def apply(self, point: np.ndarray) -> np.ndarray:
        """Map a local point to the world.

        Returns:
            The point in the world frame.
        """
        return self.pos + self.rot @ np.asarray(point, dtype=float)

    def quat(self) -> np.ndarray:
        """The rotation as a unit quaternion (w, x, y, z).

        Returns:
            The quaternion, a length-4 array.
        """
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(self.rot).ravel())
        return q

    def moved(self, delta: np.ndarray) -> Pose:
        """The same rotation, translated by a world vector.

        Returns:
            The translated pose.
        """
        return Pose(self.pos + np.asarray(delta, dtype=float), self.rot)


def pose(pos: np.ndarray | list[float], rot: np.ndarray | None = None) -> Pose:
    """A pose from a position and an optional rotation (identity).

    Returns:
        The pose (copies of the inputs).
    """
    return Pose(np.asarray(pos, dtype=float).copy(), np.eye(3) if rot is None else np.asarray(rot, dtype=float).copy())


def tool_rotation(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """Tool-site rotation whose x axis is *approach* and y axis is *closing* (orthogonalised).

    Returns:
        The 3x3 rotation matrix.
    """
    x = np.array(approach, dtype=float)  # a copy: normalised in place below
    x /= np.linalg.norm(x)
    y = np.asarray(closing, dtype=float) - x * float(np.dot(x, closing))
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def rot_z(angle: float) -> np.ndarray:
    """Rotation about z.

    Returns:
        The 3x3 rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(angle: float) -> np.ndarray:
    """Rotation about x.

    Returns:
        The 3x3 rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Rotation about y.

    Returns:
        The 3x3 rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def yaw_of(rot: np.ndarray, axis: int = 0) -> float:
    """Heading (rad) of one of the rotation's axes projected on the table.

    Returns:
        The heading in radians.
    """
    v = rot[:, axis]
    return math.atan2(v[1], v[0])


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) of the rotation taking *a* to *b*.

    Returns:
        The angle in radians, in ``[0, pi]``.
    """
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos)))


def rotation_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """World-frame rotation vector ``w`` with ``exp(w) @ a == b``.

    Returns:
        The rotation vector (axis times angle in radians).
    """
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(b @ a.T).ravel())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def rotate_by_vector(rot: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``exp(w) @ rot`` for a world-frame rotation vector *w*.

    Returns:
        The rotated 3x3 matrix.
    """
    angle = float(np.linalg.norm(w))
    if angle < MIN_ROTATION_ANGLE:
        return rot.copy()
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, w / angle, angle)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3) @ rot


def min_jerk(s: float) -> float:
    """Minimum-jerk time scaling of a fraction in [0, 1].

    Returns:
        The scaled fraction, in ``[0, 1]``.
    """
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def interpolate(a: Pose, b: Pose, s: float) -> Pose:
    """Straight line in position and slerp in rotation, at fraction *s*.

    Returns:
        The interpolated pose.
    """
    w = rotation_vector(a.rot, b.rot)
    return Pose(a.pos + s * (b.pos - a.pos), rotate_by_vector(a.rot, s * w))


def body_pose(data: mujoco.MjData, body: int) -> Pose:
    """A body's world pose.

    Returns:
        The body's pose (copies of the data's arrays).
    """
    return Pose(data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy())


def site_pose(data: mujoco.MjData, site: int) -> Pose:
    """A site's world pose.

    Returns:
        The site's pose (copies of the data's arrays).
    """
    return Pose(data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy())
