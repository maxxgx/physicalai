# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Metric scale and robot placement from an ArUco marker seen in the capture photos.

A phone capture has no scale, and nothing in it says where the robot stands. One printed marker
of known size on the table fixes both: its corners, detected in every photo and triangulated with
the COLMAP poses, give metres per capture unit (side length) and a frame on the table (its centre
and edges). The robot's base is then placed at a measured offset from the marker, in the marker's
own frame, instead of being guessed.

Marker frame: origin at the centre, ``x`` from the top-left to the top-right corner, ``y`` from the
bottom-left to the top-left corner (up in the printed image), ``z`` out of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.real2sim.colmap import SparseModel
    from physicalai_mujoco_plugin.real2sim.geometry import Similarity

MIN_VIEWS = 2
"""A corner seen in fewer photos cannot be triangulated."""


@dataclass(frozen=True)
class Marker:
    """One marker triangulated in the capture frame."""

    marker_id: int
    corners: np.ndarray
    """``(4, 3)``: top-left, top-right, bottom-right, bottom-left, in capture coordinates."""
    views: int
    """Photos the marker was detected in."""
    residual_px: float
    """Mean reprojection error of the triangulated corners, in pixels."""

    @property
    def side(self) -> float:
        """Mean edge length in capture units."""
        return float(np.mean([np.linalg.norm(self.corners[i] - self.corners[(i + 1) % 4]) for i in range(4)]))

    @property
    def center(self) -> np.ndarray:
        """Centre in capture coordinates."""
        return self.corners.mean(axis=0)

    def axes(self) -> np.ndarray:
        """Columns: the marker's x, y and z axes in capture coordinates (orthonormal).

        Returns:
            The 3x3 rotation.
        """
        x = (self.corners[1] - self.corners[0]) + (self.corners[2] - self.corners[3])
        y = (self.corners[0] - self.corners[3]) + (self.corners[1] - self.corners[2])
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
        z /= np.linalg.norm(z)
        return np.column_stack([x, np.cross(z, x), z])


def detect(
    images_dir: Path, model: SparseModel, dictionary: str = "DICT_4X4_50"
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Detect ArUco markers in every registered photo.

    Returns:
        Per marker id: ``(image_id, corners (4, 2) in pixels)`` for each photo it was found in.
    """
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # scale is only as good as the corners
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), parameters)
    found: dict[int, list[tuple[int, np.ndarray]]] = {}
    for image in model.images.values():
        photo = cv2.imread(str(Path(images_dir) / image.name), cv2.IMREAD_GRAYSCALE)
        if photo is None:
            continue
        corners, ids, _ = detector.detectMarkers(photo)
        if ids is None:
            continue
        for marker_id, quad in zip(ids.ravel(), corners, strict=True):
            found.setdefault(int(marker_id), []).append((image.image_id, quad.reshape(4, 2).astype(np.float64)))
    return found


def _projection(model: SparseModel, image_id: int) -> np.ndarray:
    image = model.images[image_id]
    camera = model.cameras[image.camera_id]
    intrinsics = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]])
    return intrinsics @ np.column_stack([image.rotation, image.translation])


def triangulate(model: SparseModel, observations: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, float]:
    """Linear (DLT) triangulation of the four corners of one marker over all its photos.

    Pixel coordinates must match ``model.cameras`` (scale the cameras to the photos first). The
    pixel centre convention follows OpenCV's detector: a corner on pixel ``(u, v)`` is at
    ``u + 0.5, v + 0.5`` in the pinhole model.

    Returns:
        The corners ``(4, 3)`` in capture coordinates and the mean reprojection error in pixels.

    Raises:
        ValueError: If the marker was seen in fewer than :data:`MIN_VIEWS` photos.
    """
    if len(observations) < MIN_VIEWS:
        msg = f"a marker needs at least {MIN_VIEWS} photos, got {len(observations)}"
        raise ValueError(msg)
    projections = [_projection(model, image_id) for image_id, _ in observations]
    corners = np.empty((4, 3))
    errors = []
    for corner in range(4):
        rows = []
        for projection, (_, quad) in zip(projections, observations, strict=True):
            u, v = quad[corner] + 0.5
            rows += [u * projection[2] - projection[0], v * projection[2] - projection[1]]
        point = np.linalg.svd(np.asarray(rows))[2][-1]
        corners[corner] = point[:3] / point[3]
        for projection, (_, quad) in zip(projections, observations, strict=True):
            image_point = projection @ np.append(corners[corner], 1.0)
            errors.append(np.linalg.norm(image_point[:2] / image_point[2] - (quad[corner] + 0.5)))
    return corners, float(np.mean(errors))


def find_marker(images_dir: Path, model: SparseModel, marker_id: int, dictionary: str = "DICT_4X4_50") -> Marker:
    """Detect and triangulate one marker.

    Returns:
        The marker in capture coordinates.

    Raises:
        ValueError: If the marker is not found in enough photos.
    """
    observations = detect(images_dir, model, dictionary).get(marker_id, [])
    if len(observations) < MIN_VIEWS:
        msg = f"marker {marker_id} found in {len(observations)} photos; it needs at least {MIN_VIEWS}"
        raise ValueError(msg)
    corners, residual = triangulate(model, observations)
    return Marker(marker_id, corners, len(observations), residual)


def mount_from_marker(
    marker: Marker, transform: Similarity, offset: tuple[float, float], yaw_deg: float
) -> tuple[tuple[float, float, float], float]:
    """``robot_mount`` on the table at ``offset`` (metres, marker x/y) from the marker centre.

    Args:
        marker: The triangulated marker.
        transform: Capture frame to table frame.
        offset: The robot base's position in the marker frame, measured on the bench.
        yaw_deg: The robot's forward direction relative to the marker's ``x`` axis.

    Returns:
        The mount position on the table top (``z = 0``) and its yaw in degrees, in the table frame.
    """
    center = transform.apply(marker.center[None])[0]
    x_axis = transform.apply_rotation(marker.axes())[:, 0]
    marker_yaw = np.arctan2(x_axis[1], x_axis[0])
    cos, sin = np.cos(marker_yaw), np.sin(marker_yaw)
    position = center[:2] + np.array([cos * offset[0] - sin * offset[1], sin * offset[0] + cos * offset[1]])
    return (float(position[0]), float(position[1]), 0.0), float(np.degrees(marker_yaw) + yaw_deg)
