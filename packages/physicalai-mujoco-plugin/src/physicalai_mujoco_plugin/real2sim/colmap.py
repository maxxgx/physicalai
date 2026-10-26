# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Read a COLMAP sparse model (``cameras.bin``, ``images.bin``, ``points3D.bin``).

COLMAP is what turns a phone video into posed photos: every capture route in the real2sim
design (COLMAP, GLOMAP, nerfstudio's ``ns-process-data``) ends in this format, and splat
trainers read it too. Only the binary format is read; it is what every tool writes by default.

COLMAP's camera looks along ``+z`` with ``y`` down, and an image stores the world-to-camera
rotation (``qvec``, w-first) and translation (``tvec``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MODEL_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
_SINGLE_FOCAL = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}
"""Models whose parameters start ``f, cx, cy``; the others start ``fx, fy, cx, cy``."""


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics of one COLMAP camera; distortion terms are kept but not modelled."""

    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    @property
    def _single_focal(self) -> bool:
        return self.model in _SINGLE_FOCAL

    @property
    def fx(self) -> float:
        """Horizontal focal length in pixels."""
        return self.params[0]

    @property
    def fy(self) -> float:
        """Vertical focal length in pixels."""
        return self.params[0] if self._single_focal else self.params[1]

    @property
    def cx(self) -> float:
        """Principal point column in pixels."""
        return self.params[1] if self._single_focal else self.params[2]

    @property
    def cy(self) -> float:
        """Principal point row in pixels."""
        return self.params[2] if self._single_focal else self.params[3]

    def scaled(self, width: int, height: int) -> Camera:
        """The same camera for images resized to ``width`` x ``height`` (e.g. ``images_4``).

        Returns:
            A ``PINHOLE`` camera; distortion terms are dropped.
        """
        sx, sy = width / self.width, height / self.height
        fx, fy, cx, cy = self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy
        return Camera(self.camera_id, "PINHOLE", width, height, (fx, fy, cx, cy))


@dataclass(frozen=True)
class Image:
    """One registered photo: its world-to-camera pose and file name."""

    image_id: int
    name: str
    camera_id: int
    rotation: np.ndarray
    """World-to-camera rotation matrix (3x3)."""
    translation: np.ndarray
    """World-to-camera translation (3,)."""

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.rotation.T @ self.translation

    @property
    def camera_to_world(self) -> np.ndarray:
        """Camera-to-world rotation; columns are the camera's x (right), y (down), z (forward)."""
        return self.rotation.T


@dataclass(frozen=True)
class SparseModel:
    """Cameras, posed images and the sparse coloured point cloud of one reconstruction."""

    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: np.ndarray
    """Point positions, ``(N, 3)`` float64."""
    colors: np.ndarray
    """Point colours, ``(N, 3)`` uint8."""
    errors: np.ndarray
    """Reprojection error per point, ``(N,)``."""
    track_lengths: np.ndarray
    """Number of images each point is seen in, ``(N,)``."""


def quat_to_matrix(qvec: np.ndarray) -> np.ndarray:
    """Rotation matrix of a w-first unit quaternion.

    Returns:
        The 3x3 rotation matrix.
    """
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_cameras(path: Path) -> dict[int, Camera]:
    """Read ``cameras.bin``.

    Returns:
        Cameras by id.
    """
    data = Path(path).read_bytes()
    (count,) = struct.unpack_from("<Q", data, 0)
    offset, cameras = 8, {}
    for _ in range(count):
        camera_id, model_id, width, height = struct.unpack_from("<iiQQ", data, offset)
        offset += 24
        name, num_params = _MODEL_PARAMS[model_id]
        params = struct.unpack_from(f"<{num_params}d", data, offset)
        offset += 8 * num_params
        cameras[camera_id] = Camera(camera_id, name, int(width), int(height), tuple(params))
    return cameras


def read_images(path: Path) -> dict[int, Image]:
    """Read ``images.bin`` (poses and names; the 2D observations are skipped).

    Returns:
        Images by id.
    """
    data = Path(path).read_bytes()
    (count,) = struct.unpack_from("<Q", data, 0)
    offset, images = 8, {}
    for _ in range(count):
        image_id, *pose, camera_id = struct.unpack_from("<i7di", data, offset)
        offset += 64
        end = data.index(b"\x00", offset)
        name = data[offset:end].decode()
        offset = end + 1
        (num_points,) = struct.unpack_from("<Q", data, offset)
        offset += 8 + 24 * num_points  # x, y (double) and point3D id (int64) per observation
        qvec, tvec = np.array(pose[:4]), np.array(pose[4:])
        images[image_id] = Image(image_id, name, camera_id, quat_to_matrix(qvec), tvec)
    return images


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read ``points3D.bin`` (the tracks are skipped).

    Returns:
        Positions ``(N, 3)``, colours ``(N, 3)`` uint8, reprojection errors and track lengths.
    """
    data = Path(path).read_bytes()
    (count,) = struct.unpack_from("<Q", data, 0)
    record = struct.Struct("<Q3d3BdQ")  # id, xyz, rgb, error, track length
    rows = np.empty((count, 8))
    offset = 8
    for i in range(count):
        values = record.unpack_from(data, offset)
        rows[i] = values[1:]
        offset += record.size + 8 * values[-1]  # image id + point2D index (int32 each) per track entry
    return rows[:, :3], rows[:, 3:6].astype(np.uint8), rows[:, 6], rows[:, 7].astype(np.int64)


def read_model(sparse_dir: Path) -> SparseModel:
    """Read ``cameras.bin``, ``images.bin`` and ``points3D.bin`` from ``sparse_dir`` (e.g. ``sparse/0``).

    Returns:
        The reconstruction.
    """
    sparse_dir = Path(sparse_dir)
    points, colors, errors, tracks = read_points(sparse_dir / "points3D.bin")
    return SparseModel(
        cameras=read_cameras(sparse_dir / "cameras.bin"),
        images=read_images(sparse_dir / "images.bin"),
        points=points,
        colors=colors,
        errors=errors,
        track_lengths=tracks,
    )
