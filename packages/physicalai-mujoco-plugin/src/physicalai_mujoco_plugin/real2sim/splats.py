# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Gaussian splats of a capture, moved into the scene's metric table frame.

A trained 3D Gaussian Splatting model (the INRIA ``.ply`` layout that gsplat, nerfstudio and brush
export) lives in the capture's COLMAP frame. :func:`transform_splats` applies the scene's
:class:`~physicalai_mujoco_plugin.real2sim.geometry.Similarity` to every Gaussian: centres,
orientations and (log) scales. Only the view-independent colour (SH degree 0) is kept here, which
is what viser draws; a renderer that needs the higher bands renders in the capture frame instead
(:class:`~physicalai_mujoco_plugin.real2sim.appearance.SplatBackground`).

Without a trained model, :func:`splats_from_points` turns the sparse COLMAP points into small
isotropic Gaussians: coarse, but enough to see the captured room around the robot in the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.real2sim.geometry import Similarity

SH_C0 = 0.28209479177387814
"""Degree-0 spherical harmonic constant: ``rgb = 0.5 + SH_C0 * f_dc``."""


@dataclass(frozen=True)
class Splats:
    """Gaussians as arrays, one row per Gaussian.

    Centres ``(N, 3)``, w-first unit quaternions ``(N, 4)``, linear (not log) scales ``(N, 3)``,
    colours ``(N, 3)`` in 0..1 and opacities ``(N,)`` in 0..1.
    """

    centers: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray

    def __len__(self) -> int:
        """Number of Gaussians.

        Returns:
            The count.
        """
        return len(self.centers)

    def covariances(self) -> np.ndarray:
        """``R S S R^T`` per Gaussian.

        Returns:
            ``(N, 3, 3)`` covariance matrices.
        """
        rotations = quats_to_matrices(self.quats)
        scaled = rotations * self.scales[:, None, :]
        return scaled @ np.swapaxes(scaled, 1, 2)

    def save(self, path: Path) -> None:
        """Write as ``.npz`` (what the viewer loads)."""
        np.savez_compressed(
            path,
            centers=self.centers,
            quats=self.quats,
            scales=self.scales,
            colors=self.colors,
            opacities=self.opacities,
        )

    @classmethod
    def load(cls, path: Path) -> Splats:
        """Read what :meth:`save` wrote.

        Returns:
            The splats.
        """
        data = np.load(path)
        return cls(data["centers"], data["quats"], data["scales"], data["colors"], data["opacities"])


def quats_to_matrices(quats: np.ndarray) -> np.ndarray:
    """Rotation matrices of w-first quaternions.

    Returns:
        ``(N, 3, 3)`` matrices.
    """
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """W-first unit quaternion of one rotation matrix.

    Returns:
        The quaternion.
    """
    m = matrix
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of w-first quaternions (broadcasts over rows).

    Returns:
        The products.
    """
    aw, ax, ay, az = np.moveaxis(np.atleast_2d(a), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.atleast_2d(b), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def read_ply_fields(path: Path) -> dict[str, np.ndarray]:
    """Read a 3DGS ``.ply`` (binary little-endian; ``x y z f_dc_* f_rest_* opacity scale_* rot_*``) raw.

    Returns:
        ``means`` ``(N, 3)``, unit w-first ``quats`` ``(N, 4)``, ``log_scales`` ``(N, 3)``,
        ``opacity_logits`` ``(N,)`` and spherical harmonics ``sh`` ``(N, K, 3)`` (``K = (degree + 1)²``).

    Raises:
        ValueError: If the file is not a binary little-endian vertex list with the 3DGS fields.
    """
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        msg = f"{path}: only binary little-endian PLY is supported"
        raise ValueError(msg)
    count, names, types = 0, [], []
    sizes = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
        elif parts[:1] == ["property"] and len(parts) == 3:  # noqa: PLR2004 - "property <type> <name>"
            types.append(sizes[parts[1]])
            names.append(parts[2])
    table = np.frombuffer(raw, dtype=np.dtype(list(zip(names, types, strict=True))), count=count, offset=end)
    needed = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "rot_0"}
    if not needed <= set(names):
        msg = f"{path}: missing 3DGS fields {sorted(needed - set(names))}"
        raise ValueError(msg)
    dc = np.column_stack([table[f"f_dc_{i}"] for i in range(3)]).astype(np.float64)
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n.rsplit("_", 1)[1]))
    # INRIA stores the higher bands channel-major: f_rest = [R coeffs..., G coeffs..., B coeffs...].
    rest = (
        np.column_stack([table[n] for n in rest_names]).reshape(count, 3, -1) if rest_names else np.zeros((count, 3, 0))
    )
    quats = np.column_stack([table[f"rot_{i}"] for i in range(4)]).astype(np.float64)
    return {
        "means": np.column_stack([table["x"], table["y"], table["z"]]).astype(np.float64),
        "quats": quats / np.linalg.norm(quats, axis=1, keepdims=True),
        "log_scales": np.column_stack([table[f"scale_{i}"] for i in range(3)]).astype(np.float64),
        "opacity_logits": table["opacity"].astype(np.float64),
        "sh": np.concatenate([dc[:, None, :], np.swapaxes(rest, 1, 2).astype(np.float64)], axis=1),
    }


def read_ply(path: Path) -> Splats:
    """Read a 3DGS ``.ply`` with activated scales, opacities and view-independent colours.

    Returns:
        The splats in the capture frame.
    """
    fields = read_ply_fields(path)
    return Splats(
        centers=fields["means"],
        quats=fields["quats"],
        scales=np.exp(fields["log_scales"]),
        colors=np.clip(0.5 + SH_C0 * fields["sh"][:, 0, :], 0.0, 1.0),
        opacities=1.0 / (1.0 + np.exp(-fields["opacity_logits"])),
    )


def transform_splats(splats: Splats, transform: Similarity) -> Splats:
    """The splats in the table frame (``p' = s R (p - o)``): centres, orientations and scales.

    Returns:
        The moved splats; colours and opacities are unchanged.
    """
    rotation = matrix_to_quat(transform.rotation)
    return Splats(
        centers=transform.apply(splats.centers),
        quats=quat_multiply(rotation, splats.quats),
        scales=splats.scales * transform.scale,
        colors=splats.colors,
        opacities=splats.opacities,
    )


def splats_from_points(points: np.ndarray, colors: np.ndarray, *, sigma: float = 0.004, opacity: float = 0.9) -> Splats:
    """Isotropic Gaussians at sparse points (a stand-in until a splat is trained).

    Returns:
        One Gaussian per point.
    """
    count = len(points)
    return Splats(
        centers=np.asarray(points, dtype=np.float64),
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        scales=np.full((count, 3), sigma),
        colors=np.asarray(colors, dtype=np.float64) / 255.0,
        opacities=np.full(count, opacity),
    )
