# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Find the table in a capture, put it in a metric z-up frame, and cut out what stands on it.

A capture comes in an arbitrary frame and scale (COLMAP's). Everything the sim needs hangs off
one surface: the table. Its plane fixes ``z`` (up, towards the cameras), the scene centre the
cameras orbit fixes the origin, the table points' principal axis fixes ``x``, and the scale comes
either from the user (``scale``) or from a floor found below the table and a known table height.

Pure numpy on purpose: a few hundred thousand sparse points need no point-cloud library, and the
plugin should not grow a heavy dependency for a proof of concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.real2sim.colmap import SparseModel

_DEGENERATE = 1e-12
_MIN_PLANE_POINTS = 3
_MIN_FLOOR_CANDIDATES = 500
"""Fewer points below the table than this: no floor in the capture."""
_MIN_FLOOR_INLIERS = 300


@dataclass(frozen=True)
class Similarity:
    """``p_table = scale * rotation @ (p_capture - origin)``: capture frame to metric table frame."""

    rotation: np.ndarray
    origin: np.ndarray
    scale: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Capture-frame points ``(N, 3)`` in the table frame.

        Returns:
            The transformed points.
        """
        return self.scale * (np.asarray(points) - self.origin) @ self.rotation.T

    def apply_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """A capture-frame orientation (columns = axes) expressed in the table frame.

        Returns:
            The rotated 3x3 matrix.
        """
        return self.rotation @ rotation

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, the inverse of :meth:`from_dict`.

        Returns:
            ``rotation``, ``origin`` and ``scale``.
        """
        return {"rotation": self.rotation.tolist(), "origin": self.origin.tolist(), "scale": self.scale}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Similarity:
        """Rebuild from :meth:`to_dict`'s output.

        Returns:
            The transform.
        """
        return cls(
            np.asarray(data["rotation"], dtype=float), np.asarray(data["origin"], dtype=float), float(data["scale"])
        )


@dataclass(frozen=True)
class Plane:
    """``normal . p + offset = 0`` with a unit normal."""

    normal: np.ndarray
    offset: float

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distances of ``points`` ``(N, 3)``, positive on the normal's side.

        Returns:
            ``(N,)`` distances.
        """
        return points @ self.normal + self.offset


@dataclass(frozen=True)
class SceneObject:
    """One thing standing on the table, in the metric table frame."""

    name: str
    points: np.ndarray
    colors: np.ndarray
    center: np.ndarray
    """Footprint centre on the table (``z = 0``)."""
    size: np.ndarray
    """Axis-aligned extent ``(dx, dy, dz)`` in metres."""


@dataclass(frozen=True)
class TableScene:
    """The table and its objects, metric and z-up, plus the transform that got them there."""

    transform: Similarity
    table_min: np.ndarray
    """Table top corner ``(x, y)`` with the smallest coordinates; the top is at ``z = 0``."""
    table_max: np.ndarray
    objects: list[SceneObject] = field(default_factory=list)
    floor_z: float | None = None
    """Floor height below the table top (negative) when a floor was found."""
    background: np.ndarray | None = None
    """Points that are neither table nor object, in the table frame (walls, chairs, plants)."""
    background_colors: np.ndarray | None = None


def look_at_center(model: SparseModel) -> np.ndarray:
    """The point closest, in least squares, to every camera's optical axis: what the capture orbits.

    Returns:
        The point in capture coordinates.
    """
    lhs, rhs = np.zeros((3, 3)), np.zeros(3)
    for image in model.images.values():
        direction = image.camera_to_world[:, 2]
        projector = np.eye(3) - np.outer(direction, direction)
        lhs += projector
        rhs += projector @ image.center
    return np.linalg.solve(lhs, rhs)


def mean_camera_up(model: SparseModel) -> np.ndarray:
    """Average of the cameras' up vectors (COLMAP's ``-y``): gravity's opposite for a handheld capture.

    Returns:
        A unit vector in capture coordinates.
    """
    up = -np.mean([image.camera_to_world[:, 1] for image in model.images.values()], axis=0)
    return up / np.linalg.norm(up)


def clean_points(model: SparseModel, *, max_error: float = 1.5, min_track: int = 3) -> np.ndarray:
    """Mask of sparse points with a small reprojection error, seen in enough images.

    Returns:
        A boolean mask over ``model.points``.
    """
    return (model.errors <= max_error) & (model.track_lengths >= min_track)


def fit_plane(
    points: np.ndarray,
    *,
    up: np.ndarray,
    threshold: float,
    max_tilt_deg: float = 30.0,
    iterations: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[Plane, np.ndarray]:
    """RANSAC the largest plane whose normal is within ``max_tilt_deg`` of ``up``; refit on inliers.

    Returns:
        The plane, normal oriented along ``up``, and its inlier mask.

    Raises:
        ValueError: If no plane is within the allowed tilt.
    """
    rng = rng or np.random.default_rng(0)
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < _DEGENERATE:
            continue
        normal /= norm
        if abs(normal @ up) < cos_tilt:
            continue
        inliers = np.abs((points - a) @ normal) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < _MIN_PLANE_POINTS:
        msg = "no plane within the allowed tilt of the cameras' up direction"
        raise ValueError(msg)
    centroid = points[best].mean(axis=0)
    normal = np.linalg.svd(points[best] - centroid, full_matrices=False)[2][-1]
    if normal @ up < 0:
        normal = -normal
    plane = Plane(normal, float(-normal @ centroid))
    return plane, np.abs(plane.distance(points)) < threshold


def _frame(normal: np.ndarray, in_plane_points: np.ndarray) -> np.ndarray:
    """Rotation whose rows are the table frame's x, y, z in capture coordinates; x = principal axis.

    Returns:
        The 3x3 rotation.
    """
    centered = in_plane_points - in_plane_points.mean(axis=0)
    projected = centered - np.outer(centered @ normal, normal)
    x = np.linalg.svd(projected, full_matrices=False)[2][0]
    x -= (x @ normal) * normal
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    return np.stack([x, y, normal])


def connected_clusters(points: np.ndarray, voxel: float) -> np.ndarray:
    """Label points by connected occupied voxels (26-neighbourhood).

    A union-find over voxels: cheap, deterministic and good enough to split objects that do not
    touch, which is all the table top needs.

    Returns:
        A cluster label per point, ``0`` .. ``k - 1``.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    index = {tuple(k): i for i, k in enumerate(unique)}
    parent = np.arange(len(unique))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
    for i, key in enumerate(unique):
        for dx, dy, dz in offsets:
            j = index.get((key[0] + dx, key[1] + dy, key[2] + dz))
            if j is not None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(unique))])
    _, labels = np.unique(roots, return_inverse=True)
    return labels[inverse]


def build_table_scene(  # noqa: PLR0914
    model: SparseModel,
    *,
    scale: float | None = None,
    table_height: float | None = None,
    plane_threshold: float = 0.01,
    min_object_height: float = 0.01,
    max_object_height: float = 0.5,
    table_margin: float = 0.02,
    voxel: float = 0.015,
    min_object_points: int = 200,
) -> TableScene:
    """Find the table, the floor (optional) and the objects; return them in a metric table frame.

    Thresholds are in metres. Before the scale is known (the table and floor search), they are
    applied as if the median camera-to-centre distance were 1 m.

    Args:
        model: The COLMAP reconstruction.
        scale: Metres per capture unit, when known (markers, LiDAR, a measured length).
        table_height: Table top height above the floor in metres; with a floor found below the
            table, it fixes the scale when ``scale`` is not given.
        plane_threshold: Distance from the plane that counts as on the table.
        min_object_height: Points lower than this above the table belong to the table (cloths, mats).
        max_object_height: Points higher than this are background (walls, plants), not objects.
        table_margin: Objects must lie this far inside the table's edge.
        voxel: Connectivity voxel for splitting objects.
        min_object_points: Smaller clusters are noise.

    Returns:
        The table scene.

    Raises:
        ValueError: Neither ``scale`` nor ``table_height`` is given, or no floor is found to
            turn ``table_height`` into a scale.
    """
    keep = clean_points(model)
    points, colors = model.points[keep], model.colors[keep]
    center = look_at_center(model)
    up = mean_camera_up(model)
    unit = float(np.median([np.linalg.norm(im.center - center) for im in model.images.values()]))

    # The table: the dominant up-facing plane near what the cameras look at.
    near = np.linalg.norm(points - center, axis=1) < 1.0 * unit
    table_plane, _ = fit_plane(points[near], up=up, threshold=plane_threshold * unit)
    rotation = _frame(
        table_plane.normal, points[near][np.abs(table_plane.distance(points[near])) < plane_threshold * unit]
    )
    origin = center - table_plane.distance(center[None])[0] * table_plane.normal

    floor_offset = _floor_offset(points, table_plane, threshold=plane_threshold * unit, below=0.2 * unit)

    if scale is None:
        if table_height is None or floor_offset is None:
            msg = "pass scale, or table_height with a capture that shows the floor below the table"
            raise ValueError(msg)
        scale = table_height / -floor_offset
    transform = Similarity(rotation, origin, float(scale))

    metric = transform.apply(points)
    on_table = np.abs(metric[:, 2]) < plane_threshold
    near_metric = np.linalg.norm(metric[:, :2], axis=1) < 1.0 * unit * scale
    table_xy = metric[on_table & near_metric, :2]
    table_min, table_max = np.percentile(table_xy, 1, axis=0), np.percentile(table_xy, 99, axis=0)

    inside = np.all((metric[:, :2] > table_min + table_margin) & (metric[:, :2] < table_max - table_margin), axis=1)
    candidate = inside & (metric[:, 2] > min_object_height) & (metric[:, 2] < max_object_height)
    objects = _segment_objects(metric[candidate], colors[candidate], voxel=voxel, min_points=min_object_points)
    used = on_table | candidate
    return TableScene(
        transform=transform,
        table_min=table_min,
        table_max=table_max,
        objects=objects,
        floor_z=None if floor_offset is None else float(floor_offset * scale),
        background=metric[~used],
        background_colors=colors[~used],
    )


def _floor_offset(points: np.ndarray, table_plane: Plane, *, threshold: float, below: float) -> float | None:
    """Signed distance of a floor parallel to the table and at least ``below`` under it, if one is found.

    Returns:
        The floor's distance from the table plane (negative), or ``None``.
    """
    under = table_plane.distance(points) < -below
    if under.sum() < _MIN_FLOOR_CANDIDATES:
        return None
    try:
        _, inliers = fit_plane(points[under], up=table_plane.normal, threshold=threshold, max_tilt_deg=5.0)
    except ValueError:
        return None
    if inliers.sum() < _MIN_FLOOR_INLIERS:
        return None
    return float(np.median(table_plane.distance(points[under][inliers])))


def _segment_objects(points: np.ndarray, colors: np.ndarray, *, voxel: float, min_points: int) -> list[SceneObject]:
    """Split the points above the table into objects, largest first.

    Returns:
        Objects with at least ``min_points`` points.
    """
    objects: list[SceneObject] = []
    if not len(points):
        return objects
    labels = connected_clusters(points, voxel)
    for label in np.argsort(-np.bincount(labels)):
        members = labels == label
        if members.sum() < min_points:
            break
        obj = points[members]
        low, high = obj.min(axis=0), obj.max(axis=0)
        objects.append(
            SceneObject(
                name=f"object{len(objects)}",
                points=obj,
                colors=colors[members],
                center=np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, 0.0]),
                size=high - np.array([low[0], low[1], 0.0]),
            )
        )
    return objects
