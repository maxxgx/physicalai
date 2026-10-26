# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Image gate: the sim seen through each capture camera, laid over the photo it came from.

The first go/no-go of the real2sim design is that sim cameras sit where the real ones were. If
they do, the table edge and every object's outline in the sim render fall on the same pixels as in
the photo. This writes one sheet per camera (photo | sim | blend with the sim's object outlines in
magenta) and reports, per object, how much of its projected sparse points the sim silhouette
covers: a number that drops as soon as scale, pose or intrinsics drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
"""Segmentation object type of geoms as a plain int; comparing an array with the enum runs per element
(about 45 ms a 640x427 frame)."""


@dataclass(frozen=True)
class GateResult:
    """The gate's reading for one capture camera."""

    camera: str
    image: str
    coverage: dict[str, float]
    """Per object: fraction of its capture points whose projection lands on its sim silhouette."""
    sheet: Path


def _render(
    model: mujoco.MjModel, data: mujoco.MjData, camera: str, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """RGB and geom-id images of ``camera``.

    Returns:
        The RGB image and the geom id per pixel (``-1`` where no geom is visible).
    """
    renderer = mujoco.Renderer(model, height, width)
    try:
        renderer.update_scene(data, camera)
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera)
        ids, kinds = np.moveaxis(renderer.render(), -1, 0)
    finally:
        renderer.close()
    return rgb, np.where(kinds == _GEOM, ids, -1)


def project(points: np.ndarray, cam: dict[str, Any]) -> np.ndarray:
    """Pixel coordinates of table-frame points in a capture camera (pinhole, image y down).

    Returns:
        ``(N, 3)``: column, row and depth in front of the camera.
    """
    xyaxes = np.asarray(cam["xyaxes"], dtype=float)
    x_axis, y_axis = xyaxes[:3], xyaxes[3:]
    z_axis = np.cross(x_axis, y_axis)  # MuJoCo camera looks along -z
    rel = points - np.asarray(cam["position"], dtype=float)
    xc, yc, depth = rel @ x_axis, rel @ y_axis, -(rel @ z_axis)
    u = cam["fx"] * xc / depth + cam["cx"]
    v = -cam["fy"] * yc / depth + cam["cy"]
    return np.column_stack([u, v, depth])


def coverage(mask: np.ndarray, points: np.ndarray, cam: dict[str, Any]) -> float | None:
    """Fraction of ``points`` in view whose projection falls on ``mask``; ``None`` when none are in view.

    Returns:
        The fraction, or ``None``.
    """
    height, width = mask.shape
    uvd = project(points, cam)
    inside = (uvd[:, 2] > 0) & (uvd[:, 0] >= 0) & (uvd[:, 0] < width) & (uvd[:, 1] >= 0) & (uvd[:, 1] < height)
    if not inside.any():
        return None
    return float(mask[uvd[inside, 1].astype(int), uvd[inside, 0].astype(int)].mean())


def run_gate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta_path: Path,
    images_dir: Path,
    out_dir: Path,
    object_points: dict[str, np.ndarray] | None = None,
) -> list[GateResult]:
    """Write one sheet per capture camera into ``out_dir`` (photo | sim | blend with outlines).

    Returns:
        Per camera, the per-object coverage and the sheet's path.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mujoco.mj_forward(model, data)
    object_points = object_points or {}
    geoms = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom") for name in object_points}
    results = []
    for cam in meta["cameras"]:
        size = (int(cam["width"]), int(cam["height"]))
        photo = cv2.resize(cv2.cvtColor(cv2.imread(str(Path(images_dir) / cam["image"])), cv2.COLOR_BGR2RGB), size)
        sim, seg = _render(model, data, cam["name"], *size)
        blend = (0.5 * photo + 0.5 * sim).astype(np.uint8)
        readings = _outline_and_measure(blend, seg, geoms, object_points, cam)
        path = out_dir / f"gate_{cam['name']}.jpg"
        sheet = cv2.cvtColor(np.hstack([photo, sim, blend]), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results.append(GateResult(cam["name"], cam["image"], readings, path))
    return results


def _outline_and_measure(
    blend: np.ndarray,
    seg: np.ndarray,
    geoms: dict[str, int],
    object_points: dict[str, np.ndarray],
    cam: dict[str, Any],
) -> dict[str, float]:
    """Draw each object's sim outline on ``blend`` in magenta; return its coverage reading.

    Returns:
        Coverage by object name, for objects with points in view.
    """
    readings = {}
    for name, geom in geoms.items():
        mask = (seg == geom).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(blend, contours, -1, (255, 0, 255), 2)
        reading = coverage(mask, object_points[name], cam)
        if reading is not None:
            readings[name] = reading
    return readings
