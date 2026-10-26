# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remove the real robot from the capture photos, so the simulated one does not leave a ghost.

A bench capture contains the real arm at the pose it had during the capture. Composited behind
the simulated arm, those pixels would show a second, frozen arm wherever the sim arm has moved
away. Knowing where the robot stands (``robot_mount``) and its joint values during the capture,
the sim renders its silhouette from every capture camera; :func:`clean_photos` dilates that mask
(calibration is never perfect) and inpaints it. The compositor then uses the cleaned photos.

Inpainting is a stopgap: it fills the arm's pixels with the surrounding texture. Splats remove the
robot properly (the design's per-link Gaussians), and a second capture with the arm moved away
fills the hole with real pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import mujoco
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


def robot_mask(
    model: mujoco.MjModel, data: mujoco.MjData, camera: str, width: int, height: int, environment: set[int]
) -> np.ndarray:
    """Pixels where a geom outside ``environment`` (the robot) is the nearest surface.

    Returns:
        A ``uint8`` mask (255 = robot) of ``height`` x ``width``.
    """
    renderer = mujoco.Renderer(model, height, width)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera)
        ids, kinds = np.moveaxis(renderer.render(), -1, 0)
    finally:
        renderer.close()
    robot = np.ones(model.ngeom, dtype=bool)
    robot[list(environment)] = False
    mask = np.zeros(ids.shape, dtype=bool)
    valid = (ids >= 0) & (kinds == _GEOM)
    mask[valid] = robot[ids[valid]]
    return mask.astype(np.uint8) * 255


def environment_geoms(model: mujoco.MjModel, meta: dict[str, object]) -> set[int]:
    """Geoms the capture contains: the table, the floor and the captured objects.

    Returns:
        Their ids.
    """
    names = ["table", "floor", *(f"{obj['name']}_geom" for obj in meta["objects"])]  # type: ignore[union-attr]
    return {i for i in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names) if i >= 0}


def clean_photos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta_path: Path,
    out_dir: Path,
    *,
    dilate_px: int = 8,
    inpaint_radius: int = 5,
    on_photo: Callable[[str, float], object] | None = None,
) -> Path:
    """Inpaint the robot out of every capture camera's photo; point ``real2sim.json`` at the result.

    ``data`` must hold the robot at its capture pose (``mj_forward`` is called here).

    Args:
        model: The scene with the robot attached.
        data: Its state at the capture pose.
        meta_path: The scene's ``real2sim.json``; ``images_dir`` moves to ``out_dir`` and the
            original is kept as ``raw_images_dir``.
        out_dir: Cleaned photos (same names) and ``masks/``.
        dilate_px: Grow the mask by this many pixels to cover calibration error and the shadow edge.
        inpaint_radius: OpenCV inpainting radius.
        on_photo: Called with each photo's name and the fraction of it that was masked.

    Returns:
        ``out_dir``.
    """
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source = Path(meta.get("raw_images_dir") or meta["images_dir"])
    out_dir = Path(out_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    mujoco.mj_forward(model, data)
    environment = environment_geoms(model, meta)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    for cam in meta["cameras"]:
        photo = cv2.imread(str(source / cam["image"]))
        height, width = photo.shape[:2]
        mask = cv2.dilate(robot_mask(model, data, cam["name"], width, height, environment), kernel)
        cv2.imwrite(str(out_dir / "masks" / f"{Path(cam['image']).stem}.png"), mask)
        cleaned = cv2.inpaint(photo, mask, inpaint_radius, cv2.INPAINT_TELEA) if mask.any() else photo
        cv2.imwrite(str(out_dir / cam["image"]), cleaned)
        if on_photo is not None:
            on_photo(cam["image"], float(mask.mean() / 255))
    meta["raw_images_dir"] = str(source)
    meta["images_dir"] = str(out_dir.resolve())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir
