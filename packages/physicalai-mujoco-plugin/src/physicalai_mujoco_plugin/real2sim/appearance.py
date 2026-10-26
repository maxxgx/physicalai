# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real appearance for capture cameras: the captured environment behind the simulated robot.

MuJoCo draws the robot well (it has the real meshes) and the environment badly (a convex hull and
a flat table). For a camera that sits where a capture photo was taken, the environment is already
known pixel for pixel, so a :class:`Compositor` keeps the sim's pixels only where something the
capture could not contain is in front: the robot, and objects that have moved since the capture.
Everywhere else the frame comes from a :class:`Background`.

Backgrounds are pluggable: :class:`PhotoBackground` is the capture photo itself (exact, but only
for that one viewpoint); a splat renderer (CUDA, ``h100s``) is the next one and also serves views
that were never photographed, such as a wrist camera. Known limits of compositing over baked
appearance, from the design: the robot casts no shadow on the background, and a moved object
leaves its captured image behind (a per-object splat removes that).
"""

from __future__ import annotations

import dataclasses
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import mujoco
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai_mujoco_plugin.cameras import CameraConfig

STREAM_WIDTH = 640
"""Width of a capture camera's default stream."""
MOVED_TOLERANCE = 0.005
"""An object farther than this (metres) from its captured position is drawn by the sim."""


class Background(Protocol):
    """The captured environment as seen by one camera, at the render's size."""

    def frame(self, width: int, height: int) -> np.ndarray:
        """The captured RGB image at ``width`` x ``height``."""
        ...


@dataclass
class PhotoBackground:
    """The capture photo, resized once to the render size."""

    path: Path
    _cache: dict[tuple[int, int], np.ndarray] | None = None

    def frame(self, width: int, height: int) -> np.ndarray:
        """The photo as RGB at ``width`` x ``height``.

        Returns:
            The resized photo (cached per size).

        Raises:
            FileNotFoundError: If the photo cannot be read.
        """
        if self._cache is None:
            self._cache = {}
        key = (width, height)
        if key not in self._cache:
            image = cv2.imread(str(self.path))
            if image is None:
                msg = f"cannot read capture photo {self.path}"
                raise FileNotFoundError(msg)
            self._cache[key] = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height), interpolation=cv2.INTER_AREA
            )
        return self._cache[key]


class Compositor:
    """Sim pixels where the robot (or a moved object) is visible, captured pixels elsewhere."""

    def __init__(
        self, model: mujoco.MjModel, background: Background, environment_geoms: set[int], objects: dict[int, np.ndarray]
    ) -> None:
        """Create a compositor for one camera.

        Args:
            model: The compiled scene.
            background: Captured appearance for this camera.
            environment_geoms: Geom ids the capture contains (table, floor, captured objects).
            objects: Captured object geom id -> its body's captured position (``real2sim.json``).
        """
        self._background = background
        self._environment = np.zeros(model.ngeom, dtype=bool)
        self._environment[list(environment_geoms)] = True
        self._objects = objects
        self._geom_body = model.geom_bodyid.copy()

    def __call__(self, renderer: mujoco.Renderer, data: mujoco.MjData, camera: str, frame: np.ndarray) -> np.ndarray:
        """Replace the environment in ``frame`` (MuJoCo's render of ``camera``) with the capture.

        Returns:
            The composited RGB frame.
        """
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(data, camera=camera)
            ids, kinds = np.moveaxis(renderer.render(), -1, 0)
        finally:
            renderer.disable_segmentation_rendering()
        sim_owned = ~self._environment.copy()
        for geom, captured in self._objects.items():
            if np.linalg.norm(data.xpos[self._geom_body[geom]] - captured) > MOVED_TOLERANCE:
                sim_owned[geom] = True
        valid = (ids >= 0) & (kinds == mujoco.mjtObj.mjOBJ_GEOM)
        mask = np.zeros(ids.shape, dtype=bool)
        mask[valid] = sim_owned[ids[valid]]
        height, width = frame.shape[:2]
        out = self._background.frame(width, height).copy()
        out[mask] = frame[mask]
        return out


def load_compositors(meta_path: Path, images_dir: Path, model: mujoco.MjModel) -> dict[str, Compositor]:
    """One photo-backed compositor per capture camera in ``real2sim.json`` that the model has.

    Returns:
        Compositors by camera name.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    environment: set[int] = set()
    for name in ("table", "floor"):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            environment.add(geom)
    objects: dict[int, np.ndarray] = {}
    for obj in meta["objects"]:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{obj['name']}_geom")
        if geom >= 0:
            environment.add(geom)
            objects[geom] = np.asarray(obj["center"], dtype=float)
    compositors = {}
    for cam in meta["cameras"]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam["name"]) < 0:
            continue
        compositors[cam["name"]] = Compositor(
            model, PhotoBackground(Path(images_dir) / cam["image"]), environment, objects
        )
    logger.info("Real appearance for cameras {}", sorted(compositors))
    return compositors


def compositor_factory(meta_path: str | Path) -> Callable[[mujoco.MjModel], dict[str, Compositor]]:
    """``CameraService.compositor_factory`` for a ``real2sim.json``; photos come from its ``images_dir``.

    Returns:
        A function building the compositors for a compiled model.
    """
    meta_path = Path(meta_path)
    images_dir = Path(json.loads(meta_path.read_text(encoding="utf-8"))["images_dir"])
    return functools.partial(load_compositors, meta_path, images_dir)


def capture_streams(meta_path: str | Path, streams: list[CameraConfig]) -> list[CameraConfig]:
    """``streams`` with each capture camera resized to its photo's aspect (640 wide).

    MuJoCo renders a camera with real intrinsics at its sensor's aspect; a 640x480 stream of a 3:2
    photo would stretch the sim against the captured background.

    Returns:
        The streams, capture cameras resized.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    sizes = {cam["name"]: (int(cam["width"]), int(cam["height"])) for cam in meta["cameras"]}
    return [
        dataclasses.replace(
            stream, width=STREAM_WIDTH, height=round(STREAM_WIDTH * sizes[stream.name][1] / sizes[stream.name][0])
        )
        if stream.name in sizes
        else stream
        for stream in streams
    ]


def show_splats(server: object, meta_path: str | Path, *, max_splats: int = 400_000) -> None:
    """Draw the scene's Gaussians (``splats.npz``) in a viser server, around the simulated robot.

    The viewer shows MuJoCo's geometry too; the splats are the captured room and table around it.
    The densest models are subsampled to ``max_splats`` to keep the browser responsive.
    """
    from physicalai_mujoco_plugin.real2sim.splats import Splats  # noqa: PLC0415

    meta_path = Path(meta_path)
    name = json.loads(meta_path.read_text(encoding="utf-8")).get("splats")
    if not name:
        return
    splats = Splats.load(meta_path.parent / name)
    keep = np.arange(len(splats))
    if len(keep) > max_splats:
        keep = np.random.default_rng(0).choice(len(keep), max_splats, replace=False)
    try:
        server.scene.add_gaussian_splats(  # type: ignore[attr-defined]
            "/real2sim/splats",
            centers=splats.centers[keep].astype(np.float32),
            covariances=splats.covariances()[keep].astype(np.float32),
            rgbs=splats.colors[keep].astype(np.float32),
            opacities=splats.opacities[keep, None].astype(np.float32),
        )
    except Exception as exc:  # noqa: BLE001 - the viewer is optional; the sim runs without the splats
        logger.warning("Could not show the captured splats in the viewer: {}", exc)
        return
    logger.info("Viewer shows {} captured Gaussians from {}", len(keep), name)
