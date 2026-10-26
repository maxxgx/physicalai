# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real appearance for capture cameras: the captured environment behind the simulated robot.

MuJoCo draws the robot well (it has the real meshes) and the environment badly (a convex hull and
a flat table). For a camera that sits where a capture photo was taken, the environment is already
known pixel for pixel, so a :class:`Compositor` keeps the sim's pixels only where something the
capture could not contain is in front: the robot, and objects that have moved since the capture.
Everywhere else the frame comes from a :class:`Background`.

Backgrounds are pluggable: :class:`PhotoBackground` is the capture photo itself (exact, but only
for that one viewpoint); :class:`SplatBackground` renders a trained Gaussian splat (gsplat, CUDA)
and also serves views that were never photographed, such as a wrist camera. Known limits of
compositing over baked
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

from physicalai_mujoco_plugin.real2sim.geometry import Similarity
from physicalai_mujoco_plugin.real2sim.splats import read_ply_fields

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai_mujoco_plugin.cameras import CameraConfig

_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
"""Segmentation object type of geoms as a plain int; comparing an array with the enum runs per element
(about 45 ms a 640x427 frame)."""

STREAM_WIDTH = 640
"""Width of a capture camera's default stream."""
MOVED_TOLERANCE = 0.005
"""An object farther than this (metres) from its captured position is drawn by the sim."""


@dataclass(frozen=True)
class View:
    """Where a camera is and how it projects, at the size being rendered."""

    position: np.ndarray
    """Camera centre in the scene (table) frame."""
    rotation: np.ndarray
    """Columns: the MuJoCo camera's x (right), y (up) and z (backwards) axes in the scene frame."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def camera_view(model: mujoco.MjModel, data: mujoco.MjData, camera: int, width: int, height: int) -> View:
    """The pinhole view of ``camera`` rendered at ``width`` x ``height``.

    A camera with a sensor size (real intrinsics) projects with its focal length and principal
    point; MuJoCo's principal point is the offset of the image centre from it. Others use ``fovy``.

    Returns:
        The view, in pixels of the render.
    """
    sensor = model.cam_sensorsize[camera]
    if sensor[0] > 0 and sensor[1] > 0:
        focal_x, focal_y, principal_x, principal_y = model.cam_intrinsic[camera]
        fx, fy = focal_x / sensor[0] * width, focal_y / sensor[1] * height
        cx, cy = width / 2 - principal_x / sensor[0] * width, height / 2 - principal_y / sensor[1] * height
    else:
        fy = height / 2 / np.tan(np.radians(model.cam_fovy[camera]) / 2)
        fx, cx, cy = fy, width / 2, height / 2
    return View(
        position=data.cam_xpos[camera].copy(),
        rotation=data.cam_xmat[camera].reshape(3, 3).copy(),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        width=width,
        height=height,
    )


class Background(Protocol):
    """The captured environment as one camera sees it."""

    def frame(self, view: View) -> np.ndarray:
        """The captured RGB image for ``view`` (``view.height`` x ``view.width``)."""
        ...


@dataclass
class PhotoBackground:
    """The capture photo, resized once to the render size."""

    path: Path
    _cache: dict[tuple[int, int], np.ndarray] | None = None

    def frame(self, view: View) -> np.ndarray:
        """The photo as RGB at the view's size; the pose is the photo's own.

        Returns:
            The resized photo (cached per size).

        Raises:
            FileNotFoundError: If the photo cannot be read.
        """
        if self._cache is None:
            self._cache = {}
        width, height = view.width, view.height
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


def capture_camera_matrices(view: View, transform: Similarity) -> tuple[np.ndarray, np.ndarray]:
    """``view`` as an OpenCV camera in the capture frame (where a splat lives).

    MuJoCo cameras look along ``-z`` with ``y`` up; OpenCV (and gsplat) along ``+z`` with ``y`` down.

    Returns:
        The 4x4 world-to-camera matrix in capture coordinates and the 3x3 pixel intrinsics.
    """
    center = transform.origin + transform.rotation.T @ view.position / transform.scale
    cam_to_capture = transform.rotation.T @ view.rotation @ np.diag([1.0, -1.0, -1.0])
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = cam_to_capture.T
    world_to_cam[:3, 3] = -cam_to_capture.T @ center
    intrinsics = np.array([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]])
    return world_to_cam, intrinsics


class SplatBackground:
    """A trained Gaussian splat rendered with gsplat (CUDA) from any view.

    Renders in the capture frame (the splat's own), so the full spherical harmonics stay valid: the
    view is moved there with the inverse of the scene's :class:`Similarity`.
    """

    def __init__(self, ply: Path, transform: Similarity, *, device: str = "cuda") -> None:
        """Load ``ply`` onto ``device``.

        Raises:
            RuntimeError: If ``device`` is CUDA and no GPU is visible.
        """
        import gsplat  # noqa: F401, PLC0415 - fail now (ImportError) rather than on the first frame
        import torch  # noqa: PLC0415

        if device.startswith("cuda") and not torch.cuda.is_available():
            msg = "no CUDA device"
            raise RuntimeError(msg)
        fields = read_ply_fields(ply)
        self._torch = torch
        self._device = device
        self._transform = transform
        tensor = functools.partial(torch.as_tensor, dtype=torch.float32, device=device)
        self._means = tensor(fields["means"])
        self._quats = tensor(fields["quats"])
        self._scales = tensor(np.exp(fields["log_scales"]))
        self._opacities = tensor(1.0 / (1.0 + np.exp(-fields["opacity_logits"])))
        self._sh = tensor(fields["sh"])
        self._sh_degree = round(np.sqrt(fields["sh"].shape[1])) - 1

    def frame(self, view: View) -> np.ndarray:
        """Render the splat from ``view``.

        Returns:
            An RGB ``uint8`` image of the view's size.
        """
        from gsplat import rasterization  # noqa: PLC0415

        torch = self._torch
        t = self._transform
        world_to_cam, intrinsics = capture_camera_matrices(view, t)
        with torch.no_grad():
            colors, _, _ = rasterization(
                self._means,
                self._quats,
                self._scales,
                self._opacities,
                self._sh,
                torch.as_tensor(world_to_cam[None], dtype=torch.float32, device=self._device),
                torch.as_tensor(intrinsics[None], dtype=torch.float32, device=self._device),
                view.width,
                view.height,
                sh_degree=self._sh_degree,
                near_plane=0.01 / t.scale,
            )
        image = colors[0].clamp(0, 1).mul(255).byte().cpu().numpy()
        return np.ascontiguousarray(image[:, :, :3])


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
        self._model = model

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
        valid = (ids >= 0) & (kinds == _GEOM)
        mask = np.zeros(ids.shape, dtype=bool)
        mask[valid] = sim_owned[ids[valid]]
        height, width = frame.shape[:2]
        camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        out = self._background.frame(camera_view(self._model, data, camera_id, width, height)).copy()
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
    splat = _splat_background(meta)
    if splat is not None:
        for camera in range(model.ncam):
            name = model.camera(camera).name
            if name and name not in compositors:
                compositors[name] = Compositor(model, splat, environment, objects)
    logger.info("Real appearance for cameras {}", sorted(compositors))
    return compositors


def _splat_background(meta: dict[str, object]) -> SplatBackground | None:
    """The trained splat as a background for cameras without a photo, where gsplat and CUDA are available.

    Returns:
        The background, or ``None`` (no trained splat, or no GPU: those cameras keep MuJoCo's frames).
    """
    ply = meta.get("splat_ply")
    if not ply:
        return None
    try:
        return SplatBackground(Path(str(ply)), Similarity.from_dict(meta["transform"]))  # type: ignore[arg-type]
    except (ImportError, RuntimeError) as exc:
        logger.warning("Splat background unavailable ({}); cameras without a photo keep MuJoCo's rendering", exc)
        return None


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
