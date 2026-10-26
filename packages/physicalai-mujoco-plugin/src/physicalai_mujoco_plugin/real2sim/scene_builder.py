# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Write a :class:`TableScene` as a robot-free scene XML for ``MuJoCoRobot(model_path=...)``.

The table top is ``z = 0``. Each object becomes a body whose collision geom is the convex hull of
its points (MuJoCo builds the hull from the mesh vertices), coloured with its median capture
colour. Capture cameras keep their real pose and intrinsics, so a sim render from ``overview``
lines up with the photo it came from. A ``real2sim.json`` beside the XML records the transform and
the cameras, for the image gate and the splat renderer.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from physicalai_mujoco_plugin.real2sim.geometry import clean_points
from physicalai_mujoco_plugin.real2sim.splats import splats_from_points

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai_mujoco_plugin.real2sim.colmap import Camera, Image, SparseModel
    from physicalai_mujoco_plugin.real2sim.geometry import SceneObject, TableScene

TABLE_THICKNESS = 0.03
HULL_VOXEL = 0.01
SPLATS_FILE = "splats.npz"
"""The scene's Gaussians in the table frame, beside ``scene.xml``; the viewer draws them."""
TABLE_COLOR_BAND = 0.005
"""Points this close to the table plane (metres) give the table its colour."""
"""Object points are thinned to one per voxel of this size before they become hull vertices."""


@dataclass(frozen=True)
class CaptureCamera:
    """A capture photo turned into a sim camera: name, image file and pinhole intrinsics."""

    name: str
    image: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    position: np.ndarray
    """Camera centre in the table frame."""
    xyaxes: np.ndarray
    """MuJoCo camera x (right) and y (up) axes in the table frame, six numbers."""


def capture_camera(name: str, image: Image, camera: Camera, scene: TableScene) -> CaptureCamera:
    """The sim camera for one photo. COLMAP looks along ``+z`` with ``y`` down; MuJoCo along ``-z``, ``y`` up.

    Returns:
        The camera in the table frame.
    """
    axes = scene.transform.apply_rotation(image.camera_to_world)
    return CaptureCamera(
        name=name,
        image=image.name,
        width=camera.width,
        height=camera.height,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        position=scene.transform.apply(image.center[None])[0],
        xyaxes=np.concatenate([axes[:, 0], -axes[:, 1]]),
    )


def select_cameras(model: SparseModel, scene: TableScene, count: int) -> list[Image]:
    """``count`` photos spread evenly around the table, by the camera's azimuth about the origin.

    The first is the one closest to ``+x``: it faces a robot mounted on the ``-x`` edge, which makes
    it the natural ``overview``.

    Returns:
        The chosen photos, without duplicates.
    """
    images = sorted(model.images.values(), key=lambda im: im.name)
    centers = scene.transform.apply(np.array([im.center for im in images]))
    azimuth = np.arctan2(centers[:, 1], centers[:, 0])
    chosen: list[int] = []
    for target in np.linspace(0.0, 2 * math.pi, count, endpoint=False):
        gap = np.abs(np.angle(np.exp(1j * (azimuth - target))))
        chosen.append(int(np.argmin(gap)))
    return [images[i] for i in dict.fromkeys(chosen)]


def _named_cameras(model: SparseModel, scene: TableScene, named: Mapping[str, str]) -> list[CaptureCamera]:
    """Capture cameras for explicitly named images, in the given order.

    Returns:
        One camera per entry.

    Raises:
        KeyError: If an image is not registered.
    """
    by_name = {image.name: image for image in model.images.values()}
    cameras = []
    for camera_name, image_name in named.items():
        if image_name not in by_name:
            msg = f"image {image_name!r} is not in the reconstruction (registered: {len(by_name)} images)"
            raise KeyError(msg)
        image = by_name[image_name]
        cameras.append(capture_camera(camera_name, image, model.cameras[image.camera_id], scene))
    return cameras


def _xyaxes_quat(xyaxes: np.ndarray) -> np.ndarray:
    """Quaternion (w-first) of the frame whose x and y axes are given; z = x cross y.

    Returns:
        The unit quaternion.
    """
    x, y = np.asarray(xyaxes[:3], dtype=float), np.asarray(xyaxes[3:], dtype=float)
    x /= np.linalg.norm(x)
    y -= (y @ x) * x
    y /= np.linalg.norm(y)
    mat = np.column_stack([x, y, np.cross(x, y)])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(-1))
    return quat


def _thin(points: np.ndarray, voxel: float) -> np.ndarray:
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def _rgba(colors: np.ndarray) -> list[float]:
    return [*(np.median(colors, axis=0) / 255.0).tolist(), 1.0]


def build_spec(
    scene: TableScene,
    cameras: list[CaptureCamera],
    *,
    name: str = "real2sim",
    table_padding: float = 0.25,
    table_color: np.ndarray | None = None,
    mount: tuple[float, float, float] = (-0.30, 0.0, 0.0),
    mount_yaw_deg: float = 0.0,
    free_objects: bool = True,
    object_mass: float = 0.2,
) -> mujoco.MjSpec:
    """The scene as an :class:`mujoco.MjSpec`.

    Args:
        scene: Table and objects in the metric table frame.
        cameras: Capture cameras, by their names.
        name: Model name.
        table_padding: The captured table area is usually only part of the table; extend the
            collision box by this much on every side so the robot and objects stay on it.
        table_color: Median RGB (0-255) of the table points, for the box colour.
        mount: ``robot_mount`` position on the table top.
        mount_yaw_deg: ``robot_mount`` yaw; 0 faces ``+x``.
        free_objects: Objects get a free joint (pushable, graspable) instead of being fixed.
        object_mass: Mass of each free object in kg (until measured).

    Returns:
        The uncompiled spec.
    """
    spec = mujoco.MjSpec()
    spec.modelname = name
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    width = max([cam.width for cam in cameras], default=640)
    height = max([cam.height for cam in cameras], default=480)
    getattr(spec.visual, "global").offwidth = max(width, 640)
    getattr(spec.visual, "global").offheight = max(height, 480)

    world = spec.worldbody
    world.add_light(pos=[1, 1, 3.5], dir=[-0.227, -0.268, -0.936], diffuse=[0.5, 0.5, 0.5])
    world.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5], castshadow=False)

    floor_z = scene.floor_z if scene.floor_z is not None else -0.75
    world.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01], pos=[0, 0, floor_z], rgba=[0.5, 0.5, 0.5, 1]
    )
    low = scene.table_min - table_padding
    high = scene.table_max + table_padding
    table_rgba = [0.55, 0.4, 0.3, 1.0] if table_color is None else [*(np.asarray(table_color) / 255.0).tolist(), 1.0]
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[(high[0] - low[0]) / 2, (high[1] - low[1]) / 2, TABLE_THICKNESS / 2],
        pos=[(high[0] + low[0]) / 2, (high[1] + low[1]) / 2, -TABLE_THICKNESS / 2],
        rgba=table_rgba,
        friction=[0.8, 0.005, 0.0001],
    )

    for obj in scene.objects:
        _add_object(spec, obj, free=free_objects, mass=object_mass)

    half = math.radians(mount_yaw_deg) / 2
    world.add_frame(name="robot_mount", pos=list(mount), quat=[math.cos(half), 0, 0, math.sin(half)])

    for cam in cameras:
        world.add_camera(
            name=cam.name,
            pos=cam.position.tolist(),
            quat=_xyaxes_quat(cam.xyaxes).tolist(),
            resolution=[cam.width, cam.height],
            focal_pixel=[cam.fx, cam.fy],
            # MuJoCo's principal point is the offset of the image centre from it (both axes), not
            # the other way round; measured against a pinhole projection to 0.3 px.
            principal_pixel=[cam.width / 2 - cam.cx, cam.height / 2 - cam.cy],
            sensor_size=[cam.width * 1e-5, cam.height * 1e-5],
        )
    return spec


def _add_object(spec: mujoco.MjSpec, obj: SceneObject, *, free: bool, mass: float) -> None:
    vertices = _thin(obj.points, HULL_VOXEL)
    # Points just above the table were cut away as table (``min_object_height``), so the hull
    # would float and drop. Project the lowest 2 cm onto ``z = 0``: the hull rests where it stood.
    low = vertices[vertices[:, 2] < vertices[:, 2].min() + 0.02]
    base = np.column_stack([low[:, :2], np.zeros(len(low))])
    vertices = np.vstack([vertices, base]) - obj.center
    spec.add_mesh(name=f"{obj.name}_hull", uservert=vertices.reshape(-1).tolist())
    body = spec.worldbody.add_body(name=obj.name, pos=obj.center.tolist())
    if free:
        body.add_freejoint(name=f"{obj.name}:joint")
    body.add_geom(
        name=f"{obj.name}_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{obj.name}_hull",
        rgba=_rgba(obj.colors),
        mass=mass if free else 0.0,
        friction=[0.8, 0.005, 0.0001],
    )


def write_scene(
    out_dir: Path,
    model: SparseModel,
    scene: TableScene,
    *,
    camera_count: int = 8,
    images_dir: Path | None = None,
    name: str = "real2sim",
    mount: tuple[float, float, float] = (-0.30, 0.0, 0.0),
    mount_yaw_deg: float = 0.0,
    free_objects: bool = True,
    named_cameras: Mapping[str, str] | None = None,
) -> Path:
    """Write ``scene.xml``, ``real2sim.json``, ``points.npz`` and ``splats.npz`` into ``out_dir``.

    Capture cameras are scaled to the photos' resolution by the caller (``model.cameras`` must
    describe the images the gate will compare against).

    Args:
        out_dir: Output directory, created if missing.
        model: The reconstruction (cameras scaled to the photos).
        scene: Table and objects from :func:`build_table_scene`.
        camera_count: Capture cameras to export, spread around the table.
        images_dir: The photos; recorded for the image gate and the appearance compositor.
        name: Model name.
        mount: ``robot_mount`` position on the table top.
        mount_yaw_deg: ``robot_mount`` yaw; 0 faces ``+x``.
        free_objects: Objects get a free joint instead of being fixed.
        named_cameras: Sim camera name -> registered image name (e.g. ``{"overview":
            "robotcam/overview.jpg"}``: a snapshot of the bench's real camera, registered in the
            same reconstruction). Without an ``overview`` entry, the first spread camera is it.

    Returns:
        The path of ``scene.xml``; a named image missing from the reconstruction raises ``KeyError``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = _named_cameras(model, scene, named_cameras or {})
    named_images = {cam.image for cam in cameras}
    spread = [image for image in select_cameras(model, scene, camera_count) if image.name not in named_images]
    cameras += [
        capture_camera(f"capture_{i:02d}", image, model.cameras[image.camera_id], scene)
        for i, image in enumerate(spread)
    ]
    if cameras and all(cam.name != "overview" for cam in cameras):
        cameras[0] = dataclasses.replace(cameras[0], name="overview")
    table_on = np.abs(scene.transform.apply(model.points)[:, 2]) < TABLE_COLOR_BAND
    table_color = np.median(model.colors[table_on], axis=0) if table_on.any() else None
    spec = build_spec(
        scene,
        cameras,
        name=name,
        table_color=table_color,
        mount=mount,
        mount_yaw_deg=mount_yaw_deg,
        free_objects=free_objects,
    )
    spec.compile()  # fail here, not in the plugin
    xml_path = out_dir / "scene.xml"
    xml_path.write_text(spec.to_xml())
    meta = {
        "images_dir": None if images_dir is None else str(Path(images_dir).resolve()),
        "transform": scene.transform.to_dict(),
        "table_min": scene.table_min.tolist(),
        "table_max": scene.table_max.tolist(),
        "floor_z": scene.floor_z,
        "objects": [
            {"name": o.name, "center": o.center.tolist(), "size": o.size.tolist(), "points": len(o.points)}
            for o in scene.objects
        ],
        "cameras": [
            {
                "name": cam.name,
                "image": cam.image,
                "width": cam.width,
                "height": cam.height,
                "fx": cam.fx,
                "fy": cam.fy,
                "cx": cam.cx,
                "cy": cam.cy,
                "position": cam.position.tolist(),
                "xyaxes": cam.xyaxes.tolist(),
            }
            for cam in cameras
        ],
    }
    # Until a splat is trained (``real2sim splat``), the sparse points stand in for it in the viewer.
    keep = clean_points(model)
    splats_from_points(scene.transform.apply(model.points[keep]), model.colors[keep]).save(out_dir / SPLATS_FILE)
    meta["splats"] = SPLATS_FILE
    (out_dir / "real2sim.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(
        out_dir / "points.npz",
        **{f"{o.name}_points": o.points for o in scene.objects},
        **{f"{o.name}_colors": o.colors for o in scene.objects},
    )
    return xml_path
