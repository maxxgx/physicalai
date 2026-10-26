# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""real2sim: COLMAP reading, table alignment, scene export and real-appearance compositing.

A synthetic capture stands in for a real one: a 0.6 m table with a 0.2 x 0.1 x 0.08 m box on it,
seen by cameras on a ring, all expressed in an arbitrary tilted, shifted and scaled capture frame
(what COLMAP produces). The pipeline must recover the metric table frame and the box from it.
"""

# MuJoCo's bindings are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import dataclasses
import json
import math
import struct
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.cameras import offscreen_rendering_available
from physicalai_mujoco_plugin.real2sim import appearance
from physicalai_mujoco_plugin.real2sim.colmap import Camera, Image, SparseModel, quat_to_matrix, read_model
from physicalai_mujoco_plugin.real2sim.gate import project
from physicalai_mujoco_plugin.real2sim.geometry import build_table_scene, connected_clusters
from physicalai_mujoco_plugin.real2sim.scene_builder import write_scene

if TYPE_CHECKING:
    from pathlib import Path

CAPTURE_SCALE = 3.0
"""Capture units per metre in the synthetic capture (the pipeline must undo it)."""
WIDTH, HEIGHT, FOCAL = 320, 240, 260.0
BOX_SIZE = np.array([0.2, 0.1, 0.08])
BOX_CENTER = np.array([0.05, -0.03])


def _capture_frame() -> tuple[np.ndarray, np.ndarray]:
    """Rotation and offset from the metric table frame to the synthetic capture frame."""
    rotation = quat_to_matrix(np.array([0.9, 0.3, -0.2, 0.25]))
    return rotation, np.array([1.5, -0.7, 2.0])


def _to_capture(points: np.ndarray) -> np.ndarray:
    rotation, offset = _capture_frame()
    return CAPTURE_SCALE * points @ rotation.T + offset


def _synthetic_model() -> SparseModel:
    rng = np.random.default_rng(1)
    table = np.column_stack([rng.uniform(-0.3, 0.3, (6000, 2)), np.zeros(6000)])
    low = np.array([*(BOX_CENTER - BOX_SIZE[:2] / 2), 0.0])
    box = low + rng.uniform(0, 1, (3000, 3)) * BOX_SIZE
    points = np.vstack([table, box])
    colors = np.vstack([np.full((len(table), 3), (150, 100, 60)), np.full((len(box), 3), (230, 200, 20))]).astype(
        np.uint8
    )

    rotation, _ = _capture_frame()
    images = {}
    for i, azimuth in enumerate(np.linspace(0, 2 * math.pi, 12, endpoint=False)):
        center = np.array([0.6 * math.cos(azimuth), 0.6 * math.sin(azimuth), 0.4])
        forward = -center / np.linalg.norm(center)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        cam_to_world = rotation @ np.column_stack([right, down, forward])  # COLMAP: x right, y down, z forward
        world_center = _to_capture(center[None])[0]
        world_to_cam = cam_to_world.T
        images[i + 1] = Image(i + 1, f"{i:03d}.png", 1, world_to_cam, -world_to_cam @ world_center)
    return SparseModel(
        cameras={1: Camera(1, "PINHOLE", WIDTH, HEIGHT, (FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2))},
        images=images,
        points=_to_capture(points),
        colors=colors,
        errors=np.full(len(points), 0.5),
        track_lengths=np.full(len(points), 5),
    )


def test_colmap_binary_round_trip(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    with (sparse / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", 2))
        f.write(struct.pack("<iiQQ", 1, 2, 640, 480) + struct.pack("<4d", 500.0, 320.0, 240.0, 0.01))  # SIMPLE_RADIAL
        f.write(struct.pack("<iiQQ", 2, 1, 800, 600) + struct.pack("<4d", 700.0, 710.0, 401.0, 299.0))  # PINHOLE
    with (sparse / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<i7di", 7, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2) + b"a.jpg\x00")
        f.write(struct.pack("<Q", 2) + struct.pack("<ddq", 1.0, 2.0, 5) * 2)
    with (sparse / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", 2))
        for pid, track in ((5, 3), (6, 1)):
            f.write(struct.pack("<Q3d3BdQ", pid, 0.1 * pid, 0.2, 0.3, 10, 20, pid, 0.7, track))
            f.write(struct.pack("<ii", 7, 0) * track)

    model = read_model(sparse)

    simple, pinhole = model.cameras[1], model.cameras[2]
    assert (simple.fx, simple.fy, simple.cx, simple.cy) == (500.0, 500.0, 320.0, 240.0)
    assert (pinhole.fx, pinhole.fy, pinhole.cx, pinhole.cy) == (700.0, 710.0, 401.0, 299.0)
    assert pinhole.scaled(400, 300).fx == pytest.approx(350.0)
    image = model.images[7]
    assert image.name == "a.jpg"
    np.testing.assert_allclose(image.center, [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(model.points, [[0.5, 0.2, 0.3], [0.6, 0.2, 0.3]])
    np.testing.assert_array_equal(model.colors[:, 2], [5, 6])
    np.testing.assert_array_equal(model.track_lengths, [3, 1])


def test_table_frame_and_object_are_recovered() -> None:
    scene = build_table_scene(_synthetic_model(), scale=1 / CAPTURE_SCALE)

    assert len(scene.objects) == 1
    box = scene.objects[0]
    # The table frame's x axis is the table points' principal axis, so the box may come out turned
    # in the plane; its height and footprint area are frame-independent.
    assert box.size[2] == pytest.approx(BOX_SIZE[2], abs=0.01)
    footprint = box.points[:, :2] - box.points[:, :2].mean(axis=0)
    extents = np.sort(np.ptp(footprint @ np.linalg.svd(footprint, full_matrices=False)[2].T, axis=0))
    np.testing.assert_allclose(extents, np.sort(BOX_SIZE[:2]), atol=0.015)
    assert np.linalg.norm(box.center[:2]) == pytest.approx(np.linalg.norm(BOX_CENTER), abs=0.01)
    np.testing.assert_allclose(scene.table_max - scene.table_min, [0.6, 0.6], atol=0.03)


def test_scale_needs_a_floor_or_a_value() -> None:
    with pytest.raises(ValueError, match="pass scale"):
        build_table_scene(_synthetic_model(), table_height=0.75)


def test_connected_clusters_split_objects_that_do_not_touch() -> None:
    a = np.random.default_rng(0).uniform(0, 0.05, (200, 3))
    labels = connected_clusters(np.vstack([a, a + [0.2, 0, 0]]), voxel=0.015)
    assert len(set(labels[:200])) == 1
    assert len(set(labels[200:])) == 1
    assert labels[0] != labels[-1]


def test_written_scene_compiles_with_capture_cameras(tmp_path: Path) -> None:
    model = _synthetic_model()
    scene = build_table_scene(model, scale=1 / CAPTURE_SCALE)

    xml = write_scene(tmp_path, model, scene, camera_count=4, images_dir=tmp_path)

    compiled = mujoco.MjModel.from_xml_path(str(xml))
    cameras = [compiled.camera(i).name for i in range(compiled.ncam)]
    assert cameras[0] == "overview"
    assert len(cameras) == 4
    np.testing.assert_allclose(compiled.cam_resolution[0], [WIDTH, HEIGHT])
    assert "robot_mount" in xml.read_text()
    meta = json.loads((tmp_path / "real2sim.json").read_text())
    assert meta["transform"]["scale"] == pytest.approx(1 / CAPTURE_SCALE)
    assert meta["images_dir"] == str(tmp_path.resolve())
    # The exported camera sits where the capture camera was, in the table frame.
    for cam in meta["cameras"]:
        assert np.linalg.norm(cam["position"][:2]) == pytest.approx(0.6, abs=0.01)
        assert cam["position"][2] == pytest.approx(0.4, abs=0.01)
    # The box's own points project into the overview camera near the image centre (cameras look at it).
    uvd = project(np.array([[*BOX_CENTER, 0.04]]), meta["cameras"][0])
    assert abs(uvd[0, 0] - WIDTH / 2) < WIDTH / 4
    assert abs(uvd[0, 1] - HEIGHT / 2) < HEIGHT / 4


def test_object_rests_on_the_table(tmp_path: Path) -> None:
    model = _synthetic_model()
    xml = write_scene(tmp_path, model, build_table_scene(model, scale=1 / CAPTURE_SCALE), camera_count=2)
    compiled = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(compiled)
    body = compiled.body("object0").id
    start = compiled.body_pos[body].copy()

    for _ in range(500):
        mujoco.mj_step(compiled, data)

    assert np.linalg.norm(data.xpos[body] - start) < 0.003


class _FakeSegmentationRenderer:
    """Stands in for ``mujoco.Renderer``: returns a fixed geom-id image in segmentation mode."""

    def __init__(self, ids: np.ndarray) -> None:
        self.ids = ids
        self.segmentation = False

    def enable_segmentation_rendering(self) -> None:
        self.segmentation = True

    def disable_segmentation_rendering(self) -> None:
        self.segmentation = False

    def update_scene(self, data: object, camera: str) -> None:
        pass

    def render(self) -> np.ndarray:
        assert self.segmentation
        kinds = np.where(self.ids >= 0, int(mujoco.mjtObj.mjOBJ_GEOM), -1)
        return np.stack([self.ids, kinds], axis=-1)


class _FlatBackground:
    def __init__(self, *, moves_objects: bool = False) -> None:
        self.views: list[appearance.View] = []
        self.moves_objects = moves_objects

    def frame(self, view: appearance.View) -> np.ndarray:
        self.views.append(view)
        return np.full((view.height, view.width, 3), 7, dtype=np.uint8)


def test_compositor_keeps_the_robot_and_moved_objects_from_the_sim() -> None:
    xml = """<mujoco><worldbody>
      <geom name="table" type="box" size="1 1 0.01"/>
      <body name="object0" pos="0 0 0.1"><freejoint/><geom name="object0_geom" type="box" size="0.05 0.05 0.05"/></body>
      <body name="link"><geom name="robot_link" type="sphere" size="0.05" pos="0.5 0 0.5"/></body>
      <camera name="cam" pos="0 -1 0.5" fovy="60"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    table, obj, link = (model.geom(n).id for n in ("table", "object0_geom", "robot_link"))
    ids = np.array([[table, obj, link, -1]])
    frame = np.full((1, 4, 3), 200, dtype=np.uint8)
    background = _FlatBackground()
    compositor = appearance.Compositor(model, background, {table, obj}, {obj: np.array([0.0, 0.0, 0.1])})
    renderer = _FakeSegmentationRenderer(ids)

    out = compositor(renderer, data, "cam", frame)
    np.testing.assert_array_equal(out[0, :, 0], [7, 7, 200, 7])  # only the robot comes from the sim
    assert not renderer.segmentation
    view = background.views[-1]
    assert (view.width, view.height, view.cx) == (4, 1, 2.0)
    np.testing.assert_allclose(view.position, [0, -1, 0.5])

    data.qpos[0] += 0.05  # the object was pushed
    mujoco.mj_forward(model, data)
    out = compositor(renderer, data, "cam", frame)
    np.testing.assert_array_equal(out[0, :, 0], [7, 200, 200, 7])

    # A background that redraws objects (a splat) keeps the pushed object, and is told where it is.
    splat_like = _FlatBackground(moves_objects=True)
    compositor = appearance.Compositor(model, splat_like, {table, obj}, {obj: np.array([0.0, 0.0, 0.1])})
    out = compositor(renderer, data, "cam", frame)
    np.testing.assert_array_equal(out[0, :, 0], [7, 7, 200, 7])
    position, _ = splat_like.views[-1].objects["object0"]
    np.testing.assert_allclose(position, [0.05, 0.0, 0.1])


@pytest.mark.slow
def test_capture_camera_intrinsics_match_mujoco_rendering() -> None:
    """MuJoCo's principal point is the offset of the image centre from it; a sign error moves every pixel."""
    if not offscreen_rendering_available():
        pytest.skip("no offscreen rendering")
    width, height, fx, fy, cx, cy = 320, 240, 250.0, 260.0, 150.0, 130.0
    cam = {"position": [0, 0, 0], "xyaxes": [1, 0, 0, 0, 1, 0], "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    point = np.array([0.1, 0.05, -1.0])
    xml = f"""<mujoco><visual><global offwidth="{width}" offheight="{height}"/></visual><worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="sphere" size="0.006" pos="{point[0]} {point[1]} {point[2]}" rgba="1 0 0 1"/>
      <camera name="c" resolution="{width} {height}" focalpixel="{fx} {fy}"
        principalpixel="{width / 2 - cx} {height / 2 - cy}" sensorsize="0.0032 0.0024"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height, width) as renderer:
        renderer.update_scene(data, "c")
        image = renderer.render()
    rows, cols = np.nonzero((image[:, :, 0] > 150) & (image[:, :, 1] < 80))

    expected = project(point[None], cam)[0]
    assert cols.mean() + 0.5 == pytest.approx(expected[0], abs=1.0)
    assert rows.mean() + 0.5 == pytest.approx(expected[1], abs=1.0)


def test_capture_streams_keep_the_capture_aspect(tmp_path: Path) -> None:
    from physicalai_mujoco_plugin.cameras import CameraConfig  # noqa: PLC0415

    meta = tmp_path / "real2sim.json"
    meta.write_text(json.dumps({"cameras": [{"name": "overview", "width": 779, "height": 520}]}))

    streams = appearance.capture_streams(meta, [CameraConfig("wrist"), CameraConfig("overview", fps=15)])

    assert (streams[0].width, streams[0].height) == (640, 480)
    assert (streams[1].width, streams[1].height, streams[1].fps) == (640, 427, 15)


def test_splat_transform_moves_covariances_with_the_frame(tmp_path: Path) -> None:
    from physicalai_mujoco_plugin.real2sim.geometry import Similarity  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.splats import Splats, transform_splats  # noqa: PLC0415

    rng = np.random.default_rng(3)
    quats = rng.normal(size=(5, 4))
    splats = Splats(
        centers=rng.normal(size=(5, 3)),
        quats=quats / np.linalg.norm(quats, axis=1, keepdims=True),
        scales=rng.uniform(0.01, 0.1, (5, 3)),
        colors=rng.uniform(0, 1, (5, 3)),
        opacities=rng.uniform(0, 1, 5),
    )
    rotation, offset = _capture_frame()
    transform = Similarity(rotation, offset, 0.4)

    moved = transform_splats(splats, transform)

    np.testing.assert_allclose(moved.centers, transform.apply(splats.centers))
    expected = 0.4**2 * rotation @ splats.covariances() @ rotation.T
    np.testing.assert_allclose(moved.covariances(), expected, atol=1e-12)
    moved.save(tmp_path / "s.npz")
    np.testing.assert_allclose(Splats.load(tmp_path / "s.npz").quats, moved.quats)


def test_read_ply_activates_3dgs_fields(tmp_path: Path) -> None:
    from physicalai_mujoco_plugin.real2sim.splats import SH_C0, read_ply  # noqa: PLC0415

    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"]
    fields += ["rot_0", "rot_1", "rot_2", "rot_3"]
    values = np.array([[1, 2, 3, 0, 1 / SH_C0 * 0.25, -1, 0, np.log(0.02), np.log(0.03), np.log(0.04), 2, 0, 0, 0]])
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
    header += "".join(f"property float {name}\n" for name in fields) + "end_header\n"
    path = tmp_path / "point_cloud.ply"
    path.write_bytes(header.encode() + values.astype("<f4").tobytes())

    splats = read_ply(path)

    np.testing.assert_allclose(splats.centers, [[1, 2, 3]])
    np.testing.assert_allclose(splats.colors, [[0.5, 0.75, 0.5 - SH_C0]], atol=1e-6)
    np.testing.assert_allclose(splats.opacities, [0.5])
    np.testing.assert_allclose(splats.scales, [[0.02, 0.03, 0.04]], rtol=1e-6)
    np.testing.assert_allclose(splats.quats, [[1, 0, 0, 0]])


def test_camera_view_matches_the_capture_intrinsics(tmp_path: Path) -> None:
    model = _synthetic_model()
    xml = write_scene(tmp_path, model, build_table_scene(model, scale=1 / CAPTURE_SCALE), camera_count=2)
    compiled = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(compiled)
    mujoco.mj_forward(compiled, data)
    cam = json.loads((tmp_path / "real2sim.json").read_text())["cameras"][0]

    view = appearance.camera_view(compiled, data, compiled.camera("overview").id, WIDTH // 2, HEIGHT // 2)

    assert (view.fx, view.fy) == pytest.approx((cam["fx"] / 2, cam["fy"] / 2))
    assert (view.cx, view.cy) == pytest.approx((cam["cx"] / 2, cam["cy"] / 2))
    np.testing.assert_allclose(view.position, cam["position"], atol=1e-6)  # XML keeps 6 digits
    np.testing.assert_allclose(view.rotation[:, 0], cam["xyaxes"][:3], atol=1e-5)
    np.testing.assert_allclose(view.rotation[:, 1], cam["xyaxes"][3:], atol=1e-5)


def test_splat_background_is_skipped_without_gsplat() -> None:
    try:
        import gsplat  # noqa: F401, PLC0415
        import torch  # noqa: PLC0415
    except ImportError:
        pass
    else:
        if torch.cuda.is_available():
            pytest.skip("gsplat and a GPU are present")
    meta = {
        "splat_ply": "missing.ply",
        "transform": {"rotation": np.eye(3).tolist(), "origin": [0, 0, 0], "scale": 1.0},
    }

    assert appearance._splat_background(meta) is None
    assert appearance._splat_background({"transform": meta["transform"]}) is None


MARKER_SIZE = 0.12
MARKER_CENTER = np.array([-0.12, 0.10, 0.0])
MARKER_YAW = math.radians(30.0)


def _marker_photos(model: SparseModel, out: Path) -> None:
    """Draw ArUco 7 lying on the synthetic table into every capture photo (by homography)."""
    import cv2  # noqa: PLC0415

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    pixels, border = 200, 40
    canvas = np.full((pixels + 2 * border,) * 2, 255, np.uint8)
    canvas[border:-border, border:-border] = cv2.aruco.generateImageMarker(dictionary, 7, pixels)
    metres = MARKER_SIZE / pixels
    middle = (pixels + 2 * border - 1) / 2
    cos, sin = math.cos(MARKER_YAW), math.sin(MARKER_YAW)

    def on_table(a: float, b: float) -> np.ndarray:  # canvas index coords -> table frame (image y down = marker -y)
        local = np.array([(a - middle) * metres, -(b - middle) * metres])
        return MARKER_CENTER + np.array([cos * local[0] - sin * local[1], sin * local[0] + cos * local[1], 0.0])

    src = np.array([[0, 0], [len(canvas) - 1, 0], [len(canvas) - 1, len(canvas) - 1], [0, len(canvas) - 1]], np.float32)
    world = _to_capture(np.array([on_table(*p) for p in src]))
    camera = model.cameras[1]
    intrinsics = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]])
    for image in model.images.values():  # photos at the camera's own resolution
        projected = (intrinsics @ (image.rotation @ world.T + image.translation[:, None])).T
        dst = (projected[:, :2] / projected[:, 2:] - 0.5).astype(np.float32)  # COLMAP pixel centres -> OpenCV
        homography = cv2.getPerspectiveTransform(src, dst)
        size = (camera.width, camera.height)
        photo = cv2.warpPerspective(canvas, homography, size, borderValue=110, flags=cv2.INTER_LINEAR)
        cv2.imwrite(str(out / image.name), photo)


def test_marker_gives_scale_and_robot_mount(tmp_path: Path) -> None:
    from physicalai_mujoco_plugin.real2sim.markers import find_marker, mount_from_marker  # noqa: PLC0415

    model = _synthetic_model()
    # Phone photos are large; a marker only ~40 px wide biases its corners inwards by ~1%.
    model = dataclasses.replace(model, cameras={1: model.cameras[1].scaled(3 * WIDTH, 3 * HEIGHT)})
    _marker_photos(model, tmp_path)

    marker = find_marker(tmp_path, model, 7)

    assert marker.views >= 8
    assert marker.residual_px < 1.0
    scale = MARKER_SIZE / marker.side
    assert scale == pytest.approx(1 / CAPTURE_SCALE, rel=0.005)
    scene = build_table_scene(model, scale=scale)
    # The table frame's own x axis is arbitrary (principal axis); compare in the marker's terms instead.
    center = scene.transform.apply(marker.center[None])[0]
    assert center[2] == pytest.approx(0.0, abs=0.003)
    mount, yaw = mount_from_marker(marker, scene.transform, (0.0, -0.25), 90.0)
    assert np.linalg.norm(np.array(mount[:2]) - center[:2]) == pytest.approx(0.25, abs=0.005)
    x_axis = scene.transform.apply_rotation(marker.axes())[:, 0]
    marker_yaw = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    assert yaw == pytest.approx(marker_yaw + 90.0)


def test_extract_frames_keeps_the_sharpest_of_each_stretch(tmp_path: Path) -> None:
    import cv2  # noqa: PLC0415

    from physicalai_mujoco_plugin.real2sim.sfm import extract_frames, sharpness  # noqa: PLC0415

    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    video = tmp_path / "orbit.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    for i in range(30):
        writer.write(sharp if i % 3 == 1 else blurred)  # one sharp frame in every stretch of three
    writer.release()

    frames = extract_frames(video, tmp_path / "frames", count=10)

    assert len(frames) == 10
    reference = sharpness(cv2.cvtColor(cv2.imread(str(frames[0])), cv2.COLOR_BGR2GRAY))
    assert reference > 2 * sharpness(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))
    assert all(f.name == f"frame_{i:04d}.jpg" for i, f in enumerate(frames))


@pytest.mark.slow
def test_clean_photos_inpaints_the_robot(tmp_path: Path) -> None:
    import cv2  # noqa: PLC0415

    from physicalai_mujoco_plugin.real2sim.robot_mask import clean_photos  # noqa: PLC0415

    if not offscreen_rendering_available():
        pytest.skip("no offscreen rendering")
    model = _synthetic_model()
    photos = tmp_path / "photos"
    photos.mkdir()
    for image in model.images.values():
        cv2.imwrite(str(photos / image.name), np.full((HEIGHT, WIDTH, 3), 90, np.uint8))
    scene_dir = tmp_path / "scene"
    xml = write_scene(
        scene_dir, model, build_table_scene(model, scale=1 / CAPTURE_SCALE), camera_count=3, images_dir=photos
    )
    spec = mujoco.MjSpec.from_file(str(xml))
    robot = spec.worldbody.add_body(name="robot", pos=[0.0, 0.0, 0.25])
    robot.add_geom(name="robot_link", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0])
    compiled = spec.compile()
    data = mujoco.MjData(compiled)
    meta = json.loads((scene_dir / "real2sim.json").read_text())
    for cam in meta["cameras"]:  # the "real" robot: a white disc where the sphere projects
        photo = cv2.imread(str(photos / cam["image"]))
        u, v, depth = project(np.array([[0.0, 0.0, 0.25]]), cam)[0]
        cv2.circle(photo, (int(u), int(v)), int(cam["fx"] * 0.05 / depth), (255, 255, 255), -1)
        cv2.imwrite(str(photos / cam["image"]), photo)

    out = clean_photos(compiled, data, scene_dir / "real2sim.json", tmp_path / "clean", dilate_px=4)

    for cam in meta["cameras"]:
        assert cv2.imread(str(out / cam["image"])).max() < 200  # the white robot is gone
        assert (out / "masks" / f"{cam['image'].rsplit('.', 1)[0]}.png").exists()
    meta = json.loads((scene_dir / "real2sim.json").read_text())
    assert meta["images_dir"] == str(out.resolve())
    assert meta["raw_images_dir"] == str(photos)


def test_capture_camera_matrices_project_like_the_sim_camera() -> None:
    """The splat renders in the capture frame; its camera must see what the sim camera sees."""
    model = _synthetic_model()
    scene = build_table_scene(model, scale=1 / CAPTURE_SCALE)
    image = model.images[3]
    cam = dataclasses.asdict(
        __import__("physicalai_mujoco_plugin.real2sim.scene_builder", fromlist=["capture_camera"]).capture_camera(
            "c", image, model.cameras[1], scene
        )
    )
    axes = np.asarray(cam["xyaxes"])
    view = appearance.View(
        position=np.asarray(cam["position"]),
        rotation=np.column_stack([axes[:3], axes[3:], np.cross(axes[:3], axes[3:])]),
        fx=cam["fx"],
        fy=cam["fy"],
        cx=cam["cx"],
        cy=cam["cy"],
        width=WIDTH,
        height=HEIGHT,
    )
    capture_points = _to_capture(np.array([[0.05, -0.03, 0.04], [0.2, 0.1, 0.0], [-0.1, 0.2, 0.06]]))
    table_points = scene.transform.apply(capture_points)  # the recovered frame, not the synthetic one

    world_to_cam, intrinsics = appearance.capture_camera_matrices(view, scene.transform)

    in_cam = (world_to_cam[:3, :3] @ capture_points.T).T + world_to_cam[:3, 3]
    pixels = (intrinsics @ in_cam.T).T
    pixels = pixels[:, :2] / pixels[:, 2:]
    np.testing.assert_allclose(pixels, project(table_points, cam)[:, :2], atol=1e-6)
    # ...and match the COLMAP camera the photo was taken with.
    colmap = (image.rotation @ capture_points.T).T + image.translation
    colmap_pixels = colmap[:, :2] / colmap[:, 2:] * FOCAL + [WIDTH / 2, HEIGHT / 2]
    np.testing.assert_allclose(pixels, colmap_pixels, atol=0.05)


def test_object_motion_in_capture_moves_points_like_the_body() -> None:
    from physicalai_mujoco_plugin.real2sim.geometry import Similarity  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.splats import quats_to_matrices  # noqa: PLC0415

    rotation, offset = _capture_frame()
    # The capture frame is CAPTURE_SCALE * metric @ rotation.T + offset; its inverse is the scene transform.
    transform = Similarity(rotation.T, offset, 1 / CAPTURE_SCALE)
    captured = np.array([0.05, -0.03, 0.0])
    yaw = math.radians(25.0)
    body_rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    body_position = np.array([0.15, 0.02, 0.0])
    on_object = np.array([[0.06, -0.01, 0.03], [0.02, -0.05, 0.07]])  # scene frame, at capture

    motion, shift = appearance.object_motion_in_capture(transform, captured, body_position, body_rotation)

    moved_capture = _to_capture(on_object) @ motion.T + shift
    expected_scene = (on_object - captured) @ body_rotation.T + body_position
    np.testing.assert_allclose(transform.apply(moved_capture), expected_scene, atol=1e-12)
    # Orientations turn with it: a Gaussian's axes in the scene frame are rotated by the body rotation.
    turned = transform.apply_rotation(motion @ quats_to_matrices(np.array([[1.0, 0, 0, 0]]))[0])
    np.testing.assert_allclose(turned, body_rotation @ transform.apply_rotation(np.eye(3)), atol=1e-12)


def test_object_members_are_the_gaussians_in_its_box() -> None:
    means = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 0.001], [0.3, 0.0, 0.05], [0.06, 0.0, 0.1]])
    inside = appearance.object_members(means, np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.12]))
    np.testing.assert_array_equal(inside, [True, False, False, True])


def test_splat_background_carries_moved_object_gaussians() -> None:
    torch = pytest.importorskip("torch")
    from physicalai_mujoco_plugin.real2sim.geometry import Similarity  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.splats import quats_to_matrices  # noqa: PLC0415

    rotation, offset = _capture_frame()
    transform = Similarity(rotation.T, offset, 1 / CAPTURE_SCALE)
    scene_means = np.array([[0.0, 0.0, 0.05], [0.5, 0.5, 0.05]])  # one on the object, one elsewhere
    background = object.__new__(appearance.SplatBackground)  # no gsplat needed to pose Gaussians
    background._torch = torch
    background._device = "cpu"
    background._transform = transform
    background._means = torch.as_tensor(_to_capture(scene_means), dtype=torch.float32)
    background._quats = torch.as_tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    background._members = {"object0": (np.zeros(3), torch.as_tensor([0]))}
    yaw = math.radians(90.0)
    turn = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    view = appearance.View(np.zeros(3), np.eye(3), 1, 1, 0, 0, 1, 1, objects={"object0": (np.array([0.1, 0, 0]), turn)})

    means, quats = background._posed(view)

    moved = transform.apply(means.numpy().astype(np.float64))
    np.testing.assert_allclose(moved, [[0.1, 0.0, 0.05], [0.5, 0.5, 0.05]], atol=1e-5)
    np.testing.assert_allclose(background._means.numpy(), _to_capture(scene_means), atol=1e-5)  # loaded copy kept
    turned = transform.apply_rotation(quats_to_matrices(quats.numpy().astype(np.float64))[0])
    np.testing.assert_allclose(turned, turn @ transform.apply_rotation(np.eye(3)), atol=1e-5)
