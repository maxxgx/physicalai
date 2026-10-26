# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``python -m physicalai_mujoco_plugin.real2sim``: build a captured scene, then check it against the photos.

``frames`` samples a phone video, ``sfm`` poses the photos (COLMAP, via pycolmap),
``build`` turns a COLMAP reconstruction and its photos into ``scene.xml`` + ``real2sim.json``;
``gate`` writes photo | sim | blend sheets for every capture camera and prints how well each object
lines up; ``splat`` adds a Gaussian splat trained on the same capture (cameras without a photo, such
as the wrist, then render it on a CUDA host). Run the result with::

    physicalai-mujoco start --profile trossen_wxai --model <out>/scene.xml --appearance <out>/real2sim.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _parse_triplet(text: str) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in text.split(","))
    return x, y, z


def _build(args: argparse.Namespace) -> int:
    from physicalai_mujoco_plugin.real2sim.colmap import read_model  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.geometry import build_table_scene  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.scene_builder import write_scene  # noqa: PLC0415

    model = read_model(Path(args.sparse))
    images = Path(args.images)
    first = next(iter(model.images.values()))
    photo = cv2.imread(str(images / first.name))
    if photo is None:
        sys.stderr.write(f"cannot read {images / first.name}\n")
        return 2
    height, width = photo.shape[:2]
    model = dataclasses.replace(model, cameras={k: c.scaled(width, height) for k, c in model.cameras.items()})
    scale_value, marker = args.scale, None
    if args.marker_size is not None:
        from physicalai_mujoco_plugin.real2sim.markers import find_marker  # noqa: PLC0415

        marker = find_marker(images, model, args.marker_id, args.marker_dict)
        scale_value = args.marker_size / marker.side
        sys.stdout.write(
            f"marker {marker.marker_id}: {marker.views} photos, residual {marker.residual_px:.2f} px, "
            f"scale {scale_value:.4f} m/unit\n"
        )
    scene = build_table_scene(model, scale=scale_value, table_height=args.table_height)
    mount, yaw = args.mount, args.yaw
    if marker is not None and args.mount_from_marker is not None:
        from physicalai_mujoco_plugin.real2sim.markers import mount_from_marker  # noqa: PLC0415

        dx, dy, dyaw = args.mount_from_marker
        mount, yaw = mount_from_marker(marker, scene.transform, (dx, dy), dyaw)
        sys.stdout.write(f"robot_mount at {np.round(mount, 3)} yaw {yaw:.1f} deg (from the marker)\n")
    xml = write_scene(
        Path(args.out),
        model,
        scene,
        camera_count=args.cameras,
        images_dir=images,
        name=args.name,
        mount=mount,
        mount_yaw_deg=yaw,
        free_objects=not args.static_objects,
    )
    sys.stdout.write(
        f"scale {scene.transform.scale:.4f} m/unit; table {np.round(scene.table_max - scene.table_min, 3)} m\n"
    )
    for obj in scene.objects:
        sys.stdout.write(
            f"{obj.name}: {len(obj.points)} points, size {np.round(obj.size, 3)} m at {np.round(obj.center[:2], 3)}\n"
        )
    sys.stdout.write(f"{xml}\n")
    return 0


def _gate(args: argparse.Namespace) -> int:
    import mujoco  # noqa: PLC0415

    from physicalai_mujoco_plugin.real2sim.gate import run_gate  # noqa: PLC0415

    scene_dir = Path(args.scene)
    meta = json.loads((scene_dir / "real2sim.json").read_text(encoding="utf-8"))
    points = np.load(scene_dir / "points.npz")
    objects = {key.removesuffix("_points"): points[key] for key in points.files if key.endswith("_points")}
    model = mujoco.MjModel.from_xml_path(str(scene_dir / "scene.xml"))
    data = mujoco.MjData(model)
    out = Path(args.out) if args.out else scene_dir / "gate"
    worst = 1.0
    for result in run_gate(model, data, scene_dir / "real2sim.json", Path(meta["images_dir"]), out, objects):
        readings = ", ".join(f"{name} {value:.3f}" for name, value in result.coverage.items())
        sys.stdout.write(f"{result.camera:12s} {result.image:24s} {readings}\n")
        worst = min([worst, *result.coverage.values()])
    sys.stdout.write(f"worst coverage {worst:.3f} (sheets in {out})\n")
    return 0 if worst >= args.min_coverage else 1


def _splat(args: argparse.Namespace) -> int:
    from physicalai_mujoco_plugin.real2sim.geometry import Similarity  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.splats import read_ply, transform_splats  # noqa: PLC0415

    scene_dir = Path(args.scene)
    meta_path = scene_dir / "real2sim.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ply = Path(args.ply).resolve()
    splats = transform_splats(read_ply(ply), Similarity.from_dict(meta["transform"]))
    splats.save(scene_dir / "splats.npz")
    meta["splats"] = "splats.npz"
    meta["splat_ply"] = str(ply)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sys.stdout.write(f"{len(splats)} Gaussians from {ply} -> {scene_dir / 'splats.npz'}\n")
    return 0


def _frames(args: argparse.Namespace) -> int:
    from physicalai_mujoco_plugin.real2sim.sfm import extract_frames  # noqa: PLC0415

    frames = extract_frames(Path(args.video), Path(args.out), count=args.count, max_side=args.max_side)
    sys.stdout.write(f"{len(frames)} frames in {args.out}\n")
    return 0


def _sfm(args: argparse.Namespace) -> int:
    try:
        from physicalai_mujoco_plugin.real2sim.sfm import run_sfm  # noqa: PLC0415

        result = run_sfm(Path(args.images), Path(args.out), matching=args.matching)
    except ImportError:
        sys.stderr.write("sfm needs pycolmap: pip install pycolmap\n")
        return 2
    sys.stdout.write(
        f"registered {result.registered}/{result.total} photos, {result.points} points\n"
        f"  --sparse {result.sparse} --images {result.images}\n"
    )
    return 0


def _clean(args: argparse.Namespace) -> int:
    from physicalai_mujoco_plugin.real2sim.robot_mask import clean_photos  # noqa: PLC0415
    from physicalai_mujoco_plugin.robot import MuJoCoRobot  # noqa: PLC0415

    scene_dir = Path(args.scene)
    robot = MuJoCoRobot(args.profile, model_path=str(scene_dir / "scene.xml"), cameras=[])
    robot.connect()
    try:
        sim = robot._sim  # noqa: SLF001 - the robot is posed directly, without physics
        if sim is None:
            sys.stderr.write("the simulation did not load\n")
            return 2
        if args.joints is not None:
            sim.channels[0].set_positions(np.array([float(v) for v in args.joints.split(",")]))
        out = clean_photos(
            sim.model,
            sim.data,
            scene_dir / "real2sim.json",
            Path(args.out) if args.out else scene_dir / "clean",
            dilate_px=args.dilate,
            on_photo=lambda name, fraction: sys.stdout.write(f"{name:24s} {100 * fraction:5.1f}% inpainted\n"),
        )
    finally:
        robot.disconnect()
    sys.stdout.write(f"cleaned photos in {out}; real2sim.json now uses them\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``build`` or ``gate`` command.

    Returns:
        The process exit code (``gate`` fails below ``--min-coverage``).
    """
    parser = argparse.ArgumentParser(
        prog="python -m physicalai_mujoco_plugin.real2sim", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    frames = sub.add_parser("frames", help="Phone video -> the sharpest frame of each stretch, as JPEGs")
    frames.add_argument("--video", required=True)
    frames.add_argument("--out", required=True, help="Directory for the frames")
    frames.add_argument("--count", type=int, default=150, help="Frames to keep")
    frames.add_argument("--max-side", type=int, default=1600, help="Downscale frames to this longest side")
    frames.set_defaults(run=_frames)

    sfm = sub.add_parser("sfm", help="Photos -> COLMAP poses and points, undistorted to pinhole (needs pycolmap)")
    sfm.add_argument("--images", required=True, help="The photos or frames")
    sfm.add_argument("--out", required=True, help="Work directory; the result is <out>/undistorted/{images,sparse}")
    sfm.add_argument("--matching", choices=["exhaustive", "sequential"], default="exhaustive")
    sfm.set_defaults(run=_sfm)

    build = sub.add_parser("build", help="COLMAP capture -> scene.xml + real2sim.json")
    build.add_argument("--sparse", required=True, help="COLMAP model directory (cameras.bin, images.bin, points3D.bin)")
    build.add_argument("--images", required=True, help="The photos the gate and the appearance use (any resolution)")
    build.add_argument("--out", required=True, help="Output directory")
    scale = build.add_mutually_exclusive_group(required=True)
    scale.add_argument("--scale", type=float, help="Metres per capture unit (from a marker or a measured length)")
    scale.add_argument("--table-height", type=float, help="Table height in metres; needs the floor in the capture")
    scale.add_argument("--marker-size", type=float, help="Side of the ArUco marker on the table, in metres")
    build.add_argument("--marker-id", type=int, default=0, help="ArUco marker id (with --marker-size)")
    build.add_argument("--marker-dict", default="DICT_4X4_50", help="ArUco dictionary name in cv2.aruco")
    build.add_argument(
        "--mount-from-marker",
        type=_parse_triplet,
        default=None,
        help="Robot base x,y (metres) and yaw (degrees) in the marker frame; overrides --mount/--yaw",
    )
    build.add_argument("--mount", type=_parse_triplet, default=(-0.30, 0.0, 0.0), help="robot_mount x,y,z on the table")
    build.add_argument("--yaw", type=float, default=0.0, help="robot_mount yaw in degrees (0 faces +x)")
    build.add_argument("--cameras", type=int, default=8, help="Capture cameras to export, spread around the table")
    build.add_argument("--name", default="real2sim", help="Model name")
    build.add_argument(
        "--static-objects", action="store_true", help="Fix the objects instead of giving them free joints"
    )
    build.set_defaults(run=_build)

    gate = sub.add_parser("gate", help="Sim vs photo sheets per capture camera, with object coverage")
    gate.add_argument("--scene", required=True, help="Directory written by build")
    gate.add_argument("--out", default=None, help="Sheet directory (default: <scene>/gate)")
    gate.add_argument("--min-coverage", type=float, default=0.9, help="Fail below this object coverage")
    gate.set_defaults(run=_gate)

    splat = sub.add_parser("splat", help="Use a trained 3DGS .ply (capture frame) as the scene's appearance")
    splat.add_argument("--scene", required=True, help="Directory written by build")
    splat.add_argument(
        "--ply", required=True, help="Trained splat (gsplat / nerfstudio / brush export) on the same capture"
    )
    splat.set_defaults(run=_splat)

    clean = sub.add_parser("clean", help="Inpaint the real robot out of the capture photos (its pose during capture)")
    clean.add_argument("--scene", required=True, help="Directory written by build")
    clean.add_argument("--profile", required=True, help="The robot profile, as for start")
    clean.add_argument(
        "--joints", default=None, help="Comma-separated joint values in the profile's units (default: home)"
    )
    clean.add_argument("--out", default=None, help="Cleaned photo directory (default: <scene>/clean)")
    clean.add_argument("--dilate", type=int, default=8, help="Grow the robot mask by this many pixels")
    clean.set_defaults(run=_clean)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
