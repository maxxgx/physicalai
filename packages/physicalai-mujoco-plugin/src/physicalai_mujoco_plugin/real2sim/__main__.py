# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``python -m physicalai_mujoco_plugin.real2sim``: build a captured scene, then check it against the photos.

``build`` turns a COLMAP reconstruction and its photos into ``scene.xml`` + ``real2sim.json``;
``gate`` writes photo | sim | blend sheets for every capture camera and prints how well each object
lines up. Run the result with::

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
    scene = build_table_scene(model, scale=args.scale, table_height=args.table_height)
    xml = write_scene(
        Path(args.out),
        model,
        scene,
        camera_count=args.cameras,
        images_dir=images,
        name=args.name,
        mount=args.mount,
        mount_yaw_deg=args.yaw,
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


def main(argv: list[str] | None = None) -> int:
    """Run the ``build`` or ``gate`` command.

    Returns:
        The process exit code (``gate`` fails below ``--min-coverage``).
    """
    parser = argparse.ArgumentParser(
        prog="python -m physicalai_mujoco_plugin.real2sim", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="COLMAP capture -> scene.xml + real2sim.json")
    build.add_argument("--sparse", required=True, help="COLMAP model directory (cameras.bin, images.bin, points3D.bin)")
    build.add_argument("--images", required=True, help="The photos the gate and the appearance use (any resolution)")
    build.add_argument("--out", required=True, help="Output directory")
    scale = build.add_mutually_exclusive_group(required=True)
    scale.add_argument("--scale", type=float, help="Metres per capture unit (from a marker or a measured length)")
    scale.add_argument("--table-height", type=float, help="Table height in metres; needs the floor in the capture")
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

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
