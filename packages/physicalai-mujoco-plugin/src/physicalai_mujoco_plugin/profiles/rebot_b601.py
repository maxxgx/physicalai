# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""reBot B601 profile: Menagerie's ``seeed_rebot_devarm`` in the LeRobot-style joint frame of ``ReBotB601RS``.

PROVISIONAL, for two reasons:

- The frame is the one PR #360 introduces (issue #343): every RS direction -1, the gripper -270
  (open) to 0 (closed) in degrees, no x6 scaling. Until #360 merges, the B601 plugin on ``main``
  still uses the previous frame (directions +1, +1, -1, -1, -1, +1, gripper /6, 0..45), so this
  profile and that driver disagree in sign on shoulder_pan, shoulder_lift and wrist_roll and in
  the gripper's sign and scale. Data recorded in either frame does not carry over to the other.
- No real arm has confirmed the mapping yet (design open issue 16.1).

How the mapping is derived:

- Menagerie's DevArm has the link origins and frames of the B601-RS URDF, so both share one zero
  pose: the folded rest pose in which the RS arm is zero-calibrated. Every offset is therefore 0.
- #360's ``00-arm-rs_asm-v3_joint_frame.urdf``, which Studio previews the real arm with, has the
  same joint axes as Menagerie (joint1 +z, joint2 -z, joint3 to joint6 +z), so every arm joint
  maps one to one (``scale`` 1); both go negative from the fold on shoulder_lift and elbow_flex.
- The driver clips actions to its public limits (``REBOT_B601_RS_JOINT_LIMITS_DEG`` in #360),
  which are tighter than Menagerie's ranges on shoulder_pan (145 vs 160 degrees), shoulder_lift
  (-170 vs -180), wrist_flex (-90..80 vs -103..97) and wrist_roll (90 vs 180). Each channel's
  ``range`` is the overlap of the two, so the simulation stops where the real arm stops. The one
  exception is elbow_flex: the driver allows -200 degrees, the model only -180 (its geometry), so
  the simulation stops 20 degrees short there.
- Menagerie's two fingers slide 0..0.05 m each from closed to open, coupled 1:1. #360's URDF
  drives them from a ``gripper_drive`` joint of -270 degrees (open) to 0, with the first finger at
  0.05 m when fully open. The real gripper does not open that far: measured on the real arm, it
  opens 85 mm at -270 and 5.4 mm at 0 (:data:`REBOT_B601_GRIPPER_SCALE`). The gripper channel drives
  both fingers through that measured mapping, each finger 2.7 to 43.7 mm (to the readings' -278
  overshoot), velocities likewise.

The wrist camera is an Intel RealSense D405 on Seeed's printed mount (reBot-DevArm
``hardware/reBot_B601_DM/3D_Printed_Parts/D405_305_Mount.step``). Its clamp (57.3 mm bore) sits on
the 57-60 mm neck of the gripper housing, after the wrist-roll joint, so the camera rolls with the
gripper. The mount holds the 42 mm camera 104 mm off the roll axis, tilted 30 degrees toward it, so
the view centre crosses the axis just past the finger tips. :data:`REBOT_B601_WRIST_CAMERA` is
that pose in ``link6``, taken from the STEP geometry. Which side of the gripper the mount faces is
symmetric in the model and assumed (``link6`` +x); check it on the arm with the joint signs.
"""

from __future__ import annotations

import json
import os
from math import radians
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from physicalai_mujoco_plugin.profiles._types import CameraMount, CameraSpec, ChannelOverride, EndEffector, RobotProfile

if TYPE_CHECKING:
    import mujoco

_GRIPPER_MAPS = {
    # name: (opening at 0 deg, opening per degree (m), the most open reading the finger range covers (deg))
    # r12 (real2sim v2, 2026-11-07): through camera B, which turned out to compensate a too-narrow map (kept to compare)
    "r12": (0.00464, 0.0002425, 270.0),
    # r16 (real2sim v2 D3a, 2026-11-08): the user's measurement fixes the opening at -270 deg to 85 mm (review NOTES.md,
    # "User statements, 2026-11-08"); the slope 0.14749 mm/deg per finger (the BOM's module-1 16-tooth pinion: 0.1396,
    # +5.65 %) from the joint fit with the wrist camera (TRAIN wrist frames), chosen on the DEV closed-on-object holds
    # (bias +0.5 mm; the BOM slope -3.0, a TRAIN-hold slope -5.0) and R-48's 18-21 mm resistor hold (19.8 mm at -49 deg);
    # the range covers the readings' -278 deg overshoot (87.4 mm; the MJCF allows 50 mm per finger)
    "r16": (0.085 - 540.0 * 0.00014749, 2.0 * 0.00014749, 278.0),
}
REBOT_B601_GRIPPER_MAP = os.environ.get("PHYSICALAI_REBOT_GRIPPER_MAP", "r16")
"""Which measured reading-to-opening map the profile uses (r16 by default; r12 kept to compare)."""
# the opening between the real fingers at a driver reading of 0 deg (closed) and per driver degree, metres (the reading
# runs -270 open to 0 closed): measured (below)
REBOT_B601_GRIPPER_OPENING_AT_ZERO, REBOT_B601_GRIPPER_OPENING_PER_DEG, _GRIPPER_MOST_OPEN_DEG = _GRIPPER_MAPS[REBOT_B601_GRIPPER_MAP]
REBOT_B601_GRIPPER_SCALE = -2.0 / REBOT_B601_GRIPPER_OPENING_PER_DEG
"""Driver gripper degrees per metre of travel of one finger (each finger moves half the opening).

Measured on the real arm (real2sim v2, the allowed robot episodes of the reBot desk): r16 (2026-11-08) opening =
5.36 mm + 0.295 mm/deg x |reading| (85 mm at -270 deg, the user's measurement; the slope from the gripper camera's
fingers fitted jointly with the camera, checked on closed-on-object holds; :data:`_GRIPPER_MAPS`). The r12 map (4.64 mm
+ 0.2425 mm/deg, 70.1 mm open) was fitted through a camera pose that compensated it. It replaces #360's URDF ratio (-270 deg = 0.05 m per finger, a 100 mm opening the real gripper does not
reach): the reBot plugin's URDF tie is knowingly broken pending an upstream fix (tests/test_twins.py)."""
REBOT_B601_GRIPPER_OFFSET = -REBOT_B601_GRIPPER_SCALE * REBOT_B601_GRIPPER_OPENING_AT_ZERO / 2.0
"""Driver degrees at a finger travel of 0 (the model's fully closed fingers, beyond the real gripper's closed stop)."""
REBOT_B601_FINGER_RANGE = (
    REBOT_B601_GRIPPER_OPENING_AT_ZERO / 2.0,
    (REBOT_B601_GRIPPER_OPENING_AT_ZERO + _GRIPPER_MOST_OPEN_DEG * REBOT_B601_GRIPPER_OPENING_PER_DEG) / 2.0,
)
"""Travel of each finger over the range of the readings (0 to -278 deg, the real readings overshoot the -270 deg command):
2.7 to 43.7 mm (r16 map)."""
REBOT_B601_GRIPPER_OPEN_TRAVEL = (REBOT_B601_GRIPPER_OPENING_AT_ZERO + 270.0 * REBOT_B601_GRIPPER_OPENING_PER_DEG) / 2.0
"""Travel of each finger at the fully open command (-270 deg): 42.5 mm, an 85 mm opening (r16 map)."""

REBOT_B601_JOINT_RANGES_DEG: dict[str, tuple[float, float] | None] = {
    "shoulder_pan": (-145.0, 145.0),
    "shoulder_lift": (-170.0, 0.0),
    "elbow_flex": None,  # the model's -180..0; the driver allows -200
    "wrist_flex": (-90.0, 80.0),
    "wrist_yaw": None,  # the model's -89.95..89.95; the driver allows -90..90
    "wrist_roll": (-90.0, 90.0),
}
"""Arm joint ranges in the driver's public degrees: #360's limits where tighter than Menagerie's, else ``None``."""


def _range(name: str) -> tuple[float, float] | None:
    limits = REBOT_B601_JOINT_RANGES_DEG[name]
    return None if limits is None else (radians(limits[0]), radians(limits[1]))


REBOT_B601_WRIST_MOUNT = Path(__file__).with_name("rebot_b601_wrist_mount.json")
"""The wrist mount from its STEP (the user's print of Seeed's ``D405_305_Mount``; real2sim v2 r15, mount_from_step.py and
build_mount_collision.py): the drawn mesh placed on the CAD gripper's bolt holes (the +x pair of the cnc7 / pla7 stack,
link6 x 30 mm, y +-11 mm; the clamp's 20.4 mm gap on the 20.1 mm stack), its convex collision pieces and planning
proxies, and the Gemini 305's 42 x 42 x 23 mm body on the cradle seat. Brown, as printed."""

REBOT_B601_MOUNT_COLOURS: dict[str, tuple[float, float, float, float]] = {"brown": (0.36, 0.20, 0.12, 1.0), "black": (0.06, 0.06, 0.065, 1.0)}
"""Print colours of the wrist mount; brown is the default (the user's print), black their likely next one."""


def _wrist_mount(colour: str = "brown") -> CameraMount | None:
    if not REBOT_B601_WRIST_MOUNT.exists():
        return None
    data = json.loads(REBOT_B601_WRIST_MOUNT.read_text())
    mount, body = data["mount"], data["camera_body"]
    as3 = lambda pts: tuple(tuple(float(v) for v in p) for p in pts)  # noqa: E731
    return CameraMount(
        vertices=as3(mount["vertices"]),
        faces=tuple(tuple(int(i) for i in f) for f in mount["faces"]),
        rgba=REBOT_B601_MOUNT_COLOURS[colour],
        pieces=tuple(as3(p) for p in mount["pieces"]),
        proxies=tuple(as3(p) for p in mount["proxies"]),
        body_pos=tuple(body["pos"]),
        body_quat=tuple(body["quat_wxyz"]),
        body_half=tuple(body["half"]),
        body_rgba=tuple(body["rgba"]),
    )


REBOT_B601_WRIST_CAMERA = CameraSpec(
    name="wrist",
    body="link6",
    # The D405's front face centre: 104 mm off the roll axis (link6 +z), 39 mm along it. The camera
    # looks along (-0.5, 0, 0.866) in link6, toward the axis and the fingers, with image up along
    # (0.866, 0, 0.5), away from the gripper; its x axis is link6 +y.
    # Measured on the real arm (phone scan, the lens midpoint triangulated in three views, 2-5 px reprojection): 21 mm
    # further along the roll axis than the mount STEP's (0.10414, 0, 0.03915).
    pos=(0.1017, 0.0031, 0.0602),
    quat=(0.18301270189221933, 0.6830127018922193, 0.6830127018922193, 0.18301270189221933),
    fovy=58.0,  # the D405's vertical field of view
    housing=(0.021, 0.021, 0.0115),  # the D405's 42 x 42 x 23 mm body
    # The printed brown shroud and bracket around the camera, measured on the real arm: the convex hull of its
    # Gaussians in a phone scan (link6 frame, outliers beyond 1.8 sigma left out; about 10 x 6 x 5 cm).
    mount_hull=(
        (
            (0.0341, 0.0125, 0.0860),
            (0.0114, -0.0223, 0.0723),
            (0.1008, 0.0213, 0.0482),
            (0.0288, -0.0139, 0.0838),
            (0.0533, 0.0194, 0.0453),
            (0.0048, 0.0379, 0.0807),
            (0.0413, -0.0170, 0.0540),
            (0.0356, -0.0174, 0.0775),
            (0.0497, 0.0053, 0.0446),
            (0.0007, -0.0194, 0.0802),
            (0.0131, -0.0191, 0.0687),
            (0.0111, 0.0301, 0.0895),
            (0.0000, 0.0316, 0.0720),
            (-0.0072, 0.0227, 0.0796),
            (0.0004, 0.0375, 0.0784),
            (0.0364, -0.0097, 0.0832),
            (0.0026, 0.0218, 0.0909),
            (-0.0065, 0.0360, 0.0856),
            (0.1008, 0.0191, 0.0528),
            (0.0121, 0.0253, 0.0910),
            (0.1120, 0.0171, 0.0421),
            (0.0101, -0.0177, 0.0699),
            (0.0953, 0.0234, 0.0399),
            (0.0794, 0.0148, 0.0383),
            (0.0145, 0.0167, 0.0925),
            (0.0652, -0.0084, 0.0467),
            (0.0115, -0.0155, 0.0858),
            (0.0030, 0.0271, 0.0684),
            (0.0809, 0.0191, 0.0402),
            (0.0105, -0.0220, 0.0809),
            (0.0119, 0.0392, 0.0773),
            (0.0082, -0.0216, 0.0741),
            (0.1054, -0.0009, 0.0404),
            (0.0245, -0.0156, 0.0834),
            (0.0098, -0.0222, 0.0807),
        ),
        (0.36, 0.20, 0.12, 1.0),
    ),
)
if (_mount := _wrist_mount()) is not None:
    from dataclasses import replace as _replace

    REBOT_B601_WRIST_CAMERA = _replace(REBOT_B601_WRIST_CAMERA, housing=None, mount_hull=None, mount_mesh=_mount)
"""The wrist camera on Seeed's ``D405_305_Mount``: the pose above (a kit may override it); the bracket and the camera's
body from the mount's STEP (:data:`REBOT_B601_WRIST_MOUNT`) when that file is present, else the scan hull above."""

REBOT_B601_JOINT_ZERO_ENV = "PHYSICALAI_REBOT_B601_JOINT_ZERO"
"""Environment key naming a JSON file with one arm unit's joint zeros (degrees, model = reading + zero), either the
mapping itself or under ``rebot_b601_joint_zero_deg`` (a real2sim kit). Unset (the default): every zero is 0 and the
profile maps readings to the model as before. These are a calibration of one physical arm, not of the B601 model."""


def _joint_zero_deg() -> dict[str, float]:
    path = os.environ.get(REBOT_B601_JOINT_ZERO_ENV)
    if not path:
        return {}
    data = json.loads(Path(path).expanduser().read_text())
    data = data.get("rebot_b601_joint_zero_deg", data)
    return {name: float(data[name]) for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll") if name in data}


REBOT_B601_JOINT_ZERO_DEG: dict[str, float] = _joint_zero_deg()
"""This process's joint zeros (:data:`REBOT_B601_JOINT_ZERO_ENV`, read at import); empty by default. The reBot desk
v2 kit holds those measured on its arm (real2sim v2 r13, fit_arm_both_views.py)."""

REBOT_B601_TIGHT_COLLISION = Path(__file__).with_name("rebot_b601_collision.json")
"""Convex pieces of the fingers' and the gripper housing's DRAWN CAD meshes (real2sim r13, build_collision_r13.py):
their union contains the drawn surface and reaches beyond it by about 2 mm at most (Menagerie's collision meshes reach
up to 4.5 mm past the fingers and 15 mm past the housing), so what the cameras show touching is what collides."""


PLAN_PROXY_GROUP = 5
"""Geom group of the planning-only proxies (``<body>_plan<i>``): never drawn, never in contact."""


def _solid_points(pts: list[list[float]], min_thick: float = 0.0003) -> np.ndarray:
    """A piece's points, given a thickness of at least *min_thick* along their least principal axis (MuJoCo cannot hull
    coplanar points): a flat piece gets a copy moved by the missing thickness."""
    p = np.asarray(pts, dtype=float); c = p.mean(axis=0)
    normal = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    thick = float(np.ptp((p - c) @ normal))
    return p if thick >= min_thick else np.vstack([p, p + (min_thick - thick) * normal])


def _tight_gripper_collision(spec: mujoco.MjSpec) -> None:
    """Replace the fingers' and the gripper housing's collision meshes by :data:`REBOT_B601_TIGHT_COLLISION`.

    The new geoms copy the old ones' contact parameters (class, group, contype/conaffinity, condim, friction, solref,
    solimp); the bodies' inertia is explicit, so masses do not change. A missing file leaves Menagerie's meshes.
    """
    import mujoco  # noqa: PLC0415

    if not REBOT_B601_TIGHT_COLLISION.exists():
        return
    data = json.loads(REBOT_B601_TIGHT_COLLISION.read_text())
    for body_name, entry in data["bodies"].items():
        body = spec.body(body_name)
        old = [g for g in body.geoms if g.contype or g.conaffinity]
        if body is None or not old:
            continue
        t = old[0]
        params = {"group": t.group, "contype": t.contype, "conaffinity": t.conaffinity, "condim": t.condim,
                  "friction": list(t.friction), "solref": list(t.solref), "solimp": list(t.solimp), "priority": t.priority,
                  "margin": t.margin, "gap": t.gap, "rgba": list(t.rgba)}
        for g in old:
            spec.delete(g)
        for i, pts in enumerate(entry["pieces"]):
            mesh = spec.add_mesh(name=f"{body_name}_tight{i}", uservert=_solid_points(pts).ravel().tolist())
            body.add_geom(name=f"{body_name}_tight{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, **params)
        # r14: planning-only proxies (each the hull of whole pieces, so they contain them): no contact, not drawn
        # (group PLAN_PROXY_GROUP, transparent, massless); the bucket demonstrator's clearance checks use them
        for i, pts in enumerate(entry.get("proxies", [])):
            mesh = spec.add_mesh(name=f"{body_name}_plan{i}", uservert=_solid_points(pts).ravel().tolist())
            body.add_geom(name=f"{body_name}_plan{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, contype=0, conaffinity=0,
                          group=PLAN_PROXY_GROUP, rgba=[0.0, 0.0, 0.0, 0.0], density=0.0)


REBOT_B601_PROFILE = RobotProfile(
    name="rebot_b601",
    display_name="reBot B601",
    menagerie_model="seeed_rebot_devarm",
    menagerie_entry="seeed_rebot_devarm",
    tier="twin",
    # Names and order of REBOT_B601_RS_JOINT_ORDER, in #360's frame; provisional (see the module docstring).
    provisional=True,
    channels=(
        ChannelOverride(name="shoulder_pan", actuators=("joint1",), range=_range("shoulder_pan"), offset=-REBOT_B601_JOINT_ZERO_DEG.get("shoulder_pan", 0.0)),
        ChannelOverride(name="shoulder_lift", actuators=("joint2",), range=_range("shoulder_lift"), offset=-REBOT_B601_JOINT_ZERO_DEG.get("shoulder_lift", 0.0)),
        ChannelOverride(name="elbow_flex", actuators=("joint3",), range=_range("elbow_flex"), offset=-REBOT_B601_JOINT_ZERO_DEG.get("elbow_flex", 0.0)),
        ChannelOverride(name="wrist_flex", actuators=("joint4",), range=_range("wrist_flex"), offset=-REBOT_B601_JOINT_ZERO_DEG.get("wrist_flex", 0.0)),
        ChannelOverride(name="wrist_yaw", actuators=("joint5",), range=_range("wrist_yaw"), offset=-REBOT_B601_JOINT_ZERO_DEG.get("wrist_yaw", 0.0)),
        ChannelOverride(name="wrist_roll", actuators=("joint6",), range=_range("wrist_roll"), offset=-REBOT_B601_JOINT_ZERO_DEG.get("wrist_roll", 0.0)),
        # Both fingers, so neither actuator fights the model's finger equality. Like every RS
        # channel, the driver reports the gripper in degrees (-270 open to 0 closed), so the unit
        # is pinned to degrees: the slide's metres stay unconverted and the scale turns them into
        # driver degrees, also under unit="normalized".
        ChannelOverride(
            name="gripper",
            actuators=("joint_left", "joint_right"),
            unit="degrees",
            scale=REBOT_B601_GRIPPER_SCALE,
            offset=REBOT_B601_GRIPPER_OFFSET,
            member_scales=(1.0,),
            range=REBOT_B601_FINGER_RANGE,
        ),
    ),
    default_unit="degrees",
    cameras=(REBOT_B601_WRIST_CAMERA,),
    default_scene="single_pick_place",
    # The model has no tool site: the point between the finger tips, on the roll axis (link6 +z).
    end_effectors=(EndEffector(body="link6", pos=(0.0, 0.0, 0.163)),),
    reach=0.544,
    customize=_tight_gripper_collision,
)

__all__ = [
    "PLAN_PROXY_GROUP",
    "REBOT_B601_JOINT_ZERO_DEG",
    "REBOT_B601_JOINT_ZERO_ENV",
    "REBOT_B601_TIGHT_COLLISION",
    "REBOT_B601_FINGER_RANGE",
    "REBOT_B601_GRIPPER_OFFSET",
    "REBOT_B601_GRIPPER_OPENING_AT_ZERO",
    "REBOT_B601_GRIPPER_OPEN_TRAVEL",
    "REBOT_B601_GRIPPER_OPENING_PER_DEG",
    "REBOT_B601_GRIPPER_SCALE",
    "REBOT_B601_JOINT_RANGES_DEG",
    "REBOT_B601_PROFILE",
    "REBOT_B601_WRIST_CAMERA",
]
