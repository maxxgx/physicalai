# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The booth task scene: two WidowX AI arms, the bin, the red bracket, the INTEL box and the blue cover.

Frame ("booth frame"): metres, origin on the table top midway between the arm bases, +x the arms'
facing, +y the robots' own left, +z up. The scene XML is written by
``~/real2sim-data/scripts/booth_task_scene.py`` from silhouette fits on calibrated video frames and
copied to ``urdf/scenes/booth_task/scene.xml``; the plugin attaches Menagerie's ``wxai_follower`` at
``left_robot_mount`` and ``right_robot_mount``.

Everything here runs before an episode starts: composing the model, the one gripper parameter set
that differs from Menagerie (:func:`tune_model`), the start state, and the robustness perturbations
(:func:`perturb`). Nothing in this module touches a running episode.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from physicalai_mujoco_plugin._urdf import get_urdf_path

if TYPE_CHECKING:
    from pathlib import Path

SCENE_ID = "booth_task"
SIDES = ("left", "right")
ARM_JOINTS = tuple(f"joint_{i}" for i in range(6))
GRIPPER_JOINT = "right_carriage_joint"
"""The driven carriage; the other one follows through the model's joint equality."""
TCP_SITE = "ee_gripper_link"
"""Menagerie's tool site: x is the approach axis, the fingers close along y."""

TIMESTEP = 1.0 / 600.0
CONTROL_PERIOD = 1.0 / 30.0
"""The demonstrator's tick: one action per 30 fps camera frame, 20 physics steps."""

SLEEP_LIFT = np.radians([0.0, 2.0, 5.0, 0.0, 0.0, 0.0])
"""Shoulder and elbow lifted 2 and 5 deg off their lower limits. The fit pins both at 0, where Menagerie's
convex collision hulls of link 2 and link 5 overlap (7.3 mm with the fitted wrist, 4.7 mm at the zero pose);
+2/+5 deg clears them, also at every corner of the D7 +-2 deg start perturbation."""
SLEEP_POSES: dict[str, np.ndarray] = {
    # Fitted to the real arms' masks over V1 0-6.9 s (13 allowed frames, joint limits enforced; silhouette IoU
    # 0.609 against 0.464 for the zero pose): ~/real2sim-data/scripts/fit_arm_pose.py -> conf/task/pose_sleep.json.
    "left": np.array([0.3959626343407479, 0.0, 0.0, 0.07243047314127601, 0.30979649425131717, -0.03646782556123335])
    + SLEEP_LIFT,
    "right": np.array([0.15509959087017722, 0.0, 0.0, 0.0021290899331034892, -0.505745092744391, 0.36483972593394465])
    + SLEEP_LIFT,
}
"""SLEEP of each arm (rad): folded back flat over its base as at the start of V1. Pan and wrist joints as fitted
(left 22.7/4.1/17.8/-2.1 deg, right 8.9/0.1/-29.0/20.9 deg for joints 0, 3, 4, 5), shoulder and elbow
:data:`SLEEP_LIFT`."""
SLEEP_GRIPPER = 0.004
"""Finger travel (m) at SLEEP: just open, so the closed fingers' pads do not touch each other."""

# --- gripper ------------------------------------------------------------------------------------
GRIPPER_FORCE_LIMIT = 100.0
"""Newtons, the carriage actuator's force range. Trossen's WidowX AI specification lists a maximum
gripping force of 100 N (docs.trossenrobotics.com/trossen_arm/main/specifications/wxai.html; the v1.5
page derives 342.9 N from the 3 Nm gripper motor). In Menagerie's model one actuator drives the right
carriage and a joint equality drives the left, so the squeeze on the object (the normal force at each
finger) is half the actuator force: at most 50 N."""
GRIPPER_KP = 4000.0
"""N/m. Menagerie's 339 N/m saturates at 6.6 N for a 35 mm part commanded fully closed, a 3 N squeeze:
it cannot hold the 0.6 kg box. The real gripper's position loop reaches its effort limit within a few
millimetres of error; 4000 N/m does so at 25 mm."""
GRIPPER_KV = 120.0
"""N s/m: damps the carriage (armature 2.57 kg in Menagerie) to a damping ratio of about 0.6."""
GRIPPER_MAX_TRAVEL = 0.044
"""Metres per finger, the real gripper's maximum finger displacement (Trossen v1.5 specification and
the driver's URDF limit, physicalai_trossen_plugin wxai_follower.urdf); 88 mm between the fingers."""
GRIPPER_SPEED = 0.0875
"""m/s per finger, the URDF velocity limit."""

FINGER_COUPLING_TIMECONST = 0.004
"""Seconds. Menagerie couples the two carriages with a soft joint equality (MuJoCo's default 0.02 s);
under the gripper's 100 N it let the fingers drift 8 mm off centre. The real fingers ride one
rack-and-pinion: a stiff coupling (2.4 timesteps) keeps them close to centre."""

JOINT_SPEED_LIMIT = np.array([3.375, 3.375, 3.375, 7.0, 7.0, 7.0])
"""rad/s, the WidowX AI joint velocity limits (Trossen v1.5 specification; wxai_follower.urdf)."""


def scene_xml_path() -> Path:
    """The registered scene's XML, bundled with the plugin.

    Returns:
        The path of ``scene.xml``.
    """
    return get_urdf_path() / "scenes" / SCENE_ID / "scene.xml"


def tune_model(model: mujoco.MjModel) -> None:
    """Apply the gripper parameters (:data:`GRIPPER_KP`, :data:`GRIPPER_KV`, :data:`GRIPPER_FORCE_LIMIT`).

    The arm actuators keep Menagerie's values. Called once on a freshly compiled model, before any
    data is stepped; idempotent.
    """
    for side in SIDES:
        actuator = model.actuator(f"{side}_{GRIPPER_JOINT}").id
        model.actuator_gainprm[actuator, 0] = GRIPPER_KP
        model.actuator_biasprm[actuator, 1] = -GRIPPER_KP
        model.actuator_biasprm[actuator, 2] = -GRIPPER_KV
        model.actuator_forcerange[actuator] = (-GRIPPER_FORCE_LIMIT, GRIPPER_FORCE_LIMIT)
        model.actuator_forcelimited[actuator] = 1
        model.actuator_ctrlrange[actuator] = (0.0, GRIPPER_MAX_TRAVEL)
        model.actuator_ctrllimited[actuator] = 1
        coupling = _gripper_equality(model, side)
        model.eq_solref[coupling] = (FINGER_COUPLING_TIMECONST, 1.0)


def _gripper_equality(model: mujoco.MjModel, side: str) -> int:
    """The joint equality that makes the left carriage follow the right one.

    Returns:
        The equality's index.

    Raises:
        ValueError: If the model has no such equality.
    """
    left = model.joint(f"{side}_left_carriage_joint").id
    right = model.joint(f"{side}_right_carriage_joint").id
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and {int(model.eq_obj1id[i]), int(model.eq_obj2id[i])} == {
            left,
            right,
        }:
            return i
    msg = f"no finger coupling for the {side} gripper"
    raise ValueError(msg)


def load_model() -> mujoco.MjModel:
    """Compose the scene with both arms, exactly as the plugin's registered scene does, and tune it.

    Returns:
        The tuned model.
    """
    from physicalai_mujoco_plugin.compose import compose_scene  # noqa: PLC0415
    from physicalai_mujoco_plugin.profiles import get_profile  # noqa: PLC0415

    model = compose_scene(scene_xml_path(), get_profile("trossen_wxai")).model
    tune_model(model)
    return model


# --- names and addresses ------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmIds:
    """Addresses of one arm in the composed model."""

    side: str
    qpos: np.ndarray
    dof: np.ndarray
    actuators: np.ndarray
    gripper_actuator: int
    gripper_qpos: tuple[int, int]
    gripper_dof: tuple[int, int]
    site: int
    kp: np.ndarray
    damping: np.ndarray
    finger_bodies: tuple[int, ...]
    arm_bodies: tuple[int, ...]


def arm_ids(model: mujoco.MjModel, side: str) -> ArmIds:
    """Resolve one arm's joints, actuators, gains and bodies.

    Returns:
        The arm's addresses.
    """
    joints = [model.joint(f"{side}_{name}") for name in ARM_JOINTS]
    actuators = np.array([model.actuator(f"{side}_{name}").id for name in ARM_JOINTS])
    right = model.joint(f"{side}_right_carriage_joint")
    left = model.joint(f"{side}_left_carriage_joint")
    finger_bodies = tuple(model.body(f"{side}_{name}").id for name in ("gripper_left", "gripper_right"))
    root = model.body(f"{side}_base_link").id
    arm_bodies = tuple(b for b in range(model.nbody) if _descends_from(model, b, root))
    return ArmIds(
        side=side,
        qpos=np.array([j.qposadr[0] for j in joints]),
        dof=np.array([j.dofadr[0] for j in joints]),
        actuators=actuators,
        gripper_actuator=model.actuator(f"{side}_{GRIPPER_JOINT}").id,
        gripper_qpos=(int(right.qposadr[0]), int(left.qposadr[0])),
        gripper_dof=(int(right.dofadr[0]), int(left.dofadr[0])),
        site=model.site(f"{side}_{TCP_SITE}").id,
        kp=model.actuator_gainprm[actuators, 0].copy(),
        damping=np.array([model.dof_damping[j.dofadr[0]] for j in joints]),
        finger_bodies=finger_bodies,
        arm_bodies=arm_bodies,
    )


def _descends_from(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body > 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


def body_pose(data: mujoco.MjData, body: int) -> tuple[np.ndarray, np.ndarray]:
    """World position and rotation matrix of a body.

    Returns:
        The position and the 3x3 rotation matrix (copies).
    """
    return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()


# --- start state and perturbations --------------------------------------------------------------


def place_start(model: mujoco.MjModel, data: mujoco.MjData, arm_offsets: dict[str, np.ndarray] | None = None) -> None:
    """Both arms at SLEEP (plus *arm_offsets*), at rest, with matching actuator targets."""
    for side in SIDES:
        ids = arm_ids(model, side)
        q = SLEEP_POSES[side] + (arm_offsets or {}).get(side, 0.0)
        lo, hi = model.jnt_range[[model.joint(f"{side}_{n}").id for n in ARM_JOINTS]].T
        q = np.clip(q, lo, hi)
        data.qpos[ids.qpos] = q
        data.qvel[ids.dof] = 0.0
        data.ctrl[ids.actuators] = q
        for adr in ids.gripper_qpos:
            data.qpos[adr] = SLEEP_GRIPPER
        data.ctrl[ids.gripper_actuator] = SLEEP_GRIPPER
    mujoco.mj_forward(model, data)


@dataclass(frozen=True)
class Perturbation:
    """One draw of RUBRIC D7's start perturbations."""

    box_dxy: tuple[float, float]
    box_dyaw: float
    bracket_dxy: tuple[float, float]
    bracket_dyaw: float
    cover_dxy: tuple[float, float]
    cover_dyaw: float
    bin_dxy: tuple[float, float]
    bin_dyaw: float
    arm_dq: dict[str, np.ndarray]
    box_mass_scale: float
    cover_mass_scale: float
    friction_scale: float

    @classmethod
    def draw(cls, rng: np.random.Generator) -> Perturbation:
        """Independent uniform draws over D7's ranges (lengths in metres, angles in radians).

        Returns:
            The draw.
        """

        def u(half: float) -> float:
            return float(rng.uniform(-half, half))

        return cls(
            box_dxy=(u(0.002), u(0.002)),
            box_dyaw=u(math.radians(2.0)),
            bracket_dxy=(u(0.015), u(0.015)),
            bracket_dyaw=u(math.radians(5.0)),
            cover_dxy=(u(0.020), u(0.020)),
            cover_dyaw=u(math.radians(10.0)),
            bin_dxy=(u(0.005), u(0.005)),
            bin_dyaw=u(math.radians(2.0)),
            arm_dq={side: rng.uniform(-math.radians(2.0), math.radians(2.0), 6) for side in SIDES},
            box_mass_scale=float(rng.uniform(0.8, 1.2)),
            cover_mass_scale=float(rng.uniform(0.8, 1.2)),
            friction_scale=float(rng.uniform(0.75, 1.25)),
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly values, angles in degrees.

        Returns:
            The draw as plain values.
        """
        return {
            "box_dxy": self.box_dxy,
            "box_dyaw_deg": math.degrees(self.box_dyaw),
            "bracket_dxy": self.bracket_dxy,
            "bracket_dyaw_deg": math.degrees(self.bracket_dyaw),
            "cover_dxy": self.cover_dxy,
            "cover_dyaw_deg": math.degrees(self.cover_dyaw),
            "bin_dxy": self.bin_dxy,
            "bin_dyaw_deg": math.degrees(self.bin_dyaw),
            "arm_dq_deg": {k: np.degrees(v).tolist() for k, v in self.arm_dq.items()},
            "box_mass_scale": self.box_mass_scale,
            "cover_mass_scale": self.cover_mass_scale,
            "friction_scale": self.friction_scale,
        }


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _rotate_body_about_z(model: mujoco.MjModel, body: int, dxy: tuple[float, float], dyaw: float) -> None:
    """Move a static (world-attached) body by *dxy* and turn it by *dyaw* about its own vertical axis."""
    model.body_pos[body, :2] += dxy
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, _yaw_quat(dyaw), model.body_quat[body].copy())
    model.body_quat[body] = quat


def perturb(model: mujoco.MjModel, data: mujoco.MjData, p: Perturbation) -> None:  # noqa: PLR0914 - one pass over every D7 draw
    """Apply one perturbation to a freshly loaded model and its start state (before the episode).

    Fixtures move with the model (static bodies); the free objects move in ``qpos``. The box moves
    with the bin, then by its own draw within the slot's play (the +-2 mm lateral draw is clipped to
    the slot, "constrained by the slot"), so it never starts inside a wall.
    """
    bin_body, bracket_body = model.body("bin").id, model.body("bracket").id
    box_body, cover_body = model.body("box").id, model.body("cover").id
    box_q = model.jnt_qposadr[model.body_jntadr[box_body]]
    cover_q = model.jnt_qposadr[model.body_jntadr[cover_body]]

    # Bin and the box standing in it: move the box rigidly with the bin first.
    bin_pos0 = model.body_pos[bin_body].copy()
    _rotate_body_about_z(model, bin_body, p.bin_dxy, p.bin_dyaw)
    rel = data.qpos[box_q : box_q + 2] - bin_pos0[:2]
    c, s = math.cos(p.bin_dyaw), math.sin(p.bin_dyaw)
    data.qpos[box_q : box_q + 2] = (
        bin_pos0[:2] + np.array(p.bin_dxy) + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
    )
    _turn_free(data, box_q, p.bin_dyaw)

    # The box's own draw, in the bin frame (x along the slot, y across it), clipped to the slot's play.
    bin_yaw = _yaw_of(model.body_quat[bin_body])
    play = slot_play(model)
    dx_slot, dy_slot = p.box_dxy
    dy_slot = float(np.clip(dy_slot, -play, play))
    # Turning in the slot eats lateral play: half the length times the yaw, at the box's ends.
    yaw_play = max(play - abs(dy_slot), 0.0) / (BOX_LENGTH / 2)
    dyaw = float(np.clip(p.box_dyaw, -yaw_play, yaw_play))
    cb, sb = math.cos(bin_yaw), math.sin(bin_yaw)
    data.qpos[box_q : box_q + 2] += [cb * dx_slot - sb * dy_slot, sb * dx_slot + cb * dy_slot]
    _turn_free(data, box_q, dyaw)

    _rotate_body_about_z(model, bracket_body, p.bracket_dxy, p.bracket_dyaw)

    data.qpos[cover_q : cover_q + 2] += p.cover_dxy
    _turn_free(data, cover_q, p.cover_dyaw)

    for body, scale in ((box_body, p.box_mass_scale), (cover_body, p.cover_mass_scale)):
        # Uniform density change: mass and inertia scale together, the centre of mass stays.
        model.body_mass[body] *= scale
        model.body_inertia[body] *= scale
    model.geom_friction[:, 0] *= p.friction_scale

    place_start(model, data, p.arm_dq)


BOX_LENGTH = 0.155


def slot_play(model: mujoco.MjModel) -> float:
    """Half the free play (m) of the box across its slot, from the compiled geometry.

    Returns:
        The play in metres, at least 0.
    """
    divider = model.geom("bin_divider_2")
    wall = model.geom("bin_wall_y+")
    inner_wall = model.geom_pos[wall.id][1] - model.geom_size[wall.id][1]
    divider_face = model.geom_pos[divider.id][1] + model.geom_size[divider.id][1]
    box_half = box_half_thickness(model)
    return max((inner_wall - divider_face) / 2 - box_half, 0.0)


def box_half_thickness(model: mujoco.MjModel) -> float:
    """Half the box's thickness (its local z), from its geoms.

    Returns:
        Half the thickness in metres.
    """
    body = model.body("box").id
    zs = [
        (model.geom_pos[g][2] - model.geom_size[g][2], model.geom_pos[g][2] + model.geom_size[g][2])
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body and model.geom_contype[g]
    ]
    return (max(z[1] for z in zs) - min(z[0] for z in zs)) / 2


def _yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _turn_free(data: mujoco.MjData, qadr: int, dyaw: float) -> None:
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, _yaw_quat(dyaw), data.qpos[qadr + 3 : qadr + 7].copy())
    data.qpos[qadr + 3 : qadr + 7] = quat


__all__ = [
    "ARM_JOINTS",
    "CONTROL_PERIOD",
    "GRIPPER_FORCE_LIMIT",
    "JOINT_SPEED_LIMIT",
    "SIDES",
    "SLEEP_POSES",
    "ArmIds",
    "Perturbation",
    "arm_ids",
    "load_model",
    "perturb",
    "place_start",
    "scene_xml_path",
    "tune_model",
]
