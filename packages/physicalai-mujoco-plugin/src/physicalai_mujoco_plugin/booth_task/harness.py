# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A reviewer-style harness for :func:`~physicalai_mujoco_plugin.booth_task.make_episode` (RUBRIC D5-D7).

It runs the loop itself (controller, then physics steps) and checks, around every controller call
and every ``mj_step``:

- the controller changed nothing but ``data.ctrl`` (qpos, qvel, act, mocap, eq_active,
  xfrc_applied, qfrc_applied, userdata, plugin state, time);
- the model is bitwise unchanged from start to end;
- no weld/connect equality or mocap body involves the box, the cover or a finger; both objects are
  free bodies with positive mass and full inertia;
- teleports: every step, each object's displacement is at most ``|v| dt + 1e-5`` m, and an object
  only accelerates upward while it touches a finger or a support;
- the worst penetration of any contact;
- arm-arm and arm-fixture contact forces (limit 20 N);
- joint speeds against the WidowX AI limits;
- success: the assembled state SA and the end state SE within TASK_SPEC/RUBRIC D5 tolerances, and
  no object dropped on the table outside its target.

Usage::

    python -m physicalai_mujoco_plugin.booth_task robustness --seeds 0-49 --perturb --jobs 8
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
import time
import traceback
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from physicalai_mujoco_plugin.booth_task.demonstrator import EPISODE_END, PHASES
from physicalai_mujoco_plugin.booth_task.episode import make_episode
from physicalai_mujoco_plugin.booth_task.frames import Pose, body_pose, yaw_of
from physicalai_mujoco_plugin.booth_task.layout import Layout
from physicalai_mujoco_plugin.booth_task.scene import JOINT_SPEED_LIMIT, SIDES, arm_ids

CONTACT_FORCE_LIMIT = 20.0
ASSEMBLED_AT = 50.0
"""Seconds: SA is checked while both arms idle at READY (V1 38-51.5 s)."""
DATA_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
    "xfrc_applied",
    "qfrc_applied",
    "userdata",
    "plugin_state",
    "time",
)
FIXTURE_BODIES = ("table", "bin", "bracket", "pole_0", "pole_1", "camera_bar")

# --- TASK_SPEC/RUBRIC D5 tolerances --------------------------------------------------------------
ANGLE_TOL_DEG = 10.0
"""Degrees: SA's tilts and alignments; SE's box upright and cover tilt."""
CENTRE_TOL = 0.01
"""Metres: SA, the box centred on the bracket and the cover centred on the box."""
RESTING_TOL = 0.003
"""Metres: SA, the box's height above resting on the bracket floor."""
SLOT_FLOOR_TOL = 0.005
"""Metres: SE, the box's bottom above the slot floor."""
COVER_RETURN_TOL = 0.03
"""Metres: SE, the cover's distance from its start."""
COVER_YAW_TOL_DEG = 15.0
"""Degrees: SE, the cover's yaw against its start."""


@dataclass
class Report:
    """What one episode did and whether it passed."""

    seed: int
    perturb: bool
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    assembled: dict[str, float | bool] = field(default_factory=dict)
    end: dict[str, float | bool] = field(default_factory=dict)
    worst_penetration_mm: float = 0.0
    worst_penetration_pair: str = ""
    max_arm_contact_force: float = 0.0
    max_arm_contact_pair: str = ""
    max_joint_speed_ratio: float = 0.0
    teleports: int = 0
    teleports_strict: int = 0
    """Informational: steps where ``|dp| > |v_before| dt + 1e-5`` (the velocity before the step; MuJoCo
    moves the position with the velocity after it, so free falls from rest show up here)."""
    unsupported_lifts: int = 0
    controller_writes: list[str] = field(default_factory=list)
    model_changed: bool = False
    table_drops: list[str] = field(default_factory=list)
    wall_s: float = 0.0
    sim_s: float = 0.0
    perturbation: dict | None = None
    worst_ik_error_mm: float = 0.0
    phase_at_failure: str = ""


def _model_snapshot(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    out = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:  # noqa: BLE001, S112 - some attributes are not readable
            continue
        if isinstance(value, np.ndarray):
            out[name] = value.copy()
    return out


def _data_snapshot(data: mujoco.MjData) -> dict[str, np.ndarray]:
    return {name: np.array(getattr(data, name), copy=True) for name in DATA_FIELDS}


def static_checks(model: mujoco.MjModel) -> list[str]:
    """Structural honesty: free objects with mass, no welds or mocap on objects or fingers.

    Returns:
        One message per problem found; empty when the model passes.
    """
    problems = []
    watched = {model.body("box").id, model.body("cover").id}
    for side in SIDES:
        watched |= set(arm_ids(model, side).finger_bodies)
        watched |= {model.body(f"{side}_carriage_left").id, model.body(f"{side}_carriage_right").id}
    for i in range(model.neq):
        if int(model.eq_type[i]) in {int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)}:
            bodies = {int(model.eq_obj1id[i]), int(model.eq_obj2id[i])}
            if bodies & watched:
                problems.append(f"equality {i} ties {bodies & watched}")
    problems.extend(f"mocap body {model.body(body).name}" for body in watched if model.body_mocapid[body] >= 0)
    for name in ("box", "cover"):
        body = model.body(name).id
        jnt = model.body_jntadr[body]
        if model.body_jntnum[body] != 1 or model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            problems.append(f"{name} is not a free body")
        if not model.body_mass[body] > 0 or not np.all(model.body_inertia[body] > 0):
            problems.append(f"{name} lacks mass or inertia")
    return problems


# --- task checks ---------------------------------------------------------------------------------


def _box_corners(layout: Layout, box: Pose) -> np.ndarray:
    b = layout.box
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    return np.array([box.apply(s * [b.length / 2, b.width / 2, b.thickness / 2]) for s in signs])


def check_assembled(layout: Layout, box: Pose, cover: Pose) -> dict[str, float | bool]:  # noqa: PLR0914 - one value per SA criterion
    """SA: box flat, fins up, in the bracket, aligned; cover straddling it, tab up, aligned.

    Returns:
        Each measured error, and ``ok`` when all are within tolerance.
    """
    bracket = layout.bracket
    fins_up = math.degrees(math.acos(min(1.0, float(box.rot[2, 2]))))
    axis = box.rot[:, 0]
    bracket_axis = bracket.rot[:, 1]
    box_yaw_err = math.degrees(
        math.acos(min(1.0, abs(float(np.dot(axis[:2], bracket_axis[:2])) / max(np.linalg.norm(axis[:2]), 1e-9))))
    )
    centre_err = float(np.linalg.norm(box.pos[:2] - bracket.pos[:2]))
    resting = float(box.pos[2] - (layout.bracket_floor + layout.box.thickness / 2))
    tab_up = math.degrees(math.acos(min(1.0, float(cover.rot[2, 2]))))
    cover_axis = cover.rot[:, 1]
    cover_yaw_err = math.degrees(
        math.acos(
            min(
                1.0,
                abs(float(np.dot(cover_axis[:2], axis[:2])))
                / max(np.linalg.norm(cover_axis[:2]) * np.linalg.norm(axis[:2]), 1e-9),
            )
        )
    )
    cover_centre_err = float(np.linalg.norm(cover.pos[:2] - box.pos[:2]))
    # Legs straddling: every box corner lies between the cover's end walls (cover frame y), and the
    # walls reach below the box's top.
    local = (_box_corners(layout, box) - cover.pos) @ cover.rot
    inside = bool(np.all(np.abs(local[:, 1]) < layout.cover.inner_length / 2))
    box_top = float(np.max(_box_corners(layout, box)[:, 2]))
    walls_below_top = bool(cover.pos[2] < box_top)
    result = {
        "box_fins_up_deg": fins_up,
        "box_axis_err_deg": box_yaw_err,
        "box_centre_err_m": centre_err,
        "box_height_above_floor_m": resting,
        "cover_tab_up_deg": tab_up,
        "cover_yaw_err_deg": cover_yaw_err,
        "cover_centre_err_m": cover_centre_err,
        "cover_legs_straddle": inside and walls_below_top,
    }
    result["ok"] = bool(
        fins_up <= ANGLE_TOL_DEG
        and box_yaw_err <= ANGLE_TOL_DEG
        and centre_err <= CENTRE_TOL
        and resting <= RESTING_TOL
        and tab_up <= ANGLE_TOL_DEG
        and cover_yaw_err <= ANGLE_TOL_DEG
        and cover_centre_err <= CENTRE_TOL
        and inside
        and walls_below_top
    )
    return result


def _slot_index(layout: Layout, model: mujoco.MjModel, point: np.ndarray) -> int:
    """Which bin slot (0 = -y-most ... 3 = +y-most) a point is in, -1 if outside the bin.

    Returns:
        The slot index, or -1 outside the bin.
    """
    local = layout.bin.inv().apply(point)
    dividers = sorted(float(model.geom_pos[model.geom(f"bin_divider_{k}").id][1]) for k in range(3))
    wall = model.geom("bin_wall_y+").id
    inner = float(model.geom_pos[wall][1] - model.geom_size[wall][1])
    if abs(local[1]) > inner:
        return -1
    return int(np.searchsorted(dividers, local[1]))


def check_end(
    model: mujoco.MjModel, layout: Layout, box: Pose, cover: Pose, box_start: Pose, cover_start: Pose
) -> dict[str, float | bool]:
    """SE: box standing in the slot it started in (the +y-most), fully down; cover near its start.

    Returns:
        Each measured error, and ``ok`` when all are within tolerance.
    """
    upright = math.degrees(math.acos(min(1.0, abs(float(box.rot[2, 1])))))  # the width axis vertical
    bottom = float(np.min(_box_corners(layout, box)[:, 2]))
    above_floor = bottom - layout.slot_floor
    slot = _slot_index(layout, model, box.pos)
    start_slot = _slot_index(layout, model, box_start.pos)
    cover_dist = float(np.linalg.norm(cover.pos[:2] - cover_start.pos[:2]))
    dyaw = math.degrees(abs((yaw_of(cover.rot, 1) - yaw_of(cover_start.rot, 1) + math.pi / 2) % math.pi - math.pi / 2))
    legs_down = math.degrees(math.acos(min(1.0, float(cover.rot[2, 2]))))
    result = {
        "box_upright_err_deg": upright,
        "box_bottom_above_slot_floor_m": above_floor,
        "box_slot": slot,
        "start_slot": start_slot,
        "cover_dist_m": cover_dist,
        "cover_yaw_err_deg": dyaw,
        "cover_tilt_deg": legs_down,
    }
    result["ok"] = bool(
        upright <= ANGLE_TOL_DEG
        and above_floor <= SLOT_FLOOR_TOL
        and slot == start_slot == 3  # noqa: PLR2004 - the +y-most of four slots
        and cover_dist <= COVER_RETURN_TOL
        and dyaw <= COVER_YAW_TOL_DEG
        and legs_down <= ANGLE_TOL_DEG
    )
    return result


# --- the run ---------------------------------------------------------------------------------------


def run(seed: int, perturb: bool, *, until: float = EPISODE_END) -> Report:  # noqa: C901, PLR0912, PLR0914, PLR0915, FBT001
    """Run one episode through every check.

    Returns:
        The episode's report.
    """
    report = Report(seed=seed, perturb=perturb)
    t_wall = time.monotonic()
    try:
        model, data, controller = make_episode(seed, perturb)
    except Exception as exc:  # noqa: BLE001
        report.failures.append(f"make_episode raised {exc!r}")
        return report
    if getattr(controller, "perturbation", None) is not None:
        report.perturbation = controller.perturbation.as_dict()
    report.failures += static_checks(model)
    model_before = _model_snapshot(model)
    layout = Layout.from_model(model)
    box_id, cover_id = model.body("box").id, model.body("cover").id
    box_start, cover_start = body_pose(data, box_id), body_pose(data, cover_id)
    objects = {"box": box_id, "cover": cover_id}
    qadr = {k: int(model.jnt_qposadr[model.body_jntadr[b]]) for k, b in objects.items()}
    vadr = {k: int(model.jnt_dofadr[model.body_jntadr[b]]) for k, b in objects.items()}
    arms = {side: arm_ids(model, side) for side in SIDES}
    owner = np.full(model.nbody, "", dtype=object)
    for side in SIDES:
        for b in arms[side].arm_bodies:
            owner[b] = side
    fixtures = {model.body(name).id for name in FIXTURE_BODIES}
    fingers = {b for side in SIDES for b in arms[side].finger_bodies}
    supports = fixtures | set(objects.values())
    table = model.body("table").id
    steps = round(controller.period / model.opt.timestep)
    dt = float(model.opt.timestep)
    force = np.zeros(6)
    assembled_checked = False
    try:  # noqa: PLR1702 - the checks run inside the step loop, around every mj_step
        while data.time < until - 1e-9:
            before = _data_snapshot(data)
            ctrl_before = data.ctrl.copy()
            controller(model, data)
            for name, value in before.items():
                if not np.array_equal(value, getattr(data, name)):
                    report.controller_writes.append(f"t={data.time:.2f} {name}")
            del ctrl_before
            for _ in range(steps):
                pos0 = {k: data.qpos[a : a + 3].copy() for k, a in qadr.items()}
                vel0 = {k: data.qvel[a : a + 3].copy() for k, a in vadr.items()}
                mujoco.mj_step(model, data)
                touching = {k: set() for k in objects}
                for i in range(data.ncon):
                    con = data.contact[i]
                    b1, b2 = int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])
                    if con.dist < -report.worst_penetration_mm / 1000:
                        report.worst_penetration_mm = -con.dist * 1000
                        report.worst_penetration_pair = f"{model.body(b1).name}/{model.body(b2).name} t={data.time:.2f}"
                    for k, b in objects.items():
                        if b in {b1, b2}:
                            touching[k].add(b2 if b1 == b else b1)
                    o1, o2 = owner[b1], owner[b2]
                    arm_arm = o1 and o2 and o1 != o2
                    arm_fixture = (o1 and b2 in fixtures) or (o2 and b1 in fixtures)
                    if arm_arm or arm_fixture:
                        mujoco.mj_contactForce(model, data, i, force)
                        f = float(np.linalg.norm(force[:3]))
                        if f > report.max_arm_contact_force:
                            report.max_arm_contact_force = f
                            report.max_arm_contact_pair = (
                                f"{model.body(b1).name}/{model.body(b2).name} t={data.time:.2f}"
                            )
                for k, b in objects.items():
                    v1 = data.qvel[vadr[k] : vadr[k] + 3]
                    dp = float(np.linalg.norm(data.qpos[qadr[k] : qadr[k] + 3] - pos0[k]))
                    if dp > max(np.linalg.norm(v1), np.linalg.norm(vel0[k])) * dt + 1e-5:
                        report.teleports += 1
                    if dp > np.linalg.norm(vel0[k]) * dt + 1e-5:
                        report.teleports_strict += 1
                    if (v1[2] - vel0[k][2]) / dt > 0.5 and not (touching[k] & (fingers | supports)):  # noqa: PLR2004
                        report.unsupported_lifts += 1
                    if table in touching[k]:
                        _note_table_contact(report, k, data, b, cover_start)
            for side in SIDES:
                ratio = float(np.max(np.abs(data.qvel[arms[side].dof]) / JOINT_SPEED_LIMIT))
                report.max_joint_speed_ratio = max(report.max_joint_speed_ratio, ratio)
            if not assembled_checked and data.time >= ASSEMBLED_AT:
                assembled_checked = True
                report.assembled = check_assembled(layout, body_pose(data, box_id), body_pose(data, cover_id))
    except Exception as exc:  # noqa: BLE001
        report.failures.append(f"raised {exc!r}: {traceback.format_exc(limit=2)}")
    report.sim_s = float(data.time)
    report.wall_s = time.monotonic() - t_wall
    report.worst_ik_error_mm = 1000 * max(arm.worst_ik_error for arm in controller.arms.values())
    if until >= EPISODE_END:
        report.end = check_end(
            model, layout, body_pose(data, box_id), body_pose(data, cover_id), box_start, cover_start
        )
    after = _model_snapshot(model)
    report.model_changed = any(not np.array_equal(v, after.get(k)) for k, v in model_before.items())
    _judge(report, until)
    return report


def _note_table_contact(report: Report, name: str, data: mujoco.MjData, body: int, cover_start: Pose) -> None:
    """Record an object touching the table away from where it belongs there."""
    pos = data.xpos[body]
    if name == "cover" and float(np.linalg.norm(pos[:2] - cover_start.pos[:2])) <= 0.03:  # noqa: PLR2004
        return
    note = f"{name} on the table at t={data.time:.2f} ({pos[0]:.3f}, {pos[1]:.3f})"
    if not report.table_drops or report.table_drops[-1].split(" at ")[0] != note.split(" at ", maxsplit=1)[0]:
        report.table_drops.append(note)


def _judge(report: Report, until: float) -> None:
    f = report.failures
    if report.controller_writes:
        f.append(f"controller wrote {report.controller_writes[:3]}")
    if report.model_changed:
        f.append("model changed during the episode")
    if report.teleports:
        f.append(f"{report.teleports} teleport steps")
    if report.unsupported_lifts:
        f.append(f"{report.unsupported_lifts} unsupported upward accelerations")
    if report.worst_penetration_mm > 2.0:  # noqa: PLR2004
        f.append(f"penetration {report.worst_penetration_mm:.2f} mm ({report.worst_penetration_pair})")
    if report.max_arm_contact_force > CONTACT_FORCE_LIMIT:
        f.append(f"arm contact {report.max_arm_contact_force:.1f} N ({report.max_arm_contact_pair})")
    if report.max_joint_speed_ratio > 1.0:
        f.append(f"joint speed {report.max_joint_speed_ratio:.2f}x the limit")
    if report.table_drops:
        f.append(f"dropped: {report.table_drops[:2]}")
    if until >= EPISODE_END:
        if not report.assembled.get("ok"):
            f.append(f"assembled state missed: {_short(report.assembled)}")
        if not report.end.get("ok"):
            f.append(f"end state missed: {_short(report.end)}")
    report.passed = not f


def _short(values: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in values.items()}


def phase_of(t: float) -> str:
    """The TASK_SPEC step at sim time *t*.

    Returns:
        The step's name, or ``done`` after the last one.
    """
    for name, t0, t1 in PHASES:
        if t0 <= t < t1:
            return name
    return "done"


def report_dict(report: Report) -> dict:
    """JSON-friendly report.

    Returns:
        The report as a dict of plain values.
    """
    return asdict(report)


def run_quiet(args: tuple[int, bool]) -> dict:
    """``run(seed, perturb)`` with logging off, as a JSON-friendly dict (for process pools).

    Returns:
        The report as a JSON-friendly dict.
    """
    from loguru import logger  # noqa: PLC0415

    logger.remove()
    seed, perturb = args
    return report_dict(run(seed, perturb))


__all__ = ["Report", "check_assembled", "check_end", "run", "run_quiet", "static_checks"]
