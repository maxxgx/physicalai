# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scripted bimanual demonstrator for the booth task (TASK_SPEC A0-A7, D1-D6).

Each arm follows a timeline of segments. A segment names a task-space goal computed when it starts
from the true object poses (privileged state): a tool pose relative to an object, or a pose for the
held object itself. Every control tick the segment interpolates its goal (minimum jerk, straight
line and slerp), inverse kinematics turns the tool pose into joint targets, and the targets go to
``data.ctrl``. Only SLEEP -> READY and back are joint-space moves (to an IK-solved READY pose).

Carrying is object-centric: the tool's target is the object's planned pose composed with the grasp
measured when the carry starts, so a grasp that landed a few millimetres off does not misplace the
object. Placements finish with a slow servo on the object's measured pose (an integral on its
position and orientation error), the way an operator nudges a part into a tight slot.

The demonstrator writes only ``data.ctrl``; it never steps the simulation or edits its state.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from physicalai_mujoco_plugin.booth_task.frames import (
    Pose,
    body_pose,
    interpolate,
    min_jerk,
    rotate_by_vector,
    rotation_vector,
)
from physicalai_mujoco_plugin.booth_task.layout import Layout, handover_box_pose
from physicalai_mujoco_plugin.booth_task.scene import CONTROL_PERIOD, SIDES, SLEEP_GRIPPER
from physicalai_mujoco_plugin.booth_task.servo import ArmServo, DynamicsModel

if TYPE_CHECKING:
    from collections.abc import Callable

    import mujoco

Side = Literal["left", "right"]

# --- grasp and handover parameters (metres, radians) ------------------------------------------------
LEFT_GRASP_X = -0.012
"""LEFT grips the top edge 12 mm toward the box's -x end from its middle ..."""
RIGHT_GRASP_X = 0.040
"""... and RIGHT the finned body 40 mm toward the +x end. LEFT's approach leans toward +x, so its
wrist is over the -x end; the fingers' convex hulls (60 mm wide at their base) stay apart while both
hold the box."""
LEFT_TILT = math.radians(30.0)
"""LEFT's approach leans forward along the box: pointing straight down the arm cannot hold the
tool more than ~0.16 m above the table at the bin (IK reach table in NOTES_demonstrator.txt)."""
TAB_TILT = math.radians(35.0)
"""RIGHT's approach to the cover's tab leans forward: the cover lies 0.52 m from the right base,
beyond the straight-down reach."""
OPEN_BOX_EDGE = 0.030
OPEN_BOX_BODY = 0.042
OPEN_TAB = 0.012
CLOSED = 0.0
"""Commanded fully closed: the servo then squeezes with its force limit's share."""

A6_PATH = {
    "height": 0.165,
    "start": (0.521, -0.089, -0.003),
    "end": (0.133, 0.043, 0.039),
    "waypoints": (
        {"t": 28.6, "pos": (0.155, -0.090, 0.167), "progress": 0.45},
        {"t": 29.8, "pos": (0.143, 0.035, 0.124), "progress": 0.80},
    ),
}
"""The real cover's path onto the bracket (V1 24.5-33 s, 80 allowed frames; real2sim scripts/fit_track.py
--both, conf/task/track_cover_A6.json, IoU 0.6-0.9): its body origin at rest before, at rest after, and two
waypoints (V1 time, position, fraction of the way)."""

A6_LIFT_HEIGHT = 0.14
A6_LIFT_LEAD = 0.2
"""The cover's lift off its spot (A6): its origin's height above the table (m) and the fraction of the way toward
the bracket it already moves. The tracked real cover rose straight up to A6_PATH's 16.5 cm there, which takes the
gripper out of the top of the overview frame (rubric D3, NOTES N5); this lift keeps it in view and still clears
the bin by COVER_TRANSFER_CLEARANCE."""
D2_LIFT_LEAD = -0.06
"""Fraction of the way toward its spot the cover moves while RIGHT lifts it off the box (D2); negative is away from
it, toward the robots. Straight up, the cover and the arm hide the whole bracket from the overview for a moment
(rubric D3, NOTES N5); a couple of centimetres toward the robots uncovers the bracket's far rim."""
D2_ARRIVAL_HEIGHT = 0.10
BESIDE_LEAD = 0.35
"""Fraction of the way out (along x) the returning cover already moves while it crosses to its spot's y (D2)."""
"""m above its spot at which the returning cover arrives over it (D2), before the slow final descent."""
COVER_TRANSFER_CLEARANCE = 0.04
"""m between the bin's top and the carried cover's underside: the cover hangs 205 mm long from its tab and
its path between the bracket and its spot passes the bin's -y corner (seed 2013 hit it at 7 cm)."""

SERVO_LIMIT = 0.01
"""m: the placement servo's largest correction."""
SERVO_DOWN_LIMIT = 0.002
"""m: its largest downward correction."""

SERVO_SETTLED_SPEED = 0.03
"""m/s: placement servos only integrate the object's pose error once it has nearly stopped, so the
arm's own lag after a transfer does not wind them up."""

READY_RIGHT = np.array([
    0.0014930653076709606,
    0.38635292495455675,
    0.0,
    0.4016163966962969,
    -0.13743615352449853,
    -0.00957286229612689,
])
"""RIGHT's READY (rad), fitted to the real right arm over V1 80-98 s while it holds still at READY (silhouette IoU
0.757; ~/real2sim-data/conf/task/pose_ready_right.json): compact, gripper held forward at mid-height, about
0.28 m ahead of the base and 0.17 m above the table. Elbow at its lower limit, as fitted (collision-free)."""
MIRROR = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
"""Joint signs of the mirror image across the arm's x-z plane: pan, wrist yaw and roll flip; the pitch joints keep."""
LEFT_READY_PAN = math.radians(25.0)
"""LEFT's READY is RIGHT's mirror image panned 25 deg outward (toward +y, over the bin). The exact mirror holds
LEFT's wrist and upper arm 0-3 mm from (and partly into) the right arm's work over the bracket while LEFT waits at
READY from A4 to D5: the cover on its way in (A6) and back (D2), the box standing up (D4). With Menagerie's convex
collision hulls a pan of 15 deg leaves 9 mm, 20 deg 23 mm, 25 deg 34 mm (seed 0, every tick of 20.5-69 s). The
real left arm also sleeps panned 22.7 deg outward (SLEEP fit); its READY was not measured (it moves at 80-98 s)."""
READY_POSES: dict[str, np.ndarray] = {
    "right": READY_RIGHT,
    "left": READY_RIGHT * MIRROR + np.array([LEFT_READY_PAN, 0.0, 0.0, 0.0, 0.0, 0.0]),
}
"""READY of each arm: RIGHT as fitted, LEFT its mirror image (both bases face +x) panned outward."""


PHASES: tuple[tuple[str, float, float], ...] = (
    ("S0 sleep", 0.0, 7.5),
    ("A0 wake", 7.5, 9.5),
    ("A1 left picks the box", 9.5, 13.0),
    ("A2 left carries and tilts", 13.0, 15.5),
    ("A3 handover left->right", 15.5, 17.5),
    ("A4 right places box in bracket", 17.5, 21.5),
    ("A5 right grasps cover tab", 21.5, 25.5),
    ("A6 right puts cover on box", 25.5, 32.6),
    ("A7 right withdraws to READY", 32.6, 38.0),
    ("SA hold READY", 38.0, 52.0),
    ("D1 right grasps cover tab", 52.0, 55.5),
    ("D2 right returns cover", 55.5, 61.5),
    ("D3 right grasps box in bracket", 61.5, 64.8),
    ("D4 right turns box toward left", 64.8, 69.0),
    ("D5 handover right->left", 69.0, 77.0),
    ("D6 left inserts box in slot", 77.0, 99.5),
    ("SE left withdraws", 99.5, 102.5),
)
"""TASK_SPEC steps with their sim-time spans; V1's own spans are in ``V1_PHASES``."""

V1_PHASES: dict[str, tuple[float, float]] = {
    "A0": (7.5, 9.5),
    "A1": (9.5, 13.0),
    "A2": (13.0, 15.5),
    "A3": (15.5, 17.5),
    "A4": (17.5, 20.5),
    "A5": (21.0, 25.5),
    "A6": (25.5, 32.6),
    "A7": (32.6, 38.0),
    "D1": (52.0, 55.5),
    "D2": (55.5, 61.5),
    "D3": (62.0, 64.5),
    "D4": (64.5, 69.0),
    "D5": (69.0, 77.0),
    "D6": (83.0, 99.5),
}
EPISODE_END = 102.5

TAU_KEYPOINTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (7.5, 7.5),  # A0 starts
    (9.5, 9.5),  # A1
    (13.0, 13.0),  # A2
    (15.5, 15.5),  # A3
    (17.5, 17.5),  # A4
    (20.5, 21.5),  # A4 ends: box released in the bracket
    (21.0, 21.5),  # A5
    (25.5, 25.5),  # A6
    (32.6, 32.6),  # A7 (the real cover is released at ~32.5 s: fit_track.py)
    (38.0, 38.0),  # READY hold
    (52.0, 52.0),  # D1
    (55.5, 55.5),  # D2
    (61.5, 61.5),  # D2 ends
    (62.0, 61.5),  # D3
    (64.5, 64.8),  # D4
    (69.0, 69.0),  # D5
    (77.0, 77.0),  # D5 ends, LEFT carries toward the bin
    (83.0, 81.0),  # D6: over the slot, aligning
    (99.5, 99.5),  # release in the slot
    (102.0, 102.5),  # LEFT withdrawn
)
"""(V1 time, sim time) pairs at the TASK_SPEC step boundaries; tau(t) interpolates between them."""


def tau(t_v1: float) -> float:
    """The sim time at which the demonstrator is in the same TASK_SPEC phase as V1 at *t_v1*.

    Returns:
        The sim time in seconds.
    """
    v1, sim = zip(*TAU_KEYPOINTS, strict=True)
    return float(np.interp(t_v1, v1, sim))


# --- segments ----------------------------------------------------------------------------------------


@dataclass
class Segment:
    """One arm's motion between two times."""

    t0: float
    t1: float
    kind: Literal["hold", "joints", "tool", "carry"]
    goal: Callable[[BoothDemonstrator], object] | None = None
    """Called once when the segment starts: a joint vector, a tool pose or an object pose."""
    grip: float | None = None
    """Finger travel goal from the segment's start; ``None`` keeps the current goal."""
    obj: str | None = None
    """``box`` or ``cover`` for carries."""
    servo: float = 0.0
    """Integral gain (1/s) on the carried object's pose error, for placements."""
    # runtime
    start: object | None = None
    end: object | None = None
    grasp: Pose | None = None
    correction_p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    correction_r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_start: np.ndarray | None = None
    q_goal: np.ndarray | None = None
    s_prev: float = 0.0


class BoothDemonstrator:
    """The booth task, both arms, as a ``controller(model, data)`` that writes ``data.ctrl``."""

    def __init__(self, model: mujoco.MjModel, *, period: float = CONTROL_PERIOD) -> None:
        """Bind to a compiled booth model (possibly perturbed)."""
        self.model = model
        self.period = period
        self.layout = Layout.from_model(model)
        self.arms: dict[str, ArmServo] = {}
        for side in SIDES:
            rest = np.array([0.0, 1.3, 1.1, 0.0, 0.0, 0.0])
            self.arms[side] = ArmServo(model, side, rest)
        self.dynamics = DynamicsModel(model)
        self.box = model.body("box").id
        self.cover = model.body("cover").id
        self.t_origin: float | None = None
        self.box_start: Pose | None = None
        self.cover_start: Pose | None = None
        self.timeline: dict[str, list[Segment]] = {}
        self._active: dict[str, int] = {}
        self._data: mujoco.MjData | None = None
        self.last_targets = np.zeros(14)
        self.ready: dict[str, Pose] = {}
        self.ready_q: dict[str, np.ndarray] = {}

    # --- lifecycle -------------------------------------------------------------------------------

    def reset(self, data: mujoco.MjData) -> None:
        """Start the script from *data*'s state (both arms asleep, objects at their start)."""
        self.t_origin = float(data.time)
        self._data = data
        for arm in self.arms.values():
            arm.sync(data)
        self.box_start = body_pose(data, self.box)
        self.cover_start = body_pose(data, self.cover)
        for side in SIDES:
            self.ready[side], self.ready_q[side] = self._ready_pose(side)
        self.timeline = {"left": self._left_script(), "right": self._right_script()}
        self._active = dict.fromkeys(SIDES, -1)

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """One control tick: write both arms' targets to ``data.ctrl``."""
        del model
        targets = self.step(data, self.period)
        self.apply(data, targets)

    def apply(self, data: mujoco.MjData, targets: np.ndarray) -> None:
        """Write 14 targets (left then right, six joints and the gripper each) to ``data.ctrl``."""
        for k, side in enumerate(SIDES):
            ids = self.arms[side].ids
            data.ctrl[ids.actuators] = targets[7 * k : 7 * k + 6]
            data.ctrl[ids.gripper_actuator] = targets[7 * k + 6]

    def step(self, data: mujoco.MjData, dt: float) -> np.ndarray:
        """Advance the script by one tick and return the 14 targets (model units), without writing them.

        Returns:
            The 14 targets, left arm then right arm.
        """
        if self.t_origin is None:
            self.reset(data)
        self._data = data
        t = float(data.time) - self.t_origin  # type: ignore[operator]
        for side in SIDES:
            self._tick_arm(side, t, dt)
        torque = self.dynamics.update(data, list(self.arms.values()))
        out = [self.arms[side].targets(data, torque, dt) for side in SIDES]
        self.last_targets = np.concatenate(out)
        return self.last_targets

    @property
    def time(self) -> float:
        """Seconds since the script started."""
        if self.t_origin is None or self._data is None:
            return 0.0
        return float(self._data.time) - self.t_origin

    @property
    def phase(self) -> str:
        """The TASK_SPEC step running now."""
        t = self.time
        for name, t0, t1 in PHASES:
            if t0 <= t < t1:
                return name
        return "done"

    @property
    def done(self) -> bool:
        """Whether the script has run to its end."""
        return self.time >= EPISODE_END

    # --- per tick --------------------------------------------------------------------------------

    def _tick_arm(self, side: str, t: float, dt: float) -> None:
        arm = self.arms[side]
        segments = self.timeline[side]
        index = self._active[side]
        while index + 1 < len(segments) and t >= segments[index + 1].t0:
            index += 1
            self._begin(side, segments[index])
        self._active[side] = index
        if index < 0:
            arm.hold(dt)
            return
        seg = segments[index]
        s = min_jerk((t - seg.t0) / max(seg.t1 - seg.t0, 1e-9))
        if seg.kind == "hold":
            arm.hold(dt)
        elif seg.kind == "joints":
            q = seg.start + s * (seg.end - seg.start)  # type: ignore[operator]
            arm.track_joints(q, dt)
        else:
            step = None
            if seg.q_goal is not None:
                step = (s - seg.s_prev) * (seg.q_goal - seg.q_start)
            seg.s_prev = s
            target = interpolate(seg.start, seg.end, s) if seg.kind == "tool" else self._carry_target(seg, s, dt)  # type: ignore[arg-type]
            arm.track_pose(target, dt, step)

    def _begin(self, side: str, seg: Segment) -> None:
        arm = self.arms[side]
        if seg.grip is not None:
            arm.set_grip(seg.grip)
        if seg.kind == "joints":
            seg.start = arm.q_cmd.copy()
            seg.end = np.asarray(seg.goal(self), dtype=float)  # type: ignore[misc]
        elif seg.kind == "tool":
            seg.start = arm.tool_command()
            seg.end = seg.goal(self)  # type: ignore[misc]
            self._plan_branch(arm, seg, seg.end)  # type: ignore[arg-type]
        elif seg.kind == "carry":
            obj = body_pose(self._data, self.box if seg.obj == "box" else self.cover)
            seg.start = obj
            seg.end = seg.goal(self)  # type: ignore[misc]
            seg.grasp = obj.inv() @ arm.tool_command()
            self._plan_branch(arm, seg, seg.end @ seg.grasp)  # type: ignore[operator]

    @staticmethod
    def _plan_branch(arm: ArmServo, seg: Segment, tool_goal: Pose) -> None:
        """Pick the IK branch of the segment's goal; the ticks follow the joint path toward it."""
        seg.q_start = arm.q_cmd.copy()
        seg.q_goal = arm.plan(tool_goal).q
        seg.s_prev = 0.0

    def _carry_target(self, seg: Segment, s: float, dt: float) -> Pose:
        planned: Pose = interpolate(seg.start, seg.end, s)  # type: ignore[arg-type]
        body = self.box if seg.obj == "box" else self.cover
        if seg.servo > 0.0 and self._speed(body) < SERVO_SETTLED_SPEED:
            actual = body_pose(self._data, body)
            # Never more than 2 mm down: an object resting on its support (the box leaning in the slot's play
            # sits ~2 mm high) must not be pressed down by the integral, or the arm drops when the fingers open.
            seg.correction_p = np.clip(
                seg.correction_p + seg.servo * (planned.pos - actual.pos) * dt,
                [-SERVO_LIMIT, -SERVO_LIMIT, -SERVO_DOWN_LIMIT],
                SERVO_LIMIT,
            )
            seg.correction_r = np.clip(
                seg.correction_r + seg.servo * rotation_vector(actual.rot, planned.rot) * dt, -0.08, 0.08
            )
        tool = planned @ seg.grasp  # type: ignore[operator]
        lever = tool.pos - planned.pos
        rot = rotate_by_vector(tool.rot, seg.correction_r)
        pos = planned.pos + rotate_by_vector(np.eye(3), seg.correction_r) @ lever + seg.correction_p
        return Pose(pos, rot)

    def _speed(self, body: int) -> float:
        """A free body's linear speed (m/s).

        Returns:
            The speed in m/s.
        """
        adr = int(self.model.jnt_dofadr[self.model.body_jntadr[body]])
        return float(np.linalg.norm(self._data.qvel[adr : adr + 3]))

    # --- goals -------------------------------------------------------------------------------------

    def obj(self, name: str) -> Pose:
        """An object's current pose.

        Returns:
            The object's pose in the booth frame.
        """
        return body_pose(self._data, self.box if name == "box" else self.cover)

    def tool(self, side: str) -> Pose:
        """An arm's commanded tool pose.

        Returns:
            The commanded tool pose in the booth frame.
        """
        return self.arms[side].tool_command()

    def _ready_pose(self, side: str) -> tuple[Pose, np.ndarray]:
        """READY as a tool pose (forward kinematics of the measured joints) and its joints.

        Returns:
            The tool pose and the joint vector.
        """
        q = np.clip(READY_POSES[side], self.arms[side].ik.lower, self.arms[side].ik.upper)
        pos, rot = self.arms[side].ik.forward(q)
        return Pose(pos, rot), q

    @staticmethod
    def handover_one() -> Pose:
        """Box pose for the left-to-right handover: tilted, fins up toward the right arm.

        Returns:
            The box pose in the booth frame.
        """
        return handover_box_pose(np.array(H1_CENTRE), H1_FINS, H1_YAW)

    @staticmethod
    def handover_two() -> Pose:
        """Box pose for the right-to-left handover: standing on its edge, held toward the left arm.

        Returns:
            The box pose in the booth frame.
        """
        return handover_box_pose(np.array(H2_CENTRE), math.pi / 2, H2_YAW)

    # --- the script ----------------------------------------------------------------------------

    def _left_script(self) -> list[Segment]:
        lay = self.layout
        edge = lay.left_edge_grasp(LEFT_GRASP_X, LEFT_TILT)
        edge2 = lay.left_edge_grasp(LEFT_GRASP_X, H2_LEFT_TILT)
        up = np.array([0.0, 0.0, 1.0])

        def back(side_pose: Pose, dist: float) -> Pose:
            return side_pose.moved(-dist * side_pose.rot[:, 0])

        return [
            Segment(0.0, 7.5, "hold", grip=SLEEP_GRIPPER),
            Segment(7.5, 9.5, "joints", lambda d: d.ready_q["left"]),
            # A1: over the box, down, close, straight up out of the slot.
            Segment(9.5, 10.9, "tool", lambda d: (d.obj("box") @ edge).moved(0.07 * up), grip=OPEN_BOX_EDGE),
            Segment(10.9, 11.7, "tool", lambda d: d.obj("box") @ edge),
            Segment(11.7, 12.2, "hold", grip=CLOSED),
            Segment(12.2, 13.0, "carry", lambda d: d.box_start.moved(0.065 * up), obj="box"),
            # A2: to the handover pose, tilting the fins toward the right arm.
            Segment(13.0, 15.5, "carry", lambda d: d.handover_one(), obj="box"),
            # A3: hold while RIGHT closes, then open and back off along the approach.
            Segment(15.5, 16.8, "hold"),
            Segment(16.8, 17.2, "hold", grip=OPEN_BOX_EDGE),
            Segment(17.2, 17.8, "tool", lambda d: back(d.tool("left"), 0.06)),
            # Up and back over its own side first: RIGHT's forearm rises toward +y on its way to the bracket.
            Segment(17.8, 18.6, "tool", lambda d: d.ready["left"].moved([0.0, 0.0, 0.07])),
            Segment(18.6, 20.0, "joints", lambda d: d.ready_q["left"]),
            Segment(20.0, 69.0, "hold"),
            # D5: over the standing box's top edge, down, close.
            Segment(69.0, 71.5, "tool", lambda d: (d.obj("box") @ edge2).moved(0.06 * up), grip=OPEN_BOX_EDGE),
            Segment(71.5, 72.7, "tool", lambda d: d.obj("box") @ edge2),
            Segment(72.7, 73.2, "hold", grip=CLOSED),
            Segment(73.2, 76.0, "hold"),
            # D6: over the slot, align, a long slow insertion, release, withdraw upward.
            Segment(76.0, 81.0, "carry", lambda d: d.over_slot(0.05), obj="box"),
            Segment(81.0, 86.0, "carry", lambda d: d.over_slot(0.045), obj="box", servo=2.0),
            Segment(86.0, 98.5, "carry", lambda d: d.over_slot(0.0), obj="box", servo=2.0),
            Segment(98.5, 99.5, "hold", grip=OPEN_BOX_EDGE),
            Segment(99.5, 102.5, "tool", lambda d: d.tool("left").moved(0.08 * up)),
        ]

    def _right_script(self) -> list[Segment]:
        lay = self.layout
        body1 = lay.right_body_grasp(RIGHT_GRASP_X, H1_RIGHT_TILT)
        body3 = lay.right_body_grasp(RIGHT_GRASP_X, D3_RIGHT_TILT)
        tab = lay.tab_grasp(TAB_TILT)
        up = np.array([0.0, 0.0, 1.0])

        def back(p: Pose, dist: float) -> Pose:
            return p.moved(-dist * p.rot[:, 0])

        return [
            Segment(0.0, 7.5, "hold", grip=SLEEP_GRIPPER),
            Segment(7.5, 9.5, "joints", lambda d: d.ready_q["right"]),
            Segment(9.5, 13.0, "hold"),
            # A2/A3: wait off the handover pose, approach along the tool axis, close.
            Segment(13.0, 15.5, "tool", lambda d: back(d.handover_one() @ body1, 0.06), grip=OPEN_BOX_BODY),
            Segment(15.5, 16.2, "tool", lambda d: d.obj("box") @ body1),
            Segment(16.2, 16.8, "hold", grip=CLOSED),
            Segment(16.8, 17.5, "hold"),
            # A4: turn it flat over the bracket, lower it in, open, rise.
            Segment(17.5, 18.0, "hold"),
            Segment(18.0, 19.2, "carry", lambda d: d.over_bracket(0.045), obj="box"),
            Segment(19.2, 20.1, "carry", lambda d: d.over_bracket(0.040), obj="box", servo=4.0),
            Segment(20.1, 20.7, "carry", lambda d: d.over_bracket(0.001), obj="box", servo=4.0),
            Segment(20.7, 21.0, "hold", grip=OPEN_BOX_BODY),
            Segment(21.0, 21.5, "tool", lambda d: d.tool("right").moved(0.06 * up)),
            # A5: over the cover's tab, down, pinch.
            Segment(21.5, 23.2, "tool", lambda d: (d.obj("cover") @ tab).moved(0.06 * up), grip=OPEN_TAB),
            Segment(23.2, 24.4, "tool", lambda d: d.obj("cover") @ tab),
            Segment(24.4, 25.5, "hold", grip=CLOSED),
            # A6: lift, carry over the box, lower onto it aligned, open. The path and its timing follow the real
            # cover's, tracked through V1 24.5-33 s (A6_PATH): up to 16.5 cm, along x toward the bracket first, then
            # across, and a slow 2.5 s descent. Transfers clear the bin: the hanging cover spans 205 mm.
            Segment(25.5, 26.4, "hold"),
            Segment(26.4, 27.4, "carry", lambda d: d.cover_raised(d.lift_toward(d.cover_start, d.cover_target(), A6_LIFT_LEAD), A6_LIFT_HEIGHT), obj="cover"),
            Segment(27.4, 28.6, "carry", lambda d: d.path_waypoint(A6_PATH, 1, d.cover_start, d.cover_target()), obj="cover"),
            Segment(28.6, 29.8, "carry", lambda d: d.path_waypoint(A6_PATH, 2, d.cover_start, d.cover_target()), obj="cover"),
            Segment(29.8, 31.0, "carry", lambda d: d.cover_target().moved(0.03 * up), obj="cover"),
            Segment(31.0, 31.6, "carry", lambda d: d.cover_target().moved(0.012 * up), obj="cover", servo=3.0),
            Segment(31.6, 32.3, "carry", lambda d: d.cover_target().moved(0.001 * up), obj="cover", servo=3.0),
            Segment(32.3, 32.6, "hold", grip=OPEN_TAB),
            # A7: rise and back to READY.
            Segment(32.6, 33.7, "tool", lambda d: d.tool("right").moved(0.06 * up)),
            Segment(33.7, 37.0, "joints", lambda d: d.ready_q["right"]),
            Segment(37.0, 52.0, "hold"),
            # D1: pinch the tab again.
            Segment(52.0, 54.2, "tool", lambda d: (d.obj("cover") @ tab).moved(0.06 * up), grip=OPEN_TAB),
            Segment(54.2, 55.0, "tool", lambda d: d.obj("cover") @ tab),
            Segment(55.0, 55.5, "hold", grip=CLOSED),
            # D2: lift it off and put it back where it started, legs down; open; withdraw.
            # (timed to the real cover's return, tracked through V1 54.5-61.5 s: down on its spot at ~58.9 s; following
            # its y-first path at its pace overran a joint's speed limit and dropped the cover, so the path is ours)
            Segment(55.5, 56.3, "carry", lambda d: d.cover_raised(d.lift_toward(d.obj("cover"), d.cover_start, D2_LIFT_LEAD)), obj="cover"),
            # Across toward its own side first, then out to the spot: the straight line takes RIGHT's forearm over
            # the bracket, hiding it from the overview (rubric D3, NOTES N5). The real cover also went y first.
            Segment(56.3, 57.1, "carry", lambda d: d.cover_raised(d.beside_bracket(d.obj("cover"), d.cover_start)), obj="cover"),
            # Descending on the way out: held at transfer height over its spot, the gripper leaves the top of the
            # overview frame. This side of the bracket is clear of the bin.
            Segment(57.1, 58.0, "carry", lambda d: d.cover_start.moved(D2_ARRIVAL_HEIGHT * up), obj="cover"),
            Segment(58.0, 58.6, "carry", lambda d: d.cover_start.moved(0.03 * up), obj="cover"),
            Segment(58.6, 59.4, "carry", lambda d: d.cover_start.moved(0.002 * up), obj="cover", servo=3.0),
            Segment(59.4, 59.8, "hold", grip=OPEN_TAB),
            Segment(59.8, 61.5, "tool", lambda d: d.tool("right").moved(0.06 * up)),
            # D3: over the lying box, down, close, lift it out flat.
            Segment(61.5, 63.0, "tool", lambda d: (d.obj("box") @ body3).moved(0.06 * up), grip=OPEN_BOX_BODY),
            Segment(63.0, 63.7, "tool", lambda d: d.obj("box") @ body3),
            Segment(63.7, 64.2, "hold", grip=CLOSED),
            Segment(64.2, 64.8, "carry", lambda d: d.obj("box").moved(0.05 * up), obj="box"),
            # D4: stand it on its edge and hold it up toward the left arm.
            Segment(64.8, 69.0, "carry", lambda d: d.handover_two(), obj="box"),
            # D5: hold while LEFT closes, then open, back off along the approach, READY.
            Segment(69.0, 73.4, "hold"),
            Segment(73.4, 73.8, "hold", grip=OPEN_BOX_BODY),
            Segment(73.8, 75.0, "tool", lambda d: back(d.tool("right"), 0.06)),
            Segment(75.0, 79.5, "joints", lambda d: d.ready_q["right"]),
            Segment(79.5, EPISODE_END, "hold"),
        ]

    def cover_raised(self, cover: Pose, height: float = 0.0) -> Pose:
        """*cover* lifted so its underside clears the bin's walls by :data:`COVER_TRANSFER_CLEARANCE`, and to
        at least *height* (m, its origin above the table).

        Returns:
            The cover pose at transfer height.
        """
        z = max(self.layout.bin_top + COVER_TRANSFER_CLEARANCE, height)
        return Pose(np.array([cover.pos[0], cover.pos[1], max(z, cover.pos[2])]), cover.rot)

    @staticmethod
    def beside_bracket(cover: Pose, goal: Pose) -> Pose:
        """*cover* moved across (along y) to *goal*'s y, keeping its x and height, turned to *goal*'s attitude.

        Returns:
            The cover pose.
        """
        return Pose(np.array([cover.pos[0] + BESIDE_LEAD * (goal.pos[0] - cover.pos[0]), goal.pos[1], cover.pos[2]]), goal.rot)

    @staticmethod
    def lift_toward(cover: Pose, goal: Pose, lead: float) -> Pose:
        """*cover* moved *lead* of the way toward *goal* in the table plane (height and attitude kept).

        Returns:
            The cover pose.
        """
        pos = cover.pos.copy()
        pos[:2] += lead * (goal.pos[:2] - cover.pos[:2])
        return Pose(pos, cover.rot)

    def path_waypoint(self, path: dict, k: int, start: Pose, target: Pose) -> Pose:
        """The *k*-th waypoint of a tracked real path (:data:`A6_PATH`, :data:`D2_PATH`), moved with the actual
        start and target: the offsets of *start* and *target* from the tracked ends are blended by the waypoint's
        progress, so a perturbed start or target keeps the path's shape. Attitude: *start*'s turning into
        *target*'s.

        Returns:
            The object pose.
        """
        w = path["waypoints"][k - 1]
        s = w["progress"]
        d0 = start.pos - np.asarray(path["start"])
        d1 = target.pos - np.asarray(path["end"])
        pos = np.asarray(w["pos"]) + (1 - s) * d0 + s * d1
        rot = interpolate(start, target, s).rot
        return Pose(pos, rot)

    def cover_target(self) -> Pose:
        """Where the cover goes on the box, from the box's measured pose.

        Returns:
            The cover pose in the booth frame.
        """
        return self.layout.cover_on_box(self.obj("box"))

    def over_bracket(self, height: float) -> Pose:
        """The box lying in the bracket, *height* above resting on its floor (the ears stand 30 mm above it).

        Returns:
            The box pose in the booth frame.
        """
        return self.layout.box_in_bracket().moved([0.0, 0.0, height])

    def over_slot(self, height: float) -> Pose:
        """The box standing in its slot as it started, *height* above fully down.

        Returns:
            The box pose in the booth frame.
        """
        return self.layout.box_in_slot(self.box_start).moved([0.0, 0.0, height])  # type: ignore[arg-type]


# --- handover parameters (found with scripts in NOTES_demonstrator.txt) -------------------------------
H1_CENTRE = (0.22, 0.0, 0.15)
H1_FINS = math.radians(45.0)
H1_YAW = math.radians(-20.0)
H1_RIGHT_TILT = 0.0
H2_CENTRE = (0.30, 0.05, 0.18)
H2_YAW = math.radians(-20.0)
H2_LEFT_TILT = math.radians(40.0)
D3_RIGHT_TILT = 0.0

__all__ = ["PHASES", "TAU_KEYPOINTS", "V1_PHASES", "BoothDemonstrator", "tau"]
