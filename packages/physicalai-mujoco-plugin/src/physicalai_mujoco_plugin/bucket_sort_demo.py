# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scripted demonstrator for the ``bucket_sort`` scene: a Franka FR3, a Panda or the FR3 Duo, with the Franka Hand.

What differs per robot (joint and actuator names, the gripper's command, the home pose, the joint
speed limits) comes from its arms in :mod:`.bucket_sort_arms`; on the Duo, each attempt goes to one
arm while the other holds its home.

It reads privileged simulation state (the instruction, the bucket and object poses) and writes the
arm's joint targets one control tick at a time, so the autopilot (``autopilot.py``) can drive the arm
or publish the targets as a virtual leader. Each episode it:

1. waits a moment for the objects to settle, then plans one attempt: the hand turns its fingers
   across the named object's short horizontal side (of the two symmetric yaws, the one whose plan
   keeps the wrist further from its limits), stops above the object, descends straight down and
   closes;
2. lifts the object, carries it above the rim of the named bucket, lowers it into the opening and
   opens the hand a little;
3. rises and returns home, then waits for the episode's success and the scene's next episode.

Motions are minimum-jerk profiles of the hand's position, yaw and opening, overlapped where one move
blends into the next (as human reaching does), turned into joint targets by damped least-squares
inverse kinematics seeded from the previous tick, so the arm never jumps to another IK branch. A
plan is checked before it runs: IK residual, joint limits with a margin, joint speed, and the
clearance of the swept hand (and of the held object) from the buckets, the other objects and the
table. Every episode draws its heights, speeds, grasp offset, yaw choice and pauses from its own
generator, drawn from the demonstrator's: the same seed repeats the same episodes.

A grasp that does not lift the object, an object that slips out on the way, or one that lands
outside the bucket gets one retry: the hand lets go and returns home, and once the object rests (on
the table, in reach, its long side level) a new attempt with fresh parameters starts from there.
After that the episode is marked ``failed`` and the demonstrator asks for a scene reset
(:meth:`BucketSortDemonstrator.take_reset_request`), so a recorder can discard the episode and the
next one starts. A target object that moves or turns before the grasp re-plans the approach; a target bucket
that moves during the carry or the descent into it, up to the release, re-plans the place. The
buckets are free bodies: their poses are read from the live data on every check and copied, with
the objects', from ``qpos`` into every plan.

In ``leader`` mode nothing but a client sends the targets back. When the arm does not reach a
plan's last target within :data:`CATCH_UP_S`, the demonstrator stops (``stalled``) instead of
failing and resetting episodes that nobody runs.

Planning an attempt takes from a tenth of a second (an FR3) to several seconds (a leaning arm, or
two arms). With ``background_planning`` (as the autopilot runs it), :meth:`BucketSortDemonstrator.step`
plans on a thread of its own, from a copy of the simulation's data, so the control tick never waits
for it: meanwhile the phase reads ``planning`` and the arm brakes to a stop or holds its targets,
and the simulation (with the virtual leader it publishes) keeps ticking.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from itertools import starmap
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.bucket_sort import (
    BUCKET_INTERIOR,
    DEFAULT_LAYOUT,
    OBJECT_Y,
    bucket_colors,
    bucket_containing,
    scene_objects,
)
from physicalai_mujoco_plugin.bucket_sort_objects import grasp_period
from physicalai_mujoco_plugin.robot_cameras import CAMERA_BODY_PLAN_MARGIN

_PROXY_NAME = __import__("re").compile(r"_plan\d+$|_pc\d+_\d+$")
_HULL_NAME = __import__("re").compile(r"_hull\d+$")

PLAN_PROXY_GROUP = 5
"""Geom group of planning-only proxies (``<body>_plan<i>``, ``<bucket>_pc<k>_<j>``; profiles/rebot_b601.py, the
bucket-v2 scene): the clearance checks use them instead of a body's collision geoms."""
from physicalai_mujoco_plugin.bucket_sort_arms import (
    FR3_ARM,
    FRANKA_GRIP_OPEN,
    FRANKA_HEIGHTS,
    BucketSortArm,
    Heights,
)

FrankaArm = BucketSortArm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from physicalai_mujoco_plugin.bucket_sort import BucketSortTask, Instruction

ARM_JOINTS = FR3_ARM.joints
ACTUATORS = (*FR3_ARM.actuators, FR3_ARM.gripper)
"""The ``franka_fr3`` channels, in action order: seven arm joints (radians in the model) and the hand's width (m)."""
HAND_BODY = "hand"
TCP_LOCAL = np.array([0.0, 0.0, 0.1034])
"""Franka's TCP in the hand frame: between the finger tips, 103.4 mm from the hand's base."""
HOME_Q = np.array(FR3_ARM.home)
"""Franka's ready pose, the scene's home: the hand 0.49 m above the table, fingers down."""
HOME_YAW = -np.pi / 2
"""World yaw of the closing axis (the hand's y axis) at :data:`HOME_Q`."""
GRIP_OPEN = FRANKA_GRIP_OPEN
"""The hand's full opening, in metres."""

FR3_DQ_MAX = np.array(FR3_ARM.dq_max)
"""FR3 joint velocity limits, rad/s (Franka's FR3 datasheet)."""
FR3_DDQ_MAX = np.array(FR3_ARM.ddq_max)
"""FR3 joint acceleration limits, rad/s^2."""
FR3_DDDQ_MAX = np.full(7, 5000.0)
"""FR3 joint jerk limits, rad/s^3."""
JOINT_MARGIN = float(np.radians(7.0))
"""The IK keeps every joint this far inside the FR3's hardware limits."""
SPEED_FRACTION = 0.5
"""A plan may use at most this fraction of the FR3's joint velocity limits."""
ACCEL_FRACTION = 0.5
"""A plan may use at most this fraction of the FR3's joint acceleration limits."""
START_BLEND_S = 0.5
"""A plan fades in any first-tick IK correction over this long."""
MAX_STRETCH = 3.0
"""A move that is too fast for the joints is stretched to at most this many times its duration."""
MAX_TICK_SPEED = 0.8
"""Joint targets never change faster than this fraction of the limits, even where the IK would jump."""

MIN_TCP_Z = 0.013
"""Lowest grasp: the finger pads reach 9 mm below the TCP, so they stay 4 mm above the table."""
CLEARANCE = {
    "bucket": 0.012,
    "objects": 0.008,
    "table": 0.002,
    "held_bucket": 0.010,
    "held_objects": 0.008,
    "arm": 0.03,
    "target": -1.0,
}
"""Smallest planned distance, in metres, from the robot (or the object it holds) to each kind of obstacle.

``arm`` is the working arm's distance from another arm that holds still (the Duo's). ``target`` (r17, D4 T6): the hand's
distance from the object it is about to grasp while it approaches and descends, before the close (a finger grazing it
there pushed its neighbours); -1 m: measured, not a budget (the scene's numeric ``r2s_target_descent_m`` sets one)."""
IK_POS_TOL = 0.003
IK_ROT_TOL = float(np.radians(2.0))
SHOULDER_SINGULAR_WIDTH = 0.3
SHOULDER_SINGULAR_WEIGHT = 1.0
IK_DAMPING = 0.03
"""Damping of the least-squares step (m, rad): small, the targets stay well inside the workspace."""
MAX_IK_STEP = 0.1
"""Largest joint change, in radians, of one IK iteration."""
CHECK_EVERY = 2
TILT_STEP = float(np.radians(5.0))
"""Steps in which the planner tries leaning the hand further (``FrankaIK.least_tilt``)."""
GRASP_TILT = float(np.radians(15.0))
"""The most the hand leans at the grasp (over the object, down to it, closing): nearly fingers-down."""
"""Clearance is checked every this many ticks of a plan."""
WRIST_PROBE_Z = 0.15
"""TCP height at which a tilted arm's joint 7 is estimated for a yaw over a point (:meth:`Planner.predicted_q7`)."""

LIFT_CHECK_DZ = 0.04
"""The grasp is checked once the hand has risen this far above the grasp height."""
LIFT_CHECK_WAIT_S = 1.0
"""The lift is judged at most this long after its planned tick, however far the arm lags."""
LIFTED_DZ = 0.02
"""A held object has risen at least this far above where it lay."""
HELD_DISTANCE = 0.045
"""A held object's centre stays within this distance of the TCP."""
SETTLE_TIMEOUT_S = 10.0
"""Give up on an object that does not come to rest within this long."""
SETTLE_CHECK_S = 1.2
"""The longest the demonstrator waits after an attempt for the placed object to rest before judging it (r16: a bound;
the judgement comes as soon as it rests)."""
BETWEEN_PAUSE_S = 0.0
"""The pause before the next object's attempt (instead of :attr:`DemoParams.start_pause_s`). r16 (D4, D7): none; the
attempt waits only on state (the next object resting, the pile still); r11-r15 waited a fixed 0.2 s, twice per object
once the transit came in."""
CATCH_UP_S = 1.5
"""At the end of a plan, wait at most this long for the arm to reach its last target."""
STALL_ERROR = 0.3
"""A joint further than this (rad) from its target for :data:`CATCH_UP_S` means nobody follows the targets.

The position servos lag up to 0.2 rad at the plan's top speed; a client easing the targets adds
its own delay."""
BRAKE_S = 0.35
"""An abandoned motion slows to a stop over this long."""
FAILURE_HOLD_S = 1.0
"""A failed episode is reported this long, at home, before the reset is requested."""
REPLAN_SHIFT = 0.012
"""Re-plan the approach when the target object moved this far (m) before the grasp."""
REPLAN_TURN = float(np.radians(5.0))
"""Re-plan the approach when the target object's grasp axis turned this far before the grasp."""
RETRY_X = (0.24, 0.52)
RETRY_Y = (OBJECT_Y[0] - 0.02, OBJECT_Y[1] + 0.02)
"""Where a retry picks the object up again: around the spawn area (``bucket_sort.OBJECT_X`` and
``OBJECT_Y``), with room for the nudge a missed grasp gives it."""
MAX_REPLANS = 3
DEFAULT_MAX_ATTEMPTS = 2
"""Attempts per episode, the retry included, unless a recorder asks for others (``RecordingOptions``)."""
STUCK_PHASES = frozenset({"carry", "lower", "release"})
"""Where a stuck start puts the hand: on the way to the bucket or over it (``stuck_start``)."""
STUCK_GRIP = (0.0, 0.03)
"""The hand's opening at a stuck start, metres: closed or almost, as if it held the object."""
STUCK_MOVE_S = (1.5, 2.5)
"""How long the move to a stuck start takes, seconds."""
STUCK_TRIES = 8
"""Poses of the planned path tried before a stuck start gives up."""
STILL_SPEED = 0.01
"""With several arms, an arm whose joints are all slower than this (rad/s) is still (``_arms_still``)."""
STILL_S = 0.1
"""How long the arms must be still on their targets before an attempt is planned."""
TARGET_TOLERANCE = 1e-4
"""How close (rad) an arm's actuator targets must be to the ones it holds; a client's degrees round a little."""
OUT_OF_REACH = 0.02
"""With several arms, a best-effort plan whose IK misses by more than this (m), or that comes too close
to an arm holding still, does not run: the episode fails instead (no arm reaches it)."""

PLAN_WAIT_S = 0.02
"""Background planning: how long the tick that starts a plan waits for it before the arm holds.

A plan done by then runs from this tick, exactly as planned in the tick; a longer one runs from the
first tick after it is done."""
PLAN_YIELD_S = 0.0005
"""Background planning: the planner lets the control thread run (releases the GIL) this often."""

Phase = Literal[
    "idle",
    "settle",
    "planning",
    "approach",
    "descend",
    "close",
    "lift",
    "carry",
    "lower",
    "release",
    "retreat",
    "return",
    "done",
    "failed",
    "stalled",
]
Outcome = Literal["running", "success", "failed", "stalled"]
PlanKind = Literal["attempt", "place", "home", "transit"]
"""A whole attempt (pick, place, home), the rest of one with the object in the hand, or the way home."""


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DemoParams:
    """Motion parameters of one attempt (metres, seconds, radians); :meth:`sample` randomizes them."""

    start_pause_s: float = 0.6
    """Wait after the episode starts, for the objects to settle."""
    go_home: bool = True
    """Whether an attempt ends with the way home. ``False`` while other objects of a kind-level instruction are still
    to be sorted: the attempt ends straight above the bucket and the next one starts there (r11: a several-object
    episode spent ~5 s per object going home and waiting)."""
    transit_speed: float = 0.42
    """Peak hand speed of the long moves (approach, carry, return), m/s."""
    fine_speed: float = 0.12
    """Peak hand speed of the descent onto the object and into the bucket, m/s."""
    lift_speed: float = 0.18
    yaw_speed: float = 1.5
    """Peak turn rate of the hand about the vertical, rad/s."""
    held_accel: float = 1.0
    """Peak hand acceleration with the object in the hand, m/s^2."""
    pregrasp_height: float = 0.10
    """Height of the stop above the grasp."""
    lift_height: float = 0.10
    """Height above the grasp of the vertical lift, before the carry blends in."""
    carry_z: float = 0.23
    """TCP height of the carry: the held object passes 6 cm or more above the 0.12 m rims."""
    release_z: float = 0.14
    """TCP height where the hand opens over the bucket."""
    grasp_offset: float = 0.0
    """Grasp this far (m) from the object's centre along its long side."""
    grasp_dz: float = 0.002
    """Grasp this far above the object's centre (or :data:`MIN_TCP_Z`)."""
    yaw_noise: float = 0.0
    """Turn of the closing axis away from square to the object's short side."""
    open_margin: float = 0.025
    """Opening beyond the object's width on the way down."""
    release_margin: float = 0.02
    """Opening beyond the object's width to let it go."""
    close_s: float = 0.6
    settle_s: float = 0.3
    """Pause with the hand closed before lifting."""
    release_pause_s: float = 0.2
    """Pause over the bucket before opening."""
    after_release_s: float = 0.4
    blend: float = 0.25
    """How much of a move overlaps the next one, as a fraction of its duration (0: stop between moves)."""
    place_offset: tuple[float, float] = (0.0, -0.01)
    """Release point in the bucket frame, from the interior's centre: across, along (+ toward the open end)."""
    reorient: bool = True
    """Turn the object during the carry so its long side runs along the bucket."""
    release_yaw_noise: float = 0.0
    yaw_choice: float = 0.5
    """Uniform draw that picks between two grasp yaws whose plans are about as good."""
    close_drift_mps: float = 0.0
    """r15 (operator drift, :func:`operator_drift`): once the fingers hold the object, the hand starts rising at this
    speed through the rest of the close and the settle, instead of standing still (0: it stands still)."""
    release_creep_mps: float = 0.0
    """The hand keeps sinking at this speed during :attr:`release_pause_s` (at most :data:`RELEASE_CREEP_MAX_M`)."""
    release_rise_mps: float = 0.0
    """Once the opening fingers clear the object, the hand rises at this speed through :attr:`after_release_s`."""
    motion_ramp: float | None = None
    """r18 (D7, :func:`operator_timing`): the scripted moves' speed profile (:func:`move_progress`); ``None``: minimum jerk."""
    release_open_mps: float | None = None
    """r18: the release's finger opening speed (m/s of width); ``None``: a fixed 0.45 s opening (r1-r17)."""
    arrive_wait_s: float = 0.1
    """How long a plan's last target stands before the arm counts as arrived (joints within 0.02 rad of it; r1-r17 0.1 s)."""

    @classmethod
    def for_heights(cls, heights: Heights = FRANKA_HEIGHTS) -> DemoParams:
        """Return the defaults, with the heights in the middle of *heights* for an arm other than a Franka.

        Returns:
            The parameters.
        """
        if heights == FRANKA_HEIGHTS:
            return cls()

        def mid(span: tuple[float, float]) -> float:
            return (span[0] + span[1]) / 2.0

        return cls(
            pregrasp_height=mid(heights.pregrasp_height),
            lift_height=mid(heights.lift_height),
            carry_z=mid(heights.carry_z),
            release_z=mid(heights.release_z),
        )

    @classmethod
    def sample(cls, rng: np.random.Generator, heights: Heights = FRANKA_HEIGHTS) -> DemoParams:
        """Draw a randomized set of parameters, every one inside what the scene allows (and *heights*).

        Returns:
            The parameters.
        """
        u = rng.uniform
        return cls(
            start_pause_s=float(u(0.4, 1.2)),
            transit_speed=float(u(0.30, 0.55)),
            fine_speed=float(u(0.08, 0.16)),
            lift_speed=float(u(0.12, 0.24)),
            yaw_speed=float(u(1.0, 2.0)),
            held_accel=float(u(0.7, 1.2)),
            pregrasp_height=float(u(*heights.pregrasp_height)),
            lift_height=float(u(*heights.lift_height)),
            carry_z=float(u(*heights.carry_z)),
            release_z=float(u(*heights.release_z)),
            grasp_offset=float(u(-0.006, 0.006)),
            grasp_dz=float(u(0.0, 0.004)),
            yaw_noise=float(np.radians(u(-7.0, 7.0))),
            open_margin=float(u(0.018, 0.035)),
            release_margin=float(u(0.012, 0.03)),
            close_s=float(u(0.45, 0.8)),
            settle_s=float(u(0.15, 0.5)),
            release_pause_s=float(u(0.0, 0.4)),
            after_release_s=float(u(0.25, 0.6)),
            blend=float(u(0.1, 0.35)),
            place_offset=(float(u(-0.008, 0.008)), float(u(-0.03, 0.01))),
            reorient=bool(rng.random() < 0.75),  # noqa: PLR2004
            release_yaw_noise=float(np.radians(u(-10.0, 10.0))),
            yaw_choice=float(rng.random()),
        )


# ----------------------------------------------------------------------
# Minimum-jerk motion script
# ----------------------------------------------------------------------


def min_jerk(s: np.ndarray | float) -> np.ndarray:
    """Return the minimum-jerk profile ``10 s^3 - 15 s^4 + 6 s^5`` of a move's progress, clipped to [0, 1]."""
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


MIN_JERK_PHASES = frozenset({"descend", "close", "lower", "release"})
"""r18: the moves that end at or near a contact keep minimum jerk under a ramped profile (DemoParams.motion_ramp): the hand
comes down onto the object, the fingers close and the object is lowered into the bucket and let go as in r1-r17 (a ramped
descent is still 10 % above the bottom at 80 % of its time, against 5.8 %, and the fingers' early close is timed for the
latter: default 31026, a finger 12 mm into a battery). Free-space moves ramp: approach, lift, carry, retreat, return, home."""


def move_progress(s: np.ndarray | float, ramp: float | None = None) -> np.ndarray:
    """A scripted move's progress (0..1) at normalised time *s* (r18, D7): minimum jerk (*ramp* ``None``, r1-r17, ~17 % of
    every move below a tenth of its peak speed), or the operator-like profile: a smooth (quintic) speed ramp over the share
    *ramp* of the move at each end and a constant speed between (:attr:`DemoParams.motion_ramp`)."""
    if ramp is None or ramp >= 0.5:  # noqa: PLR2004 - a half-and-half ramp is minimum jerk's shape
        return min_jerk(s)
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0); vc = 1.0 / (1.0 - ramp)
    area = lambda x: 2.5 * x**4 - 3.0 * x**5 + x**6   # noqa: E731 - the integral of the smoothstep 10x^3-15x^4+6x^5
    return np.where(s < ramp, vc * ramp * area(np.clip(s / ramp, 0, 1)),
                    np.where(s > 1.0 - ramp, 1.0 - vc * ramp * area(np.clip((1.0 - s) / ramp, 0, 1)), vc * (0.5 * ramp + (s - ramp))))


def move_duration(distance: float, peak_speed: float, minimum: float = 0.3, peak_accel: float | None = None,
                  ramp: float | None = None) -> float:
    """Return how long a move of *distance* takes at a peak speed of *peak_speed*, in the profile *ramp*
    (:func:`move_progress`).

    Minimum jerk: its peak speed is 1.875 times its mean speed, its peak acceleration ``5.77 distance / T^2``. The ramped
    profile (r18, the arm's moves; the fingers keep minimum jerk): ``1 / (1 - r)`` times the mean speed and ``1.875 / (r (1 - r)) distance / T^2`` (r = 0.25: 1.33 and 10).

    Returns:
        The duration in seconds, at least *minimum*, and long enough to stay under *peak_accel*.
    """
    if ramp is None or ramp >= 0.5:  # noqa: PLR2004 - move_progress is minimum jerk there
        speed_k, accel_k = 1.875, 5.77
    else:
        speed_k, accel_k = 1.0 / (1.0 - ramp), 1.875 / (ramp * (1.0 - ramp))
    duration = max(minimum, speed_k * abs(distance) / peak_speed)
    if peak_accel is not None:
        duration = max(duration, float(np.sqrt(accel_k * abs(distance) / peak_accel)))
    return duration


@dataclass
class _Move:
    phase: Phase
    t0: float
    duration: float
    delta: np.ndarray
    target: np.ndarray
    """The state ``(x, y, z, yaw, grip, tilt)`` the script reaches once this move (and those before it) end."""
    q_delta: np.ndarray | None = None
    """Change of the joint-space reference posture over this move (see :meth:`Planner.solve`)."""
    ramp: float | None = None
    """r18: this move's speed profile (:func:`move_progress`; ``None``: minimum jerk)."""


class MotionScript:
    """A sum of minimum-jerk moves of the state ``(x, y, z, yaw, grip, tilt)``.

    Each move adds its displacement with a minimum-jerk profile; a move may start before the
    previous one ends (``blend``), which rounds the corner between them while keeping the speed,
    acceleration and jerk continuous.
    """

    def __init__(self, start: np.ndarray, slow: dict[str, float] | None = None) -> None:
        """Start at *start*, ``(x, y, z, yaw, grip[, tilt])``; moves of a phase in *slow* take that many times longer.

        The tilt (how far the hand leans outward, :func:`lean`) is in the state only when *start* has it.
        """
        self.slow = dict(slow or {})
        self.ramp: float | None = None   # r18: the moves' speed profile (move_progress), from DemoParams.motion_ramp
        self.start = np.asarray(start, dtype=np.float64).copy()
        self.end = self.start.copy()
        self.moves: list[_Move] = []
        self.t_end = 0.0

    def add(
        self,
        phase: Phase,
        *,
        pos: np.ndarray | None = None,
        yaw: float | None = None,
        grip: float | None = None,
        duration: float,
        blend: float = 0.0,
        tilt: float | None = None,
    ) -> None:
        """Move to the given position, yaw, opening and tilt (the others keep their value) in *duration* seconds.

        Args:
            phase: The phase it reports while it runs.
            pos: TCP position.
            yaw: Closing-axis yaw, unwrapped: the hand turns the way the numbers say.
            grip: Commanded opening.
            duration: Seconds.
            blend: Fraction of the previous move it overlaps.
            tilt: How far the hand leans outward.
        """
        duration = float(duration) * self.slow.get(phase, 1.0)
        target = self.end.copy()
        if pos is not None:
            target[:3] = pos
        if yaw is not None:
            target[3] = yaw
        if grip is not None:
            target[4] = grip
        if tilt is not None:
            target[5] = tilt  # a script of a leaning arm, whose state has the tilt
        last = self.moves[-1] if self.moves else None
        t0 = self.t_end
        if last is not None and blend > 0.0 and np.any(last.delta):
            t0 = max(last.t0, self.t_end - blend * last.duration)
        ramp = None if phase in MIN_JERK_PHASES else self.ramp
        self.moves.append(_Move(phase, t0, float(duration), target - self.end, target.copy(), ramp=ramp))
        self.end = target
        self.t_end = max(self.t_end, t0 + float(duration))

    def pause(self, phase: Phase, seconds: float) -> None:
        """Hold still for *seconds*, reporting *phase*."""
        if seconds > 0.0:
            self.moves.append(_Move(phase, self.t_end, float(seconds), np.zeros(len(self.end)), self.end.copy()))
            self.t_end += float(seconds)

    def at(self, t: np.ndarray) -> np.ndarray:
        """Return the state at times *t*, shape ``(len(t), 5)``."""
        t = np.asarray(t, dtype=np.float64)
        state = np.repeat(self.start[None, :], len(t), axis=0)
        for move in self.moves:
            if np.any(move.delta):
                x = (t - move.t0) / move.duration
                step = move_progress(x, move.ramp)[:, None] * move.delta[None, :]
                if move.ramp is not None:   # r18: the fingers keep minimum jerk (they slow into a contact); the arm ramps
                    step[:, 4] = min_jerk(x) * move.delta[4]
                state += step
        return state

    def reference(self, t: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        """Return the joint-space reference posture at times *t*, shape ``(len(t), 7)``.

        It moves like the state, a minimum-jerk profile per move, by each move's ``q_delta``.
        """
        t = np.asarray(t, dtype=np.float64)
        ref = np.repeat(np.asarray(q_start, dtype=np.float64)[None, :], len(t), axis=0)
        for move in self.moves:
            if move.q_delta is not None and np.any(move.q_delta):
                ref += move_progress((t - move.t0) / move.duration, move.ramp)[:, None] * move.q_delta[None, :]
        return ref

    def phases(self, t: np.ndarray) -> list[Phase]:
        """Return the phase at each time: that of the last move started by then."""
        starts = np.array([move.t0 for move in self.moves])
        out: list[Phase] = []
        for value in np.asarray(t):
            index = int(np.searchsorted(starts, value + 1e-9, side="right")) - 1
            out.append(self.moves[max(index, 0)].phase)
        return out

    @property
    def end_tilt(self) -> float:
        """The tilt the script ends at; 0 for a script without one."""
        return float(self.end[5]) if len(self.end) > 5 else 0.0  # noqa: PLR2004

    def first_time(self, phase: Phase) -> float:
        """Return when the first move of *phase* starts, or the end of the script.

        Returns:
            Seconds from the start.
        """
        return next((move.t0 for move in self.moves if move.phase == phase), self.t_end)


# ----------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------


def hand_rotation(yaw: float) -> np.ndarray:
    """Return the hand's world orientation with the fingers down and the closing axis at *yaw*.

    Returns:
        The 3x3 rotation; its columns are the hand's x, y (closing) and z (finger) axes.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]])


def lean(rotation: np.ndarray, outward: np.ndarray, tilt: float) -> np.ndarray:
    """Return *rotation* turned by *tilt* about the level axis across *outward*: the fingers lean outward.

    Args:
        rotation: An orientation, its third column the approach axis (fingers down: straight down).
        outward: The horizontal direction from the arm's base to the hand.
        tilt: The angle, radians; positive turns the approach toward *outward*.

    Returns:
        The turned 3x3 rotation.
    """
    axis = np.array([outward[1], -outward[0], 0.0])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9 or abs(tilt) < 1e-12:  # noqa: PLR2004
        return rotation
    axis /= norm
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    turn = np.eye(3) + np.sin(tilt) * cross + (1.0 - np.cos(tilt)) * cross @ cross
    return turn @ rotation


def wrap_near(angle: float, reference: float, period: float = 2.0 * np.pi) -> float:
    """Return *angle* plus a whole number of periods, as close as possible to *reference*.

    Returns:
        The shifted angle.
    """
    return float(angle + period * np.round((reference - angle) / period))


# ----------------------------------------------------------------------
# Planning off the control tick
# ----------------------------------------------------------------------


class PlanCancelledError(Exception):
    """A plan on its own thread was cancelled: the demonstrator was reset while it planned."""


_planning = threading.local()
"""``job``: the :class:`_PlanJob` whose thread this is, on a planning thread."""


def _checkpoint() -> None:
    """On a planning thread, let the control thread run now and then, and stop a cancelled plan.

    The planner calls it in its inner loops. Planning is mostly short Python and NumPy calls that
    hold the GIL; a thread that holds it hands it over only every ``sys.getswitchinterval()``
    (5 ms), and the control tick needs it after each physics step, so without these pauses a tick
    takes ten times as long. Elsewhere it does nothing. A cancelled plan stops here, by
    :class:`PlanCancelledError`.
    """
    job = getattr(_planning, "job", None)
    if job is not None:
        job.checkpoint()


class _PlanJob:
    """One plan on its own thread, which has the demonstrator's state to itself until it ends.

    Meanwhile the control thread only outputs :attr:`hold`, and the demonstrator's
    :meth:`~BucketSortDemonstrator.reset` marks the job :attr:`cancelled` instead of resetting; the
    control thread collects the job once :attr:`done` is set (``BucketSortDemonstrator._finish``).
    """

    def __init__(
        self, work: Callable[[], object], hold: np.ndarray | None, brake_ticks: int, after: _PlanJob | None
    ) -> None:
        self.hold = hold
        """Targets per tick while it plans, model units: where the arm is, then its braking; the last holds.
        ``None``: hold the actuators' targets."""
        self.brake_ticks = brake_ticks
        """Braking ticks in :attr:`hold` (after its first row), which the plan starts with."""
        self.ticks = 0
        """Control ticks since the one that started it."""
        self.cancelled = False
        self.status: dict[str, object] | None = None
        """The demonstrator's status when it started, which it reports meanwhile."""
        self.result: object = None
        self.error: Exception | None = None
        self.done = threading.Event()
        self._work = work
        self._after = after
        self._yielded = 0.0
        self.thread = threading.Thread(target=self._main, name="bucket-sort-planner", daemon=True)

    def start(self) -> None:
        """Start planning."""
        self.thread.start()

    def checkpoint(self) -> None:
        """Release the GIL every :data:`PLAN_YIELD_S`; raise once cancelled.

        Raises:
            PlanCancelledError: When the plan was cancelled.
        """
        if self.cancelled:
            raise PlanCancelledError
        now = time.perf_counter()
        if now - self._yielded >= PLAN_YIELD_S:
            # sleep(0) keeps the GIL on macOS; any real sleep hands it to a waiting thread.
            time.sleep(1e-6)
            self._yielded = time.perf_counter()

    def _main(self) -> None:
        _planning.job = self
        try:
            if self._after is not None:
                self._after.thread.join()  # a cancelled plan still on its way out
            self._yielded = time.perf_counter()
            self.result = self._work()
        except PlanCancelledError:
            self.cancelled = True
        except Exception as exc:  # noqa: BLE001 - handed to the control thread
            self.error = exc
        finally:
            _planning.job = None
            self.done.set()


class _WorkerJob:
    """The planning worker's stand-in for :class:`_PlanJob`: :func:`_checkpoint` stops a cancelled plan."""

    def __init__(self, flag: object) -> None:
        self._flag = flag

    def checkpoint(self) -> None:
        if self._flag.value:
            raise PlanCancelledError


def _plan_transit_core(planner: Planner, snapshot: SceneSnapshot, start: tuple, q: np.ndarray, target: str, bucket: str,
                       params: DemoParams, dt: float) -> JointPlan:
    """r16 transit (BucketSortDemonstrator._plan_and_run_transit): a collision-checked move over *target* at the carry
    height, as a function of its inputs only (r17: so the planning worker can make it; no random draws)."""
    planner.load(snapshot)
    position, _ = planner.object_pose(target)
    here = np.asarray(start[:3], dtype=np.float64)
    over = np.array([position[0], position[1], max(float(params.carry_z), float(here[2]))])

    def build(slow: dict[str, float]) -> tuple[MotionScript, float]:
        script = MotionScript(start, slow)
        script.ramp = params.motion_ramp
        yaw = float(start[3])
        tilt = None if planner.ik.plain else planner.tilt_at(over, yaw)
        script.add("approach", pos=over, duration=max(move_duration(float(np.linalg.norm(over - here)), params.transit_speed, 0.8, ramp=params.motion_ramp),
                                                      move_duration((tilt or 0.0) - script.end_tilt, params.yaw_speed, 0.8, ramp=params.motion_ramp)), tilt=tilt)
        return script, yaw

    go_home = getattr(planner, "_go_home", True); planner._go_home = False   # no way home: the attempt follows
    try:
        return planner.checked(build, q, dt, kind="transit", target=target, bucket=bucket)
    finally:
        planner._go_home = go_home


def _plan_worker_main(conn: object, model: object, arms: tuple, flag: object) -> None:
    """The planning worker: its own planners on its own copy of the model; one attempt plan per request."""
    import os  # noqa: PLC0415

    parent = os.getppid()
    planners = tuple(Planner(model, arm, tuple(o for o in arms if o is not arm)) for arm in arms)
    _planning.job = _WorkerJob(flag)
    while True:
        try:
            # wake every second: a parent that died without closing us (SIGTERM, SIGKILL) leaves the pipe open on some
            # systems; an orphan is re-parented (to launchd on macOS), so a changed parent pid also ends the worker
            while not conn.poll(1.0):
                if os.getppid() != parent:
                    return
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        if msg[0] == "transit":   # r17: the transit too (it held the GIL on a thread: physics at 0.82 x real time)
            _, job_id, active, snapshot, q, start, target, bucket, params, dt, geom_bits = msg
            try:
                model.geom_contype[:], model.geom_conaffinity[:], model.geom_group[:] = geom_bits
                reply = (job_id, "ok", _plan_transit_core(planners[active], snapshot, start, q, target, bucket, params, dt), None)
            except PlanCancelledError:
                reply = (job_id, "cancelled", None, None)
            except Exception as exc:  # noqa: BLE001 - reported to the demonstrator, which plans itself then
                reply = (job_id, "error", repr(exc), None)
            finally:
                flag.value = 0
            try:
                conn.send(reply)
            except (BrokenPipeError, OSError):
                return
            continue
        (job_id, active, snapshot, q, start, instruction, params, dt, rng_state, geom_bits) = msg
        try:
            contype, conaffinity, group = geom_bits
            model.geom_contype[:], model.geom_conaffinity[:], model.geom_group[:] = contype, conaffinity, group
            rng = None
            if rng_state is not None:
                rng = np.random.default_rng(); rng.bit_generator.state = rng_state
            arm = arms[active]
            plan = _plan_attempt_core(planners[active], arm, len(arms) > 1 or arm.max_tilt > 0.0, snapshot, q, start,
                                      instruction, params, dt, rng)
            try:
                conn.send((job_id, "ok", plan, None if rng is None else rng.bit_generator.state))
            except (BrokenPipeError, OSError):
                return   # the demonstrator is gone
        except PlanCancelledError:
            reply = (job_id, "cancelled", None, None)
        except Exception as exc:  # noqa: BLE001 - reported to the demonstrator, which plans itself then
            reply = (job_id, "error", repr(exc), None)
        else:
            reply = None
        finally:
            flag.value = 0
        if reply is not None:
            try:
                conn.send(reply)
            except (BrokenPipeError, OSError):
                return   # the demonstrator is gone


class _PlanProcess:
    """Attempt planning in a process of its own (:meth:`BucketSortDemonstrator.plan_attempt`).

    Planning is seconds of short Python and NumPy calls that hold the GIL; on a planning thread it still shares the
    interpreter with the control loop and the camera streams, which slow down for as long as it runs. Here the
    planning thread only waits on a pipe (the GIL released) while the worker plans on its own copy of the model. The
    worker gets the model's mutable contact bits with every request; the episode's generator state goes there and
    comes back, so a plan and the draws after it are the same as planned in-process. Any failure of the worker falls
    back to planning in-process.
    """

    def __init__(self, model: object, arms: tuple) -> None:
        import multiprocessing as mp  # noqa: PLC0415

        # "spawn" on every platform (macOS's default too): a fresh interpreter, never a fork of a process that holds
        # OpenGL contexts, threads and the camera workers' pipes
        ctx = mp.get_context("spawn")
        self._flag = ctx.Value("b", 0)
        self._conn, child = ctx.Pipe()
        # r18 (5(a)): with PLAN_PARALLEL the worker starts its own candidate pool, which a daemonic process may not; it then
        # is not a daemon, and is closed at exit (it also ends when its parent goes, below)
        daemon = PLAN_PARALLEL <= 0
        self._proc = ctx.Process(target=_plan_worker_main, args=(child, model, arms, self._flag), name="bucket-sort-plan-worker", daemon=daemon)
        self._proc.start()
        child.close()
        if not daemon:
            import atexit  # noqa: PLC0415
            import weakref  # noqa: PLC0415

            atexit.register(_PlanProcess._close_at_exit, weakref.ref(self))
        self._next = 0
        self._lock = threading.Lock()

    @staticmethod
    def _close_at_exit(ref: object) -> None:
        self = ref()
        if self is None or not self._proc.is_alive():
            return
        try:
            self._flag.value = 1
            self._conn.send(None)
        except (BrokenPipeError, OSError, ValueError):
            pass
        self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.terminate()

    @property
    def alive(self) -> bool:
        return self._proc.is_alive()

    @property
    def pid(self) -> int | None:
        """The worker's process id."""
        return self._proc.pid

    def plan_transit(self, active, snapshot, q, start, target, bucket, params, dt, model) -> JointPlan | None:  # noqa: ANN001, PLR0913
        """r17: the transit planned in the worker; ``None`` when it cannot (the caller plans itself)."""
        with self._lock:
            if not self.alive:
                return None
            self._next += 1; job_id = self._next
            bits = (np.array(model.geom_contype), np.array(model.geom_conaffinity), np.array(model.geom_group))
            try:
                self._conn.send(("transit", job_id, active, snapshot, q, start, target, bucket, params, dt, bits))
            except (BrokenPipeError, OSError):
                return None
            while True:
                if self._conn.poll(0.02):
                    try:
                        rid, status, plan, _ = self._conn.recv()
                    except (EOFError, OSError):
                        return None
                    if rid != job_id:
                        continue
                    if status == "cancelled":
                        raise PlanCancelledError
                    if status != "ok":
                        logger.warning("Demonstrator: the planning worker failed a transit ({}); planning here", plan)
                        return None
                    return plan
                _checkpoint()

    def plan_attempt(self, active, snapshot, q, start, instruction, params, dt, rng, model) -> JointPlan | None:  # noqa: ANN001, PLR0913
        """Plan in the worker; ``None`` when it cannot (dead, or an error): the caller plans itself.

        Raises:
            PlanCancelledError: When the calling plan job is cancelled meanwhile.
        """
        with self._lock:
            if not self.alive:
                return None
            self._next += 1; job_id = self._next
            bits = (np.array(model.geom_contype), np.array(model.geom_conaffinity), np.array(model.geom_group))
            try:
                self._conn.send((job_id, active, snapshot, q, start, instruction, params, dt,
                                 None if rng is None else rng.bit_generator.state, bits))
            except (BrokenPipeError, OSError):
                return None
            while True:
                if self._conn.poll(0.02):
                    try:
                        rid, status, plan, rng_state = self._conn.recv()
                    except (EOFError, OSError):
                        return None
                    if rid != job_id:
                        continue
                    if status == "cancelled":
                        raise PlanCancelledError
                    if status != "ok":
                        logger.warning("Demonstrator: the planning worker failed ({}); planning here", plan)
                        return None
                    if rng is not None and rng_state is not None:
                        rng.bit_generator.state = rng_state
                    return plan
                job = getattr(_planning, "job", None)
                if job is not None and getattr(job, "cancelled", False):
                    self._flag.value = 1   # the worker stops at its next checkpoint and answers "cancelled"
                if not self.alive:
                    return None

    def close(self) -> None:
        try:
            self._conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.terminate()


class FrankaIK:
    """Damped least-squares IK of a Franka arm's TCP: a position, fingers down, the closing axis at a yaw.

    The FR3 has one joint more than the task needs. That redundancy (where the wrist sits on a
    small circle about the hand's vertical axis, with the elbow swinging along) is resolved in the
    task's null space: :meth:`solve` pulls toward a reference posture, and :meth:`optimize` moves
    the arm away from its joint limits, which matters most for the elbow (folded over the near
    objects, nearly straight over the far end of a bucket). Joints stay :data:`JOINT_MARGIN` inside
    the hardware limits; a joint pinned there drops out of the solve (an active set). It runs on a
    private ``MjData``, so the simulation is untouched.

    The arm (:class:`~.bucket_sort_arms.BucketSortArm`) names the joints, the tool and the home pose;
    it defaults to the ``franka_fr3`` profile's FR3. A six-joint arm has no redundancy left. An arm
    with a :attr:`~.bucket_sort_arms.BucketSortArm.max_tilt` may lean the hand outward from the base
    (:func:`lean`) by a *tilt* the planner chooses, and its yaw is the closing axis' with the lean
    taken out.
    """

    def __init__(self, model: object, arm: FrankaArm = FR3_ARM) -> None:
        """Resolve *arm*'s joints and hand on *model*."""
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = model
        self.arm = arm
        self.data = mujoco.MjData(model)
        joints = [model.joint(arm.name(name)).id for name in arm.joints]
        self.qpos_adr = np.array([int(model.jnt_qposadr[j]) for j in joints])
        self.dof_adr = np.array([int(model.jnt_dofadr[j]) for j in joints])
        self.hard_lower = np.array([float(model.jnt_range[j][0]) for j in joints])
        self.hard_upper = np.array([float(model.jnt_range[j][1]) for j in joints])
        margin = JOINT_MARGIN if arm.joint_margin is None else arm.joint_margin
        self.lower = self.hard_lower + margin
        self.upper = self.hard_upper - margin
        self._centre = (self.hard_lower + self.hard_upper) / 2.0
        self._half = (self.hard_upper - self.hard_lower) / 2.0
        self.finger_adr = np.array([int(model.jnt_qposadr[model.joint(arm.name(name)).id]) for name in arm.fingers])
        site_rotation = np.eye(3)
        if arm.tool_site is not None:
            site = int(model.site(arm.name(arm.tool_site)).id)
            self.hand = int(model.site_bodyid[site])
            self.tcp_local = np.array(model.site_pos[site], dtype=np.float64)
            flat = np.zeros(9)
            mujoco.mju_quat2Mat(flat, model.site_quat[site])
            site_rotation = flat.reshape(3, 3)
        else:
            self.hand = int(model.body(arm.name(arm.tool_body)).id)
            self.tcp_local = np.array(arm.tcp, dtype=np.float64)
        approach = site_rotation @ np.asarray(arm.approach, dtype=np.float64)
        closing = site_rotation @ np.asarray(arm.closing, dtype=np.float64)
        self.tool_axes = np.column_stack([np.cross(closing, approach), closing, approach])
        """The hand's axes in the tool body's frame, as :func:`hand_rotation` orders them: across, closing, approach."""
        self.plain = arm.max_tilt <= 0.0 and np.array_equal(self.tool_axes, np.eye(3))
        """Fingers always down on a Franka Hand: the orientation is the tool body's, unchanged."""
        self.grip_open = float(arm.grip_open)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self.damping = IK_DAMPING
        self.home = np.array(arm.home)
        """The arm's home pose: plans start and end there."""
        mujoco.mj_kinematics(model, self.data)
        base = np.array(self.data.xpos[model.jnt_bodyid[joints[0]]][:2], dtype=np.float64)
        self.base_xy = np.zeros(2) if np.allclose(base, 0.0, atol=1e-12) else base
        """Where the base joint's axis meets the table: the hand leans outward from it."""
        self.home_yaw = float(arm.home_yaw) if arm.home_yaw is not None else self.fk(self.home)[1]
        """World yaw of the closing axis at :attr:`home`."""

    def set_arm(self, q: np.ndarray, grip: float | None = None) -> None:
        """Put the private model's arm at *q* (and the fingers at half *grip* each) and update its kinematics."""
        self.data.qpos[self.qpos_adr] = q
        if grip is not None:
            self.data.qpos[self.finger_adr] = min(max(grip, 0.0), self.grip_open) / 2.0
        self._mujoco.mj_kinematics(self.model, self.data)

    def tcp(self, data: object | None = None) -> np.ndarray:
        """Return the TCP's world position (of *data*, by default the private one)."""
        data = self.data if data is None else data
        return data.xpos[self.hand] + data.xmat[self.hand].reshape(3, 3) @ self.tcp_local

    def outward(self, pos: np.ndarray) -> np.ndarray:
        """Return the horizontal unit direction from the base's axis to *pos* (zero over the base).

        Returns:
            ``(x, y)``.
        """
        out = np.asarray(pos[:2], dtype=np.float64) - self.base_xy
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 1e-9 else np.zeros(2)  # noqa: PLR2004

    def hand_axes(self, data: object | None = None) -> np.ndarray:
        """Return the hand's world axes in :func:`hand_rotation`'s order: across, closing, approach.

        Returns:
            The 3x3 rotation.
        """
        data = self.data if data is None else data
        return data.xmat[self.hand].reshape(3, 3) @ self.tool_axes

    def tilt(self, data: object | None = None) -> float:
        """Return how far the hand leans outward from fingers-down, radians.

        Returns:
            The angle; 0 for an arm that keeps its fingers down.
        """
        if self.plain:
            return 0.0
        data = self.data if data is None else data
        approach = self.hand_axes(data)[:, 2]
        out = self.outward(self.tcp(data))
        return float(np.arctan2(approach[0] * out[0] + approach[1] * out[1], -approach[2]))

    def target_rotation(self, pos: np.ndarray, yaw: float, tilt: float = 0.0) -> np.ndarray:
        """Return the tool body's world orientation for the hand at *pos*, closing along *yaw*, leaning *tilt*.

        Returns:
            The 3x3 rotation.
        """
        if self.plain:
            return hand_rotation(yaw)
        return lean(hand_rotation(yaw), self.outward(pos), tilt) @ self.tool_axes.T

    def yaw(self, data: object | None = None) -> float:
        """Return the closing axis' world yaw, in ``(-pi, pi]``, with the hand's lean taken out."""
        data = self.data if data is None else data
        if self.plain:
            axis = data.xmat[self.hand].reshape(3, 3)[:, 1]
        else:
            axes = lean(self.hand_axes(data), self.outward(self.tcp(data)), -self.tilt(data))
            axis = axes[:, 1]
        return float(np.arctan2(axis[1], axis[0]))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the TCP position and closing-axis yaw at *q*.

        Returns:
            ``(position, yaw)``; the yaw is wrapped to ``(-pi, pi]``.
        """
        self.set_arm(q)
        return self.tcp(), self.yaw()

    def continuous_yaw(self, q: np.ndarray) -> float:
        """Return the closing-axis yaw at *q* on the branch the planner uses: home is :attr:`home_yaw`.

        Turning the hand about the vertical turns the wrist joint (joint 7 on a Franka arm) the other
        way, and the base joint turns it along, so ``home_yaw + (q1 - q1_home) - (q7 - q7_home)``
        tells which turn of the wrapped yaw is meant. On a tilted base (the Duo's) the base joint
        turns the hand less than that, but far less than the half turn that would pick the wrong branch.

        Returns:
            The yaw in radians.
        """
        _, yaw = self.fk(q)
        w = self.arm.wrist
        return wrap_near(yaw, self.home_yaw + float(q[0] - self.home[0]) - float(q[w] - self.home[w]))

    def limit_gradient(self, q: np.ndarray) -> np.ndarray:
        """Return the descent direction of ``sum(((q - centre) / half range) ** 6)``.

        The cost is flat in the middle of each joint's range and steep near its ends.

        Returns:
            Per joint, unscaled.
        """
        x = (q - self._centre) / self._half
        gradient = -6.0 * x**5 / self._half
        if self.arm.shoulder_singularity:
            # Joint 2 near zero lines joints 1 and 3 up (a shoulder singularity): push it off zero.
            width = SHOULDER_SINGULAR_WIDTH
            gradient[1] += SHOULDER_SINGULAR_WEIGHT * 2.0 * q[1] / width**2 * np.exp(-((q[1] / width) ** 2))
        return gradient

    def margin(self, q: np.ndarray) -> float:
        """Return the smallest distance (rad) of any joint to its hardware limit."""
        return float(min((q - self.hard_lower).min(), (self.hard_upper - q).min()))

    def solve(  # noqa: PLR0914 - one numerical routine; splitting it hides the algorithm
        self,
        q0: np.ndarray,
        pos: np.ndarray,
        yaw: float,
        *,
        iters: int = 6,
        posture: np.ndarray | None = None,
        posture_gain: float = 0.5,
        limit_gain: float = 0.0,
        null_step: float = 0.05,
        tol: float = 2e-5,
        tilt: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        """Return joints near *q0* that put the TCP at *pos*, fingers down (or leaning *tilt*), closing along *yaw*.

        Args:
            q0: Seed.
            pos: TCP target.
            yaw: Closing-axis yaw.
            iters: Gauss-Newton iterations at most.
            posture: Where the null space pulls (each iteration, by *posture_gain* of the gap).
            posture_gain: Fraction of the gap to *posture* closed per iteration.
            limit_gain: Step along :meth:`limit_gradient` per iteration.
            null_step: Largest null-space change per iteration (rad).
            tol: Position tolerance (m) to stop early; rotation uses ten times it in radians.
            tilt: How far the hand leans outward (:func:`lean`).

        Returns:
            ``(q, position error in m, orientation error in rad)``.
        """
        cache = getattr(self, "_solve_cache", None)
        key = None
        if cache is not None:   # an attempt plan solves the same pose from the same seed many times (r10: 48 %)
            key = (np.asarray(q0, dtype=np.float64).tobytes(), np.asarray(pos, dtype=np.float64).tobytes(), float(yaw), iters,
                   None if posture is None else np.asarray(posture, dtype=np.float64).tobytes(), posture_gain, limit_gain,
                   null_step, tol, float(tilt))
            hit = cache.get(key)
            if hit is not None:
                # the solver leaves its data at the kinematics of the joints it returns, and callers read it
                self.data.qpos[self.qpos_adr] = hit[0]
                self._mujoco.mj_kinematics(self.model, self.data)
                self._mujoco.mj_comPos(self.model, self.data)
                return hit[0].copy(), hit[1], hit[2]
        result = self._solve(q0, pos, yaw, iters=iters, posture=posture, posture_gain=posture_gain, limit_gain=limit_gain,
                             null_step=null_step, tol=tol, tilt=tilt)
        if key is not None and len(cache) < SOLVE_CACHE_MAX:
            cache[key] = (result[0].copy(), result[1], result[2])
        return result

    def solve_cache(self, on: bool) -> None:  # noqa: FBT001
        """Remember :meth:`solve`'s results (exact inputs -> exact outputs) while *on*; cleared either way. Plain
        kinematics, no scene: the same inputs always give the same joints, so a remembered one is the one it would solve."""
        self._solve_cache = {} if on else None

    def _solve(  # noqa: PLR0914 - one numerical routine; splitting it hides the algorithm
        self,
        q0: np.ndarray,
        pos: np.ndarray,
        yaw: float,
        *,
        iters: int = 6,
        posture: np.ndarray | None = None,
        posture_gain: float = 0.5,
        limit_gain: float = 0.0,
        null_step: float = 0.05,
        tol: float = 2e-5,
        tilt: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        """:meth:`solve` without its memory."""
        mujoco, model, data = self._mujoco, self.model, self.data
        q = np.clip(np.asarray(q0, dtype=np.float64).copy(), self.lower, self.upper)
        r_des = self.target_rotation(pos, yaw, tilt)
        quat, e_rot = np.zeros(4), np.zeros(3)
        n = len(q)
        eye6, eye_n = np.eye(6), np.eye(n)
        pos_err, rot_err = np.inf, np.inf
        for it in range(iters + 1):
            _checkpoint()
            data.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            p = self.tcp()
            e_pos = pos - p
            mujoco.mju_mat2Quat(quat, (r_des @ data.xmat[self.hand].reshape(3, 3).T).ravel())
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
            pull = np.zeros(n)
            if posture is not None:
                pull += posture_gain * (posture - q)
            if limit_gain > 0.0:
                pull += limit_gain * self.limit_gradient(q)
            size = float(np.abs(pull).max())
            if size > null_step:
                pull *= null_step / size
            converged = pos_err < tol and rot_err < tol * 10.0 and size < 1e-4  # noqa: PLR2004
            if converged or it == iters:
                break
            mujoco.mj_jac(model, data, self._jacp, self._jacr, p, self.hand)
            jac = np.vstack([self._jacp[:, self.dof_adr], self._jacr[:, self.dof_adr]])
            error = np.concatenate([e_pos, e_rot])
            free = np.ones(n, dtype=bool)
            dq = np.zeros(n)
            for _pass in range(n):
                jf = jac * free
                pinv = jf.T @ np.linalg.inv(jf @ jf.T + self.damping**2 * eye6)
                dq = pinv @ error + (eye_n - pinv @ jf) @ (pull * free)
                dq *= free
                blocked = free & (((q >= self.upper - 1e-9) & (dq > 0)) | ((q <= self.lower + 1e-9) & (dq < 0)))
                if not blocked.any():
                    break
                free &= ~blocked
            biggest = float(np.max(np.abs(dq)))
            if biggest > MAX_IK_STEP:
                dq *= MAX_IK_STEP / biggest
            q = np.clip(q + dq, self.lower, self.upper)
        return q, pos_err, rot_err

    def optimize(
        self, q0: np.ndarray, pos: np.ndarray, yaw: float, iters: int = 80, tilt: float = 0.0
    ) -> tuple[np.ndarray, float, float]:
        """Solve for *pos* and *yaw* (leaning *tilt*), then move along the null space away from the joint limits.

        The result stays on the seed's side of the null space, so consecutive key poses solved from
        each other connect without a jump.

        Returns:
            ``(q, position error in m, orientation error in rad)``.
        """
        q, e_pos, e_rot = self.solve(q0, pos, yaw, iters=30, tilt=tilt)
        return (
            self.solve(q, pos, yaw, iters=iters, limit_gain=0.02, null_step=0.03, tol=1e-6, tilt=tilt)
            if iters
            else (q, e_pos, e_rot)
        )

    def least_tilt(
        self, q0: np.ndarray, pos: np.ndarray, yaw: float, max_tilt: float | None = None
    ) -> tuple[float, np.ndarray] | None:
        """Return the smallest lean, in steps of :data:`TILT_STEP` up to *max_tilt*, that reaches *pos*.

        Args:
            q0: Joints to solve from.
            pos: The TCP's position.
            yaw: The closing yaw.
            max_tilt: The most the hand may lean; ``None`` is the arm's ``max_tilt``.

        The first lean the IK reaches within half its tolerances wins; failing that, the most allowed
        lean if it reaches within them.

        Returns:
            ``(tilt, q)``: the lean and the joints there; ``None`` when no lean up to *max_tilt* reaches.
        """
        limit = self.arm.max_tilt if max_tilt is None else min(max_tilt, self.arm.max_tilt)
        steps = int(np.floor(limit / TILT_STEP + 1e-9)) + 1
        tilt, (q, e_pos, e_rot) = 0.0, (np.asarray(q0, dtype=np.float64), np.inf, np.inf)
        for k in range(steps):
            tilt = k * TILT_STEP
            q, e_pos, e_rot = self.optimize(q0, pos, yaw, iters=0, tilt=tilt)
            if e_pos <= IK_POS_TOL / 2 and e_rot <= IK_ROT_TOL / 2:
                return tilt, q
        if e_pos <= IK_POS_TOL and e_rot <= IK_ROT_TOL:
            return tilt, q
        return None


# ----------------------------------------------------------------------
# Scene geometry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectShape:
    """An object's collision box in its body frame; the hand closes across its long side."""

    half: np.ndarray
    """Half extents along the body's x, y and z."""
    round_width: float | None = None
    """For a body round about its long side (cylinders along it, the resistor): its diameter where the hand closes,
    at the middle (the resistor's waist). Two pads close on a round body at its diameter whatever it has rolled to or
    whatever the heading, so the width the hand closes on is this, not the box's projection."""

    @property
    def long_axis(self) -> int:
        """The body axis of the longest side."""
        return int(np.argmax(self.half))

    @property
    def half_length(self) -> float:
        """Half the longest side."""
        return float(self.half[self.long_axis])

    def grasp_axes(self, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Return how to grasp the object lying with *rotation*: across its long side, horizontally.

        This holds lying flat and also rolled about the long side (the resistor rolls), as long as
        the long side lies roughly level.

        Returns:
            ``(closing axis, long axis, width)``: horizontal unit vectors in the world frame and the
            box's width along the closing axis; ``None`` when the long side is not level.
        """
        along = rotation[:, self.long_axis].copy()
        if abs(float(along[2])) > 0.35:  # noqa: PLR2004 - standing on end, or nearly
            # A squat object (a cube, a cylinder, a cross) lies on another face as well: grasp it across
            # its widest level side instead. An elongated one (the resistor) standing on end is not grasped.
            others = sorted(float(h) for i, h in enumerate(self.half) if i != self.long_axis)
            if self.half_length > 1.3 * others[-1]:  # noqa: PLR2004
                if abs(float(along[2])) < 0.85:  # noqa: PLR2004 - leaning: neither lying nor standing
                    return None
                # standing on its end (the real resistors stand about one in five): grasped from above across its
                # round body at any heading (:meth:`BucketSortPlanner.grasp_period`); the axis here is just one of them
                closing = np.array([1.0, 0.0, 0.0])
                width = float(2.0 * np.sum(self.half * np.abs(rotation.T @ closing))) if self.round_width is None else self.round_width
                return closing, np.array([0.0, 1.0, 0.0]), width
            level = [i for i in range(3) if abs(float(rotation[2, i])) <= 0.35]  # noqa: PLR2004
            if not level:
                return None
            along = rotation[:, max(level, key=lambda i: float(self.half[i]))].copy()
        along[2] = 0.0
        along /= np.linalg.norm(along)
        closing = np.array([-along[1], along[0], 0.0])
        width = float(2.0 * np.sum(self.half * np.abs(rotation.T @ closing)))
        if self.round_width is not None and abs(float(rotation[2, self.long_axis])) <= 0.35:  # noqa: PLR2004 - lying
            width = self.round_width
        return closing, along, width


def object_shape(model: object, name: str) -> ObjectShape:
    """Measure object *name*'s collision hull in its body frame.

    Returns:
        Its box and grasp axis.
    """
    import mujoco  # noqa: PLC0415

    body = int(model.body(name).id)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    rotation = np.zeros(9)
    cylinders = []
    for geom in np.flatnonzero(np.asarray(model.geom_bodyid) == body):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ""
        if _PROXY_NAME.search(geom_name):
            continue   # r14 planning proxies: not the object's shape
        # a collision geom (named <body>_hull<i>; while parked its contacts may be off, r14) or any other colliding geom
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0 and not _HULL_NAME.search(geom_name):
            continue
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        mesh = int(model.geom_dataid[geom])
        if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_MESH) and mesh >= 0:
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            local = model.mesh_vert[start : start + count]
        else:   # a primitive (the resistor's cap / waist cylinders): its own bounding box (geom_dataid is -1 there)
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
            local = model.geom_aabb[geom, :3] + corners * model.geom_aabb[geom, 3:]
        points = local @ rotation.reshape(3, 3).T + model.geom_pos[geom]
        lo, hi = np.minimum(lo, points.min(axis=0)), np.maximum(hi, points.max(axis=0))
        if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            cylinders.append((rotation.reshape(3, 3)[:, 2].copy(), np.asarray(model.geom_pos[geom], dtype=float), float(model.geom_size[geom, 0])))
    shape = ObjectShape(half=(hi - lo) / 2.0)
    if cylinders and all(abs(axis[shape.long_axis]) > 0.999 for axis, _, _ in cylinders):  # noqa: PLR2004 - all along the long side
        centre = (hi + lo) / 2.0
        middle = min(cylinders, key=lambda c: abs(float(c[1][shape.long_axis] - centre[shape.long_axis])))
        shape = ObjectShape(half=shape.half, round_width=2.0 * middle[2])
    return shape


class _Clearance:
    """Distances from the hand (and a held object) to the buckets, the other objects and the table.

    With *others*, also from the whole working arm to those arms, which hold still.
    """

    def __init__(
        self, model: object, data: object, arm: FrankaArm = FR3_ARM, others: tuple[FrankaArm, ...] = ()
    ) -> None:
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = model
        self.data = data

        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]

        proxy_name = __import__("re").compile(r"_plan\d+$|_pc\d+_\d+$")

        def is_proxy(g: int) -> bool:   # r14 planning-only proxies, by name (<body>_plan<i>, <bucket>_pc<k>_<j>): their
            return bool(proxy_name.search(names[g]))   # group and contact bits change (compositor, parking on the shelf)

        def collision_geoms(body_names: tuple[str, ...], *, use_proxies: bool = True) -> np.ndarray:
            """Per body its planning proxies if it has any (each the hull of whole collision pieces: they contain the
            collision, so clearing them clears it; far fewer to check), else its collision geoms."""
            out = []
            for name in body_names:
                b = int(model.body(name).id)
                body_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == b]
                proxies = [g for g in body_geoms if is_proxy(g)] if use_proxies else []
                out += proxies if proxies else [g for g in body_geoms if (model.geom_contype[g] or model.geom_conaffinity[g]) and not is_proxy(g)]
            return np.array(out, dtype=int)

        # a bucket content item's proxies count while the item is shown (bucket fill: its key collision geom has contact)
        key_of = {}
        for g in range(model.ngeom):
            n = names[g]
            if is_proxy(g) and "_pc" in n:
                head, rest = n.split("_pc", 1)
                key = f"{head}_item{rest.split('_', 1)[0]}"
                key_of[g] = int(model.geom(key).id) if key in names else -1
        self._content_key = key_of
        self._buckets_by_fill: dict[bytes, np.ndarray] = {}
        self._bucket_keys: np.ndarray | None = None

        hand = tuple(arm.name(body) for body in arm.hand_bodies)
        self.robot = collision_geoms((*hand, *(arm.name(link) for link in reversed(arm.links))))
        # the fingers against the table set how low a grasp goes: their fine pieces (few pairs; a proxy's extra reach
        # would lift every grasp)
        self.fingers = collision_geoms(tuple(arm.name(body) for body in arm.finger_bodies), use_proxies=False)
        # The base (link 0) never moves: its distance from another arm's base is the mount's.
        moving = tuple(arm.name(link) for link in arm.arm_links[1:])
        self.arm = collision_geoms((*hand, *moving)) if others else None
        self.other_arms = (
            collision_geoms(tuple(o.name(b) for o in others for b in (*o.arm_links, *o.hand_bodies)))
            if others
            else None
        )
        self.buckets = collision_geoms(tuple(f"bucket_{color}" for color in bucket_colors(model)))
        self.objects = {name: collision_geoms((name,)) for name in scene_objects(model)}
        self.table = np.array([int(model.geom("table_top").id)])
        self._fromto = np.zeros(6)
        self._body_radius: dict[int, float] = {}
        self._group_bodies: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._others: dict[str, np.ndarray] = {}

    def _near_bodies(self, group_a: np.ndarray, group_b: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
        """*group_a* and *group_b* cut to the geoms of bodies whose bounding spheres (each body's geoms about its
        origin, rigid, so computed once) come within *cap* of each other (r13: hundreds of small collision pieces per
        bucket and hand, so the pairwise test runs on the near bodies only)."""
        data, model = self.data, self.model
        out = []
        for group in (group_a, group_b):
            key = (id(group), len(group))
            cached = self._group_bodies.get(key)
            if cached is None or cached[0] is not group:
                bodies = np.asarray(model.geom_bodyid[group]); cached = (group, bodies, np.unique(bodies))
                self._group_bodies[key] = cached
            bodies = cached[1]
            for b in cached[2]:
                if int(b) not in self._body_radius:   # over ALL the body's collision geoms: valid for any subset of them
                    g = np.flatnonzero((np.asarray(model.geom_bodyid) == b) & ((np.asarray(model.geom_contype) != 0) | (np.asarray(model.geom_conaffinity) != 0)))
                    g = g if len(g) else group[bodies == b]
                    self._body_radius[int(b)] = float((np.linalg.norm(data.geom_xpos[g] - data.xpos[b], axis=1) + model.geom_rbound[g]).max())
            out.append(bodies)
        ba, bb = out
        ua, ub = self._group_bodies[(id(group_a), len(group_a))][2], self._group_bodies[(id(group_b), len(group_b))][2]
        ca, cb = data.xpos[ua], data.xpos[ub]
        ra = np.array([self._body_radius[int(b)] for b in ua]); rb = np.array([self._body_radius[int(b)] for b in ub])
        gap = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2) - ra[:, None] - rb[None, :]
        near = gap < cap
        keep_a, keep_b = ua[near.any(axis=1)], ub[near.any(axis=0)]
        group_a, group_b = group_a[np.isin(ba, keep_a)], group_b[np.isin(bb, keep_b)]
        # then each side's geoms against the other side's bounding sphere
        for _ in range(2):
            if not len(group_a) or not len(group_b):
                break
            pa = data.geom_xpos[group_a]; c = pa.mean(axis=0); r = float((np.linalg.norm(pa - c, axis=1) + model.geom_rbound[group_a]).max())
            pb = data.geom_xpos[group_b]
            group_b = group_b[np.linalg.norm(pb - c, axis=1) - model.geom_rbound[group_b] < r + cap]
            group_a, group_b = group_b, group_a
        return group_a, group_b

    def _min_distance(self, group_a: np.ndarray, group_b: np.ndarray, cap: float = 0.05) -> float:
        data, model = self.data, self.model
        group_a, group_b = self._near_bodies(group_a, group_b, cap)
        if not len(group_a) or not len(group_b):
            return cap
        pa, pb = data.geom_xpos[group_a], data.geom_xpos[group_b]
        ra, rb = model.geom_rbound[group_a], model.geom_rbound[group_b]
        centre = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        bound = centre - ra[:, None] - rb[None, :]   # a lower bound of each pair's distance (bounding spheres)
        ii, jj = np.nonzero(bound < cap)
        best = cap
        # nearest bound first, and stop once no remaining pair can be closer: exact, and few mj_geomDistance calls when
        # the geoms are many small convex pieces (the r13 collision built from the drawn meshes)
        for k in np.argsort(bound[ii, jj], kind="stable"):
            if bound[ii[k], jj[k]] >= best:
                break
            distance = self._mujoco.mj_geomDistance(model, data, int(group_a[ii[k]]), int(group_b[jj[k]]), best, self._fromto)
            best = min(best, float(distance))
        return best

    def _active_buckets(self) -> np.ndarray:
        """The bucket geoms to check: content items' proxies only for the items the bucket fill shows (cached per fill)."""
        if not self._content_key:
            return self.buckets
        model = self.model
        if self._bucket_keys is None:
            self._bucket_keys = np.array([self._content_key.get(int(g), -1) for g in self.buckets], dtype=int)
        keys = self._bucket_keys; k = np.maximum(keys, 0)
        shown = (keys < 0) | (np.asarray(model.geom_contype)[k] != 0) | (np.asarray(model.geom_conaffinity)[k] != 0)
        key = shown.tobytes()
        cached = self._buckets_by_fill.get(key)
        if cached is None:
            cached = self._buckets_by_fill[key] = self.buckets[shown]
        return cached

    def measure(self, *, target: str, held: bool) -> dict[str, float]:
        """Return the smallest distance to each kind of obstacle at the private data's current pose.

        Args:
            target: The object being picked: the fingers may touch it, and it is the held object.
            held: Whether the hand carries *target* (its pose in the private data must follow the hand).

        Returns:
            Distances in metres by :data:`CLEARANCE` key, capped at 5 cm.
        """
        others = self._others.get(target)
        if others is None:   # one array per target, so the near-body cache (keyed by the array) hits
            others = self._others[target] = np.concatenate([geoms for name, geoms in self.objects.items() if name != target])
        buckets = self._active_buckets()
        out = {
            "bucket": self._min_distance(self.robot, buckets),
            "objects": self._min_distance(self.robot, others),
            "table": self._min_distance(self.fingers, self.table),
        }
        if held:
            out["held_bucket"] = self._min_distance(self.objects[target], buckets)
            out["held_objects"] = self._min_distance(self.objects[target], others)
        if self.arm is not None and self.other_arms is not None:
            out["arm"] = self._min_distance(self.arm, self.other_arms)
        return out


# ----------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------


@dataclass
class JointPlan:
    """Joint targets for every control tick of one plan, with what was checked about them."""

    kind: PlanKind
    dt: float
    q: np.ndarray
    """Arm targets, ``(ticks, 7)``, radians."""
    grip: np.ndarray
    """Hand opening targets, ``(ticks,)``, metres."""
    phases: list[Phase]
    cartesian: np.ndarray
    """The scripted ``(x, y, z, yaw, grip, tilt)`` per tick."""
    lift_check: int | None = None
    """Tick at which the grasp is checked."""
    hold_until: int | None = None
    """Last tick (exclusive) the object must still be in the hand: the start of the release."""
    carry_start: int | None = None
    """First tick of the carry: from there to the release the bucket's pose is checked every tick."""
    bucket_pose: tuple[np.ndarray, float] | None = None
    """The target bucket's position and yaw the plan was made for."""
    yaw_grasp: float = 0.0
    yaw_release: float = 0.0
    pos_error: float = 0.0
    rot_error: float = 0.0
    clearance: dict[str, float] = field(default_factory=dict)
    limit_margin: float = np.inf
    """Smallest distance (rad) of any joint to its hardware limit."""
    peak_speed_ratio: float = 0.0
    """Largest joint speed as a fraction of the FR3's limit."""
    peak_accel_ratio: float = 0.0
    """Largest joint acceleration as a fraction of the FR3's limit."""
    problems: list[str] = field(default_factory=list)
    unreached: bool = False
    """A pose no allowed lean of the hand reaches (``Planner.tilt_at``): the plan never runs."""
    brake_ticks: int = 0
    """Ticks of braking in front of the plan (:meth:`BucketSortDemonstrator._run`)."""

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return not self.problems

    def __len__(self) -> int:
        """Return the number of ticks.

        Returns:
            The plan's length.
        """
        return len(self.q)

    def targets(self, tick: int) -> np.ndarray:
        """Return the targets of *tick*, clamped to the plan's ends.

        Returns:
            The joints (rad), then the opening (m).
        """
        tick = min(max(tick, 0), len(self.q) - 1)
        return np.array([*self.q[tick], self.grip[tick]])


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth each column of *values* with a Gaussian of *sigma* samples, holding the end values.

    Returns:
        The smoothed array; the first and last rows are unchanged when the ends are at rest.
    """
    radius = max(1, int(np.ceil(3.0 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([np.repeat(values[:1], radius, axis=0), values, np.repeat(values[-1:], radius, axis=0)])
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(values.shape[1])], axis=1)


def joint_smoothness(q: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Peak joint speed, acceleration and jerk of a target trajectory, by finite differences.

    Returns:
        ``velocity``, ``acceleration`` and ``jerk``: per-joint peaks (rad/s, rad/s^2, rad/s^3).
    """
    q = np.asarray(q, dtype=np.float64)
    zeros = np.zeros(q.shape[1])
    v = np.diff(q, axis=0) / dt if len(q) > 1 else np.zeros((1, q.shape[1]))
    a = np.diff(v, axis=0) / dt if len(v) > 1 else np.zeros((1, q.shape[1]))
    j = np.diff(a, axis=0) / dt if len(a) > 1 else np.zeros((1, q.shape[1]))
    return {
        "velocity": np.abs(v).max(axis=0) if len(v) else zeros,
        "acceleration": np.abs(a).max(axis=0) if len(a) else zeros,
        "jerk": np.abs(j).max(axis=0) if len(j) else zeros,
    }


@dataclass
class SceneSnapshot:
    """What a plan needs from the simulation: the objects and buckets (free joints, in ``qpos``) as they are now."""

    qpos: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray

    @classmethod
    def take(cls, data: object) -> SceneSnapshot:
        """Copy the poses out of *data*.

        Returns:
            The snapshot.
        """
        return cls(np.array(data.qpos), np.array(data.mocap_pos), np.array(data.mocap_quat))


SOLVE_CACHE_MAX = 200_000
"""Most remembered IK solves within one attempt plan (:meth:`FrankaIK.solve_cache`)."""


def brisk_for_several(params: DemoParams, rng: np.random.Generator | None) -> DemoParams:
    """The timing of an episode that sorts several objects (r11, D4: the hang limit is 3 x the median episode, and
    each object costs about one single-object episode): every speed and pause at the brisk end of its range of
    :meth:`DemoParams.sample`. With *rng* (the randomised mixes), drawn from the brisk quarter of each range; without
    (the default, unrandomised speeds), the brisk end itself."""
    brisk = {"transit_speed": (0.30, 0.55, +1), "fine_speed": (0.08, 0.16, +1), "lift_speed": (0.12, 0.24, +1),
             "close_s": (0.45, 0.8, -1), "release_pause_s": (0.0, 0.4, -1), "after_release_s": (0.25, 0.6, -1), "start_pause_s": (0.4, 1.2, -1)}
    values = {}
    for name, (lo, hi, sign) in brisk.items():
        quarter = (hi - lo) / 4
        end = hi if sign > 0 else lo
        values[name] = float(end - sign * (rng.uniform(0.0, quarter) if rng is not None else 0.0))
    return replace(params, **values)


OPERATOR_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
OPERATOR_CLOSING_MM_S = (9.4, 15.0, 23.5, 38.2, 87.7)
"""r15 (D7): the real operator's hand speed (mm/s, the point between the finger tips by the arm's kinematics) while the
real gripper closes, quantiles p10-p90 of the reviewer's allowed export, TRAIN episodes 0, 2, 6, 7, 9 (143 frames;
scripts/real2sim/v2, sim/r15/real_drift_stats.json). The real hand almost never stands still while the gripper moves."""
OPERATOR_OPENING_MM_S = (9.7, 16.1, 45.2, 108.8, 178.1)
"""The same while the real gripper opens (131 frames); about half of it rising."""
OPERATOR_START_STILL_Q = ((0.0, 0.25, 0.5, 0.75, 1.0), (0.0, 0.27, 0.63, 0.73, 0.80))
"""The real arm's stillness from a recording's first frame to its first motion (0.1 deg per frame), TRAIN episodes 0, 2,
6, 7, 9 sorted, as a quantile curve (probability, seconds). DEV 1, 3, 10: 0.50 / 0.73 / 0.90 s."""
START_OVERHEAD_S = 0.20
"""What a recording adds to the start pause before the first motion: Studio's start after the go and the plan's ramp to
0.1 deg per frame (r16 check 3, 20 episodes: 0.17-0.26 s, median 0.20)."""


def start_pause(u: float) -> float:
    """r16: the start pause for a uniform draw *u*: the TRAIN curve less :data:`START_OVERHEAD_S` (at least 0), so that
    the recorded start stillness follows the real one (p10 / median / p90 about 0.20 / 0.63 / 0.78 s against TRAIN 0.14 /
    0.67 / 0.81; the overhead floors the shortest). Chosen on DEV over U(0, 0.62) s (0.26 / 0.51 / 0.76)."""
    return max(0.0, float(np.interp(u, *OPERATOR_START_STILL_Q)) - START_OVERHEAD_S)


CLOSE_DRIFT_MAX_MPS = 0.04
"""The rise while the fingers squeeze is capped at the TRAIN p75 (a faster one would lift before the grip forms)."""
RELEASE_CREEP_MAX_M = 0.01
"""The sink before the release goes at most 1 cm (a third of :attr:`BucketSortDemonstrator.RELEASE_GAP`)."""
RELEASE_RISE_MAX_MPS = 0.11
"""The rise after the release is capped at the TRAIN p75."""
CONTACT_MARGIN_S = 0.1
CLOSE_EARLY_MAX = 0.5
"""How much of the descent the fingers' closing may overlap at most (its share of the descent; r15-r17 0.5); the fingers
still reach the object's width only CONTACT_MARGIN_S after the descent ends."""
"""How long a plan's last target stands before the arm counts as arrived (with the joints within 0.02 rad of it)."""
"""The rise starts this long after the commanded opening reaches the object's width (the fingers lag the command)."""
RELEASE_CLEAR_M = 0.01
"""The rise after the release starts once the commanded opening is this much wider than the object's widest section."""
RELEASE_RISE_MAX_M = 0.03
"""The rise after the release goes at most 3 cm (the return starts from there; higher, a pose near the top of the
workspace can fall out of reach)."""


def operator_drift(params: DemoParams, rng: np.random.Generator, scale: float = 1.0) -> DemoParams:
    """r15 (D7, the coordinator's plan: real drift and pause statistics, TRAIN fit, DEV choice): the hand keeps moving
    where the scripted demonstrator stood still with the gripper moving, at speeds drawn from the real operator's
    (:data:`OPERATOR_CLOSING_MM_S`, :data:`OPERATOR_OPENING_MM_S`, the p10-p90 range, times *scale*), and the start pause
    drawn from the real one (:func:`start_pause`). The motion is commanded arm motion (the plan's checks see
    it); the directions are the safe ones (up after the grip forms, up once the fingers clear the object), not the real
    operator's mostly sideways jitter, which would nudge the object before contact.

    Returns:
        The parameters with the drift.
    """
    q = OPERATOR_QUANTILES
    closing = float(np.interp(rng.uniform(q[0], q[-1]), q, OPERATOR_CLOSING_MM_S)) / 1000.0 * scale
    creep = float(np.interp(rng.uniform(q[0], q[-1]), q, OPERATOR_CLOSING_MM_S)) / 1000.0 * scale
    opening = float(np.interp(rng.uniform(q[0], q[-1]), q, OPERATOR_OPENING_MM_S)) / 1000.0 * scale
    # (r15: no sink before the release: below the release height the plan's rim clearance was set for, the housing met a
    # bucket wall (record mix 7401); the draw stays, so the other draws keep their values)
    del creep
    return replace(params, start_pause_s=start_pause(float(rng.uniform(0.0, 1.0))), close_drift_mps=min(closing, CLOSE_DRIFT_MAX_MPS),
                   release_rise_mps=min(opening, RELEASE_RISE_MAX_MPS))


OPERATOR_TIMING_R18 = {"motion_ramp": None, "release_open_mps": 0.118, "arrive_wait_s": 0.0, "speed_factor": 1.6,
                       "settle_s": (0.0, 0.15), "after_release_s": (0.0, 0.15)}
"""r18 (D7, the coordinator's retiming: fitted on TRAIN 0, 2, 6, 7, 9 allowed states, chosen on DEV 1, 3, 10 by the
still-run histogram and speed-bin distance, gated live in Studio; scripts/real2sim/v2/timing_profile.py, sim/r18/timing):
the release opening at the operator's 0.118 m/s of width (117-119 mm/s on every single-stage TRAIN open); no fixed wait at
a plan's end (arrival stays a state: within 0.02 rad); the hold after the close and after the release 0-0.15 s (the
operator's: 0); the transit, fine and lift speeds x1.6. Joint speeds and accelerations stay under the TRAIN and DEV
operator's maximum per joint (sim/r18/timing/speed_gate.json). A retry or plan variant drawn later keeps it (_timed_like).
Tried, not adopted: the ramped move profile (motion_ramp 0.25, free-space moves; move_progress): the best offline profile
(DEV 18.1 against 23.7) but twice the transits over the joint acceleration bound, each one planned afresh live: stalls of
4-10 s in Studio (sim/r18/studio_gate). The profile stays available (motion_ramp), off."""


def operator_timing(params: DemoParams, rng: np.random.Generator, timing: dict = OPERATOR_TIMING_R18) -> DemoParams:
    """r18 (D7): the real operator's timing (:data:`OPERATOR_TIMING_R18`), after every other draw, from its own stream.

    Returns:
        The parameters with the timing.
    """
    f = float(timing["speed_factor"])
    ff, fl = float(timing.get("fine_factor", f)), float(timing.get("lift_factor", f))
    return replace(params, settle_s=float(rng.uniform(*timing["settle_s"])), after_release_s=float(rng.uniform(*timing["after_release_s"])),
                   transit_speed=params.transit_speed * f, fine_speed=params.fine_speed * ff, lift_speed=params.lift_speed * fl,
                   motion_ramp=timing["motion_ramp"], release_open_mps=timing["release_open_mps"], arrive_wait_s=timing["arrive_wait_s"])


def _timed_like(new: DemoParams, old: DemoParams, rng: np.random.Generator) -> DemoParams:
    """r18: a new draw (a retry, a plan variant) keeps the episode's timing: :func:`operator_timing` when *old* has it."""
    return operator_timing(new, rng) if old.motion_ramp is not None else new


def _drift_of(params: DemoParams) -> dict:
    return {k: getattr(params, k) for k in ("close_drift_mps", "release_creep_mps", "release_rise_mps")}


def inverse_min_jerk(y: float, ramp: float | None = None) -> float:
    """The progress s in [0, 1] at which :func:`move_progress` (with *ramp*) reaches *y* (clipped to [0, 1])."""
    y = float(np.clip(y, 0.0, 1.0)); lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if float(move_progress(mid, ramp)) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# r16 (coordinator, the batch-vs-alone residual): the demonstrator's time thresholds count whole physics steps since
# the episode's start instead of differences of the ever-growing sim time (Demonstrator._now)
TICK_CLOCK = True
AHEAD_BEST_EFFORT = True
"""r17: a best-effort plan made ahead may be used (:meth:`BucketSortDemonstrator._take_ahead`)."""
AHEAD_QUALITY_M = 0.0005
AHEAD_QUALITY_DEG = 0.5
"""r17 (coordinator): a best-effort plan made ahead is used when, on the actual scene, the target lies within 0.5 mm and
0.5 deg of where it was planned for (its IK aims there) and no clearance of the exact re-check is more than 0.5 mm worse
than at planning time; otherwise the attempt is planned afresh."""
AHEAD_MOVED_M = 0.002
"""r16 (plan ahead): a plan made ahead is used only if its target moved at most this since it was predicted (the other
objects are covered by the exact clearance re-check on the actual scene)."""
AHEAD_TURNED_DEG = 2.0

OBJECT_CUT_M = 0.006
"""A best-effort plan that brings the hand or the held object this far into another object is not run (r11: a 14 mm
cut squeezed a CPU into the battery being grasped, 4.1 mm penetration); the attempt fails or retries instead."""


def _cuts_objects(plan: JointPlan) -> bool:
    """Whether *plan* failed its checks and cuts into another object by more than :data:`OBJECT_CUT_M`."""
    if plan is None or plan.ok:
        return False
    worst = min(plan.clearance.get("objects", np.inf), plan.clearance.get("held_objects", np.inf))
    return bool(worst < -OBJECT_CUT_M)


SCREEN_FIRST = True
"""r16: plan attempts screened first, then (only when none passes) in full again from the same random state
(:func:`_plan_attempt_core`). Exact because the planner's checks are deterministic; tests that feed the planner
non-deterministic stand-in plans turn it off."""
BUCKET_HIT_M = CAMERA_BODY_PLAN_MARGIN
"""r16: a failed plan whose bucket clearance is below minus this counts as bringing the hand into a bucket
(:func:`_hits_bucket`). The clearance is measured on the planning proxies, not the collision: the part that meets a rim
is the camera body, whose proxy is :data:`~physicalai_mujoco_plugin.robot_cameras.CAMERA_BODY_PLAN_MARGIN` (5 mm) larger
every way than its collision box (the bucket's proxies reach up to 5 mm more), so a plan clearance between 0 and -5 mm
does not touch. Measured (real2sim v2 r16): planned -5.2 / -8.1 mm, real contact 0.19 / 0.08 mm deep."""


def _hits_bucket(plan: JointPlan) -> bool:
    """r16 (housing contacts at the true 120 mm bucket height): whether *plan* failed its checks and brings the hand (its
    housing, the camera body) into a bucket (bucket clearance below 0). With other targets left, another goes first and
    this one waits; otherwise the attempt is refused instead of run as a best effort (a re-ordering, no margin change)."""
    if plan is None or plan.ok:
        return False
    return bool(plan.clearance.get("bucket", np.inf) < -BUCKET_HIT_M)


def _may_run_plan(plan: JointPlan) -> bool:
    """:meth:`BucketSortDemonstrator._may_run` as a function (the planning worker has no demonstrator)."""
    return not plan.unreached and plan.pos_error <= OUT_OF_REACH and plan.clearance.get("arm", np.inf) >= CLEARANCE["arm"]


def _plan_attempt_core(
    planner: Planner,
    arm: FrankaArm,
    gated: bool,  # noqa: FBT001
    snapshot: SceneSnapshot,
    q: np.ndarray,
    start: np.ndarray,
    instruction: Instruction,
    params: DemoParams,
    dt: float,
    rng: np.random.Generator | None,
) -> JointPlan:
    """The body of :meth:`BucketSortDemonstrator.plan_attempt` (see there), for *planner* and *arm*; *gated* as
    :attr:`BucketSortDemonstrator._gated`. Runs in the demonstrator's process or in its planning worker."""
    if not SCREEN_FIRST:
        planner.ik.solve_cache(True)
        try:
            return _plan_attempt_cascade(planner, arm, gated, snapshot, q, start, instruction, params, dt, rng)
        finally:
            planner.ik.solve_cache(False)
    planner.ik.solve_cache(True)
    state = rng.bit_generator.state if rng is not None else None
    try:
        # r16 (planning stalls): screen first. A candidate that already fails (joints, IK) skips its clearance sweep, and
        # a sweep stops at the first tick past a clearance budget. A passing candidate is never cut short, so the plan
        # returned is the one the full cascade returns (it returns the first passing candidate). Only when none passes
        # does the cascade run again in full, from the same random state, to rank the failures by their true depths.
        planner.screening = True
        planner._screen_record = record = {}
        plan = _plan_attempt_cascade(planner, arm, gated, snapshot, q, start, instruction, params, dt, rng)
        if plan.ok:
            return plan
        planner.screening = False
        planner._screen_record = None
        planner._screen_reuse = record
        if rng is not None:
            rng.bit_generator.state = state
        return _plan_attempt_cascade(planner, arm, gated, snapshot, q, start, instruction, params, dt, rng)
    finally:
        planner.screening = False
        planner._screen_record = None
        planner._screen_reuse = None
        planner.ik.solve_cache(False)


def _plan_attempt_cascade(
    planner: Planner,
    arm: FrankaArm,
    gated: bool,  # noqa: FBT001
    snapshot: SceneSnapshot,
    q: np.ndarray,
    start: np.ndarray,
    instruction: Instruction,
    params: DemoParams,
    dt: float,
    rng: np.random.Generator | None,
) -> JointPlan:
    planner.load(snapshot)
    position, rotation = planner.object_pose(instruction.object)
    axis = planner.grasp_axis_yaw(instruction.object, rotation) + params.yaw_noise
    # a symmetric object (a cube, a hexagon, a round piece) is grasped across the same width at more yaws: the
    # wrist's 90 deg either way then reaches more of them
    candidates = planner.yaw_candidates(
        axis, float(start[3]), position[:2], period=planner.grasp_period(instruction.object, rotation)
    )
    if len(candidates) == 2:  # noqa: PLR2004
        margins = [planner.key_margin(q, instruction, params, yaw) for yaw in candidates]
        if abs(margins[0] - margins[1]) < np.radians(10.0):
            candidates = candidates if params.yaw_choice < 0.5 else candidates[::-1]  # noqa: PLR2004
        elif margins[1] > margins[0]:
            candidates = candidates[::-1]
    safer, safest = arm.heights.safe
    variants = [
        params,
        replace(params, release_z=max(params.release_z, safer[0]), carry_z=max(params.carry_z, safer[1])),
        replace(
            params, release_z=safest[0], carry_z=safest[1], blend=0.0, place_offset=(0.0, -0.01), grasp_offset=0.0
        ),
    ]
    if rng is not None:
        variants += [_timed_like(replace(DemoParams.sample(rng, arm.heights), go_home=params.go_home, **_drift_of(params)), params, rng)
                     for _ in range(2)]
    # r18 (5(a), exact parallel planning): every candidate starts from the same planner state (_planner_canon: the IK
    # data and the planner's per-script fields as they are here), so its plan does not depend on which candidates ran
    # before it; the candidates of a stage may then run on several workers (PLAN_PARALLEL) and are reduced in their list
    # order: the first that passes, else the least severe (ties: the earlier; where gated, one that may run first)
    canon = _planner_canon(planner)
    best: JointPlan | None = None
    eligible: JointPlan | None = None
    """Where :attr:`_gated`, the best plan that may run at all (:meth:`_may_run`)."""
    try:
        main = [(variant, float(yaw), ("main", variant, float(yaw))) for variant in variants for yaw in candidates]
        for plan in _evaluate_candidates(planner, canon, snapshot, start, instruction, q, dt, main):
            if plan.ok:
                return plan
            if best is None or _less_severe(plan, best):
                best = plan
            if gated and _may_run_plan(plan) and (eligible is None or _less_severe(plan, eligible)):
                eligible = plan
        # None passed. Before running a best effort (which may push a neighbour or miss), try the grasp yaws not yet
        # tried: a symmetric object (a cube, a cross) offers an axis every quarter turn, and another axis may clear
        # the objects around it; and a narrower pre-grasp opening, whose fingers come down closer to the object.
        tried = {round(float(y), 4) for y in candidates}
        extra = [y for y in planner.all_yaws(axis, float(start[3]), position[:2], period=planner.grasp_period(instruction.object, rotation))
                 if round(float(y), 4) not in tried]
        narrow = replace(params, open_margin=min(params.open_margin, 0.012))
        more = [(variant, float(yaw), ("more", variant, float(yaw))) for variant, yaws in ((params, extra), (narrow, list(candidates) + extra)) for yaw in yaws]
        for plan in _evaluate_candidates(planner, canon, snapshot, start, instruction, q, dt, more):
            if plan.ok:
                return plan
            if best is None or _less_severe(plan, best):
                best = plan
            if gated and _may_run_plan(plan) and (eligible is None or _less_severe(plan, eligible)):
                eligible = plan
        if eligible is not None:
            best = eligible  # a plan that would be refused must not hide one that may run
        return best  # type: ignore[return-value]
    finally:
        _planner_restore(planner, canon)


SEVERITY_TIE = 1e-9
"""r18 (5(a)): severities this close are a tie (the earlier candidate wins). Floating point is not bit-identical across
processes (numpy's vector reductions depend on memory alignment), and a 1e-14 difference must not pick another plan."""


def _less_severe(plan: JointPlan, other: JointPlan) -> bool:
    return _severity(plan) < _severity(other) - SEVERITY_TIE


PLAN_PARALLEL = int(__import__("os").environ.get("PHYSICALAI_PLAN_PARALLEL", "0"))
_POOLS: dict = {}


def _candidate_worker_main(conn: object, model: object, arm: object, others: tuple, flag: object) -> None:
    """A candidate pool worker: its own planner on its own copy of the model; one candidate per request (r18)."""
    import os  # noqa: PLC0415

    parent = os.getppid()
    planner = Planner(model, arm, others)
    _planning.job = _WorkerJob(flag)
    while True:
        try:
            while not conn.poll(1.0):
                if os.getppid() != parent:
                    return
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        idx, snapshot, canon, start, instruction, q, dt, variant, yaw, key, screening, bits = msg
        try:
            model.geom_contype[:], model.geom_conaffinity[:], model.geom_group[:] = bits
            planner.screening, planner._screen_record, planner._screen_reuse = screening, None, None
            planner.ik.solve_cache(True)
            try:
                reply = (idx, "ok", _evaluate_candidate(planner, canon, snapshot, start, instruction, q, dt, variant, yaw, key))
            finally:
                planner.ik.solve_cache(False)
        except PlanCancelledError:
            reply = (idx, "cancelled", None)
        except Exception as exc:  # noqa: BLE001 - the caller evaluates it itself then
            reply = (idx, "error", repr(exc))
        try:
            conn.send(reply)
        except (BrokenPipeError, OSError):
            return


class _CandidatePool:
    """PLAN_PARALLEL worker processes evaluating a cascade stage's candidates (r18, 5(a)); results in list order."""

    def __init__(self, model: object, arm: object, others: tuple, n: int) -> None:
        import multiprocessing as mp  # noqa: PLC0415

        ctx = mp.get_context("spawn")
        self._flag = ctx.Value("b", 0); self._conns, self._procs = [], []; self._gen = 0; self._lock = threading.Lock()
        for k in range(n):
            a, child = ctx.Pipe()
            proc = ctx.Process(target=_candidate_worker_main, args=(child, model, arm, others, self._flag), name=f"bucket-sort-candidates-{k}", daemon=True)
            proc.start(); child.close(); self._conns.append(a); self._procs.append(proc)

    def evaluate(self, planner: Planner, canon: dict, snapshot: SceneSnapshot, start: np.ndarray, instruction: Instruction,
                 q: np.ndarray, dt: float, cands: list) -> list | None:
        """Every candidate's plan, in list order; ``None`` when a worker fails or another planning thread has the pool
        (the caller evaluates them itself: the same plans)."""
        if not all(p.is_alive() for p in self._procs) or not self._lock.acquire(blocking=False):
            return None
        try:
            return self._evaluate(planner, canon, snapshot, start, instruction, q, dt, cands)
        finally:
            self._lock.release()

    def _evaluate(self, planner: Planner, canon: dict, snapshot: SceneSnapshot, start: np.ndarray, instruction: Instruction,
                  q: np.ndarray, dt: float, cands: list) -> list | None:
        m = planner.model
        bits = (np.array(m.geom_contype), np.array(m.geom_conaffinity), np.array(m.geom_group))
        screening = bool(getattr(planner, "screening", False))
        self._gen += 1; gen = self._gen; self._flag.value = 0   # replies of an earlier, cancelled stage are ignored
        msg = lambda i: ((gen, i), snapshot, canon, start, instruction, q, dt, cands[i][0], cands[i][1], cands[i][2], screening, bits)  # noqa: E731
        # one candidate in flight per worker: a large message must not wait on a pipe its worker does not read
        pending = list(range(len(cands))); busy: dict[int, int] = {}
        out: dict[int, object] = {}; failed = False
        try:
            for w, c in enumerate(self._conns):
                if pending:
                    k = pending.pop(0); c.send(msg(k)); busy[w] = k
            while len(out) < len(cands):
                for w, c in enumerate(self._conns):
                    while c.poll(0.0):
                        (g, idx), status, plan = c.recv()
                        if g != gen:
                            continue
                        out[idx] = plan; busy.pop(w, None)
                        if status == "cancelled":
                            raise PlanCancelledError
                        if status != "ok":
                            failed = True
                        if pending:
                            k = pending.pop(0); c.send(msg(k)); busy[w] = k
                _checkpoint()
                if len(out) < len(cands):
                    self._conns[0].poll(0.005)
        except PlanCancelledError:
            self._flag.value = 1   # the workers stop this stage's candidates; the next stage clears it
            raise
        except (BrokenPipeError, OSError):
            return None
        return None if failed else [out[i] for i in range(len(cands))]


def _candidate_pool(planner: Planner) -> _CandidatePool | None:
    key = (id(planner.model), getattr(planner.arm, "name", str(planner.arm)))
    pool = _POOLS.get(key)
    if pool is None:
        try:
            pool = _POOLS[key] = _CandidatePool(planner.model, planner.arm, getattr(planner, "_others", ()), PLAN_PARALLEL)
        except Exception as exc:  # noqa: BLE001 - planning goes on one candidate after another
            logger.warning("Demonstrator: no candidate pool ({}); planning one candidate after another", exc)
            return None
    return pool
"""r18 (5(a)): worker processes that evaluate a cascade stage's candidates in parallel (0: one after another, here)."""
_CANON_FIELDS = ("_go_home", "_held_below", "_lowered", "_no_lowering", "held_width", "unreached", "_wrist_probe")


def _planner_canon(planner: Planner) -> dict:
    """The planner state a candidate starts from: the IK data's coordinates and the per-script fields (r18)."""
    return {"qpos": np.array(planner.ik.data.qpos), **{k: getattr(planner, k, None) for k in _CANON_FIELDS},
            "_has": {k: hasattr(planner, k) for k in _CANON_FIELDS}}


def _planner_restore(planner: Planner, canon: dict) -> None:
    planner.ik.data.qpos[:] = canon["qpos"]
    for k in _CANON_FIELDS:
        if canon["_has"][k]:
            setattr(planner, k, canon[k])
        elif hasattr(planner, k):
            delattr(planner, k)
    planner._mujoco.mj_kinematics(planner.model, planner.ik.data)


def _evaluate_candidate(planner: Planner, canon: dict, snapshot: SceneSnapshot, start: np.ndarray, instruction: Instruction,
                        q: np.ndarray, dt: float, variant: DemoParams, yaw: float, key: tuple) -> JointPlan:
    """One attempt candidate from the canonical state (r18)."""
    planner.load(snapshot)
    _planner_restore(planner, canon)

    def build(slow: dict[str, float]) -> tuple[MotionScript, float]:
        return planner.attempt_script(start, instruction, variant, grasp_yaw=yaw, slow=slow)

    plan = planner.checked(build, q, dt, kind="attempt", target=instruction.object, bucket=instruction.bucket, key=key)
    plan.yaw_grasp = yaw
    return plan


def _evaluate_candidates(planner: Planner, canon: dict, snapshot: SceneSnapshot, start: np.ndarray, instruction: Instruction,
                         q: np.ndarray, dt: float, cands: list) -> Iterator[JointPlan]:
    """The candidates' plans in list order: one after another (stopping after one that passes), or all on the
    candidate pool's workers when PLAN_PARALLEL is set (the same plans: each starts from *canon*)."""
    pool = _candidate_pool(planner) if PLAN_PARALLEL > 0 and len(cands) > 1 else None
    if pool is not None:
        plans = pool.evaluate(planner, canon, snapshot, start, instruction, q, dt, cands)
        if plans is not None:
            yield from plans
            return
    for variant, yaw, key in cands:
        plan = _evaluate_candidate(planner, canon, snapshot, start, instruction, q, dt, variant, yaw, key)
        yield plan
        if plan.ok:
            return


class Planner:
    """Build and check joint plans of one arm for the bucket_sort scene."""

    def __init__(self, model: object, arm: FrankaArm = FR3_ARM, others: tuple[FrankaArm, ...] = ()) -> None:
        """Bind to the scene model and the *arm* that works; planning runs on private data.

        Args:
            model: The bucket_sort model.
            arm: The arm the plans move.
            others: Arms that hold still meanwhile; the plans keep clear of them.
        """
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = model
        self.arm = arm
        self.ik = FrankaIK(model, arm)
        self.dq_max = np.array(arm.dq_max)
        self.ddq_max = np.array(arm.ddq_max)
        self.clearance = _Clearance(model, self.ik.data, arm, others)
        self._others = tuple(others)   # r18: the candidate pool's planners are made alike
        self._wrist_probe: tuple[tuple[float, float], float, float] | None = None
        """The last point :meth:`predicted_q7` solved a tilted arm for: the point, joint 7 and the yaw there."""
        objects = scene_objects(model)
        self.shapes = {name: object_shape(model, name) for name in objects}
        self._object_qpos = {name: int(model.jnt_qposadr[model.joint(f"{name}:joint").id]) for name in objects}
        self._object_body = {name: int(model.body(name).id) for name in objects}
        self._bucket_body = {color: int(model.body(f"bucket_{color}").id) for color in bucket_colors(model)}
        self.held_width = 0.04
        """The width the last attempt script grasps; the place re-plan and the clearance use it."""
        self.unreached = False
        """Whether the script being built has a pose no allowed lean reaches (:meth:`tilt_at`)."""
        self.tick_null_step = 0.01
        """Largest null-space change per IK iteration while following the reference posture."""
        self.tick_limit_gain = 0.0
        self.tick_iters = 3
        self.smooth_ticks = 3.0
        """Standard deviation, in ticks, of the Gaussian that smooths the joint targets after the IK."""
        """Step along the joint-limit gradient per tick, so a joint slows before its soft limit."""

    # -- scene readout ----------------------------------------------------

    def load(self, snapshot: SceneSnapshot) -> None:
        """Put the snapshot's objects and buckets into the private data."""
        data = self.ik.data
        data.qpos[:] = snapshot.qpos
        data.mocap_pos[:] = snapshot.mocap_pos
        data.mocap_quat[:] = snapshot.mocap_quat
        self._mujoco.mj_kinematics(self.model, data)

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the object's position and rotation matrix in the private data (after :meth:`load`)."""
        body = self._object_body[name]
        return self.ik.data.xpos[body].copy(), self.ik.data.xmat[body].reshape(3, 3).copy()

    def bucket_pose(self, color: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the bucket's position and rotation matrix in the private data (after :meth:`load`)."""
        body = self._bucket_body[color]
        return self.ik.data.xpos[body].copy(), self.ik.data.xmat[body].reshape(3, 3).copy()

    def carry_held(self, name: str, live: object, q: np.ndarray) -> None:
        """Move object *name* in the private data from the hand in *live* to the hand at *q* (after :meth:`load`).

        The object keeps its pose relative to the hand, so a plan that starts at *q* (where braking
        stops) sees it where the hand will hold it then, not where it is now.
        """
        body, adr = self._object_body[name], self._object_qpos[name]
        hand_rot = live.xmat[self.ik.hand].reshape(3, 3)
        relative_pos = hand_rot.T @ (live.xpos[body] - self.ik.tcp(live))
        relative_rot = hand_rot.T @ live.xmat[body].reshape(3, 3)
        data = self.ik.data
        self.ik.set_arm(q)
        hand_rot = data.xmat[self.ik.hand].reshape(3, 3)
        data.qpos[adr : adr + 3] = self.ik.tcp() + hand_rot @ relative_pos
        quat = np.zeros(4)
        self._mujoco.mju_mat2Quat(quat, (hand_rot @ relative_rot).ravel())
        data.qpos[adr + 3 : adr + 7] = quat
        self._mujoco.mj_kinematics(self.model, data)

    # -- grasp geometry ---------------------------------------------------

    def grasp_axes(self, name: str, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the closing axis, the long axis and the width to grasp object *name* across.

        An object whose long side is not level gets its flat-lying axes (and is unlikely to be held).

        Returns:
            Horizontal world unit vectors and the width in metres.
        """
        shape = self.shapes[name]
        axes = shape.grasp_axes(rotation)
        if axes is None:
            axes = shape.grasp_axes(np.eye(3))
        return axes  # type: ignore[return-value]

    def standing(self, name: str, rotation: np.ndarray) -> bool:
        """Whether elongated object *name* stands on its end (its long side within ~30 deg of the vertical)."""
        shape = self.shapes[name]
        return abs(float(rotation[2, shape.long_axis])) >= 0.85  # noqa: PLR2004

    def grasp_period(self, name: str, rotation: np.ndarray) -> float:
        """The symmetry of the closing yaw: a standing object is round about the vertical (tried every 30 deg)."""
        return float(np.pi / 6) if self.standing(name, rotation) else grasp_period(name)

    def grasp_axis_yaw(self, name: str, rotation: np.ndarray) -> float:
        """Return the world yaw of the closing axis across the object (one of two, pi apart).

        Returns:
            The yaw in radians.
        """
        closing, _, _ = self.grasp_axes(name, rotation)
        return float(np.arctan2(closing[1], closing[0]))

    def clear_width(self, name: str) -> float:
        """The widest section across a round body's long side (a dumbbell's caps): what the open fingers must clear
        on the way down; 0 for other bodies (their box width is the grasp width)."""
        shape = self.shapes[name]
        if shape.round_width is None:
            return 0.0
        return float(2.0 * max(h for i, h in enumerate(shape.half) if i != shape.long_axis))

    def grasp_width(self, name: str, rotation: np.ndarray) -> float:
        """Return the width the hand closes on, at most the hand's opening."""
        return min(self.ik.grip_open, self.grasp_axes(name, rotation)[2])

    def grasp_point(self, name: str, position: np.ndarray, rotation: np.ndarray, params: DemoParams) -> np.ndarray:
        """Return the TCP's grasp position: the centre, shifted along the long side, at a safe height.

        Returns:
            The position.
        """
        _, along, _ = self.grasp_axes(name, rotation)
        shape = self.shapes[name]
        if self.standing(name, rotation) or shape.round_width is not None:
            # on its end: its middle; a body round about its long side with a waist (the v2 desk's dumbbell resistor):
            # across its waist, as the real fingers close on its bands
            point = position.copy()
        else:
            # Keep the fingers (2 cm wide) on the object.
            limit = shape.half_length - 0.012
            point = position + along * float(np.clip(params.grasp_offset, -limit, limit))
        point[2] = max(float(position[2]), self.arm.min_tcp_z) + params.grasp_dz
        return point

    def predicted_q7(self, yaw: float, xy: np.ndarray) -> float:
        """Estimate joint 7 for the hand at *xy* with closing-axis yaw *yaw* (see :meth:`FrankaIK.continuous_yaw`).

        An upright arm at the origin turns its base toward *xy*. A tilted one (the Duo's) is solved
        for: the hand over *xy* at its home yaw, from home, then joint 7 turns against the rest of the yaw.

        Returns:
            Radians.
        """
        ik, w = self.ik, self.arm.wrist
        if self.arm.upright:
            q1 = float(np.arctan2(xy[1] - ik.base_xy[1], xy[0] - ik.base_xy[0]))
            return float(ik.home[w] + (q1 - ik.home[0]) - (yaw - ik.home_yaw))
        key = (float(xy[0]), float(xy[1]))
        if self._wrist_probe is None or self._wrist_probe[0] != key:
            q, _, _ = ik.solve(ik.home, np.array([*key, WRIST_PROBE_Z]), ik.home_yaw, iters=30)
            self._wrist_probe = (key, float(q[w]), ik.continuous_yaw(q))
        _, q7, solved_yaw = self._wrist_probe
        return float(q7 - (yaw - solved_yaw))

    def yaw_candidates(self, axis_yaw: float, current: float, xy: np.ndarray, *, period: float = np.pi) -> list[float]:
        """Return the closing yaws that line up with *axis_yaw*, nearest to *current* first.

        The hand is symmetric under half a turn, so each axis has a yaw every pi; the candidates are
        the two nearest to *current* whose joint 7 stays inside its range.

        Returns:
            One or two yaws, unwrapped near *current*.
        """
        base = wrap_near(axis_yaw, current, period)
        options = [base + k * period for k in (-2, -1, 0, 1, 2)]
        w = self.arm.wrist
        limit_lo, limit_hi = self.ik.lower[w] + 0.1, self.ik.upper[w] - 0.1
        feasible = [y for y in options if limit_lo <= self.predicted_q7(y, xy) <= limit_hi]
        feasible.sort(key=lambda y: abs(y - current))
        if feasible:
            return feasible[:2]
        # none keeps the 0.1 rad margin: the two that pass the joint's own limits by the most (or miss them by the
        # least), not the one nearest the current yaw, which may be far outside (a grasp the IK then cannot reach)
        def overshoot(y: float) -> float:
            q7 = self.predicted_q7(y, xy)
            return max(self.ik.lower[w] - q7, q7 - self.ik.upper[w])
        return sorted(options, key=overshoot)[:2]

    def all_yaws(self, axis_yaw: float, current: float, xy: np.ndarray, *, period: float = np.pi) -> list[float]:
        """Every closing yaw lining up with *axis_yaw* (one per *period*) whose predicted joint 7 stays inside its
        hard limits, nearest to *current* first: the wider set :meth:`plan_attempt` falls back on.

        Returns:
            Yaws, unwrapped near *current*.
        """
        base = wrap_near(axis_yaw, current, period)
        w = self.arm.wrist
        options = [base + k * period for k in range(-4, 5)]
        inside = [y for y in options if self.ik.lower[w] <= self.predicted_q7(y, xy) <= self.ik.upper[w]]
        return sorted(inside, key=lambda y: abs(y - current))

    def key_margin(self, q: np.ndarray, instruction: Instruction, params: DemoParams, yaw: float) -> float:
        """Return the joint-limit margin of an attempt's key postures with grasp yaw *yaw* (after :meth:`load`).

        Returns:
            The smallest margin (rad) at the grasp and over the bucket, each solved from the one before.
        """
        name = instruction.object
        position, rotation = self.object_pose(name)
        grasp = self.grasp_point(name, position, rotation, params)
        key, _, _ = self.ik.optimize(q, grasp, yaw, iters=40, tilt=self.tilt_at(grasp, yaw, grasp=True) or 0.0)
        margin = self.ik.margin(key)
        bucket_pos, bucket_rot = self.bucket_pose(instruction.bucket)
        across = float(np.arctan2(bucket_rot[1, 0], bucket_rot[0, 0]))
        release_yaw = self.yaw_candidates(across, yaw, bucket_pos[:2])[0] if params.reorient else yaw
        over = np.array([bucket_pos[0], bucket_pos[1], params.release_z])
        tilt = self.tilt_at(over, release_yaw) or 0.0
        key, e_pos, _ = self.ik.optimize(key, over, release_yaw, iters=40, tilt=tilt)
        return min(margin, self.ik.margin(key)) - (1.0 if e_pos > IK_POS_TOL else 0.0)

    def tilt_at(self, pos: np.ndarray, yaw: float, *, grasp: bool = False) -> float | None:
        """Return how far the hand leans at *pos* closing along *yaw*: the least that reaches it.

        At the *grasp* (over the object and down to it) the hand leans :data:`GRASP_TILT` at most.
        When no lean reaches, the script uses the most allowed and :attr:`unreached` marks it, so
        :meth:`checked` fails the plan and it never runs.

        Returns:
            Radians, ``None`` for an arm that keeps its fingers down.
        """
        if self.arm.max_tilt <= 0.0:
            return None
        limit = min(GRASP_TILT, self.arm.max_tilt) if grasp else self.arm.max_tilt
        found = self.ik.least_tilt(self.ik.home, pos, yaw, limit)
        if found is None:
            self.unreached = True
            return limit
        return found[0]

    # -- building ---------------------------------------------------------

    def attempt_script(  # noqa: PLR0914
        self,
        start: np.ndarray,
        instruction: Instruction,
        params: DemoParams,
        *,
        grasp_yaw: float,
        slow: dict[str, float] | None = None,
    ) -> tuple[MotionScript, float]:
        """Script one pick-and-place-and-home attempt from *start* (after :meth:`load`).

        Args:
            start: ``(x, y, z, yaw, grip)`` now.
            instruction: What to move where.
            params: Heights, speeds and pauses.
            grasp_yaw: The closing yaw to grasp with (unwrapped).
            slow: Stretch the moves of these phases by these factors.

        Returns:
            The script and the release yaw.
        """
        p = params
        name = instruction.object
        position, rotation = self.object_pose(name)
        grasp = self.grasp_point(name, position, rotation, p)
        self._held_below = self._below_tcp(name, grasp)
        self.held_width = self.grasp_width(name, rotation)
        script = MotionScript(start, slow)
        script.ramp = params.motion_ramp
        x0 = start[:3]
        # open to clear the widest section, not the closing width: a dumbbell's caps are wider than its waist (r11, T6:
        # a narrow opening at the waist width clipped a standing resistor's cap on the way down)
        open_width = min(self.ik.grip_open, max(self.held_width, self.clear_width(name)) + p.open_margin)
        pre = grasp + np.array([0.0, 0.0, p.pregrasp_height])
        # Get up and clear first when starting low and away from the object (a retry).
        if x0[2] < pre[2] - 0.02 and np.linalg.norm(x0[:2] - grasp[:2]) > 0.03:  # noqa: PLR2004
            up = np.array([x0[0], x0[1], max(pre[2], x0[2] + 0.05)])
            script.add(
                "approach",
                pos=up,
                grip=max(start[4], open_width),
                duration=move_duration(up[2] - x0[2], p.lift_speed, 0.5, ramp=p.motion_ramp),
                tilt=self.tilt_at(up, float(script.end[3])),
            )
        here, leaning = script.end[:3], script.end_tilt
        tilt = self.tilt_at(pre, grasp_yaw, grasp=True)
        script.add(
            "approach",
            pos=pre,
            yaw=grasp_yaw,
            grip=open_width,
            duration=max(
                move_duration(float(np.linalg.norm(pre - here)), p.transit_speed, 0.8, ramp=p.motion_ramp),
                move_duration(grasp_yaw - script.end[3], p.yaw_speed, 0.8, ramp=p.motion_ramp),
                move_duration((tilt or 0.0) - script.end_tilt, p.yaw_speed, 0.8, ramp=p.motion_ramp),
            ),
            blend=p.blend,
            tilt=tilt,
        )
        # The descent starts once the approach has leaned the hand to its grasp tilt: blended into
        # a lean that is still changing, it would lean more than GRASP_TILT on the way down.
        script.add(
            "descend",
            pos=grasp,
            duration=move_duration(p.pregrasp_height, p.fine_speed, 0.6),   # a MIN_JERK_PHASES move
            blend=0.0 if tilt is not None and abs(tilt - leaning) > 1e-9 else p.blend,  # noqa: PLR2004
            tilt=self.tilt_at(grasp, grasp_yaw, grasp=True),
        )
        squeeze = self.arm.squeeze
        width_from = float(script.end[4]); width_to = 0.0 if squeeze is None else max(0.0, self.held_width - squeeze)
        touch = inverse_min_jerk((width_from - self.held_width) / max(width_from - width_to, 1e-9)) * p.close_s   # the fingers: min jerk
        early = 0.0
        # r15: not for a round body (a resistor rolls into the grip while the fingers squeeze it; rising then left it held
        # off-centre and it fell outside the bucket, default seed 5783)
        close_drift = p.close_drift_mps if self.clear_width(name) <= 0.0 else 0.0
        if close_drift > 0.0 and script.moves:
            # r15 (operator drift): the fingers start closing while the descent ends; the descent is over CONTACT_MARGIN_S
            # before the fingers reach the object's width, so the hand never moves sideways or down against it
            early = min(max(0.0, touch - CONTACT_MARGIN_S), CLOSE_EARLY_MAX * script.moves[-1].duration)
        script.add("close", grip=width_to, duration=p.close_s, blend=early / script.moves[-1].duration if early > 0.0 else 0.0)
        if close_drift > 0.0:
            # r15 (operator drift): rising through the settle. r16: only once the close has ended (rising during the
            # squeeze shifted flat objects in the grip too: default 5703 and 5747, a battery dropped outside)
            rise_s = p.settle_s
            rise = min(close_drift * rise_s, 0.5 * p.lift_height)
            script.add("close", pos=grasp + np.array([0.0, 0.0, rise]), duration=rise_s)
        else:
            script.pause("close", p.settle_s)
        lift = grasp + np.array([0.0, 0.0, p.lift_height])
        script.add(
            "lift",
            pos=lift,
            duration=move_duration(p.lift_height, p.lift_speed, 0.6, p.held_accel, ramp=p.motion_ramp),
            tilt=self.tilt_at(lift, grasp_yaw),
        )

        yaw_release = self._add_place(script, instruction, params, grasp_yaw, (position - grasp)[:2])
        self.add_return(script, p)
        return script, yaw_release

    def place_script(
        self,
        start: np.ndarray,
        instruction: Instruction,
        params: DemoParams,
        offset: np.ndarray,
        slow: dict[str, float] | None = None,
    ) -> tuple[MotionScript, float]:
        """Script the rest of an attempt from *start* with the object in the hand (after :meth:`load`).

        Args:
            start: ``(x, y, z, yaw, grip)`` now.
            instruction: What to move where.
            params: Heights, speeds and pauses.
            offset: The held object's centre from the TCP, horizontally.
            slow: Stretch the moves of these phases by these factors.

        Returns:
            The script and the release yaw.
        """
        script = MotionScript(start, slow)
        script.ramp = params.motion_ramp
        if start[2] < params.carry_z - REPLAN_SHIFT:  # re-planned on the way down into the bucket: rise first
            rise = params.carry_z - float(start[2])
            up = np.array([start[0], start[1], params.carry_z])
            script.add(
                "carry",
                pos=up,
                duration=move_duration(rise, params.lift_speed, 0.5, ramp=params.motion_ramp),
                tilt=self.tilt_at(up, float(start[3])),
            )
        yaw_release = self._add_place(script, instruction, params, float(start[3]), np.asarray(offset)[:2])
        self.add_return(script, params)
        return script, yaw_release

    def _add_place(  # noqa: PLR0914
        self, script: MotionScript, instruction: Instruction, p: DemoParams, yaw: float, offset: np.ndarray
    ) -> float:
        """Add the carry over the bucket, the descent into its opening and the release.

        The object's centre, not the TCP, goes over the release point; *offset* (the centre from the
        TCP) turns with the hand.

        Returns:
            The release yaw.
        """
        bucket_pos, bucket_rot = self.bucket_pose(instruction.bucket)
        (x_lo, x_hi), (y_lo, y_hi), _ = BUCKET_INTERIOR
        local = np.array([(x_lo + x_hi) / 2 + p.place_offset[0], (y_lo + y_hi) / 2 + p.place_offset[1], 0.0])
        place = bucket_pos + bucket_rot @ local
        across = float(np.arctan2(bucket_rot[1, 0], bucket_rot[0, 0]))
        yaw_release = self.yaw_candidates(across + p.release_yaw_noise, yaw, place[:2])[0] if p.reorient else yaw
        turn = yaw_release - yaw
        c, s = np.cos(turn), np.sin(turn)
        offset = np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]])
        over = np.array([place[0] - offset[0], place[1] - offset[1], p.carry_z])
        tilt = self.tilt_at(over, yaw_release)
        script.add(
            "carry",
            pos=over,
            yaw=yaw_release,
            duration=max(
                move_duration(float(np.linalg.norm(over - script.end[:3])), p.transit_speed, 0.8, p.held_accel, ramp=p.motion_ramp),
                move_duration(turn, p.yaw_speed, 0.8, ramp=p.motion_ramp),
                move_duration((tilt or 0.0) - script.end_tilt, p.yaw_speed, 0.8, ramp=p.motion_ramp),
            ),
            blend=p.blend,
            tilt=tilt,
        )
        release_z = self.release_height(place, p)
        release = np.array([over[0], over[1], release_z])
        script.add(
            "lower",
            pos=release,
            # (at the lift speed: the release now goes up to 9 cm deeper into the bucket, the fine speed made it slow)
            duration=move_duration(p.carry_z - release_z, max(p.fine_speed, p.lift_speed), 0.5, p.held_accel),   # MIN_JERK_PHASES
            blend=p.blend,
            tilt=self.tilt_at(release, yaw_release),
        )
        if p.release_creep_mps > 0.0 and p.release_pause_s > 0.0:
            # r15 (operator drift): the hand keeps sinking a little before it opens, and rises once the fingers clear
            creep = min(p.release_creep_mps * p.release_pause_s, RELEASE_CREEP_MAX_M)
            script.add("release", pos=release - np.array([0.0, 0.0, creep]), duration=p.release_pause_s)
        else:
            script.pause("release", p.release_pause_s)
        width_from = float(script.end[4]); width_to = min(self.ik.grip_open, self.held_width + p.release_margin)
        # r18 (D7): the fingers open at the operator's speed (DemoParams.release_open_mps), r1-r17 over a fixed 0.45 s
        open_s = 0.45 if p.release_open_mps is None else max(0.1, abs(width_to - width_from) / p.release_open_mps)
        # the rise waits until the fingers clear the object's WIDEST section (a resistor held at its waist has wider caps)
        widest = max(self.held_width, self.clear_width(instruction.object))
        clear = inverse_min_jerk((widest + RELEASE_CLEAR_M - width_from) / max(width_to - width_from, 1e-9)) * open_s
        # r15 (operator drift): the fingers start opening while the hand still sinks (the object stays held until they clear it)
        early = min(max(0.0, clear - CONTACT_MARGIN_S), 0.5 * p.release_pause_s) if p.release_creep_mps > 0.0 and p.release_pause_s > 0.0 else 0.0
        script.add("release", grip=width_to, duration=open_s, blend=early / p.release_pause_s if early > 0.0 else 0.0)
        if p.release_rise_mps > 0.0:
            overlap = max(0.0, open_s - clear)
            rise_s = overlap + p.after_release_s
            rise = min(p.release_rise_mps * rise_s, RELEASE_RISE_MAX_M)
            script.add("release", pos=script.end[:3] + np.array([0.0, 0.0, rise]), duration=max(rise_s, 1e-3), blend=overlap / open_s)
        else:
            script.pause("release", p.after_release_s)
        return yaw_release

    RELEASE_GAP, RELEASE_DROP_MAX = 0.03, 0.09
    """The release (r11, the r09/r10 3.3 mm release penetration: a battery dropped 15 cm onto an empty bucket's floor):
    the held object's bottom goes to RELEASE_GAP above whatever lies under the release point (the floor or the
    contents), the TCP at most RELEASE_DROP_MAX (9 cm) below the drawn ``release_z`` (the plan's checks still guard the rims)."""

    def _below_tcp(self, name: str, grasp: np.ndarray) -> float:
        """How far the object reaches below the TCP once grasped at *grasp* (it lies on the table now)."""
        import mujoco  # noqa: PLC0415
        m, d = self.model, self.ik.data
        body = int(m.body(name).id); low = np.inf
        for g in np.flatnonzero(np.asarray(m.geom_bodyid) == body):
            if int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0:
                continue
            low = min(low, float(d.geom_xpos[g][2] - m.geom_rbound[g]) if m.geom_rbound[g] > 0 else float(d.geom_xpos[g][2]))
        aabb_low = float(d.xpos[body][2] - self.shapes[name].half.min()) if name in self.shapes else low
        bottom = max(low, aabb_low) if np.isfinite(low) else aabb_low
        return max(0.0, float(grasp[2]) - bottom)

    def _ray_start_z(self) -> float:
        """Where the release's downward ray starts: 3 cm above the bucket mesh's top (0.115 m without a bucket mesh)."""
        if not hasattr(self, "_ray_z"):
            m = self.model; z = 0.115
            try:
                k = int(m.mesh("bucket").id); a, n = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
                tops = [float(m.geom_pos[g][2]) + float(m.mesh_vert[a:a + n, 2].max()) for g in range(m.ngeom)
                        if m.geom_type[g] == 7 and int(m.geom_dataid[g]) == k]   # mjGEOM_MESH
                if tops: z = max(tops) + 0.03
            except KeyError:
                pass
            self._ray_z = z
        return self._ray_z

    def release_height(self, place: np.ndarray, p: DemoParams) -> float:
        """The TCP height of the release over *place* (see :attr:`RELEASE_GAP`)."""
        import mujoco  # noqa: PLC0415
        below = getattr(self, "_held_below", None)
        self._lowered = False
        if below is None or getattr(self, "_no_lowering", False):
            return p.release_z
        m, d = self.model, self.ik.data
        # r17 (D4, 622597): the ray starts above the bucket's own top (the drawn mesh's height in this model), not at a
        # fixed 0.115 m: with the 120 mm buckets, contents heaped near the rim lay above that start, the release was
        # lowered onto them and a held resistor was pressed 3.24 mm into a content item
        start = np.array([place[0], place[1], self._ray_start_z()]); geomid = np.array([-1], dtype=np.int32)
        groups = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
        dist = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), groups, 1, -1, geomid)
        if dist < 0:
            return p.release_z
        surface = float(start[2] - dist)
        z = float(min(p.release_z, max(surface + below + self.RELEASE_GAP, p.release_z - self.RELEASE_DROP_MAX)))
        self._lowered = z < p.release_z - 1e-6
        return z

    def add_return(self, script: MotionScript, p: DemoParams) -> None:
        """Add the retreat straight up; :meth:`solve` adds the joint-space way home after it. With ``p.go_home``
        false (more objects of the instruction to sort), only the retreat straight up: the plan ends there."""
        self._go_home = bool(getattr(p, "go_home", True))
        here = script.end[:3]
        retreat_z = max(here[2] + 0.06, self.arm.heights.retreat_z)
        up = np.array([here[0], here[1], retreat_z])
        script.add(
            "retreat",
            pos=up,
            duration=move_duration(retreat_z - here[2], p.lift_speed, 0.5, ramp=p.motion_ramp),
            tilt=self.tilt_at(up, float(script.end[3])),
        )
        if self.arm.max_tilt > 0.0 and self._go_home:
            # Home is low for a six-joint arm: go there in a straight line, over the buckets'
            # rims, not in joint space, which dips the hand. The joint-space way home is then short.
            ik = self.ik
            home, _ = ik.fk(ik.home)
            yaw = wrap_near(ik.continuous_yaw(ik.home), float(script.end[3]), np.pi)
            script.add(
                "retreat",
                pos=home,
                yaw=yaw,
                tilt=ik.tilt(),
                duration=max(
                    move_duration(float(np.linalg.norm(home - up)), p.transit_speed, 0.8, ramp=p.motion_ramp),
                    move_duration(yaw - script.end[3], p.yaw_speed, 0.8, ramp=p.motion_ramp),
                    move_duration(script.end_tilt, p.yaw_speed, 0.8, ramp=p.motion_ramp),
                ),
                blend=p.blend,
            )

    def checked(
        self,
        build: Callable[[dict[str, float]], tuple[MotionScript, float]],
        q: np.ndarray,
        dt: float,
        *,
        kind: PlanKind,
        target: str,
        bucket: str,
        held_from: float | None = None,
        key: object = None,
    ) -> JointPlan:
        """Build a script, solve it, and stretch the moves that are too fast until the joints keep to the limits.

        Args:
            build: Returns the script and its release yaw for a ``slow`` map (phase to stretch factor).
            q: Joint angles at the start.
            dt: Control period.
            kind: What the plan is for.
            target: The object being picked or held.
            bucket: The bucket it goes to; the plan records its pose.
            held_from: When the hand holds the object; ``None`` from the first ``lift`` move.

        Returns:
            The last plan solved (``problems`` says whether it passed).
        """
        # r16: the full pass after a screening pass that found no plan reuses each candidate whose screened sweep ran in
        # full (its result is the full one), leaving the data as that sweep did; only the cut ones are solved again
        reuse = getattr(self, "_screen_reuse", None)
        if key is not None and reuse is not None and key in reuse:
            plan = reuse[key]
            if self._as_swept(plan, target, plan.held_ticks):
                return plan
        plan = self._checked(build, q, dt, kind=kind, target=target, bucket=bucket, held_from=held_from)
        record = getattr(self, "_screen_record", None)
        if key is not None and record is not None and not getattr(plan, "screen_partial", False):
            record[key] = plan
        return plan

    def _checked(
        self,
        build: Callable[[dict[str, float]], tuple[MotionScript, float]],
        q: np.ndarray,
        dt: float,
        *,
        kind: PlanKind,
        target: str,
        bucket: str,
        held_from: float | None = None,
    ) -> JointPlan:
        slow: dict[str, float] = {}
        plan: JointPlan | None = None
        bucket_pos, bucket_rot = self.bucket_pose(bucket)
        for _ in range(4):
            self.unreached = False
            script, yaw_release = build(slow)
            start = script.first_time("lift") + 0.1 if held_from is None else held_from
            release = next((m.t0 for m in script.moves if m.phase == "release"), script.t_end)
            # r16: the clearance sweep only for the plan kept (a too-fast plan is stretched and solved again)
            plan = self.solve(script, q, dt, kind=kind, target=target, held_from=start, held_until=release, clearance=False)
            plan.yaw_release = yaw_release
            velocity = np.diff(plan.q, axis=0) / dt
            speed = (np.abs(velocity) / self.dq_max).max(axis=1)
            accel = (np.abs(np.diff(velocity, axis=0)) / (dt * self.ddq_max)).max(axis=1)
            factors: dict[str, float] = {}
            for k in np.flatnonzero(speed > 0.9 * SPEED_FRACTION):
                ratio = float(speed[k] / (0.85 * SPEED_FRACTION))
                factors[plan.phases[k + 1]] = max(factors.get(plan.phases[k + 1], 1.0), ratio)
            for k in np.flatnonzero(accel > 0.9 * ACCEL_FRACTION):
                ratio = float(np.sqrt(accel[k] / (0.85 * ACCEL_FRACTION)))
                factors[plan.phases[k + 1]] = max(factors.get(plan.phases[k + 1], 1.0), ratio)
            factors.pop("return", None)  # the way home is timed from its own speed limit
            too_fast = plan.peak_speed_ratio > SPEED_FRACTION or plan.peak_accel_ratio > ACCEL_FRACTION
            if not too_fast or not factors:
                break
            for phase, factor in factors.items():
                slow[phase] = min(MAX_STRETCH, slow.get(phase, 1.0) * factor)
        assert plan is not None  # noqa: S101 - the loop runs at least once
        plan.held_ticks = self._held_ticks(start, release, dt)   # r16: what the sweep used, for an exact re-check later
        plan.held_width = float(self.held_width)
        self._check_clearance(plan, target, plan.held_ticks)
        if getattr(self, "_lowered", False) and not getattr(self, "_no_lowering", False) and any(
            pr.startswith("bucket clearance") for pr in plan.problems
        ):   # the deeper release (release_height) reaches a rim: the same plan at the drawn release height instead
            self._no_lowering = True
            try:
                return self._checked(build, q, dt, kind=kind, target=target, bucket=bucket, held_from=held_from)
            finally:
                self._no_lowering = False
        plan.bucket_pose = (bucket_pos, float(np.arctan2(bucket_rot[1, 0], bucket_rot[0, 0])))
        self._annotate(plan)
        if self.unreached:
            plan.unreached = True
            limit = np.degrees(min(GRASP_TILT, self.arm.max_tilt))
            plan.problems.append(
                f"no lean reaches a key pose (at most {limit:.0f} deg at the grasp, "
                f"{np.degrees(self.arm.max_tilt):.0f} deg elsewhere)"
            )
        return plan

    @staticmethod
    def _annotate(plan: JointPlan) -> None:
        """Mark the ticks where the demonstrator checks the grasp, the hold and the bucket."""
        lift_ticks = [k for k, phase in enumerate(plan.phases) if phase == "lift"]
        if lift_ticks:
            z0 = plan.cartesian[lift_ticks[0], 2]
            above = [k for k in lift_ticks if plan.cartesian[k, 2] >= z0 + LIFT_CHECK_DZ]
            plan.lift_check = above[0] if above else lift_ticks[-1]
        release_ticks = [k for k, phase in enumerate(plan.phases) if phase == "release"]
        plan.hold_until = release_ticks[0] if release_ticks else None
        carry_ticks = [k for k, phase in enumerate(plan.phases) if phase == "carry"]
        plan.carry_start = carry_ticks[0] if carry_ticks else None
        if plan.kind == "place":
            plan.lift_check = None

    def solve(  # noqa: PLR0914
        self,
        script: MotionScript,
        q_start: np.ndarray,
        dt: float,
        *,
        kind: PlanKind,
        target: str,
        held_from: float | None = None,
        held_until: float | None = None,
        home_speed: float = 0.3,
        home_blend: float = 0.3,
        clearance: bool = True,
    ) -> JointPlan:
        """Turn *script* into joint targets every *dt*, then the way home, and check them (after :meth:`load`).

        Args:
            script: The motion.
            q_start: Joint angles at the start (the IK's first seed).
            dt: Control period.
            kind: What the plan is for.
            target: The object being picked (and held from *held_from* to *held_until*, in seconds).
            held_from: When the hand starts carrying the object.
            held_until: When it lets go.
            home_speed: After the script, the arm returns home (:attr:`FrankaIK.home`) in joint space, each
                joint at most this fraction of its speed and acceleration limits.
            home_blend: Fraction of the script's last move that the way home overlaps.
            clearance: Sweep the clearances too (:meth:`checked` defers it to the plan it keeps).

        Returns:
            The plan; ``problems`` lists every check that failed.
        """
        q_start = np.asarray(q_start, dtype=np.float64).copy()
        q_script, pos_error, rot_error = self._follow_script(script, q_start, dt)
        n_script = len(q_script)
        cartesian = script.at(np.arange(n_script) * dt)
        phases = script.phases(np.arange(n_script) * dt)
        if not getattr(self, "_go_home", True):   # more objects to sort: no way home, the plan ends above the bucket
            home_speed = 0.0
        # The way home, in joint space: a minimum-jerk move of every joint onto the ready pose,
        # starting while the last move of the script ends.
        delta = self.ik.home - q_script[-1]
        if home_speed <= 0.0:
            delta = np.zeros_like(delta)
        home_s = 1.0 if home_speed <= 0.0 else max(
            1.2,
            1.875 * float(np.max(np.abs(delta) / (home_speed * self.dq_max))),
            float(np.sqrt(5.77 * np.max(np.abs(delta) / (home_speed * self.ddq_max)))),
        )
        last = script.moves[-1] if script.moves else None
        start = script.t_end - (home_blend * last.duration if last is not None and np.any(last.delta) else 0.0)
        ticks = max(n_script, int(np.ceil((start + home_s) / dt)) + 1)
        ramp = min_jerk((np.arange(ticks) * dt - start) / home_s)
        extra = ticks - n_script
        q = np.vstack([q_script, np.repeat(q_script[-1:], extra, axis=0)]) + ramp[:, None] * delta
        cartesian = np.vstack([cartesian, np.repeat(cartesian[-1:], extra, axis=0)])
        grip = cartesian[:, 4] + ramp * (self.ik.grip_open - cartesian[-1, 4])
        home_tick = int(np.ceil(start / dt))
        all_phases: list[Phase] = [*phases, *([phases[-1]] * extra)]
        returning: list[Phase] = ["return"] * (ticks - home_tick)
        all_phases = [*all_phases[:home_tick], *returning]
        if home_speed <= 0.0:   # no way home: exactly the script (the retreat's end is at rest)
            q, cartesian, grip, all_phases = q_script, cartesian[:n_script], cartesian[:n_script, 4].copy(), list(phases)
        plan = JointPlan(
            kind=kind,
            dt=dt,
            q=q,
            grip=grip,
            phases=all_phases,
            cartesian=cartesian,
            pos_error=pos_error,
            rot_error=rot_error,
        )
        self._check_joints(plan)
        held = self._held_ticks(held_from, held_until, dt)
        if clearance or not self._as_swept(plan, target, held):
            self._check_clearance(plan, target, held)
        return plan

    @staticmethod
    def _held_ticks(held_from: float | None, held_until: float | None, dt: float) -> tuple[int, int] | None:
        if held_from is None or held_until is None:
            return None
        return (int(np.ceil(held_from / dt)), int(np.floor(held_until / dt)))

    def _as_swept(self, plan: JointPlan, target: str, held: tuple[int, int] | None) -> bool:
        """Leave the private data as a full clearance sweep of *plan* leaves it (the arm at the last swept tick, the
        object at rest, the kinematics of that tick), since the next script is built from that data. ``False`` when
        that tick holds the object (then only the sweep itself reproduces it)."""
        k = ((len(plan.q) - 1) // CHECK_EVERY) * CHECK_EVERY
        if held is not None and held[0] <= k < held[1]:
            return False
        ik, data = self.ik, self.ik.data
        adr = self._object_qpos[target]; rest = np.array(data.qpos[adr : adr + 7])
        ik.set_arm(plan.q[k], float(plan.grip[k]))
        self._mujoco.mj_kinematics(self.model, data)
        data.qpos[adr : adr + 7] = rest
        return True

    def _follow_script(  # noqa: PLR0914 - one numerical routine; splitting it hides the algorithm
        self, script: MotionScript, q_start: np.ndarray, dt: float
    ) -> tuple[np.ndarray, float, float]:
        """Solve the IK every tick along *script*, following its reference posture in the null space.

        Key postures come first: at the end of each move, the IK solution furthest from the joint
        limits on the previous key's side of the null space. The reference posture moves between
        them like the hand does. The targets are then smoothed, and any first-tick correction fades in.

        Returns:
            ``(targets (ticks, joints), largest position error, largest orientation error)``.
        """
        ik, data = self.ik, self.ik.data
        key = q_start
        for move in script.moves:
            tilt = float(move.target[5]) if len(move.target) > 5 else 0.0  # noqa: PLR2004
            if not np.any(move.delta[:4]) and not np.any(move.delta[5:]):
                move.q_delta = None
                continue
            new_key, _, _ = ik.optimize(key, move.target[:3], float(move.target[3]), tilt=tilt)
            move.q_delta = new_key - key
            key = new_key
        n_script = max(2, int(np.ceil(script.t_end / dt)) + 1)
        times = np.arange(n_script) * dt
        cartesian = script.at(times)
        tilts = cartesian[:, 5] if cartesian.shape[1] > 5 else np.zeros(n_script)  # noqa: PLR2004
        reference = script.reference(times, q_start)
        q_script = np.zeros((n_script, len(q_start)))
        current = q_start.copy()
        pos_error = rot_error = 0.0
        step_limit = MAX_TICK_SPEED * self.dq_max * dt
        for k in range(n_script):
            solved, e_pos, e_rot = ik.solve(
                current,
                cartesian[k, :3],
                float(cartesian[k, 3]),
                tilt=float(tilts[k]),
                iters=self.tick_iters if k else 30,
                posture=reference[k] if self.tick_null_step > 0.0 else None,
                null_step=self.tick_null_step,
                limit_gain=self.tick_limit_gain,
            )
            # Never faster than MAX_TICK_SPEED of the limits, even where the IK would jump.
            step = np.clip(solved - current, -step_limit, step_limit)
            if np.any(step != solved - current):
                ik.set_arm(current + step)
                e_pos = float(np.linalg.norm(cartesian[k, :3] - ik.tcp()))
            current += step
            q_script[k] = current
            pos_error, rot_error = max(pos_error, e_pos), max(rot_error, e_rot)
        if self.smooth_ticks > 0.0:
            # Round off the kinks a joint limit or the null space leaves.
            q_script = gaussian_smooth(q_script, self.smooth_ticks)
        # The first IK tick snaps a hand that is not quite level (after braking, say) onto the task;
        # fade that correction in over START_BLEND_S instead, so the targets start where the arm is.
        fade = 1.0 - min_jerk(times / START_BLEND_S)
        q_script -= fade[:, None] * (q_script[0] - q_start)[None, :]
        # Measure the task again on the targets as they will run (after the fade-in).
        for k in range(int(np.ceil(START_BLEND_S / dt)), n_script, CHECK_EVERY):
            ik.set_arm(q_script[k])
            target = ik.target_rotation(cartesian[k, :3], float(cartesian[k, 3]), float(tilts[k]))
            rot = data.xmat[ik.hand].reshape(3, 3) @ target.T
            pos_error = max(pos_error, float(np.linalg.norm(cartesian[k, :3] - ik.tcp())))
            rot_error = max(rot_error, float(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))))
        return q_script, pos_error, rot_error

    def _check_joints(self, plan: JointPlan) -> None:
        """Record the IK residual, the joint-limit margin and the peak joint speed and acceleration."""
        ik, q = self.ik, plan.q
        if plan.pos_error > IK_POS_TOL or plan.rot_error > IK_ROT_TOL:
            plan.problems.append(f"IK residual {plan.pos_error * 1000:.1f} mm, {np.degrees(plan.rot_error):.1f} deg")
        plan.limit_margin = float(min((q - ik.hard_lower).min(), (ik.hard_upper - q).min()))
        smooth = joint_smoothness(q, plan.dt)
        plan.peak_speed_ratio = float((smooth["velocity"] / self.dq_max).max())
        plan.peak_accel_ratio = float((smooth["acceleration"] / self.ddq_max).max())
        if plan.peak_speed_ratio > SPEED_FRACTION:
            plan.problems.append(f"joint speed {plan.peak_speed_ratio:.0%} of the {self.arm.label} limit")
        if plan.peak_accel_ratio > ACCEL_FRACTION:
            plan.problems.append(f"joint acceleration {plan.peak_accel_ratio:.0%} of the {self.arm.label} limit")

    def arm_clearance(self, q: np.ndarray, snapshot: SceneSnapshot) -> float:
        """Return how close the working arm comes to the arms holding still, at each row of *q* (capped at 5 cm).

        Every row is checked: braking is short. The arms holding still stand as in *snapshot*.

        Returns:
            Metres; infinite without other arms.
        """
        clearance = self.clearance
        if clearance.arm is None or clearance.other_arms is None:
            return float(np.inf)
        self.load(snapshot)
        worst = float(np.inf)
        for row in np.atleast_2d(q):
            _checkpoint()
            self.ik.set_arm(row)
            worst = min(worst, clearance._min_distance(clearance.arm, clearance.other_arms))  # noqa: SLF001
        return worst

    def _check_clearance(self, plan: JointPlan, target: str, held: tuple[int, int] | None) -> None:
        """Sweep the hand (and, between the *held* ticks, the object in it) and record the clearances."""
        ik, data = self.ik, self.ik.data
        adr = self._object_qpos[target]
        position, rotation = self.object_pose(target)
        relative: tuple[np.ndarray, np.ndarray] | None = None
        rest_qpos = np.array(data.qpos[adr : adr + 7])
        worst: dict[str, float] = dict.fromkeys(CLEARANCE, float(np.inf))
        quat = np.zeros(4)
        screening = getattr(self, "screening", False)
        # While a deeper release may still be retried at the drawn height (:meth:`checked` retries when a bucket clearance
        # problem is found), that problem decides what happens next: the sweep runs until it finds one, whatever else
        # fails; otherwise it stops at any problem, or is not needed at all when the plan fails already.
        retry_possible = getattr(self, "_lowered", False) and not getattr(self, "_no_lowering", False)
        stop_keys = [key for key in CLEARANCE if key.startswith("bucket")] if retry_possible else list(CLEARANCE)
        if screening and plan.problems and not retry_possible and self._as_swept(plan, target, held):   # r16: it fails already
            plan.clearance = {}
            plan.screen_partial = True
            return
        stopped = False
        for k in range(0, len(plan.q), CHECK_EVERY):
            _checkpoint()
            holding = held is not None and held[0] <= k < held[1]
            ik.set_arm(plan.q[k], self.held_width if holding else float(plan.grip[k]))
            if holding:
                hand_pos, hand_rot = ik.tcp(), data.xmat[ik.hand].reshape(3, 3).copy()
                if relative is None:
                    # As grasped: the object sits where it lies now, relative to the closed hand.
                    relative = (hand_rot.T @ (position - hand_pos), hand_rot.T @ rotation)
                data.qpos[adr : adr + 3] = hand_pos + hand_rot @ relative[0]
                self._mujoco.mju_mat2Quat(quat, (hand_rot @ relative[1]).ravel())
                data.qpos[adr + 3 : adr + 7] = quat
            else:
                data.qpos[adr : adr + 7] = rest_qpos
            self._mujoco.mj_kinematics(self.model, data)
            for key, value in self.clearance.measure(target=target, held=holding).items():
                worst[key] = min(worst[key], value)
            if not holding and plan.phases[k] in ("approach", "descend"):   # r17: the hand against the target before the close
                worst["target"] = min(worst["target"], self.clearance._min_distance(self.clearance.robot, self.clearance.objects[target]))
            if screening and any(worst[key] < self._budget(key) for key in stop_keys):
                stopped = True
                plan.screen_partial = True
                break   # r16: a problem for certain; the sweep's remaining ticks cannot make it pass
        data.qpos[adr : adr + 7] = rest_qpos
        if stopped and not self._as_swept(plan, target, held):   # the last tick holds the object (rare): the full sweep
            self.screening = False
            try:
                self._check_clearance(plan, target, held)
            finally:
                self.screening = True
            return
        plan.clearance = {key: value for key, value in worst.items() if np.isfinite(value)}
        plan.target_budget = self._budget("target")   # r17: the scene's, for _severity
        plan.problems += [
            f"{key} clearance {value * 1000:.1f} mm" for key, value in plan.clearance.items() if value < self._budget(key)
        ]

    def _budget(self, key: str) -> float:
        """CLEARANCE[key]; the target's from the scene's numeric ``r2s_target_descent_m`` when it has one (r17)."""
        if key != "target":
            return CLEARANCE[key]
        if not hasattr(self, "_target_budget"):
            try:
                self._target_budget = float(self.model.numeric("r2s_target_descent_m").data[0])
            except KeyError:
                self._target_budget = CLEARANCE["target"]
        return self._target_budget


# ----------------------------------------------------------------------
# The demonstrator
# ----------------------------------------------------------------------


def _severity(plan: JointPlan) -> float:
    """Rank failing plans by how far past the budgets they go; IK error and clearance weigh most.

    Returns:
        A score; lower is better.
    """
    return (
        (1e6 if plan.unreached else 0.0)  # after any plan that may run
        + max(0.0, plan.pos_error - IK_POS_TOL) * 1000.0
        + max(0.0, plan.peak_speed_ratio - SPEED_FRACTION)
        + max(0.0, plan.peak_accel_ratio - ACCEL_FRACTION)
        + sum(max(0.0, (getattr(plan, "target_budget", CLEARANCE[key]) if key == "target" else CLEARANCE[key]) - value) * 1000.0
              for key, value in plan.clearance.items())
    )


@dataclass(frozen=True)
class StuckStart:
    """The move to a stuck start: targets for every control tick, model units in the channels' order."""

    targets: np.ndarray
    """``(ticks, channels)``: the working arm goes there, the others hold their targets."""
    arm: int
    """The arm that moves, and works the episode's first attempt."""


def smoothstep(s: np.ndarray) -> np.ndarray:
    """Return ``3 s^2 - 2 s^3`` of *s* clipped to [0, 1]: from rest to rest.

    Returns:
        The profile.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@dataclass
class DemoStats:
    """Counters over the demonstrator's lifetime."""

    ahead_used: int = 0
    """r16: next-object attempts started from the plan made ahead (it passed the exact re-check on the actual state)."""
    ahead_replanned: int = 0
    """r16: next-object attempts planned afresh (no plan ahead, another target, or the re-check failed)."""
    ahead_why: dict = field(default_factory=dict)
    """r16: why each plan made ahead was not used, counted by reason."""

    episodes: int = 0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    replans: int = 0
    plan_problems: int = 0
    """Plans that ran although no candidate passed every check."""


@dataclass
class _Episode:
    serial: int
    instruction: Instruction | None
    rng: np.random.Generator
    params: DemoParams
    start_time: float
    attempt: int = 0
    outcome: Outcome = "running"
    failure: str | None = None
    replans: int = 0
    reasons: list[str] = field(default_factory=list)
    """Why each attempt ended early, in order."""
    retry_pending: bool = False
    """Back home after a failed attempt, the next one starts once the object rests."""
    reset_requested: bool = False
    targets_done: int = 0
    """Objects of a kind-level instruction already sorted in this episode."""


class BucketSortDemonstrator:
    """Put the instructed object in the instructed bucket, one control tick at a time.

    With several arms (the Duo's), one works an attempt while the others hold their targets: each
    attempt goes to the arm that reaches both the object and the bucket, the nearer one first, and
    its plans keep clear of the arms that hold still.

    With *background_planning*, a plan that :meth:`step` starts (an attempt, a re-plan, the way
    home) runs on a thread of its own, from a copy of the data (``mujoco.mj_copyData``); the live
    data stays the control thread's. Until it is done, :attr:`phase` reads ``planning`` and
    :meth:`step` outputs where the arm was, then the braking the plan starts with, then holds; the
    plan then runs from the tick it is done, past the braking already output. A plan done within
    :data:`PLAN_WAIT_S` runs from the tick that started it, as without a thread. :meth:`reset`
    while it plans discards it. :meth:`stuck_start` has :meth:`begin_stuck_start` for the same.
    Called directly, the other methods plan in the calling thread.
    """

    def __init__(  # noqa: PLR0915 - one field per piece of state, each documented
        self,
        model: object,
        task: BucketSortTask,
        *,
        arms: tuple[FrankaArm, ...] = (FR3_ARM,),
        rng: np.random.Generator | None = None,
        randomize: bool = True,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        reset_on_failure: bool = True,
        background_planning: bool = False,
        plan_process: bool | None = None,
    ) -> None:
        """Bind to the scene model and its task.

        Args:
            model: The bucket_sort model.
            task: The scene's task (``Sim.on_reset``): the instruction and the success test.
            arms: The robot's arms in channel order (:data:`~.bucket_sort_arms.BUCKET_SORT_ARMS`);
                the default is the ``franka_fr3`` profile's FR3.
            rng: Draws each episode's own generator; the same seed repeats the same episodes.
            randomize: Draw each attempt's parameters (:meth:`DemoParams.sample`); otherwise the defaults.
            max_attempts: Attempts per episode, the retry included.
            reset_on_failure: Ask for a scene reset after a failed episode (:meth:`take_reset_request`).
            background_planning: Plan on a thread of its own, so :meth:`step` returns at once.
            plan_process: Plan attempts in a worker process (:class:`_PlanProcess`); ``None``: with background
                planning, unless the environment sets ``PHYSICALAI_PLAN_PROCESS=0``.
        """
        from physicalai_mujoco_plugin.profiles.derive import control_kind  # noqa: PLC0415

        self.model = model
        self.task = task
        self.arms = tuple(arms)
        self.planners = tuple(Planner(model, arm, tuple(o for o in self.arms if o is not arm)) for arm in self.arms)
        self._active = 0
        self.stats = DemoStats()
        self.randomize = randomize
        self.max_attempts = int(max_attempts)
        self.reset_on_failure = reset_on_failure
        self._rng = rng if rng is not None else np.random.default_rng()
        # r16 (plan ahead): the next object's attempt, planned while the current one releases and retreats
        self._ahead: dict | None = None
        self._ahead_planner: Planner | None = None
        self._transited = False   # r16: the move over the current next object is done (or was tried)
        # r16 (plan before record): with the start gate on, an episode's first attempt waits at "ready" until a go
        self._start_gate = False
        self._await_go = False
        self._go = False
        self._go_time: float | None = None
        self._actuators = np.array([
            int(model.actuator(arm.name(name)).id) for arm in self.arms for name in (*arm.actuators, arm.gripper)
        ])
        """Every arm's actuators, in output order: seven joints, then the hand, arm by arm."""
        # ctrl per model-unit target: 1 for the FR3, 6375 for the Panda's finger tendon (255 per 0.04 m).
        self._ctrl_per_unit = np.array([
            control_kind(model, int(a))[1] * float(model.actuator_gear[a, 0]) for a in self._actuators
        ])
        self._grip_per_width = np.array([arm.grip_per_width for arm in self.arms])
        self._rows: list[tuple[int, int]] = []
        """Each arm's first output and joint count: its joints, then its gripper."""
        for arm in self.arms:
            start = self._rows[-1][0] + self._rows[-1][1] + 1 if self._rows else 0
            self._rows.append((start, len(arm.joints)))
        self._grip_rows = np.array([start + count for start, count in self._rows])
        self._followers = [
            (int(row), int(model.actuator(arm.name(name)).id), float(self._ctrl_per_unit[row]))
            for arm, row in zip(self.arms, self._grip_rows, strict=True)
            for name in arm.gripper_followers
        ]
        """Actuators that take a gripper's target too (the reBot's second finger): output row, id, ctrl per unit."""
        self._hold: list[np.ndarray | None] = [None] * len(self.arms)
        self._still_since: float | None = None
        self._pinned_arm: int | None = None
        """The arm a stuck start moved, which works the episode's first attempt (several arms)."""
        """Since when every arm stands still on its targets (several arms)."""
        """The targets each arm that does not work holds, model units (seven joints and the opening in metres)."""
        self._object_body = {name: int(model.body(name).id) for name in scene_objects(model)}
        self._bucket_body = {color: int(model.body(f"bucket_{color}").id) for color in bucket_colors(model)}
        self._q: np.ndarray | None = None
        self._q_prev: np.ndarray | None = None
        self._braking_from = self.ik.home.copy()
        self._grip = self.arms[0].grip_open
        self._generation: int | None = None
        self._episode: _Episode | None = None
        self._plan: JointPlan | None = None
        self._tick = 0
        self._phase: Phase = "idle"
        self._since = 0.0
        self._t_anchor = 0.0   # r16: sim time at the episode's start (the tick clock's zero)
        self._plan_end: float | None = None
        self._lag_since: float | None = None
        self._lift_check_since: float | None = None
        self._rest_z = 0.0
        self._planned_object: np.ndarray | None = None
        self._planned_axis = 0.0
        self._reset_requested = False
        self.plans: list[JointPlan] = []
        """Every plan of the current episode, for diagnostics."""
        self.background_planning = background_planning
        import os  # noqa: PLC0415

        env = os.environ.get("PHYSICALAI_PLAN_PROCESS")
        self._use_plan_process = (background_planning if env is None else env not in ("0", "off", "false")) if plan_process is None else plan_process
        self._plan_proc: _PlanProcess | None = None
        self._job: _PlanJob | None = None
        """The plan on its own thread, until the control thread collects it (:meth:`_finish`)."""
        self._housing: tuple | None = None   # (hand housing bodies, objects and buckets): _check_housing
        self._collateral: dict | None = None   # _check_collateral
        self._pile_dofs: tuple | None = None; self._pile_still_since: float | None = None   # _pile_still
        self._scratch: object | None = None
        """The planning thread's copy of the data."""
        self._deferring = False
        """Inside :meth:`step` with background planning: plans go to a thread."""
        self._last_call: float | None = None
        """When :meth:`step` (or :meth:`poll_stuck_start`) last returned, monotonic seconds."""
        self._stuck_result: StuckStart | None = None

    # -- readouts ---------------------------------------------------------

    @property
    def planner(self) -> Planner:
        """The planner of the arm that works now."""
        return self.planners[self._active]

    @property
    def ik(self) -> FrankaIK:
        """The IK of the arm that works now."""
        return self.planners[self._active].ik

    @property
    def active_arm(self) -> FrankaArm:
        """The arm that works now; the others hold still."""
        return self.arms[self._active]

    @property
    def phase(self) -> Phase:
        """What the demonstrator is doing: ``planning`` while a plan runs on its own thread."""
        job = self._job
        if job is not None:
            return "idle" if job.cancelled else "planning"
        return self._phase

    @property
    def planning(self) -> bool:
        """Whether a plan is in flight: its thread runs, or ran and the plan waits for the next tick.

        A plan :meth:`reset` cancelled is not in flight.
        """
        job = self._job
        return job is not None and not job.cancelled

    @property
    def planner_running(self) -> bool:
        """Whether a planning thread still runs, and reads the model: a cancelled one until it stops."""
        job = self._job
        return job is not None and not job.done.is_set()

    def reap(self) -> None:
        """Collect a cancelled plan whose thread has stopped, and apply the reset that waited for it.

        :meth:`step` collects plans too, but a cancelled one may never see another step (the
        autopilot switched off): the control loop calls this every tick.
        """
        job = self._job
        if job is not None and job.cancelled and job.done.is_set():
            self._finish(job)

    @property
    def target(self) -> str | None:
        """The object of the current instruction, if any."""
        episode = self._current_episode()
        return episode.instruction.object if episode is not None and episode.instruction is not None else None

    def _current_episode(self) -> _Episode | None:
        """Return the episode in progress; none once :meth:`reset` cancelled the plan in flight.

        Returns:
            The episode, or ``None``.
        """
        job = self._job
        return None if job is not None and job.cancelled else self._episode

    @property
    def last_plan(self) -> JointPlan | None:
        """The plan that ran last."""
        return self.plans[-1] if self.plans else None

    def set_start_gate(self, *, enabled: bool | None = None, go: bool = False) -> None:
        """r16 (plan before record): ``enabled`` switches the start gate; ``go`` releases the episode waiting at it (a
        go before the episode is ready counts for it, so a recorder may send it right after it starts recording)."""
        if enabled is not None:
            self._start_gate = bool(enabled)
            if not enabled:
                self._await_go = False
        if go:
            self._go = True

    def status(self) -> dict[str, object]:
        """Return the episode status for ``/health``: instruction, attempt, outcome and why it failed.

        ``outcome`` is ``running``, ``success``, ``failed`` (a recorder discards the episode) or
        ``stalled`` (the arm does not follow the published targets). ``clean`` is false once an
        attempt went wrong (a miss, a slip, a drop), even when the episode still succeeds, for
        example by a dropped object landing in the bucket; a recorder that keeps only smooth
        demonstrations keeps clean successes.

        While a plan runs on its own thread, the status is the one from when it started.
        """
        job = self._job
        if job is not None and not job.cancelled and job.status is not None:
            return dict(job.status)
        episode = self._current_episode()
        return {
            "ready": bool(self._await_go and not self._go),   # r16: planned, waiting at the start gate for a go
            "clean": not episode.reasons if episode is not None else None,
            "reasons": list(episode.reasons) if episode is not None else [],
            "instruction": episode.instruction.text if episode is not None and episode.instruction else None,
            "episode": episode.serial if episode is not None else None,
            "attempt": episode.attempt if episode is not None else 0,
            "max_attempts": self.max_attempts,
            "outcome": episode.outcome if episode is not None else None,
            "failure": episode.failure if episode is not None else None,
        }

    def take_reset_request(self) -> bool:
        """Return whether a failed episode wants a scene reset now, once.

        Returns:
            ``True`` once per failed episode.
        """
        if self._job is not None:
            return False  # the planning thread has the state
        requested, self._reset_requested = self._reset_requested, False
        return requested

    def reset(self) -> None:
        """Forget the episode in progress; the next :meth:`step` starts a new one from the arm's pose.

        A plan in flight is discarded: the reset takes effect once its thread ends, within a few
        milliseconds (:func:`_checkpoint`); until then :meth:`step` holds the actuators' targets.
        A finished plan not collected yet, and a stuck start not polled yet, are discarded too.
        """
        job = self._job
        if job is not None and not job.done.is_set():
            job.cancelled = True
            return
        self._job = None
        self._stuck_result = None
        self._reset_state()

    def close(self, timeout_s: float = 2.0) -> None:
        """Cancel a plan in flight and wait for its thread (the scene is unloaded); stop the planning worker."""
        job = self._job
        if job is not None:
            job.cancelled = True
            job.thread.join(timeout_s)
            self.reap()
        if self._plan_proc is not None:
            self._plan_proc.close(); self._plan_proc = None

    def _plan_process_for_call(self) -> _PlanProcess | None:
        """The planning worker, started on first use; ``None`` when planning in this process."""
        if not self._use_plan_process:
            return None
        if self._plan_proc is None or not self._plan_proc.alive:
            try:
                self._plan_proc = _PlanProcess(self.model, self.arms)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Demonstrator: no planning worker ({}); planning in this process", exc)
                self._use_plan_process = False
                return None
        return self._plan_proc

    def _reset_state(self) -> None:
        self._q = None
        self._q_prev = None
        self._plan = None
        self._episode = None
        self._generation = None
        self._phase = "idle"
        self._reset_requested = False
        self._pinned_arm = None

    # -- control ----------------------------------------------------------

    def step(self, data: object, dt: float, *, apply: bool = True) -> np.ndarray:
        """Advance by one control tick of *dt* seconds.

        With *apply*, the targets are written to the actuators; otherwise they are only returned.

        Returns:
            Eight targets per arm in model units, the channels' order (``ACTUATORS`` for the FR3): the
            seven arm joints (radians) and the hand's actuator (the FR3's opening in metres).
        """
        job = self._job
        targets = self._while_planning(job, data, dt) if job is not None else None
        if targets is None:
            new_episode = self._episode is None or self._generation != self.task.reset_count
            if self._q is None or new_episode:
                self._take_targets(data)
            if new_episode:
                self._start_episode(data)
            self._q_prev = self._q.copy()
            self._deferring = self.background_planning
            try:
                self._advance(data, dt)
            finally:
                self._deferring = False
            self._check_housing(data)
            job = self._job
            # A plan just went to its thread, which has the state now: output where the arm is.
            targets = job.hold[0].copy() if job is not None else self._targets(self._q, self._grip)  # type: ignore[index, arg-type]
        self._last_call = time.monotonic()
        if apply:
            self.write_targets(data, targets)
        return targets

    def _check_housing(self, data: object) -> None:
        """In a scene whose model sets the custom numeric ``r2s_housing_contact_unclean`` (the reBot desk v2), a touch of
        the hand's housing (its hand bodies other than the fingers) on an object or a bucket makes the episode unclean:
        that collision reaches up to ~8 mm past the drawing, so a recording must not show it touching (real2sim r13)."""
        import mujoco  # noqa: PLC0415

        model, episode = self.model, self._episode
        if episode is None:
            return
        if self._housing is None:
            on = False
            try:
                on = bool(model.numeric("r2s_housing_contact_unclean").data[0])
            except KeyError:
                pass
            arm = self.active_arm
            hand = {int(model.body(arm.name(b)).id) for b in arm.hand_bodies} - {int(model.body(arm.name(b)).id) for b in arm.finger_bodies}
            targets = {int(model.body(n).id) for n in (*scene_objects(model), *(f"bucket_{c}" for c in bucket_colors(model)))}
            self._housing = (hand, targets) if on else ((), ())
        self._check_collateral(data, episode)
        hand, targets = self._housing
        if not hand:
            return
        for i in range(int(data.ncon)):
            c = data.contact[i]; b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 in hand and b2 in targets) or (b2 in hand and b1 in targets):
                other = b2 if b1 in hand else b1
                reason = f"the hand's housing touched {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)}"
                if reason not in episode.reasons:
                    episode.reasons.append(reason); logger.warning("Demonstrator: {} (the episode is not clean)", reason)
                return

    COLLATERAL_LIFT_M = 0.005
    """A non-target object this far above the lowest it has rested this episode was carried along (r14, R-45)."""

    def _check_collateral(self, data: object, episode: _Episode) -> None:
        """In a scene whose model sets ``r2s_collateral_unclean``: an object the instruction does not name that rises
        more than :data:`COLLATERAL_LIFT_M` above the lowest it has been this episode (picked up with the target, wedged
        between the fingers) makes the episode unclean (its text label would be wrong for a policy)."""
        model = self.model
        if self._collateral is None:
            try:
                on = bool(model.numeric("r2s_collateral_unclean").data[0])
            except KeyError:
                on = False
            self._collateral = {"on": on, "episode": None, "low": {}}
        state = self._collateral
        if not state["on"] or self.task.instruction is None:
            return
        if state["episode"] is not episode:
            named = set(getattr(self.task.instruction, "all_targets", None) or (self.task.instruction.object,))
            others = [n for n in getattr(self.task.layout, "objects", ()) if n not in named]
            state.update(episode=episode, low={int(model.body(n).id): np.inf for n in others})
        if self._plan is None and all(np.isinf(v) for v in state["low"].values()):
            # r16 (D4): not before the episode's first motion: the reset's pile is still falling and bouncing then (a
            # 7 mm bounce at 0.2 s flagged a CPU the arm never touched); the first plan starts once the pile is still
            return
        xpos = data.xpos
        for body, low in state["low"].items():
            z = float(xpos[body][2])
            if z < low:
                state["low"][body] = z
            elif z > low + self.COLLATERAL_LIFT_M:
                import mujoco  # noqa: PLC0415

                reason = f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)} was carried along"
                if reason not in episode.reasons:
                    episode.reasons.append(reason); logger.warning("Demonstrator: {} (the episode is not clean)", reason)

    # -- planning off the control tick ------------------------------------

    def _while_planning(self, job: _PlanJob, data: object, dt: float) -> np.ndarray | None:
        """Return this tick's targets while *job* plans; ``None`` once it is collected, and the tick runs as usual.

        The tick waits for the plan up to the rest of a control period since the last one returned
        (:meth:`_pace`), so a loop faster than real time slows to it while it plans.

        Returns:
            The held or braking targets, model units.
        """
        job.ticks += 1
        if not job.done.wait(self._pace(dt)):
            if job.cancelled or job.hold is None:
                return self._actuator_targets(data)  # the reset put the arm elsewhere
            return job.hold[min(job.ticks, len(job.hold) - 1)].copy()
        self._finish(job)
        return None

    def _pace(self, dt: float) -> float:
        """Return how long to wait for a plan this tick: what is left of *dt* since the last call returned.

        Returns:
            Seconds, at most *dt*.
        """
        if self._last_call is None:
            return 0.0
        return min(dt, max(0.0, dt - (time.monotonic() - self._last_call)))

    def _finish(self, job: _PlanJob) -> None:
        """Collect a finished *job*: discard it after a reset, else skip the braking ticks already output.

        What the planning thread raised is raised here, as if it had planned in this thread.
        """
        self._job = None
        if job.cancelled:
            self._reset_state()
            self._stuck_result = None
            return
        if job.error is not None:
            raise job.error
        played = min(job.ticks - 1, job.brake_ticks)
        plan = self._plan
        if played > 0 and plan is not None and plan.brake_ticks == job.brake_ticks:
            self._tick = played
            targets = plan.targets(played - 1)
            self._q, self._grip = targets[:-1], float(targets[-1])

    def _actuator_targets(self, data: object) -> np.ndarray:
        """Return every arm's actuator targets in model units, as :meth:`step` returns them.

        Returns:
            Eight values per arm.
        """
        targets = np.concatenate([self._ctrl_targets(data, arm) for arm in range(len(self.arms))])
        targets[self._grip_rows] *= self._grip_per_width
        return targets

    def _plan_then(self, data: object, dt: float, brake: np.ndarray | None, work: Callable[[object], None]) -> None:
        """Run *work* on *data*: plan, then start the plan (:meth:`_run`), after *brake*.

        Within :meth:`step` with background planning, *work* runs on a thread with a copy of
        *data*; meanwhile the arm brakes along *brake* and then holds. The braking's clearance from
        the arms holding still is checked first, here, since it runs before the plan is done.
        """
        if not self._deferring or getattr(_planning, "job", None) is not None:
            work(data)
            return
        braking = brake is not None and len(brake) > 0
        if braking:
            reason = self._brake_blocked(data, brake)  # type: ignore[arg-type]
            if reason is not None:
                self._stop_and_fail(data, dt, None, reason, freeze_at=self._braking_from)
                return
        rows = [self._braking_from, *brake] if braking else [self._q]  # type: ignore[misc]
        hold = np.stack([self._targets(q, self._grip) for q in rows])  # type: ignore[arg-type]
        self._dispatch(data, work, hold, len(rows) - 1)

    def _dispatch(
        self, data: object, work: Callable[[object], object], hold: np.ndarray | None, brake_ticks: int
    ) -> None:
        """Start *work* on its own thread with a copy of *data*; collect it if done within :data:`PLAN_WAIT_S`.

        A cancelled plan still on its way out finishes first; its thread has the scratch data, so this
        one gets its own.
        """
        import mujoco  # noqa: PLC0415

        previous = self._job  # cancelled, on its way out: the new thread waits for it
        if previous is not None:
            previous.cancelled = True
        if self._scratch is None or previous is not None:
            self._scratch = mujoco.MjData(self.model)
        scratch = self._scratch
        mujoco.mj_copyData(scratch, self.model, data)

        def run() -> object:
            if previous is not None:
                self._reset_state()  # what collecting the cancelled plan would have done
            return work(scratch)

        job = _PlanJob(run, hold, brake_ticks, previous)
        job.status = self.status()  # before it is the job in flight: a cancelled one reports the reset
        self._job = job
        job.start()
        if job.done.wait(PLAN_WAIT_S):
            self._finish(job)

    def _take_targets(self, data: object) -> None:
        """Start from the arms as they are: the working arm's joints, every other arm's targets."""
        # A scene reset puts the arm home: start from there, not from the last episode's targets.
        self._q = np.array(data.qpos[self.ik.qpos_adr])
        self._grip = self._ctrl_targets(data, self._active)[-1]
        self._hold = [None if i == self._active else self._ctrl_targets(data, i) for i in range(len(self.arms))]

    def _targets(self, q: np.ndarray, grip: float) -> np.ndarray:
        """Return every channel's target, model units: the working arm at *q* and *grip* (m), the others holding.

        Returns:
            Eight values per arm.
        """
        targets = np.concatenate([
            [*q, grip] if i == self._active else self._hold[i]  # type: ignore[list-item]
            for i in range(len(self.arms))
        ])
        targets[self._grip_rows] *= self._grip_per_width
        return targets

    def write_targets(self, data: object, targets: np.ndarray) -> None:
        """Write targets in model units (:meth:`step`'s) to the actuators."""
        data.ctrl[self._actuators] = targets * self._ctrl_per_unit
        for row, actuator, per_unit in self._followers:
            data.ctrl[actuator] = targets[row] * per_unit

    def stuck_start(self, data: object, dt: float, rng: np.random.Generator) -> StuckStart | None:
        """Plan the move to a pose a policy gets stuck in, for the episode about to start (after a reset).

        The pose comes from the attempt the demonstrator plans for this episode's instruction and
        layout: a tick of its carry, its descent into the bucket or its release, where the hand would
        hold the object. The hand gets there closed or almost (:data:`STUCK_GRIP`) and empty, by a
        smoothstep move in joint space of :data:`STUCK_MOVE_S`. The planned attempt passed every check
        (reach, joint limits and speed, clearance, and on the Duo the other arm's); the move is checked
        for the hand's clearance from the buckets, the other objects, the table and the other arm.
        With several arms, the planned arm moves and works the first attempt; the others hold.

        Returns:
            The move, or ``None`` when no instruction, no plan that passes every check, or no clear
            move was found; the demonstrator then starts as usual.
        """
        instruction = self.task.instruction
        if instruction is None:
            return None
        self._reset_state()
        self._take_targets(data)
        snapshot = SceneSnapshot.take(data)
        heights = self.active_arm.heights
        params = DemoParams.sample(rng, heights) if self.randomize else DemoParams.for_heights(heights)
        if self._operator_timing() >= 18 and self.randomize:  # noqa: PLR2004 - r18: the scene's operator timing here too
            params = operator_timing(params, rng)
        if len(self.arms) > 1:
            plan = self._plan_with_an_arm(snapshot, instruction, params, dt, rng)
        else:
            plan = self.plan_attempt(snapshot, self._q, self._start_state(), instruction, params, dt, rng, report=False)  # type: ignore[arg-type]
        ticks = (
            [] if plan is None or not plan.ok else [k for k, phase in enumerate(plan.phases) if phase in STUCK_PHASES]
        )
        move = self._stuck_move(snapshot, plan, ticks, dt, rng) if plan is not None and ticks else None
        self._reset_state()  # the episode starts from the stuck pose, at the next step
        if move is None:
            return None
        self._pinned_arm = self._active if len(self.arms) > 1 else None
        logger.info(
            "Demonstrator: a stuck start for the {}, {:.1f} s away", self.active_arm.label, len(move.targets) * dt
        )
        return move

    def begin_stuck_start(self, data: object, dt: float, rng: np.random.Generator) -> None:
        """Start :meth:`stuck_start`; with background planning on its own thread. :meth:`poll_stuck_start` returns it.

        The autopilot must not step meanwhile; a :meth:`reset` discards the move.
        """
        self._stuck_result = None
        if not self.background_planning:
            self._stuck_result = self.stuck_start(data, dt, rng)
            return

        def work(copy: object) -> None:
            self._stuck_result = self.stuck_start(copy, dt, rng)

        self._dispatch(data, work, None, 0)

    def poll_stuck_start(self, dt: float) -> tuple[bool, StuckStart | None]:
        """Return whether the stuck start of :meth:`begin_stuck_start` still plans, else its move (once).

        While it plans, the call waits up to the rest of a control period of *dt* (as :meth:`step` does).

        Returns:
            ``(True, None)`` while it plans, then ``(False, move)``; the move is ``None`` when there is none.
        """
        job = self._job
        if job is not None:
            if not job.done.wait(self._pace(dt)):
                self._last_call = time.monotonic()
                return True, None
            self._finish(job)
        self._last_call = time.monotonic()
        move, self._stuck_result = self._stuck_result, None
        return False, move

    def _stuck_move(
        self, snapshot: SceneSnapshot, plan: JointPlan, ticks: list[int], dt: float, rng: np.random.Generator
    ) -> StuckStart | None:
        """Return a clear smoothstep move from the current targets to a stuck pose of *plan*, if one is found.

        Returns:
            The move, or ``None`` after :data:`STUCK_TRIES` poses whose move is not clear.
        """
        planner, instruction = self.planner, self.task.instruction
        start, grip0 = np.asarray(self._q, dtype=np.float64), float(self._grip)
        for _ in range(STUCK_TRIES):
            goal = plan.q[int(rng.choice(ticks))]
            grip = float(rng.uniform(*STUCK_GRIP))
            count = max(2, round(float(rng.uniform(*STUCK_MOVE_S)) / dt))
            s = smoothstep(np.arange(1, count + 1) / count)
            q = start[None, :] + s[:, None] * (goal - start)[None, :]
            opening = grip0 + s * (grip - grip0)
            move = JointPlan(
                kind="home", dt=dt, q=q, grip=opening, phases=["return"] * count, cartesian=np.zeros((count, 5))
            )
            planner.load(snapshot)
            planner._check_clearance(move, instruction.object, None)  # noqa: SLF001  # type: ignore[union-attr]
            if move.problems:
                continue
            targets = np.stack(list(starmap(self._targets, zip(q, opening, strict=True))))
            return StuckStart(targets=targets, arm=self._active)
        return None

    def _ctrl_targets(self, data: object, arm: int) -> np.ndarray:
        """Return arm *arm*'s current targets from ``ctrl``: seven joints and the opening in metres.

        Returns:
            Eight values.
        """
        start, count = self._rows[arm]
        rows = slice(start, start + count + 1)
        targets = np.array(data.ctrl[self._actuators[rows]]) / self._ctrl_per_unit[rows]
        opening = targets[count] / self._grip_per_width[arm]
        targets[count] = float(np.clip(opening, 0.0, self.arms[arm].grip_open))
        return targets

    def _switch_arm(self, arm: int) -> None:
        """Let arm *arm* work from its held targets; the one that worked holds its own."""
        hold = self._hold[arm]
        if arm == self._active or hold is None or self._q is None:  # only the working arm has no hold
            return
        self._hold[self._active] = np.array([*self._q, self._grip])
        self._hold[arm] = None
        self._active = arm
        count = self._rows[arm][1]
        self._q, self._grip = hold[:count].copy(), float(hold[count])
        self._q_prev = self._q.copy()
        self._braking_from = self._q.copy()

    def _start_episode(self, data: object) -> None:
        self._generation = self.task.reset_count
        # r16: the reset's own episode seed (a fixed-seed reset repeats the episode alone or within any batch); the
        # demonstrator's stream only for a task that does not draw one
        self._await_go, self._go, self._go_time = False, False, None   # r16: each episode needs its own go
        self._ahead = None   # r16: a plan made ahead belongs to its episode
        self._transited = False
        task_seed = getattr(self.task, "episode_seed", None)
        seed = int(task_seed) if task_seed is not None else int(self._rng.integers(2**63))
        rng = np.random.default_rng(seed)
        heights = self.active_arm.heights
        params = DemoParams.sample(rng, heights) if self.randomize else DemoParams.for_heights(heights)
        targets = len(getattr(self.task.instruction, "all_targets", None) or ()) if self.task.instruction is not None else 0
        if targets > 1:   # r11 (D4 hangs): several objects to sort, the brisk end of each declared range (MULTI_OBJECT_BRISK)
            params = brisk_for_several(params, rng if self.randomize else None)
        drift = self._operator_drift_scale()
        if drift > 0.0 and self.randomize:   # r15 (D7): a scene that asks for the real operator's drift (the reBot desk v2)
            # its own stream (from the episode's seed): the episode's other draws stay those of a scene without drift
            params = operator_drift(params, np.random.default_rng([seed, 15]), drift)
        if self._operator_timing() >= 18 and self.randomize:   # noqa: PLR2004 - r18 (D7): a scene that asks for the operator's timing
            params = operator_timing(params, np.random.default_rng([seed, 18]))
            if targets > 1:
                params = replace(params, start_pause_s=min(params.start_pause_s, brisk_for_several(params, None).start_pause_s))
        self.stats.episodes += 1
        self._t_anchor = float(data.time)
        self._episode = _Episode(self.stats.episodes, self.task.instruction, rng, params, float(data.time))
        self._plan = None
        self.plans = []
        self._set_phase("settle" if self.task.instruction is not None else "idle", data)

    def _operator_timing(self) -> int:
        """The custom numeric ``r2s_operator_timing`` of the scene (18: :func:`operator_timing`), else 0; the environment's
        ``R2S_OPERATOR_TIMING`` overrides it (an audit switch, like ``R2S_SENSOR_NOISE``)."""
        import os  # noqa: PLC0415

        if os.environ.get("R2S_OPERATOR_TIMING", "") != "":
            return int(os.environ["R2S_OPERATOR_TIMING"])
        try:
            return int(self.model.numeric("r2s_operator_timing").data[0])
        except KeyError:
            return 0

    def _operator_drift_scale(self) -> float:
        """The custom numeric ``r2s_operator_drift`` of the scene (the speed scale of :func:`operator_drift`), else 0."""
        try:
            return float(self.model.numeric("r2s_operator_drift").data[0])
        except KeyError:
            return 0.0

    def _now(self, data: object) -> float:
        """The demonstrator's clock: whole physics steps since the episode started, times the step (r16). Sim time keeps
        running across resets, so ``data.time`` differences carry float rounding that depends on how long the process
        ran before: a seed alone and as the n-th of a batch could cross a threshold one tick apart. With
        ``TICK_CLOCK`` off, the raw sim time (r15)."""
        if not TICK_CLOCK:
            return float(data.time)
        step = float(self.model.opt.timestep)
        return round((float(data.time) - self._t_anchor) / step) * step

    def _set_phase(self, phase: Phase, data: object) -> None:
        self._phase = phase
        self._since = self._now(data)
        if phase == "settle":
            self._still_since = None  # the arms must stand still anew before this attempt
            self._pile_still_since = None  # and a pile must be still anew

    def _advance(self, data: object, dt: float) -> None:
        episode = self._episode
        now = self._now(data)
        if episode is None or episode.instruction is None or self._phase in {"idle", "stalled"}:
            return
        if self._phase == "settle":
            self._settle(data, dt, now)
            return
        settling = episode.outcome == "running" and self._phase in {"retreat", "return", "done"}
        if settling and self.task.evaluate(self.model, data).success:
            episode.outcome = "success"
            self.stats.successes += 1
        plan = self._plan
        if plan is not None and self._await_go:
            # r16: at the start gate. After the go, the episode's start pause (the real operator's), then the plan
            self._phase = "ready"
            if self._go and self._go_time is None:
                self._go_time = now
                logger.info("Demonstrator: go; start pause {:.3f} s", episode.params.start_pause_s)
            if self._go_time is not None and now - self._go_time >= episode.params.start_pause_s:
                self._await_go = False
            return
        if plan is not None:
            if self._tick < len(plan):
                self._follow(data, plan, dt)
            elif self._arrived(data, plan, now):
                self._plan = None
                if plan.kind == "transit":   # r16: over the next object; its attempt (planned meanwhile) comes next
                    self._set_phase("settle", data)
                    return
                if episode.retry_pending and episode.outcome == "running":
                    episode.retry_pending = False
                    self._set_phase("settle", data)
                else:
                    self._set_phase("failed" if episode.outcome == "failed" else "done", data)
            return
        if self._phase == "done" and episode.outcome == "running" and (
            # r16 (D4, D7): judged once the released object rests, with no fixed minimum (r11-r15: at least 0.4 s between
            # objects, a fixed 1.2 s after the return home); SETTLE_CHECK_S only bounds the wait
            now - self._since >= SETTLE_CHECK_S or self._resting(data)
        ):
            self._judge(data, dt)
        elif (
            self._phase == "failed"
            and self.reset_on_failure
            and not episode.reset_requested
            and now - self._since >= FAILURE_HOLD_S
        ):
            episode.reset_requested = self._reset_requested = True

    def _settle(self, data: object, dt: float, now: float) -> None:
        """Start the attempt once the pause is over and the object rests; a retry also needs it graspable."""
        episode = self._episode
        still = self._arms_still(data, now)
        pause = episode.params.start_pause_s if episode.targets_done == 0 else BETWEEN_PAUSE_S  # type: ignore[union-attr]
        if self._start_gate and episode.targets_done == 0 and episode.attempt == 0:   # type: ignore[union-attr]
            pause = 0.0   # r16: the start pause comes after the plan, at the gate
        if now - self._since >= pause and self._resting(data) and still and self._pile_still(data, now):  # type: ignore[union-attr]
            if episode.attempt == 0:  # type: ignore[union-attr]
                instruction = episode.instruction  # type: ignore[union-attr]
                if episode.targets_done > 0 and not self._transited and len(self.arms) == 1:   # type: ignore[union-attr]
                    # r16 (mid-episode stalls): move over the next object while its attempt is planned
                    self._transited = True
                    self._plan_then(data, dt, None, lambda live: self._plan_and_run_transit(live, dt))
                    return
                if len(instruction.targets) > 1:   # a kind-level instruction: the pile has settled, pick the best one now
                    state = self.task.evaluate(self.model, data)
                    left = [n for n in instruction.targets if state.in_bucket[n] != instruction.bucket]
                    pick = self._best_target(data, left) if left else instruction.object
                    if pick != instruction.object:
                        episode.instruction = replace(instruction, object=pick)  # type: ignore[union-attr]
                        if not self._resting(data):
                            return
                self._begin_attempt(data, dt)
            elif self._graspable(data):
                self._begin_attempt(data, dt, new_attempt=True)
            elif len(episode.instruction.targets) > 1 and self._switch_after_failure(data):  # type: ignore[union-attr]
                self._begin_attempt(data, dt, new_attempt=True)
            else:
                self._fail(data, f"{episode.reasons[-1]}, and it cannot be picked up where it lies", dt)  # type: ignore[union-attr]
        elif now - self._since > SETTLE_TIMEOUT_S and not still:
            # No way home either: it would be checked against an arm that is not where it will be.
            self._stop_and_fail(data, dt, None, "the arms do not come to rest", freeze_at=self._q)
        elif now - self._since > SETTLE_TIMEOUT_S:
            self._fail(data, "the object does not come to rest", dt)

    def _arms_still(self, data: object, now: float) -> bool:
        """Return whether every arm has stood still on its targets for :data:`STILL_S` (several arms).

        A plan is checked against the arms that hold still where they stand when it is planned. An arm
        still moving (toward a target a client sent before the autopilot took over, say) would stand
        elsewhere: so every arm's actuator targets must be the ones the demonstrator gives it (the
        working arm's, the holds), and every joint slower than :data:`STILL_SPEED`. A position servo
        with a fixed target that has come to rest stays there, so the snapshot holds for the plan.

        Returns:
            ``True`` with one arm, which has no other arm to keep clear of.
        """
        if len(self.arms) == 1:
            return True
        for index, planner in enumerate(self.planners):
            start, count = self._rows[index]
            target = self._q if index == self._active else self._hold[index][:count]  # type: ignore[index]
            rows = slice(start, start + count)
            targets = np.array(data.ctrl[self._actuators[rows]]) / self._ctrl_per_unit[rows]
            moving = np.abs(np.array(data.qvel[planner.ik.dof_adr])).max() >= STILL_SPEED
            if moving or np.abs(targets - target).max() > TARGET_TOLERANCE:
                self._still_since = None
                return False
        if self._still_since is None:
            self._still_since = now
        return now - self._still_since >= STILL_S

    def _follow(self, data: object, plan: JointPlan, dt: float) -> None:
        """Output the plan's next targets, and check that the attempt still goes as planned."""
        episode = self._episode
        tick = self._tick
        if self._lagging(data, plan, tick):
            return
        targets = plan.targets(tick)
        self._q, self._grip = targets[:-1], float(targets[-1])
        self._phase = plan.phases[tick]
        self._tick += 1
        if plan.kind in ("home", "transit") or episode is None or episode.instruction is None:
            return
        name = episode.instruction.object
        if self._phase in {"approach", "descend"} and self._planned_object is not None:
            body = self._object_body[name]
            moved = np.linalg.norm(data.xpos[body][:2] - self._planned_object[:2])
            axis = self.planner.grasp_axis_yaw(name, data.xmat[body].reshape(3, 3))
            # The hand closes the same way turned by half a turn.
            turned = abs(wrap_near(axis, self._planned_axis, np.pi) - self._planned_axis)
            if moved > REPLAN_SHIFT or turned > REPLAN_TURN:
                self._replan(data, dt, f"the {name} moved", attempt=True)
            return
        if plan.lift_check is not None and tick == plan.lift_check:
            self._lift_check_since = self._now(data)
        if self._lift_check_since is not None:
            # The arm lags its targets (the servos, a client's easing): judge the lift once the
            # hand itself is up, or after LIFT_CHECK_WAIT_S at the latest.
            up = float(self.ik.tcp(data)[2]) >= plan.cartesian[plan.lift_check or 0, 2] - 0.005
            if not up and self._now(data) - self._lift_check_since < LIFT_CHECK_WAIT_S:
                return
            self._lift_check_since = None
            if not self._holding(data, name, lifted=True):
                self._retry_or_fail(data, "the grasp did not lift the object", dt)
            return
        if plan.hold_until is not None and (plan.lift_check or 0) < tick < plan.hold_until:
            if not self._holding(data, name, lifted=False):
                self._retry_or_fail(data, "the object slipped out of the hand", dt)
                return
            if plan.carry_start is not None and tick >= plan.carry_start and self._bucket_moved(data, plan):
                self._replan(data, dt, f"the {episode.instruction.bucket} bucket moved", attempt=False)

    def _lagging(self, data: object, plan: JointPlan, tick: int) -> bool:
        """Stall when the arm stays far from the targets: in ``leader`` mode, nobody sends them back.

        Returns:
            Whether the demonstrator just stalled.
        """
        error = float(np.abs(np.array(data.qpos[self.ik.qpos_adr]) - plan.q[max(tick - 1, 0)]).max())
        now = self._now(data)
        if error <= STALL_ERROR:
            self._lag_since = None
            return False
        if self._lag_since is None:
            self._lag_since = now
        if now - self._lag_since < CATCH_UP_S:
            return False
        self._stall(error)
        return True

    def _stall(self, error: float) -> None:
        episode = self._episode
        if episode is not None:
            episode.outcome, episode.failure = "stalled", "the arm does not follow the targets"
        logger.warning("Demonstrator: the arm does not follow its targets (error {:.2f} rad); stopping", error)
        self._plan, self._plan_end, self._lag_since = None, None, None
        self._phase = "stalled"

    def _bucket_moved(self, data: object, plan: JointPlan) -> bool:
        if plan.bucket_pose is None:
            return False
        color = self._episode.instruction.bucket  # type: ignore[union-attr]
        body = self._bucket_body[color]
        rotation = data.xmat[body].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        position, planned_yaw = plan.bucket_pose
        turned = abs(wrap_near(yaw - planned_yaw, 0.0))
        return bool(np.linalg.norm(data.xpos[body][:2] - position[:2]) > REPLAN_SHIFT or turned > np.radians(5.0))

    def _holding(self, data: object, name: str, *, lifted: bool) -> bool:
        position = data.xpos[self._object_body[name]]
        near = float(np.linalg.norm(position - self.ik.tcp(data))) < HELD_DISTANCE
        return near and (not lifted or float(position[2]) > self._rest_z + LIFTED_DZ)

    def _arrived(self, data: object, plan: JointPlan, now: float) -> bool:
        """Return whether the arm reached the plan's last target; one that stays far away stalls the demonstrator.

        In ``leader`` mode a client sends the targets back; when none does, the arm never gets there.
        """
        if self._plan_end is None:
            self._plan_end = now
        error = float(np.abs(np.array(data.qpos[self.ik.qpos_adr]) - plan.q[-1]).max())
        waited = now - self._plan_end
        arrive_wait = self._episode.params.arrive_wait_s if self._episode is not None else 0.1
        if (error < 0.02 and waited >= arrive_wait) or (waited >= CATCH_UP_S and error <= STALL_ERROR):  # noqa: PLR2004
            self._plan_end = None
            return True
        if waited >= CATCH_UP_S:
            self._stall(error)
        return False

    def _judge(self, data: object, dt: float) -> None:
        """Back home after an attempt: did the object land in the bucket, and nothing else?"""
        episode = self._episode
        instruction = episode.instruction  # type: ignore[union-attr]
        state = self.task.evaluate(self.model, data)
        if state.success:
            episode.outcome = "success"  # type: ignore[union-attr]
            self.stats.successes += 1
            return
        targets = instruction.all_targets
        start = self.task.start_in_bucket
        others = [
            name
            for name, color in state.in_bucket.items()
            if color is not None and color != start.get(name) and name not in targets
        ]
        if others:
            self._fail(data, f"the {others[0]} ended up in a bucket", dt)
            return
        wrong = [name for name in targets if name != instruction.object and state.in_bucket[name] not in (None, instruction.bucket)]
        if wrong:
            self._fail(data, f"the {wrong[0]} ended up in the {state.in_bucket[wrong[0]]} bucket", dt)
            return
        where = state.in_bucket[instruction.object]
        if where == instruction.bucket:
            if not state.resting[instruction.object]:
                return  # in place but still moving (or touched): judge again next tick
            left = [name for name in targets if state.in_bucket[name] != instruction.bucket]
            if left:   # a kind-level instruction: on to the next object of the kind
                self._next_target(data, left)
            return
        reason = f"the object landed in the {where} bucket" if where else "the object landed outside the bucket"
        self._retry_or_fail(data, reason, dt)

    def _next_target(self, data: object, left: list[str]) -> None:
        """Pick the next object of a kind-level instruction and start on it: the highest graspable one (the top of a
        pile first), then the nearest to the arm's base; with none graspable, the highest (its attempt may fail)."""
        episode = self._episode
        pick = self._best_target(data, left)
        logger.info("Demonstrator: {} placed; next {} ({} left)", episode.instruction.object, pick, len(left))  # type: ignore[union-attr]
        episode.instruction = replace(episode.instruction, object=pick)  # type: ignore[union-attr, arg-type]
        episode.attempt = 0  # type: ignore[union-attr]
        episode.replans = 0  # type: ignore[union-attr]
        episode.targets_done += 1  # type: ignore[union-attr]
        self._transited = False
        self._set_phase("settle", data)

    def _switch_after_failure(self, data: object) -> bool:
        """A kind-level retry whose object cannot be picked up where it now lies: go on with another of the kind still
        on the table that can (this attempt counts); ``False`` when none can."""
        episode = self._episode
        instruction = episode.instruction  # type: ignore[union-attr]
        state = self.task.evaluate(self.model, data)
        left = [
            n for n in instruction.targets
            if n != instruction.object and state.in_bucket[n] != instruction.bucket and self._graspable_name(data, n)
            and state.resting[n]
        ]
        if not left:
            return False
        pick = self._best_target(data, left)
        logger.info("Demonstrator: {} cannot be picked up where it lies; {} instead", instruction.object, pick)
        episode.instruction = replace(instruction, object=pick)  # type: ignore[union-attr]
        return True

    def _best_target(self, data: object, left: list[str]) -> str:
        """The object of a kind-level instruction to pick next: the highest graspable one (the top of a pile first),
        then the nearest to the arm's base; with none graspable, the highest."""
        bodies = {name: self._object_body[name] for name in left}
        graspable = [name for name in left if self._graspable_name(data, name)]
        free = [name for name in graspable if not self._covered(data, name)]
        return max(
            free or graspable or left,
            key=lambda name: (round(float(data.xpos[bodies[name]][2]), 3), -float(np.hypot(*data.xpos[bodies[name]][:2]))),
        )

    def _covered(self, data: object, name: str) -> bool:
        """Whether another object lies on top of *name*: its centre over *name*'s footprint and clearly higher. A
        grasp there closes on both, or the one on top pins it down."""
        footprints = self.task.footprints(self.model)
        here = data.xpos[self._object_body[name]]
        own = footprints[name]
        for other, body in self._object_body.items():
            if other == name or other not in footprints:
                continue
            there = data.xpos[body]
            if (
                float(np.hypot(*(there[:2] - here[:2]))) < 0.7 * (own.radius + footprints[other].radius)
                and there[2] > here[2] + own.half_height
                and there[2] < 0.15  # noqa: PLR2004 - on the table, not parked or in the air
            ):
                return True
        return False

    # -- attempts ---------------------------------------------------------

    def _start_state(self) -> np.ndarray:
        q = self._q if self._q is not None else self.ik.home
        tcp, _ = self.ik.fk(q)
        yaw = self.ik.continuous_yaw(q)
        if self.ik.plain:
            return np.array([*tcp, yaw, self._grip])
        return np.array([*tcp, yaw, self._grip, self.ik.tilt()])

    def _set_go_home(self, data: object) -> None:
        """Whether this attempt goes home at its end (:attr:`DemoParams.go_home`): not while another object of a
        kind-level instruction is still to be sorted."""
        episode = self._episode
        instruction = episode.instruction  # type: ignore[union-attr]
        targets = tuple(getattr(instruction, "all_targets", None) or (instruction.object,))
        go = True
        if len(targets) > 1:
            state = self.task.evaluate(self.model, data)
            go = not [n for n in targets if n != instruction.object and state.in_bucket.get(n) != instruction.bucket]
        if episode.params.go_home != go:  # type: ignore[union-attr]
            episode.params = replace(episode.params, go_home=go)  # type: ignore[union-attr]

    PILE_STILL_MPS = 0.005
    PILE_STILL_RADPS = 0.05
    PILE_STILL_MPS2 = 0.5
    PILE_STILL_S = 0.0
    PILE_STILL_MAX_S = 3.0

    def _pile_still(self, data: object, now: float) -> bool:
        """In a pile (r14, T6): every placed object below :data:`PILE_STILL_MPS` and :data:`PILE_STILL_RADPS` before the
        hand sets off (a neighbour still creeping or toppling was the commonest pre-grasp "push", with no hand contact);
        at most :data:`PILE_STILL_MAX_S` after the settle phase began. Scattered tables: True. r16 (D4, D7): a state, no
        fixed duration (r14-r15 held the linear test for 0.3 s); the angular test catches a toppling body at its turn, the
        acceleration test (:data:`PILE_STILL_MPS2`) a body at rest only for the instant: a reset's spawned pile (zero
        velocity, falling at g) or a bounce's apex."""
        layout = getattr(self.task, "layout", None)
        if layout is None or not getattr(layout, "pile", False):
            return True
        if now - self._since >= self.PILE_STILL_MAX_S:
            return True
        model = self.model
        if self._pile_dofs is None or self._pile_dofs[0] is not layout:
            dofs = [int(model.jnt_dofadr[model.joint(f"{n}:joint").id]) for n in layout.objects]
            self._pile_dofs = (layout, np.array(dofs, dtype=int))
        dofs = self._pile_dofs[1]
        qvel = np.asarray(data.qvel)
        speed = np.linalg.norm(qvel[dofs[:, None] + np.arange(3)], axis=1) if len(dofs) else np.zeros(0)
        spin = np.linalg.norm(qvel[dofs[:, None] + 3 + np.arange(3)], axis=1) if len(dofs) else np.zeros(0)
        accel = np.linalg.norm(np.asarray(data.qacc)[dofs[:, None] + np.arange(3)], axis=1) if len(dofs) else np.zeros(0)
        if (speed < self.PILE_STILL_MPS).all() and (spin < self.PILE_STILL_RADPS).all() and (accel < self.PILE_STILL_MPS2).all():
            if self._pile_still_since is None: self._pile_still_since = now
            return now - self._pile_still_since >= self.PILE_STILL_S
        self._pile_still_since = None
        return False

    def _resting(self, data: object) -> bool:
        name = self._episode.instruction.object  # type: ignore[union-attr]
        return bool(self.task.evaluate(self.model, data).resting[name])

    def _graspable(self, data: object) -> bool:
        """Return whether the instructed object lies where an attempt can pick it up.

        Returns:
            ``True`` on the table, in reach, out of the buckets and with its long side level.
        """
        return self._graspable_name(data, self._episode.instruction.object)  # type: ignore[union-attr]

    def _graspable_name(self, data: object, name: str) -> bool:
        """:meth:`_graspable` for object *name*."""
        body = self._object_body[name]
        position = data.xpos[body]
        rotation = data.xmat[body].reshape(3, 3)
        area = self.task.area
        retry_x, retry_y = RETRY_X, RETRY_Y
        if area is not DEFAULT_LAYOUT:  # the same margins about the robot's own object area
            retry_x = (area.object_x[0] - 0.06, area.object_x[1] + 0.04)
            retry_y = (area.object_y[0] - 0.02, area.object_y[1] + 0.02)
        return bool(
            retry_x[0] <= position[0] <= retry_x[1]
            and retry_y[0] <= position[1] <= retry_y[1]
            and position[2] < 0.05  # noqa: PLR2004
            and self.planner.shapes[name].grasp_axes(rotation) is not None
            and bucket_containing(self.model, data, position) is None
        )

    def _retry_or_fail(self, data: object, reason: str, dt: float) -> None:
        """After a failed attempt: let go, go home, and retry once the object rests; or fail the episode."""
        episode = self._episode
        if episode is None:
            return
        episode.reasons.append(reason)
        if episode.attempt >= self.max_attempts:
            self._fail(data, reason, dt)
            return
        logger.info("Demonstrator: {}; retrying", reason)
        self.stats.retries += 1
        if self.randomize:
            episode.params = _timed_like(replace(DemoParams.sample(episode.rng, self.active_arm.heights), **_drift_of(episode.params)),
                                         episode.params, episode.rng)
        episode.retry_pending = True
        self._go_home(data, dt)

    def _fail(self, data: object, reason: str, dt: float) -> None:
        """Mark the episode failed, let go of anything held and go home; the reset follows."""
        episode = self._episode
        if episode is None:
            return
        if episode.outcome != "failed":
            self.stats.failures += 1
            note = getattr(self.task, "note_failure", None)
            if note is not None:
                note()   # a stratified retry keeps this episode's stratum for the next layout
        episode.outcome, episode.failure = "failed", reason
        logger.info("Demonstrator: episode {} failed: {}", episode.serial, reason)
        self._go_home(data, dt)

    def _go_home(self, data: object, dt: float) -> None:
        """Brake, open the hand, rise and return home.

        The arm may be moving when an attempt is abandoned: the targets first slow to a stop at a
        constant deceleration over :data:`BRAKE_S`, so no joint jumps from speed to rest.
        """
        brake = self._brake(dt)
        self._plan_then(data, dt, brake, lambda live: self._go_home_after(live, dt, brake))

    def _go_home_after(self, data: object, dt: float, brake: np.ndarray) -> None:
        episode = self._episode
        self.planner.load(SceneSnapshot.take(data))
        script = MotionScript(self._start_state())
        script.add("release", grip=self.ik.grip_open, duration=0.4)
        self.planner.add_return(script, episode.params)  # type: ignore[union-attr]
        plan = self.planner.solve(script, self._q, dt, kind="home", target=episode.instruction.object)  # type: ignore[union-attr]
        self._run(plan, data, brake)

    def _brake(self, dt: float) -> np.ndarray:
        """Return targets that slow the arm from its current target speed to a stop.

        The next plan is planned from where they stop (``self._q`` is the last of them until
        :meth:`_run` puts the braking in front of that plan).

        Returns:
            ``(ticks, 7)`` joint targets.
        """
        q = np.asarray(self._q, dtype=np.float64)
        velocity = (q - self._q_prev) / dt if self._q_prev is not None else np.zeros(len(q))
        t = np.arange(1, round(BRAKE_S / dt) + 1) * dt
        brake = q[None, :] + velocity[None, :] * (t - t**2 / (2.0 * BRAKE_S))[:, None]
        brake = np.clip(brake, self.ik.lower, self.ik.upper)
        self._braking_from = q.copy()
        self._q = brake[-1].copy()
        return brake

    def _replan(self, data: object, dt: float, reason: str, *, attempt: bool) -> None:
        """Plan again from here: the whole attempt (before the grasp) or the place (with the object held)."""
        episode = self._episode
        if episode is None:
            return
        if episode.replans >= MAX_REPLANS:
            self._fail(data, f"{reason} too often", dt)
            return
        episode.replans += 1
        self.stats.replans += 1
        logger.info("Demonstrator: {}; planning again", reason)
        brake = self._brake(dt)
        if attempt:
            self._begin_attempt(data, dt, new_attempt=False, brake=brake)
        else:
            self._begin_place(data, dt, brake=brake)

    def _begin_attempt(
        self, data: object, dt: float, *, new_attempt: bool = True, brake: np.ndarray | None = None
    ) -> None:
        """Plan an attempt from the current targets and start it."""
        episode = self._episode
        if episode is None or episode.instruction is None:
            return
        self._set_go_home(data)
        if new_attempt or episode.attempt == 0:
            episode.attempt += 1
            self.stats.attempts += 1
        self._plan_then(data, dt, brake, lambda live: self._plan_and_run_attempt(live, dt, brake))

    def _plan_and_run_attempt(self, data: object, dt: float, brake: np.ndarray | None) -> None:
        episode = self._episode
        snapshot = SceneSnapshot.take(data)
        if brake is None and len(self.arms) > 1:
            plan = self._plan_with_an_arm(snapshot, episode.instruction, episode.params, dt, episode.rng)
            if plan is None:
                instruction = episode.instruction
                self._fail(data, f"no arm reaches the {instruction.object} and the {instruction.bucket} bucket", dt)
                return
        else:
            plan = self._take_ahead(snapshot) if self._ahead is not None else None   # r16: planned ahead, re-checked
            if plan is None:
                plan = self.plan_attempt(
                    snapshot, self._q, self._start_state(), episode.instruction, episode.params, dt, episode.rng
                )
            if self._gated and (plan.unreached or plan.pos_error > OUT_OF_REACH) and len(episode.instruction.targets) > 1:
                plan = self._another_target_plan(data, snapshot, plan, dt)
            if _cuts_objects(plan) and len(episode.instruction.targets) > 1:
                plan = self._another_target_plan(data, snapshot, plan, dt, cut=True)
            if _cuts_objects(plan):   # r11: a glitch is worse than a failed attempt (the recorder discards those)
                cut = -min(plan.clearance.get("objects", 0.0), plan.clearance.get("held_objects", 0.0)) * 1000
                self._fail(data, f"every plan cuts into another object (by {cut:.0f} mm)", dt)
                return
            if _hits_bucket(plan) and len(episode.instruction.targets) > 1:   # r16: another target first, this one later
                plan = self._another_target_plan(data, snapshot, plan, dt, bucket=True)
            if _hits_bucket(plan):   # r16: the last target (or none other workable): refused, not run into the bucket
                depth = -plan.clearance.get("bucket", 0.0) * 1000
                self._fail(data, f"every plan brings the hand into a bucket (by {depth:.0f} mm)", dt)
                return
        self.planner.load(snapshot)
        position, rotation = self.planner.object_pose(episode.instruction.object)
        self._rest_z = float(position[2])
        self._planned_object = position.copy()
        self._planned_axis = self.planner.grasp_axis_yaw(episode.instruction.object, rotation)
        self._run(plan, data, brake)

    def _plan_and_run_transit(self, data: object, dt: float) -> None:
        """r16 (mid-episode stalls; coordinator's conditions): a collision-checked move over the next object, at the carry
        height, as an operator moves over it; its attempt is planned meanwhile from where the move ends (:meth:`_plan_ahead`),
        and used only after the same exact clearance sweep on the actual scene (:meth:`_take_ahead`). A move that fails a
        check is not run: the attempt is planned from here as before."""
        episode = self._episode
        instruction = episode.instruction  # type: ignore[union-attr]
        snapshot = SceneSnapshot.take(data)
        start = self._start_state(); p = episode.params  # type: ignore[union-attr]
        process = self._plan_process_for_call()   # r17: in the planning worker when there is one (not on a GIL thread)
        plan = None
        if process is not None:
            plan = process.plan_transit(self._active, snapshot, self._q, start, instruction.object, instruction.bucket, p, dt, self.model)
        if plan is None:
            plan = _plan_transit_core(self.planner, snapshot, start, self._q, instruction.object, instruction.bucket, p, dt)
        if not plan.ok:
            logger.info("Demonstrator: no move over the {} ({}); planning from here", instruction.object, plan.problems[:2])
            return
        self._run(plan, data)
        self._plan_ahead(data, plan, target=instruction.object)

    def _plan_ahead(self, data: object, plan: JointPlan, *, target: str | None = None) -> None:
        """r16 (plan ahead): once the current object is released, plan the attempt on the next object of a kind-level
        instruction, on a thread, from a predicted scene: every object where it is now, the released one out of the way
        (it is falling into the bucket), the arm where this plan ends. :meth:`_take_ahead` uses it only for the same
        target and start and only after the exact clearance sweep on the actual scene passes."""
        import threading  # noqa: PLC0415

        episode = self._episode
        self._ahead = None
        if episode is None or episode.instruction is None or len(self.arms) > 1:
            return
        instruction = episode.instruction
        targets = tuple(getattr(instruction, "all_targets", None) or (instruction.object,))
        if len(targets) < 2:
            return
        state = self.task.evaluate(self.model, data)
        snapshot = SceneSnapshot.take(data)
        if target is not None:   # the transit's: the attempt on this target from where the move ends
            pick = target
        else:
            left = [n for n in targets if n != instruction.object and state.in_bucket.get(n) != instruction.bucket]
            if not left:
                return
            pick = self._best_target(data, left)
            body = self._object_body[instruction.object]
            adr = int(self.model.jnt_qposadr[self.model.body_jntadr[body]])
            snapshot.qpos[adr : adr + 3] = (0.0, 0.0, -5.0)   # the released object: out of the way
        last = plan.targets(len(plan) - 1)
        q_end, grip_end = np.asarray(last[:-1], dtype=np.float64).copy(), float(last[-1])
        saved = (self._q, self._grip)
        self._q, self._grip = q_end, grip_end
        try:
            start = self._start_state()
        finally:
            self._q, self._grip = saved
        go_home = not [n for n in targets if n != pick and (target is not None or n != instruction.object) and state.in_bucket.get(n) != instruction.bucket]
        params = replace(episode.params, go_home=go_home)
        rng = np.random.Generator(np.random.PCG64()); rng.bit_generator.state = episode.rng.bit_generator.state
        nxt = replace(instruction, object=pick)
        a_pick = int(self.model.jnt_qposadr[self.model.body_jntadr[self._object_body[pick]]])
        job: dict = {"target": pick, "q": q_end, "params": params, "rng": rng, "plan": None, "serial": episode.serial,
                     "pick_adr": a_pick, "pick_pose": snapshot.qpos[a_pick : a_pick + 7].copy()}
        process = self._plan_process_for_call()
        if process is None and self._ahead_planner is None:
            arm = self.active_arm
            self._ahead_planner = Planner(self.model, arm, tuple(o for o in self.arms if o is not arm))

        def work() -> None:
            result = None
            if process is not None:
                result = process.plan_attempt(self._active, snapshot, q_end, start, nxt, params, plan.dt, rng, self.model)
            if result is None:
                result = _plan_attempt_core(self._ahead_planner, self.active_arm, self._gated, snapshot, q_end, start, nxt, params, plan.dt, rng)
            job["plan"] = result

        job["thread"] = threading.Thread(target=work, name="plan-ahead", daemon=True)
        job["thread"].start()
        self._ahead = job

    def _take_ahead(self, snapshot: SceneSnapshot) -> JointPlan | None:
        """r16 (plan ahead): the plan made ahead for this attempt, if it is for the same object, start and parameters,
        it passed every check, and the same clearance sweep as a fresh plan's, on the actual scene (*snapshot*), passes
        too; else ``None`` (plan afresh). Runs in the planning job (it waits for the plan made ahead if still running)."""
        job, self._ahead = self._ahead, None
        episode = self._episode
        if job is None or episode is None:
            return None
        why = ("another episode" if job["serial"] != episode.serial else "another target" if job["target"] != episode.instruction.object
               else "not a first attempt" if episode.attempt != 1 else "another start" if self._q is None or not np.allclose(job["q"], self._q, atol=1e-9)
               else "other parameters" if job["params"] != episode.params else None)
        if why is not None:
            self.stats.ahead_replanned += 1; self.stats.ahead_why[why] = self.stats.ahead_why.get(why, 0) + 1
            return None
        a, was = job["pick_adr"], job["pick_pose"]; now = snapshot.qpos[a : a + 7]
        turned = 2.0 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(was[3:], now[3:]))))))
        moved = np.linalg.norm(now[:3] - was[:3]) > AHEAD_MOVED_M or turned > AHEAD_TURNED_DEG
        if moved:   # the target moved since (the drop nudged it): the grasp was aimed at where it was (other objects:
            # the clearance re-check below)
            self.stats.ahead_replanned += 1; self.stats.ahead_why["the target moved"] = self.stats.ahead_why.get("the target moved", 0) + 1
            return None
        job["thread"].join()
        plan = job["plan"]
        # r17 (live stalls): a best-effort plan made ahead is used too when the scene it was made for is the actual one:
        # the target where it was (float tolerance, not the 2 mm a passing plan may move: the grasp's IK aims at it) and
        # the exact clearance re-check below reproducing every planning-time clearance and problem; a fresh plan would
        # search the same space from the same state and random draws
        best_effort = plan is not None and not plan.ok and getattr(plan, "held_ticks", "missing") != "missing" and AHEAD_BEST_EFFORT
        if best_effort:   # r17 (coordinator: a quality check): the target where the plan aimed (its IK holds), within tolerance
            moved_m = float(np.linalg.norm(now[:3] - was[:3]))
            if not (plan.problems and moved_m <= AHEAD_QUALITY_M and turned <= AHEAD_QUALITY_DEG):
                best_effort = False
        if plan is None or (not plan.ok and not best_effort) or getattr(plan, "held_ticks", "missing") == "missing":
            why = "no plan" if plan is None else f"it failed a check: {plan.problems[0][:60]}" if not plan.ok else "no held ticks"
            self.stats.ahead_replanned += 1; self.stats.ahead_why[why] = self.stats.ahead_why.get(why, 0) + 1
            return None
        planner = self.planner
        planner.load(snapshot)
        held_width = planner.held_width
        planner.held_width = plan.held_width
        before = list(plan.problems); clearance_before = dict(plan.clearance)
        if best_effort:   # the re-check recomputes the clearance problems from scratch
            plan.problems = [q for q in plan.problems if "clearance" not in q]
        try:
            planner._check_clearance(plan, episode.instruction.object, plan.held_ticks)
        finally:
            planner.held_width = held_width
        if best_effort and (set(plan.clearance) != set(clearance_before)
                            or any(plan.clearance[k] < clearance_before[k] - AHEAD_QUALITY_M for k in clearance_before)):
            plan.problems, plan.clearance = before, clearance_before
            self.stats.ahead_replanned += 1; self.stats.ahead_why["a best effort not reproduced"] = self.stats.ahead_why.get("a best effort not reproduced", 0) + 1
            return None
        if plan.problems != before:   # the actual scene fails a clearance the predicted one passed
            self.stats.ahead_replanned += 1; self.stats.ahead_why["the re-check failed"] = self.stats.ahead_why.get("the re-check failed", 0) + 1
            logger.info("Demonstrator: the plan made ahead fails on the actual scene ({}); planning afresh", plan.problems[len(before):])
            return None
        episode.rng.bit_generator.state = job["rng"].bit_generator.state   # as if planned now
        self.stats.ahead_used += 1
        if best_effort:
            self.stats.ahead_best_effort_used = getattr(self.stats, "ahead_best_effort_used", 0) + 1
            self._report_problems(plan)
        logger.info("Demonstrator: plan ahead used for {} (used {}, planned afresh {}: {})", episode.instruction.object,
                    self.stats.ahead_used, self.stats.ahead_replanned, self.stats.ahead_why)
        return plan

    def _another_target_plan(
        self, data: object, snapshot: SceneSnapshot, plan: JointPlan, dt: float, *, cut: bool = False, bucket: bool = False
    ) -> JointPlan:
        """A kind-level instruction whose chosen object is out of reach: plan for its other objects still on the table
        (graspable ones, highest first) and switch to the first one the hand reaches; else keep *plan* (it fails)."""
        episode = self._episode
        instruction = episode.instruction  # type: ignore[union-attr]
        state = self.task.evaluate(self.model, data)
        left = [
            n for n in instruction.targets
            if n != instruction.object and state.in_bucket[n] != instruction.bucket and self._graspable_name(data, n)
        ]
        left.sort(key=lambda n: -float(data.xpos[self._object_body[n]][2]))
        for name in left:
            other = replace(instruction, object=name)
            candidate = self.plan_attempt(snapshot, self._q, self._start_state(), other, episode.params, dt, episode.rng)  # type: ignore[union-attr]
            if (not candidate.unreached and candidate.pos_error <= OUT_OF_REACH and not (cut and _cuts_objects(candidate))
                    and not (bucket and (_hits_bucket(candidate) or _cuts_objects(candidate)))):
                why = "boxed in" if cut else ("against a bucket" if bucket else "out of reach")
                logger.info("Demonstrator: {} {}; {} first", instruction.object, why, name)
                episode.instruction = other  # type: ignore[union-attr]
                return candidate
        return plan

    def _plan_with_an_arm(
        self, snapshot: SceneSnapshot, instruction: Instruction, params: DemoParams, dt: float, rng: np.random.Generator
    ) -> JointPlan | None:
        """Plan an attempt with the arm that suits it, and let that arm work (several arms, at rest).

        Every arm plans, in the order of :meth:`_arm_preference`: the first whose plan passes every
        check works, else the best of the plans that may run (:meth:`_may_run`: within
        :data:`OUT_OF_REACH` of their path and clear of the other arm), however well a plan that may
        not run scores. The ranking only orders the arms:
        its probe does not try the other grasp yaw or the turn for the release, which a plan may need.

        Returns:
            The plan, for the arm that now works; ``None`` when no arm can do the attempt.
        """
        preference = {i: self._arm_preference(i, snapshot, instruction, params) for i in range(len(self.arms))}
        order = sorted(preference, key=lambda i: preference[i])
        episode = self._episode
        if self._pinned_arm is not None:
            if episode is None or episode.attempt <= 1:
                order = [self._pinned_arm]  # a stuck start's arm recovers on its own: it works the first attempt
            self._pinned_arm = None
        best: tuple[int, JointPlan] | None = None
        """The best plan that may run (:meth:`_may_run`), and its arm."""
        closest: JointPlan | None = None
        """The best of the plans that may not, for the log."""
        for arm in order:
            self._switch_arm(arm)
            plan = self.plan_attempt(snapshot, self._q, self._start_state(), instruction, params, dt, rng, report=False)  # type: ignore[arg-type]
            if plan.ok:
                logger.info("Demonstrator: the {} works this attempt", self.active_arm.label)
                return plan
            if self._may_run(plan) and (best is None or _severity(plan) < _severity(best[1])):
                best = (arm, plan)
            elif best is None and (closest is None or _severity(plan) < _severity(closest)):
                closest = plan
        if best is None:
            if closest is not None:
                logger.info("Demonstrator: no arm can do this attempt ({})", "; ".join(closest.problems))
            return None
        self._switch_arm(best[0])
        logger.info("Demonstrator: the {} works this attempt", self.active_arm.label)
        self._report_problems(best[1])
        return best[1]

    def _arm_preference(
        self, arm: int, snapshot: SceneSnapshot, instruction: Instruction, params: DemoParams
    ) -> tuple[bool, float]:
        """Rank arm *arm* for an attempt: whether it reaches the grasp and the bucket from home, and how far it goes.

        The probe keeps the object's grasp yaw over the bucket; an arm it misses may still plan the
        attempt with the other yaw or a turn for the release, so it ranks such an arm last.

        Returns:
            A sort key; lower is better.
        """
        planner = self.planners[arm]
        ik = planner.ik
        planner.load(snapshot)
        position, rotation = planner.object_pose(instruction.object)
        grasp = planner.grasp_point(instruction.object, position, rotation, params)
        bucket, _ = planner.bucket_pose(instruction.bucket)
        over = np.array([bucket[0], bucket[1], params.release_z])
        q_grasp, e_grasp, _ = ik.optimize(ik.home, grasp, planner.grasp_axis_yaw(instruction.object, rotation), 0)
        _, e_over, _ = ik.optimize(q_grasp, over, ik.continuous_yaw(q_grasp), 0)
        home, _ = ik.fk(ik.home)
        reaches = e_grasp <= IK_POS_TOL and e_over <= IK_POS_TOL
        distance = float(np.linalg.norm((grasp - home)[:2]) + np.linalg.norm((bucket - home)[:2]))
        return (not reaches, distance)

    def _begin_place(self, data: object, dt: float, brake: np.ndarray | None = None) -> None:
        """Plan the carry and the release again, with the object in the hand, and start it."""
        self._plan_then(data, dt, brake, lambda live: self._plan_and_run_place(live, dt, brake))

    def _plan_and_run_place(self, data: object, dt: float, brake: np.ndarray | None) -> None:
        episode = self._episode
        instruction = episode.instruction  # type: ignore[union-attr]
        self.planner.load(SceneSnapshot.take(data))
        # Plan from where braking stops, with the object where the hand will hold it then.
        self.planner.carry_held(instruction.object, data, self._q)
        start = self._start_state()
        position, _ = self.planner.object_pose(instruction.object)
        offset = (position - start[:3])[:2]
        plan = self.planner.checked(
            lambda slow: self.planner.place_script(start, instruction, episode.params, offset, slow),  # type: ignore[union-attr]
            self._q,
            dt,
            kind="place",
            target=instruction.object,
            bucket=instruction.bucket,
            held_from=0.0,
        )
        self._run(plan, data, brake)

    def plan_attempt(
        self,
        snapshot: SceneSnapshot,
        q: np.ndarray,
        start: np.ndarray,
        instruction: Instruction,
        params: DemoParams,
        dt: float,
        rng: np.random.Generator | None = None,
        *,
        report: bool = True,
    ) -> JointPlan:
        """Plan one attempt of the working arm: the better grasp yaw first, then the other, then safer variants.

        The two yaws are ranked by the joint-limit margin of their key postures (grasp, then
        release, solved from each other). When the margins are within 10 deg, ``params.yaw_choice``
        ranks them instead, so both yaws appear in the demonstrations.

        With *rng*, two fresh parameter draws come last: a path near the shoulder singularity
        (objects close in front of the base) often clears with other heights and speeds.

        Without *report*, a plan that fails a check is neither counted nor logged.

        Returns:
            The first plan that passes every check, else the one with the fewest problems.
        """
        plan = None
        process = self._plan_process_for_call()
        if process is not None:
            plan = process.plan_attempt(self._active, snapshot, q, start, instruction, params, dt, rng, self.model)
        if plan is None:   # in this process (no worker, or the worker is unavailable)
            plan = _plan_attempt_core(self.planner, self.active_arm, self._gated, snapshot, q, start, instruction, params, dt, rng)
        if report and not plan.ok:
            self._report_problems(plan)
        return plan


    @property
    def _gated(self) -> bool:
        """Whether :meth:`_run` refuses best-effort plans out of reach: with several arms, or a leaning one.

        :meth:`plan_attempt` then prefers a plan that may run (:meth:`_may_run`) to any it would refuse.
        One Franka runs its best-effort plans as before.
        """
        return len(self.arms) > 1 or self.active_arm.max_tilt > 0.0

    @staticmethod
    def _may_run(plan: JointPlan) -> bool:
        """Return whether a best-effort plan may run where :attr:`_gated`: near its path and clear of the other arm.

        Near is within :data:`OUT_OF_REACH`; clear is :data:`CLEARANCE` ``arm``.

        Returns:
            ``True`` for a plan that misses some check but neither of these.
        """
        return (
            not plan.unreached
            and plan.pos_error <= OUT_OF_REACH
            and plan.clearance.get("arm", np.inf) >= CLEARANCE["arm"]
        )

    def _too_close_to_an_arm(self, plan: JointPlan) -> bool:
        """Return whether *plan* brings the working arm closer than :data:`CLEARANCE` ``arm`` to an arm holding still.

        Every plan runs through :meth:`_run`, the re-planned approach and place and the way home
        included, so none moves the working arm into the other.
        """
        return len(self.arms) > 1 and plan.clearance.get("arm", np.inf) < CLEARANCE["arm"]

    def _stop_and_fail(
        self,
        data: object,
        dt: float,
        brake: np.ndarray | None,
        reason: str,
        *,
        freeze_at: np.ndarray | None = None,
    ) -> None:
        """Stop and hold; the episode fails and the scene resets after :data:`FAILURE_HOLD_S`.

        The arm stops where *brake* ends (checked clear of the other arm), else at once on *freeze_at*
        (the target before a braking that was not clear), else where its targets are. No other plan
        runs from here: one that goes home could come as close to the other arm.
        """
        episode = self._episode
        if brake is not None and len(brake):
            q = np.vstack([brake, brake[-1:]])
            self._q = self._braking_from  # this tick still outputs where the arm is now
        else:
            self._q = np.array(freeze_at if freeze_at is not None else self._q, dtype=np.float64)
            q = self._q[None, :].copy()
        ticks = len(q)
        state = self._start_state()
        stop = JointPlan(
            kind="home",
            dt=dt,
            q=q,
            grip=np.full(ticks, self._grip),
            phases=["failed"] * ticks,
            cartesian=np.repeat(state[None, :], ticks, axis=0),
            brake_ticks=len(brake) if brake is not None else 0,
        )
        if episode is not None:
            if episode.outcome != "failed":  # a failed episode on its way home keeps its reason
                self.stats.failures += 1
                episode.outcome, episode.failure = "failed", reason
            episode.reasons.append(reason)
        logger.warning("Demonstrator: {}; stopping, the scene resets next", reason)
        self._plan = stop
        self.plans.append(stop)
        self._tick = 0
        self._plan_end = None
        self._lag_since = None
        self._lift_check_since = None
        self._set_phase("failed", data)

    def _brake_blocked(self, data: object, brake: np.ndarray | None) -> str | None:
        """Return why *brake* must not run, with several arms: it would reach into an arm holding still.

        The arm then stops on the last target instead, which a checked plan (or the hold) put there.

        Returns:
            The reason, or ``None`` when the braking is clear (or there is none, or one arm).
        """
        if len(self.arms) == 1 or brake is None or not len(brake):
            return None
        clearance = self.planner.arm_clearance(brake, SceneSnapshot.take(data))
        if clearance >= CLEARANCE["arm"]:
            return None
        needed = CLEARANCE["arm"] * 1000
        return f"braking would come {clearance * 1000:.0f} mm from the other arm (at least {needed:.0f} mm)"

    def _report_problems(self, plan: JointPlan) -> None:
        self.stats.plan_problems += 1
        logger.warning("Demonstrator: no plan passed every check ({}); running the best one", "; ".join(plan.problems))

    def _run(self, plan: JointPlan, data: object, brake: np.ndarray | None = None) -> None:
        """Start *plan*, after *brake*; with several arms, every plan passes one gate here first.

        The braking and the plan must keep clear of the arms holding still, and an attempt or place
        plan must stay within :data:`OUT_OF_REACH` of its path; otherwise the arm stops
        (:meth:`_stop_and_fail`). One arm runs its plans as before: the FR3's best-effort plans included.
        """
        if len(self.arms) > 1:
            needed = CLEARANCE["arm"] * 1000
            reason = self._brake_blocked(data, brake)
            if reason is not None:
                self._stop_and_fail(data, plan.dt, None, reason, freeze_at=self._braking_from)
                return
            if self._too_close_to_an_arm(plan):
                distance = plan.clearance["arm"] * 1000
                reason = (
                    f"the {plan.kind} plan would come {distance:.0f} mm from the other arm (at least {needed:.0f} mm)"
                )
                self._stop_and_fail(data, plan.dt, brake, reason)
                return
        if self._gated:
            # A plan out of reach would close on nothing or let go outside the bucket: several arms
            # and the leaning arms refuse it (the checked braking still stops the arm); one Franka
            # runs its best-effort plans as before.
            if plan.unreached:
                self._stop_and_fail(data, plan.dt, brake, f"no lean of the hand reaches the {plan.kind} plan's poses")
                return
            if plan.kind != "home" and plan.pos_error > OUT_OF_REACH:
                reason = (
                    f"the {plan.kind} plan misses its path by {plan.pos_error * 1000:.0f} mm "
                    f"(at most {OUT_OF_REACH * 1000:.0f} mm): out of reach"
                )
                self._stop_and_fail(data, plan.dt, brake, reason)
                return
        if brake is not None and len(brake):
            # Prefix the braking ticks; the plan's own ticks shift by as many.
            n = len(brake)
            plan.brake_ticks = n
            plan.q = np.vstack([brake, plan.q])
            plan.grip = np.concatenate([np.full(n, self._grip), plan.grip])
            plan.cartesian = np.vstack([np.repeat(plan.cartesian[:1], n, axis=0), plan.cartesian])
            plan.phases = [plan.phases[0]] * n + plan.phases
            for name in ("lift_check", "hold_until", "carry_start"):
                tick = getattr(plan, name)
                if tick is not None:
                    setattr(plan, name, tick + n)
            self._q = self._braking_from  # this tick still outputs where the arm is now
        self._plan = plan
        self.plans.append(plan)
        self._tick = 0
        self._plan_end = None
        self._lag_since = None
        self._lift_check_since = None
        self._set_phase(plan.phases[0], data)
        episode = self._episode
        if (self._start_gate and plan.kind == "attempt" and episode is not None and episode.targets_done == 0
                and episode.attempt == 1):
            # r16 (plan before record): planned; hold here until the recorder's go, then the start pause, then move
            self._await_go, self._go_time = True, None
            self._phase = "ready"


__all__ = [
    "ACTUATORS",
    "FR3_DDDQ_MAX",
    "FR3_DDQ_MAX",
    "FR3_DQ_MAX",
    "HOME_Q",
    "BucketSortDemonstrator",
    "DemoParams",
    "FrankaIK",
    "JointPlan",
    "MotionScript",
    "Planner",
    "SceneSnapshot",
    "hand_rotation",
    "joint_smoothness",
    "min_jerk",
    "move_duration",
    "object_shape",
    "wrap_near",
]
