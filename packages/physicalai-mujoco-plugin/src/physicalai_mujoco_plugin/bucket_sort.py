# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Layout, instructions and success for the ``bucket_sort`` scene.

Three buckets (black, blue, red) stand in front of the arm, and three objects (a resistor, a 9 V
battery and a CPU) lie between the arm and the buckets. Each episode names one object and one
bucket, "put the CPU in the red bucket". A reset:

- lays the buckets out anywhere in their area (:data:`BUCKET_X`, :data:`BUCKET_Y`), each turned
  up to :data:`BUCKET_YAW_RANGE` away from facing the robot, at least :data:`BUCKET_GAP` apart and
  in the arm's reach, and gives them the colours in a random order, so neither the place nor the
  order tells which bucket is meant (:func:`sample_layout`). The buckets are free bodies: the arm
  can push them and the viewer or ``POST /objects/{joint}/pose`` can move them, and the success
  test follows them wherever they are;
- drops the objects at random positions and headings in their area (:data:`OBJECT_X`,
  :data:`OBJECT_Y`), apart from each other and far enough from the buckets that a hand grasping
  from above clears the bucket walls;
- samples the instruction from the allowed (object, bucket) pairs, all nine by default, so pairs
  can be held out of training (:func:`sample_instruction`, :class:`BucketSortTask`), and words it
  in one of many ways (:data:`PHRASES`): "Drop the 9V battery in the blue bin.", "Can you put the
  chip in the red box?".

The episode succeeds when the named object rests inside the named bucket, out of the gripper,
and no other object is in a bucket (:meth:`BucketSortTask.evaluate`). :class:`BucketSortEpisode`
reports it in ``/health`` and the viewer, and starts a new episode after the success dwell.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.bucket_sort_objects import (
    KINDS,
    OBJECT_COLORS,
    PARK_CONTYPE,
    PARK_SPACING,
    PARKING,
    REAL_KINDS,
    kind_of,
    real_kind,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

OBJECTS = ("resistor", "battery", "cpu")
"""Object names: each is a body with the free joint ``<name>:joint``."""
OBJECT_LABELS = {"resistor": "resistor", "battery": "9 V battery", "cpu": "CPU"}
"""How the canonical instruction (:attr:`Instruction.canonical_text`) names each object."""
BUCKET_COLORS = ("black", "blue", "red")
"""Bucket colours: each bucket is the body ``bucket_<colour>`` with the free joint ``bucket_<colour>:joint``."""
BUCKET_JOINTS = tuple(f"bucket_{color}:joint" for color in BUCKET_COLORS)
"""The buckets' free joints, in :data:`BUCKET_COLORS` order."""
ALL_PAIRS: tuple[tuple[str, str], ...] = tuple((obj, color) for obj in OBJECTS for color in BUCKET_COLORS)
"""Every (object, bucket colour) pair an instruction can name."""
KNOWN_BUCKET_COLORS = (*BUCKET_COLORS, "green")
"""Every colour a scene's bucket may have. ``bucket_sort`` has :data:`BUCKET_COLORS`; a scene with other
buckets (``bucket_sort_desk``'s four) is read from its model (:func:`bucket_colors`)."""
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}
"""How an instruction counts the buckets ("of the four bins, ...")."""

BUCKET_SIZE = (0.150, 0.2025, 0.120)
"""The bucket's outer size in its own frame, in metres: across (x), long axis (y) and depth (z).

It lies on its long side with its opening up (``scripts/build_bucket_sort_meshes.py``)."""
BUCKET_YAW = float(np.pi / 2)
"""The turn about the vertical, in radians (counter-clockwise seen from above), that points a
bucket's open end straight at the robot.

The bucket's open end, above its low lip, is at +y in its own frame; a quarter turn points it at
the robot (-x), with the long axis along x. A reset adds up to :data:`BUCKET_YAW_RANGE` to it."""
BUCKET_INTERIOR = ((-0.069, 0.069), (-0.095, 0.095), (0.006, 0.120))
"""The space inside a bucket in its own frame, ``(x, y, z)`` ranges in metres: between its walls,
above its bottom, up to its rim."""

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
# Every range below is in the world frame, in metres (the robot base at the origin, facing +x).

TABLE_X = (-0.17, 0.83)
"""The table top's extent along x (``scene.xml``)."""
TABLE_Y = (-0.9, 0.9)
"""The table top's extent along y."""
TABLE_MARGIN = 0.03
"""Smallest distance from a bucket's footprint to the table's edges."""
BUCKET_X = (0.50, 0.72)
"""Where bucket centres are drawn along x, before the checks below."""
BUCKET_Y = (-0.40, 0.40)
"""Where bucket centres are drawn along y. The buckets are drawn one by one, each anywhere, so
they stand in a row, staggered, or two behind each other."""
BUCKET_YAW_RANGE = float(np.radians(25.0))
"""Largest turn of a bucket away from :data:`BUCKET_YAW`, either way, in radians: the open end
still faces the robot's side of the table."""
BUCKET_GAP = 0.03
"""Smallest gap between two buckets' footprints (their rotated outer rectangles)."""
OBJECT_X = (0.30, 0.48)
"""Where object centres are drawn along x."""
OBJECT_Y = (-0.34, 0.34)
"""Where object centres are drawn along y."""
OBJECT_GAP = 0.02
"""Smallest gap between the circles that enclose two objects at any heading."""
OBJECT_BUCKET_GAP = 0.06
"""Smallest gap between an object's enclosing circle and a bucket's footprint."""
REACH_RADIUS = (0.35, 0.76)
"""Horizontal distance from the base's axis, ``(nearest, farthest)``, at which the hand reaches
every object centre and every point of :data:`BUCKET_REACH_POINTS`.

Measured with damped least-squares IK of the FR3 with the fingers down, for four hand headings
(each with its half-turn twin), the null space pulling the joints away from their limits: at
0.35 m every joint stays at least 9 deg inside its limits from the table up to 0.16 m (the grasp
and the stop above it); nearer, the folded elbow comes within 7 deg. At 0.76 m the margin is over
45 deg up to 0.27 m (the carry over a bucket), and the reach ends at about 0.80 m. The nearest
radius also keeps the objects away from the arm's base."""
BUCKET_REACH_POINTS = ((0.0, -0.04), (0.0, 0.0), (0.0, 0.04))
"""Points in a bucket's frame, ``(x, y)``, that must be in :data:`REACH_RADIUS`: where a hand
over the opening lets an object go."""
HAND_HALF_SIZE = (0.110, 0.045)
"""Half the footprint of the Franka Hand about the grasp point, in metres: along the fingers'
closing axis and across it.

The hand's housing spans 0.106 m either way along the closing axis and 0.032 m across, from
0.03 m above the fingertips, so it is below a bucket's rim when the hand grasps an object on the
table. The margins cover a grasp turned 7 deg or shifted 6 mm along the object."""
HAND_BUCKET_GAP = 0.015
"""Smallest gap between the footprint of a hand grasping an object (:data:`HAND_HALF_SIZE`,
closing across the object's short side) and a bucket's footprint."""
FINGER_HALF_SIZE = (0.068, 0.012)
"""Half the footprint of the open fingers about the grasp point: along the closing axis and across."""
FINGER_OBJECT_GAP = 0.01
"""Smallest gap between the open fingers grasping one object and another object's enclosing circle."""
LAYOUT_DRAWS = 200
"""Draws per bucket or object before a layout is started over."""
LAYOUT_RESTARTS = 50
"""Layouts started over before :data:`FALLBACK_LAYOUT` is used (it never was in 100 000 seeds)."""
SLOT_NAMES = ("left", "centre", "right")
"""Names of the buckets from left to right as seen from behind the robot (decreasing y)."""
DROP_HEIGHT = 0.002
"""How far above the table an object is released at reset, in metres."""
SETTLE_SPEED = 0.02
"""Largest speed, in m/s, at which an object counts as resting."""

Pair = tuple[str, str]


@dataclass(frozen=True)
class Pose2D:
    """A pose on the table: the centre's ``x`` and ``y`` in metres and the ``yaw`` in radians."""

    x: float
    y: float
    yaw: float

    def as_dict(self) -> dict[str, float]:
        """Return the pose for JSON."""
        return {"x": self.x, "y": self.y, "yaw": self.yaw}


@dataclass(frozen=True)
class ObjectFootprint:
    """What the layout needs to know of an object lying flat."""

    radius: float
    """Horizontal radius at any heading, about the body origin."""
    half_height: float
    """Height of the body origin above the table."""
    closing_yaw: float
    """Heading in the body frame of the object's short horizontal side, across which a hand closes:
    0 for the body's x axis, a quarter turn for its y axis."""


@dataclass(frozen=True)
class Layout:
    """Where a reset puts the buckets and the objects."""

    buckets: dict[str, Pose2D]
    """Each bucket's pose by colour; the yaw includes :data:`BUCKET_YAW`."""
    objects: dict[str, Pose2D]
    """Each object's pose by name."""
    draws: int = 0
    """Candidate poses drawn to find this layout."""
    fallback: bool = False
    """Whether no layout was found and :data:`FALLBACK_LAYOUT` was used."""
    parked: tuple[str, ...] = ()
    """Objects of a pool scene that sit out this episode, parked on the hidden shelf."""
    pile: bool = False
    """Whether the objects lie in a pile (close together, some touching) rather than scattered."""
    object_colors: dict[str, str] = field(default_factory=dict)
    """The colour name each recoloured pool object got (:data:`bucket_sort_objects.OBJECT_COLORS`)."""
    targets: tuple[str, ...] = ()
    """The objects an instruction may name: in a pile, those a hand can close on without touching another;
    empty means every placed object."""
    heights: dict[str, float] = field(default_factory=dict)
    """Extra drop height of an object laid on top of another in a pile (metres); absent for one on the table."""
    upright: frozenset[str] = frozenset()
    """Objects standing on their end (:attr:`TaskLayout.upright`)."""

    @property
    def colors_left_to_right(self) -> tuple[str, ...]:
        """The bucket colours by decreasing y: left to right as seen from behind the robot."""
        return tuple(sorted(self.buckets, key=lambda color: -self.buckets[color].y))


FALLBACK_LAYOUT = Layout(
    buckets={color: Pose2D(0.645, y, BUCKET_YAW) for color, y in zip(BUCKET_COLORS, (0.22, 0.0, -0.22), strict=True)},
    objects={name: Pose2D(0.40, y, 0.0) for name, y in zip(OBJECTS, (0.17, 0.0, -0.17), strict=True)},
    fallback=True,
)
"""A layout that passes every check, for a generator that finds none."""


def _rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _rectangle(pose: Pose2D, half: tuple[float, float], yaw: float | None = None) -> np.ndarray:
    # The corners, counter-clockwise, of a rectangle of half sizes *half* (along its x and y) at *pose*.
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * np.asarray(half)
    return np.array([pose.x, pose.y]) + corners @ _rotation(pose.yaw if yaw is None else yaw).T


def _segment_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    # The distance from each of *points* (n, 2) to the boundary of *polygon* (m, 2).
    edges = np.roll(polygon, -1, axis=0) - polygon
    offset = points[:, None, :] - polygon[None, :, :]
    t = np.clip(np.einsum("nmi,mi->nm", offset, edges) / np.einsum("mi,mi->m", edges, edges), 0.0, 1.0)
    return np.linalg.norm(offset - t[..., None] * edges[None], axis=2).min(axis=1)


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    # Whether two convex polygons overlap: the separating axis test.
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        pa, pb = a @ normals.T, b @ normals.T
        if np.any((pa.max(axis=0) < pb.min(axis=0)) | (pb.max(axis=0) < pa.min(axis=0))):
            return False
    return True


def polygon_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Return the distance between two convex polygons (corners in order), or ``-1`` if they overlap.

    Returns:
        Metres.
    """
    if _overlap(a, b):
        return -1.0
    return float(min(_segment_distances(a, b).min(), _segment_distances(b, a).min()))


def circle_gap(centre: tuple[float, float], radius: float, polygon: np.ndarray) -> float:
    """Return the distance from a circle to a convex polygon (corners counter-clockwise), negative when they overlap.

    Returns:
        Metres.
    """
    point = np.asarray(centre, dtype=np.float64)
    edges = np.roll(polygon, -1, axis=0) - polygon
    inside = np.all(edges[:, 0] * (point[1] - polygon[:, 1]) - edges[:, 1] * (point[0] - polygon[:, 0]) >= 0.0)
    distance = float(_segment_distances(point[None], polygon)[0])
    return (-distance if inside else distance) - radius


def _far_apart(a: tuple[float, float], b: tuple[float, float], reach: float) -> bool:
    # Shapes within radii that add up to *reach* (with the gap) cannot come closer than the gap.
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 > reach**2


_BUCKET_RADIUS = float(np.hypot(BUCKET_SIZE[0], BUCKET_SIZE[1]) / 2.0)


SCATTER_MOST = 6
"""(The reach area holds no more than 6 of the real objects apart; :attr:`TaskLayout.scatter_kind_counts` may ask for fewer.)"""
SCATTER_MOST_REAL_LIKE = 5
"""Most objects a scattered kind-level table holds: the reBot's reach area (about 12 x 48 cm) has room for no more of
the real objects apart; a fuller table is a pile (:meth:`BucketSortTask._draw_kind_objects`)."""

NOVEL_TARGETS: dict[str, tuple[str, ...]] = {}
"""Unfamiliar everyday objects an episode may name (:attr:`TaskLayout.novel_target_fraction`): body kind -> its names
("mug", ...). Empty: none are modelled yet (the user decides later); the plumbing refuses a fraction above 0."""


@dataclass(frozen=True)
class TaskLayout:
    """Where one robot's layouts put the buckets and the objects (:func:`sample_layout`).

    The objects and buckets keep their size; only where they may stand changes per robot. The
    default is the Franka arms' layout, the module's constants; a smaller arm gets its areas and
    reach closer in (``scene_registry``'s ``profile_task_layouts``).
    """

    bucket_x: tuple[float, float] = BUCKET_X
    bucket_y: tuple[float, float] = BUCKET_Y
    object_x: tuple[float, float] = OBJECT_X
    object_y: tuple[float, float] = OBJECT_Y
    reach: tuple[float, float] = REACH_RADIUS
    """Distance from an arm's base axis at which it reaches every object centre."""
    bucket_reach: tuple[float, float] = REACH_RADIUS
    """Distance from an arm's base axis at which it reaches every point of :data:`BUCKET_REACH_POINTS`."""
    row: tuple[tuple[str, Pose2D], ...] = ()
    """A row of buckets measured on a real setup: each colour with its pose (yaw included). A reset
    moves each by up to :attr:`row_jitter` and, with :attr:`shuffle_row`, deals the colours to the
    row's places at random; empty draws the buckets anywhere in :attr:`bucket_x` and :attr:`bucket_y`."""
    row_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Largest move of a row bucket from its place: along x and y (metres) and its turn (radians)."""
    shuffle_row: bool = False
    """Deal the row's colours to its places at random at each reset."""
    table_x: tuple[float, float] | None = None
    """The table top's extent along x; ``None`` for :data:`TABLE_X`."""
    table_y: tuple[float, float] | None = None
    """The table top's extent along y; ``None`` for :data:`TABLE_Y`."""
    bucket_gap: float | None = None
    """Least gap between two buckets; ``None`` for :data:`BUCKET_GAP`. A real row can stand closer."""
    fill: tuple[float, float] | None = None
    """How full each bucket's contents are at a reset, a fraction drawn per bucket from this range; ``None``
    leaves them as the scene wrote them. The contents are geoms of the bucket body named
    ``bucket_<colour>_item<k>`` (collision) and ``bucket_<colour>_item<k>_<part>`` (visual), the lowest first;
    a lower fill hides the top ones (:func:`set_bucket_fill`)."""
    bases: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    """The arms' base axes on the table; an object or bucket point must be in reach of one of them."""
    hand_half_size: tuple[float, float] = HAND_HALF_SIZE
    """Half the hand's footprint about the grasp point (:data:`HAND_HALF_SIZE`)."""
    finger_half_size: tuple[float, float] = FINGER_HALF_SIZE
    face_bases: bool = False
    """Turn each bucket's open end toward its nearest base (plus the usual turn either way), not toward -x."""
    objects_reach_all: bool = False
    """Every object in reach of every base, not just one: with two arms, whichever arm reaches the
    named bucket also reaches the named object."""
    active_objects: tuple[int, int] | None = None
    """How many of a pool scene's objects an episode uses, drawn from this range (both ends included);
    ``None`` uses every object of the scene. The others are parked (:data:`bucket_sort_objects.PARKING`)."""
    pile_probability: float = 0.0
    """Probability that an episode lays its objects in a pile instead of scattering them."""
    pile_gap: tuple[float, float] = (0.0, 0.008)
    """Gap between the enclosing circles of neighbours in a pile, drawn per object (metres)."""
    pile_finger_gap: float = 0.004
    """Least gap between the open fingers on a pile's target and its neighbours."""
    sequence_probability: float = 0.0
    """Probability that, after a success, the next reset keeps the table as it is (objects already sorted
    stay in their buckets) and names another object still on the table: sorting a pile one by one."""
    real_look_probability: float = 1.0
    """Probability that an episode shows the real table and exposure; otherwise a reset draws a table colour and
    pattern (:data:`TABLE_LOOKS`) and an exposure gain, written to the model's ``r2s_*`` numerics, which a real2sim
    compositor reads (a scene without them ignores this)."""
    exposure_range: tuple[float, float] = (1.0, 1.0)
    """Exposure gain drawn for an episode that does not show the real look."""
    real_bucket_probability: float = 1.0
    """Probability that the buckets show their real colours; otherwise each shows a distinct colour of
    :data:`BUCKET_PALETTE` (:func:`randomize_buckets`)."""
    kind_counts: tuple[tuple[str, int, int], ...] | None = None
    """Kind-level episodes with the real objects: ``(kind, low, high)`` per real kind
    (:data:`bucket_sort_objects.REAL_KINDS`), how many of its pool bodies (``obj_<kind>_<k>``) a fresh layout lays out,
    drawn uniformly with both ends included and at least one of the named kind. The instruction then names a kind
    and a bucket, and every object of that kind on the table is a target (:attr:`Instruction.targets`), sorted one
    after another. ``None`` keeps per-object instructions."""
    distractors: tuple[int, int] = (0, 0)
    """With :attr:`kind_counts`: how many generic pool shapes (:data:`bucket_sort_objects.KINDS`) lie among the real
    objects as unnamed distractors, drawn uniformly; an instruction never names one."""
    scatter_finger_clearance: bool = True
    """Whether scattered objects must also leave room for the open fingers between them (``False``: only the object
    gap, so a crowded table, as on the real desk, still scatters; the demonstrator picks among neighbours as in a pile)."""
    stack_probability: float = 0.0
    """In a pile, the probability that an object is laid on top of one already placed (it drops onto it and settles,
    overlapping and partly hiding it) rather than beside it."""
    real_pairs: tuple[Pair, ...] = ()
    """With :attr:`kind_counts`: the (kind, bucket colour) pairs of the real task set."""
    real_pair_probability: float = 0.0
    """With :attr:`kind_counts`: probability that an episode's pair is drawn from :attr:`real_pairs` (uniformly);
    otherwise from the other allowed pairs (uniformly)."""
    upright: tuple[tuple[str, float], ...] = ()
    """With :attr:`kind_counts`: ``(kind, probability)``: each object of that real kind laid out on the table stands on its
    end with this probability (the real resistors: about one in five), never in a stack."""
    scatter_kind_counts: tuple[tuple[str, int, int], ...] | None = None
    """With :attr:`kind_counts`: the counts per real kind of a scattered table (at least one of the named kind, at most
    :data:`SCATTER_MOST` objects in all), when it differs from a pile's; ``None`` uses :attr:`kind_counts`."""
    stratified_retry: bool = False
    """With :attr:`kind_counts`: after an episode the demonstrator failed (:meth:`BucketSortTask.note_failure`), the next fresh layout keeps its stratum (the
    arrangement, the named kind and the number of targets) and draws everything else anew, so recording only the
    successes keeps the specified mix of strata instead of under-representing the harder ones."""
    novel_target_fraction: float = 0.0
    """Fraction of episodes whose target would be an unfamiliar everyday object (:data:`NOVEL_TARGETS`). No such assets
    exist: any value above 0 raises until they do."""

    def __post_init__(self) -> None:
        """Refuse settings that cannot be met."""
        if self.novel_target_fraction > 0.0 and not NOVEL_TARGETS:
            msg = "novel_target_fraction > 0 needs novel target objects (NOVEL_TARGETS is empty)"
            raise ValueError(msg)

    @property
    def hand_radius(self) -> float:
        """The radius of the circle around :attr:`hand_half_size`."""
        return float(np.hypot(*self.hand_half_size))

    @property
    def finger_radius(self) -> float:
        """The radius of the circle around :attr:`finger_half_size`."""
        return float(np.hypot(*self.finger_half_size))

    def nearest_base(self, x: float, y: float) -> tuple[float, float]:
        """Return the base nearest to ``(x, y)``.

        Returns:
            Its position.
        """
        return min(self.bases, key=lambda base: (x - base[0]) ** 2 + (y - base[1]) ** 2)


DEFAULT_LAYOUT = TaskLayout()
"""The Franka arms' layout: a base at the origin and the areas of the module's constants."""


def _current(layout: TaskLayout) -> TaskLayout:
    """Return *layout*, or for :data:`DEFAULT_LAYOUT` the module's constants as they are now.

    Returns:
        The layout to use.
    """
    if layout is not DEFAULT_LAYOUT:
        return layout
    return TaskLayout(
        bucket_x=BUCKET_X,
        bucket_y=BUCKET_Y,
        object_x=OBJECT_X,
        object_y=OBJECT_Y,
        reach=REACH_RADIUS,
        bucket_reach=REACH_RADIUS,
        hand_half_size=HAND_HALF_SIZE,
        finger_half_size=FINGER_HALF_SIZE,
    )


_HAND_RADIUS = DEFAULT_LAYOUT.hand_radius
_FINGER_RADIUS = DEFAULT_LAYOUT.finger_radius


def bucket_footprint(pose: Pose2D) -> np.ndarray:
    """Return a bucket's outer rectangle on the table at *pose*, corners counter-clockwise.

    Returns:
        ``(4, 2)`` corners in metres.
    """
    return _rectangle(pose, (BUCKET_SIZE[0] / 2.0, BUCKET_SIZE[1] / 2.0))


def _grasp_rectangle(pose: Pose2D, footprint: ObjectFootprint, half: tuple[float, float]) -> np.ndarray:
    # The rectangle *half* (along the closing axis, across) of a hand grasping an object at *pose*.
    return _rectangle(pose, half, pose.yaw + footprint.closing_yaw)


def _in_reach(
    x: float,
    y: float,
    radius: tuple[float, float] = REACH_RADIUS,
    bases: tuple[tuple[float, float], ...] = ((0.0, 0.0),),
) -> bool:
    for base_x, base_y in bases:
        dx, dy = x - base_x, y - base_y
        if radius[0] ** 2 <= dx * dx + dy * dy <= radius[1] ** 2:
            return True
    return False


def bucket_problems(
    pose: Pose2D, others: Iterable[Pose2D] = (), *, first: bool = False, layout: TaskLayout = DEFAULT_LAYOUT
) -> list[str]:
    """Return what keeps a bucket at *pose* out of a layout with the buckets *others*.

    Args:
        pose: The bucket's pose.
        others: The buckets placed before it.
        first: Stop at the first problem.
        layout: The robot's layout, for the reach.

    Returns:
        The problems; empty when the pose is fine.
    """
    problems: list[str] = []
    corners = bucket_footprint(pose)
    layout = _current(layout)
    (x_lo, x_hi), (y_lo, y_hi) = layout.table_x or TABLE_X, layout.table_y or TABLE_Y
    if (
        corners[:, 0].min() < x_lo + TABLE_MARGIN
        or corners[:, 0].max() > x_hi - TABLE_MARGIN
        or corners[:, 1].min() < y_lo + TABLE_MARGIN
        or corners[:, 1].max() > y_hi - TABLE_MARGIN
    ):
        problems.append("table edge")
        if first:
            return problems
    turn = _rotation(pose.yaw)
    points = [np.array([pose.x, pose.y]) + turn @ point for point in BUCKET_REACH_POINTS]
    if not all(_in_reach(float(x), float(y), layout.bucket_reach, layout.bases) for x, y in points):
        problems.append("reach")
        if first:
            return problems
    gap = BUCKET_GAP if layout.bucket_gap is None else layout.bucket_gap
    for other in others:
        if _far_apart((pose.x, pose.y), (other.x, other.y), 2.0 * _BUCKET_RADIUS + gap):
            continue
        if polygon_gap(corners, bucket_footprint(other)) < gap:
            problems.append("bucket gap")
            break
    return problems


def object_problems(
    name: str,
    pose: Pose2D,
    footprints: Mapping[str, ObjectFootprint],
    buckets: Iterable[Pose2D],
    others: Mapping[str, Pose2D] | None = None,
    *,
    first: bool = False,
    layout: TaskLayout = DEFAULT_LAYOUT,
    fingers_clear: bool = True,
) -> list[str]:
    """Return what keeps object *name* at *pose* out of a layout with *buckets* and the objects *others*.

    Args:
        name: The object.
        pose: Its pose.
        footprints: Every object's footprint (:func:`object_footprint`).
        buckets: The buckets' poses.
        others: The objects placed before it.
        first: Stop at the first problem.
        layout: The robot's layout, for the reach and the hand's footprint.
        fingers_clear: Also require room for the open fingers between it and the others.

    Returns:
        The problems, sorted; empty when the pose is fine.
    """
    problems: set[str] = set()
    default = layout is DEFAULT_LAYOUT
    layout = _current(layout)
    shape = footprints[name]
    centre = (pose.x, pose.y)
    in_reach = (
        all(_in_reach(pose.x, pose.y, layout.reach, (base,)) for base in layout.bases)
        if layout.objects_reach_all
        else _in_reach(pose.x, pose.y, layout.reach, layout.bases)
    )
    if not in_reach:
        problems.add("reach")
        if first:
            return sorted(problems)
    # on the desk with a margin: the reach is a ring about the base, so without this an object could be placed beside or
    # behind the base, past the desk's back edge, and fall off (r08 D4, seed 233291)
    table_x, table_y = layout.table_x or TABLE_X, layout.table_y or TABLE_Y
    if (
        pose.x - shape.radius < table_x[0] + TABLE_MARGIN
        or pose.x + shape.radius > table_x[1] - TABLE_MARGIN
        or pose.y - shape.radius < table_y[0] + TABLE_MARGIN
        or pose.y + shape.radius > table_y[1] - TABLE_MARGIN
    ):
        problems.add("off table")
        if first:
            return sorted(problems)
    hand_radius, finger_radius = (
        (_HAND_RADIUS, _FINGER_RADIUS) if default else (layout.hand_radius, layout.finger_radius)
    )
    hand = _grasp_rectangle(pose, shape, layout.hand_half_size)
    for bucket in buckets:
        if _far_apart(
            centre,
            (bucket.x, bucket.y),
            _BUCKET_RADIUS + max(shape.radius + OBJECT_BUCKET_GAP, hand_radius + HAND_BUCKET_GAP),
        ):
            continue
        corners = bucket_footprint(bucket)
        if circle_gap(centre, shape.radius, corners) < OBJECT_BUCKET_GAP:
            problems.add("object near bucket")
        if polygon_gap(hand, corners) < HAND_BUCKET_GAP:
            problems.add("hand near bucket")
        if first and problems:
            return sorted(problems)
    fingers = _grasp_rectangle(pose, shape, layout.finger_half_size)
    for other, other_pose in (others or {}).items():
        other_shape = footprints[other]
        other_centre = (other_pose.x, other_pose.y)
        if not _far_apart(centre, other_centre, shape.radius + other_shape.radius + OBJECT_GAP):
            problems.add("object gap")
        elif fingers_clear and not _far_apart(
            centre, other_centre, finger_radius + max(shape.radius, other_shape.radius) + FINGER_OBJECT_GAP
        ) and (
            circle_gap(other_centre, other_shape.radius, fingers) < FINGER_OBJECT_GAP
            or circle_gap(centre, shape.radius, _grasp_rectangle(other_pose, other_shape, layout.finger_half_size))
            < FINGER_OBJECT_GAP
        ):
            problems.add("fingers near object")
        if first and problems:
            break
    return sorted(problems)


TABLE_LOOKS: dict[str, tuple[float, float, float]] = {
    "white": (0.88, 0.88, 0.86),
    "light grey": (0.70, 0.71, 0.72),
    "dark grey": (0.32, 0.33, 0.35),
    "black": (0.10, 0.10, 0.11),
    "beige": (0.82, 0.74, 0.60),
    "light wood": (0.76, 0.60, 0.42),
    "dark wood": (0.42, 0.28, 0.18),
    "green mat": (0.18, 0.38, 0.27),
    "blue": (0.30, 0.45, 0.65),
    "red": (0.60, 0.20, 0.18),
}
"""Table colours a randomised episode draws from (albedo), with a pattern from :data:`TABLE_PATTERNS`."""
TABLE_PATTERNS = ("plain", "grain", "speckle", "grid")
"""Patterns the compositor lays over a randomised table colour (0, 1, 2, 3 in ``r2s_table_look``)."""


def _numeric(model: object, name: str) -> int | None:
    import mujoco  # noqa: PLC0415

    k = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name))
    return int(model.numeric_adr[k]) if k >= 0 else None


LOOK_KNOBS: dict[str, dict[str, object]] = {
    "table": {"what": "table-top colour (K1)", "values": "real, or one of TABLE_LOOKS x TABLE_PATTERNS, albedo x U(0.92, 1.06)"},
    "exposure": {"what": "image exposure gain (K7)", "range": [0.8, 1.2], "default": 1.0},
    "white_balance": {"what": "colour temperature as R/B gains 1 +- 0.08 t (K5/K7)", "range": [-1.0, 1.0], "default": 0.0},
    "light": {"what": "sim key light: direction offset (m at 2.4 m) and intensity (K5)", "range": [[-0.6, 0.6], [-0.6, 0.6], [0.75, 1.25]], "default": [0, 0, 1]},
    "camera_jitter": {"what": "overview and wrist extrinsics: +-8 mm, +-1.5 deg per axis (K6)", "default": 0},
    "noise": {"what": "sensor noise sigma, 8-bit levels (K7)", "range": [0.0, 3.0], "default": 0.0},
    "blur": {"what": "image blur sigma, px (K7)", "range": [0.0, 0.8], "default": 0.0},
}
"""The randomised look's knobs, ranges and defaults (the real configuration is the default of each)."""


def randomize_look(model: object, rng: np.random.Generator, layout: TaskLayout) -> dict[str, object]:
    """Draw the episode's table look, light, camera jitter and image model into the model's ``r2s_*`` numerics.

    ``r2s_table_look`` holds ``(use, r, g, b, pattern, scale, phase)`` (``use`` 0 keeps the real table),
    ``r2s_exposure`` the gain, ``r2s_wb`` R/G/B gains, ``r2s_light`` ``(dx, dy, intensity)``, ``r2s_cam_jitter``
    ``(overview dx dy dz rx ry rz, wrist dx dy dz rx ry rz)`` in metres and radians, ``r2s_image`` ``(noise, blur)``.
    A real2sim compositor reads them (:mod:`.real2sim.desk_compositor`); a scene without them ignores this. With
    :attr:`TaskLayout.real_look_probability` everything is the real configuration.

    Returns:
        What was drawn, for the episode's description (``variant``).
    """
    look_adr = _numeric(model, "r2s_table_look")
    if look_adr is None:
        return {}
    real = rng.random() < layout.real_look_probability
    table = np.zeros(7)
    exposure, wb_t, light, jitter, noise, blur = 1.0, 0.0, np.array([0.0, 0.0, 1.0]), np.zeros(12), 0.0, 0.0
    look: dict[str, object] = {"variant": "default" if real else "rand", "table": "real"}
    if not real:
        if rng.random() < 0.85:  # most randomised episodes recolour the table; some keep the real one
            name = list(TABLE_LOOKS)[int(rng.integers(len(TABLE_LOOKS)))]
            pattern = int(rng.integers(len(TABLE_PATTERNS)))
            rgb = np.clip(np.asarray(TABLE_LOOKS[name]) * rng.uniform(0.92, 1.06, size=3), 0.0, 1.0)
            table = np.array([1.0, *rgb, pattern, float(rng.uniform(0.6, 1.6)), float(rng.uniform(0.0, 6.283))])
            look.update(table=name, pattern=TABLE_PATTERNS[pattern])
        exposure = float(rng.uniform(*layout.exposure_range))
        wb_t = float(rng.uniform(-1.0, 1.0))
        light = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(0.75, 1.25)])
        jitter = np.concatenate([np.r_[rng.uniform(-0.008, 0.008, 3), rng.uniform(-0.026, 0.026, 3)] for _ in range(2)])
        noise, blur = float(rng.uniform(0.0, 3.0)), float(rng.uniform(0.0, 0.8))
    model.numeric_data[look_adr : look_adr + 7] = table
    for name, values in (("r2s_exposure", [exposure]), ("r2s_wb", [1 + 0.08 * wb_t, 1.0, 1 - 0.08 * wb_t]),
                         ("r2s_light", light), ("r2s_cam_jitter", jitter), ("r2s_image", [noise, blur])):
        adr = _numeric(model, name)
        if adr is not None:
            model.numeric_data[adr : adr + len(values)] = values
    look.update(exposure=round(exposure, 4), white_balance=round(wb_t, 4), light=np.round(light, 4).tolist(),
                camera_jitter=np.round(jitter, 5).tolist(), noise=round(noise, 3), blur=round(blur, 3))
    return look


BUCKET_PALETTE: dict[str, tuple[float, float, float]] = {
    "black": (0.10, 0.10, 0.11),
    "green": (0.15, 0.55, 0.28),
    "red": (0.80, 0.12, 0.10),
    "blue": (0.12, 0.32, 0.80),
    "yellow": (0.92, 0.78, 0.12),
    "orange": (0.95, 0.45, 0.10),
    "purple": (0.45, 0.22, 0.65),
    "white": (0.88, 0.88, 0.86),
    "grey": (0.50, 0.51, 0.52),
    "pink": (0.95, 0.50, 0.66),
}
"""Colours a recoloured bucket may show (albedo), by the name instructions use."""


def randomize_buckets(model: object, rng: np.random.Generator, layout: TaskLayout) -> dict[str, str]:
    """Draw the colour each bucket shows this episode, distinct across buckets, into ``r2s_bucket_rgb`` and the sim
    buckets' materials; the real colours with :attr:`TaskLayout.real_bucket_probability`.

    ``r2s_bucket_rgb`` holds ``(use, r, g, b)`` per bucket in :func:`bucket_colors` order; ``use`` 0 keeps the real one.

    Returns:
        The shown colour name of each bucket body (identity when nothing was recoloured or the scene has no numeric).
    """
    import mujoco  # noqa: PLC0415

    colors = bucket_colors(model)
    k = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "r2s_bucket_rgb"))
    if k < 0 or rng.random() < layout.real_bucket_probability:
        names = {c: c for c in colors}
    else:
        pick = rng.permutation(len(BUCKET_PALETTE))[: len(colors)]
        names = {c: list(BUCKET_PALETTE)[int(i)] for c, i in zip(colors, pick, strict=True)}
    if k >= 0:
        adr = int(model.numeric_adr[k])
        for i, c in enumerate(colors):
            shown = names[c]
            rgb = np.asarray(BUCKET_PALETTE[shown]) if shown != c else np.zeros(3)
            model.numeric_data[adr + 4 * i : adr + 4 * i + 4] = [float(shown != c), *rgb]
            material = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f"bucket_{c}"))
            if material >= 0:
                if not hasattr(randomize_buckets, "_real"):
                    randomize_buckets._real = {}  # type: ignore[attr-defined]
                real = randomize_buckets._real.setdefault((id(model), material), model.mat_rgba[material, :3].copy())  # type: ignore[attr-defined]
                model.mat_rgba[material, :3] = rgb if shown != c else real
    return names


def _draw_bucket(rng: np.random.Generator, layout: TaskLayout, place: Pose2D | None) -> Pose2D:
    """Draw a bucket pose: near its row *place*, or anywhere in the layout's bucket area.

    Returns:
        The pose.
    """
    if place is not None:
        dx, dy, dyaw = layout.row_jitter
        return Pose2D(
            place.x + float(rng.uniform(-dx, dx)),
            place.y + float(rng.uniform(-dy, dy)),
            place.yaw + float(rng.uniform(-dyaw, dyaw)),
        )
    x, y = float(rng.uniform(*layout.bucket_x)), float(rng.uniform(*layout.bucket_y))
    yaw = BUCKET_YAW
    if layout.face_bases:
        base_x, base_y = layout.nearest_base(x, y)
        yaw += float(np.arctan2(y - base_y, x - base_x))
    return Pose2D(x, y, yaw + float(rng.uniform(-BUCKET_YAW_RANGE, BUCKET_YAW_RANGE)))


def sample_layout(
    rng: np.random.Generator,
    footprints: Mapping[str, ObjectFootprint],
    layout: TaskLayout = DEFAULT_LAYOUT,
    colors: tuple[str, ...] = BUCKET_COLORS,
    objects: tuple[str, ...] = OBJECTS,
    required: str | None = None,
    chosen: tuple[str, ...] | None = None,
    pile: bool | None = None,
    protect: frozenset[str] = frozenset(),
    upright: frozenset[str] = frozenset(),
) -> Layout:
    """Draw a layout of the buckets and the objects that passes every check.

    The colours, then the objects, are placed in a random order, each drawn uniformly in its area
    (and heading range) until it passes :func:`bucket_problems` or :func:`object_problems` with
    those placed before it; a bucket or object that finds no place in :data:`LAYOUT_DRAWS` draws
    starts the layout over. The same generator state gives the same layout.

    Args:
        rng: The generator.
        footprints: Each object's :class:`ObjectFootprint` (:func:`object_footprint`).
        layout: Where the robot's buckets and objects may stand; with a :attr:`TaskLayout.row`, the
            buckets stand in that row instead.
        colors: The colours of the scene's buckets (:func:`bucket_colors`); a row names its own.
        objects: The scene's objects (:func:`scene_objects`); with :attr:`TaskLayout.active_objects`, a
            random subset of them is laid out and the rest parked.
        required: An object that must be laid out (and, in a pile, be a target).
        chosen: The objects to lay out (the rest parked), instead of drawing them (:attr:`TaskLayout.active_objects`).
        pile: Lay them in a pile (``True``) or scatter them (``False``) instead of drawing it
            (:attr:`TaskLayout.pile_probability`).
        protect: Objects (the instruction's targets) that no other object may be stacked on in a pile: freeing one
            would mean moving an object the instruction does not name.
        upright: Objects standing on their end: *footprints* must hold their upright footprints; never stacked.

    Returns:
        The layout; :data:`FALLBACK_LAYOUT` (or the row as measured) when :data:`LAYOUT_RESTARTS`
        layouts all failed.
    """
    draws = 0
    default = layout is DEFAULT_LAYOUT
    layout = _current(layout)
    colors = tuple(color for color, _ in layout.row) or colors
    # the arrangement is drawn once per layout (redrawing it at every restart would favour piles, which fit more often)
    use_pile = pile if pile is not None else layout.pile_probability > 0.0 and rng.random() < layout.pile_probability
    for restart in range(LAYOUT_RESTARTS):
        if not use_pile and layout.pile_probability > 0.0 and restart == LAYOUT_RESTARTS // 2:
            use_pile = True   # the scattered objects found no room in half the restarts: a pile instead
            logger.debug("Bucket sort: no room to scatter {} objects; laying them in a pile", len(chosen or objects))
        buckets: dict[str, Pose2D] = {}
        places = [pose for _, pose in layout.row]
        if layout.row and layout.shuffle_row:
            places = [places[int(i)] for i in rng.permutation(len(places))]
        for index in rng.permutation(len(colors)):
            for _ in range(LAYOUT_DRAWS):
                draws += 1
                pose = _draw_bucket(rng, layout, places[int(index)] if layout.row else None)
                if not bucket_problems(
                    pose, buckets.values(), first=True, layout=DEFAULT_LAYOUT if default else layout
                ):
                    buckets[colors[int(index)]] = pose
                    break
            else:
                break
        if len(buckets) < len(colors):
            continue
        if chosen is not None:
            pass
        elif layout.active_objects is None:
            chosen = objects
        else:
            low, high = layout.active_objects  # type: ignore[misc]
            count = int(rng.integers(min(low, len(objects)), min(high, len(objects)) + 1))
            pool = [o for o in objects if o != required]
            picked = {pool[int(i)] for i in rng.choice(len(pool), size=count - (required is not None), replace=False)}
            chosen = tuple(o for o in objects if o in picked or o == required)
        if use_pile:
            placed = _pile(rng, footprints, layout, chosen, buckets.values(), default=default, protect=protect, upright=upright)
            draws += 1
            if placed is None:
                continue
            pile_objects, targets, heights = placed
            if required is not None and required not in targets:
                continue
            return Layout(
                buckets={color: buckets[color] for color in colors},
                objects={name: pile_objects[name] for name in chosen},
                draws=draws,
                parked=tuple(name for name in objects if name not in chosen),
                pile=True,
                targets=targets,
                heights=heights,
                upright=frozenset(n for n in upright if n in chosen),
            )
        objects_placed: dict[str, Pose2D] = {}
        for index in rng.permutation(len(chosen)):
            name = chosen[int(index)]
            for _ in range(LAYOUT_DRAWS):
                draws += 1
                pose = Pose2D(
                    float(rng.uniform(*layout.object_x)),
                    float(rng.uniform(*layout.object_y)),
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                )
                area = DEFAULT_LAYOUT if default else layout
                if not object_problems(
                    name, pose, footprints, buckets.values(), objects_placed, first=True, layout=area,
                    fingers_clear=layout.scatter_finger_clearance,
                ):
                    objects_placed[name] = pose
                    break
            else:
                break
        if len(objects_placed) == len(chosen):
            return Layout(
                buckets={color: buckets[color] for color in colors},
                objects={name: objects_placed[name] for name in chosen},
                draws=draws,
                parked=tuple(name for name in objects if name not in chosen),
                upright=frozenset(n for n in upright if n in chosen),
            )
    logger.warning("Bucket sort: no layout found in {} draws; using the fallback layout", draws)
    fallback_buckets = dict(layout.row) if layout.row else FALLBACK_LAYOUT.buckets
    if objects != OBJECTS:  # a pool scene: three of its objects in the fallback's places, the rest parked
        fallback_objects = dict(zip(objects, FALLBACK_LAYOUT.objects.values(), strict=False))
        parked = tuple(objects[len(fallback_objects) :])
        return Layout(
            buckets=fallback_buckets, objects=fallback_objects, draws=draws, fallback=True, parked=parked
        )
    return Layout(buckets=fallback_buckets, objects=FALLBACK_LAYOUT.objects, draws=draws, fallback=True)


def _pile(
    rng: np.random.Generator,
    footprints: Mapping[str, ObjectFootprint],
    layout: TaskLayout,
    names: tuple[str, ...],
    buckets: Iterable[Pose2D],
    *,
    default: bool,
    protect: frozenset[str] = frozenset(),
    upright: frozenset[str] = frozenset(),
) -> tuple[dict[str, Pose2D], tuple[str, ...], dict[str, float]] | None:
    """Lay *names* out in a pile: each next to one already placed, their enclosing circles a few mm apart, or (with
    :attr:`TaskLayout.stack_probability`) on top of it, a little off its centre, to drop and settle there.

    Every object must still pass the reach and bucket checks of :func:`object_problems`; the pile's
    targets are the objects whose open fingers clear their neighbours by :attr:`TaskLayout.pile_finger_gap`.

    Returns:
        The poses, the targets and the extra drop heights of stacked objects, or ``None`` when no pile fits or no
        object can be a target.
    """
    buckets = list(buckets)
    area = DEFAULT_LAYOUT if default else layout
    finger_half = layout.finger_half_size
    for _ in range(20):
        placed: dict[str, Pose2D] = {}
        heights: dict[str, float] = {}
        centre = (float(rng.uniform(*layout.object_x)), float(rng.uniform(*layout.object_y)))
        for name in (names[int(i)] for i in rng.permutation(len(names))):
            radius = footprints[name].radius
            on_top = None
            if placed and name not in upright and layout.stack_probability > 0.0 and rng.random() < layout.stack_probability:
                # an object the instruction does not name never lies on one it names (nor anything near it); nothing
                # lies on a standing object, and a standing one is never stacked
                bases = [n for n in placed if n not in upright and (name in protect or (n not in protect and not _near_any(placed[n], placed, protect, footprints)))]
                on_top = bases[int(rng.integers(len(bases)))] if bases else None
            for _ in range(LAYOUT_DRAWS):
                yaw = float(rng.uniform(0.0, 2.0 * np.pi))
                if on_top is not None:
                    below = placed[on_top]
                    offset = float(rng.uniform(0.0, 0.5 * footprints[on_top].radius))
                    angle = float(rng.uniform(0.0, 2.0 * np.pi))
                    pose = Pose2D(below.x + offset * np.cos(angle), below.y + offset * np.sin(angle), yaw)
                    if object_problems(name, pose, footprints, buckets, None, first=True, layout=area):
                        continue
                    # drop from above everything its footprint overlaps, not only the one it lands on
                    under = [
                        n for n, o in placed.items()
                        if not _far_apart((pose.x, pose.y), (o.x, o.y), radius + footprints[n].radius)
                    ]
                    if name not in protect and any(n in protect for n in under):
                        continue   # its overhang would reach a named object
                    placed[name] = pose
                    heights[name] = max(heights.get(n, 0.0) + 2.0 * footprints[n].half_height for n in under) + 0.003
                    break
                if not placed:
                    x, y = centre
                else:
                    anchor = list(placed.values())[int(rng.integers(len(placed)))]
                    anchor_name = next(n for n, p in placed.items() if p is anchor)
                    distance = radius + footprints[anchor_name].radius + float(rng.uniform(*layout.pile_gap))
                    angle = float(rng.uniform(0.0, 2.0 * np.pi))
                    x, y = anchor.x + distance * np.cos(angle), anchor.y + distance * np.sin(angle)
                pose = Pose2D(float(x), float(y), yaw)
                if any(
                    not _far_apart((pose.x, pose.y), (o.x, o.y), radius + footprints[n].radius + layout.pile_gap[0] - 1e-6)
                    for n, o in placed.items()
                    # beside the objects on the table, and below any stacked one whose underside is lower than its top
                    if n not in heights or heights[n] < 2.0 * footprints[name].half_height + 0.003
                ):
                    continue
                if object_problems(name, pose, footprints, buckets, None, first=True, layout=area):
                    continue
                placed[name] = pose
                break
            else:
                break
        if len(placed) != len(names):
            continue
        targets = tuple(
            name
            for name in names
            if name not in heights and not any(n in heights for n in names if n != name and _covers(placed[n], placed[name], footprints[name].radius))
            and all(
                circle_gap((o.x, o.y), footprints[n].radius, _grasp_rectangle(placed[name], footprints[name], finger_half))
                >= layout.pile_finger_gap
                for n, o in placed.items()
                if n != name and n not in heights
            )
        )
        if targets:
            return placed, targets, heights
    return None


def _near_any(pose: Pose2D, placed: Mapping[str, Pose2D], names: frozenset[str], footprints: Mapping[str, ObjectFootprint]) -> bool:
    """Whether *pose* is within reach of an overhang of any of *names* already placed (its circle touches theirs)."""
    return any(
        not _far_apart((pose.x, pose.y), (placed[n].x, placed[n].y), 2.5 * footprints[n].radius)
        for n in names if n in placed
    )


def _covers(top: Pose2D, below: Pose2D, radius: float) -> bool:
    """Whether an object stacked at *top* lies over one at *below* (its centre within *radius* of it)."""
    return bool(np.hypot(top.x - below.x, top.y - below.y) < radius)


# ----------------------------------------------------------------------
# Instructions
# ----------------------------------------------------------------------

VARY_INSTRUCTION_TEXT = True
"""Whether instructions are worded in many ways (:data:`PHRASES`) by default; ``False`` always
gives the canonical sentence. :class:`BucketSortTask` can override it."""

Weighted = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class InstructionPhrases:
    """The words an instruction is made of; :data:`PHRASES` is the scene's set.

    A worded instruction is, with probability :attr:`canonical_fraction`, the canonical sentence
    (:attr:`Instruction.canonical_text`, "put the CPU in the red bucket"); otherwise a template
    filled with a name of the object, a container word and the bucket's colour, in sentence case,
    with an optional "please" and a final full stop. Every choice is weighted (the weights of a
    list need not sum to one).

    To extend it, add entries to the lists below. A name must name only its object (no word of
    another object's names, and no colour word), and a template must hold ``{object}`` once and
    either ``{bucket}`` (the colour and a container word: "red bin") or ``{containers}`` before
    ``{color} one`` ("of the three bins, ... the red one"); ``tests/test_bucket_sort.py`` checks
    that every sentence names the right object and colour and nothing else.
    """

    canonical_fraction: float
    """Probability of the canonical sentence, word for word."""
    templates: tuple[tuple[str, float, Literal["imperative", "question", "plain"]], ...]
    """``(template, weight, kind)``: an imperative can take "please" before or after it, a
    question takes it after "can you"/"could you" and ends in a question mark, a plain sentence takes none."""
    object_names: dict[str, Weighted]
    """Names of each object, used after "the"."""
    containers: Weighted
    """Words for a bucket."""
    plurals: dict[str, str]
    """The plural of each container word."""
    please: float
    """Probability of "please" in a template that can take it."""
    please_first: float
    """Of those, probability that "please" opens the sentence rather than ending it."""
    full_stop: float
    """Probability of a full stop after a sentence that is not a question."""


PHRASES = InstructionPhrases(
    canonical_fraction=0.3,
    templates=(
        ("put the {object} in the {bucket}", 2.0, "imperative"),
        ("put the {object} into the {bucket}", 1.0, "imperative"),
        ("place the {object} in the {bucket}", 1.0, "imperative"),
        ("place the {object} into the {bucket}", 1.0, "imperative"),
        ("drop the {object} in the {bucket}", 1.0, "imperative"),
        ("move the {object} to the {bucket}", 1.0, "imperative"),
        ("sort the {object} into the {bucket}", 1.0, "imperative"),
        ("take the {object} and put it in the {bucket}", 1.0, "imperative"),
        ("pick up the {object} and drop it in the {bucket}", 1.0, "imperative"),
        ("grab the {object} and place it in the {bucket}", 1.0, "imperative"),
        ("can you put the {object} in the {bucket}", 1.0, "question"),
        ("could you place the {object} into the {bucket}", 1.0, "question"),
        ("I want the {object} in the {bucket}", 1.0, "plain"),
        ("the {object} goes in the {bucket}", 1.0, "plain"),
        ("the {object} belongs in the {bucket}", 1.0, "plain"),
        ("of the {count} {containers}, put the {object} in the {color} one", 1.0, "plain"),
    ),
    object_names={
        "battery": (
            ("battery", 3.0),
            ("9 V battery", 2.0),
            ("9V battery", 1.0),
            ("9 volt battery", 1.0),
            ("nine volt battery", 1.0),
        ),
        "cpu": (("CPU", 3.0), ("processor", 1.5), ("chip", 1.5), ("microchip", 1.0), ("computer chip", 1.0)),
        "resistor": (("resistor", 1.0),),
    },
    containers=(("bucket", 3.0), ("bin", 2.0), ("box", 1.5), ("container", 1.5), ("tub", 1.0)),
    plurals={"bucket": "buckets", "bin": "bins", "box": "boxes", "container": "containers", "tub": "tubs"},
    please=0.25,
    please_first=0.6,
    full_stop=0.5,
)
"""The scene's instruction wording (:class:`InstructionPhrases`).

The resistor has no other name: it is the longest of the three objects, so "small resistor" would
point at the wrong one, and "the resistor" is what the templates already say."""


def _choose(rng: np.random.Generator, options: Weighted) -> str:
    return options[int(rng.choice(len(options), p=_normalized([weight for _, weight in options])))][0]


def canonical_text(obj: str, color: str) -> str:
    """Return the canonical instruction, e.g. ``put the CPU in the red bucket``.

    Returns:
        The sentence.
    """
    return f"put the {OBJECT_LABELS[obj]} in the {color} bucket"


def word_instruction(
    rng: np.random.Generator,
    obj: str,
    color: str,
    phrases: InstructionPhrases = PHRASES,
    *,
    count: int = 3,
    names: Weighted | None = None,
) -> str:
    """Word the instruction "put *obj* in the *color* bucket" at random (:class:`InstructionPhrases`).

    Args:
        rng: The generator.
        obj: One of :data:`OBJECTS`.
        color: One of the scene's bucket colours.
        phrases: The words to use.
        count: How many buckets the scene has.
        names: Names of the object to choose from, weighted; ``None`` for :attr:`InstructionPhrases.object_names`.

    Returns:
        The sentence.
    """
    if rng.random() < phrases.canonical_fraction:
        if names is not None:  # a pool object: its canonical label is the first name
            return f"put the {names[0][0]} in the {color} bucket"
        return canonical_text(obj, color)
    index = int(rng.choice(len(phrases.templates), p=_normalized([weight for _, weight, _ in phrases.templates])))
    template, _, kind = phrases.templates[index]
    name = _choose(rng, names if names is not None else phrases.object_names[obj])
    container = _choose(rng, phrases.containers)
    text = template.format(
        object=name,
        bucket=f"{color} {container}",
        containers=phrases.plurals[container],
        color=color,
        count=NUMBER_WORDS.get(count, str(count)),
    )
    if kind != "plain" and rng.random() < phrases.please:
        if kind == "question":
            text = text.replace(" you ", " you please ", 1)
        elif rng.random() < phrases.please_first:
            text = f"please {text}"
        else:
            text = f"{text}, please"
    if kind == "question":
        text += "?"
    elif rng.random() < phrases.full_stop:
        text += "."
    return text[0].upper() + text[1:]


REAL_CONTAINER_NOUNS: dict[str, str] = {"red": "bin", "blue": "bucket", "green": "box", "black": "box"}
"""The container words of the real reBot task strings ("put the resistors inside the red bin", "Put the CPU in the
blue bucket", "... green box", "... black box"), by the colour the container shows."""
CONTAINER_NOUNS: tuple[str, ...] = ("bin", "bucket", "box")
CONTAINER_PLURALS: dict[str, str] = {"bin": "bins", "bucket": "buckets", "box": "boxes"}
REAL_NOUN_PROBABILITY = 0.85
"""Probability that a container is called by its real word; otherwise by one of the other two, uniformly."""

KIND_NAMES: dict[str, tuple[Weighted, Weighted]] = {
    "resistor": ((("resistor", 1.0),), (("resistors", 1.0),)),
    "cpu": ((("CPU", 3.0), ("chip", 0.5), ("processor", 0.5)), (("CPUs", 3.0), ("chips", 0.5), ("processors", 0.5))),
    "battery": (
        (("battery", 3.0), ("9V battery", 2.0), ("9 V battery", 1.0), ("9 volt battery", 1.0)),
        (("batteries", 3.0), ("9V batteries", 2.0), ("9 V batteries", 1.0), ("9 volt batteries", 1.0)),
    ),
}
"""Names of each real kind, ``(singular, plural)``, weighted, the first canonical. No colour words: colour words name
containers only."""


@dataclass(frozen=True)
class KindPhrases:
    """How a kind-level instruction is worded (:func:`word_kind_instruction`); :data:`KIND_PHRASES` is the set."""

    real_form: float
    """Probability of the real task strings' form: "put the {object} inside/in the {colour} {noun}"."""
    real_form_lower: float
    """Of those, probability of a lower-case start (the real strings have both)."""
    real_form_full_stop: float
    """Of those, probability of a final full stop (the real strings have none)."""
    plural: float
    """With several targets, probability of the plural ("the resistors"); otherwise the collective singular, as the
    real "Put the CPU in the blue bucket" said of two CPUs."""
    templates: tuple[tuple[str, float, Literal["imperative", "question", "plain"]], ...]
    """Paraphrases: ``{object}``, ``{bucket}`` (colour and noun) or ``{containers}``/``{color}``, and ``{it}``,
    ``{go}``, ``{belong}`` that agree with a plural object."""
    please: float
    please_first: float
    full_stop: float


KIND_PHRASES = KindPhrases(
    real_form=0.5,
    real_form_lower=0.5,
    real_form_full_stop=0.2,
    plural=0.75,
    templates=(
        ("put the {object} in the {bucket}", 2.0, "imperative"),
        ("put the {object} into the {bucket}", 1.0, "imperative"),
        ("place the {object} in the {bucket}", 1.0, "imperative"),
        ("place the {object} inside the {bucket}", 1.0, "imperative"),
        ("drop the {object} into the {bucket}", 1.0, "imperative"),
        ("move the {object} to the {bucket}", 1.0, "imperative"),
        ("sort the {object} into the {bucket}", 1.0, "imperative"),
        ("take the {object} and put {it} in the {bucket}", 1.0, "imperative"),
        ("pick up the {object} and drop {it} in the {bucket}", 1.0, "imperative"),
        ("grab the {object} and place {it} inside the {bucket}", 1.0, "imperative"),
        ("can you put the {object} in the {bucket}", 1.0, "question"),
        ("could you place the {object} inside the {bucket}", 1.0, "question"),
        ("I want the {object} in the {bucket}", 1.0, "plain"),
        ("the {object} {go} in the {bucket}", 1.0, "plain"),
        ("the {object} {belong} in the {bucket}", 1.0, "plain"),
        ("of the {count} {containers}, put the {object} in the {color} one", 1.0, "plain"),
    ),
    please=0.25,
    please_first=0.6,
    full_stop=0.5,
)


def container_noun(rng: np.random.Generator, color: str) -> str:
    """Return the word for the *color* container: its real word (:data:`REAL_CONTAINER_NOUNS`) with
    :data:`REAL_NOUN_PROBABILITY`, else another of bin / bucket / box."""
    real = REAL_CONTAINER_NOUNS.get(color)
    if real is not None and rng.random() < REAL_NOUN_PROBABILITY:
        return real
    options = [noun for noun in CONTAINER_NOUNS if noun != real]
    return options[int(rng.integers(len(options)))]


def word_kind_instruction(
    rng: np.random.Generator, kind: str, count: int, color: str, *, buckets: int = 4, phrases: KindPhrases = KIND_PHRASES
) -> tuple[str, str]:
    """Word "put every *kind* on the table in the *color* container" (``count`` objects) at random.

    Returns:
        ``(sentence, canonical sentence)``.
    """
    plural = count > 1 and rng.random() < phrases.plural
    singular_names, plural_names = KIND_NAMES[kind]
    noun = container_noun(rng, color)
    canonical = f"put the {(plural_names if plural else singular_names)[0][0]} in the {color} {REAL_CONTAINER_NOUNS.get(color, 'bucket')}"
    name = _choose(rng, plural_names if plural else singular_names)
    if rng.random() < phrases.real_form:
        text = f"put the {name} {'inside' if rng.random() < 0.5 else 'in'} the {color} {noun}"
        if rng.random() >= phrases.real_form_lower:
            text = text[0].upper() + text[1:]
        if rng.random() < phrases.real_form_full_stop:
            text += "."
        return text, canonical
    index = int(rng.choice(len(phrases.templates), p=_normalized([w for _, w, _ in phrases.templates])))
    template, _, form = phrases.templates[index]
    text = template.format(
        object=name,
        bucket=f"{color} {noun}",
        containers=CONTAINER_PLURALS[noun],
        color=color,
        count=NUMBER_WORDS.get(buckets, str(buckets)),
        it="them" if plural else "it",
        go="go" if plural else "goes",
        belong="belong" if plural else "belongs",
    )
    if form != "plain" and rng.random() < phrases.please:
        if form == "question":
            text = text.replace(" you ", " you please ", 1)
        elif rng.random() < phrases.please_first:
            text = f"please {text}"
        else:
            text = f"{text}, please"
    if form == "question":
        text += "?"
    elif rng.random() < phrases.full_stop:
        text += "."
    return text[0].upper() + text[1:], canonical


def draw_kind_pair(
    rng: np.random.Generator, allowed: tuple[Pair, ...], layout: TaskLayout, kinds: Iterable[str] | None = None
) -> Pair:
    """Draw a (kind, bucket colour) pair: from :attr:`TaskLayout.real_pairs` with :attr:`TaskLayout.real_pair_probability`,
    else from the other *allowed* pairs; only kinds in *kinds* (those on the table) when given.

    Raises:
        ValueError: If no allowed pair has a kind in *kinds*.
    """
    pairs = tuple(p for p in allowed if kinds is None or p[0] in set(kinds))
    if not pairs:
        msg = "no allowed (kind, bucket) pair for the kinds on the table"
        raise ValueError(msg)
    real = tuple(p for p in pairs if p in layout.real_pairs)
    other = tuple(p for p in pairs if p not in layout.real_pairs)
    pick = real if real and (not other or rng.random() < layout.real_pair_probability) else other
    return pick[int(rng.integers(len(pick)))]


def _normalized(weights: list[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    return values / values.sum()


@dataclass(frozen=True)
class Instruction:
    """One episode's instruction: put ``object`` in the ``bucket`` bucket.

    Two instructions are equal when they name the same object and bucket, however they are worded.
    """

    object: str
    """One of :data:`OBJECTS`."""
    bucket: str
    """One of :data:`BUCKET_COLORS`."""
    wording: str | None = field(default=None, compare=False)
    """The sentence this episode uses; ``None`` for the canonical one."""
    label: str | None = field(default=None, compare=False)
    """How the canonical sentence names the object ("red cube"); ``None`` for :data:`OBJECT_LABELS`."""
    bucket_label: str | None = field(default=None, compare=False)
    """The colour the bucket shows this episode when a reset recoloured it (:func:`randomize_buckets`); ``None`` when
    it shows :attr:`bucket`, its body's colour."""
    kind: str | None = None
    """A kind-level instruction's real kind (:data:`bucket_sort_objects.REAL_KINDS`): it names the kind, and
    :attr:`object` is just the target picked first; ``None`` for an instruction naming one object."""
    targets: tuple[str, ...] = ()
    """Every object the instruction asks for: all of :attr:`kind` on the table at the episode's start. Empty means
    (:attr:`object`,)."""
    canonical: str | None = field(default=None, compare=False)
    """A kind-level instruction's canonical sentence ("put the resistors in the red bin")."""

    @property
    def all_targets(self) -> tuple[str, ...]:
        """The objects that must end in the bucket: :attr:`targets`, else :attr:`object` alone."""
        return self.targets or (self.object,)

    @property
    def shown_bucket(self) -> str:
        """The bucket's colour as the cameras show it and the sentence names it."""
        return self.bucket_label or self.bucket

    @property
    def canonical_text(self) -> str:
        """The instruction in its fixed form, e.g. ``put the CPU in the red bucket``."""
        if self.canonical is not None:
            return self.canonical
        if self.label is not None:
            return f"put the {self.label} in the {self.shown_bucket} bucket"
        return canonical_text(self.object, self.shown_bucket)

    @property
    def text(self) -> str:
        """The instruction as given to the robot: :attr:`wording`, else :attr:`canonical_text`."""
        return self.wording if self.wording is not None else self.canonical_text

    def as_dict(self) -> dict[str, str]:
        """Return the instruction for JSON: ``object``, ``bucket``, ``text`` and ``canonical_text``."""
        out = {
            "object": self.object,
            "bucket": self.bucket,
            "bucket_shown": self.shown_bucket,
            "text": self.text,
            "canonical_text": self.canonical_text,
        }
        if self.kind is not None:
            out["kind"] = self.kind
            out["targets"] = list(self.targets)
        return out


def check_pairs(
    pairs: Iterable[Pair] | None, colors: tuple[str, ...] = BUCKET_COLORS, objects: tuple[str, ...] = OBJECTS
) -> tuple[Pair, ...]:
    """Validate a set of allowed (object, bucket colour) pairs.

    Args:
        pairs: The allowed pairs; ``None`` allows every pair of :data:`OBJECTS` and *colors*.
        colors: The scene's bucket colours (:func:`bucket_colors`).

    Returns:
        The pairs without repeats, in :data:`ALL_PAIRS` order.

    Raises:
        ValueError: If a pair names an unknown object or colour, or no pair is left.
    """
    every = (
        ALL_PAIRS
        if colors == BUCKET_COLORS and objects == OBJECTS
        else tuple((obj, color) for obj in objects for color in colors)
    )
    if pairs is None:
        return every
    wanted = {(str(obj), str(color)) for obj, color in pairs}
    unknown = sorted(wanted - set(every))
    if unknown:
        msg = f"Unknown (object, bucket) pairs {unknown}; objects are {objects}, buckets {colors}"
        raise ValueError(msg)
    if not wanted:
        msg = "At least one (object, bucket) pair must be allowed"
        raise ValueError(msg)
    return tuple(pair for pair in every if pair in wanted)


def sample_instruction(
    rng: np.random.Generator,
    allowed_pairs: Iterable[Pair] | None = None,
    *,
    vary_text: bool | None = None,
    colors: tuple[str, ...] = BUCKET_COLORS,
    objects: tuple[str, ...] = OBJECTS,
    candidates: tuple[str, ...] | None = None,
    labels: Mapping[str, tuple[str, Weighted]] | None = None,
    bucket_names: Mapping[str, str] | None = None,
) -> Instruction:
    """Draw an instruction uniformly from the allowed (object, bucket colour) pairs, then word it.

    Args:
        rng: The generator.
        allowed_pairs: The pairs to draw from (:func:`check_pairs`); ``None`` draws from all nine.
        vary_text: Word it at random (:func:`word_instruction`) or use the canonical sentence;
            ``None`` follows :data:`VARY_INSTRUCTION_TEXT`.
        colors: The scene's bucket colours (:func:`bucket_colors`).
        objects: The scene's objects (:func:`scene_objects`).
        candidates: The objects this episode may name (placed and graspable); ``None`` for all of *objects*.
        labels: For a pool scene, each object's canonical label and weighted names (:func:`object_labels`).
        bucket_names: The colour each bucket body shows this episode, when recoloured (:func:`randomize_buckets`).

    Returns:
        The instruction.
    """
    pairs = check_pairs(allowed_pairs, colors, objects)
    if candidates is not None:
        narrowed = tuple(pair for pair in pairs if pair[0] in candidates)
        pairs = narrowed or pairs
    obj, color = pairs[int(rng.integers(len(pairs)))]
    vary = VARY_INSTRUCTION_TEXT if vary_text is None else vary_text
    label, names = labels[obj] if labels is not None and obj in labels else (None, None)
    shown = (bucket_names or {}).get(color, color)
    wording = word_instruction(rng, obj, shown, count=len(colors), names=names) if vary else None
    return Instruction(object=obj, bucket=color, wording=wording, label=label, bucket_label=shown if shown != color else None)


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s_ = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


@dataclass(frozen=True)
class TaskState:
    """Where the objects are relative to the buckets, and whether the instruction is done."""

    in_bucket: dict[str, str | None]
    """The bucket colour each object's centre is inside, or ``None``."""
    resting: dict[str, bool]
    """Whether each object is still and out of the robot's grasp."""
    success: bool
    """The named object rests in the named bucket, and no other object is in a bucket."""
    buckets: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    """Each bucket's pose in the world frame: ``position`` (metres, the centre of its footprint)
    and ``wxyz``."""

    def as_dict(self) -> dict[str, object]:
        """Return the state for JSON."""
        return {
            "in_bucket": dict(self.in_bucket),
            "resting": dict(self.resting),
            "success": self.success,
            "buckets": {color: dict(pose) for color, pose in self.buckets.items()},
        }


class BucketSortTask:
    """The scene's reset and success test, with the current layout and instruction.

    Call it as the scene's reset callback, ``task(model, data, rng)``: it puts the arm at home,
    lays the buckets and objects out and samples the instruction. The allowed pairs and the
    wording can be changed at any time and apply from the next reset.
    """

    def __init__(
        self,
        *,
        allowed_pairs: Iterable[Pair] | None = None,
        place_home: Callable[[object, object], None] | None = None,
        robot_roots: tuple[int, ...] = (),
        vary_text: bool | None = None,
        layout: TaskLayout = DEFAULT_LAYOUT,
    ) -> None:
        """Create the task; nothing is laid out until the first reset.

        Args:
            allowed_pairs: The (object, bucket colour) pairs instructions are drawn from
                (:func:`check_pairs`); ``None`` allows all nine.
            place_home: Puts the arm at its home pose at rest; ``None`` leaves the arm where it is.
            robot_roots: Root body of every robot: objects touching them are held, not resting.
            vary_text: Word instructions at random, or always use the canonical sentence;
                ``None`` follows :data:`VARY_INSTRUCTION_TEXT` at each reset.
            layout: Where the robot's buckets and objects may stand.
        """
        self._requested = None if allowed_pairs is None else tuple(allowed_pairs)
        self.colors: tuple[str, ...] = BUCKET_COLORS
        """The scene's bucket colours (:func:`bucket_colors`), read at each reset."""
        self._kind_mode = layout.kind_counts is not None
        self._allowed = check_pairs(self._requested, self.colors, self._pair_objects(OBJECTS)) if not self._kind_mode else ()
        self._place_home = place_home
        self._robot_roots = tuple(robot_roots)
        self._vary_text = vary_text
        self.area = layout
        """Where the robot's buckets and objects may stand (:class:`TaskLayout`)."""
        self._footprints: tuple[object, dict[str, ObjectFootprint]] | None = None
        self.instruction: Instruction | None = None
        """The current instruction; ``None`` before the first reset."""
        self.layout: Layout | None = None
        """Where the last reset put the buckets and objects; ``None`` before the first reset."""
        self.slot_colors: tuple[str, ...] = self.colors
        """The bucket colours left to right (by decreasing y) as the last reset put them."""
        self.reset_count = 0
        """How many resets have run: a new value means a new layout and instruction."""
        self.objects: tuple[str, ...] = OBJECTS
        """The scene's objects (:func:`scene_objects`), read at each reset; a pool scene has more than it places."""
        self.labels: dict[str, tuple[str, Weighted]] = {}
        """For a pool scene, each placed object's canonical label and names this episode (:func:`object_labels`)."""
        self.start_in_bucket: dict[str, str | None] = {}
        """Where each object was at the episode's start: success needs every other object to stay so."""
        self.sequence_index = 0
        """How many episodes in a row have sorted from the same table (0 for a fresh layout)."""
        self.look: dict[str, object] = {}
        self.bucket_names: dict[str, str] = {}
        """The colour each bucket body shows this episode (:func:`randomize_buckets`)."""
        """The episode's table look and exposure (:func:`randomize_look`); empty for a scene without them."""
        self._continue_next = False
        self._held_stratum: tuple[bool, str, int] | None = None
        self._failed = False
        self.stratum_retry = False
        """Whether the current fresh layout re-drew the stratum of a failed episode (:attr:`TaskLayout.stratified_retry`)."""
        self.required: str | None = None
        """An object every reset lays out and names (a reviewer's or a test's pair); ``None`` draws freely."""

    @property
    def allowed_pairs(self) -> tuple[Pair, ...]:
        """The (object, bucket colour) pairs the next instructions are drawn from."""
        return self._allowed

    def set_allowed_pairs(self, pairs: Iterable[Pair] | None) -> None:
        """Draw later instructions from *pairs* (:func:`check_pairs`); ``None`` allows all nine.

        The current instruction stays until the next reset.
        """
        self._requested = None if pairs is None else tuple(pairs)
        self._allowed = check_pairs(self._requested, self.colors, self._pair_objects(self.objects))

    def _pair_objects(self, objects: tuple[str, ...]) -> tuple[str, ...]:
        """What pairs name: the scene's objects, or with kind-level instructions the real kinds it holds."""
        if not self._kind_mode:
            return objects
        kinds = {real_kind(name) for name in objects} - {None}
        return tuple(kind for kind in REAL_KINDS if kind in kinds)

    @property
    def vary_text(self) -> bool:
        """Whether the next resets word the instruction at random."""
        return VARY_INSTRUCTION_TEXT if self._vary_text is None else self._vary_text

    def set_vary_text(self, vary: bool | None) -> None:  # noqa: FBT001
        """Word later instructions at random (``True``) or canonically (``False``).

        ``None`` follows :data:`VARY_INSTRUCTION_TEXT`.
        """
        self._vary_text = vary

    @property
    def canonical_text(self) -> str | None:
        """The current instruction in its fixed form (:attr:`Instruction.canonical_text`); ``None`` before a reset."""
        return self.instruction.canonical_text if self.instruction is not None else None

    def footprints(self, model: object) -> dict[str, ObjectFootprint]:
        """Return each object's :class:`ObjectFootprint` on *model*, computed once per model."""
        if self._footprints is None or self._footprints[0] is not model:
            self._footprints = (model, {name: object_footprint(model, name) for name in scene_objects(model)})
        return self._footprints[1]

    def note_failure(self) -> None:
        """The demonstrator failed this episode: with :attr:`TaskLayout.stratified_retry`, the next fresh layout keeps its
        stratum. A reset for any other reason (a success, a user's reset) draws a fresh one."""
        self._failed = True

    def offer_continue(self) -> None:
        """Let the next reset keep the table (after a success): with :attr:`TaskLayout.sequence_probability`, it names
        another object still on the table instead of laying out a new one (a sorting sequence)."""
        self._continue_next = True

    def _on_table(self, model: object, data: object) -> tuple[str, ...]:
        """Return the placed objects resting on the table in the object area, out of every bucket."""
        layout = self.layout
        if layout is None:
            return ()
        area = _current(self.area)
        names = []
        for name in layout.objects:
            position = data.xpos[_body(model, name)]
            if (
                bucket_containing(model, data, position) is None
                and area.object_x[0] - 0.05 <= position[0] <= area.object_x[1] + 0.03
                and area.object_y[0] - 0.03 <= position[1] <= area.object_y[1] + 0.03
                and position[2] < 0.05  # noqa: PLR2004 - lying on the table, not on top of a bucket
            ):
                names.append(name)
        return tuple(names)

    def _continue(self, model: object, data: object, rng: np.random.Generator) -> bool:
        """Start the next instruction on the table as it is; ``False`` when no object is left to sort."""
        import mujoco  # noqa: PLC0415

        mujoco.mj_forward(model, data)
        remaining = self._on_table(model, data)
        if not remaining or self.layout is None:
            return False
        if self._place_home is not None:
            self._place_home(model, data)
        self.start_in_bucket = {
            name: bucket_containing(model, data, data.xpos[_body(model, name)]) for name in self.objects
        }
        self.sequence_index += 1
        if self._kind_mode:
            kinds = {real_kind(name) for name in remaining} - {None}
            if not any(pair[0] in kinds for pair in self._allowed):
                return False
            pair = draw_kind_pair(rng, self._allowed, self.area, kinds)
            self.instruction = self._kind_instruction(rng, pair, remaining)
        else:
            self.instruction = sample_instruction(
                rng,
                self._allowed,
                vary_text=self.vary_text,
                colors=self.colors,
                objects=self.objects,
                candidates=remaining,
                labels=self.labels or None,
                bucket_names=self.bucket_names,
            )
        self.reset_count += 1
        mujoco.mj_forward(model, data)
        logger.info(
            "Bucket sort: {!r} ({}); sequence step {} with {} objects left",
            self.instruction.text,
            self.instruction.canonical_text,
            self.sequence_index,
            len(remaining),
        )
        return True

    def __call__(self, model: object, data: object, rng: np.random.Generator) -> None:
        """Reset: the arm at home, a new layout (:func:`sample_layout`) and a new instruction; then
        :attr:`episode_seed`, the last draw, so the scene's own draws are those of before (r16)."""
        self._reset_scene(model, data, rng)
        # r16: no solver warm start carried over from the previous episode (a dropped object's path depended on what
        # ran before it: a batch's 7418 penetrated 7.3 mm, alone 0.6 mm)
        data.qacc_warmstart[:] = 0.0
        data.qvel[:] = 0.0   # every body placed at rest (parked objects too: the solver couples them all)
        self.episode_seed = int(rng.integers(2**63))

    episode_seed: int | None = None
    """The demonstrator's seed for the episode this reset starts (r16): drawn from the reset's RNG, so a reset with a
    fixed seed repeats the demonstrator's choices too, alone or after any other episodes (the demonstrator's own
    stream, made once per scene, did not repeat; real2sim r15 notes/R6_LOCKSTEP_DESIGN.md)."""

    def _reset_scene(self, model: object, data: object, rng: np.random.Generator) -> None:
        import mujoco  # noqa: PLC0415

        keep, self._continue_next = self._continue_next, False
        if keep and self.area.sequence_probability > 0.0 and rng.random() < self.area.sequence_probability:
            if self._continue(model, data, rng):
                return
        if self._place_home is not None:
            self._place_home(model, data)
        footprints = self.footprints(model)
        colors = bucket_colors(model)
        objects = scene_objects(model)
        if colors != self.colors or objects != self.objects or not self._allowed:
            self.colors, self.objects = colors, objects
            self._allowed = check_pairs(self._requested, colors, self._pair_objects(objects))
        if self._kind_mode:
            self._kind_reset(model, data, rng, footprints)
            return
        layout = sample_layout(rng, footprints, self.area, colors, objects, required=self.required)
        if layout.parked or any(kind_of(name) for name in layout.objects):
            object_colors, self.labels = recolor_pool(model, rng, tuple(layout.objects))
            layout = replace(layout, object_colors=object_colors)
            park_objects(model, data, layout.parked, layout.objects)
        else:
            self.labels = {}
        if self.area.fill is not None:
            set_bucket_fill(model, {color: float(rng.uniform(*self.area.fill)) for color in colors})
        for color, pose in layout.buckets.items():
            _set_free_pose(model, data, f"bucket_{color}:joint", (pose.x, pose.y, 0.0), pose.yaw)
        for name, pose in layout.objects.items():
            _set_free_pose(
                model, data, f"{name}:joint", (pose.x, pose.y, footprints[name].half_height + DROP_HEIGHT), pose.yaw
            )
        self.look = randomize_look(model, rng, self.area)
        self.bucket_names = randomize_buckets(model, rng, self.area)
        self.layout = layout
        self.slot_colors = layout.colors_left_to_right
        self.instruction = sample_instruction(
            rng,
            self._allowed,
            vary_text=self.vary_text,
            colors=self.colors,
            objects=self.objects,
            candidates=(self.required,) if self.required in layout.objects else (layout.targets or tuple(layout.objects)),
            labels=self.labels or None,
            bucket_names=self.bucket_names,
        )
        self.start_in_bucket = dict.fromkeys(self.objects)
        self.sequence_index = 0
        self.reset_count += 1
        mujoco.mj_forward(model, data)
        logger.info(
            "Bucket sort: {!r} ({}); buckets left to right {}",
            self.instruction.text,
            self.instruction.canonical_text,
            ", ".join(self.slot_colors),
        )

    # -- kind-level episodes (TaskLayout.kind_counts) ------------------------------------------------------------

    def _upright_footprint(self, model: object, name: str) -> ObjectFootprint:
        cache = self.__dict__.setdefault("_upright_fp", {})
        if (id(model), name) not in cache:
            cache[(id(model), name)] = object_footprint_upright(model, name)
        return cache[(id(model), name)]

    def _kind_bodies(self) -> dict[str, list[str]]:
        bodies: dict[str, list[str]] = {}
        for name in self.objects:
            kind = real_kind(name)
            if kind is not None and kind_of(name) is not None:   # the pool bodies obj_<kind>_<k>
                bodies.setdefault(kind, []).append(name)
        return bodies

    def _draw_kind_objects(
        self, rng: np.random.Generator, kind: str, most: int | None = None, named: int | None = None, counts: tuple | None = None
    ) -> tuple[str, ...]:
        """The objects of a fresh kind-level layout: counts per real kind (:attr:`TaskLayout.kind_counts`, at least one
        of *kind*) and the unnamed distractors (:attr:`TaskLayout.distractors`); with *most*, drawn again until no more
        than that many (a scattered table, :data:`SCATTER_MOST`)."""
        for _ in range(200):
            chosen = self._draw_kind_objects_once(rng, kind, counts)
            if (most is None or len(chosen) <= most) and (named is None or sum(real_kind(n) == kind for n in chosen) == named):
                return chosen
        return chosen

    def _draw_kind_objects_once(self, rng: np.random.Generator, kind: str, counts: tuple | None = None) -> tuple[str, ...]:
        bodies = self._kind_bodies()
        chosen: list[str] = []
        for name, low, high in counts or self.area.kind_counts or ():
            have = bodies.get(name, [])
            count = int(rng.integers(low, high + 1))
            count = min(max(count, 1 if name == kind else 0), len(have))
            chosen += [have[int(i)] for i in rng.choice(len(have), size=count, replace=False)] if count else []
        generic = [name for name in self.objects if kind_of(name) in KINDS]
        low, high = self.area.distractors
        count = min(int(rng.integers(low, high + 1)), len(generic))
        chosen += [generic[int(i)] for i in rng.choice(len(generic), size=count, replace=False)] if count else []
        return tuple(name for name in self.objects if name in chosen)

    def _kind_instruction(
        self, rng: np.random.Generator, pair: Pair, on_table: tuple[str, ...], first: str | None = None
    ) -> Instruction:
        kind, color = pair
        targets = tuple(name for name in on_table if real_kind(name) == kind)
        shown = (self.bucket_names or {}).get(color, color)
        text, canonical = word_kind_instruction(rng, kind, len(targets), shown, buckets=len(self.colors))
        if not self.vary_text:
            text = canonical
        return Instruction(
            object=first if first in targets else targets[0],
            bucket=color,
            wording=text,
            bucket_label=shown if shown != color else None,
            kind=kind,
            targets=targets,
            canonical=canonical,
        )

    def _kind_reset(self, model: object, data: object, rng: np.random.Generator, footprints: dict[str, ObjectFootprint]) -> None:
        """A fresh kind-level episode: the pair, the objects (real kinds and distractors), their layout, the look."""
        import mujoco  # noqa: PLC0415

        on_scene = self._pair_objects(self.objects)
        required_kind = real_kind(self.required) if self.required else None
        held = self._held_stratum if (self.area.stratified_retry and self._failed) else None
        self._failed = False
        if held is not None:   # the previous episode failed: the same arrangement, kind and number of targets
            pile, kind, named = held
            pair = draw_kind_pair(rng, self._allowed, self.area, (kind,))
        else:
            pair = draw_kind_pair(rng, self._allowed, self.area, (required_kind,) if required_kind else on_scene)
            pile = self.area.pile_probability > 0.0 and rng.random() < self.area.pile_probability
            named = None
        scatter_counts = self.area.scatter_kind_counts
        most = None if pile else (SCATTER_MOST_REAL_LIKE if scatter_counts else SCATTER_MOST)
        chosen = self._draw_kind_objects(rng, pair[0], most, named, None if pile else scatter_counts)
        named = frozenset(n for n in chosen if real_kind(n) == pair[0])
        p_up = dict(self.area.upright)
        upright = frozenset(n for n in chosen if real_kind(n) in p_up and rng.random() < p_up[real_kind(n)])
        footprints = {**footprints, **{n: self._upright_footprint(model, n) for n in upright}}
        layout = sample_layout(rng, footprints, self.area, self.colors, self.objects, chosen=chosen, pile=pile, protect=named, upright=upright)
        object_colors, self.labels = recolor_pool(model, rng, tuple(layout.objects))
        layout = replace(layout, object_colors=object_colors)
        park_objects(model, data, layout.parked, layout.objects)
        if self.area.fill is not None:
            set_bucket_fill(model, {color: float(rng.uniform(*self.area.fill)) for color in self.colors})
        for color, pose in layout.buckets.items():
            _set_free_pose(model, data, f"bucket_{color}:joint", (pose.x, pose.y, 0.0), pose.yaw)
        for name, pose in layout.objects.items():
            z = footprints[name].half_height + DROP_HEIGHT + layout.heights.get(name, 0.0)
            if name in layout.upright:   # standing on its end, turned yaw about the vertical
                q = np.zeros(4); import mujoco as _mj  # noqa: PLC0415
                _mj.mju_mat2Quat(q, (_yaw_matrix(pose.yaw) @ upright_rotation(model, name)).ravel())
                _set_free_quat(model, data, f"{name}:joint", (pose.x, pose.y, z), q)
            else:
                _set_free_pose(model, data, f"{name}:joint", (pose.x, pose.y, z), pose.yaw)
        self.look = randomize_look(model, rng, self.area)
        self.bucket_names = randomize_buckets(model, rng, self.area)
        self.layout = layout
        self.slot_colors = layout.colors_left_to_right
        kinds_here = {real_kind(name) for name in layout.objects} - {None}
        if pair[0] not in kinds_here:   # a fallback layout without the drawn kind
            pair = draw_kind_pair(rng, self._allowed, self.area, kinds_here)
        # the target picked first: one on the table (not stacked), a pile target when there is one
        candidates = [n for n in layout.objects if real_kind(n) == pair[0] and n not in layout.heights]
        preferred = [n for n in candidates if not layout.targets or n in layout.targets]
        first = (preferred or candidates or [None])[0]
        self.instruction = self._kind_instruction(rng, pair, tuple(layout.objects), first)
        self._held_stratum = (layout.pile, pair[0], len(self.instruction.targets))
        self.stratum_retry = held is not None
        self.start_in_bucket = dict.fromkeys(self.objects)
        self.sequence_index = 0
        self.reset_count += 1
        mujoco.mj_forward(model, data)
        logger.info(
            "Bucket sort: {!r} ({}); {} targets; buckets left to right {}",
            self.instruction.text,
            self.instruction.canonical_text,
            len(self.instruction.targets),
            ", ".join(self.slot_colors),
        )

    def evaluate(self, model: object, data: object) -> TaskState:
        """Check where the objects are and whether the current instruction is done.

        Returns:
            The state; never a success before the first reset.
        """
        held = self._bodies_touching_robot(model, data)
        in_bucket: dict[str, str | None] = {}
        resting: dict[str, bool] = {}
        for name in self.objects:
            body = _body(model, name)
            dof = int(model.jnt_dofadr[_joint(model, f"{name}:joint")])
            in_bucket[name] = bucket_containing(model, data, data.xpos[body])
            speed = float(np.linalg.norm(data.qvel[dof : dof + 3]))
            resting[name] = speed <= SETTLE_SPEED and body not in held
        instruction = self.instruction
        targets = instruction.all_targets if instruction is not None else ()
        success = (
            instruction is not None
            and all(in_bucket[name] == instruction.bucket and resting[name] for name in targets)
            and all(
                in_bucket[name] == self.start_in_bucket.get(name)
                for name in self.objects
                if name not in targets
            )
        )
        buckets = {
            color: {
                "position": [float(v) for v in data.xpos[body]],
                "wxyz": [float(v) for v in data.xquat[body]],
            }
            for color in self.colors
            for body in (_body(model, f"bucket_{color}"),)
        }
        return TaskState(in_bucket=in_bucket, resting=resting, success=success, buckets=buckets)

    def describe(self) -> dict[str, object]:
        """Return the instruction, the buckets left to right as laid out, and the allowed pairs, for JSON.

        ``slots`` lists the buckets by decreasing y at the last reset (``left``, ``centre``,
        ``right`` as seen from behind the robot) with that reset's centre and yaw; ``buckets`` in
        the episode status has their live poses.
        """
        layout = self.layout
        return {
            "instruction": self.instruction.as_dict() if self.instruction is not None else None,
            "slots": [
                {"slot": slot, "color": color, **(layout.buckets[color].as_dict() if layout is not None else {})}
                for slot, color in zip(slot_names(len(self.slot_colors)), self.slot_colors, strict=True)
            ],
            "allowed_pairs": [list(pair) for pair in self._allowed],
            "vary_text": self.vary_text,
            "reset_count": self.reset_count,
            **(
                {
                    "objects": list(layout.objects),
                    "object_labels": {name: label for name, (label, _) in self.labels.items()},
                    "pile": layout.pile,
                    "sequence_index": self.sequence_index,
                    "stratum": list(self._held_stratum) if self._held_stratum else None,
                    "stratum_retry": self.stratum_retry,
                    "look": dict(self.look),
                    "bucket_shown": dict(self.bucket_names),
                }
                if layout is not None and self.labels
                else {}
            ),
        }

    def _bodies_touching_robot(self, model: object, data: object) -> set[int]:
        ncon = int(data.ncon)
        if ncon == 0 or not self._robot_roots:
            return set()
        robot = np.isin(np.asarray(model.body_rootid), self._robot_roots)
        bodies = np.asarray(model.geom_bodyid)
        b1 = bodies[np.asarray(data.contact.geom1[:ncon])]
        b2 = bodies[np.asarray(data.contact.geom2[:ncon])]
        r1, r2 = robot[b1], robot[b2]
        return {int(b) for b in np.concatenate([b2[r1 & ~r2], b1[r2 & ~r1]])}


def _set_free_quat(model: object, data: object, joint_name: str, position: tuple[float, float, float], quat: np.ndarray) -> None:
    """Put a free joint at *position* with orientation *quat* (w, x, y, z), at rest."""
    joint = _joint(model, joint_name)
    adr, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    data.qpos[adr : adr + 3] = position
    data.qpos[adr + 3 : adr + 7] = quat
    data.qvel[dof : dof + 6] = 0.0


def _set_free_pose(
    model: object, data: object, joint_name: str, position: tuple[float, float, float], yaw: float
) -> None:
    """Put a free joint at *position*, turned *yaw* about the vertical, at rest."""
    joint = _joint(model, joint_name)
    adr, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    data.qpos[adr : adr + 3] = position
    data.qpos[adr + 3 : adr + 7] = _yaw_quat(yaw)
    data.qvel[dof : dof + 6] = 0.0


_HIDDEN_GROUP = 5
"""The geom group hidden contents move to: no renderer draws it by default."""


def set_bucket_fill(model: object, fractions: Mapping[str, float]) -> dict[str, int]:
    """Show the lowest *fraction* of each bucket's contents and hide the rest (drawn nowhere, no contacts).

    The first call remembers each content geom's group and contact bits; later calls restore them for
    the items shown.

    Args:
        model: The scene's model.
        fractions: The fraction (0 to 1) of each colour's items to show; colours left out are unchanged.

    Returns:
        How many items each named colour shows.
    """
    import mujoco  # noqa: PLC0415

    cache = _CONTENTS.get(id(model))
    if cache is None:
        cache = {}
        for geom in range(int(model.ngeom)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if not name.startswith("bucket_") or "_item" not in name:
                continue
            color, rest = name[len("bucket_") :].split("_item", 1)
            item = int(rest.split("_", 1)[0])
            original = (int(model.geom_group[geom]), int(model.geom_contype[geom]), int(model.geom_conaffinity[geom]))
            cache.setdefault(color, {}).setdefault(item, []).append((geom, original))
        _CONTENTS[id(model)] = cache
    shown: dict[str, int] = {}
    for color, fraction in fractions.items():
        items = cache.get(color, {})
        count = round(float(np.clip(fraction, 0.0, 1.0)) * len(items))
        for item, geoms in items.items():
            for geom, (group, contype, conaffinity) in geoms:
                visible = item < count
                model.geom_group[geom] = group if visible else _HIDDEN_GROUP
                model.geom_contype[geom] = contype if visible else 0
                model.geom_conaffinity[geom] = conaffinity if visible else 0
        shown[color] = count
    return shown


_CONTENTS: dict[int, dict[str, dict[int, list[tuple[int, tuple[int, int, int]]]]]] = {}
"""Per model, each colour's content geoms by item, with their original group and contact bits."""


_SCENE_OBJECTS: dict[int, tuple[object, tuple[str, ...]]] = {}


def scene_objects(model: object) -> tuple[str, ...]:
    """Return the objects of *model*: the v1 objects it has (:data:`OBJECTS` order), then its pool bodies.

    A pool body is ``obj_<kind>_<k>`` with the free joint ``<name>:joint`` (:mod:`.bucket_sort_objects`).

    Returns:
        The body names.
    """
    import mujoco  # noqa: PLC0415

    cached = _SCENE_OBJECTS.get(id(model))
    if cached is not None and cached[0] is model:
        return cached[1]
    body = mujoco.mjtObj.mjOBJ_BODY
    names = [name for name in OBJECTS if mujoco.mj_name2id(model, body, name) >= 0]
    for index in range(int(model.nbody)):
        name = mujoco.mj_id2name(model, body, index) or ""
        if kind_of(name) is not None:
            names.append(name)
    result = tuple(names)
    _SCENE_OBJECTS[id(model)] = (model, result)
    return result


def object_labels(
    names: Iterable[str], colors: Mapping[str, str]
) -> dict[str, tuple[str, Weighted]]:
    """Return each object's canonical label and weighted names: "red cube" for a pool object, the v1 names else.

    Returns:
        ``{name: (label, ((name, weight), ...))}``.
    """
    labels: dict[str, tuple[str, Weighted]] = {}
    for name in names:
        kind = kind_of(name)
        real = real_kind(name)
        if kind is None or real is not None:
            key = real or name
            labels[name] = (OBJECT_LABELS[key], PHRASES.object_names[key])
            continue
        color = colors[name]
        nouns = KINDS[kind].nouns
        labels[name] = (f"{color} {nouns[0][0]}", tuple((f"{color} {noun}", weight) for noun, weight in nouns))
    return labels


def recolor_pool(
    model: object, rng: np.random.Generator, placed: tuple[str, ...]
) -> tuple[dict[str, str], dict[str, tuple[str, Weighted]]]:
    """Give each placed pool object a colour (its material ``<name>``), no two of one kind alike.

    Returns:
        The colour names by object and :func:`object_labels` of the placed objects.
    """
    import mujoco  # noqa: PLC0415

    palette = tuple(OBJECT_COLORS)
    taken: dict[str, set[str]] = {}
    colors: dict[str, str] = {}
    for name in placed:
        kind = kind_of(name)
        if kind is None or kind not in KINDS or not KINDS[kind].colorable:
            continue
        free = [color for color in palette if color not in taken.setdefault(kind, set())]
        color = free[int(rng.integers(len(free)))]
        taken[kind].add(color)
        colors[name] = color
        material = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name))
        if material >= 0:
            jitter = rng.uniform(0.94, 1.04, size=3)
            model.mat_rgba[material, :3] = np.clip(np.asarray(OBJECT_COLORS[color]) * jitter, 0.0, 1.0)
    return colors, object_labels(placed, colors)


_POOL_CONTACTS: dict[int, tuple[object, dict[int, tuple[int, int]]]] = {}


_PROXY_NAME = __import__("re").compile(r"_plan\d+$")


def mujoco_name(model: object, geom: int) -> str:
    import mujoco  # noqa: PLC0415

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""


def _pool_contacts(model: object, name: str, *, parked: bool) -> None:
    """Give object *name*'s collision geoms the shelf's contact bits (parked) or their own (placed).

    Parked below the floor, an object must not touch the floor plane, which would push it up through
    the table; it touches only the shelf (:data:`bucket_sort_objects.PARK_CONTYPE`).
    """
    entry = _POOL_CONTACTS.get(id(model))
    if entry is None or entry[0] is not model:
        entry = (model, {})
        _POOL_CONTACTS[id(model)] = entry
    originals = entry[1]
    body = _body(model, name)
    geoms = [int(g) for g in np.flatnonzero(np.asarray(model.geom_bodyid) == body)]
    # r14: an object with planning proxies (``<name>_plan<i>``, hulls of its collision pieces) rests on the shelf on
    # them alone while parked (far fewer contacts than its many fine pieces); placed, its pieces collide again
    proxies = {g for g in geoms if _PROXY_NAME.search(mujoco_name(model, g))}
    for geom in geoms:
        if geom in proxies:
            model.geom_contype[geom] = model.geom_conaffinity[geom] = PARK_CONTYPE if parked else 0
            continue
        if geom not in originals:
            contype, conaffinity = int(model.geom_contype[geom]), int(model.geom_conaffinity[geom])
            if contype == 0 and conaffinity == 0:
                continue  # a visual geom
            originals[geom] = (contype, conaffinity)
        contype, conaffinity = ((0, 0) if proxies else (PARK_CONTYPE, PARK_CONTYPE)) if parked else originals[geom]
        model.geom_contype[geom] = contype
        model.geom_conaffinity[geom] = conaffinity


def park_objects(model: object, data: object, names: Iterable[str], placed: Iterable[str] = ()) -> None:
    """Put *names* on the hidden shelf below the floor (:data:`bucket_sort_objects.PARKING`), at rest.

    The *placed* pool objects get their own contacts back.
    """
    for name in placed:
        _pool_contacts(model, name, parked=False)
    for index, name in enumerate(names):
        _pool_contacts(model, name, parked=True)
        x = PARKING[0] + (index % 6) * PARK_SPACING
        y = PARKING[1] + (index // 6) * PARK_SPACING
        _set_free_pose(model, data, f"{name}:joint", (x, y, PARKING[2] + 0.05), 0.0)


def bucket_colors(model: object) -> tuple[str, ...]:
    """Return the colours of the buckets *model* has (bodies ``bucket_<colour>``), in :data:`KNOWN_BUCKET_COLORS` order.

    Returns:
        The colours.
    """
    import mujoco  # noqa: PLC0415

    body = mujoco.mjtObj.mjOBJ_BODY
    return tuple(color for color in KNOWN_BUCKET_COLORS if mujoco.mj_name2id(model, body, f"bucket_{color}") >= 0)


def slot_names(count: int) -> tuple[str, ...]:
    """Name *count* buckets left to right as seen from behind the robot: :data:`SLOT_NAMES` for three.

    Returns:
        The names.
    """
    return SLOT_NAMES if count == len(SLOT_NAMES) else tuple(f"slot {i + 1}" for i in range(count))


def bucket_containing(model: object, data: object, point: np.ndarray) -> str | None:
    """Return the colour of the bucket whose interior (:data:`BUCKET_INTERIOR`) holds *point*.

    The interior moves with each bucket's current pose (``data.xpos`` and ``data.xmat``, as of the
    last ``mj_forward`` or step), so it follows a bucket that was pushed or dragged.

    Returns:
        The colour, or ``None`` outside every bucket.
    """
    for color in bucket_colors(model):
        body = _body(model, f"bucket_{color}")
        local = data.xmat[body].reshape(3, 3).T @ (np.asarray(point) - data.xpos[body])
        if all(lo <= value <= hi for value, (lo, hi) in zip(local, BUCKET_INTERIOR, strict=True)):
            return color
    return None


def object_extent(model: object, name: str) -> tuple[float, float]:
    """Return an object's horizontal radius at any heading and its half height, lying as written.

    Both come from the bounding boxes of the object's collision geoms about its body origin.

    Returns:
        ``(radius, half_height)`` in metres.
    """
    import mujoco  # noqa: PLC0415

    body = _body(model, name)
    radius, half_height = 0.0, 0.0
    rotation = np.zeros(9)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    for geom in np.flatnonzero(np.asarray(model.geom_bodyid) == body):
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        points = (
            model.geom_pos[geom]
            + (model.geom_aabb[geom, :3] + corners * model.geom_aabb[geom, 3:]) @ rotation.reshape(3, 3).T
        )
        radius = max(radius, float(np.hypot(points[:, 0], points[:, 1]).max()))
        half_height = max(half_height, float(-points[:, 2].min()))
    return radius, half_height


def object_footprint(model: object, name: str) -> ObjectFootprint:
    """Return what the layout needs of an object (:class:`ObjectFootprint`), from its collision geoms.

    Returns:
        The footprint.
    """
    import mujoco  # noqa: PLC0415

    radius, half_height = object_extent(model, name)
    body = _body(model, name)
    rotation = np.zeros(9)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    half = np.zeros(2)
    for geom in np.flatnonzero(np.asarray(model.geom_bodyid) == body):
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        points = (
            model.geom_pos[geom]
            + (model.geom_aabb[geom, :3] + corners * model.geom_aabb[geom, 3:]) @ rotation.reshape(3, 3).T
        )
        half = np.maximum(half, np.abs(points[:, :2]).max(axis=0))
    closing_yaw = 0.0 if half[0] <= half[1] else float(np.pi / 2)
    return ObjectFootprint(radius=radius, half_height=half_height, closing_yaw=closing_yaw)


def _hull_corners(model: object, name: str) -> np.ndarray:
    """The corners of the object's collision geoms' boxes in its body frame."""
    import mujoco  # noqa: PLC0415

    body = _body(model, name); rotation = np.zeros(9); out = []
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    for geom in np.flatnonzero(np.asarray(model.geom_bodyid) == body):
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        out.append(model.geom_pos[geom] + (model.geom_aabb[geom, :3] + corners * model.geom_aabb[geom, 3:]) @ rotation.reshape(3, 3).T)
    return np.concatenate(out)


def upright_rotation(model: object, name: str) -> np.ndarray:
    """The body rotation that stands object *name* on its end: its longest horizontal body axis turned vertical."""
    points = _hull_corners(model, name); half = np.abs(points[:, :2]).max(axis=0)
    if half[1] >= half[0]:   # long along body y: a quarter turn about x takes y to z
        return np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    return np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])   # long along x: a quarter turn about y takes x to z


def object_footprint_upright(model: object, name: str) -> ObjectFootprint:
    """The :class:`ObjectFootprint` of object *name* standing on its end (:func:`upright_rotation`)."""
    points = _hull_corners(model, name) @ upright_rotation(model, name).T
    return ObjectFootprint(radius=float(np.hypot(points[:, 0], points[:, 1]).max()), half_height=float(-points[:, 2].min()), closing_yaw=0.0)


Phase = Literal["idle", "success_hold"]


class BucketSortEpisode:
    """Report the task in ``/health`` and the viewer, and start a new episode after a success.

    It has the controls of the other episode helpers (``active``, the success dwell). Once the
    instruction has held for the dwell, it resets the scene as ``POST /reset`` does.
    """

    def __init__(
        self,
        task: BucketSortTask,
        reset_scene: Callable[[], None],
        *,
        success_dwell_s: float = 5.0,
        active: bool = True,
    ) -> None:
        """Watch *task*; *reset_scene* is the owner's scene reset, which the automatic resets share.

        Going through the owner keeps a fixed seed repeating the layout and drops viewer drags,
        as a manual reset does.
        """
        self.task = task
        self._reset_scene = reset_scene
        self._dwell_s = float(success_dwell_s)
        self._active = active
        self._phase: Phase = "idle"
        self._success_since: float | None = None
        self._resets_suspended = False
        self._countdown_s: float | None = None
        self._episode_count = 0
        self._state: TaskState | None = None

    @property
    def active(self) -> bool:
        """Whether a success starts a new episode after the dwell."""
        return self._active

    def set_active(self, active: bool) -> None:  # noqa: FBT001
        """Pause or resume the automatic reset; pausing cancels a running countdown."""
        self._active = active
        if not active:
            self.notify_manual_reset()

    @property
    def dwell_s(self) -> float:
        """Seconds the instruction must stay done before the next episode starts."""
        return self._dwell_s

    def set_dwell(self, dwell_s: float) -> None:
        """Change the success dwell; a running countdown uses the new value."""
        self._dwell_s = float(dwell_s)

    def notify_manual_reset(self) -> None:
        """Clear the success countdown after an explicit reset."""
        self._phase = "idle"
        self._success_since = None
        self._countdown_s = None

    def status(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot for ``/health`` and the viewer."""
        state = self._state
        return {
            "enabled": True,
            "kind": "bucket_sort",
            "active": self._active,
            "phase": self._phase,
            "episode_count": self._episode_count,
            "success_dwell_s": self._dwell_s,
            "countdown_s": self._countdown_s,
            **self.task.describe(),
            "success": state.success if state is not None else False,
            "in_bucket": dict(state.in_bucket) if state is not None else dict.fromkeys(self.task.objects),
            "buckets": dict(state.buckets) if state is not None else {},
        }

    def suspend_resets(self, suspended: bool) -> None:  # noqa: FBT001
        """Keep evaluating the task but never reset the scene, while something else owns the episodes.

        Automatic Studio recording ends and resets each episode itself; :attr:`active` and the
        dwell stay as the user set them for afterwards.
        """
        if suspended and not self._resets_suspended:
            self.notify_manual_reset()  # drop a success countdown in progress
        self._resets_suspended = suspended

    def update(self, model: object, data: object) -> None:
        """Evaluate the task for one tick, and reset the scene once a success has held for the dwell."""
        state = self.task.evaluate(model, data)
        self._state = state
        if not self._active or self._resets_suspended:
            return
        if not state.success:
            if self._phase == "success_hold":
                logger.info("Bucket sort: the instruction is no longer met; countdown cancelled")
            self.notify_manual_reset()
            return
        now = float(data.time)
        if self._success_since is None:
            self._success_since = now
            self._phase = "success_hold"
            logger.info("Bucket sort: success; next episode in {:.1f} s", self._dwell_s)
        self._countdown_s = max(0.0, self._dwell_s - (now - self._success_since))
        if self._countdown_s > 0.0:
            return
        self.task.offer_continue()
        self._reset_scene()
        self._episode_count += 1
        self.notify_manual_reset()
        self._state = self.task.evaluate(model, data)


def _body(model: object, name: str) -> int:
    import mujoco  # noqa: PLC0415

    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if body < 0:
        msg = f"The bucket_sort model has no body {name!r}"
        raise ValueError(msg)
    return body


def _joint(model: object, name: str) -> int:
    import mujoco  # noqa: PLC0415

    joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if joint < 0:
        msg = f"The bucket_sort model has no joint {name!r}"
        raise ValueError(msg)
    return joint


__all__ = [
    "ALL_PAIRS",
    "BUCKET_COLORS",
    "BUCKET_GAP",
    "BUCKET_INTERIOR",
    "BUCKET_JOINTS",
    "BUCKET_SIZE",
    "BUCKET_X",
    "BUCKET_Y",
    "BUCKET_YAW",
    "BUCKET_YAW_RANGE",
    "DEFAULT_LAYOUT",
    "KNOWN_BUCKET_COLORS",
    "OBJECTS",
    "OBJECT_LABELS",
    "OBJECT_X",
    "OBJECT_Y",
    "PHRASES",
    "REACH_RADIUS",
    "VARY_INSTRUCTION_TEXT",
    "BucketSortEpisode",
    "BucketSortTask",
    "Instruction",
    "InstructionPhrases",
    "Layout",
    "ObjectFootprint",
    "Pose2D",
    "TaskLayout",
    "BUCKET_PALETTE",
    "LOOK_KNOBS",
    "TABLE_LOOKS",
    "TABLE_PATTERNS",
    "TaskState",
    "bucket_colors",
    "bucket_containing",
    "bucket_footprint",
    "bucket_problems",
    "canonical_text",
    "check_pairs",
    "object_extent",
    "object_footprint",
    "object_labels",
    "object_problems",
    "park_objects",
    "randomize_buckets",
    "randomize_look",
    "recolor_pool",
    "scene_objects",
    "sample_instruction",
    "sample_layout",
    "set_bucket_fill",
    "word_instruction",
]
