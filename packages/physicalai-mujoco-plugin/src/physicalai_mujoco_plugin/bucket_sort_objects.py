# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A pool of mixed graspable objects for the bucket-sort scenes (``bucket_sort_desk_v2``).

The v1 scenes have three fixed objects (a resistor, a 9 V battery and a CPU). A pool scene has more
bodies than an episode uses: each reset picks some of them, gives the plain shapes a colour from
:data:`OBJECT_COLORS`, lays them out (scattered or in a pile) and parks the rest out of sight on a
hidden shelf below the floor. Instructions name an object by its colour and shape ("the red cube")
or, for the real objects, by name ("the 9 V battery").

A pool body is named ``obj_<kind>_<k>`` (:data:`KINDS`), with the free joint ``<name>:joint`` and a
material of its own, ``<name>`` (recoloured per episode). Its collision and visual geoms are meshes
(inline convex hulls), so :func:`bucket_sort_demo.object_shape` measures them like the v1 objects.
:func:`pool_xml` writes the bodies for a scene XML.

Every kind is grasped from above, across its narrow horizontal side: the reBot's fingers open to
0.10 m, so every kind is at most 0.07 m across that side and at least 0.018 m tall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBJECT_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.80, 0.10, 0.09),
    "orange": (0.95, 0.45, 0.08),
    "yellow": (0.95, 0.80, 0.12),
    "green": (0.15, 0.58, 0.22),
    "blue": (0.12, 0.30, 0.82),
    "purple": (0.45, 0.20, 0.65),
    "pink": (0.95, 0.50, 0.66),
    "white": (0.90, 0.90, 0.88),
    "black": (0.08, 0.08, 0.09),
    "brown": (0.45, 0.28, 0.14),
}
"""Named colours a plain pool object can take (linear-ish sRGB the renderer uses as albedo)."""


@dataclass(frozen=True)
class ObjectKind:
    """One shape of the pool."""

    nouns: tuple[tuple[str, float], ...]
    """Names of the shape after the colour ("red **cube**"), weighted; the first is canonical."""
    colorable: bool = True
    """Whether a reset recolours it; a fixed object (the battery) has no colour word."""
    density: float = 400.0
    """Kilograms per cubic metre of its hull: light plastic or wood."""


KINDS: dict[str, ObjectKind] = {
    "cube": ObjectKind((("cube", 1.0),)),
    "brick": ObjectKind((("brick", 2.0), ("rectangle", 1.0))),
    "cylinder": ObjectKind((("cylinder", 2.0), ("round piece", 1.0))),
    "puck": ObjectKind((("disc", 2.0), ("puck", 1.0))),
    "hex": ObjectKind((("hexagon", 2.0), ("hex", 1.0))),
    "prism": ObjectKind((("triangle", 2.0), ("wedge", 1.0))),
    "bar": ObjectKind((("bar", 2.0), ("stick", 1.0))),
    "cross": ObjectKind((("cross", 2.0), ("plus", 1.0))),
    "arch": ObjectKind((("arch", 2.0), ("bridge", 1.0))),
}
"""The plain kinds, by the ``<kind>`` part of a body name. No noun shares a word with another kind's, and no
two objects on the table have the same colour and kind, so "the red cube" names one object."""


def _box(sx: float, sy: float, sz: float, cx: float = 0.0, cy: float = 0.0, cz: float = 0.0) -> np.ndarray:
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return corners * np.array([sx, sy, sz]) / 2.0 + np.array([cx, cy, cz])


def _prism(radius: float, height: float, sides: int, phase: float = 0.0, stretch: float = 1.0) -> np.ndarray:
    angles = phase + np.arange(sides) * 2.0 * np.pi / sides
    ring = np.stack([radius * stretch * np.cos(angles), radius * np.sin(angles)], axis=1)
    return np.concatenate([
        np.column_stack([ring, np.full(sides, -height / 2.0)]),
        np.column_stack([ring, np.full(sides, height / 2.0)]),
    ])


def _rounded_triangle(radius: float, height: float, cut: float) -> np.ndarray:
    """An equilateral prism with its corners cut flat (each cut parallel to the opposite side), so a parallel gripper
    closing across it meets a flat on both sides, not a sharp edge on one."""
    corners = np.stack([radius * np.cos(np.arange(3) * 2 * np.pi / 3), radius * np.sin(np.arange(3) * 2 * np.pi / 3)], 1)
    ring = []
    for i in range(3):
        c, a, b = corners[i], corners[(i - 1) % 3], corners[(i + 1) % 3]
        ring += [c + (a - c) / np.linalg.norm(a - c) * cut, c + (b - c) / np.linalg.norm(b - c) * cut]
    ring = np.array(ring)
    return np.concatenate([np.column_stack([ring, np.full(6, -height / 2)]), np.column_stack([ring, np.full(6, height / 2)])])


SHAPES: dict[str, tuple[np.ndarray, ...]] = {
    # Each kind's convex parts in its body frame (metres), the body origin at the centre of its bounding
    # box; x is the long horizontal side, so the hand closes along y.
    "cube": (_box(0.036, 0.034, 0.034),),
    "brick": (_box(0.064, 0.032, 0.026),),
    "cylinder": (_prism(0.019, 0.034, 24),),
    "puck": (_prism(0.026, 0.020, 28),),
    "hex": (_prism(0.022, 0.030, 6),),
    "prism": (_rounded_triangle(0.027, 0.030, 0.008),),
    "bar": (_box(0.096, 0.022, 0.022),),
    "cross": (_box(0.060, 0.020, 0.024), _box(0.020, 0.060, 0.024)),
    "arch": (_box(0.062, 0.028, 0.012, cz=0.008), _box(0.016, 0.028, 0.016, cx=-0.023, cz=-0.006),
             _box(0.016, 0.028, 0.016, cx=0.023, cz=-0.006)),
}
"""Convex parts of each kind (vertex sets; MuJoCo builds the hulls)."""


def _centred(parts: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    points = np.concatenate(parts)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    return tuple(part - centre for part in parts)


SHAPES = {kind: _centred(parts) for kind, parts in SHAPES.items()}


@dataclass(frozen=True)
class RealKind:
    """A real task object a pool scene holds several of (``obj_<kind>_<k>``), built from the scene's own meshes."""

    hull: str
    """The scene's collision mesh."""
    parts: tuple[tuple[str, str], ...]
    """``(visual mesh, material)`` pairs, all already in the scene."""
    mass: float
    """Kilograms."""


REAL_KINDS: dict[str, RealKind] = {
    "resistor": RealKind(
        "resistor_hull",
        (
            ("resistor_body", "resistor_body"),
            ("resistor_band_brown", "band_brown"),
            ("resistor_band_red", "band_red"),
            ("resistor_band_yellow", "band_yellow"),
        ),
        0.040,
    ),
    "battery": RealKind(
        "battery_hull",
        (("battery_metal", "battery_metal"), ("battery_label", "battery_label"), ("battery_band", "battery_band")),
        0.045,
    ),
    "cpu": RealKind("cpu_hull", (("cpu_body", "cpu_body"), ("cpu_marking", "cpu_marking"), ("cpu_legs", "cpu_legs")), 0.030),
}
"""The real objects of the reBot desk (the oversized resistor model, the 9 V battery, the 8086 CPU model): a scene may
hold several of each as pool bodies; they are never recoloured and are named by what they are, never by a colour."""

REAL_POOL: tuple[tuple[str, int], ...] = (("resistor", 4), ("cpu", 3), ("battery", 2))
"""How many bodies of each real kind :func:`real_pool_xml` writes by default (the most one episode lays out)."""

GRASP_PERIODS: dict[str, float] = {
    "cube": float(np.pi / 2),
    "cross": float(np.pi / 2),
    "hex": float(np.pi / 3),
    "cylinder": float(np.pi / 6),
    "puck": float(np.pi / 6),
}
"""Turns of the closing axis about the vertical that grasp a kind across the same width (its symmetry): a cube is
grasped across either pair of faces, a hexagon across any of its three, a round piece at any heading (tried every
30 deg). Other kinds have only the two closing yaws half a turn apart."""


def grasp_period(name: str) -> float:
    """Return the symmetry period of the closing yaw for object *name* (:data:`GRASP_PERIODS`, else pi)."""
    kind = kind_of(name)
    return GRASP_PERIODS.get(kind, float(np.pi)) if kind is not None else float(np.pi)


DEFAULT_POOL: tuple[tuple[str, int], ...] = (
    ("cube", 3),
    ("brick", 2),
    ("cylinder", 2),
    ("puck", 1),
    ("hex", 1),
    ("bar", 1),
    ("cross", 1),
    ("arch", 1),
)
"""How many bodies of each kind :func:`pool_xml` writes by default (plus the v1 objects the scene keeps). The prism
(a triangle) is defined but left out: a parallel gripper closes on it face against corner, and it slipped in 3 of 60
episodes in the hardest part of the reach even with its corners cut."""

PARKING = (0.0, 0.0, -2.0)
"""Where the hidden shelf parks unused objects, below the floor, in the world frame."""
PARK_SPACING = 0.15
"""Spacing of parked objects on the shelf."""
PARK_CONTYPE = 4
"""Contact bit of the shelf and of parked objects: they meet each other and the shelf, nothing else."""


def kind_of(name: str) -> str | None:
    """Return the pool kind of body *name* (``obj_<kind>_<k>``, a :data:`KINDS` or :data:`REAL_KINDS` kind), or
    ``None`` for another body."""
    if not name.startswith("obj_"):
        return None
    kind = name[len("obj_") :].rsplit("_", 1)[0]
    return kind if kind in KINDS or kind in REAL_KINDS else None


def is_real(name: str) -> bool:
    """Return whether body *name* is one of the real objects (:data:`REAL_KINDS`), by name or as a pool body."""
    return name in REAL_KINDS or kind_of(name) in REAL_KINDS


def real_kind(name: str) -> str | None:
    """Return the real kind of body *name*: ``resistor`` for ``resistor`` or ``obj_resistor_2``; ``None`` otherwise."""
    if name in REAL_KINDS:
        return name
    kind = kind_of(name)
    return kind if kind in REAL_KINDS else None


def body_names(pool: tuple[tuple[str, int], ...] = DEFAULT_POOL) -> tuple[str, ...]:
    """Return the body names :func:`pool_xml` (or :func:`real_pool_xml`) writes for *pool*."""
    return tuple(f"obj_{kind}_{k}" for kind, count in pool for k in range(count))


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{v:.5f}" for v in np.asarray(values).ravel())


def pool_xml(pool: tuple[tuple[str, int], ...] = DEFAULT_POOL, *, friction: str = "0.9 0.01 0.0001", contact: str = "") -> tuple[str, str]:
    """Write the pool's assets and bodies for a scene XML.

    Returns:
        ``(asset, worldbody)`` XML fragments: meshes and one material per body, then the bodies (parked
        on the shelf, which is written too) with their free joints.
    """
    assets: list[str] = []
    bodies: list[str] = []
    for kind, _count in pool:
        for part, verts in enumerate(SHAPES[kind]):
            assets.append(f'<mesh name="pool_{kind}_{part}" vertex="{_fmt(verts)}"/>')
    for index, name in enumerate(body_names(pool)):
        kind = kind_of(name)
        assert kind in KINDS  # noqa: S101 - body_names writes only pool kinds
        assets.append(f'<material name="{name}" rgba="0.5 0.5 0.5 1" specular="0.2" shininess="0.3"/>')
        x, y = (index % 6) * PARK_SPACING, (index // 6) * PARK_SPACING
        lines = [f'<body name="{name}" pos="{x:.3f} {y:.3f} {PARKING[2] + 0.05:.3f}">', f'  <freejoint name="{name}:joint"/>']
        for part in range(len(SHAPES[kind])):
            lines.append(
                f'  <geom name="{name}_hull{part}" type="mesh" mesh="pool_{kind}_{part}" group="3" rgba="0.5 0.5 0.5 0" '
                f'condim="4" friction="{friction}" density="{KINDS[kind].density:.0f}"{" " + contact if contact else ""}/>'
            )
            lines.append(
                f'  <geom name="{name}_vis{part}" type="mesh" mesh="pool_{kind}_{part}" material="{name}" '
                'contype="0" conaffinity="0" group="2" density="0"/>'
            )
        lines.append("</body>")
        bodies.extend(lines)
    shelf = (
        f'<geom name="pool_shelf" type="box" pos="{PARKING[0] + 0.4:.3f} {PARKING[1] + 0.2:.3f} {PARKING[2]:.3f}" '
        f'size="1.0 0.6 0.01" group="5" rgba="0 0 0 0" contype="{PARK_CONTYPE}" conaffinity="{PARK_CONTYPE}" solref="0.004 1" solmix="100"/>'
    )
    return "\n".join(assets), "\n".join([shelf, *bodies])


def real_pool_xml(
    pool: tuple[tuple[str, int], ...] = REAL_POOL,
    *,
    collision_class: str = "object_collision",
    contact: str = "",
    masses: dict[str, float] | None = None,
    collision_parts: dict[str, list[str]] | None = None,
    plan_parts: dict[str, list[str]] | None = None,
) -> str:
    """Write the worldbody XML of the real objects' pool bodies (the scene has their meshes and materials).

    Bodies start parked on the shelf (:data:`PARKING`), after the generic pool's places; a reset places some.
    *masses* overrides :attr:`RealKind.mass` per kind (a scene whose meshes are scaled to the measured size).
    *collision_parts* replaces a kind's convex hull by several geoms, each given as XML attributes (type, size, pos or
    fromto, mass): the resistor is a dumbbell, and a convex hull would fill its waist where the fingers close.
    *plan_parts* adds per kind planning-only geoms (XML attributes, e.g. a proxy mesh with no contact, an unused geom
    group and no mass), named ``<body>_plan<i>``, which a demonstrator may check clearance against.

    Returns:
        The bodies.
    """
    lines: list[str] = []
    index = 0
    for kind, count in pool:
        spec = REAL_KINDS[kind]
        for k in range(count):
            name = f"obj_{kind}_{k}"
            x, y = (index % 6) * PARK_SPACING, 0.6 + (index // 6) * PARK_SPACING
            index += 1
            lines += [f'<body name="{name}" pos="{x:.3f} {y:.3f} {PARKING[2] + 0.05:.3f}">', f'  <freejoint name="{name}:joint"/>']
            parts = (collision_parts or {}).get(kind)
            if parts:
                lines += [f'  <geom class="{collision_class}" name="{name}_hull{i}" {attrs}{" " + contact if contact else ""}/>' for i, attrs in enumerate(parts)]
            else:
                lines.append(f'  <geom class="{collision_class}" name="{name}_hull0" mesh="{spec.hull}" mass="{(masses or {}).get(kind, spec.mass)}"{" " + contact if contact else ""}/>')
            lines += [f'  <geom name="{name}_plan{i}" {attrs}/>' for i, attrs in enumerate((plan_parts or {}).get(kind, []))]
            lines += [f'  <geom class="visual" mesh="{mesh}" material="{material}"/>' for mesh, material in spec.parts]
            lines.append("</body>")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POOL",
    "REAL_KINDS",
    "REAL_POOL",
    "RealKind",
    "is_real",
    "real_kind",
    "real_pool_xml",
    "GRASP_PERIODS",
    "KINDS",
    "OBJECT_COLORS",
    "PARKING",
    "PARK_CONTYPE",
    "PARK_SPACING",
    "SHAPES",
    "ObjectKind",
    "body_names",
    "grasp_period",
    "kind_of",
    "pool_xml",
]
