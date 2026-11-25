# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Object geometry, grasps and placement targets of the booth task, read from the compiled model.

Frames:

- box ``B``: origin at the centre of its 155 x 95 x 35 mm envelope; x along the long axis, y across
  the width, z through the thickness with the finned body on +z ("fins up" when lying). It starts
  standing on its long edge in the +y-most bin slot: y up, fins toward -y.
- cover ``C``: origin at the middle of its underside; y along the plateau (the long axis), z up; the
  tab stands on the plateau's middle, thin along y.
- tool site: x is the approach axis (from the wrist toward the fingertips), the fingers close
  along y, and the wrist camera sits on the site's +z side.

Grasps are tool poses in an object's frame. Every target is a pose of an object or of the tool in
the booth frame, computed from the true object poses (privileged state), never a joint vector.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from physicalai_mujoco_plugin.booth_task.frames import Pose, pose, rot_x, rot_z, tool_rotation

PAD_HALF = 0.008
"""Half the finger pad's length along the approach axis; the tool site is at the pad's far end."""


@dataclass(frozen=True)
class BoxShape:
    """The box's dimensions (m), from its collision geoms."""

    length: float
    width: float
    thickness: float
    base: float
    body_length: float
    body_width: float


@dataclass(frozen=True)
class CoverShape:
    """The cover's dimensions (m), from its collision geoms."""

    height: float
    """Plateau top above the underside."""
    tab_height: float
    inner_length: float
    """Between the inner faces of the end walls."""
    plateau_underside: float


@dataclass(frozen=True)
class Layout:
    """The fixtures and object shapes of one compiled (possibly perturbed) model."""

    box: BoxShape
    cover: CoverShape
    bracket: Pose
    """Bracket body frame: origin at the middle of its underside, y along its long axis."""
    bracket_floor: float
    """Height of the bracket floor's top (m)."""
    bracket_rim: float
    bracket_ear: float
    bin: Pose
    slot_center: np.ndarray
    """The +y-most slot's middle on its floor, in the booth frame."""
    slot_floor: float
    bin_top: float
    """Height of the bin's walls and dividers (m)."""

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> Layout:  # noqa: PLR0914 - one read per geom and body
        """Read the layout from the model's geoms and static body poses.

        Returns:
            The layout of *model*.
        """
        box_body = model.body("box").id
        box_geoms = {model.geom(g).name: g for g in range(model.ngeom) if model.geom_bodyid[g] == box_body}
        base, body = box_geoms["box_base"], box_geoms["box_body"]
        box = BoxShape(
            length=2 * float(model.geom_size[base][0]),
            width=2 * float(model.geom_size[base][1]),
            thickness=2 * float(model.geom_size[base][2] + model.geom_size[body][2]),
            base=2 * float(model.geom_size[base][2]),
            body_length=2 * float(model.geom_size[body][0]),
            body_width=2 * float(model.geom_size[body][1]),
        )
        plateau = model.geom("cover_plateau").id
        tab = model.geom("cover_tab").id
        wall = model.geom("cover_wall_+").id
        top = float(model.geom_pos[plateau][2] + model.geom_size[plateau][2])
        cover = CoverShape(
            height=top,
            tab_height=2 * float(model.geom_size[tab][2]),
            inner_length=2 * float(model.geom_pos[wall][1] - model.geom_size[wall][1]),
            plateau_underside=float(model.geom_pos[plateau][2] - model.geom_size[plateau][2]),
        )
        bracket_body = model.body("bracket").id
        bracket = _static_pose(model, bracket_body)
        floor = model.geom("bracket_floor").id
        rim = model.geom("bracket_rim_x+").id
        ear = model.geom("bracket_ear_++").id
        bin_body = model.body("bin").id
        bin_pose = _static_pose(model, bin_body)
        divider = model.geom("bin_divider_2").id
        wall_y = model.geom("bin_wall_y+").id
        slot_y = (
            float(model.geom_pos[divider][1] + model.geom_size[divider][1])
            + float(model.geom_pos[wall_y][1] - model.geom_size[wall_y][1])
        ) / 2
        bin_base = model.geom("bin_base").id
        slot_floor = float(model.geom_pos[bin_base][2] + model.geom_size[bin_base][2])
        return cls(
            box=box,
            cover=cover,
            bracket=bracket,
            bracket_floor=float(model.geom_pos[floor][2] + model.geom_size[floor][2]),
            bracket_rim=float(model.geom_pos[rim][2] + model.geom_size[rim][2]),
            bracket_ear=float(model.geom_pos[ear][2] + model.geom_size[ear][2]),
            bin=bin_pose,
            slot_center=bin_pose.apply([0.0, slot_y, slot_floor]),
            slot_floor=slot_floor,
            bin_top=float(model.geom_pos[wall_y][2] + model.geom_size[wall_y][2]),
        )

    # --- grasps (tool pose in the object's frame) ----------------------------------------------

    def left_edge_grasp(self, x: float, tilt: float) -> Pose:
        """LEFT on the box's top long edge (box standing on it), across the thickness.

        The fingers close on the two broad faces 20 mm below the edge, where the finned body is
        under the pads; the approach leans *tilt* toward the box's +x end (pointing down and forward
        when the box is in its slot), and the wrist camera sits on the +x side.

        Returns:
            The tool pose in the box's frame.
        """
        b = self.box
        approach = np.array([math.sin(tilt), -math.cos(tilt), 0.0])
        pad = np.array([x, b.width / 2 - 0.020, 0.0])
        return Pose(pad + PAD_HALF * approach, tool_rotation(approach, [0.0, 0.0, -1.0]))

    def right_body_grasp(self, x: float, tilt: float) -> Pose:
        """RIGHT across the finned body's width (72 mm), from the fins side.

        The pads cover the body's sides from 1 mm above the base plate to the top; the approach
        leans *tilt* toward the box's +x end, and the wrist camera sits on the -x side.

        Returns:
            The tool pose in the box's frame.
        """
        b = self.box
        approach = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
        body_top = b.thickness / 2
        pad = np.array([x, 0.0, body_top - 0.010])
        return Pose(pad + PAD_HALF * approach, tool_rotation(approach, [0.0, -1.0, 0.0]))

    def tab_grasp(self, tilt: float) -> Pose:
        """RIGHT pinching the cover's tab across its thickness, from above, leaning *tilt* toward +x.

        Returns:
            The tool pose in the cover's frame.
        """
        c = self.cover
        approach = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
        pad = np.array([0.0, 0.0, c.height + 0.013])
        return Pose(pad + PAD_HALF * approach, tool_rotation(approach, [0.0, 1.0, 0.0]))

    # --- object targets ---------------------------------------------------------------------------

    def box_in_bracket(self) -> Pose:
        """The box lying in the bracket: fins up, long axis along the bracket's, centred, resting.

        Its +x end (where RIGHT holds it) points to the bracket's -y end, toward the right arm.

        Returns:
            The box pose in the booth frame.
        """
        rot = self.bracket.rot @ rot_z(-math.pi / 2)
        z = self.bracket_floor + self.box.thickness / 2
        return Pose(np.array([self.bracket.pos[0], self.bracket.pos[1], z]), rot)

    def box_in_slot(self, start: Pose) -> Pose:
        """The box standing upright in its slot, aligned with it, fully down on the slot floor.

        Along the slot it goes back where it started; it faces the way it started (fins toward -y).

        Returns:
            The box pose in the booth frame.
        """
        rot = self.bin.rot @ rot_x(math.pi / 2)
        if float(np.dot(rot[:, 0], start.rot[:, 0])) < 0:  # keep the end it started with toward +x
            rot = self.bin.rot @ rot_z(math.pi) @ rot_x(math.pi / 2)
        z = self.slot_floor + self.box.width / 2
        centre = self.slot_center
        # Along the slot keep the start position; across it, the slot's middle.
        along = self.bin.rot[:, 0]
        offset = float(np.dot(start.pos - centre, along))
        pos = centre + offset * along
        return Pose(np.array([pos[0], pos[1], z]), rot)

    def cover_on_box(self, box: Pose) -> Pose:
        """The cover straddling the lying box: long axes parallel, centred on it, tab up.

        It comes to rest on the box's top or, where the bracket's ears stand under its walls, on the
        ears; the target height is the higher of the two.

        Returns:
            The cover pose in the booth frame.
        """
        long_axis = box.rot[:, 0]
        yaw = math.atan2(long_axis[1], long_axis[0]) - math.pi / 2
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2  # the cover is symmetric end for end
        on_box = box.pos[2] + self.box.thickness / 2 - self.cover.plateau_underside
        z = max(on_box, self.bracket_ear)
        return Pose(np.array([box.pos[0], box.pos[1], z]), rot_z(yaw))


def _static_pose(model: mujoco.MjModel, body: int) -> Pose:
    """World pose of a body attached directly to the world, from the model (no data needed).

    Returns:
        The body's pose in the booth frame.
    """
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, model.body_quat[body])
    return pose(model.body_pos[body], m.reshape(3, 3))


def handover_box_pose(centre: np.ndarray, fins_from_vertical: float, yaw: float = 0.0) -> Pose:
    """Box pose for a handover: long axis along +x turned by *yaw*, fins leaning toward -y.

    ``fins_from_vertical`` is the angle between the fins' normal and straight up: 90 deg is the box
    standing on its edge with the fins toward -y (as in the slot), 0 deg lying fins up.

    Returns:
        The box pose in the booth frame.
    """
    # Lying fins up is the identity; standing in the slot is +90 deg about x.
    return Pose(np.asarray(centre, dtype=float), rot_z(yaw) @ rot_x(fins_from_vertical))


__all__ = ["BoxShape", "CoverShape", "Layout", "handover_box_pose"]
