# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""r13: what the cameras show touching is what collides.

1. The fingers' and the gripper housing's collision coincides with their drawn surfaces: no drawn point outside the
   collision by more than 1 mm; the fingers' collision beyond the drawn surface within 2 mm at the 95th percentile. The
   housing (its drawing ends ~70 mm above the fingertips, beyond any object on the desk; the demonstrator runs count
   its contacts) is held to what r13 reached: 5 mm (p95), 8 mm (max); Menagerie's reached 15 mm.
2. In a demonstrator grasp (sim, the bucket-v2 desk with its kit), while both fingers hold the object, every contact
   point between a finger and the held object projects onto the drawn finger in the wrist and the overview camera
   (segmentation of the drawn geoms only): within 2 mm at the contact's depth (the drawing-collision tolerance) + 1 px.
Needs the bucket-v2 kit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import re as _re; PROXY = _re.compile(r"_plan\d+$|_pc\d+_\d+$")   # r14 planning-only proxies

ROOT = Path(os.environ.get("REAL2SIM_ROOT", Path.home() / "real2sim-data/bucket-v2")).expanduser()
KIT = ROOT / "kit/real2sim.json"
V2 = Path(__file__).resolve().parents[1] / "scripts/real2sim/v2"
pytestmark = pytest.mark.skipif(not KIT.exists(), reason="needs the bucket-v2 kit")


@pytest.fixture(scope="module")
def bs():
    os.environ.setdefault("REAL2SIM_ROOT", str(ROOT)); os.environ.setdefault("KIT", str(KIT)); os.environ.setdefault("PHYSICALAI_PLAN_PROCESS", "0")
    sys.path.insert(0, str(V2))
    import bucketsim_v2 as b

    return b


def _mesh(m, bid, drawn):
    import mujoco

    Vs, Fs, off = [], [], 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        if (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0) != drawn: continue
        if PROXY.search(m.geom(g).name or ""): continue   # r14 planning proxies: neither drawn nor colliding
        r = np.zeros(9); mujoco.mju_quat2Mat(r, m.geom_quat[g]); mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        Vs.append(m.mesh_vert[a:a + n].astype(float) @ r.reshape(3, 3).T + m.geom_pos[g]); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n
    return Vs, Fs


@pytest.mark.parametrize(("body", "p95_mm", "max_mm"), [("gripper_left", 2.0, 4.5), ("gripper_right", 2.0, 4.5), ("gripper_end", 5.0, 8.0)])
def test_drawn_and_collision_coincide(bs, body, p95_mm, max_mm) -> None:
    from collision_decomp import report

    robot = bs.make_robot(0); m = robot._sim.model; bid = m.body(body).id
    Vs, Fs = _mesh(m, bid, True); hulls, _ = _mesh(m, bid, False)
    rep = report(np.vstack(Vs), np.vstack(Fs), hulls, spacing=0.0008)
    assert rep["drawn_outside_collision"]["max_mm"] <= 1.0, rep
    assert rep["collision_beyond_drawn"]["p95_mm"] <= p95_mm and rep["collision_beyond_drawn"]["max_mm"] <= max_mm, rep


def _project(m, d, cam, P):
    c = m.camera(cam).id; R = d.cam_xmat[c].reshape(3, 3); f = 0.5 * 480 / np.tan(np.radians(m.cam_fovy[c]) / 2)
    q = (P - d.cam_xpos[c]) @ R   # camera frame: x right, y up, -z forward
    return np.c_[320 + f * q[:, 0] / -q[:, 2], 240 - f * q[:, 1] / -q[:, 2]], -q[:, 2], f


def test_held_contacts_on_drawn_fingers(bs) -> None:
    import mujoco

    robot = bs.make_robot(5700); m, d = robot._sim.model, robot._sim.data
    bs.reset(robot, 5700); demo_ok = False
    fl, fr = m.body("gripper_left").id, m.body("gripper_right").id
    drawn_fingers = [g for g in range(m.ngeom) if m.geom_bodyid[g] in (fl, fr) and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and not PROXY.search(m.geom(g).name or "")]
    seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering()
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator
    held_ticks, checked = 0, 0
    for _ in range(int(30 / bs.DT)):
        bs.tick(robot, demo)
        touch: dict[int, set] = {}; pts: dict[int, list] = {}
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for fb, ob in ((b1, b2), (b2, b1)):
                if fb in (fl, fr) and m.body(ob).name.startswith("obj_"):
                    touch.setdefault(int(ob), set()).add(int(fb)); pts.setdefault(int(ob), []).append(c.pos.copy())
        held = [ob for ob, fs in touch.items() if len(fs) == 2]
        if not held: continue
        held_ticks += 1
        if held_ticks % 10: continue   # every 10th held tick (0.2 s)
        P = np.array(pts[held[0]])
        for cam in ("wrist", "overview"):
            keep = m.geom_rgba.copy(); grp = m.geom_group.copy()
            try:
                m.geom_rgba[:, 3] = 0; m.geom_rgba[drawn_fingers, 3] = 1; m.geom_group[:] = 3; m.geom_group[drawn_fingers] = 0
                o = mujoco.MjvOption(); o.geomgroup[:] = 0; o.geomgroup[0] = 1
                seg.update_scene(d, camera=cam, scene_option=o); s = seg.render()
            finally:
                m.geom_rgba[:] = keep; m.geom_group[:] = grp
            finger = (s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(s[..., 0], drawn_fingers)
            uv, z, f = _project(m, d, cam, P)
            ys, xs = np.nonzero(finger)
            for (u, v), zz in zip(uv, z):
                if zz <= 0 or not (0 <= u < 640 and 0 <= v < 480) or not len(xs): continue
                tol = f * 0.002 / zz + 1.0
                gap = np.hypot(xs - u, ys - v).min()
                assert gap <= tol, f"{cam}: contact at ({u:.0f}, {v:.0f}) is {gap:.1f} px from the drawn fingers (tolerance {tol:.1f} px)"
                checked += 1
        if held_ticks >= 60: demo_ok = True; break
    assert demo_ok and checked > 0, f"no hold of 1.2 s in 30 s (held ticks {held_ticks})"
