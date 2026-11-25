# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""r14: the planning-only proxies CONTAIN the fine collision pieces (the coordinator's condition: a path that clears the
proxies cannot touch the physics geometry). Checked on the compiled bucket-v2 scene, per body that has proxies: every
vertex of every collision piece (the convex hull MuJoCo collides) lies inside the union of the body's proxies. Also:
proxies neither collide nor are drawn while an object is placed. Needs the bucket-v2 kit."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get("REAL2SIM_ROOT", Path.home() / "real2sim-data/bucket-v2")).expanduser()
KIT = ROOT / "kit/real2sim.json"
V2 = Path(__file__).resolve().parents[1] / "scripts/real2sim/v2"
pytestmark = pytest.mark.skipif(not KIT.exists(), reason="needs the bucket-v2 kit")
PROXY = re.compile(r"_plan\d+$|_pc\d+_\d+$")


@pytest.fixture(scope="module")
def model():
    os.environ.setdefault("REAL2SIM_ROOT", str(ROOT)); os.environ.setdefault("KIT", str(KIT)); os.environ.setdefault("PHYSICALAI_PLAN_PROCESS", "0")
    sys.path.insert(0, str(V2))
    import bucketsim_v2 as b

    robot = b.make_robot(0); b.reset(robot, 0)
    return robot._sim.model


def _points(m, g):
    import mujoco

    r = np.zeros(9); mujoco.mju_quat2Mat(r, m.geom_quat[g]); R = r.reshape(3, 3)
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); P = m.mesh_vert[a:a + n].astype(float)
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        rad, h = m.geom_size[g][:2]; th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ring = np.c_[rad * np.cos(th), rad * np.sin(th), np.zeros(48)]; P = np.vstack([ring - [0, 0, h], ring + [0, 0, h]])
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        P = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * m.geom_size[g]
    else:
        return None
    return P @ R.T + m.geom_pos[g]


@pytest.mark.parametrize("body", ["gripper_left", "gripper_right", "gripper_end", "link6", "obj_cpu_0", "obj_battery_0", "obj_resistor_0", "bucket_red"])
def test_proxies_contain_the_collision(model, body) -> None:
    from collision_decomp import inside_hulls

    m = model; bid = m.body(body).id
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    names = {g: m.geom(g).name or "" for g in gs}
    proxies = [g for g in gs if PROXY.search(names[g])]
    pieces = [g for g in gs if not PROXY.search(names[g]) and (m.geom_contype[g] or m.geom_conaffinity[g] or re.search(r"_hull\d+$|_tight\d+$|_c\d+$", names[g]))]
    pieces = [g for g in pieces if not (body.startswith("bucket_") and "_item" in names[g])]   # contents: their own _pc proxies
    assert proxies and pieces, (len(proxies), len(pieces))
    hulls = [_points(m, g) for g in proxies]
    P = np.vstack([p for p in (_points(m, g) for g in pieces) if p is not None])
    inside = inside_hulls(P, hulls, eps=2e-6)
    assert inside.all(), f"{(~inside).sum()} of {len(P)} collision vertices outside the proxies"


def test_proxies_never_collide_or_draw(model) -> None:
    m = model
    for g in range(m.ngeom):
        if PROXY.search(m.geom(g).name or "") and m.body(m.geom_bodyid[g]).name not in ():
            body = m.body(m.geom_bodyid[g]).name
            if body.startswith("obj_") and m.geom_contype[g]:   # a parked object rests on its proxies (shelf contact type only)
                assert int(m.geom_contype[g]) == 4 and int(m.geom_conaffinity[g]) == 4, body
                continue
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, m.geom(g).name
            assert m.geom_rgba[g][3] == 0.0, m.geom(g).name
