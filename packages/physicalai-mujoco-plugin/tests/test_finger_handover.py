# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""r17 (R-56, coordinator ruling during r17): the phone path's finger hand-over (the arm splat's fingers instead of the CAD
fingers' colour) acts only when a caller turns it on (the matched-state HELDOUT render); robot cameras, recordings, VIDEO and
demos never compute it. Needs the bucket-v2 kit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get("REAL2SIM_ROOT", Path.home() / "real2sim-data/bucket-v2")).expanduser()
KIT = ROOT / "kit/real2sim.json"
V2 = Path(__file__).resolve().parents[1] / "scripts/real2sim/v2"
pytestmark = pytest.mark.skipif(not KIT.exists(), reason="needs the bucket-v2 kit")


@pytest.fixture(scope="module")
def setup():
    os.environ.setdefault("REAL2SIM_ROOT", str(ROOT)); os.environ.setdefault("KIT", str(KIT)); os.environ.setdefault("PHYSICALAI_PLAN_PROCESS", "0")
    sys.path.insert(0, str(V2))
    import bucketsim_v2 as b

    robot = b.make_robot(0); b.reset(robot, 0); b.matched_state(robot)
    return b, robot, b.compositor(robot, 320, 240, show_objects=False)


def test_off_by_default_and_never_computed_unless_on(setup, monkeypatch) -> None:
    b, robot, comp = setup
    assert comp.phone_finger_handover is False

    def boom() -> float:
        raise AssertionError("the finger hand-over was computed")

    monkeypatch.setattr(comp, "_robot_splat_weight", boom)
    comp.render_camera("overview"); comp.render_camera("wrist")   # robot cameras (recordings, ROBOTCAM)
    cams = b.pinhole_cameras(); T, K, _, _ = next(iter(cams.values()))
    comp.render_sfm(T, K)   # a phone view with the switch off (the VIDEO's case)


def test_on_at_the_calibrated_pose_leaves_the_seg_alone(setup) -> None:
    b, robot, comp = setup
    cams = b.pinhole_cameras(); T, K, _, _ = next(iter(cams.values()))
    comp.phone_finger_handover = False; off = comp.render_sfm(T, K, outputs=True)
    comp.phone_finger_handover = True
    try:
        assert comp._robot_splat_weight() == pytest.approx(1.0)   # matched state: the calibrated pose
        on = comp.render_sfm(T, K, outputs=True)
    finally:
        comp.phone_finger_handover = False
    np.testing.assert_array_equal(on["seg"], off["seg"])   # D3's geometry overlays still see the CAD fingers
