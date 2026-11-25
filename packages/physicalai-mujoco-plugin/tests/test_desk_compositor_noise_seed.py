"""The sensor noise seeds OpenCV's generator from the sim time: it must stay a valid C int for long sessions."""
import types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from physicalai_mujoco_plugin.real2sim import desk_compositor as dc


@pytest.mark.parametrize("t", [0.0, 270.0, 300.0, 3600.0 * 24])
def test_sensor_noise_seed_fits_c_int_for_long_sim_times(t):
    comp = types.SimpleNamespace(kit=types.SimpleNamespace(sensor_noise={"read_lin": 1e-3, "shot_lin": 5e-5}),
                                 d=types.SimpleNamespace(time=t), m=None)
    comp.m = types.SimpleNamespace()
    img = np.full((48, 64, 3), 128, np.uint8)
    orig = dc.mujoco.mj_id2name
    dc.mujoco.mj_id2name = lambda *a: "overview"
    try:
        out = dc.Compositor._sensor_noise(comp, img, 0)
    finally:
        dc.mujoco.mj_id2name = orig
    assert out.shape == img.shape and out.dtype == np.uint8
