"""r18 (D7 retiming, 5(a) exact parallel planning): the move profile, the operator timing draw and the tie rule."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from physicalai_mujoco_plugin import bucket_sort_demo as bsd


@pytest.mark.parametrize("ramp", [None, 0.5, 0.25, 0.15])
def test_move_progress_ends_and_monotonic(ramp: float | None) -> None:
    s = np.linspace(0.0, 1.0, 2001)
    y = bsd.move_progress(s, ramp)
    assert y[0] == pytest.approx(0.0, abs=1e-12)
    assert y[-1] == pytest.approx(1.0, abs=1e-12)
    assert (np.diff(y) >= -1e-12).all()
    v = np.gradient(y, s)
    assert v[0] == pytest.approx(0.0, abs=1e-3)   # starts and ends at rest
    assert v[-1] == pytest.approx(0.0, abs=1e-3)
    assert np.abs(np.diff(v)).max() < 0.01   # the speed is continuous (no step)


def test_move_progress_defaults_to_min_jerk() -> None:
    s = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(bsd.move_progress(s), bsd.min_jerk(s))
    np.testing.assert_allclose(bsd.move_progress(s, 0.5), bsd.min_jerk(s))


def test_ramp_spends_less_time_slow() -> None:
    s = np.linspace(0.0, 1.0, 10001)

    def slow_share(ramp: float | None) -> float:
        v = np.gradient(bsd.move_progress(s, ramp), s)
        return float((v < 0.1 * v.max()).mean())

    assert slow_share(0.25) < slow_share(None) - 0.04


@pytest.mark.parametrize("ramp", [None, 0.25])
@pytest.mark.parametrize("y", [0.0, 0.1, 0.37, 0.5, 0.9, 1.0])
def test_inverse_matches_profile(ramp: float | None, y: float) -> None:
    assert float(bsd.move_progress(bsd.inverse_min_jerk(y, ramp), ramp)) == pytest.approx(y, abs=1e-9)


def test_script_ramp_keeps_the_ends() -> None:
    start = np.array([0.2, 0.0, 0.1, 0.0, 0.03])
    for ramp in (None, 0.25):
        script = bsd.MotionScript(start)
        script.ramp = ramp
        script.add("approach", pos=np.array([0.3, 0.1, 0.2]), duration=1.0)
        states = script.at(np.array([0.0, script.t_end]))
        np.testing.assert_allclose(states[0][:3], start[:3], atol=1e-9)
        np.testing.assert_allclose(states[-1][:3], [0.3, 0.1, 0.2], atol=1e-9)


def test_operator_timing_draw() -> None:
    p = bsd.DemoParams()
    a = bsd.operator_timing(p, np.random.default_rng([7, 18]))
    b = bsd.operator_timing(p, np.random.default_rng([7, 18]))
    assert a == b   # its own stream: repeatable from the episode's seed
    t = bsd.OPERATOR_TIMING_R18
    assert a.motion_ramp == t["motion_ramp"]
    assert a.release_open_mps == t["release_open_mps"]
    assert a.arrive_wait_s == t["arrive_wait_s"]
    assert a.transit_speed == pytest.approx(p.transit_speed * t["speed_factor"])
    assert a.fine_speed == pytest.approx(p.fine_speed * t["speed_factor"])
    assert a.lift_speed == pytest.approx(p.lift_speed * t["speed_factor"])
    assert t["settle_s"][0] <= a.settle_s <= t["settle_s"][1]
    assert t["after_release_s"][0] <= a.after_release_s <= t["after_release_s"][1]


def test_defaults_keep_r17_timing() -> None:
    p = bsd.DemoParams()
    assert p.motion_ramp is None
    assert p.release_open_mps is None
    assert p.arrive_wait_s == pytest.approx(0.1)


def test_severity_tie_goes_to_the_earlier(monkeypatch: pytest.MonkeyPatch) -> None:
    sev = {"a": 0.3540555877611876, "b": 0.3540555877611729, "c": 0.35}
    monkeypatch.setattr(bsd, "_severity", lambda plan: sev[plan.name])
    a, b, c = (SimpleNamespace(name=k) for k in "abc")
    # a 1e-14 difference (floating point across processes) is a tie: neither replaces the other
    assert not bsd._less_severe(b, a)
    assert not bsd._less_severe(a, b)
    assert bsd._less_severe(c, a)


@pytest.mark.parametrize("ramp", [None, 0.25, 0.15])
@pytest.mark.parametrize(("distance", "speed", "accel"), [(0.10, 0.2, 1.0), (0.03, 0.4, 0.5), (0.2, 0.3, None)])
def test_move_duration_keeps_the_caps(ramp: float | None, distance: float, speed: float, accel: float | None) -> None:
    T = bsd.move_duration(distance, speed, 0.0, accel, ramp)
    t = np.linspace(0.0, T, 20001)
    x = distance * bsd.move_progress(t / T, ramp)
    v = np.gradient(x, t)
    a = np.gradient(v, t)
    assert v.max() <= speed * 1.001
    if accel is not None:
        assert np.abs(a).max() <= accel * 1.01


def test_new_draws_keep_the_timing() -> None:
    rng = np.random.default_rng(3)
    timed = bsd.operator_timing(bsd.DemoParams(), np.random.default_rng(4))
    assert bsd._timed_like(bsd.DemoParams(), timed, rng).motion_ramp == bsd.OPERATOR_TIMING_R18["motion_ramp"]
    assert bsd._timed_like(bsd.DemoParams(), bsd.DemoParams(), rng).motion_ramp is None


def test_fingers_keep_min_jerk_under_a_ramp() -> None:
    start = np.array([0.2, 0.0, 0.1, 0.0, 0.08])
    script = bsd.MotionScript(start)
    script.ramp = 0.25
    script.add("approach", pos=np.array([0.2, 0.0, 0.05]), grip=0.02, duration=1.0)   # a ramped move that also opens
    t = np.linspace(0.0, 1.0, 11)
    states = script.at(t)
    np.testing.assert_allclose(states[:, 4], 0.08 + bsd.min_jerk(t) * (0.02 - 0.08), atol=1e-12)
    np.testing.assert_allclose(states[:, 2], 0.1 + bsd.move_progress(t, 0.25) * (0.05 - 0.1), atol=1e-12)


def test_contact_moves_keep_min_jerk() -> None:
    start = np.array([0.2, 0.0, 0.2, 0.0, 0.08])
    script = bsd.MotionScript(start)
    script.ramp = 0.25
    script.add("approach", pos=np.array([0.3, 0.0, 0.2]), duration=1.0)
    script.add("descend", pos=np.array([0.3, 0.0, 0.1]), duration=1.0)
    assert script.moves[0].ramp == 0.25
    assert script.moves[1].ramp is None
    t = np.linspace(1.0, 2.0, 11)
    np.testing.assert_allclose(script.at(t)[:, 2], 0.2 + bsd.min_jerk(t - 1.0) * (0.1 - 0.2), atol=1e-12)
