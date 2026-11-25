"""Booth task: two WidowX AI arms, the scripted IK demonstrator and its physical-honesty checks (real MuJoCo, no display)."""

import math

import mujoco
import numpy as np
import pytest
from loguru import logger

from physicalai_mujoco_plugin.booth_task import EPISODE_END, make_episode, run_episode
from physicalai_mujoco_plugin.booth_task.frames import body_pose
from physicalai_mujoco_plugin.booth_task.harness import DATA_FIELDS, run, static_checks
from physicalai_mujoco_plugin.booth_task.layout import Layout
from physicalai_mujoco_plugin.booth_task.scene import (
    GRIPPER_FORCE_LIMIT,
    JOINT_SPEED_LIMIT,
    Perturbation,
    SLEEP_POSES,
    arm_ids,
    slot_play,
)
from physicalai_mujoco_plugin.scene_registry import get_scene, list_scenes_for

pytestmark = pytest.mark.requires_download  # Menagerie's WidowX AI

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")


@pytest.fixture(autouse=True)
def _quiet() -> None:
    logger.remove()


@pytest.fixture(scope="module")
def nominal() -> tuple:
    return make_episode(0, perturb=False)


def test_scene_is_registered_for_two_widowx_ai_arms() -> None:
    from physicalai_mujoco_plugin.profiles import get_profile

    scene = get_scene("booth_task")
    assert scene.arm_counts == (2,)
    assert "booth_task" in list_scenes_for(get_profile("trossen_wxai"), 2)
    assert "booth_task" not in list_scenes_for(get_profile("trossen_wxai"), 1)
    assert scene.scene_xml_path.exists()


def test_plugin_exposes_the_composite_robot_names_in_degrees_and_metres() -> None:
    from physicalai_mujoco_plugin.robot import MuJoCoRobot

    robot = MuJoCoRobot("trossen_wxai", scene="booth_task", bimanual=True, cameras=[], rate_hz=30, seed=0)
    assert robot.joint_names == [f"{side}_{joint}" for side in ("left", "right") for joint in JOINTS]
    robot.connect()
    try:
        obs = robot.get_observation()
        # SLEEP: pan and wrist as fitted to V1 (left pan 22.7 deg, right 8.9 deg), shoulder and elbow lifted
        # 2 and 5 degrees off their limits; grippers in metres.
        assert obs.joint_positions[0] == pytest.approx(22.7, abs=1.0)
        assert obs.joint_positions[7] == pytest.approx(8.9, abs=1.0)
        assert obs.joint_positions[1] == pytest.approx(2.0, abs=1.0)
        assert obs.joint_positions[2] == pytest.approx(5.0, abs=1.5)
        assert 0.0 <= obs.joint_positions[6] < 0.01
        assert robot._automation.autopilot.available
        model = robot._sim.model
        gripper = model.actuator("left_right_carriage_joint").id
        assert model.actuator_forcerange[gripper][1] == GRIPPER_FORCE_LIMIT
    finally:
        robot.disconnect()


def test_model_parameters_stay_within_the_rubric_limits(nominal: tuple) -> None:
    model, _, _ = nominal
    assert model.opt.timestep <= 0.002
    assert np.all(model.geom_friction[:, 0] <= 1.5)
    assert np.all(model.geom_friction[:, 1] <= 10 * 0.005)
    assert np.all(model.geom_friction[:, 2] <= 10 * 0.0001)
    assert static_checks(model) == []
    for side in ("left", "right"):
        actuator = model.actuator(f"{side}_right_carriage_joint").id
        assert model.actuator_forcerange[actuator][1] <= GRIPPER_FORCE_LIMIT <= 100.0
    # The arm actuators keep Menagerie's gains.
    assert model.actuator_gainprm[model.actuator("left_joint_1").id, 0] == pytest.approx(83.69874)


def test_static_checks_catch_a_weld_on_the_box() -> None:
    from physicalai_mujoco_plugin.booth_task.scene import scene_xml_path, tune_model
    from physicalai_mujoco_plugin.compose import compose_scene_spec
    from physicalai_mujoco_plugin.profiles import get_profile

    spec = compose_scene_spec(scene_xml_path(), get_profile("trossen_wxai"))
    weld = spec.add_equality()
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.name1, weld.name2 = "box", "left_gripper_left"
    model = spec.compile()
    tune_model(model)
    assert static_checks(model)


def test_start_state_is_sleep_with_the_box_standing_in_its_slot(nominal: tuple) -> None:
    model, data, _ = nominal
    for side in ("left", "right"):
        # make_episode lets everything settle for 0.5 s first: the arms sag a fraction of a degree.
        np.testing.assert_allclose(data.qpos[arm_ids(model, side).qpos], SLEEP_POSES[side], atol=np.radians(1.0))
    assert data.time == 0.0
    layout = Layout.from_model(model)
    box = body_pose(data, model.body("box").id)
    assert abs(box.rot[2, 1]) > math.cos(math.radians(6))  # the 95 mm width is vertical (it leans in the slot's play)
    assert box.pos[2] == pytest.approx(layout.slot_floor + layout.box.width / 2, abs=0.002)
    mujoco.mj_forward(model, data)
    assert min((data.contact[i].dist for i in range(data.ncon)), default=0.0) > -0.001


@pytest.mark.parametrize("seed", range(20))
def test_perturbations_stay_within_the_d7_ranges(seed: int) -> None:
    draw = Perturbation.draw(np.random.default_rng(seed))
    assert max(abs(v) for v in draw.box_dxy) <= 0.002
    assert abs(draw.box_dyaw) <= math.radians(2)
    assert max(abs(v) for v in draw.bracket_dxy) <= 0.015
    assert abs(draw.bracket_dyaw) <= math.radians(5)
    assert max(abs(v) for v in draw.cover_dxy) <= 0.020
    assert abs(draw.cover_dyaw) <= math.radians(10)
    assert max(abs(v) for v in draw.bin_dxy) <= 0.005
    assert abs(draw.bin_dyaw) <= math.radians(2)
    assert all(np.all(np.abs(v) <= math.radians(2)) for v in draw.arm_dq.values())
    assert 0.8 <= draw.box_mass_scale <= 1.2
    assert 0.8 <= draw.cover_mass_scale <= 1.2
    assert 0.75 <= draw.friction_scale <= 1.25


def test_perturbed_box_starts_clear_of_the_slot_walls() -> None:
    for seed in range(5):
        model, data, _ = make_episode(seed, perturb=True)
        assert slot_play(model) > 0
        mujoco.mj_forward(model, data)
        box = model.body("box").id
        worst = min(
            (data.contact[i].dist for i in range(data.ncon) if box in (model.geom_bodyid[data.contact[i].geom1], model.geom_bodyid[data.contact[i].geom2])),
            default=0.0,
        )
        assert worst > -0.0015


def test_controller_writes_only_ctrl() -> None:
    model, data, controller = make_episode(3, perturb=True)
    steps = round(controller.period / model.opt.timestep)
    while data.time < 10.5:
        before = {name: np.array(getattr(data, name), copy=True) for name in DATA_FIELDS}
        controller(model, data)
        for name, value in before.items():
            assert np.array_equal(value, getattr(data, name)), name
        for _ in range(steps):
            mujoco.mj_step(model, data)


def test_left_lifts_the_box_out_of_its_slot() -> None:
    model, data, controller = make_episode(0, perturb=False)
    start = body_pose(data, model.body("box").id).pos
    run_episode(model, data, controller, until=13.0)
    lifted = body_pose(data, model.body("box").id).pos
    assert lifted[2] - start[2] > 0.05
    # Held only by the left fingers: nothing else touches the box.
    fingers = set(arm_ids(model, "left").finger_bodies)
    box = model.body("box").id
    for i in range(data.ncon):
        pair = {int(model.geom_bodyid[data.contact[i].geom1]), int(model.geom_bodyid[data.contact[i].geom2])}
        if box in pair:
            assert pair - {box} <= fingers
    assert controller.arms["left"].worst_ik_error < 0.002


def test_joint_speed_limits_are_the_real_arm_s() -> None:
    np.testing.assert_allclose(JOINT_SPEED_LIMIT, [3.375, 3.375, 3.375, 7.0, 7.0, 7.0])


@pytest.mark.slow
def test_full_cycle_passes_every_check() -> None:
    report = run(0, perturb=False)
    assert report.passed, report.failures
    assert report.sim_s >= EPISODE_END - 1e-6
    assert report.worst_penetration_mm <= 2.0
    assert report.max_joint_speed_ratio <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize("seed", [11, 2013])  # 2013: the cover once hit the bin on its way back (D2)
def test_perturbed_cycles_pass(seed: int) -> None:
    report = run(seed, perturb=True)
    assert report.passed, report.failures
