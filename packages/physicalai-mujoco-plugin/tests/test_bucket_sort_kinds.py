# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Kind-level bucket-sort episodes with the real objects (TaskLayout.kind_counts; the reBot v2 desk)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from physicalai_mujoco_plugin.bucket_sort import (
    BUCKET_INTERIOR,
    CONTAINER_NOUNS,
    KIND_NAMES,
    REAL_CONTAINER_NOUNS,
    SCATTER_MOST,
    TaskLayout,
    container_noun,
    draw_kind_pair,
    word_kind_instruction,
)
from physicalai_mujoco_plugin.bucket_sort_arms import REBOT_DESK_V2_DEFAULT, REBOT_DESK_V2_LAYOUT
from physicalai_mujoco_plugin.bucket_sort_objects import KINDS, OBJECT_COLORS, real_kind

COLORS = ("red", "blue", "green", "black", "orange", "grey")


def _names(kind: str) -> set[str]:
    singular, plural = KIND_NAMES[kind]
    return {n for n, _ in singular} | {n for n, _ in plural}


@pytest.mark.parametrize("kind", ["resistor", "cpu", "battery"])
def test_kind_instruction_names_the_kind_and_the_container_only(kind: str) -> None:
    rng = np.random.default_rng(0)
    object_words = {w for kind_ in KINDS.values() for noun, _ in kind_.nouns for w in noun.split()}
    for count in (1, 2, 4):
        for color in COLORS:
            for _ in range(40):
                text, canonical = word_kind_instruction(rng, kind, count, color)
                low = text.lower()
                assert any(name.lower() in low for name in _names(kind)), text
                # the colour appears once, naming the container ("red bin") or "the red one"
                assert len(re.findall(rf"\b{color}\b", low)) == 1, text
                assert re.search(rf"\b{color} (bin|bucket|box|one)\b", low), text
                # no colour word for an object, no generic shape noun
                for other in OBJECT_COLORS:
                    if other != color:
                        assert not re.search(rf"\b{other}\b", low), text
                assert not (set(re.findall(r"[a-z]+", low)) & (object_words - {"bar", "plus"})), text
                assert canonical.startswith("put the ") and color in canonical


def test_plural_agrees_and_only_with_several_targets() -> None:
    rng = np.random.default_rng(1)
    plurals = {n for _, plural in KIND_NAMES.values() for n, _ in plural}
    seen_plural = 0
    for _ in range(400):
        text, _ = word_kind_instruction(rng, "resistor", 1, "red")
        assert "resistors" not in text
        text, _ = word_kind_instruction(rng, "cpu", 3, "blue")
        if any(re.search(rf"\b{re.escape(p)}\b", text) for p in plurals):
            seen_plural += 1
            assert not re.search(r"\b(it|goes|belongs)\b", text), text
    assert 0.6 < seen_plural / 400 < 0.9   # KIND_PHRASES.plural = 0.75


def test_real_container_nouns_mostly() -> None:
    rng = np.random.default_rng(2)
    for color, real in REAL_CONTAINER_NOUNS.items():
        nouns = Counter(container_noun(rng, color) for _ in range(2000))
        assert set(nouns) <= set(CONTAINER_NOUNS)
        assert 0.8 < nouns[real] / 2000 < 0.9


def test_real_pairs_half_the_time() -> None:
    rng = np.random.default_rng(3)
    allowed = tuple((k, c) for k in ("resistor", "cpu", "battery") for c in ("black", "blue", "red", "green"))
    picks = [draw_kind_pair(rng, allowed, REBOT_DESK_V2_LAYOUT) for _ in range(4000)]
    real = sum(p in REBOT_DESK_V2_LAYOUT.real_pairs for p in picks) / len(picks)
    assert 0.45 < real < 0.55
    assert len(set(picks)) == 12
    assert {p[0] for p in (draw_kind_pair(rng, allowed, REBOT_DESK_V2_LAYOUT, ("cpu",)) for _ in range(50))} == {"cpu"}


def test_novel_targets_refused_until_assets_exist() -> None:
    assert REBOT_DESK_V2_LAYOUT.novel_target_fraction == 0.0
    with pytest.raises(ValueError, match="novel"):
        TaskLayout(novel_target_fraction=0.1)
    with pytest.raises(ValueError, match="novel"):
        replace(REBOT_DESK_V2_DEFAULT, novel_target_fraction=0.2)


@pytest.fixture(scope="module")
def desk():
    from physicalai_mujoco_plugin.robot import MuJoCoRobot

    robot = MuJoCoRobot("rebot_b601", scene="bucket_sort_desk_v2", cameras=[], http_port=0, seed=0)
    robot.connect()
    robot._automation.set_autopilot("off")  # noqa: SLF001
    yield robot
    robot.disconnect()


def _reset(robot, seed: int):
    robot._set_seed(seed)  # noqa: SLF001
    robot._reset()  # noqa: SLF001
    robot._set_seed(None)  # noqa: SLF001
    return robot._sim.on_reset  # noqa: SLF001


def test_desk_v2_resets_lay_out_real_objects_and_target_every_one_of_the_kind(desk) -> None:
    piles = upright = resistors = 0
    for seed in range(60):
        task = _reset(desk, 7000 + seed)
        ins, layout = task.instruction, task.layout
        kinds = Counter(real_kind(n) for n in layout.objects)
        assert None not in kinds, "generic shapes are off by default (distractors 0)"
        if layout.pile:
            assert 1 <= kinds["resistor"] <= 4 and 1 <= kinds["cpu"] <= 3 and kinds["battery"] <= 2
        else:   # a real-like scattered table: one to five objects
            assert kinds["resistor"] <= 3 and kinds["cpu"] <= 2 and kinds["battery"] <= 1
            assert 1 <= len(layout.objects) <= 5
        for name in layout.upright:
            assert real_kind(name) == "resistor" and name not in layout.heights
        assert ins.kind in kinds and kinds[ins.kind] >= 1
        assert set(ins.targets) == {n for n in layout.objects if real_kind(n) == ins.kind}
        assert ins.object in ins.targets
        piles += layout.pile
        upright += len(layout.upright); resistors += kinds["resistor"]
    assert 0.15 < piles / 60 < 0.45   # REBOT_DESK_V2: pile_probability 0.3
    assert 0.08 < upright / resistors < 0.40   # about one resistor in five stands on its end


def test_kind_success_needs_every_target(desk) -> None:
    import mujoco

    task = _reset(desk, 7101)
    for seed in range(7102, 7200):
        if len(task.instruction.targets) >= 2:
            break
        task = _reset(desk, seed)
    m, d = desk._sim.model, desk._sim.data  # noqa: SLF001
    ins = task.instruction
    assert len(ins.targets) >= 2
    pose = task.layout.buckets[ins.bucket]

    def into_bucket(name: str, k: int) -> None:
        adr = m.jnt_qposadr[m.joint(f"{name}:joint").id]
        dof = m.jnt_dofadr[m.joint(f"{name}:joint").id]
        c, s = np.cos(pose.yaw), np.sin(pose.yaw)
        dx, dy = 0.0, (k - 1) * 0.04
        d.qpos[adr : adr + 3] = (pose.x + c * dx - s * dy, pose.y + s * dx + c * dy, BUCKET_INTERIOR[2][0] + 0.03)
        d.qvel[dof : dof + 6] = 0.0

    for k, name in enumerate(ins.targets[:-1]):
        into_bucket(name, k)
    mujoco.mj_forward(m, d)
    assert not task.evaluate(m, d).success
    into_bucket(ins.targets[-1], len(ins.targets) - 1)
    mujoco.mj_forward(m, d)
    state = task.evaluate(m, d)
    assert all(state.in_bucket[n] == ins.bucket for n in ins.targets)
    assert state.success


def test_stratified_retry_keeps_the_stratum_after_a_failure_only(desk) -> None:
    task = desk._sim.on_reset  # noqa: SLF001
    saved = task.area
    task.area = replace(REBOT_DESK_V2_LAYOUT, sequence_probability=0.0)
    try:
        assert task.area.stratified_retry
        _reset(desk, 9000)
        for seed in range(9001, 9013):   # the demonstrator failed each one
            before = (task.layout.pile, task.instruction.kind, len(task.instruction.targets))
            task.note_failure()
            _reset(desk, seed)
            assert task.stratum_retry
            assert (task.layout.pile, task.instruction.kind, len(task.instruction.targets)) == before
        strata = set()
        for seed in range(9100, 9130):   # after a success or a plain reset the next layout draws a fresh stratum
            if seed % 2:
                task.offer_continue()
            _reset(desk, seed)
            assert not task.stratum_retry
            strata.add((task.layout.pile, task.instruction.kind, len(task.instruction.targets)))
        assert len(strata) > 3
    finally:
        task.area = saved


@pytest.mark.parametrize("kind", ["resistor", "cpu", "battery"])
@pytest.mark.parametrize("bucket", ["black", "green", "red", "blue"])
def test_a_forced_pair_names_that_kind_and_bucket(desk, kind: str, bucket: str) -> None:
    """DEMO's way of forcing a pair (r09 D4: it passed a body where the kind-level task wants a kind, and raised)."""
    task = desk._sim.on_reset  # noqa: SLF001
    body = next(n for n in task.objects if real_kind(n) == kind)
    task.required = body
    task.set_allowed_pairs([(kind, bucket)])
    try:
        for seed in (1, 2):
            task = _reset(desk, 7300 + seed)
            assert (task.instruction.kind, task.instruction.bucket) == (kind, bucket)
            assert real_kind(task.instruction.object) == kind
    finally:
        task.required = None
        task.set_allowed_pairs(None)
