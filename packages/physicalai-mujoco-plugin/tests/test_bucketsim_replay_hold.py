# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ROBOTCAM_REPLAY at a frame where the real gripper holds a resistor: it runs and draws the fingers where the recorded
reading puts them (r13: no replay-only drawing rule). Needs the bucket-v2 kit and the reviewer's export of the allowed
episodes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("REAL2SIM_ROOT", Path.home() / "real2sim-data/bucket-v2")).expanduser()
KIT = ROOT / "kit/real2sim.json"
STATES = ROOT / "review/export/robotcam_allowed/states_allowed.parquet"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/real2sim/v2/bucketsim_v2.py"

pytestmark = pytest.mark.skipif(not (KIT.exists() and STATES.exists()), reason="needs the bucket-v2 kit and export")


def test_replay_at_a_held_resistor(tmp_path: Path) -> None:
    env = dict(os.environ, REAL2SIM_ROOT=str(ROOT), KIT=str(KIT))
    # allowed episode 7 carries a resistor at 4.833 s (reading -49 deg, in the hold band)
    run = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "replay", "--states", str(STATES), "--times", "7:4.8333,7:0.1667",
                          "--out", str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert run.returncode == 0, run.stderr[-3000:]
    frames = json.loads((tmp_path / "replay.json").read_text())["frames"]
    assert len(frames) == 2
    held, opened = frames
    assert not any(k.endswith("_finger_opening_mm") for f in (held, opened) for k in f["state_public_used"])   # r13: no replay-only rule
    # r12 layout rule: the source episodes are recorded per bucket (episode 7: red, green, black from 6, 7, 9, 10)
    src = held["bucket_layout_from_episode"]
    assert src["red"]["allowed_pooled"] == [6, 7, 9, 10] and src["blue"]["allowed_pooled"] == [0, 1, 2, 3, 6, 7, 9, 10]


def test_layout_rule_per_bucket() -> None:
    """U-4 red (the user: "red moved on ep2 onwards and again on ep7", counting from 1): red {0}, {1-5}, {6-10};
    green and black {0-3}, {4-10}; blue one pool. Only the pools of allowed episodes are used."""
    lay = json.loads(KIT.read_text())["bucket_layouts_episodes"]["bucket_segments"]
    pooled = {c: {e: s["allowed_pooled"] for s in segs for e in s["episodes"]} for c, segs in lay.items()}
    assert pooled["red"][0] == [0] and pooled["red"][4] == pooled["red"][5] == [1, 2, 3] and pooled["red"][8] == [6, 7, 9, 10]
    assert pooled["green"][3] == pooled["black"][3] == [0, 1, 2, 3] and pooled["green"][4] == pooled["black"][8] == [6, 7, 9, 10]
    assert all(pooled["blue"][e] == [0, 1, 2, 3, 6, 7, 9, 10] for e in range(11))
    for c, segs in lay.items():   # every episode in exactly one segment, and no held-out episode pooled
        assert sorted(e for s in segs for e in s["episodes"]) == list(range(11)), c
        assert not {4, 5, 8} & {e for s in segs for e in s["allowed_pooled"]}, c
