# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""reBot B601 joint zeros (real2sim r13): a calibration of ONE arm, applied only when PHYSICALAI_REBOT_B601_JOINT_ZERO
names a file holding them; unset, the profile maps readings to the model exactly as before."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROBE = """
import json
from physicalai_mujoco_plugin.profiles.rebot_b601 import REBOT_B601_PROFILE
print(json.dumps({c.name: [c.scale, c.offset] for c in REBOT_B601_PROFILE.channels}))
"""


def _profile(env: dict[str, str]) -> dict:
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_off_by_default_leaves_the_profile_unchanged() -> None:
    env = {k: v for k, v in os.environ.items() if k != "PHYSICALAI_REBOT_B601_JOINT_ZERO"}
    channels = _profile(env)
    assert all(offset == 0.0 for name, (_, offset) in channels.items() if name != "gripper")


def test_a_unit_file_sets_the_zeros(tmp_path: Path) -> None:
    unit = tmp_path / "kit.json"
    unit.write_text(json.dumps({"rebot_b601_joint_zero_deg": {"elbow_flex": 3.924, "wrist_flex": -4.957, "fitted_from": "test"}}))
    env = {**{k: v for k, v in os.environ.items() if k != "PHYSICALAI_REBOT_B601_JOINT_ZERO"}, "PHYSICALAI_REBOT_B601_JOINT_ZERO": str(unit)}
    off = _profile({k: v for k, v in os.environ.items() if k != "PHYSICALAI_REBOT_B601_JOINT_ZERO"})
    on = _profile(env)
    assert on["elbow_flex"][1] == -3.924 and on["wrist_flex"][1] == 4.957   # model = reading + zero
    assert {k: v for k, v in on.items() if k not in ("elbow_flex", "wrist_flex")} == {k: v for k, v in off.items() if k not in ("elbow_flex", "wrist_flex")}
