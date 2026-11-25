# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FIT_LAYOUT (r12, U-5): on an allowed episode's overview video it reproduces the builder's stored per-episode fit, and
ROBOTCAM_REPLAY --layout FILE with the default rule's poses renders exactly as --layout episode. Allowed episodes only;
needs the bucket-v2 kit and the reviewer's export of the allowed episodes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("REAL2SIM_ROOT", Path.home() / "real2sim-data/bucket-v2")).expanduser()
KIT = ROOT / "kit/real2sim.json"
EXPORT = ROOT / "review/export/robotcam_allowed"
STATES = EXPORT / "states_allowed.parquet"
STORED = ROOT / "align/bucket_layouts_episodes.json"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/real2sim/v2/bucketsim_v2.py"

pytestmark = pytest.mark.skipif(not (KIT.exists() and STATES.exists() and STORED.exists()), reason="needs the bucket-v2 kit and export")


def _run(tmp_path: Path, *args: str) -> None:
    env = dict(os.environ, REAL2SIM_ROOT=str(ROOT), KIT=str(KIT), PHYSICALAI_PLAN_PROCESS="0")
    run = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *args], cwd=tmp_path, env=env, capture_output=True,
                         text=True, timeout=900, check=False)
    assert run.returncode == 0, run.stderr[-3000:]


def test_fit_layout_reproduces_the_stored_fit(tmp_path: Path) -> None:
    out = tmp_path / "e7.json"
    _run(tmp_path, "fit_layout", "--video", str(EXPORT / "ep007_overview.mp4"), "--t0", "0", "--t1", "100",
         "--states", str(STATES), "--episode", "7", "--out", str(out))
    fit, stored = json.loads(out.read_text()), json.loads(STORED.read_text())["episodes"]["7"]
    assert fit["poses"] == stored["poses"] and fit["iou_fitted"] == stored["iou_fitted"]
    assert [f"overview_e007_t{f['episode_time_ms']:07d}.png" for f in fit["frame_list"]] == stored["frame_list"]
    assert (tmp_path / "e7_overlay.jpg").stat().st_size > 0


def test_replay_layout_file_equals_the_default(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPT.parent))
    os.environ.setdefault("KIT", str(KIT))
    import bucketsim_v2 as b

    _, poses = b.episode_layout(7)
    lay = tmp_path / "default_e7.json"
    lay.write_text(json.dumps({"poses": poses}))
    for name, layout in (("episode", "episode"), ("file", str(lay))):
        _run(tmp_path, "replay", "--states", str(STATES), "--times", "7:2.0", "--layout", layout, "--out", str(tmp_path / name))
    for cam in ("overview", "gripper"):
        a = (tmp_path / "episode" / f"007_002.0000_{cam}.png").read_bytes()
        assert a == (tmp_path / "file" / f"007_002.0000_{cam}.png").read_bytes(), cam
    rec = json.loads((tmp_path / "file" / "replay.json").read_text())
    assert rec["layout"].startswith("file ") and rec["frames"][0]["bucket_layout_from_episode"]["file"] == str(lay.resolve())
    assert "statement rule" in json.loads((tmp_path / "episode" / "replay.json").read_text())["layout"]
