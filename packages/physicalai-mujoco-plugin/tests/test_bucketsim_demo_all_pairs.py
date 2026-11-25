# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The real2sim DEMO command, run literally as the MANIFEST gives it, for every declared type x bucket (r09 D4, R-36:
it raised for every pair). Needs the bucket-v2 kit; set R2S_DELIVERY to a frozen delivery to run that delivery's own
MANIFEST command (its scripts and kit), otherwise the worktree's script with REAL2SIM_ROOT's kit."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("REAL2SIM_ROOT", Path.home() / "real2sim-data/bucket-v2")).expanduser()
KIT = ROOT / "kit/real2sim.json"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/real2sim/v2/bucketsim_v2.py"
TYPES = ("resistor", "cpu", "battery")
BUCKETS = ("black", "green", "red", "blue")

pytestmark = pytest.mark.skipif(not KIT.exists() and not os.environ.get("R2S_DELIVERY"), reason="needs the bucket-v2 real2sim kit")


def _demo_command(tmp: Path) -> str:
    seeds = ",".join(str(9101 + i) for i in range(len(TYPES) * len(BUCKETS)))
    pairs = ",".join(f"{t}:{b}" for t in TYPES for b in BUCKETS)
    out, dump = tmp / "demo.json", tmp / "dump"
    delivery = os.environ.get("R2S_DELIVERY")
    if delivery:   # the MANIFEST's DEMO row, verbatim, with only its seeds, pairs and output paths replaced
        manifest = (Path(delivery) / "MANIFEST.md").read_text()
        row = next(line for line in manifest.splitlines() if line.startswith("| DEMO |"))
        command = re.search(r"`([^`]*bucketsim_v2\.py demo [^`]*)`", row).group(1)
        command = re.sub(r"--seeds \S+", f"--seeds {seeds}", command)
        command = re.sub(r"--pairs \S+", f"--pairs {pairs}", command)
        command = re.sub(r"--out \S+", f"--out {out}", command)
        return re.sub(r"--dump \S+", f"--dump {dump}", command)
    env = f"REAL2SIM_ROOT={shlex.quote(str(ROOT))} KIT={shlex.quote(str(KIT))}"
    return (f"cd {shlex.quote(str(tmp))} && {env} {shlex.quote(sys.executable)} -I -B {shlex.quote(str(SCRIPT))} demo "
            f"--seeds {seeds} --pairs {pairs} --variant default --out {out} --dump {dump}")


def test_demo_runs_every_declared_type_and_bucket(tmp_path: Path) -> None:
    run = subprocess.run(_demo_command(tmp_path), shell=True, capture_output=True, text=True, timeout=1800, check=False)  # noqa: S602
    assert run.returncode == 0, run.stderr[-3000:]
    episodes = json.loads((tmp_path / "demo.json").read_text())
    assert sorted(e["pair"] for e in episodes) == sorted(f"{t}:{b}" for t in TYPES for b in BUCKETS)
    for e in episodes:
        kind, bucket = e["pair"].split(":")
        assert e["bucket"] == bucket
        assert e["object"].startswith(f"obj_{kind}_"), e   # the forced pair's kind is the instruction's
        assert not e["glitch"], e
    assert sum(e["success"] for e in episodes) >= 10, [(e["pair"], e["failure"]) for e in episodes if not e["success"]]
