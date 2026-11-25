"""The builder's entry to the plugin's desk compositor (physicalai_mujoco_plugin.real2sim.desk_compositor) with this
project's asset paths: a delivery folder's own copies first, else the working files."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from physicalai_mujoco_plugin.real2sim import desk_compositor as _dc
from physicalai_mujoco_plugin.real2sim.desk_compositor import (  # noqa: F401
    COLORS, LABEL_BUCKET, LABEL_OBJECT, LABEL_OVERVIEW, LABEL_ROBOT, OUTLINE, Calibration, DeskKit, apply_look, setup_lighting,
)

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402
HERE = Path(__file__).resolve().parent.parent   # a delivery folder when run from one: its own copies come first


def asset(name: str, live: Path) -> Path:
    return HERE / name if (HERE / name).exists() else live


SPLAT = Path(os.environ.get("BUCKET_SPLAT", asset("splat.ply", ROOT / "splat/v1c/out/export_30000.ply")))
CALIBRATION = asset("calibration.json", ROOT / "calibration.json")


def default_kit() -> DeskKit:
    return DeskKit(splat=SPLAT, calibration=CALIBRATION, look=asset("robot_look.json", ROOT / "align/robot_look.json"),
                   robot_gs=Path(os.environ.get("BUCKET_ROBOT_GS", asset("robot_gs.ply", ROOT / "align/robot_gs.ply"))))


class Compositor(_dc.Compositor):
    """The plugin's compositor with this project's kit (the calibration argument is kept for the callers)."""

    def __init__(self, m, d, cal=None, *, width: int, height: int, show_objects: bool = True, robot_splat: bool = True) -> None:
        super().__init__(m, d, default_kit(), width=width, height=height, show_objects=show_objects, robot_splat=robot_splat)


def calibration() -> Calibration:
    return Calibration(CALIBRATION)
