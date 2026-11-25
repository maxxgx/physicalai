"""Shared helpers: the project folder, held-out rule, PINHOLE cameras of the undistorted SfM.

REAL2SIM_ROOT names the project folder (video/, colmap/, splat/, align/, calibration.json, notes/, review/, deliveries/)."""
import json
import os
from pathlib import Path
import numpy as np

if not os.environ.get("REAL2SIM_ROOT"):
    raise SystemExit("Set REAL2SIM_ROOT to the project folder (it holds colmap/, splat/, align/, calibration.json)")
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser().resolve()
_HERE = Path(__file__).resolve().parent
PLUGIN = _HERE.parent / "plugin" if (_HERE.parent / "plugin").exists() else _HERE.parents[1]
"""The plugin package (src/, urdf/): a delivery's frozen copy when run from one, else this checkout."""
_HFILE = ROOT / "review/HELDOUT_PHONE.json"
_H = json.load(open(_HFILE)) if _HFILE.exists() else {"windows_s": []}

def held_t(t):
    g = _H.get("guard_s", 0.5)
    return any(t0 - g - 1e-6 <= t <= t1 + g + 1e-6 for t0, t1 in _H["windows_s"])

def held(name):
    """Window or guard frame, by name: fNNNNN with NNNNN the index at the project's naming rate (v2: round(t*30))."""
    return held_t(int(Path(name).stem[1:6]) / float(_H.get("source_fps", 15)))

def pinhole_cameras(path=ROOT / "colmap/pinhole_cameras.json"):
    """{name: (world_to_cam 4x4, K, w, h)} in the SfM frame (undistorted PINHOLE model)."""
    out = {}
    for n, c in json.load(open(path)).items():
        T = np.eye(4); T[:3] = np.array(c["cam_from_world"])
        out[n] = (T, np.array(c["K"]), c["w"], c["h"])
    return out
