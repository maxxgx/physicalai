"""The review's held-out rule (REAL2SIM_ROOT/review/HELDOUT_PHONE.json): names fNNNNN with NNNNN = round(t*30).
held(name) is True for window AND guard frames: filter every preview, sheet, fit and training set with it."""
import json, os
from pathlib import Path
ROOT = Path(os.environ.get("REAL2SIM_ROOT", "~/real2sim-data/bucket-v2")).expanduser().resolve()
_HERE = Path(__file__).resolve().parent
_SRC = next(p for p in (_HERE / "HELDOUT_PHONE.json", _HERE.parent / "HELDOUT_PHONE.json", ROOT / "review/HELDOUT_PHONE.json") if p.exists())
_H = json.load(open(_SRC))   # a delivery carries its own copy (never the review folder's)
_W = [tuple(w) for w in _H["windows_s"]]; _G = float(_H["guard_s"]); _FPS = float(_H["source_fps"]); EPS = 1e-6
def in_window(t): return any(a - EPS <= t <= b + EPS for a, b in _W)
def in_heldout(t): return any(a - _G - EPS <= t <= b + _G + EPS for a, b in _W)
def t_of(name): return int(Path(name).stem[1:6]) / _FPS
def held(name): return in_heldout(t_of(name))
def allowed(names): return [n for n in names if not held(n)]
