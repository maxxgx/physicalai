"""Refit the robot's base, joints and finger opening at the chosen scale (calibration.json), starting from it.
Prints the result; --write stores joints, opening and rms into calibration.json."""
import json, sys, numpy as np
from pathlib import Path
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
import fit_robot as R, cad
cal = json.load(open(ROOT / "calibration.json")); B = json.load(open(ROOT / "align/buckets3d.json"))
k = cal["scale"]["chosen_units_per_m"] / B["s_units_per_m"]
rb = json.load(open(ROOT / "align/robot_fit.json"))
def pts(p):  # base x, y, yaw, dz, joints 1-6, finger opening (each finger, m)
    x, y, yaw, dz = p[:4]; P = (cad.arm_points(np.r_[p[4:10], p[10]], R.local) - R.B0) * k
    c, s = np.cos(yaw), np.sin(yaw); return P @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [x, y, dz]
def loss(p, T=0.02):
    P = pts(p); a = np.minimum(R.treeA.query(P)[0], T) ** 2
    b = np.minimum(R.cKDTree(P[R.is_lime]).query(R.L)[0], T) ** 2
    return a.mean() + b.mean()
best = None
for g0 in (0.0, 0.01, 0.025, 0.04):
    p0 = np.r_[rb["base_xy_m"], cal["robot"]["base_yaw_in_plane_rad"], rb["base_dz_m"], cal["robot"]["joints_rad"], g0]
    r = minimize(loss, p0, method="Powell", bounds=[(None, None)] * 10 + [(0.0, 0.05)], options={"maxiter": 12000, "xtol": 1e-6, "ftol": 1e-11})
    print("g0", g0, "rms mm", round(np.sqrt(r.fun / 2) * 1000, 3), "grip mm", round(r.x[10] * 1000, 1), "q deg", np.degrees(r.x[4:10]).round(1), flush=True)
    if best is None or r.fun < best.fun: best = r
if "--write" in sys.argv:
    cal["robot"]["joints_rad"] = best.x[4:10].tolist(); cal["robot"]["joints_deg"] = np.degrees(best.x[4:10]).round(2).tolist()
    cal["robot"]["finger_opening_m"] = float(best.x[10]); cal["robot"]["fit_rms_mm"] = float(np.sqrt(best.fun / 2) * 1000)
    json.dump(cal, open(ROOT / "calibration.json", "w"), indent=1); print("written")
