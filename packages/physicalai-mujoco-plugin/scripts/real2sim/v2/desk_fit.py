"""The desk's outline and height from the dense points (allowed views), in plane metres of align/buckets3d.json at the
chosen scale S: the white points within 6 mm of the desk plane near the buckets -> largest blob on a 1 cm grid -> minimum-
area rectangle; the floor: the densest height below -0.4 m. Writes align/desk.json in the scene frame (needs the robot base:
plane-metre frame position and yaw from align/robot_fit.json and the frame scale).  usage: desk_fit.py S"""
import json, os, sys, numpy as np, cv2
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
S = float(sys.argv[1]); B = json.load(open(ROOT / "align/buckets3d.json")); sb = B["s_units_per_m"]; pl = B["plane"]
o, u, v, n = (np.array(pl[k]) for k in "ouvn")
Z = np.load(ROOT / "align/dense_pts.npz"); sel = np.random.default_rng(5).choice(len(Z["P"]), 4_000_000, replace=False)
P, C = Z["P"][sel].astype(np.float64), Z["C"][sel]
Q = np.c_[(P - o) @ u, (P - o) @ v, (P - o) @ n] / S
hsv = cv2.cvtColor(C.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
white = (hsv[:, 1] < 45) & (hsv[:, 2] > 150) & (np.abs(Q[:, 2]) < 0.006)
bc = np.mean([[b["x"] * sb / S, b["y"] * sb / S] for b in B["buckets"].values()], 0)
q = Q[white & (np.hypot(Q[:, 0] - bc[0], Q[:, 1] - bc[1]) < 1.6)][:, :2]
g = 0.01; lo = q.min(0); ij = np.floor((q - lo) / g).astype(int); occ = np.zeros(ij.max(0)[::-1] + 1, np.uint8); occ[ij[:, 1], ij[:, 0]] = 1
occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
k, lab, st, _ = cv2.connectedComponentsWithStats(occ); i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
ys, xs = np.nonzero(lab == i); pts = (np.c_[xs, ys] * g + lo + g / 2).astype(np.float32)
(cx, cy), (w, h), ang = cv2.minAreaRect(pts)
fl = Q[(Q[:, 2] < -0.4) & (Q[:, 2] > -1.2) & (np.hypot(Q[:, 0] - cx, Q[:, 1] - cy) < 2.0)][:, 2]
hist, edges = np.histogram(fl, bins=np.arange(-1.2, -0.4, 0.005)); floor = float(edges[np.argmax(hist)] + 0.0025)
R = json.load(open(ROOT / "align/robot_fit.json")); k_ = S / sb
bx, by = np.array(R["base_xy_m"]) * sb / S; yaw = R["base_yaw"]
c, s_ = np.cos(-yaw), np.sin(-yaw); Rz = np.array([[c, -s_], [s_, c]])
ctr = Rz @ (np.array([cx, cy]) - [bx, by]); a = np.radians(ang) - yaw
half = [float(w / 2), float(h / 2)]
out = {"method": "white dense points (allowed-view splat depth + photo colour) within 6 mm of the desk plane, largest blob, minimum-area rectangle; height above the floor = densest floor height below the plane",
       "center": ctr.tolist(), "half": half, "yaw": float(a), "height_m": float(-floor), "sigma_m": 0.01, "blob_cells": int(len(xs))}
json.dump(out, open(ROOT / "align/desk.json", "w"), indent=1); print(json.dumps(out))
