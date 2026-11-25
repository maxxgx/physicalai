"""Buckets from their wall points in the desk-plane frame: per bucket (hand seed, align/bucket_seeds.json), the coloured
(black: dark) dense points 1.5-11 cm above the desk within 0.5 SfM units of the seed, outliers beyond 3 MAD dropped, then
a minimum-area rectangle = the bucket's outer footprint. Scale = the CAD outer size (150 x 202.5 mm) over the measured
sides (8 sides, mean and spread). Yaw is known modulo pi here (the open end is resolved with the robot's position).
Reads align/plane_fine.npy, align/dense_pts.npz; writes align/buckets3d.json (same schema as fit_desk_buckets.py)."""
import json, os, sys, numpy as np, cv2
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
Z = np.load(ROOT / "align/dense_pts.npz"); sel = np.random.default_rng(1).choice(len(Z["P"]), 6_000_000, replace=False)
P, C = Z["P"][sel].astype(np.float64), Z["C"][sel]; del Z
pl = np.load(ROOT / "align/plane_fine.npy"); n, d = pl[:3], pl[3]
SEEDS = json.load(open(ROOT / "align/bucket_seeds.json"))["seeds_sfm"]
# orient as fit_desk_buckets.py does (buckets above the desk)
near = np.zeros(len(P), bool)
for X in SEEDS.values():
    X = np.array(X); near |= np.linalg.norm((P - X) - np.outer((P - X) @ n, n), axis=1) < 0.4
hs = P[near] @ n + d; hs = hs[np.abs(hs) > 0.04]
if np.median(hs) < 0: n, d = -n, -d
u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(n, u); o = -d * n
Q = np.c_[(P - o) @ u, (P - o) @ v, P @ n + d]
hsv = cv2.cvtColor(C.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float); H, S, V = hsv[:, 0], hsv[:, 1] / 255, hsv[:, 2] / 255
masks = {"green": (H >= 45) & (H <= 85) & (S > 0.3), "red": ((H <= 8) | (H >= 170)) & (S > 0.45),
         "blue": (H >= 98) & (H <= 125) & (S > 0.35), "black": (V < 0.25)}
s_guess = 4.0; rect = {}; sides = []
for col, m_ in masks.items():
    c0 = np.array([(np.array(SEEDS[col]) - o) @ u, (np.array(SEEDS[col]) - o) @ v])
    q = Q[m_ & (np.hypot(Q[:, 0] - c0[0], Q[:, 1] - c0[1]) < 0.75) & (Q[:, 2] > 0.015 * s_guess) & (Q[:, 2] < 0.11 * s_guess)]
    # the bucket: the connected blob of its colour on a 4 mm occupancy grid (closed), the one nearest the seed
    g = 0.016; lo = c0 - 0.8; N = int(1.6 / g)
    ij = np.floor((q[:, :2] - lo) / g).astype(int); ok = (ij >= 0).all(1) & (ij < N).all(1)
    occ = np.zeros((N, N), np.uint8); np.add.at(occ, (ij[ok, 1], ij[ok, 0]), 1)
    occ = (occ >= 2).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    k, lab, stats, cents = cv2.connectedComponentsWithStats(occ)
    big = [i for i in range(1, k) if stats[i, cv2.CC_STAT_AREA] > 150]
    best = min(big, key=lambda i: np.hypot(*(cents[i] * g + lo - c0)))
    ys, xs = np.nonzero(lab == best)
    pts = np.c_[xs, ys].astype(np.float32) * g + lo + g / 2
    q = pts
    (cx, cy), (w, h), ang = cv2.minAreaRect(q.astype(np.float32))
    short, long_ = sorted((w, h)); yaw_long = np.radians(ang) + (0 if h >= w else np.pi / 2)   # direction of the long side
    rect[col] = {"centre": [float(cx), float(cy)], "short": float(short), "long": float(long_), "yaw_long": float(yaw_long), "points": int(len(q))}
    sides += [short / 0.150, long_ / 0.2025]
    print(col, len(q), "points; sides (units)", round(short, 4), round(long_, 4), "-> s", round(short / 0.150, 3), round(long_ / 0.2025, 3), flush=True)
s = float(os.environ.get("S_OVERRIDE", np.median(sides))); sd = float(np.std(sides) / np.median(sides) * 100)
print("scale", round(s, 4), "units/m; spread", round(sd, 2), "%", flush=True)
buckets = {}
for col, r_ in rect.items():
    # the CAD bucket's long axis is its +y; yaw is the turn that puts +y along the measured long side (mod pi for now)
    buckets[col] = {"x": r_["centre"][0] / s, "y": r_["centre"][1] / s, "yaw": float(r_["yaw_long"] - np.pi / 2), "yaw_mod_pi": True}
json.dump({"s_units_per_m": s, "scale_spread_pct": sd, "sides_s": sides, "rect_units": rect, "method": "min-area rectangle of wall points, CAD outer size",
           "rms_mm": None, "per_bucket": {c: {"s": float(np.mean([rect[c]["short"] / 0.150, rect[c]["long"] / 0.2025])), "rms_mm": None} for c in rect},
           "plane": {"o": o.tolist(), "u": u.tolist(), "v": v.tolist(), "n": n.tolist()}, "buckets": buckets},
          open(ROOT / "align/buckets3d.json", "w"), indent=1)
# robot points for fit_robot.py (plane metres)
Qm = Q / s
lime = (H >= 24) & (H <= 42) & (S > 0.45) & (V > 0.35) & (Qm[:, 2] > 0.003) & (Qm[:, 2] < 0.8)
Lp = Qm[lime]; c0 = np.median(Lp[:, :2], 0)
for _ in range(3):
    dd = np.linalg.norm(Lp[:, :2] - c0, axis=1); c0 = np.median(Lp[dd < np.percentile(dd, 60), :2], 0)
Lp = Lp[np.linalg.norm(Lp[:, :2] - c0, axis=1) < 0.5]
other = ~((S < 0.15) & (V > 0.65)) & (Qm[:, 2] > 0.003) & (Qm[:, 2] < 0.8) & ~lime
Op = Qm[other]; Op = Op[np.linalg.norm(Op[:, :2] - c0, axis=1) < 0.55]
for b in buckets.values():
    Op = Op[np.linalg.norm(Op[:, :2] - [b["x"], b["y"]], axis=1) > 0.13]
def vox(A, h=0.004):
    _, i = np.unique(np.floor(A / h).astype(np.int64), axis=0, return_index=True); return A[i]
Lp, Op = vox(Lp), vox(Op)
np.savez_compressed(ROOT / "align/robot_pts.npz", lime=np.c_[Lp, np.zeros((len(Lp), 3))], other=np.c_[Op, np.zeros((len(Op), 3))])
print("robot points: lime", len(Lp), "other", len(Op), "lime centre (m)", c0.round(3), flush=True)
