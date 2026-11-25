"""Desk plane and the four buckets from dense coloured points (align/dense_pts.npz: splat depth + photo colour; SfM frame), with the bucket CAD and one shared
scale. Also writes the robot's points (lime shells and other non-white points near them) for fit_robot.py.

1. Desk plane: RANSAC on white Gaussians (S < 0.15, V > 0.65), refined by least squares; normal toward the buckets.
2. Bucket points: by colour above the desk (green, red, blue saturated; black dark, near the row between red and blue).
3. CAD fit: the bucket mesh surface (cad.bucket_body_points), shared scale s (SfM units per metre), per bucket x, y, yaw;
   robust (truncated at 2 cm) data-to-model distances; several starts.
Writes align/plane_fine.npy, align/buckets3d.json (v1 format, 4 buckets), align/robot_pts.npz (v1 format, plane metres).
usage: fit_desk_buckets.py SPLAT.ply [ORDER]   ORDER: the row's colours left to right as the phone sees them from the
front edge (default green,red,black,blue)"""
import json, os, sys, numpy as np, cv2
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import minimize
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cad  # noqa: E402
from physicalai_mujoco_plugin.real2sim.splats import read_ply  # noqa: E402

Z = np.load(ROOT / "align/dense_pts.npz"); sel = np.random.default_rng(1).choice(len(Z["P"]), 3_000_000, replace=False)
P0, C0 = Z["P"][sel].astype(np.float64), Z["C"][sel]; del Z
_, first = np.unique(np.floor(P0 / 0.012).astype(np.int64), axis=0, return_index=True)   # ~3 mm voxels (SfM units ~4/m)
P = P0[first]; C = C0[first].astype(np.float64) / 255
print("dense points", len(P0), "-> voxels", len(P), flush=True)
hsv = cv2.cvtColor((C.reshape(-1, 1, 3) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float); H, S, V = hsv[:, 0], hsv[:, 1] / 255, hsv[:, 2] / 255
rng = np.random.default_rng(0)
# the desk: white points within 3 SfM units (about 0.8 m) of the buckets' saturated colours (the ceiling and walls are
# white too, and far from them)
sat = ((((H >= 45) & (H <= 85)) | (H <= 7) | (H >= 172) | ((H >= 100) & (H <= 122))) & (S > 0.55) & (V > 0.3))
core = np.median(P[sat], 0)
for _ in range(3):
    dd = np.linalg.norm(P[sat] - core, axis=1); core = np.median(P[sat][dd < np.percentile(dd, 40)], 0)
white = np.flatnonzero((S < 0.15) & (V > 0.65) & (np.linalg.norm(P - core, axis=1) < 3.0))
white = white[rng.choice(len(white), min(50000, len(white)), replace=False)]
print("white points", len(white), flush=True)
thr = 0.015   # SfM units (~4 mm): dense depth points are noisy at that level
best = (0, None)
PW = P[white]
for _ in range(1500):
    a, b, c = PW[rng.integers(0, len(PW), 3)]
    n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
    if nn < 1e-9: continue
    n /= nn; d = -n @ a; inl = int((np.abs(PW @ n + d) < thr).sum())
    if inl > best[0]: best = (inl, (n, d))
n, d = best[1]
inl = white[np.abs(P[white] @ n + d) < thr]
cen = P[inl].mean(0); _, _, vt = np.linalg.svd(P[inl] - cen); n = vt[2]; d = -n @ cen
# the normal points to the side holding the coloured buckets
# (oriented by the points around the hand-placed bucket seeds: the buckets stand on the desk; the wooden floor below
# the desk is saturated too, so "the colourful side" would point the wrong way)
SEEDS = json.load(open(ROOT / "align/bucket_seeds.json"))["seeds_sfm"]
near_seed = np.zeros(len(P), bool)
for X in SEEDS.values():
    X = np.array(X); near_seed |= np.linalg.norm((P - X) - np.outer((P - X) @ n, n), axis=1) < 0.4
hs = P[near_seed] @ n + d; hs = hs[np.abs(hs) > 0.04]
if np.median(hs) < 0: n, d = -n, -d
print("desk plane inliers", len(inl), "of", len(white), "white", flush=True)
np.save(ROOT / "align/plane_fine.npy", np.r_[n, d])
u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(n, u); o = -d * n
Q = np.c_[(P - o) @ u, (P - o) @ v, P @ n + d]   # plane frame, SfM units; z = height above the desk
masks = {"green": (H >= 45) & (H <= 85) & (S > 0.35) & (V > 0.2), "red": ((H <= 7) | (H >= 172)) & (S > 0.5) & (V > 0.25),
         "blue": (H >= 100) & (H <= 122) & (S > 0.5) & (V > 0.2)}
cent = {c_: np.array([(np.array(X) - o) @ u, (np.array(X) - o) @ v]) for c_, X in SEEDS.items()}
s0 = float(np.mean([np.linalg.norm(cent[a_] - cent[b_]) for a_, b_ in (("green", "red"), ("red", "black"), ("black", "blue"))]) / 0.19)
print("bucket seeds (plane units)", {k: v_.round(3).tolist() for k, v_ in cent.items()}, "scale guess", round(s0, 3), flush=True)
data = {}
for c_ in ("green", "red", "blue", "black"):
    if c_ == "black":
        m_ = (V < 0.2) & (S < 0.5)
    else:
        m_ = masks[c_]
    q = Q[m_]; dd = np.linalg.norm(q[:, :2] - cent[c_], axis=1)
    q = q[(dd < 0.13 * s0) & (q[:, 2] > 0.004 * s0) & (q[:, 2] < 0.13 * s0)]
    q = q[rng.choice(len(q), min(5000, len(q)), replace=False)]
    data[c_] = q; print(c_, len(q), "points", flush=True)
model = cad.bucket_body_points(20000); tree = cKDTree(model)
COLS = ("green", "red", "black", "blue")

def resid(c_, qq, sc):
    x, y, yaw = qq; Qm = data[c_] / sc - [x, y, 0]; cy, sy = np.cos(-yaw), np.sin(-yaw)
    return np.minimum(tree.query(Qm @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]).T)[0], 0.02)

def loss(p, cols=COLS):
    pen = 1e3 * max(0.0, abs(np.log(p[0] / s0)) - 0.35) ** 2   # keep the scale within 30 % of the guess
    return pen + sum(np.mean(resid(c_, p[1 + 3 * i:4 + 3 * i], p[0]) ** 2) for i, c_ in enumerate(cols))

row = cent["blue"] - cent["green"]; row_yaw = np.arctan2(row[1], row[0])
res = []
for f in (0.95, 1.05):
    for yaw0 in (row_yaw + np.pi / 2, row_yaw - np.pi / 2):
        for jit in (0.0,):
            p0 = [s0 * f]
            for c_ in COLS:
                m_ = np.median(data[c_][:, :2], 0) / (s0 * f); p0 += [m_[0], m_[1], yaw0 + jit * (c_ == "green")]
            r = minimize(loss, np.array(p0), method="Powell", options={"maxiter": 4000, "xtol": 1e-5, "ftol": 1e-9})
            res.append(r); print("start", f, round(np.degrees(yaw0)), jit, "rms mm", round(np.sqrt(r.fun / 4) * 1000, 2), "s", round(r.x[0], 4), flush=True)
r = min(res, key=lambda r: r.fun); p = r.x
per = {}
for i, c_ in enumerate(COLS):
    q0 = np.r_[p[0], p[1 + 3 * i:4 + 3 * i]]
    rr = minimize(lambda q: loss(np.r_[q[0], q[1:]], (c_,)), q0, method="Powell", options={"xtol": 1e-7, "ftol": 1e-11})
    per[c_] = {"s": float(rr.x[0]), "rms_mm": float(np.sqrt(rr.fun) * 1000)}
out = {"s_units_per_m": float(p[0]), "per_bucket": per, "rms_mm": float(np.sqrt(r.fun / 4) * 1000),
       "per_bucket_s_sd_pct": float(np.std([v_["s"] for v_ in per.values()]) / p[0] * 100),
       "plane": {"o": o.tolist(), "u": u.tolist(), "v": v.tolist(), "n": n.tolist()},
       "buckets": {c_: dict(zip(("x", "y", "yaw"), map(float, p[1 + 3 * i:4 + 3 * i]))) for i, c_ in enumerate(COLS)}}
json.dump(out, open(ROOT / "align/buckets3d.json", "w"), indent=1); print(json.dumps(out, indent=1)[:1500], flush=True)

# robot points (plane metres) for fit_robot.py: lime shells, and other non-white points above the desk near them
sc = p[0]; Qm = Q / sc
lime = (H >= 24) & (H <= 42) & (S > 0.45) & (V > 0.35) & (Qm[:, 2] > 0.003) & (Qm[:, 2] < 0.8)
Lp = Qm[lime]; c0 = np.median(Lp[:, :2], 0)
for _ in range(3):
    dd = np.linalg.norm(Lp[:, :2] - c0, axis=1); c0 = np.median(Lp[dd < np.percentile(dd, 60), :2], 0)
Lp = Lp[np.linalg.norm(Lp[:, :2] - c0, axis=1) < 0.5]
other = ~((S < 0.15) & (V > 0.65)) & (Qm[:, 2] > 0.003) & (Qm[:, 2] < 0.8) & ~lime
Op = Qm[other]; Op = Op[np.linalg.norm(Op[:, :2] - c0, axis=1) < 0.55]
for b in out["buckets"].values():
    Op = Op[np.linalg.norm(Op[:, :2] - [b["x"], b["y"]], axis=1) > 0.13]
cl, co = C[lime] * 255, C[other] * 255
np.savez_compressed(ROOT / "align/robot_pts.npz", lime=np.c_[Lp, np.zeros((len(Lp), 3))], other=np.c_[Op, np.zeros((len(Op), 3))])
print("robot points: lime", len(Lp), "other", len(Op), "lime centre (m)", c0.round(3), flush=True)
