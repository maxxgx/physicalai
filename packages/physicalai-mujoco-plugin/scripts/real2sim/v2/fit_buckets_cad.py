"""Each bucket's pose (x, y, yaw) by a 3D fit of the bucket CAD surface to its coloured dense points, at a given scale,
starting from align/buckets3d.json (the rectangle fit: centre, long-side direction) with both open-end directions; then
each bucket's own scale with its pose free (a scale reference). Updates align/buckets3d.json (poses, per-bucket scale,
rms).  usage: fit_buckets_cad.py S_UNITS_PER_M"""
import json, os, sys, numpy as np, cv2
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import minimize
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cad  # noqa: E402
S = float(sys.argv[1])
B = json.load(open(ROOT / "align/buckets3d.json")); pl = B["plane"]; o, u, v, n = (np.array(pl[k]) for k in "ouvn")
Z = np.load(ROOT / "align/dense_pts.npz"); sel = np.random.default_rng(4).choice(len(Z["P"]), 6_000_000, replace=False)
P, C = Z["P"][sel].astype(np.float64), Z["C"][sel]; del Z
Q = np.c_[(P - o) @ u, (P - o) @ v, (P - o) @ n] / S          # plane frame, metres at scale S
hsv = cv2.cvtColor(C.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float); H, Sat, V = hsv[:, 0], hsv[:, 1] / 255, hsv[:, 2] / 255
masks = {"green": (H >= 45) & (H <= 85) & (Sat > 0.3), "red": ((H <= 8) | (H >= 170)) & (Sat > 0.5) & (V > 0.3),
         "blue": (H >= 98) & (H <= 125) & (Sat > 0.35), "black": (V < 0.22) & (Sat < 0.6)}
model = cad.bucket_body_points(20000); tree = cKDTree(model)
rng = np.random.default_rng(0); out = {}
for col, m_ in masks.items():
    r0 = B["rect_units"][col]; c0 = np.array(r0["centre"]) / S
    q = Q[m_ & (np.hypot(Q[:, 0] - c0[0], Q[:, 1] - c0[1]) < 0.14) & (Q[:, 2] > 0.004) & (Q[:, 2] < 0.13)]
    q = q[rng.choice(len(q), min(4000, len(q)), replace=False)]
    def resid(p, s_=1.0):
        x, y, yaw = p; Qm = q / s_ - [x, y, 0]; cy, sy = np.cos(-yaw), np.sin(-yaw)
        return np.minimum(tree.query(Qm @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]).T)[0], 0.015)
    best = None
    for yaw0 in (r0["yaw_long"] - np.pi / 2, r0["yaw_long"] + np.pi / 2):
        for dyaw in (-0.15, 0.0, 0.15):
            r = minimize(lambda p: np.mean(resid(p) ** 2), [c0[0], c0[1], yaw0 + dyaw], method="Powell", options={"xtol": 1e-5, "ftol": 1e-10})
            if best is None or r.fun < best.fun: best = r
    rs = minimize(lambda p: np.mean(resid(p[1:], p[0]) ** 2), np.r_[1.0, best.x], method="Powell", options={"xtol": 1e-5, "ftol": 1e-10})
    out[col] = {"x": float(best.x[0]), "y": float(best.x[1]), "yaw": float(best.x[2]), "rms_mm": float(np.sqrt(best.fun) * 1000),
                "own_scale_factor": float(rs.x[0]), "points": int(len(q))}
    print(col, len(q), "pts; rms mm", round(np.sqrt(best.fun) * 1000, 2), "yaw deg", round(np.degrees(best.x[2]), 1),
          "own scale x", round(rs.x[0], 4), "rms", round(np.sqrt(rs.fun) * 1000, 2), flush=True)
B["s_units_per_m"] = S
B["buckets"] = {c: {k: out[c][k] for k in ("x", "y", "yaw")} for c in out}
B["per_bucket"] = {c: {"s": S * out[c]["own_scale_factor"], "rms_mm": out[c]["rms_mm"]} for c in out}
B["bucket_scale_units_per_m"] = float(np.median([S * out[c]["own_scale_factor"] for c in out]))
B["bucket_scale_spread_pct"] = float(np.std([out[c]["own_scale_factor"] for c in out]) * 100)
B["method"] = "bucket CAD surface fitted to each bucket's coloured dense points (pose at the chosen scale, then its own scale)"
json.dump(B, open(ROOT / "align/buckets3d.json", "w"), indent=1)
print("bucket scale (median of own scales)", round(B["bucket_scale_units_per_m"], 4), "spread %", round(B["bucket_scale_spread_pct"], 2))
