"""3D fit of the bucket CAD to dense bucket points (splat depth at bucket-coloured pixels), shared scale.
Robust data->model distance (Huber-like, truncated at 2 cm). Writes align/buckets3d.json."""
import json, sys, numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import ROOT
import cad

pl = np.load(ROOT / "align/plane_fine.npy"); n, d = pl[:3], pl[3]
u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(n, u); o = -d * n
Z = np.load(ROOT / "align/bucket_pts.npz"); rng = np.random.default_rng(0)
data = {}
for c in ("green", "red", "blue"):
    P = Z[c]; uv = np.c_[(P - o) @ u, (P - o) @ v]; med = np.median(uv, 0)
    P = P[np.linalg.norm(uv - med, axis=1) < 0.5]; P = P[rng.choice(len(P), min(15000, len(P)), replace=False)]
    data[c] = np.c_[(P - o) @ u, (P - o) @ v, (P - o) @ n]  # plane frame, SfM units
model = cad.bucket_body_points(20000); tree = cKDTree(model)

def resid(c, q, s):
    x, y, yaw = q; Q = data[c] / s - [x, y, 0]; cy, sy = np.cos(-yaw), np.sin(-yaw)
    Qb = Q @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]).T
    dist, _ = tree.query(Qb); return np.minimum(dist, 0.02)

def loss(p, cols=("green", "red", "blue")):
    s = p[0]; return sum(np.mean(resid(c, p[1 + 3 * i:4 + 3 * i], s) ** 2) for i, c in enumerate(cols))

if __name__ == "__main__":
    best = None
    for s0 in (3.1, 3.25, 3.4):
        for yaw0 in (0, np.pi / 2, np.pi, -np.pi / 2):
            p0 = [s0]
            for c in ("green", "red", "blue"):
                m = data[c][:, :2].mean(0) / s0; p0 += [m[0], m[1], yaw0]
            r = minimize(loss, np.array(p0), method="Powell", options={"maxiter": 8000, "xtol": 1e-5, "ftol": 1e-9})
            if best is None or r.fun < best.fun: best = r
            print(s0, round(yaw0, 2), "rms mm", round(np.sqrt(r.fun / 3) * 1000, 2), "s", round(r.x[0], 4), flush=True)
    p = best.x
    # per-bucket scale check: refit each bucket alone with its own scale
    per = {}
    for i, c in enumerate(("green", "red", "blue")):
        q0 = np.r_[p[0], p[1 + 3 * i:4 + 3 * i]]
        r = minimize(lambda q: loss(np.r_[q[0], q[1:]], (c,)), q0, method="Powell", options={"xtol": 1e-6, "ftol": 1e-10})
        per[c] = float(r.x[0])
    out = {"s_units_per_m": float(p[0]), "per_bucket_s": per, "rms_mm": float(np.sqrt(best.fun / 3) * 1000),
           "plane": {"o": o.tolist(), "u": u.tolist(), "v": v.tolist(), "n": n.tolist()},
           "buckets": {c: dict(zip(("x", "y", "yaw"), map(float, p[1 + 3 * i:4 + 3 * i]))) for i, c in enumerate(("green", "red", "blue"))}}
    json.dump(out, open(ROOT / "align/buckets3d.json", "w"), indent=1); print(json.dumps(out, indent=1)[:900])
