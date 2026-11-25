"""Fit the reBot B601 CAD to the splat robot points: base x, y, yaw, height offset (plate), a free scale factor k
(robot size / bucket-derived metres, the second scale reference) and joints 1-6. Writes align/robot_fit.json."""
import json, sys, numpy as np, mujoco
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
import cad

Z = np.load(ROOT / "align/robot_pts.npz"); rng = np.random.default_rng(0)
def vox(P, h=0.004):
    _, i = np.unique(np.floor(P[:, :3] / h).astype(np.int64), axis=0, return_index=True); return P[i, :3]
L = vox(Z["lime"]); A = np.vstack([L, vox(Z["other"])])
c0 = np.median(L[:, :2], 0); A = A[(np.linalg.norm(A[:, :2] - c0, axis=1) < 0.45) & (A[:, 2] < 0.6)]
L = L[rng.choice(len(L), min(6000, len(L)), replace=False)]
print("lime", len(L), "all", len(A), flush=True); treeA = cKDTree(A)
m, d = cad.model(); local = cad.geom_local_points(per_geom=250)
rgba = lambda g: m.mat_rgba[m.geom_matid[g], :3] if m.geom_matid[g] >= 0 else m.geom_rgba[g, :3]
lime_g = [g for g in local if rgba(g)[1] > 0.6 and rgba(g)[2] < 0.4 and rgba(g)[0] < rgba(g)[1]]
print("lime geoms", lime_g, flush=True)
gid = np.concatenate([[g] * len(p) for g, (_, p) in local.items()]); is_lime = np.isin(gid, lime_g)
base = m.body("base_link"); B0 = d.xpos[base.id].copy()

def model_pts(p):
    x, y, yaw, dz, logk = p[:5]; P = (cad.arm_points(np.r_[p[5:11], 0.0], local) - B0) * np.exp(logk)
    c, s = np.cos(yaw), np.sin(yaw); return P @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [x, y, dz]

def loss(p, T=0.02):
    P = model_pts(p)
    a = np.minimum(treeA.query(P)[0], T) ** 2          # every model surface point should sit on splat geometry
    b = np.minimum(cKDTree(P[is_lime]).query(L)[0], T) ** 2  # every lime splat point should sit on a lime shell
    return a.mean() + b.mean()

if __name__ == "__main__":
    starts = []
    for yaw in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        for dx in (-0.12, 0, 0.12):
            for dy in (-0.12, 0, 0.12):
                p = np.r_[c0 + [dx, dy], yaw, 0.0, 0.0, np.zeros(6)]
                starts.append((loss(p, 0.05), p))
    starts.sort(key=lambda t: t[0]); res = []
    for _, p in starts[:6]:  # coarse: base only, then everything
        r = minimize(lambda b: loss(np.r_[b, 0.0, 0.0, np.zeros(6)], 0.05), p[:4], method="Powell")
        r = minimize(loss, np.r_[r.x, 0.0, np.zeros(6)], method="Powell", options={"maxiter": 20000, "xtol": 1e-5, "ftol": 1e-10})
        res.append(r); print("rms mm", round(np.sqrt(r.fun / 2) * 1000, 2), "k", round(np.exp(r.x[4]), 4), "base", r.x[:4].round(3), "q deg", np.degrees(r.x[5:]).round(1), flush=True)
    r = min(res, key=lambda r: r.fun); p = r.x
    out = {"base_xy_m": p[:2].tolist(), "base_yaw": float(p[2]), "base_dz_m": float(p[3]), "k_robot_over_bucket": float(np.exp(p[4])),
           "joints_rad": p[5:11].tolist(), "rms_mm": float(np.sqrt(r.fun / 2) * 1000), "frame": "desk plane frame of align/buckets3d.json, metres"}
    json.dump(out, open(ROOT / "align/robot_fit.json", "w"), indent=1); print(json.dumps(out))
