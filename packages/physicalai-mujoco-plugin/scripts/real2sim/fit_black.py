"""Place the black bucket: dark splat-depth points in a box past the green bucket, CAD fit at the shared scale.
Adds "black" to align/buckets3d.json."""
import json, sys, cv2, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, held, pinhole_cameras
import cad
from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer

fit = json.load(open(ROOT / "align/buckets3d.json")); s = fit["s_units_per_m"]
pl = fit["plane"]; o, u, v, n = (np.array(pl[k]) for k in "ouvn")
g, b = fit["buckets"]["green"], fit["buckets"]["blue"]
step = (np.array([g["x"], g["y"]]) - [b["x"], b["y"]]) / 2  # bucket pitch along the row
guess = np.array([g["x"], g["y"]]) + step
print("guess", guess.round(3), "pitch m", np.linalg.norm(step).round(3))
r = MetalSplatRenderer(ROOT / "splat/v0/out/export_30000.ply")
cams = pinhole_cameras(); names = sorted(nm for nm in cams if not held(nm))[::3]
pts = []
for nm in names:
    T, K, w, h = cams[nm]; w2, h2 = w // 2, h // 2; K2 = K.copy(); K2[:2] /= 2
    out = r.render(T, K2, w2, h2); D = out["median_depth"]
    img = cv2.resize(cv2.imread(str(ROOT / "colmap/brush_all/images" / nm)), (w2, h2), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); dark = (hsv[..., 2] < 70) & (out["alpha"] > 0.9) & (D > 0)
    ys, xs = np.nonzero(dark); z = D[ys, xs]
    Xc = np.c_[(xs + 0.5 - K2[0, 2]) / K2[0, 0] * z, (ys + 0.5 - K2[1, 2]) / K2[1, 1] * z, z]
    Pw = (Xc - T[:3, 3]) @ T[:3, :3]
    q = np.c_[(Pw - o) @ u, (Pw - o) @ v, (Pw - o) @ n] / s
    keep = (np.linalg.norm(q[:, :2] - guess, axis=1) < 0.13) & (q[:, 2] > 0.004) & (q[:, 2] < 0.13)
    pts.append(q[keep])
Q = np.concatenate(pts); rng = np.random.default_rng(0); Q = Q[rng.choice(len(Q), min(15000, len(Q)), replace=False)]
print("dark points", len(Q), "z p99.5 mm", round(np.percentile(Q[:, 2], 99.5) * 1000, 1))
tree = cKDTree(cad.bucket_body_points(20000))
def loss(p):
    x, y, yaw = p; c, sn = np.cos(-yaw), np.sin(-yaw)
    Qb = (Q - [x, y, 0]) @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]).T
    return np.mean(np.minimum(tree.query(Qb)[0], 0.02) ** 2)
best = min((minimize(loss, [*Q[:, :2].mean(0), y0], method="Powell") for y0 in (0, np.pi / 2, np.pi, -np.pi / 2)), key=lambda r: r.fun)
x, y, yaw = best.x; print("black", round(x, 4), round(y, 4), round(yaw, 3), "rms mm", round(np.sqrt(best.fun) * 1000, 2))
fit["buckets"]["black"] = {"x": float(x), "y": float(y), "yaw": float(yaw)}; fit["black_rms_mm"] = float(np.sqrt(best.fun) * 1000)
json.dump(fit, open(ROOT / "align/buckets3d.json", "w"), indent=1)
