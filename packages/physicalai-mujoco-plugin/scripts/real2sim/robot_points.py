"""Dense robot points from splat median depth over training views, in the desk-plane frame (metres, shared bucket scale):
lime pixels (the arm's lime shells) and all non-white pixels above the desk near the lime cluster. Writes align/robot_pts.npz."""
import json, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, held, pinhole_cameras
from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer

fit = json.load(open(ROOT / "align/buckets3d.json")); s = fit["s_units_per_m"]
pl = fit["plane"]; o, u, v, n = (np.array(pl[k]) for k in "ouvn")
r = MetalSplatRenderer(ROOT / "splat/v0/out/export_30000.ply")
cams = pinhole_cameras(); names = sorted(nm for nm in cams if not held(nm))[::2]
lime, other = [], []
for nm in names:
    T, K, w, h = cams[nm]; w2, h2 = w // 2, h // 2; K2 = K.copy(); K2[:2] /= 2
    out = r.render(T, K2, w2, h2); D = out["median_depth"]
    img = cv2.resize(cv2.imread(str(ROOT / "colmap/brush_all/images" / nm)), (w2, h2), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); ok = (out["alpha"] > 0.9) & (D > 0)
    is_lime = (hsv[..., 0] >= 28) & (hsv[..., 0] <= 45) & (hsv[..., 1] > 110) & (hsv[..., 2] > 90)
    white = (hsv[..., 1] < 40) & (hsv[..., 2] > 170)
    for mask, acc in ((ok & is_lime, lime), (ok & ~is_lime & ~white, other)):
        ys, xs = np.nonzero(mask); z = D[ys, xs]
        Xc = np.c_[(xs + 0.5 - K2[0, 2]) / K2[0, 0] * z, (ys + 0.5 - K2[1, 2]) / K2[1, 1] * z, z]
        Pw = (Xc - T[:3, 3]) @ T[:3, :3]
        q = np.c_[(Pw - o) @ u, (Pw - o) @ v, (Pw - o) @ n] / s
        acc.append(np.c_[q[(q[:, 2] > 0.003) & (q[:, 2] < 0.8)], img[ys, xs][(q[:, 2] > 0.003) & (q[:, 2] < 0.8)]])
L = np.concatenate(lime); O = np.concatenate(other)
c = np.median(L[:, :2], 0); print("lime", len(L), "centre m", c.round(3), "z range", np.percentile(L[:, 2], [5, 95]).round(3))
L = L[np.linalg.norm(L[:, :2] - c, axis=1) < 0.5]; O = O[np.linalg.norm(O[:, :2] - c, axis=1) < 0.55]
# drop the buckets' footprints
for b in fit["buckets"].values():
    O = O[np.linalg.norm(O[:, :2] - [b["x"], b["y"]], axis=1) > 0.13]
print("kept lime", len(L), "other", len(O))
np.savez_compressed(ROOT / "align/robot_pts.npz", lime=L, other=O)
