"""Project the fitted robot (align/robot_fit.json) and buckets onto training frames; writes align/robot_check.jpg."""
import json, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, held, pinhole_cameras
import fit_robot as R
from overlay_buckets import bucket_world
import cad

fit = json.load(open(ROOT / "align/buckets3d.json")); s = fit["s_units_per_m"]
pl = fit["plane"]; o, u, v, n = (np.array(pl[k]) for k in "ouvn")
rb = json.load(open(ROOT / "align/robot_fit.json"))
p = np.r_[rb["base_xy_m"], rb["base_yaw"], rb["base_dz_m"], np.log(rb["k_robot_over_bucket"]), rb["joints_rad"]]
Q = R.model_pts(p); lime = R.is_lime
W = o + s * (Q[:, :1] * u + Q[:, 1:2] * v + Q[:, 2:3] * n)
cams = pinhole_cameras(); names = sorted(nm for nm in cams if not held(nm))
pick = [names[int(i)] for i in np.linspace(0, len(names) - 1, 6)]
bpts = cad.bucket_body_points(3000)
tiles = []
for nm in pick:
    T, K, w, h = cams[nm]; img = cv2.imread(str(ROOT / "colmap/brush_all/images" / nm))
    def draw(P, col):
        X = P @ T[:3, :3].T + T[:3, 3]; m = X[:, 2] > 0; uv = (X[m] @ K.T)[:, :2] / X[m, 2:3]
        for x, y in uv.astype(int):
            if 0 <= x < w and 0 <= y < h: img[y, x] = col
    draw(W[~lime], (255, 255, 0)); draw(W[lime], (255, 0, 255))
    for c, b in fit["buckets"].items(): draw(bucket_world(fit, b, bpts), (255, 255, 255))
    tiles.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
cv2.imwrite(str(ROOT / "align/robot_check.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in (0, 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("ok")
