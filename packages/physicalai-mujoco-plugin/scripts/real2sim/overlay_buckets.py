"""Project the fitted bucket CAD (align/buckets3d.json) onto training frames as coloured outlines; writes align/buckets3d_check.jpg."""
import json, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, held, pinhole_cameras
import cad

def bucket_world(fit, b, pts):
    pl = fit["plane"]; o, u, v, n = (np.array(pl[k]) for k in "ouvn"); s = fit["s_units_per_m"]
    c, sn = np.cos(b["yaw"]), np.sin(b["yaw"]); R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    q = pts @ R.T + [b["x"], b["y"], 0]
    return o + s * (q[:, :1] * u + q[:, 1:2] * v + q[:, 2:3] * n)

if __name__ == "__main__":
    fit = json.load(open(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "align/buckets3d.json")))
    cams = pinhole_cameras(); names = sorted(n for n in cams if not held(n))
    pick = [names[int(i)] for i in np.linspace(0, len(names) - 1, 6)]
    pts = cad.bucket_body_points(4000)
    col = {"green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 0, 0), "black": (255, 0, 255)}
    tiles = []
    for nm in pick:
        T, K, w, h = cams[nm]; img = cv2.imread(str(ROOT / "colmap/brush_all/images" / nm))
        for c, b in fit["buckets"].items():
            P = bucket_world(fit, b, pts); X = P @ T[:3, :3].T + T[:3, 3]; X = X[X[:, 2] > 0]
            uv = (X @ K.T)[:, :2] / X[:, 2:3]
            for x, y in uv.astype(int):
                if 0 <= x < w and 0 <= y < h: img[y, x] = col[c]
        tiles.append(cv2.resize(img, (640, 360)))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in (0, 3)])
    cv2.imwrite(str(ROOT / "align/buckets3d_check.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); print("ok", pick)
