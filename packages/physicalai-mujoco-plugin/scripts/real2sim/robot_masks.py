"""Training masks (255 = use, 0 = ignore) that hide the real robot: its CAD at the calibrated base pose and joints,
projected into every frame, closed and dilated (cables and blur). Writes align/masks_robot/<frame>.png and a sheet."""
import json, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, held, pinhole_cameras
import cad

cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]
M, t, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
local = cad.geom_local_points(per_geom=3000)
W = cad.arm_points(np.r_[cal["robot"]["joints_rad"], 0.0], local)           # sim world, base at the origin
# the wrist camera's printed mount (brown, not in the arm CAD): a 7 cm ball about the wrist's geometry
m_, _ = cad.model(); wrist = np.concatenate([np.full(len(p_), g) for g, (_, p_) in local.items()])
wb = {m_.body(b).id for b in ("link6", "gripper_end")}
c6 = W[np.isin([m_.geom_bodyid[g] for g in wrist], list(wb))].mean(0)
ball = np.random.default_rng(0).normal(size=(20000, 3)); ball = c6 + 0.07 * ball / np.linalg.norm(ball, axis=1, keepdims=True) * np.random.default_rng(1).random((20000, 1)) ** (1 / 3)
W = np.vstack([W, ball])
P = (W - t) @ M / sc                                                          # -> SfM (M orthonormal)
DIL = int(sys.argv[1]) if len(sys.argv) > 1 else 21
out = ROOT / "align/masks_robot"; out.mkdir(exist_ok=True)
cams = pinhole_cameras(); sheet = []
for i, (nm, (T, K, w, h)) in enumerate(sorted(cams.items())):
    X = P @ T[:3, :3].T + T[:3, 3]; X = X[X[:, 2] > 0.05]
    uv = np.round((X @ K.T)[:, :2] / X[:, 2:3]).astype(int); ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    m = np.zeros((h, w), np.uint8); m[uv[ok, 1], uv[ok, 0]] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIL, DIL)))
    cv2.imwrite(str(out / (Path(nm).stem + ".png")), 255 - m)
    if i % 60 == 0 and not held(nm):   # the sheet shows training frames only
        img = cv2.imread(str(ROOT / "colmap/pinhole/images" / nm)); img[m > 0] = (0.4 * img[m > 0] + [0, 0, 150]).astype(np.uint8)
        sheet.append(cv2.resize(img, (480, 270)))
while len(sheet) % 3: sheet.append(np.zeros_like(sheet[0]))
cv2.imwrite(str(ROOT / "align/masks_robot_sheet.jpg"), np.vstack([np.hstack(sheet[j:j + 3]) for j in range(0, len(sheet), 3)]))
print("masks", len(cams))
