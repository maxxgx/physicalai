"""r16 (D6): phone-path renders of ALLOWED registered frames (asserted outside every held-out window +- guard), real |
render | |difference| sheets and a per-frame PSNR after a per-channel affine colour match (the builder's proxy, not the
reviewer's metric). usage: phone_allowed_check.py OUT_DIR FRAME[,FRAME...]"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); H = json.load(open(ROOT / "review/HELDOUT_PHONE.json"))
allowed = lambda n: not any(a - H["guard_s"] <= int(n[1:6]) / 30.0 <= bb + H["guard_s"] for a, bb in H["windows_s"])
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
names = [n if n.endswith(".jpg") else n + ".jpg" for n in sys.argv[2].split(",")]; assert all(map(allowed, names))
cams = b.pinhole_cameras(); robot = b.make_robot(0); b.matched_state(robot)
_, _, W, Hh = cams[names[0]]; comp = b.compositor(robot, W, Hh, show_objects=False); res = {}
for n in names:
    T, K, w, h = cams[n]; b.matched_state(robot, int(n[1:6]) / 30.0)
    r = comp.render_sfm(T, K)["rgb"].astype(np.float64); real = cv2.resize(cv2.imread(str(ROOT / "colmap/images" / n)), (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float64)
    cc = np.stack([np.polyval(np.polyfit(r[..., c].ravel(), real[..., c].ravel(), 1), r[..., c]) for c in range(3)], -1)
    mse = float(((np.clip(cc, 0, 255) - real) ** 2).mean()); res[n] = round(10 * np.log10(255 ** 2 / mse), 2)
    diff = np.abs(np.clip(cc, 0, 255) - real).mean(-1)
    tiles = [cv2.resize(x, (480, int(480 * h / w))) for x in (real[:, :, ::-1].astype(np.uint8), r[:, :, ::-1].astype(np.uint8), cv2.applyColorMap(np.clip(diff * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO))]
    cv2.imwrite(str(OUT / f"{n[:-4]}.jpg"), np.hstack(tiles)); print(n, res[n], flush=True)
json.dump(res, open(OUT / "psnr.json", "w"), indent=1)
