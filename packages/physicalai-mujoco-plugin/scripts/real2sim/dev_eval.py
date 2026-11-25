"""Builder's own dev evaluation on allowed frames (notes/DEV_SPLIT.json), never the reviewer's held-out frames:
matched-state composites at the dev poses vs the undistorted dev frames, psnr_cc as phone_eval.py computes it
(per-channel gain/bias, scale 0.5 INTER_AREA, eroded 5 px), plus the same without the robot's pixels.
usage: dev_eval.py SPLAT.ply [--label NAME]"""
import argparse, json, os, sys, cv2, numpy as np
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("splat"); ap.add_argument("--label", default=""); a = ap.parse_args()
os.environ["BUCKET_SPLAT"] = str(Path(a.splat).expanduser())
sys.path.insert(0, str(Path(__file__).parent)); sys.argv = ["x"]
import bucketsim
from common import ROOT, held, pinhole_cameras
from compositor import Calibration, Compositor, calibration
dev = json.load(open(ROOT / "notes/DEV_SPLIT.json"))["frames"]; assert not any(held(n) for n in dev)
cams = pinhole_cameras(); T0, K0, W, H = cams[dev[0]]
robot = bucketsim.make_robot(0); bucketsim.matched_state(robot)
comp = Compositor(robot._sim.model, robot._sim.data, calibration(), width=W, height=H, show_objects=False)
def colour_fit(r, q, m):
    o = r.copy()
    for c in range(3):
        A = np.stack([r[..., c][m], np.ones(int(m.sum()))], 1); g, b = np.linalg.lstsq(A, q[..., c][m], rcond=None)[0]; o[..., c] = np.clip(g * r[..., c] + b, 0, 1)
    return o
psnr = lambda x, y, m: float(-10 * np.log10(np.mean((x[m] - y[m]) ** 2) + 1e-12))
rows = []
for n in dev:
    T, K, w, h = cams[n]; r = comp.render_sfm(T, K, outputs=True)
    ren = r["rgb"].astype(np.float32) / 255; real = cv2.imread(str(ROOT / "colmap/brush_all/images" / n))[:, :, ::-1].astype(np.float32) / 255
    m = cv2.erode(np.ones((h, w), np.uint8), np.ones((5, 5), np.uint8)) > 0
    robot_px = cv2.dilate((r["seg"] == 10).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    sz = (round(w * 0.5), round(h * 0.5)); real = cv2.resize(real, sz, interpolation=cv2.INTER_AREA); ren = cv2.resize(ren, sz, interpolation=cv2.INTER_AREA)
    m = cv2.resize(m.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0; rp = cv2.resize(robot_px.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
    cc = colour_fit(ren, real, m)
    rows.append({"name": n, "psnr_cc": psnr(cc, real, m), "psnr_cc_norobot": psnr(cc, real, m & ~rp), "robot_frac": float(rp.mean())})
P = np.array([r["psnr_cc"] for r in rows]); Q = np.array([r["psnr_cc_norobot"] for r in rows])
res = {"label": a.label, "splat": a.splat, "n": len(rows), "psnr_cc_median": float(np.median(P)), "psnr_cc_worst": float(P.min()),
       "psnr_cc_norobot_median": float(np.median(Q)), "rows": rows}
out = ROOT / "notes/dev_eval"; out.mkdir(exist_ok=True); json.dump(res, open(out / f"{a.label or Path(a.splat).parent.parent.name}.json", "w"), indent=1)
print(f"DEV {a.label}: psnr_cc median {np.median(P):.2f} worst {P.min():.2f} | without robot {np.median(Q):.2f} | robot share {np.mean([r['robot_frac'] for r in rows]):.3f}")
