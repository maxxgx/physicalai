"""Builder's dev check of the robot-camera realism (not the reviewer's grading): ROBOTCAM_REPLAY at ALLOWED export frames
(review/export/robotcam_allowed: states_allowed.parquet, and frames_x5 decoded from the export), then the final-table
metrics as RUBRIC 3.x defines them, re-implemented here (no reviewer code is imported): per-channel gain/bias colour
fit over the valid mask, psnr_cc, lpips_cc (AlexNet, spatial, mean over the mask), sharpness ratio; labels 20-99
excluded, dilated 12 px, the mask eroded 5 px. Writes OUT/metrics.json and side-by-side sheets.
usage: dev_replay.py EXPORT FRAMES_X5 OUT [--n 30] [--video-time T] [--seed 0] [--env KEY=VAL ...]"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("export"); ap.add_argument("frames"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=30); ap.add_argument("--video-time", type=float, default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-render", action="store_true"); ap.add_argument("--layout", default="episode", choices=("episode", "matched")); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); F = Path(a.frames)
names = sorted(p.name for p in F.glob("overview_e*_t*.png"))
rng = np.random.default_rng(a.seed); pick = sorted(rng.choice(len(names), size=min(a.n, len(names)), replace=False))
items = []
for i in pick:
    n = names[i]; ep = int(n.split("_e")[1][:3]); ms = int(n.split("_t")[1][:7]); items.append((ep, ms / 1000.0, n))
times = ",".join(f"{ep}:{t:.4f}" for ep, t, _ in items)
here = Path(__file__).resolve().parent
if not a.skip_render:
    cmd = [sys.executable, "-I", "-B", str(here / "bucketsim_v2.py"), "replay", "--states", str(Path(a.export) / "states_allowed.parquet"),
           "--times", times, "--out", str(out / "renders"), "--contents-mask", "--layout", a.layout] + (["--video-time", str(a.video_time)] if a.video_time is not None else [])
    subprocess.run(cmd, check=True, env=os.environ.copy())

import lpips, torch
dev = "mps" if torch.backends.mps.is_available() else "cpu"
net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
load = lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
def colour_fit(ren, real, m):
    o = ren.copy()
    for c in range(3):
        A = np.stack([ren[..., c][m], np.ones(int(m.sum()))], 1); g, b = np.linalg.lstsq(A, real[..., c][m], rcond=None)[0]
        o[..., c] = np.clip(g * ren[..., c] + b, 0, 1)
    return o
def grad(x):
    g = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
k = np.ones((25, 25), np.uint8); rows = {"overview": [], "gripper": []}; sheets = {"overview": [], "gripper": []}
for ep, t, n in items:
    stem = f"{ep:03d}_{t:08.4f}"
    for cam in ("overview", "gripper"):
        real = load(F / n.replace("overview_", f"{cam}_")); ren = load(out / "renders" / f"{stem}_{cam}.png")
        seg = cv2.imread(str(out / "renders" / f"{stem}_{cam}_seg.png"), cv2.IMREAD_UNCHANGED); seg = seg[..., 0] if seg.ndim == 3 else seg
        e = ((seg >= 20) & (seg <= 99)).astype(np.uint8); m = cv2.dilate(e, k) == 0
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        cc = colour_fit(ren, real, m)
        with torch.no_grad():
            tt = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(dev) * 2 - 1
            lmap = net(tt(real), tt(cc))[0, 0].cpu().numpy()
        gr, gs = grad(real), grad(cc)
        r = {"frame": n, "episode": ep, "t": t, "psnr_raw": float(-10 * np.log10(np.mean((ren[m] - real[m]) ** 2) + 1e-12)),
             "psnr_cc": float(-10 * np.log10(np.mean((cc[m] - real[m]) ** 2) + 1e-12)), "lpips_cc": float(lmap[m].mean()),
             "sharp_ratio": float(gs[m].mean() / (gr[m].mean() + 1e-6)), "valid_frac": float(m.mean()),
             "mean_rgb_real": (real[m].mean(0) * 255).round(1).tolist(), "mean_rgb_sim": (ren[m].mean(0) * 255).round(1).tolist()}
        rows[cam].append(r)
        if len(sheets[cam]) < 6:
            sheets[cam].append(np.hstack([cv2.resize((real[:, :, ::-1] * 255).astype(np.uint8), (320, 240)), cv2.resize((ren[:, :, ::-1] * 255).astype(np.uint8), (320, 240))]))
summ = {cam: {key: float(np.median([r[key] for r in rs])) for key in ("psnr_raw", "psnr_cc", "lpips_cc", "sharp_ratio")} for cam, rs in rows.items()}
json.dump({"source": "review/export/robotcam_allowed (allowed episodes only)", "video_time": a.video_time, "n": len(items), "median": summ, "frames": rows}, open(out / "metrics.json", "w"), indent=1)
for cam, s in sheets.items():
    cv2.imwrite(str(out / f"sheet_{cam}.jpg"), np.vstack(s))
print(json.dumps(summ))
