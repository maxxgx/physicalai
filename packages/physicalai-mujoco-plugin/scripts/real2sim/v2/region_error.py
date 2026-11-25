"""Where the replay error lives: for a dev_replay output (renders + metrics frames), the colour-fitted squared error per
region of the sim segmentation (desk 5, buckets 1-4, robot 10, background 0, housings 11/12), as a share of the total
and as each region's own psnr_cc, per camera. usage: region_error.py DEV_REPLAY_DIR FRAMES_X5"""
import sys, json, numpy as np, cv2
from pathlib import Path
D, F = Path(sys.argv[1]), Path(sys.argv[2]); M = json.load(open(D / "metrics.json"))
groups = {"desk": [5], "buckets": [1, 2, 3, 4], "bucket walls": [1, 2, 3, 4], "bucket contents": [1, 2, 3, 4], "robot": [10], "background": [0], "housings": [11, 12]}
for cam in ("overview", "gripper"):
    sse = {g: 0.0 for g in groups}; npx = {g: 0 for g in groups}; sse_nc, npx_nc = [0.0], [0]
    for r in M["frames"][cam]:
        stem = f"{r['episode']:03d}_{r['t']:08.4f}"
        real = cv2.imread(str(F / r["frame"].replace("overview_", f"{cam}_")))[:, :, ::-1].astype(np.float32) / 255
        ren = cv2.imread(str(D / "renders" / f"{stem}_{cam}.png"))[:, :, ::-1].astype(np.float32) / 255
        seg = cv2.imread(str(D / "renders" / f"{stem}_{cam}_seg.png"), cv2.IMREAD_UNCHANGED); seg = seg[..., 0] if seg.ndim == 3 else seg
        m = cv2.erode((cv2.dilate(((seg >= 20) & (seg <= 99)).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        cc = ren.copy()
        for c in range(3):
            A = np.stack([ren[..., c][m], np.ones(int(m.sum()))], 1); g, b = np.linalg.lstsq(A, real[..., c][m], rcond=None)[0]; cc[..., c] = np.clip(g * ren[..., c] + b, 0, 1)
        e = ((cc - real) ** 2).sum(-1)
        cp = D / "renders" / f"{stem}_{cam}_contents.png"
        contents = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE) > 127 if cp.exists() else np.zeros(seg.shape, bool)
        for g, labs in groups.items():
            mk = m & np.isin(seg, labs)
            if g == "bucket walls": mk &= ~contents
            if g == "bucket contents": mk &= contents
            sse[g] += float(e[mk].sum()); npx[g] += int(mk.sum())
        mk = m & ~contents; sse_nc[0] += float(e[mk].sum()); npx_nc[0] += int(mk.sum())
    tot = sum(v for g, v in sse.items() if g not in ("bucket walls", "bucket contents"))
    print(cam, {g: {"share_of_error": round(sse[g] / tot, 3), "share_of_pixels": round(npx[g] / max(sum(npx.values()), 1), 3),
                    "psnr_cc": round(float(-10 * np.log10(sse[g] / max(npx[g], 1) / 3 + 1e-12)), 2)} for g in groups if npx[g]},
          "| all regions without the sim bucket contents: psnr_cc", round(float(-10 * np.log10(sse_nc[0] / max(npx_nc[0], 1) / 3 + 1e-12)), 2),
          "with:", round(float(-10 * np.log10(tot / max(sum(npx[g] for g in groups if g not in ("bucket walls", "bucket contents")), 1) / 3 + 1e-12)), 2))
