"""r14 (D1, coordinator): a global tone curve for the overview composite, fitted on TRAIN allowed frames, chosen on DEV.

psnr_cc fits each image's gain and bias per channel before comparing, so only the SHAPE of a tone curve can change it.
Fitted: per channel a monotone piecewise-linear curve f (8 knots on [0, 1]) applied to the overview's output, minimising
sum over TRAIN images of |a_ic f(sim) + b_ic - real|^2 over the valid mask (labels 20-99 dilated 12 px, eroded 5 px, as
dev_replay.py), alternating f and the per-image gains. Frames: ROBOTCAM_REPLAY at allowed export frames, TRAIN episodes
0, 2, 6, 7, 9 (8 each), DEV 1, 3, 10 (10 each; reported, never fitted). Reports DEV median psnr_cc before / after; with
--write and a DEV gain, composes f into the kit's overview LUT (camera_models.overview.lut: y' = f(y) at its 16 knots).
usage: fit_overview_tone.py EXPORT FRAMES_X5 OUT_DIR [--write]"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, cv2

EXPORT, F, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent; KIT = Path(os.environ["KIT"]).expanduser()
rng = np.random.default_rng(0); KN = np.linspace(0, 1, 8)
def pick(eps, per):
    out = []
    for ep in eps:
        names = sorted(p.name for p in F.glob(f"overview_e{ep:03d}_t*.png"))
        for i in sorted(rng.choice(len(names), size=min(per, len(names)), replace=False)):
            out.append((ep, int(names[i].split("_t")[1][:7]) / 1000.0, names[i]))
    return out
sets = {"train": pick((0, 2, 6, 7, 9), 8), "dev": pick((1, 3, 10), 10)}
for split, items in sets.items():
    if not (OUT / split / "replay.json").exists():
        subprocess.run([sys.executable, "-I", "-B", str(HERE / "bucketsim_v2.py"), "replay", "--states", str(EXPORT / "states_allowed.parquet"),
                        "--times", ",".join(f"{ep}:{t:.4f}" for ep, t, _ in items), "--out", str(OUT / split)], check=True, env=os.environ.copy())
load = lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64) / 255.0
def pairs(split):
    out = []
    for ep, t, n in sets[split]:
        stem = f"{ep:03d}_{t:08.4f}"; ren = load(OUT / split / f"{stem}_overview.png"); real = load(F / n)
        seg = cv2.imread(str(OUT / split / f"{stem}_overview_seg.png"), cv2.IMREAD_UNCHANGED); seg = seg[..., 0] if seg.ndim == 3 else seg
        e = ((seg >= 20) & (seg <= 99)).astype(np.uint8); m = cv2.dilate(e, np.ones((25, 25), np.uint8)) == 0
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        idx = rng.choice(int(m.sum()), size=min(20000, int(m.sum())), replace=False)
        out.append((ren[m][idx], real[m][idx], ren, real, m))
    return out
TR, DV = pairs("train"), pairs("dev")
def basis(x):   # linear interpolation weights on the knots
    i = np.clip(np.searchsorted(KN, x, side="right") - 1, 0, len(KN) - 2); w = (x - KN[i]) / (KN[i + 1] - KN[i])
    B = np.zeros((len(x), len(KN))); B[np.arange(len(x)), i] = 1 - w; B[np.arange(len(x)), i + 1] = w; return B
LAM = float(os.environ.get("TONE_LAMBDA", "0"))   # chosen on DEV
theta = np.tile(KN, (3, 1))
for it in range(6):
    for c in range(3):
        rows_A, rows_b = [], []
        for s, r, *_ in TR:
            B = basis(s[:, c]); fs = B @ theta[c]; A = np.stack([fs, np.ones_like(fs)], 1); g, b0 = np.linalg.lstsq(A, r[:, c], rcond=None)[0]
            # f(0) = 0 and f(1) = 1 fixed (scale and offset are the per-image gains' job): only the interior knots fitted
            rows_A.append(g * B[:, 1:-1]); rows_b.append(r[:, c] - b0 - g * B[:, -1])
        Am, bm = np.vstack(rows_A), np.concatenate(rows_b)
        if LAM > 0:   # smoothness: second differences of the whole curve (endpoints fixed) pulled toward 0
            D = np.zeros((len(KN) - 2, len(KN))); [D.__setitem__((i, slice(i, i + 3)), [1, -2, 1]) for i in range(len(KN) - 2)]
            Am = np.vstack([Am, np.sqrt(LAM) * D[:, 1:-1]]); bm = np.concatenate([bm, -np.sqrt(LAM) * (D[:, -1] * 1.0)])
        inner = np.linalg.lstsq(Am, bm, rcond=None)[0]
        theta[c] = np.maximum.accumulate(np.clip(np.r_[0.0, inner, 1.0], 0, 1))
def psnr_cc(items, th):
    out = []
    for _, _, ren, real, m in items:
        cc = ren.copy()
        for c in range(3):
            v = (basis(ren[..., c].ravel()) @ th[c]).reshape(ren.shape[:2])
            A = np.stack([v[m], np.ones(int(m.sum()))], 1); g, b0 = np.linalg.lstsq(A, real[..., c][m], rcond=None)[0]
            cc[..., c] = np.clip(g * v + b0, 0, 1)
        out.append(-10 * np.log10(np.mean((cc[m] - real[m]) ** 2) + 1e-12))
    return float(np.median(out))
ident = np.tile(KN, (3, 1))
res = {"knots": KN.tolist(), "curve": theta.round(4).tolist(), "train_frames": [n for *_, n in sets["train"]], "dev_frames": [n for *_, n in sets["dev"]],
       "train_psnr_cc": [psnr_cc(TR, ident), psnr_cc(TR, theta)], "dev_psnr_cc": [psnr_cc(DV, ident), psnr_cc(DV, theta)]}
print(json.dumps({k: v for k, v in res.items() if "psnr" in k or k == "curve"}), flush=True)
json.dump(res, open(OUT / "tone_fit.json", "w"), indent=1)
if "--write" in sys.argv and res["dev_psnr_cc"][1] > res["dev_psnr_cc"][0]:
    kit = json.load(open(KIT)); lut = kit["camera_models"]["overview"]["lut"]
    lut["y"] = [list(np.clip(basis(np.asarray(lut["y"][c], float)) @ theta[c] * (max(lut["y"][c]) - min(lut["y"][c])) / 1.0 + 0, 0, 1).round(4))
                if False else list(np.round(np.interp(np.asarray(lut["y"][c], float), KN, theta[c]), 4)) for c in range(3)]
    kit["camera_models"]["overview"]["tone_r14"] = {"curve_knots": KN.tolist(), "curve": theta.round(4).tolist(), "fitted_from": "fit_overview_tone.py (r14, D1): TRAIN 0, 2, 6, 7, 9 (40 frames), chosen on DEV 1, 3, 10 (30 frames); lists in real/tone_fit.json", "dev_psnr_cc": res["dev_psnr_cc"]}
    json.dump(kit, open(KIT, "w"), indent=1); print("kit overview LUT composed with the curve")
