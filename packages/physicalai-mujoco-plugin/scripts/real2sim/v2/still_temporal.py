"""r16 (D7, approved camera-model fit): the temporal variation of STILL content, real (allowed export) against sim.

Still runs: >= 12 consecutive frames with every recorded channel (arm and gripper) changing < 0.05 deg per frame. Per
run, in decoded frames (as recorded, after AV1): (1) the global mean luma per frame (exposure / flicker): its structure
function D(k) = median |m(t + k) - m(t)| for k = 1, 2, 4, 8, 16 frames; (2) the low-frequency field: luma minus the
run's median frame, block-averaged 8 x 8 (1/8 resolution), the global mean removed: the median per-block |change| at the
same lags; (3) the full-resolution per-pixel |change| at lag 1 (the fine noise). TRAIN (0, 2, 6, 7, 9) / DEV (1, 3, 10)
for the real export, all episodes for a Studio dataset. Luma in 0-255.
usage: still_temporal.py DATASET_OR_EXPORT_DIR CAMERA OUT_JSON   (Studio venv python: pandas, cv2)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, cv2

D, CAM, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
export = (D / "states_allowed.parquet").exists()
if export:
    df = pd.read_parquet(D / "states_allowed.parquet").sort_values(["episode_index", "timestamp"])
else:
    df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
LAGS = (1, 2, 4, 8, 16)

def stream(paths):
    for v in paths:
        c = cv2.VideoCapture(str(v))
        while True:
            ok, f = c.read()
            if not ok: break
            yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)

def runs_of(still, n=12):
    out, start = [], None
    for i, s in enumerate(np.r_[still, False]):
        if s and start is None: start = i
        if not s and start is not None:
            if i - start >= n: out.append((start, i))
            start = None
    return out

def analyse(eps):
    gm, lf, fine = {k: [] for k in LAGS}, {k: [] for k in LAGS}, []
    src = None if export else stream(sorted((D / f"videos/observation.images.{CAM}").rglob("*.mp4")))
    for ep, g in df.groupby("episode_index"):
        S = np.stack(g["observation.state"].values).astype(float); dq = np.r_[np.inf, np.abs(np.diff(S, axis=0)).max(1)]
        frames = list(stream([D / f"ep{int(ep):03d}_{CAM}.mp4"])) if export else [next(src) for _ in range(len(g))]
        if export and int(ep) not in eps: continue
        for a, b in runs_of(dq[: len(frames)] < 0.05):
            F = np.stack(frames[a:b]); med = np.median(F, axis=0)
            m = F.reshape(len(F), -1).mean(1)
            R = F - med; h, w = R.shape[1] // 8 * 8, R.shape[2] // 8 * 8
            L = R[:, :h, :w].reshape(len(F), h // 8, 8, w // 8, 8).mean((2, 4)); L = L - L.mean((1, 2), keepdims=True)
            for k in LAGS:
                if len(F) > k:
                    gm[k] += np.abs(m[k:] - m[:-k]).tolist(); lf[k].append(float(np.median(np.abs(L[k:] - L[:-k]))))
            fine.append(float(np.median(np.abs(F[1:] - F[:-1]))))
    q = lambda v: round(float(np.median(v)), 4) if len(v) else None
    return {"runs": len(fine), "global_mean_D": {k: q(v) for k, v in gm.items()}, "lowfreq_field_D": {k: q(v) for k, v in lf.items()},
            "fine_pixel_lag1": q(fine)}

res = {"source": str(D), "camera": CAM}
if export:
    res["train"] = analyse({0, 2, 6, 7, 9}); res["dev"] = analyse({1, 3, 10})
else:
    res["sim"] = analyse(None)
for k, v in res.items():
    if isinstance(v, dict): print(k, json.dumps(v))
json.dump(res, open(OUT, "w"), indent=1)
