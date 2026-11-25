"""Temporal sensor noise of a recorded dataset (LeRobot v3, as Studio writes it), per camera: in still segments (no
joint moves more than 0.05 deg for >= 8 consecutive frames), the per-pixel temporal standard deviation against the
pixel's mean level, fitted as sigma(I)^2 = read^2 + shot * I (8-bit levels). Run it on the real robot episodes (only the
ALLOWED ones, after the lock) and on a simulated dataset recorded through the same Studio pipeline, so both carry the
same video coding: the kit's ``sensor_noise`` {read, shot} is then set so the simulated recording matches the real one.
usage: noise_measure.py DATASET_DIR [--episodes 0,1,...] [--cams overview,wrist]   (Studio venv python: pandas, cv2)"""
import argparse, json, numpy as np, pandas as pd, cv2
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--episodes"); ap.add_argument("--cams", default="overview,wrist")
a = ap.parse_args(); D = Path(a.dataset)
df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index").reset_index(drop=True)
keep_eps = None if not a.episodes else {int(e) for e in a.episodes.split(",")}
st = np.stack(df["observation.state"].values); ep = df["episode_index"].to_numpy()
still = np.r_[False, (np.abs(np.diff(st, axis=0)).max(1) < 0.05) & (ep[1:] == ep[:-1])]
runs, i = [], 0   # [start, end) of still runs >= 8 frames
while i < len(still):
    if still[i]:
        j = i
        while j < len(still) and still[j]: j += 1
        if j - i >= 8 and (keep_eps is None or ep[i] in keep_eps): runs.append((i - 1, j))
        i = j
    else: i += 1
out = {}
for cam in a.cams.split(","):
    frames = []
    for v in sorted((D / f"videos/observation.images.{cam}").rglob("*.mp4")):
        c = cv2.VideoCapture(str(v))
        while True:
            ok, f = c.read()
            if not ok: break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    frames = np.array(frames[:len(df)]); lv, var = [], []
    for s, e in runs:
        block = frames[s:e].astype(np.float32); m = block.mean(0); v = block.var(0, ddof=1)
        lv.append(m[::4, ::4].ravel()); var.append(v[::4, ::4].ravel())
    if not lv:
        out[cam] = {"still_runs": 0}; continue
    lv, var = np.concatenate(lv), np.concatenate(var)
    bins = np.arange(0, 257, 16); mid, med = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (lv >= lo) & (lv < hi) & (lv > 3) & (lv < 252)   # clipped pixels have no noise to measure
        if sel.sum() > 200: mid.append((lo + hi) / 2); med.append(float(np.median(var[sel])))
    A = np.c_[np.ones(len(mid)), mid]; (r2, shot), *_ = np.linalg.lstsq(A, np.array(med), rcond=None)
    out[cam] = {"still_runs": len(runs), "frames_in_runs": int(sum(e - s for s, e in runs)), "read": float(np.sqrt(max(r2, 0))),
                "shot": float(max(shot, 0)), "sigma_by_level": {f"{m:.0f}": round(float(np.sqrt(v)), 3) for m, v in zip(mid, med)}}
    print(cam, json.dumps(out[cam]))
json.dump(out, open(D.name + "_noise.json", "w"), indent=1)
