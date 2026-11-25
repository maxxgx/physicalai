"""r16 (D7, planning stalls): how still the arm is in a recorded Studio dataset, against the real allowed export.

Per frame the largest change of the six arm joints since the previous frame (deg, the recorded state). Near-still:
< 0.1 deg. Still runs: consecutive near-still frames, in frames (1/30 s). Reported: the near-still share, the run
lengths (p50 / p90 / max), the share of frames in runs of 0-1, 1-3, 3-7, 7-15, >= 15 s, the start still (first frame to
the first motion) per episode; the real export split TRAIN (0, 2, 6, 7, 9) / DEV (1, 3, 10).
usage: rec_still.py DATASET_DIR STATES_ALLOWED_PARQUET OUT_JSON   (Studio venv python: pandas)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

def frames_of(df):
    for ep, g in df.groupby("episode_index"):
        yield ep, np.stack(g["observation.state"].values).astype(float)

def stats(episodes):
    near_n = tot = 0; runs = []; starts = []
    for _, S in episodes:
        dq = np.abs(np.diff(S[:, :6], axis=0)).max(1); still = dq < 0.1
        tot += len(dq); near_n += int(still.sum())
        moving = np.flatnonzero(~still); starts.append(float(moving[0] + 1) / 30 if len(moving) else len(dq) / 30)
        n = 0
        for s in np.r_[still, False]:
            if s: n += 1
            elif n: runs.append(n); n = 0
    r = np.array(runs) if runs else np.zeros(1)
    bins = {}
    for lo, hi in ((0, 1), (1, 3), (3, 7), (7, 15), (15, 1e9)):
        sel = (r >= lo * 30) & (r < hi * 30); bins[f"{lo}-{hi if hi < 1e9 else 'inf'} s"] = round(100 * float(r[sel].sum()) / max(tot, 1), 2)
    return {"frames": tot, "near_still_pct": round(100 * near_n / max(tot, 1), 2), "still_runs": len(runs),
            "run_frames_p50_p90_max": [float(np.percentile(r, 50)), round(float(np.percentile(r, 90)), 1), int(r.max())],
            "pct_of_frames_in_runs": bins, "start_still_s_median_p10_p90": [round(float(np.median(starts)), 2)] + np.percentile(starts, [10, 90]).round(2).tolist(),
            "start_still_s": [round(x, 2) for x in starts][:60]}

D, REAL, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sim = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
real = pd.read_parquet(REAL).sort_values(["episode_index", "timestamp"])
res = {"sim": stats(frames_of(sim)),
       "real_train": stats((e, S) for e, S in frames_of(real) if e in (0, 2, 6, 7, 9)),
       "real_dev": stats((e, S) for e, S in frames_of(real) if e in (1, 3, 10))}
for k, v in res.items(): print(k, json.dumps({x: y for x, y in v.items() if x != "start_still_s"}))
json.dump(res, open(OUT, "w"), indent=1)
