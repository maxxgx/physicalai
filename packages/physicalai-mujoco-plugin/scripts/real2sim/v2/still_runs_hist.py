"""r17 (D7 analysis, coordinator): still-run lengths, sim (a Studio dataset) against real (the allowed export, TRAIN 0, 2,
6, 7, 9 and DEV 1, 3, 10). A frame is still when every recorded channel changes < 0.1 deg from the previous frame; runs of
consecutive still frames; the share of all frames in runs of each length bin (frames at 30 fps: 1, 2-3, 4-9, 10-29
(< 1 s), 1-3 s, 3-7 s, 7-15 s, >= 15 s). usage: still_runs_hist.py DATASET EXPORT OUT_JSON   (Studio venv: pandas)"""
import glob, json, sys
import numpy as np, pandas as pd
BINS = [(1, 1), (2, 3), (4, 9), (10, 29), (30, 89), (90, 209), (210, 449), (450, 10 ** 9)]
LAB = ["1", "2-3", "4-9", "10-29", "1-3 s", "3-7 s", "7-15 s", ">=15 s"]
def hist(groups):
    tot, inbin = 0, np.zeros(len(BINS))
    for S in groups:
        tot += len(S); still = np.r_[False, np.abs(np.diff(S, axis=0)).max(1) < 0.1]; st = None
        for i, s in enumerate(np.r_[still, False]):
            if s and st is None: st = i
            if not s and st is not None:
                n = i - st; inbin[[k for k, (a, b) in enumerate(BINS) if a <= n <= b][0]] += n; st = None
    return {"frames": int(tot), "near_still_pct": round(100 * inbin.sum() / tot, 2), "pct_in_runs": dict(zip(LAB, (100 * inbin / tot).round(2).tolist()))}
D, E, OUT = sys.argv[1:4]
sim = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{D}/data/**/*.parquet", recursive=True))]).sort_values("index")
real = pd.read_parquet(f"{E}/states_allowed.parquet").sort_values(["episode_index", "timestamp"])
st = lambda df, eps=None: [np.stack(g["observation.state"].values).astype(float) for e, g in df.groupby("episode_index") if eps is None or int(e) in eps]
res = {"sim": hist(st(sim)), "real_train": hist(st(real, {0, 2, 6, 7, 9})), "real_dev": hist(st(real, {1, 3, 10}))}
for k, v in res.items(): print(k, json.dumps(v))
json.dump(res, open(OUT, "w"), indent=1)
