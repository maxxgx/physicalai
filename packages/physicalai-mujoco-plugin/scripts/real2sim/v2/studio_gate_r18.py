"""r18 D7 gate (coordinator): a Studio recording, per camera and per episode, measured the way the rubric measures it.

  duplicates   dscheck's definition: consecutive frames whose grey 160x120 mean |diff| < 0.25, counted while the arm
               moves (any joint change > 1e-4 deg since the previous frame), per camera, per episode and in all;
  timing       timing_profile.profile on the recorded states (still-run histogram, speed bins, events) and its distance
               to the real TRAIN and DEV profiles (sim/r18/timing/real.json, allowed export only);
  stalls       per episode the longest still run (every channel < 0.1 deg per frame) in seconds, and where it starts.
Live re-plan stalls are in the recorded states, so they count here. usage: studio_gate_r18.py DATASET_DIR REAL_JSON OUT_JSON
(run with the Studio venv's python: pandas, cv2)"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_profile import distance, profile  # noqa: E402

D, REAL, OUT = Path(sys.argv[1]), json.load(open(sys.argv[2])), sys.argv[3]
info = json.load(open(D / "meta/info.json"))
df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index").reset_index(drop=True)
st = np.stack(df["observation.state"].values).astype(float)
ep = df["episode_index"].values; newep = np.r_[True, np.diff(ep) != 0]
dq = np.r_[0, np.abs(np.diff(st, axis=0)).max(1)]; dq[newep] = 0
moving = (dq > 1e-4) & ~newep
res = {"dataset": str(D), "fps": info["fps"], "episodes": int(info["total_episodes"]), "frames": int(len(df)), "cameras": {}}
for cam in [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]:
    d = []
    for v in sorted((D / f"videos/observation.images.{cam}").rglob("*.mp4")):
        c = cv2.VideoCapture(str(v)); prev = None
        while True:
            ok, f = c.read()
            if not ok:
                break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
            d.append(float(np.abs(g - prev).mean()) if prev is not None else np.inf); prev = g
    n = min(len(d), len(df)); d = np.array(d[:n]); dup = (d < 0.25) & ~newep[:n]; mv = moving[:n]
    per = {}
    for e in np.unique(ep[:n]):
        k = ep[:n] == e
        per[int(e)] = round(100 * float((dup & mv & k).sum()) / max(int((mv & k).sum()), 1), 2)
    res["cameras"][cam] = {"frames": int(n), "dup_while_moving_pct": round(100 * float((dup & mv).sum()) / max(int(mv.sum()), 1), 2),
                           "dup_all_pct": round(100 * float(dup.mean()), 2), "per_episode_while_moving_pct": per,
                           "worst_episode": max(per.items(), key=lambda kv: kv[1]) if per else None}
episodes = [st[ep == e] for e in np.unique(ep)]
prof = profile(episodes)
res["timing"] = {"sim": prof, "distance_train": distance(prof, REAL["train"]), "distance_dev": distance(prof, REAL["dev"]),
                 "real_train": {k: REAL["train"][k] for k in ("near_still_pct", "still_runs_pct", "speed_bins_pct")},
                 "real_dev": {k: REAL["dev"][k] for k in ("near_still_pct", "still_runs_pct", "speed_bins_pct")}}
stalls = {}
for e, S in zip(np.unique(ep), episodes):
    still = np.r_[False, np.abs(np.diff(S, axis=0)).max(1) < 0.1]; best, at, st0 = 0, None, None
    for i, x in enumerate(np.r_[still, False]):
        if x and st0 is None:
            st0 = i
        if not x and st0 is not None:
            if i - st0 > best:
                best, at = i - st0, st0
            st0 = None
    stalls[int(e)] = {"longest_still_s": round(best / info["fps"], 2), "starts_s": round((at or 0) / info["fps"], 2), "episode_s": round(len(S) / info["fps"], 2)}
res["stalls"] = stalls
res["longest_stall_s"] = max(v["longest_still_s"] for v in stalls.values()) if stalls else None
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({"cameras": {c: {k: v[k] for k in ("dup_while_moving_pct", "worst_episode")} for c, v in res["cameras"].items()},
                  "distance_train": res["timing"]["distance_train"], "distance_dev": res["timing"]["distance_dev"],
                  "near_still_pct": prof["near_still_pct"], "still_runs_pct": prof["still_runs_pct"], "speed_bins_pct": prof["speed_bins_pct"],
                  "longest_stall_s": res["longest_stall_s"]}))
