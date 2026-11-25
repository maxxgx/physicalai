"""D7 S3 (r10): repeated frames per episode of a Studio dataset, by the rubric's definition (cv2 decode, grey, INTER_AREA
to 160x120, mean |diff| < 0.25 = identical to the previous frame), overall and where the arm moves (a joint changed by
more than 1e-4 since the previous frame). Also per episode: whether the plugin drew a pile (from the plugin log's
reset lines, matched by order) and the arm's mean joint speed. usage: dup_per_episode.py DATASET_DIR [PLUGIN_LOG] OUT_JSON"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
D = Path(sys.argv[1]); out = sys.argv[-1]; plog = sys.argv[2] if len(sys.argv) > 3 else None
df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
eps = pd.concat([pd.read_parquet(p) for p in sorted((D / "meta/episodes").rglob("*.parquet"))]).sort_values("episode_index")
st = np.stack(df["observation.state"].values); dq = np.r_[0, np.abs(np.diff(st, axis=0)).max(1)]
newep = np.r_[True, np.diff(df["episode_index"].values) != 0]; moving = (dq > 1e-4) & ~newep
res = {"dataset": str(D), "definition": __doc__.split("\n")[0], "episodes": []}
cams = sorted(p.name.replace("observation.images.", "") for p in (D / "videos").iterdir())
same = {}
for cam in cams:
    flags = []
    for f in sorted((D / f"videos/observation.images.{cam}").rglob("*.mp4")):
        cap = cv2.VideoCapture(str(f)); prev = None
        while True:
            ok, fr = cap.read()
            if not ok: break
            g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
            flags.append(prev is not None and float(np.abs(g - prev).mean()) < 0.25); prev = g
    same[cam] = np.array(flags[: len(df)])
    if len(flags) != len(df): res.setdefault("notes", []).append(f"{cam}: {len(flags)} decoded frames vs {len(df)} rows")
piles = []
if plog:
    for line in open(plog):
        m = re.search(r"Bucket sort: .*\((put [^)]*)\); (\d+) targets", line)
        if m: piles.append(line)
for e in sorted(df["episode_index"].unique()):
    sel = (df["episode_index"].values == e); sel_first = sel & ~newep
    row = {"episode": int(e), "frames": int(sel.sum()), "task": str(df.loc[sel, "task_index"].iloc[0]) if "task_index" in df else None,
           "mean_joint_speed_deg_s": round(float(dq[sel_first].mean() * 30), 1)}
    for cam in cams:
        s = same[cam][: len(sel)]
        row[cam] = {"identical_pct": round(100 * float(s[sel_first].mean()), 1), "identical_while_moving_pct": round(100 * float(s[sel_first & moving].mean()), 1) if (sel_first & moving).any() else None}
    res["episodes"].append(row)
for cam in cams:
    s = same[cam]; first = ~newep[: len(s)]
    res[cam] = {"identical_pct": round(100 * float(s[first].mean()), 1), "identical_while_moving_pct": round(100 * float(s[first & moving[: len(s)]].mean()), 1)}
json.dump(res, open(out, "w"), indent=1)
print({cam: res[cam] for cam in cams})
for r in res["episodes"]: print(r["episode"], r["frames"], r["mean_joint_speed_deg_s"], {c: r[c]["identical_pct"] for c in cams})
