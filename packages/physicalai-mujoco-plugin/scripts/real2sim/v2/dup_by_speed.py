"""r18 (D7 / U-2): where a Studio recording's duplicate frames fall, by the arm's speed at that frame. A duplicate is
dscheck's (grey 160x120 mean |diff| < 0.25 to the previous frame, within an episode); the speed is the largest joint
change since the previous frame (deg). Duplicates at high arm speed are repeated camera frames (a stale stream);
at low speed, frames in which the arm moved too little to change the picture. usage: dup_by_speed.py DATASET_DIR"""
import json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
D = Path(sys.argv[1]); info = json.load(open(D / "meta/info.json"))
df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index").reset_index(drop=True)
st = np.stack(df["observation.state"].values).astype(float); ep = df["episode_index"].values
newep = np.r_[True, np.diff(ep) != 0]; sp = np.r_[0, np.abs(np.diff(st[:, :6], axis=0)).max(1)]; sp[newep] = 0
gm = np.r_[0, np.abs(np.diff(st[:, 6]))]; gm[newep] = 0
BINS = [(1e-4, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 1e9)]
out = {}
for cam in [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]:
    d = []
    for v in sorted((D / f"videos/observation.images.{cam}").rglob("*.mp4")):
        c = cv2.VideoCapture(str(v)); prev = None
        while True:
            ok, f = c.read()
            if not ok: break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
            d.append(float(np.abs(g - prev).mean()) if prev is not None else np.inf); prev = g
    n = min(len(d), len(df)); d = np.array(d[:n]); dup = (d < 0.25) & ~newep[:n]; exact = (d < 0.02) & ~newep[:n]
    rows = {}
    for lo, hi in BINS:
        k = (sp[:n] >= lo) & (sp[:n] < hi)
        rows[f"{lo:g}-{hi:g}"] = {"frames": int(k.sum()), "dup_pct": round(100 * float((dup & k).sum()) / max(int(k.sum()), 1), 1),
                                 "exact_repeat_pct": round(100 * float((exact & k).sum()) / max(int(k.sum()), 1), 1)}
    mv = sp[:n] > 1e-4
    out[cam] = {"dup_while_moving_pct": round(100 * float((dup & mv).sum()) / max(int(mv.sum()), 1), 2),
                "exact_repeats_while_moving_pct": round(100 * float((exact & mv).sum()) / max(int(mv.sum()), 1), 2),
                "dup_while_moving_share_by_arm_speed": {b: round(100 * float((dup & mv & (sp[:n] >= lo) & (sp[:n] < hi)).sum()) / max(int((dup & mv).sum()), 1), 1) for b, (lo, hi) in zip(rows, BINS)},
                "by_arm_speed_deg_per_frame": rows,
                "dup_while_only_gripper_moves_pct": round(100 * float((dup & ~mv & (gm[:n] > 1e-4)).sum()) / max(int((~mv & (gm[:n] > 1e-4)).sum()), 1), 1)}
print(json.dumps(out, indent=1))
