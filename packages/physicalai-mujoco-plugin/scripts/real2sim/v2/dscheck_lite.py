"""Builder's check of a Studio dataset (not the reviewer's dscheck): fps, episodes, frames, tasks, timestamp steps, and
consecutive identical frames per camera, split by whether the arm moved (a frozen frame while the arm moves is a stale
camera). usage: dscheck_lite.py DATASET_DIR   (run with the Studio venv's python: pandas, cv2)"""
import sys, json, numpy as np, pandas as pd, cv2
from pathlib import Path
D = Path(sys.argv[1]); info = json.load(open(D / "meta/info.json"))
df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
st = np.stack(df["observation.state"].values); dq = np.r_[0, np.abs(np.diff(st, axis=0)).max(1)]
mv = dq > 1e-4   # any joint change (dscheck)
mv1 = dq > 1.0   # clearly moving: a joint moved > 1 deg since the previous frame
newep = np.r_[True, np.diff(df["episode_index"].values) != 0]
steps = df.groupby("episode_index")["timestamp"].apply(lambda s: np.diff(s.values))
print(json.dumps({"fps": info["fps"], "episodes": info["total_episodes"], "frames": info["total_frames"],
                  "frames_per_episode": df.groupby("episode_index").size().tolist(),
                  "timestamp_step_ms": [round(float(min(v.min() for v in steps)) * 1000, 2), round(float(max(v.max() for v in steps)) * 1000, 2)],
                  "tasks": pd.read_parquet(D / "meta/tasks.parquet").index.tolist()[:10]}))
for cam in [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]:
    d = []
    for v in sorted((D / f"videos/observation.images.{cam}").rglob("*.mp4")):
        c = cv2.VideoCapture(str(v)); prev = None
        while True:
            ok, f = c.read()
            if not ok: break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32); d.append(float(np.abs(g - prev).mean()) if prev is not None else np.inf); prev = g
    d = np.array(d[:len(df)]); dup = (d < 0.25) & ~newep[:len(d)]   # the rubric's dscheck definition: mean |diff| < 0.25 at 160x120
    m1 = mv1[:len(d)]
    print(f"{cam}: {len(d)} frames, identical to the previous {dup.sum()} ({dup.mean() * 100:.1f} %), while the arm moves {(dup & mv[:len(d)]).sum()} ({(dup & mv[:len(d)]).sum() / max(mv.sum(), 1) * 100:.1f} % of moving frames), while a joint moves > 1 deg {(dup & m1).sum()} ({(dup & m1).sum() / max(m1.sum(), 1) * 100:.1f} % of {m1.sum()})")
