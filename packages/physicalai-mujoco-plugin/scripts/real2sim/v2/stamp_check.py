"""Diagnostics: true staleness of a Studio dataset recorded with R2S_STAMP=1 (a frame counter in each frame's top-left
corner). usage: stamp_check.py DATASET_DIR   (Studio venv's python)"""
import sys, numpy as np, pandas as pd, cv2
from pathlib import Path
D = Path(sys.argv[1]); df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
st = np.stack(df["observation.state"].values); mv1 = np.r_[0, np.abs(np.diff(st, axis=0)).max(1)] > 1.0
newep = np.r_[True, np.diff(df["episode_index"].values) != 0]
for cam in sorted(p.name.replace("observation.images.", "") for p in (D / "videos").iterdir()):
    ids = []
    for v in sorted((D / f"videos/observation.images.{cam}").rglob("*.mp4")):
        c = cv2.VideoCapture(str(v))
        while True:
            ok, f = c.read()
            if not ok: break
            g = cv2.cvtColor(f[0:8, 0:16], cv2.COLOR_BGR2GRAY).astype(float)
            ids.append(sum((1 << b) for b in range(8) if g[2:6, b * 2:b * 2 + 2].mean() > 127))
    ids = np.array(ids[:len(df)]); same = np.r_[False, ids[1:] == ids[:-1]] & ~newep[:len(ids)]
    jump = np.r_[0, (ids[1:] - ids[:-1]) % 256]
    print(f"{cam}: stale {same.sum()} of {len(ids)} ({same.mean() * 100:.1f} %), stale while a joint moves > 1 deg {(same & mv1[:len(ids)]).sum()} of {mv1.sum()}; frames skipped per step: median {np.median(jump[~newep[:len(ids)]]):.0f}, p90 {np.percentile(jump[~newep[:len(ids)]], 90):.0f}")
