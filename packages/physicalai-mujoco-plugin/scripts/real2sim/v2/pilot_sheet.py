"""Pilot contact sheet: N random episodes of a Studio dataset (simulated frames only), each at its first grasp: the first frame
at the episode's most-closed gripper reading (within 5 deg), and 0.5 s later (lifting). Columns: overview, wrist at the grasp; overview, wrist 0.5 s later. Episode and
frame stamped. usage: pilot_sheet.py DATASET_DIR OUT_JPG [--n 6] [--seed 0]   (Studio venv python: pandas, cv2)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, cv2

D, OUT = Path(sys.argv[1]), sys.argv[2]
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6
rng = np.random.default_rng(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0)
info = json.load(open(D / "meta/info.json"))
df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
cams = [k for k in info["features"] if k.startswith("observation.images.")]

def frames_of(cam):   # global frame index -> image, streamed (episodes in order across the chunk files)
    for v in sorted((D / f"videos/{cam}").rglob("*.mp4")):
        c = cv2.VideoCapture(str(v))
        while True:
            ok, f = c.read()
            if not ok: break
            yield f

eps = sorted(df.episode_index.unique()); pick = sorted(rng.choice(eps, size=min(N, len(eps)), replace=False))
want = {}
for ep in pick:
    g = df[df.episode_index == ep]; S = np.stack(g["observation.state"].values).astype(float); grip = S[:, 6]
    # the grasp: the first frame at the episode's most-closed reading (within 5 deg; readings run -270 open to 0 closed)
    k = int(np.argmax(grip >= grip.max() - 5.0))
    base = int(g["index"].iloc[0]); want[ep] = (base + k, base + min(k + 15, len(grip) - 1), k)
need = {i for a, b, _ in want.values() for i in (a, b)}
imgs = {}
for cam in cams:
    for i, f in enumerate(frames_of(cam)):
        if i in need: imgs[(cam, i)] = cv2.resize(f, (320, 240))
        if i > max(need): break
rows = []
for ep in pick:
    a, b, k = want[ep]; tiles = []
    for idx in (a, b):
        for cam in cams:
            t = imgs.get((cam, idx), np.zeros((240, 320, 3), np.uint8)).copy()
            cv2.putText(t, f"ep {ep} f{idx - (a - k)} {cam.split('.')[-1]}{' +0.5s' if idx == b else ''}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
    rows.append(np.hstack(tiles))
cv2.imwrite(OUT, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("sheet", OUT, "episodes", [int(e) for e in pick])
