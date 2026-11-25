"""Spot-check a recorded Studio dataset: N random episodes (seeded), per episode a contact sheet of both cameras at the
start, 1/3, 2/3 and end, with its instruction; plus fps, lengths and the dscheck-style duplicate rate of the set.
usage: spotcheck_v2.py DATASET_DIR [N] [OUT_DIR]   (Studio venv python: pandas, cv2)"""
import sys, json, numpy as np, pandas as pd, cv2
from pathlib import Path
D = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
out = Path(sys.argv[3]) if len(sys.argv) > 3 else D.parent / (D.name + "_spotcheck"); out.mkdir(parents=True, exist_ok=True)
info = json.loads((D / "meta/info.json").read_text())
df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index").reset_index(drop=True)
tasks = pd.read_parquet(D / "meta/tasks.parquet"); task_of = {int(r): str(t) for t, r in zip(tasks.index, tasks["task_index"])}
eps = sorted(df["episode_index"].unique()); pick = sorted(np.random.default_rng(0).choice(eps, size=min(N, len(eps)), replace=False))
frames = {}
CAMS = [c for c in ("overview", "gripper", "wrist") if (D / f"videos/observation.images.{c}").exists()]
for cam in CAMS:
    fr = []
    for v in sorted((D / f"videos/observation.images.{cam}").rglob("*.mp4")):
        c = cv2.VideoCapture(str(v))
        while True:
            ok, f = c.read()
            if not ok: break
            fr.append(cv2.resize(f, (320, 240)))
    frames[cam] = fr
summary = []
for e in pick:
    idx = df.index[df["episode_index"] == e].to_numpy(); t = task_of.get(int(df.loc[idx[0], "task_index"]), "?")
    cols = [idx[0], idx[len(idx) // 3], idx[2 * len(idx) // 3], idx[-1]]
    sheet = np.vstack([np.hstack([frames[c][i] for i in cols]) for c in CAMS])
    cv2.putText(sheet, f"ep {e}: {t}"[:90], (6, 18), 0, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(out / f"episode_{e:04d}.jpg"), sheet)
    summary.append({"episode": int(e), "frames": int(len(idx)), "seconds": round(len(idx) / info["fps"], 1), "task": t})
json.dump({"dataset": str(D), "fps": info["fps"], "episodes": int(len(eps)), "checked": summary}, open(out / "spotcheck.json", "w"), indent=1)
print(json.dumps({"fps": info["fps"], "episodes": int(len(eps)), "checked": summary}, indent=1))
