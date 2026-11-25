"""r13 (coordinator, pilot defect): does each recorded episode's first frame show its own scene, not the previous one?

Per episode and camera (cv2 decode, grey, INTER_AREA 160x120, as dscheck): the frame-to-frame change over the first
20 frames. A reset between episodes is a scene cut (a new table, new objects, the arm back home): mean |diff| far above
the episode's own motion. A cut at frame k >= 1 means frames 0..k-1 are the previous scene. Also: frame 0 against the
previous episode's last frame (a stale start matches it). r14: also whether two consecutive episodes start in the SAME
scene (the r13 review's run on one fixed seed): the same instruction and the overview's frame 0 within 1.5 grey levels
of the previous episode's frame 0. Writes per-episode cut index (or none), the repeats and a summary.
usage: first_frame_check.py DATASET_DIR OUT_JSON [--cut 12]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2

D = Path(sys.argv[1]); OUT = sys.argv[2]; CUT = float(sys.argv[sys.argv.index("--cut") + 1]) if "--cut" in sys.argv else 12.0
eps = pd.concat([pd.read_parquet(p) for p in sorted((D / "meta/episodes").rglob("*.parquet"))]).sort_values("episode_index")
cams = sorted(p.name.replace("observation.images.", "") for p in (D / "videos").iterdir())
res = {"dataset": str(D), "cut_threshold_grey_levels": CUT, "episodes": []}
frames = {}
for cam in cams:
    seq = []
    for f in sorted((D / f"videos/observation.images.{cam}").rglob("*.mp4")):
        cap = cv2.VideoCapture(str(f))
        while True:
            ok, im = cap.read()
            if not ok: break
            seq.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32))
    frames[cam] = seq
starts = {}; idx = 0
for _, r in eps.iterrows():   # episodes are consecutive in the concatenated video files (LeRobot v3)
    starts[int(r.episode_index)] = (idx, int(r.length)); idx += int(r.length)
bad = {c: 0 for c in cams}
for e, (s, n) in starts.items():
    row = {"episode": e, "frames": n}
    for cam in cams:
        seq = frames[cam]
        if s + n > len(seq): row[cam] = {"error": "video shorter than the episode table"}; continue
        d = [float(np.abs(seq[s + k + 1] - seq[s + k]).mean()) for k in range(min(20, n - 1))]
        cut = next((k + 1 for k, v in enumerate(d) if v > CUT), None)
        prev = float(np.abs(seq[s] - seq[s - 1]).mean()) if s > 0 else None
        row[cam] = {"cut_at_frame": cut, "max_diff_first20": round(max(d), 2), "frame0_vs_previous_last": round(prev, 2) if prev is not None else None}
        if cut is not None: bad[cam] += 1
    res["episodes"].append(row)
tasks = pd.read_parquet(D / "meta/tasks.parquet").reset_index()
task_text = {int(r["task_index"]): str(r["task"] if "task" in r else r.iloc[0]) for _, r in tasks.iterrows()}
df0 = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
ep_task = {int(e): task_text.get(int(g.task_index.iloc[0])) for e, g in df0.groupby("episode_index")}
repeats = []
order = sorted(starts)
for prev, e in zip(order[:-1], order[1:]):
    cam = "overview" if "overview" in frames else cams[0]
    d0 = float(np.abs(frames[cam][starts[e][0]] - frames[cam][starts[prev][0]]).mean())
    same = d0 < 1.5 and ep_task.get(e) == ep_task.get(prev)
    if same: repeats.append({"episode": e, "previous": prev, "frame0_diff": round(d0, 2), "task": ep_task.get(e)})
res["same_scene_as_previous"] = repeats
res["summary"] = {"consecutive_same_scene": len(repeats), "episodes": len(starts), "episodes_with_a_cut_after_frame_0": bad,
                  "any_camera": sum(any(isinstance(r.get(c), dict) and r[c].get("cut_at_frame") for c in cams) for r in res["episodes"])}
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res["summary"]))
