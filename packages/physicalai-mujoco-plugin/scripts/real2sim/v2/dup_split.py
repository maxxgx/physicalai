"""r15 (D7, coordinator): what the duplicates are made of. Per camera, among the frames dscheck counts as duplicates
(mean |diff| of 160x120 grey < 0.25 against the previous frame, same episode) while the arm moves (any joint channel
changes > 1e-4): EXACT repeats (the full-resolution decoded frame identical to the previous one: the source sent the
same image again, a stream-timing repeat) against NEAR repeats (a new image that changes almost nothing after the
codec: near-still content), each split by the arm's speed (largest joint change per frame, deg: < 0.1 near-still,
0.1-0.3, >= 0.3). Works on a Studio dataset or on the reviewer's allowed export (videos ep*_overview.mp4 /
ep*_gripper.mp4 + states_allowed.parquet).
usage: dup_split.py DATASET_DIR|EXPORT_DIR OUT_JSON   (Studio venv python: pandas, cv2)"""
import sys, json, numpy as np, pandas as pd, cv2
from pathlib import Path

D, OUT = Path(sys.argv[1]), sys.argv[2]
export = (D / "states_allowed.parquet").exists()
if export:
    df = pd.read_parquet(D / "states_allowed.parquet").sort_values(["episode_index", "timestamp"])
    cams = {"overview": "overview", "gripper": "gripper"}
    videos = {c: {int(p.name[2:5]): p for p in D.glob(f"ep*_{c}.mp4")} for c in cams}
else:
    df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
    info = json.load(open(D / "meta/info.json")); cams = {k.split(".")[-1]: k for k in info["features"] if k.startswith("observation.images.")}
res = {"source": str(D), "cameras": {}}
for cam in cams:
    rows = []
    for ep, g in df.groupby("episode_index"):
        S = np.stack(g["observation.state"].values).astype(float); dq = np.r_[np.inf, np.abs(np.diff(S, axis=0)).max(1)]
        dq_arm = np.r_[np.inf, np.abs(np.diff(S[:, :6], axis=0)).max(1)]
        if export:
            v = videos[cam].get(int(ep))
        else:
            vs = sorted((D / f"videos/{cams[cam]}").rglob("*.mp4")); v = None
        if export and v is None: continue
        rows.append((ep, dq, dq_arm, v))
    def stream(paths):
        for v in paths:
            c = cv2.VideoCapture(str(v))
            while True:
                ok, f = c.read()
                if not ok: break
                yield f
    src = None if export else stream(sorted((D / f"videos/{cams[cam]}").rglob("*.mp4")))   # episodes in order
    n_dup = n_mov = 0; cls = {"exact": {"<0.1": 0, "0.1-0.3": 0, ">=0.3": 0}, "near": {"<0.1": 0, "0.1-0.3": 0, ">=0.3": 0}}
    for ep, dq, dq_arm, v in rows:
        fr = stream([v]) if export else (next(src, None) for _ in range(len(dq)))
        prev = prevg = None
        for i, f in zip(range(len(dq)), fr):
            if f is None: break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev is not None and dq[i] > 1e-4:
                n_mov += 1
                if np.abs(g - prevg).mean() < 0.25:
                    n_dup += 1; kind = "exact" if np.array_equal(f, prev) else "near"
                    s = dq_arm[i]; cls[kind]["<0.1" if s < 0.1 else ("0.1-0.3" if s < 0.3 else ">=0.3")] += 1
            prev, prevg = f, g
    pct = lambda x: round(100.0 * x / max(n_mov, 1), 2)
    res["cameras"][cam] = {"moving_frames": n_mov, "dup_pct_while_moving": pct(n_dup),
                           "exact_pct": pct(sum(cls["exact"].values())), "near_pct": pct(sum(cls["near"].values())),
                           "exact_by_arm_speed_pct": {k: pct(v) for k, v in cls["exact"].items()},
                           "near_by_arm_speed_pct": {k: pct(v) for k, v in cls["near"].items()}}
    print(cam, json.dumps(res["cameras"][cam]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
