"""r16 (D7, overview duplicates): how much consecutive frames differ, real against sim, by the arm's speed.

dscheck's measure: mean |diff| of consecutive 160x120 grey frames (INTER_AREA); a duplicate below 0.25. Per camera and
arm-speed bin (largest joint change per frame, deg: < 0.1, 0.1-0.3, 0.3-1, >= 1): the diff's p10 / p50 / p90 and the
share under 0.25; plus the share of frames whose full-resolution decoded image is identical to the previous one
(exact repeats). A Studio dataset (sim) or the reviewer's allowed export (real).
usage: frame_diff.py DATASET_OR_EXPORT_DIR OUT_JSON   (Studio venv python: pandas, cv2)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, cv2

D, OUT = Path(sys.argv[1]), sys.argv[2]
export = (D / "states_allowed.parquet").exists()
if export:
    df = pd.read_parquet(D / "states_allowed.parquet").sort_values(["episode_index", "timestamp"]); cams = ("overview", "gripper")
else:
    df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
    info = json.load(open(D / "meta/info.json")); cams = tuple(k.split(".")[-1] for k in info["features"] if k.startswith("observation.images."))

def stream(paths):
    for v in paths:
        c = cv2.VideoCapture(str(v))
        while True:
            ok, f = c.read()
            if not ok: break
            yield f

BINS = ((0, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 1e9))
res = {"source": str(D), "cameras": {}}
for cam in cams:
    rows = []   # (arm speed, diff, exact)
    src = None if export else stream(sorted((D / f"videos/observation.images.{cam}").rglob("*.mp4")))
    for ep, g in df.groupby("episode_index"):
        S = np.stack(g["observation.state"].values).astype(float); dq = np.r_[np.nan, np.abs(np.diff(S[:, :6], axis=0)).max(1)]
        fr = stream([D / f"ep{int(ep):03d}_{cam}.mp4"]) if export else (next(src, None) for _ in range(len(g)))
        prev = prevg = None
        for i, f in zip(range(len(g)), fr):
            if f is None: break
            gimg = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev is not None: rows.append((dq[i], float(np.abs(gimg - prevg).mean()), bool(np.array_equal(f, prev))))
            prev, prevg = f, gimg
    a = np.array([r[0] for r in rows]); dd = np.array([r[1] for r in rows]); ex = np.array([r[2] for r in rows])
    out = {}
    for lo, hi in BINS:
        sel = (a >= lo) & (a < hi)
        if sel.sum() < 5: continue
        out[f"{lo}-{hi if hi < 1e9 else 'inf'} deg"] = {"n": int(sel.sum()), "diff_p10_p50_p90": np.percentile(dd[sel], [10, 50, 90]).round(3).tolist(),
                                                      "dup_pct": round(100 * float((dd[sel] < 0.25).mean()), 2), "exact_pct": round(100 * float(ex[sel].mean()), 2)}
    res["cameras"][cam] = out
    print(cam, json.dumps(out), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
