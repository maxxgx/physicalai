"""D1 (r11, U-4: the buckets never moved): why the per-episode bucket fits differ, from allowed overview frames only.
(1) the spread of the per-episode fits (align/bucket_layouts_episodes.json); (2) camera drift: the shift of each
episode's temporal median against episode 0's on static regions only (the room above the desk and the desk's near
part, buckets and the arm's reach masked), by phase correlation; (3) nudges: each bucket's colour-mask centroid and
area in the first vs the last 5 frames of every episode (the arm away from the bucket in both, checked by the
mask area). usage: layout_spread.py FRAMES_X5 OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np, cv2
F = Path(sys.argv[1]); ROOT = Path.home() / "real2sim-data/bucket-v2"
lay = json.load(open(ROOT / "align/bucket_layouts_episodes.json"))["episodes"]
EPS = sorted(int(e) for e in lay); COLORS = ("black", "green", "red", "blue")
res = {"source": "review/export/robotcam_allowed overview frames (frames_x5)", "fits": {}, "drift_px": {}, "nudges": {}}
# (1) spread
for c in COLORS:
    P = np.array([[lay[str(e)]["poses"][c]["x"], lay[str(e)]["poses"][c]["y"], np.degrees(lay[str(e)]["poses"][c]["yaw"])] for e in EPS])
    res["fits"][c] = {"x_range_mm": round(float(np.ptp(P[:, 0])) * 1000, 1), "y_range_mm": round(float(np.ptp(P[:, 1])) * 1000, 1), "yaw_range_deg": round(float(np.ptp(P[:, 2])), 1),
                      "per_episode_mm_deg": {e: [round(float(v), 4) if i < 2 else round(float(v), 1) for i, v in enumerate(p)] for e, p in zip(EPS, P)}}
    print("fit spread", c, res["fits"][c]["x_range_mm"], "mm x,", res["fits"][c]["y_range_mm"], "mm y,", res["fits"][c]["yaw_range_deg"], "deg")
def frames(ep): return sorted(F.glob(f"overview_e{ep:03d}_t*.png"))
def median(fs): return np.median(np.stack([cv2.imread(str(f)) for f in fs]), axis=0).astype(np.uint8)
meds = {e: median(frames(e)[:: max(1, len(frames(e)) // 25)]) for e in EPS}
static = np.zeros((480, 640), np.float64); static[:150, :] = 1.0   # the room above the desk (no bucket, no arm reach)
win = cv2.createHanningWindow((640, 480), cv2.CV_64F) * static
g0 = cv2.cvtColor(meds[EPS[0]], cv2.COLOR_BGR2GRAY).astype(np.float64)
for e in EPS:
    g = cv2.cvtColor(meds[e], cv2.COLOR_BGR2GRAY).astype(np.float64)
    (dx, dy), r = cv2.phaseCorrelate(g0.copy(), g.copy(), win.copy())
    res["drift_px"][e] = [round(dx, 2), round(dy, 2), round(r, 3)]
print("camera drift vs e0 (dx, dy, peak):", res["drift_px"])
def masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); h, s, v = (hsv[..., i].astype(int) for i in range(3)); out = {}
    out["red"] = ((h < 10) | (h > 168)) & (s > 110) & (v > 60); out["green"] = (h >= 60) & (h <= 95) & (s > 70) & (v > 60)
    out["blue"] = (h >= 100) & (h <= 130) & (s > 110) & (v > 50)
    for c in ("red", "green", "blue"):
        n, lab, st, _ = cv2.connectedComponentsWithStats(out[c].astype(np.uint8)); out[c] = lab == (1 + int(np.argmax(st[1:, 4]))) if n > 1 else out[c]
    return out
for e in EPS:
    fs = frames(e); a, b = masks(median(fs[:5])), masks(median(fs[-5:])); row = {}
    for c in ("red", "green", "blue"):
        ya, xa = np.nonzero(a[c]); yb, xb = np.nonzero(b[c])
        row[c] = {"centroid_shift_px": [round(float(xb.mean() - xa.mean()), 1), round(float(yb.mean() - ya.mean()), 1)], "area_start_end": [int(a[c].sum()), int(b[c].sum())],
                  "iou": round(float((a[c] & b[c]).sum() / max((a[c] | b[c]).sum(), 1)), 3)}
    res["nudges"][e] = {"start_frames": [f.name for f in fs[:5]], "end_frames": [f.name for f in fs[-5:]], **row}
    print("episode", e, "start->end", {c: (row[c]["centroid_shift_px"], row[c]["iou"]) for c in row})
json.dump(res, open(sys.argv[2], "w"), indent=1)
