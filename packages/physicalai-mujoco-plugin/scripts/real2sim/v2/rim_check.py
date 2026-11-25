"""r16 (bucket height 120 mm, the user's statement): do the buckets' rims sit where the real ones do in the robot
cameras? On a dev_replay output (its allowed frames, real frames_x5 against the renders): per frame, camera and bucket
colour (red, green, blue: HSV masks, the same rule for real and render, blobs >= 300 px), the top edge per image column;
over the columns both show, the median of (real - render) rows (> 0: the real rim lower in the image). Per camera and
split (TRAIN 0, 2, 6, 7, 9 / DEV 1, 3, 10): the median signed offset and the median |offset| over frames, px.
usage: rim_check.py DEV_REPLAY_DIR FRAMES_X5 OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np, cv2
D, F, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
RULES = {"red": [((0, 120, 60), (8, 255, 255)), ((172, 120, 60), (180, 255, 255))], "green": [((40, 80, 40), (85, 255, 255))],
         "blue": [((95, 90, 40), (130, 255, 255))]}
def mask(img, c):
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); m = np.zeros(h.shape[:2], np.uint8)
    for lo, hi in RULES[c]: m |= cv2.inRange(h, np.array(lo), np.array(hi))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    if n <= 1 or st[1:, 4].max() < 300: return None
    return lab == 1 + int(np.argmax(st[1:, 4]))
def top(m):
    has = m.any(0); return np.where(has, m.argmax(0), -1)
meta = json.load(open(D / "metrics.json")); res = {}
for cam, rows in meta["frames"].items():
    for split, eps in (("train", {0, 2, 6, 7, 9}), ("dev", {1, 3, 10})):
        offs = []
        for r in rows:
            if r["episode"] not in eps: continue
            stem = f"{r['episode']:03d}_{r['t']:08.4f}"; ren = cv2.imread(str(D / "renders" / f"{stem}_{cam}.png"))
            real = cv2.imread(str(F / r["frame"].replace("overview_", f"{cam}_")))
            if ren is None or real is None: continue
            real = cv2.resize(real, (ren.shape[1], ren.shape[0]))
            for c in RULES:
                a, b = mask(real, c), mask(ren, c)
                if a is None or b is None: continue
                ta, tb = top(a), top(b); ok = (ta >= 0) & (tb >= 0)
                if ok.sum() >= 15: offs.append(float(np.median(ta[ok] - tb[ok])))
        res[f"{cam}_{split}"] = {"n": len(offs), "signed_px": round(float(np.median(offs)), 1) if offs else None, "abs_px": round(float(np.median(np.abs(offs))), 1) if offs else None}
print(json.dumps(res)); json.dump(res, open(OUT, "w"), indent=1)
