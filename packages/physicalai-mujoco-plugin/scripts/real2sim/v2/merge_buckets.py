"""Write align/buckets3d.json (the schema make_calibration_v2.py reads) from the top-down rectangles
(align/buckets_topdown.json) at scale S: plane metres = units / S. Yaw: the CAD's +y (the open end, above the low lip)
along the measured long side, pointing toward the robot base (checked on allowed frames f01588, f01806: with the open
ends away from the base every sim outline showed its tall end wall where the real bucket's low lip is). The black bucket's long side is ambiguous in the render (its contents smear
over the rim): its axis is the one of its two that is closer to its neighbours' (red, blue) long sides.
usage: merge_buckets.py S_FRAME BASE_X_UNITS BASE_Y_UNITS   (S_FRAME: the plane-metre frame the robot fit used, 3.9)"""
import json, os, sys, numpy as np
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
S, bx, by = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
T = json.load(open(ROOT / "align/buckets_topdown.json")); B = json.load(open(ROOT / "align/buckets3d.json"))
def axis(a): return np.array([np.cos(a), np.sin(a)])
ref = np.mean([axis(2 * T["buckets"][c]["yaw_long"]) for c in ("red", "blue")], 0); ref_ang = np.arctan2(ref[1], ref[0]) / 2
out = {}
for c, r in T["buckets"].items():
    cen = np.array(r["centre_plane_units"]); a = r["yaw_long"]
    if c == "black":   # pick the rectangle side closer to the neighbours' long axis
        cand = [a, a + np.pi / 2]; a = min(cand, key=lambda t: abs(np.sin(t - ref_ang)))
    d = axis(a)
    if d @ (cen - [bx, by]) > 0: d = -d          # open end toward the robot base
    phi = np.arctan2(d[1], d[0])
    out[c] = {"x": float(cen[0] / S), "y": float(cen[1] / S), "yaw": float(phi - np.pi / 2)}
sides = [v for c in ("green", "red", "blue") for v in (T["buckets"][c]["s_short"], T["buckets"][c]["s_long"])]
B.update({"s_units_per_m": S, "buckets": out, "bucket_scale_units_per_m": float(np.mean(sides)),
          "bucket_scale_sd_pct": float(np.std(sides) / np.mean(sides) * 100), "bucket_scale_sides": sides,
          "per_bucket": {c: {"s": float(np.mean([T["buckets"][c]["s_short"], T["buckets"][c]["s_long"]])), "rms_mm": None} for c in out},
          "method": "straight-down splat render (no photo), rim outlines by colour, min-area rectangles; scale from the green, red and blue rims (CAD 150 x 202.5 mm), black excluded (contents smear over its rim)"})
json.dump(B, open(ROOT / "align/buckets3d.json", "w"), indent=1)
print({c: {k: round(v, 4) for k, v in b.items()} for c, b in out.items()}, "bucket scale", round(np.mean(sides), 4), "sd %", round(np.std(sides) / np.mean(sides) * 100, 2))
