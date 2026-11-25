"""Bucket fit error as a function of the scale s (poses re-fitted at each s), plus the rim height and footprint
read straight from the points: how strongly the buckets pin s."""
import sys, numpy as np
from pathlib import Path
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
import fit_buckets3d as F
import json
from common import ROOT
fit = json.load(open(ROOT / "align/buckets3d.json"))
p0 = []
for c in ("green", "red", "blue"):
    b = fit["buckets"][c]; p0 += [b["x"], b["y"], b["yaw"]]
for s in (3.0, 3.1, 3.2, 3.3, 3.4):
    r = minimize(lambda q: F.loss(np.r_[s, q]), np.array(p0), method="Powell", options={"xtol": 1e-5, "ftol": 1e-10})
    print("s", s, "rms mm", round(np.sqrt(r.fun / 3) * 1000, 2), flush=True)
for c in ("green", "red", "blue"):
    P = F.data[c]
    z = P[:, 2]; h = np.percentile(z, 99.5)
    # footprint across the bucket's short axis at the rim band (walls only), in the fitted yaw
    b = fit["buckets"][c]; cy, sy = np.cos(-b["yaw"]), np.sin(-b["yaw"])
    Q = (P[:, :2] / fit["s_units_per_m"] - [b["x"], b["y"]]) @ np.array([[cy, -sy], [sy, cy]]).T * fit["s_units_per_m"]
    band = (z > 0.6 * h)
    wx = np.percentile(Q[band, 0], 99.5) - np.percentile(Q[band, 0], 0.5)
    print(c, "rim units", round(h, 4), "-> s", round(h / 0.12, 3), "| width units", round(wx, 4), "-> s", round(wx / 0.150, 3))
