"""r17 (R-56 condition: drawn = collision for what the phone path now draws): the arm splat's finger Gaussians against the
fingers' collision pieces, in the finger body's frame (they ride on the finger, so at any opening, the CAL one included):
how far each Gaussian centre lies outside every collision piece (0 inside), mm: median / p95 / max, for the Gaussians of
opacity > 0.5 (what draws visibly) and for all. usage: finger_splat_vs_collision.py OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
src = (HERE / "relabel_finger_gaussians.py").read_text(); src = src[:src.index("keep = solid(")]
OUT = sys.argv[1]; sys.argv = ["x", "/dev/null"]
g = {"__file__": str(HERE / "relabel_finger_gaussians.py"), "__name__": "r"}; exec(compile(src, "r", "exec"), g)
from collision_decomp import outside_distance_hulls
from physicalai_mujoco_plugin.profiles import rebot_b601 as prof
m, d0, W, f = g["m"], g["d0"], g["W"], g["f"]; names = np.array(json.load(open(g["path"].with_name(g["path"].stem + "_links.json")))["links"])
C = json.load(open(prof.REBOT_B601_TIGHT_COLLISION))["bodies"]; op = 1 / (1 + np.exp(-np.asarray(f["opacity_logits"], float)))
st = lambda x: {"n": int(len(x)), "median_mm": round(float(np.median(x)) * 1000, 2), "p95_mm": round(float(np.percentile(x, 95)) * 1000, 2), "max_mm": round(float(x.max()) * 1000, 2)}
res = {}
for fb in ("gripper_left", "gripper_right"):
    bid = m.body(fb).id; R, p = d0.xmat[bid].reshape(3, 3), d0.xpos[bid]
    P = (W[names == fb] - p) @ R; o = op[names == fb]
    dist = outside_distance_hulls(P, [np.asarray(q, float) for q in C[fb]["pieces"]])
    res[fb] = {"opacity_over_0.5": st(dist[o > 0.5]), "all": st(dist)}; print(fb, res[fb], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
