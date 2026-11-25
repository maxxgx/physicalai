"""r17 (R-56 / D2 H1): how far the drawn CAD fingers sit from the real fingers the arm splat recorded. At the splat's rest
pose (its links' joints, opening 0.09 m), the finger-labelled Gaussians (opacity > 0.5, scale < 3 mm; within 15 mm of the
drawn finger) against the drawn finger's surface (0.3 mm samples): the translation t that minimises the median distance of
(Gaussians + t) to the surface (a robust ICP, translation only), per finger, in the finger body's frame (x, y across the
rail, z along it). usage: finger_splat_offset.py OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
src = (HERE / "relabel_finger_gaussians.py").read_text(); src = src[:src.index("keep = solid(")]
OUT = sys.argv[1]; sys.argv = ["x", "/dev/null"]
g = {"__file__": str(HERE / "relabel_finger_gaussians.py"), "__name__": "r"}; exec(compile(src, "r", "exec"), g)
m, d0, W, f, names, drawn_mesh = g["m"], g["d0"], g["W"], g["f"], np.array(json.load(open(g["path"].with_name(g["path"].stem + "_links.json")))["links"]), None
from collision_decomp import oriented_samples
op = 1 / (1 + np.exp(-np.asarray(f["opacity_logits"], float))); sc = np.exp(np.asarray(f["log_scales"], float)).max(1) * g["cal"].scale
res = {}
for fb in ("gripper_left", "gripper_right"):
    bid = m.body(fb).id; Vs, Fs, off = [], [], 0
    for gg in range(m.ngeom):
        if m.geom_bodyid[gg] != bid or m.geom_contype[gg] != 0 or m.geom_group[gg] >= 3 or m.geom_type[gg] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid = m.geom_dataid[gg]; V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        Vs.append(V @ d0.geom_xmat[gg].reshape(3, 3).T + d0.geom_xpos[gg]); Fs.append(m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]] + off); off += len(V)
    S, _ = oriented_samples(np.vstack(Vs), np.vstack(Fs), 0.0003, np.random.default_rng(0)); tree = cKDTree(S)
    P = W[(names == fb) & (op > 0.5) & (sc < 0.003)]; P = P[tree.query(P)[0] < 0.015]
    t = np.zeros(3)
    for _ in range(30):
        dist, k = tree.query(P + t); keep = dist < np.percentile(dist, 80)
        t = t + np.median((S[k] - (P + t))[keep], axis=0)
    R = d0.xmat[bid].reshape(3, 3); dist = tree.query(P + t)[0]
    res[fb] = {"n": int(len(P)), "t_world_mm": (t * 1000).round(2).tolist(), "t_body_mm": (R.T @ t * 1000).round(2).tolist(),
               "median_dist_mm_before_after": [round(float(np.median(tree.query(P)[0])) * 1000, 2), round(float(np.median(dist)) * 1000, 2)]}
    print(fb, res[fb], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
