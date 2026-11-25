"""r16 (coordinator: a pre-existing breach of drawn = collision, fixed): the fingers' collision reached up to 2.56 mm past
their drawing because r13's decomposition stopped at its 96-piece cap with 68 pieces over its 0.9 mm tolerance (and r15's
sliver extrusion added a little). Re-decomposed from the drawn fingers (collision_decomp.decompose, PCA frame) with a
larger cap and a smaller smallest box; every piece thinner than 1.5 mm then extruded along its thin axis toward the drawn
solid's inside (thicken_slivers.py's rule, r15's multi-contact lesson). The planning proxies are kept. Prints the signed
audit (collision_decomp.report) before / after; with --write, writes profiles/rebot_b601_collision.json (backup beside).
usage: redecompose_fingers.py PLUGIN_DIR --pieces N [--tol-mm 1.0] [--min-box-mm 1.5] [--write]"""
import json, os, re, shutil, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial import ConvexHull
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
src = (HERE / "trim_finger_collision.py").read_text(); src = src[src.index("import bucketsim_v2"):src.index("def clip")]
plugin = Path(sys.argv[1]); exec(compile(src.replace("plugin = Path(sys.argv[1]); ", "plugin = Path(sys.argv[1]); _x = "), "trim_head", "exec"))
from collision_decomp import decompose, report, winding
arg = lambda k, d: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
N, TOLD, BOX = int(arg("--pieces", 160)), arg("--tol-mm", 1.0) / 1000, arg("--min-box-mm", 1.5) / 1000

def thin(P):
    c = P.mean(0)
    try: cand = ConvexHull(P).equations[:, :3]
    except Exception: cand = np.linalg.svd(P - c, full_matrices=False)[2]
    w = np.ptp((P - c) @ cand.T, axis=0); return cand[int(np.argmin(w))], float(w.min()), c

def thicken_inward(P, V, F):
    n, t, c = thin(P)
    if t >= MIN - 1e-9: return P, False
    w = winding(np.array([c + n * MIN, c - n * MIN]), V, F)
    side = 1.0 if w[0] > 0.5 >= w[1] else (-1.0 if w[1] > 0.5 >= w[0] else 0.0)
    h = (P - c) @ n; ext = MIN - t
    Q = np.vstack([P + n * ext / 2 * (h >= np.median(h))[:, None], P - n * ext / 2 * (h < np.median(h))[:, None]]) if side == 0.0 else np.vstack([P, P + side * n * ext])
    return Q[ConvexHull(Q).vertices], True

path = plugin / "src/physicalai_mujoco_plugin/profiles/rebot_b601_collision.json"; C = json.load(open(path)); out = {}
for body in ("gripper_left", "gripper_right"):
    V, F = drawn(body); e = C["bodies"][body]; before = e["report"]
    hulls, info = decompose(V, F, tol=TOLD, min_box=BOX, max_pieces=N)
    res = [thicken_inward(np.asarray(h, float), V, F) for h in hulls]; hulls = [p for p, _ in res]
    after = report(V, F, hulls); out[body] = {"info": info, "thickened": sum(k for _, k in res), "before": before, "after": after}
    print(body, json.dumps(out[body]), flush=True)
    e["pieces"] = [h.round(6).tolist() for h in hulls]; e["decomposition_r16"] = {**info, "min_box_m": BOX, "thickened_to_mm": MIN * 1000, "report_r13_r15": before}; e["report"] = after
if "--write" in sys.argv:
    shutil.copyfile(path, path.with_name("rebot_b601_collision.pre_r16_redecomp.json.bak")); json.dump(C, open(path, "w")); print("written")
