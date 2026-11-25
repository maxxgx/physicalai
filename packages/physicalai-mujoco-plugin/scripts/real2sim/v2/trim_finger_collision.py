"""r16 (coordinator: a pre-existing breach of drawn = collision, fixed): the fingers' collision reached up to 2.56 mm past
their drawing (r13's decomposition left 68 pieces over its 0.9 mm tolerance; the r15 sliver extrusion added a little).
Per finger piece, while its hull pokes out of the drawn solid by more than TOL (its surface sampled at 0.4 mm, the
winding number for inside, the distance to the drawn surface), it is cut by the drawn surface's tangent plane at its
worst point, moved 0.2 mm inward (the convex piece cannot follow a concave drawing: cutting, not moving vertices). A
piece left thinner than MIN (1.5 mm, r15's multi-contact lesson) is then extruded along its thin axis toward the drawn
solid's inside, as thicken_slivers.py does. Writes profiles/rebot_b601_collision.json (a backup beside it first) and
prints the signed audit (collision_decomp.report) before / after per finger.
usage: trim_finger_collision.py PLUGIN_DIR [--tol-mm 0.8]"""
import json, os, re, shutil, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial import ConvexHull, cKDTree
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import bucketsim_v2 as b
from collision_decomp import winding, report, oriented_samples, hull_surface

plugin = Path(sys.argv[1]); TOL = float(sys.argv[sys.argv.index("--tol-mm") + 1]) / 1000 if "--tol-mm" in sys.argv else 0.0008
INSET, MIN = 0.0002, 0.0015
PROXY = re.compile(r"_plan\d+$|_pc\d+_\d+$")
robot = b.make_robot(0); b.reset(robot, 0); m = robot._sim.model

def qm(q):
    x = np.zeros(9); mujoco.mju_quat2Mat(x, q); return x.reshape(3, 3)

def drawn(body):
    bid = m.body(body).id; Vs, Fs, off = [], [], 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] or m.geom_conaffinity[g] or PROXY.search(m.geom(g).name or ""): continue
        if re.search(r"_hull\d+$|_tight\d+$|_item", m.geom(g).name or ""): continue
        mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        Vs.append(m.mesh_vert[a:a + n].astype(float) @ qm(m.geom_quat[g]).T + m.geom_pos[g]); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n
    return np.vstack(Vs), np.vstack(Fs)

def clip(P, n, c):
    """The convex hull of P cut to {x: x . n <= c}."""
    H = ConvexHull(P); s = P @ n - c; keep = P[s <= 0]
    edges = {tuple(sorted((a, bb))) for t in H.simplices for a, bb in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))}
    cut = [P[i] + (P[j] - P[i]) * (s[i] / (s[i] - s[j])) for i, j in edges if (s[i] > 0) != (s[j] > 0)]
    Q = np.vstack([keep] + ([np.array(cut)] if cut else []))
    return Q[ConvexHull(Q).vertices] if len(Q) >= 4 else None

def thin(P):
    c = P.mean(0)
    try: cand = ConvexHull(P).equations[:, :3]
    except Exception: cand = np.linalg.svd(P - c, full_matrices=False)[2]
    w = np.ptp((P - c) @ cand.T, axis=0); return cand[int(np.argmin(w))], float(w.min()), c

def trim(pieces, V, F):
    S, N = oriented_samples(V, F, 0.0003, np.random.default_rng(0)); tree = cKDTree(S); rng = np.random.default_rng(1)
    out, cuts, thick = [], 0, 0
    for p in pieces:
        P = np.asarray(p, float)
        for _ in range(12):
            Q = hull_surface(ConvexHull(P), 0.0004, rng); o = winding(Q, V, F) < 0.5
            if not o.any(): break
            d, i = tree.query(Q[o]); k = int(np.argmax(d))
            if d[k] <= TOL: break
            n = N[i[k]]; R = clip(P, n, float(S[i[k]] @ n) - INSET)
            if R is None: break
            P = R; cuts += 1
        n, t, c = thin(P)
        if t < MIN - 1e-9:   # as thicken_slivers.py: toward the drawn solid's inside
            w = winding(np.array([c + n * MIN, c - n * MIN]), V, F)
            side = 1.0 if w[0] > 0.5 >= w[1] else (-1.0 if w[1] > 0.5 >= w[0] else 0.0)
            h = (P - c) @ n; ext = MIN - t
            P = np.vstack([P + n * ext / 2 * (h >= np.median(h))[:, None], P - n * ext / 2 * (h < np.median(h))[:, None]]) if side == 0.0 else np.vstack([P, P + side * n * ext])
            P = P[ConvexHull(P).vertices]; thick += 1
        out.append(P.round(6).tolist())
    return out, cuts, thick

path = plugin / "src/physicalai_mujoco_plugin/profiles/rebot_b601_collision.json"
shutil.copyfile(path, path.with_name("rebot_b601_collision.pre_r16_trim.json.bak"))
C = json.load(open(path)); summary = {}
for body in ("gripper_left", "gripper_right"):
    V, F = drawn(body); e = C["bodies"][body]; before = report(V, F, [np.asarray(p, float) for p in e["pieces"]])
    e["pieces"], cuts, thick = trim(e["pieces"], V, F); after = report(V, F, [np.asarray(p, float) for p in e["pieces"]])
    e["trim_r16"] = {"tol_mm": TOL * 1000, "inset_mm": INSET * 1000, "cuts": cuts, "rethickened": thick, "report_before": before}; e["report"] = after
    summary[body] = {"cuts": cuts, "rethickened": thick, "before": before, "after": after}; print(body, json.dumps(summary[body]), flush=True)
json.dump(C, open(path, "w"))
