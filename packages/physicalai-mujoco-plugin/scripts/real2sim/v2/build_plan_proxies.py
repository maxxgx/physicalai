"""r14 (D7: speed): planning-only proxies. The demonstrator's clearance checks run on these, physics keeps the fine
collision pieces of r13 (built from the drawn meshes).

Each proxy is the convex hull of a group of a body's fine pieces (merged greedily, below), so it CONTAINS
those pieces: a path that clears the proxies cannot touch the physics geometry (the coordinator's condition; checked
here and by tests/test_plan_proxies.py with the signed audit's point-in-hull test). Pieces are merged greedily while the
merged hull reaches at most TOL beyond the drawn surface: fingers 3 mm, housing 6 mm, the bucket shell and the objects
5 mm. Written next to the pieces: profiles/rebot_b601_collision.json ("proxies" per body) and
align/collision_r13.json ("proxies" per kind and for the bucket). Reported: how far each proxy set reaches beyond the
drawn surface (p95 / max, mm), for information (proxies are conservative by design).
usage: build_plan_proxies.py PLUGIN_DIR"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collision_decomp import inside_hulls, thicken

plugin = Path(sys.argv[1]); ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
TOL = {"gripper_left": 0.003, "gripper_right": 0.003, "gripper_end": 0.006, "bucket": 0.005, "cpu": 0.005, "battery": 0.005, "resistor": 0.005}
BODY = {"gripper_left": "gripper_left", "gripper_right": "gripper_right", "gripper_end": "gripper_end", "bucket": "bucket_red", "cpu": "obj_cpu_0", "battery": "obj_battery_0", "resistor": "obj_resistor_0"}
import bucketsim_v2 as b
robot = b.make_robot(0); m = robot._sim.model


def proxies(pieces: list, tol: float, S: np.ndarray, N: np.ndarray, max_proxies: int | None = None) -> list:
    """Greedy merging: start from the fine pieces; repeatedly merge the pair of neighbouring proxies (5 nearest by
    centroid) whose merged hull reaches least beyond the drawn surface (samples S with outward normals N: a hull point
    on the outer side of its nearest sample, its distance), while that reach stays <= tol. Each proxy is a hull of
    whole pieces, so the proxies contain the pieces (asserted)."""
    import heapq
    from scipy.spatial import cKDTree
    tree = cKDTree(S); rng = np.random.default_rng(0)
    from collision_decomp import hull_surface
    def reach(P):
        Q = hull_surface(ConvexHull(P), 0.0012, rng); dist, k = tree.query(Q)
        out = ((Q - S[k]) * N[k]).sum(1) > 0.0006
        return float(dist[out].max()) if out.any() else 0.0
    cur = {i: thicken(np.asarray(p, float)) for i, p in enumerate(pieces)}
    cur = {i: P[ConvexHull(P).vertices] for i, P in cur.items()}
    nxt = len(cur); heap = []
    def push_pairs(i):
        ids = [j for j in cur if j != i]
        if not ids: return
        c = np.array([cur[j].mean(0) for j in ids]); d = np.linalg.norm(c - cur[i].mean(0), axis=1)
        for j in np.array(ids)[np.argsort(d)[:5]]:
            M = np.vstack([cur[i], cur[int(j)]]); r = reach(M)
            if r <= tol: heapq.heappush(heap, (r, min(i, int(j)), max(i, int(j))))
    for i in list(cur): push_pairs(i)
    while heap:
        r, i, j = heapq.heappop(heap)
        if i not in cur or j not in cur: continue
        M = np.vstack([cur.pop(i), cur.pop(j)]); M = M[ConvexHull(M).vertices]; cur[nxt] = M; push_pairs(nxt); nxt += 1
    # each vertex 50 um further from the proxy's centroid: the scene writes the pieces at 10 um (cylinders) and the
    # proxies at 1 um resolution; the margin keeps the containment exact after that rounding
    out = [c_ + (P - c_) * (1 + 5e-5 / np.maximum(np.linalg.norm(P - c_, axis=1, keepdims=True), 1e-9)) for P in cur.values() for c_ in [P.mean(0)]]
    V = np.vstack([np.asarray(p, float) for p in pieces]); assert inside_hulls(V, out, eps=1e-6).all(), "a proxy does not contain its pieces"
    return out


def drawn_samples(body: str):
    """The body's drawn surface (bucketsim's model; the bucket: its shell) with outward face normals."""
    import mujoco
    from collision_decomp import oriented_samples
    bid = m.body(body).id; Vs, Fs, off = [], [], 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] or m.geom_conaffinity[g] or m.geom_group[g] >= 3: continue
        if body.startswith("bucket_") and m.mesh(m.geom_dataid[g]).name != "bucket": continue
        mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        r = np.zeros(9); mujoco.mju_quat2Mat(r, m.geom_quat[g])
        Vs.append(m.mesh_vert[a:a + n].astype(float) @ r.reshape(3, 3).T + m.geom_pos[g]); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n
    return oriented_samples(np.vstack(Vs), np.vstack(Fs), 0.0008, np.random.default_rng(0))


def cyl_points(c):
    a, b, r = np.array(c["from"]), np.array(c["to"]), c["radius"]; ax = b - a; ax /= np.linalg.norm(ax)
    u = np.cross(ax, [1, 0, 0] if abs(ax[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); v = np.cross(ax, u)
    th = np.linspace(0, 2 * np.pi, 32, endpoint=False); ring = (np.outer(np.cos(th), u) + np.outer(np.sin(th), v)) * r / np.cos(np.pi / 32)
    return np.vstack([ring + a, ring + b])


rp = plugin / "src/physicalai_mujoco_plugin/profiles/rebot_b601_collision.json"; R = json.loads(rp.read_text())
for body, e in R["bodies"].items():
    S, N = drawn_samples(BODY[body]); e["proxies"] = [p.round(6).tolist() for p in proxies(e["pieces"], TOL[body], S, N)]
    print(body, len(e["pieces"]), "pieces ->", len(e["proxies"]), "proxies", flush=True)
rp.write_text(json.dumps(R))
op = ROOT / "align/collision_r13.json"; O = json.loads(op.read_text())
for kind, e in O["kinds"].items():
    pieces = [cyl_points(c) for c in e["cylinders"]] if "cylinders" in e else e["pieces"]
    S, N = drawn_samples(BODY[kind]); e["proxies"] = [p.round(6).tolist() for p in proxies(pieces, TOL[kind], S, N)]
    print(kind, len(pieces), "pieces ->", len(e["proxies"]), "proxies", flush=True)
op.write_text(json.dumps(O))
