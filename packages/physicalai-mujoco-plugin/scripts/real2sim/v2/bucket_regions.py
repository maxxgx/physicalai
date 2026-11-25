"""r13 (coordinator condition 2): the bucket's drawn surface against its collision, PER REGION where objects touch.

The drawn bucket shell (the scene's mesh at its height scale, bucket frame: open end +y, floor at z ~ 0..6 mm) sampled
with its face normals; each sample classed as
  inner floor: facing up (n_z > 0.7) and below 20 mm;
  rim:         within 6 mm of the top;
  inner wall:  facing the bucket's inside (horizontal normal toward the vertical axis through the floor's centre);
  outside:     the rest (outer faces, the underside).
Per region: drawn_outside_collision (drawn surface outside every piece, distance to the pieces) and
collision_beyond_drawn (the pieces' surface outside the drawn solid, its distance to the drawn surface, assigned to the
region of the nearest drawn point): median / p95 / max, mm.  usage: bucket_regions.py COLLISION_JSON OUT_JSON [--boxes]
(--boxes: the upstream five boxes of the r12 scene instead of the pieces)"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial import ConvexHull, cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
from collision_decomp import winding, hull_surface, outside_distance_hulls

COL, OUT = sys.argv[1], sys.argv[2]
robot = b.make_robot(0); m = robot._sim.model; bid = m.body("bucket_red").id
def qm(q):
    r = np.zeros(9); mujoco.mju_quat2Mat(r, q); return r.reshape(3, 3)
Vs, Fs, off = [], [], 0
for g in range(m.ngeom):
    if m.geom_bodyid[g] == bid and m.geom_contype[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name == "bucket":
        mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        Vs.append(m.mesh_vert[a:a + n].astype(float) @ qm(m.geom_quat[g]).T + m.geom_pos[g]); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n
V, F = np.vstack(Vs), np.vstack(Fs)
if "--boxes" in sys.argv:   # the r12 boxes (the scene's template), from the r12 scene copy
    import re
    xml = (Path.home() / "real2sim-data/bucket-v2/sim/scene_v2_pre_r13.xml").read_text()
    hulls = []
    for line in xml.splitlines():
        if 'class="bucket_collision" name="bucket_red_' not in line: continue
        pos = np.array([float(x) for x in re.search(r'pos="([^"]+)"', line)[1].split()]); size = np.array([float(x) for x in re.search(r'size="([^"]+)"', line)[1].split()])
        e = re.search(r'euler="([^"]+)"', line); R = np.eye(3)
        if e:
            ax = float(e[1].split()[0]); R = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        C = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
        hulls.append(C @ R.T + pos)
else:
    hulls = [np.array(p) for p in json.load(open(COL))["kinds"]["bucket"]["pieces"]]

rng = np.random.default_rng(0)
A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
k = int(A.sum() / 0.0007 ** 2); idx = rng.choice(len(F), k, p=A / A.sum()); u, v = rng.random(k), rng.random(k); f = u + v > 1; u[f], v[f] = 1 - u[f], 1 - v[f]
T = V[F[idx]]; S = T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0])
N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
# orient the normals outward of the solid: a point a little along +n is outside (winding < 0.5)
probe = winding(S[::50] + 0.0005 * N[::50], V, F); flip = np.median(probe) > 0.5
if flip: N = -N
top = V[:, 2].max(); floor_c = np.array([0.0, float(np.median(S[(N[:, 2] > 0.7) & (S[:, 2] < 0.02)][:, 1])) if ((N[:, 2] > 0.7) & (S[:, 2] < 0.02)).any() else 0.0])
to_axis = floor_c - S[:, :2]; inward = (N[:, :2] * to_axis).sum(1) > 0
region = np.full(len(S), "outside", dtype=object)
region[(N[:, 2] > 0.7) & (S[:, 2] < 0.02)] = "inner floor"
region[(region == "outside") & (np.abs(N[:, 2]) <= 0.7) & inward] = "inner wall"
region[S[:, 2] > top - 0.006] = "rim"
unc = outside_distance_hulls(S, hulls)
Q = np.vstack([hull_surface(ConvexHull(h), 0.0007, rng) for h in hulls]); w = winding(Q, V, F); o = w < 0.5
dq, nn = cKDTree(S).query(Q); prot = np.where(o, dq, 0.0); qreg = region[nn]
st = lambda x: {"median_mm": round(float(np.median(x)) * 1000, 2), "p95_mm": round(float(np.percentile(x, 95)) * 1000, 2), "max_mm": round(float(x.max()) * 1000, 2), "n": int(len(x))} if len(x) else None
res = {"collision": "upstream boxes (r12)" if "--boxes" in sys.argv else f"{len(hulls)} pieces ({COL})", "normals_flipped": bool(flip), "regions": {}}
for r in ("inner floor", "inner wall", "rim", "outside"):
    res["regions"][r] = {"drawn_outside_collision": st(unc[region == r]), "collision_beyond_drawn": st(prot[qreg == r])}
    print(r, json.dumps(res["regions"][r]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
