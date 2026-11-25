"""r13 item 4: every difference between what is drawn and what collides, in mm, signed (collision_decomp.report).

Per body kind of the scene as the recordings build it (bucketsim_v2.make_robot: the profile's gripper collision, the
scene's objects, buckets and contents): the drawn geoms (contype = conaffinity = 0, not the compositor's helpers) and
the collision geoms of the body, each collision geom as the point set MuJoCo hulls (a mesh's vertices, a box's corners,
a cylinder's rims, a sphere's or capsule's sampled surface), and:
  collision_beyond_drawn: the collision surface OUTSIDE the drawn solid, its distance to the drawn surface (an object
                          stops there with a visible gap);
  drawn_outside_collision: the drawn surface outside every collision piece, its distance to them (drawn, but nothing
                          there to touch).
median / p95 / max, mm (sampling ~0.6 mm). A bucket: its shell against its walls and floor; its content items as their
kind (pieces placed as the scene writes them, checked on one item per kind and bucket).
usage: audit_visual_collision.py OUT_JSON [--bodies a,b,...]"""
import re as _re; PROXY = _re.compile(r"_plan\d+$|_pc\d+_\d+$")   # r14 planning-only proxies
import json, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
from collision_decomp import report

OUT = sys.argv[1]; ONLY = sys.argv[sys.argv.index("--bodies") + 1].split(",") if "--bodies" in sys.argv else None
robot = b.make_robot(0); b.reset(robot, 0); m = robot._sim.model
GEOM = mujoco.mjtGeom


def qm(q):
    r = np.zeros(9); mujoco.mju_quat2Mat(r, q); return r.reshape(3, 3)


def to_body(g, P):
    return P @ qm(m.geom_quat[g]).T + m.geom_pos[g]


def collision_points(g):
    t, s = int(m.geom_type[g]), m.geom_size[g]
    if t == GEOM.mjGEOM_MESH:
        mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); P = m.mesh_vert[a:a + n].astype(float)
    elif t == GEOM.mjGEOM_BOX:
        P = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s
    elif t == GEOM.mjGEOM_CYLINDER:
        th = np.linspace(0, 2 * np.pi, 64, endpoint=False); ring = np.c_[s[0] * np.cos(th) / np.cos(np.pi / 64), s[0] * np.sin(th) / np.cos(np.pi / 64), np.zeros(64)]
        P = np.vstack([ring - [0, 0, s[1]], ring + [0, 0, s[1]]])
    elif t in (GEOM.mjGEOM_SPHERE, GEOM.mjGEOM_CAPSULE, GEOM.mjGEOM_ELLIPSOID):
        u = np.random.default_rng(0).normal(size=(400, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        P = u * (s[:3] if t == GEOM.mjGEOM_ELLIPSOID else s[0])
        if t == GEOM.mjGEOM_CAPSULE: P[:, 2] += np.where(P[:, 2] >= 0, s[1], -s[1])
    else:
        return None
    return to_body(g, P)


def drawn_mesh(gs):
    Vs, Fs, off = [], [], 0
    for g in gs:
        if m.geom_type[g] != GEOM.mjGEOM_MESH: continue
        mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        Vs.append(to_body(g, m.mesh_vert[a:a + n].astype(float))); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n
    return (np.vstack(Vs), np.vstack(Fs)) if Vs else (None, None)


def is_drawn(g):
    name = m.geom(g).name or ""
    return m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and not name.endswith(("wrist_mount", "wrist_housing")) and not PROXY.search(name)


res = {"method": __doc__.split("\n\n")[0], "bodies": {}}
seen = set()
for bid in range(1, m.nbody):
    name = m.body(bid).name; kind = name.rstrip("0123456789").rstrip("_")
    if ONLY and name not in ONLY: continue
    if kind in seen and not name.startswith("bucket_"): continue
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_type[g] != GEOM.mjGEOM_PLANE]
    col = [g for g in gs if not (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0)]
    drawn = [g for g in gs if is_drawn(g)]
    if not col or not drawn: continue
    seen.add(kind)
    if name.startswith("bucket_"):
        nm = lambda g: m.geom(g).name or ""
        shell_d = [g for g in drawn if "_item" not in nm(g)]; shell_c = [g for g in col if "_item" not in nm(g)]
        V, F = drawn_mesh(shell_d); rep = report(V, F, [collision_points(g) for g in shell_c], spacing=0.0006)
        res["bodies"][name] = {"part": "bucket shell", "collision": f"{len(shell_c)} geoms ({', '.join(sorted({GEOM(int(m.geom_type[g])).name[7:].lower() for g in shell_c}))})", **rep}
        print(name, "shell", rep, flush=True)
        items = sorted({nm(g).split("_item")[1].split("_")[0] for g in drawn if "_item" in nm(g)}, key=int)
        done_kinds = set()
        for k in items:
            idr = [g for g in drawn if nm(g).startswith(f"{name}_item{k}_")]; ic = [g for g in col if nm(g) == f"{name}_item{k}" or nm(g).startswith(f"{name}_item{k}_c")]
            mesh_kind = next((mm for mm in ("resistor", "cpu", "battery") if any(mm in (m.mesh(m.geom_dataid[g]).name or "") for g in idr)), "?")
            if mesh_kind in done_kinds or not ic: continue
            done_kinds.add(mesh_kind); V, F = drawn_mesh(idr); rep = report(V, F, [collision_points(g) for g in ic], spacing=0.0006)
            res["bodies"][f"{name}_item{k}"] = {"part": f"bucket content ({mesh_kind})", "collision": f"{len(ic)} pieces", **rep}
            print(f"{name}_item{k}", mesh_kind, rep, flush=True)
        continue
    V, F = drawn_mesh(drawn)
    if V is None: continue
    pts = [p for p in (collision_points(g) for g in col) if p is not None]
    rep = report(V, F, pts, spacing=0.0006)
    res["bodies"][name] = {"part": kind, "collision": f"{len(col)} geoms ({', '.join(sorted({GEOM(int(m.geom_type[g])).name[7:].lower() for g in col}))})", **rep}
    print(name, rep, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
json.dump(res, open(OUT, "w"), indent=1)
