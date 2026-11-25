"""r13 item 1 + 4: collision built from what is drawn (collision_decomp.decompose), for every body that touches objects.

- reBot fingers (gripper_left, gripper_right) and the gripper housing (gripper_end): convex pieces of their drawn CAD
  meshes, in the body frame -> src/physicalai_mujoco_plugin/profiles/rebot_b601_collision.json (the profile's
  customize step replaces the Menagerie collision meshes with these).
- The bucket shell (r13, after the audit: the five upstream boxes missed the tapered drawn walls by up to 27 mm): convex
  pieces of the drawn bucket mesh at the scene's height scale.
- The CPU and the 9 V battery: convex pieces of their drawn meshes at the scene's scale, in the object frame;
  the resistor: coaxial cylinders following the drawn radial profile (it rolls; segments where the radius varies by
  <= tol) -> ALIGN/collision_r13.json, which build_scene_v2.py writes into the scene (pool objects and bucket contents).
Every piece set comes with its report (collision beyond the drawn surface / drawn surface outside collision, mm).
usage: build_collision_r13.py PLUGIN_DIR [--only a,b] [--tol MM] [--pieces N]"""
import re as _re; PROXY = _re.compile(r"_plan\d+$|_pc\d+_\d+$")   # r14 planning-only proxies
import json, os, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
from collision_decomp import decompose, report

plugin = Path(sys.argv[1]); ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
TOL = float(sys.argv[sys.argv.index("--tol") + 1]) / 1000 if "--tol" in sys.argv else 0.0012
PIECES = int(sys.argv[sys.argv.index("--pieces") + 1]) if "--pieces" in sys.argv else 64
robot = b.make_robot(0); b.reset(robot, 0); m = robot._sim.model


def qm(q):
    x = np.zeros(9); mujoco.mju_quat2Mat(x, q); return x.reshape(3, 3)


def drawn_mesh(body: str):
    bid = m.body(body).id; Vs, Fs, off = [], [], 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and not PROXY.search(m.geom(g).name or ""):
            mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            Vs.append(m.mesh_vert[a:a + n].astype(float) @ qm(m.geom_quat[g]).T + m.geom_pos[g]); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n
    return np.vstack(Vs), np.vstack(Fs)


def resistor_cylinders(V, F, tol):
    """Coaxial cylinders along the drawn resistor's long axis: the radial profile in 0.5 mm bins, merged greedily while
    the radius range in a segment stays <= tol; each segment's radius is its largest drawn radius."""
    c = (V.max(0) + V.min(0)) / 2; ax = int(np.argmax(V.max(0) - V.min(0)))
    from collision_decomp import sample_surface
    S = sample_surface(V, F, 0.0003, np.random.default_rng(0)); x = S[:, ax]; rr = np.linalg.norm(np.delete(S - c, ax, axis=1), axis=1)
    edges = np.arange(x.min(), x.max() + 0.0005, 0.0005); prof = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        sel = (x >= a_) & (x < b_)
        if sel.any(): prof.append((a_, b_, float(rr[sel].max())))
    segs = []
    for a_, b_, r in prof:
        if segs and max(segs[-1][3], r) - min(segs[-1][4], r) <= tol:
            s = segs[-1]; segs[-1] = (s[0], b_, max(s[2], r), max(s[3], r), min(s[4], r))
        else:
            segs.append((a_, b_, r, r, r))
    cyl = []
    for a_, b_, r, _, _ in segs:
        p0, p1 = c.copy(), c.copy(); p0[ax], p1[ax] = a_, b_; cyl.append({"from": p0.round(6).tolist(), "to": p1.round(6).tolist(), "radius": round(r, 6)})
    # pieces as points for the report: each cylinder sampled on its rims
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False); hulls = []
    for cy in cyl:
        p0, p1, r = np.array(cy["from"]), np.array(cy["to"]), cy["radius"]; others = [i for i in range(3) if i != ax]
        ring = np.zeros((48, 3)); ring[:, others[0]] = r * np.cos(th) / np.cos(np.pi / 48); ring[:, others[1]] = r * np.sin(th) / np.cos(np.pi / 48)
        hulls.append(np.vstack([ring + p0, ring + p1]))
    return cyl, hulls


out_robot = plugin / "src/physicalai_mujoco_plugin/profiles/rebot_b601_collision.json"
out_obj = ROOT / "align/collision_r13.json"
R = json.loads(out_robot.read_text()) if out_robot.exists() else {"bodies": {}}
O = json.loads(out_obj.read_text()) if out_obj.exists() else {"kinds": {}}
for name in ("gripper_left", "gripper_right", "gripper_end"):
    if ONLY and name not in ONLY: continue
    V, F = drawn_mesh(name); t = time.time()
    hulls, info = decompose(V, F, tol=TOL, max_pieces=PIECES)
    rep = report(V, F, hulls); print(name, info, rep, "%.0fs" % (time.time() - t), flush=True)
    R["bodies"][name] = {"pieces": [h.round(6).tolist() for h in hulls], "decomposition": info, "report": rep}
    R["method"] = __doc__.split("\n\n")[0] + " The pieces are convex point sets in the body frame (MuJoCo takes their hulls)."
    out_robot.write_text(json.dumps(R))
for kind, body in (("cpu", "obj_cpu_0"), ("battery", "obj_battery_0"), ("resistor", "obj_resistor_0"), ("bucket", "bucket_red")):
    if ONLY and kind not in ONLY: continue
    V, F = drawn_mesh(body); t = time.time()
    if kind == "bucket":   # the shell alone (bucket_red has no contents), in the bucket's frame (height scale applied)
        bid = m.body(body).id; Vs, Fs, off = [], [], 0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid and m.geom_contype[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name == "bucket":
                mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                Vs.append(m.mesh_vert[a:a + n].astype(float) @ qm(m.geom_quat[g]).T + m.geom_pos[g]); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n
        V, F = np.vstack(Vs), np.vstack(Fs)
    if kind == "resistor":
        cyl, hulls = resistor_cylinders(V, F, TOL); info = {"cylinders": len(cyl)}; rep = report(V, F, hulls)
        O["kinds"][kind] = {"cylinders": cyl, "decomposition": info, "report": rep}
    else:
        hulls, info = decompose(V, F, tol=TOL, max_pieces=PIECES, **({"inside": "normal", "min_box": 0.004} if kind == "bucket" else {})); rep = report(V, F, hulls)
        O["kinds"][kind] = {"pieces": [h.round(6).tolist() for h in hulls], "decomposition": info, "report": rep}
    print(kind, info, rep, "%.0fs" % (time.time() - t), flush=True)
    O["frame"] = "the object's body frame at the scene's scale (align/object_size.json)"; out_obj.write_text(json.dumps(O))
