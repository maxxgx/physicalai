"""r15 (D4, seed 648379): no convex collision piece thinner than MIN_MM.

MuJoCo 3.16's multi-contact (its default) reported -9.03 mm for a CPU grazing a 0.3 mm-thick content-item piece whose
true signed distance was -0.73 mm (sim/r15/d4_648379_ccd.log); the same piece made >= 1 mm thick gives the true depth
(-1.15 mm at 1 mm). Disabling multi-contact instead fixed the depth but let flat holds slip (sim/r15/slip_ablate.txt).
Here every piece thinner than MIN_MM (collision_decomp.thicken's 0.3 mm slabs, where the drawn surface is flat) is
extruded along its thin axis to MIN_MM, toward the drawn solid's inside (winding number of the two candidate points;
symmetric where neither side is inside), so it reaches no further past the drawing than before where the solid is at
least MIN_MM thick there. Every other piece stays byte-identical. Applies to the objects and content items (cpu,
battery: align/collision_r13.json), the bucket shell, the fingers and the housing (profiles/rebot_b601_collision.json)
and the wrist mount (profiles/rebot_b601_wrist_mount.json). Reports per body the pieces changed and the signed audit
(collision_decomp.report) before / after. Proxies: rerun build_plan_proxies.py (and this script refits the mount's).
usage: thicken_slivers.py PLUGIN_DIR [--min-mm 1.5]   (r15: 1.5, so the scaled-down content items stay ~1 mm)"""
import json, os, sys
from pathlib import Path
import numpy as np, mujoco
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import re; PROXY = re.compile(r"_plan\d+$|_pc\d+_\d+$")
import bucketsim_v2 as b
from collision_decomp import winding, report, inside_hulls, oriented_samples
from scipy.spatial import ConvexHull

plugin = Path(sys.argv[1]); ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
MIN = float(sys.argv[sys.argv.index("--min-mm") + 1]) / 1000 if "--min-mm" in sys.argv else 0.0015
robot = b.make_robot(0); b.reset(robot, 0); m = robot._sim.model

def qm(q):
    x = np.zeros(9); mujoco.mju_quat2Mat(x, q); return x.reshape(3, 3)

def drawn(body, mesh_name=None):
    bid = m.body(body).id; Vs, Fs, off = [], [], 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] or m.geom_conaffinity[g] or PROXY.search(m.geom(g).name or ""): continue
        if re.search(r"_hull\d+$|_tight\d+$|_item", m.geom(g).name or ""): continue
        if mesh_name and m.mesh(m.geom_dataid[g]).name != mesh_name: continue
        mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        Vs.append(m.mesh_vert[a:a + n].astype(float) @ qm(m.geom_quat[g]).T + m.geom_pos[g]); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n
    return np.vstack(Vs), np.vstack(Fs)

def fix(pieces, V, F):
    out, changed = [], 0
    for p in pieces:
        P = np.asarray(p, float); c = P.mean(0)
        try:   # the thinnest width: over the hull's face normals (the principal axis misses wedge-shaped pieces)
            H = ConvexHull(P); cand = H.equations[:, :3]
        except Exception:
            cand = np.linalg.svd(P - c, full_matrices=False)[2]
        w = np.ptp((P - c) @ cand.T, axis=0); n = cand[int(np.argmin(w))]; t = float(w.min())
        if t >= MIN - 1e-9:
            out.append(p); continue
        w = winding(np.array([c + n * MIN, c - n * MIN]), V, F)
        side = 1.0 if w[0] > 0.5 >= w[1] else (-1.0 if w[1] > 0.5 >= w[0] else 0.0)
        h = (P - c) @ n; ext = MIN - t
        if side == 0.0: Q = np.vstack([P + n * ext / 2 * (h >= np.median(h))[:, None], P - n * ext / 2 * (h < np.median(h))[:, None]])
        else: Q = np.vstack([P, P + side * n * ext])
        Q = Q[ConvexHull(Q).vertices]; out.append(Q.round(6).tolist()); changed += 1
    return out, changed

summary = {}
# objects and content items (one shape per kind), and the bucket shell
O_path = ROOT / "align/collision_r13.json"; O = json.load(open(O_path))
for kind, body, mesh in (("cpu", "obj_cpu_0", None), ("battery", "obj_battery_0", None), ("bucket", "bucket_red", "bucket")):
    V, F = drawn(body, mesh)
    if kind == "bucket":   # the shell at the bucket's height scale (build_collision_r13.py's frame)
        bid = m.body(body).id; Vs, Fs, off = [], [], 0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid and m.geom_contype[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name == "bucket":
                mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                Vs.append(m.mesh_vert[a:a + n].astype(float) @ qm(m.geom_quat[g]).T + m.geom_pos[g]); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n
        V, F = np.vstack(Vs), np.vstack(Fs)
    e = O["kinds"][kind]; before = report(V, F, [np.asarray(p, float) for p in e["pieces"]])
    e["pieces"], k = fix(e["pieces"], V, F); after = report(V, F, [np.asarray(p, float) for p in e["pieces"]])
    e["slivers_r15"] = {"min_mm": MIN * 1000, "thickened": k, "report_before": before}; e["report"] = after
    summary[kind] = {"thickened": k, "of": len(e["pieces"]), "before": before, "after": after}; print(kind, summary[kind], flush=True)
json.dump(O, open(O_path, "w"))
# the robot's tight pieces
R_path = plugin / "src/physicalai_mujoco_plugin/profiles/rebot_b601_collision.json"; R = json.load(open(R_path))
for body in ("gripper_left", "gripper_right", "gripper_end"):
    V, F = drawn(body); e = R["bodies"][body]; before = report(V, F, [np.asarray(p, float) for p in e["pieces"]])
    e["pieces"], k = fix(e["pieces"], V, F); after = report(V, F, [np.asarray(p, float) for p in e["pieces"]])
    e["slivers_r15"] = {"min_mm": MIN * 1000, "thickened": k, "report_before": before}; e["report"] = after
    summary[body] = {"thickened": k, "of": len(e["pieces"]), "before": before, "after": after}; print(body, summary[body], flush=True)
R_path.write_text(json.dumps(R))
# the wrist mount (link6 frame; its drawn mesh is in the file), with its proxies refitted
M_path = plugin / "src/physicalai_mujoco_plugin/profiles/rebot_b601_wrist_mount.json"; M = json.load(open(M_path)); mt = M["mount"]
V, F = np.asarray(mt["vertices"], float), np.asarray(mt["faces"]); before = report(V, F, [np.asarray(p, float) for p in mt["pieces"]])
mt["pieces"], k = fix(mt["pieces"], V, F); after = report(V, F, [np.asarray(p, float) for p in mt["pieces"]])
src = (HERE / "build_plan_proxies.py").read_text(); ns = {"np": np}
exec("from scipy.spatial import ConvexHull\nfrom collision_decomp import inside_hulls, thicken\n" + src[src.index("def proxies("):src.index("def drawn_samples(")], ns)
S, N = oriented_samples(V, F, 0.0008, np.random.default_rng(0))
px = ns["proxies"]([np.asarray(p, float) for p in mt["pieces"]], 0.006, S, N); mt["proxies"] = [p.round(6).tolist() for p in px]
mt["slivers_r15"] = {"min_mm": MIN * 1000, "thickened": k, "report_before": before}; mt["report"] = after
M_path.write_text(json.dumps(M)); summary["wrist_mount"] = {"thickened": k, "of": len(mt["pieces"]), "before": before, "after": after, "proxies": len(px)}
print("wrist_mount", summary["wrist_mount"], flush=True)
json.dump(summary, open(ROOT / "sim/r15/thicken_slivers.json", "w"), indent=1)
