"""r16 (D3b): the arm splat's Gaussians that sit inside a finger's drawn solid at the splat's rest pose but ride on
another link (gripper_end, link6, the other finger) stay where the real fingers stood (opening 0.09 m) when the sim
fingers move: the "real finger remnants" beside the drawn fingers in close phone views. Relabelled to the finger they lie
in: inside that finger's drawn meshes (voxels filled, 1 mm, dilated one voxel) and inside no drawn mesh of gripper_end or
link6 (which keep what lies in them). Writes OUT_STEM.ply (a copy) and OUT_STEM_links.json; prints the counts.
usage: relabel_finger_gaussians.py OUT_STEM   (kit robot_gs -> OUT_STEM.ply)"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, mujoco, trimesh
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
from physicalai_mujoco_plugin.real2sim.splats import read_ply_fields
OUT = Path(sys.argv[1]); PITCH = 0.001; DILATE = int(os.environ.get("DILATE", "1"))   # voxels (mm)
robot = b.make_robot(0); b.matched_state(robot); m = robot._sim.model
comp = b.compositor(robot, 640, 480, show_objects=False); cal = comp.cal; path = comp.kit.robot_gs
meta = json.load(open(path.with_name(path.stem + "_links.json"))); names = np.array(meta["links"], dtype=object)
f = read_ply_fields(path); X = np.asarray(f["means"], float)
W = (X * cal.scale) @ cal.M.T + (cal.t + np.array([0, 0, cal.plate]))   # SfM -> world, as the compositor's inverse
d0 = mujoco.MjData(m); d0.qpos[:] = robot._sim.data.qpos
for i, v in enumerate(meta["joints_rad"][:6]): d0.qpos[m.jnt_qposadr[m.joint(f"joint{i + 1}").id]] = v
for n in ("joint_left", "joint_right"): d0.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["joints_rad"][6] / 2
mujoco.mj_forward(m, d0)

def solid(body, dilate):
    """The body's drawn meshes at the rest pose, as a test: points -> inside (filled voxels)."""
    bid = m.body(body).id; tests = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_contype[g] != 0 or m.geom_group[g] >= 3 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]; V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        F = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        Vw = V @ d0.geom_xmat[g].reshape(3, 3).T + d0.geom_xpos[g]
        vg = trimesh.Trimesh(Vw, F, process=False).voxelized(PITCH).fill(); occ = vg.matrix
        if dilate: occ = ndimage.binary_dilation(occ, iterations=dilate)
        tests.append((occ, np.linalg.inv(vg.transform)))
    def inside(P):
        out = np.zeros(len(P), bool)
        for occ, inv in tests:
            ijk = np.rint(P @ inv[:3, :3].T + inv[:3, 3]).astype(int)
            ok = np.all((ijk >= 0) & (ijk < occ.shape), axis=1)
            out[np.flatnonzero(ok)[occ[tuple(ijk[ok].T)]]] = True
        return out
    return inside

keep = solid("gripper_end", 0)(W) | solid("link6", 0)(W)
new = names.copy(); counts = {}
for fb in ("gripper_left", "gripper_right"):
    hit = solid(fb, DILATE)(W) & ~keep & (names != fb)
    counts[fb] = {str(k): int((hit & (names == k)).sum()) for k in np.unique(names[hit])}; new[hit] = fb
print("relabelled:", counts, "of", len(names))
# r17 (R-56, f03484's right finger drawn twice): Gaussians riding on a finger but more than DROP_MM outside its drawn solid
# at the rest pose are a second copy of the finger beside the CAD one; they get the label "dropped" (the compositor
# draws no Gaussian so labelled)
DROP = float(os.environ.get("DROP_MM", "0")); dropped = {}
if DROP > 0:
    for fb in ("gripper_left", "gripper_right"):
        far = (new == fb) & ~solid(fb, int(round(DROP)))(W); dropped[fb] = int(far.sum()); new[far] = "dropped"
    print("dropped:", dropped)
# r17 (R-56 condition: drawn = collision): with the CAD fingers left out of a phone view's colour, the splat's finger
# Gaussians ARE the drawn fingers; those whose centre lies more than DROP_COLLISION_MM outside every finger collision
# piece (finger body frame; they ride on the finger, so at any opening) are labelled "dropped" too
DROPC = float(os.environ.get("DROP_COLLISION_MM", "0"))
if DROPC > 0:
    from collision_decomp import outside_distance_hulls
    from physicalai_mujoco_plugin.profiles import rebot_b601 as prof
    CP = json.load(open(prof.REBOT_B601_TIGHT_COLLISION))["bodies"]
    for fb in ("gripper_left", "gripper_right"):
        bid = m.body(fb).id; R, pb = d0.xmat[bid].reshape(3, 3), d0.xpos[bid]; idx = np.flatnonzero(new == fb)
        dist = outside_distance_hulls((W[idx] - pb) @ R, [np.asarray(q, float) for q in CP[fb]["pieces"]])
        far = idx[dist > DROPC / 1000.0]; new[far] = "dropped"; dropped[fb + "_vs_collision"] = int(len(far))
    print("dropped:", dropped)
shutil.copyfile(path, OUT.with_suffix(".ply"))
json.dump({**meta, "links": [str(v) for v in new], "r16_relabel": {"rule": __doc__.split("usage")[0].strip(), "from": str(path.name), "counts": counts},
                                    "r17_dropped": {"outside_drawn_mm": DROP, "outside_collision_mm": DROPC, "counts": dropped}},
          open(OUT.with_name(OUT.name + "_links.json"), "w"))
