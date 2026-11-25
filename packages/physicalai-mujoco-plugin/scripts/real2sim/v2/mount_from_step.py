"""r15 (D2, the user's mount): place the wrist camera mount's STEP geometry on the CAD gripper and derive the camera prior.

Inputs: the STEP's tessellation and features (step_to_mesh.py: $REAL2SIM_ROOT/mount/mesh.stl, features.json; mm), the
CAD gripper of the rebot_b601 profile (compiled bucket-v2 scene). No camera frame is read.
1. Bolt pattern. The mount's clamp has two M3 clearance holes (r 1.65) parallel to its z axis, 22 mm apart; the CAD
   gripper has two such pairs (cnc7 r 1.6 over pla7 r 1.35, parallel to the roll axis, link6 x = +-30, y = +-11). The
   mount goes on the +x pair (the side the desk and finger fits put the camera, r12 / r13). Rotation: the mount's z along
   the roll axis (its camera looks toward +z, the fingers), its y radial (+x), so x_link6 = y_step, y_link6 = -x_step,
   z_link6 = z_step. In-plane offset: the hole centres (least squares, both holes). Along the axis: the clamp's two plates
   (gap 20.4 mm) hold the cnc7 + pla7 stack (link6 z 63.0-83.1, 20.1 mm): centred in the gap.
2. Checks: hole residual (mm), the clamp's arc (r 28.65) against the roll axis, the gap against the stack, and the mount
   mesh against the gripper's tight collision pieces (vertices inside: count, depth).
3. Camera prior: the 42 x 42 x 23 mm camera's back sits on the cradle plate (the face opposite the screw counterbores,
   normal = the optical axis), centred on the cradle (its lips) and on its two M3 holes; the front glass 23 mm out; the
   colour origin (Gemini 305 datasheet 4.9 / 4.10: the left RGB module's optical centre, 3.95 mm behind the glass, the
   left module 9 mm left of centre, half the 18 mm baseline; image up away from the gripper as the profile has it).
   Compared with the kit's camera B (r13 fit) and v1's phone-scan lens midpoint, in mm and degrees.
Writes $REAL2SIM_ROOT/mount/mount_link6.json (transform, checks, prior pose, differences) and mount_link6.stl (metres,
link6). usage: mount_from_step.py"""
import json, os, sys
from pathlib import Path
import numpy as np, mujoco, trimesh
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); MD = ROOT / "mount"; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import bucketsim_v2 as b
from collision_decomp import inside_hulls
feats = json.load(open(MD / "features.json")); mesh = trimesh.load(MD / "mesh.stl", process=True)

# --- the mount's own features (mm, STEP frame)
cyl = feats["cylinders"]
bolt = [c for c in cyl if abs(c["radius"] - 1.65) < 0.02 and abs(c["axis"][2]) > 0.99]
holes_s = sorted({(round(c["point"][0], 3), round(c["point"][1], 3)) for c in bolt})          # axis points (x, y)
assert len(holes_s) == 2, holes_s
seat = [c for c in cyl if abs(c["radius"] - 1.65) < 0.02 and abs(c["axis"][2]) < 0.99]          # the camera's two M3 holes
cb = [c for c in cyl if abs(c["radius"] - 2.9) < 0.02]                                           # their counterbores
n_seat = -np.asarray(cb[0]["axis"], float); n_seat /= np.linalg.norm(n_seat)                      # sign fixed below
arc = [c for c in cyl if abs(c["radius"] - 28.65) < 0.02]
planes = feats["planes"]

# the cradle plate: the two large planes with normal +-n_seat; the camera's back on the one away from the counterbores
pl = [p for p in planes if abs(abs(np.dot(p["normal"], n_seat)) - 1) < 1e-3 and p["area"] > 1000]
pl.sort(key=lambda p: np.dot(p["centroid"], n_seat))
cb_level = np.mean([np.dot(c["centroid"], n_seat) for c in cb])
lv = [np.dot(p["centroid"], n_seat) for p in pl]
back_face = pl[int(np.argmax(np.abs(np.array(lv) - cb_level)))]                                   # farther from the counterbores
axis_s = np.asarray(back_face["normal"], float)                                                   # outward = the optical axis
seat_pts = np.array([c["point"] for c in seat]); seat_axis = np.asarray(seat[0]["axis"], float)
# hole centres on the back face plane
level = np.dot(back_face["centroid"], axis_s)
seat_c = np.unique(np.round(seat_pts + ((level - seat_pts @ axis_s) / np.dot(seat_axis, axis_s))[:, None] * seat_axis, 2), axis=0)
# the cradle's lips (planes perpendicular to the plate, across the camera's height): the inner faces bound the camera
u_dir = np.cross(axis_s, [1.0, 0, 0]); u_dir /= np.linalg.norm(u_dir)
lips = sorted([p for p in planes if abs(abs(np.dot(p["normal"], u_dir)) - 1) < 1e-3 and p["area"] > 300], key=lambda p: np.dot(p["centroid"], u_dir))
seat_centre_s = seat_c.mean(0)
lip_u = [np.dot(p["centroid"], u_dir) for p in lips]

# --- the CAD gripper (link6 frame, mm): hole loops parallel to the roll axis
robot = b.make_robot(0); m = robot._sim.model; d = mujoco.MjData(m); mujoco.mj_forward(m, d)
l6 = m.body("link6").id; R6 = d.xmat[l6].reshape(3, 3); p6 = d.xpos[l6]
def mesh_link6(name):
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name == name and m.geom_contype[g] == 0 and m.geom_rgba[g][3] > 0:
            mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; V = m.mesh_vert[a:a + n]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            W = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]; return (W - p6) @ R6 * 1000, m.mesh_face[fa:fa + fn]
    raise KeyError(name)
def hole_loops(V, F, rmin, rmax):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    E = E[(np.abs(V[E[:, 0], 2] - V[E[:, 1], 2]) < 0.02) & (np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) < 2.0)]
    k, lab = connected_components(sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))), directed=False)
    out = []
    for c in range(k):
        idx = np.flatnonzero(lab == c)
        if len(idx) < 6: continue
        P = V[idx, :2]; s = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P ** 2).sum(1), rcond=None)[0]; r = np.sqrt(s[2] + s[0] ** 2 + s[1] ** 2)
        if rmin < r < rmax and np.abs(np.hypot(P[:, 0] - s[0], P[:, 1] - s[1]) - r).max() < 0.15: out.append((s[0], s[1], r, V[idx, 2].mean()))
    return np.array(out)
Vc, Fc = mesh_link6("cnc7"); Vp, Fp = mesh_link6("pla7_green")
hc = hole_loops(Vc, Fc, 1.5, 1.7); hc = hc[(np.abs(hc[:, 0] - 30) < 2) & (np.abs(np.abs(hc[:, 1]) - 11) < 2)]   # the +x pair (link6 x 30, y +-11)
holes_6 = np.array([hc[hc[:, 1] > 0][:, :2].mean(0), hc[hc[:, 1] < 0][:, :2].mean(0)])
near = lambda V: V[(V[:, 0] > 20) & (V[:, 0] < 42) & (np.abs(V[:, 1]) < 20)]
stack = (near(Vc)[:, 2].min(), near(Vp)[:, 2].max())

# --- the transform: R (STEP -> link6) fixed by the layout; t from the holes (xy) and the stack (z)
R = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
hs = np.array([[x, y, 0.0] for x, y in holes_s]) @ R.T
hs = hs[np.argsort(-hs[:, 1])]; h6 = holes_6[np.argsort(-holes_6[:, 1])]
txy = (h6 - hs[:, :2]).mean(0); hole_res = np.linalg.norm(h6 - hs[:, :2] - txy, axis=1)
bot = [p for p in planes if p["normal"][2] > 0.999 and abs(p["centroid"][2] - 17.4) < 1.0]
top = [p for p in planes if p["normal"][2] < -0.999 and abs(p["centroid"][2] - 37.8) < 1.0]
gap = (bot[0]["centroid"][2], top[0]["centroid"][2])
tz = (stack[0] + stack[1]) / 2 - (gap[0] + gap[1]) / 2
t = np.r_[txy, tz]
to6 = lambda P: np.asarray(P, float) @ R.T + t
arc_c = to6(arc[0]["point"])[:2]

# mount mesh in link6 against the gripper's tight collision pieces
Vm = to6(mesh.vertices)
tight = json.load(open(Path(b.__file__).resolve().parents[3] / "src/physicalai_mujoco_plugin/profiles/rebot_b601_collision.json"))
hulls = []
for body in ("gripper_end", "link6"):
    if body not in tight["bodies"]: continue
    bid = m.body(body).id; Rb = d.xmat[bid].reshape(3, 3); pb = d.xpos[bid]
    for piece in tight["bodies"][body]["pieces"] if isinstance(tight["bodies"][body], dict) else tight["bodies"][body]:
        P = np.asarray(piece, float); hulls.append(((P @ Rb.T + pb) - p6) @ R6 * 1000)
ins = inside_hulls(Vm, hulls, eps=1e-6) if hulls else np.zeros(len(Vm), bool)

# --- camera prior (link6, metres)
back_c = seat_centre_s + (level - seat_centre_s @ axis_s) * axis_s
front_s = back_c + 23.0 * axis_s
up_s = u_dir if np.dot(to6(back_c + u_dir) - to6(back_c), [1, 0, 0]) > 0 else -u_dir          # image up: away from the axis
fwd6 = R @ axis_s; up6 = R @ up_s; x6 = np.cross(-up6, fwd6)                                   # image x = (-up) x fwd (y down)
front6 = to6(front_s); optical6 = front6 - 3.95 * fwd6 - 9.0 * x6
Rc = np.c_[x6, up6, -fwd6]                                                                      # MuJoCo camera: x right, y up, -z forward
q = Rot.from_matrix(Rc).as_quat(); quat = [q[3], q[0], q[1], q[2]]
kit = json.load(open(os.environ["KIT"])); camB = kit["cameras"]["wrist"]
def pose_diff(pos_m, quat_wxyz):
    dp = np.asarray(pos_m) * 1000 - optical6
    Rb_ = Rot.from_quat(np.r_[quat_wxyz[1:], quat_wxyz[0]]).as_matrix(); fb = -Rb_[:, 2]
    ang = np.degrees(np.arccos(np.clip(np.dot(fb, fwd6), -1, 1)))
    rel = Rot.from_matrix(Rc.T @ Rb_).as_rotvec(degrees=True)
    along = float(np.dot(dp, fwd6))
    return {"dpos_mm": dp.round(1).tolist(), "dist_mm": round(float(np.linalg.norm(dp)), 1), "along_axis_mm": round(along, 1),
            "across_axis_mm": round(float(np.linalg.norm(dp - along * fwd6)), 1), "axis_angle_deg": round(float(ang), 2),
            "rotation_vector_deg_cam_frame": rel.round(2).tolist(), "rotation_deg": round(float(np.linalg.norm(rel)), 2)}
out = {"source": "inputs/camera_mount/D405_305_Mount.step (user-supplied, Seeed reBot-DevArm b601-camera-mounts) via step_to_mesh.py; the profile's CAD gripper",
       "step_to_link6": {"R": R.tolist(), "t_mm": t.round(3).tolist(), "note": "p_link6_mm = R @ p_step_mm + t"},
       "checks": {"step_bolt_holes_xy_mm": holes_s, "cad_holes_link6_xy_mm": holes_6.round(2).tolist(), "hole_residual_mm": hole_res.round(2).tolist(),
                  "cad_hole_radius_mm": hc[:, 2].round(2).tolist(), "step_hole_radius_mm": 1.65,
                  "clamp_gap_step_z_mm": [round(g, 2) for g in gap], "clamp_gap_mm": round(gap[1] - gap[0], 2),
                  "stack_link6_z_mm": [round(s, 2) for s in stack], "stack_mm": round(stack[1] - stack[0], 2),
                  "arc_r28.65_centre_link6_xy_mm": arc_c.round(2).tolist(),
                  "mount_vertices_inside_gripper_collision": int(ins.sum()), "mount_vertices": int(len(Vm)),
                  "mount_bbox_link6_mm": Vm.min(0).round(1).tolist() + Vm.max(0).round(1).tolist()},
       "camera_prior": {"front_centre_link6_mm": front6.round(2).tolist(), "optical_centre_link6_mm": optical6.round(2).tolist(),
                        "forward_link6": fwd6.round(4).tolist(), "image_up_link6": up6.round(4).tolist(), "image_x_link6": x6.round(4).tolist(),
                        "pos_m": (optical6 / 1000).round(5).tolist(), "quat_wxyz": np.round(quat, 6).tolist(),
                        "assumptions": "Gemini 305 42 x 42 x 23 mm, back on the cradle plate, centred on its 2 M3 holes; colour origin 3.95 mm behind the glass, 9 mm left (half the 18 mm baseline); vertical offset of the RGB module 0"},
       "camera_B_minus_prior": pose_diff(camB["pos"], camB["quat"]),
       "v1_phone_scan_lens_midpoint_minus_prior_front_mm": (np.array([101.7, 3.1, 60.2]) - front6).round(1).tolist()}
json.dump(out, open(MD / "mount_link6.json", "w"), indent=1)
tm = trimesh.Trimesh(Vm / 1000.0, mesh.faces); tm.export(MD / "mount_link6.stl")
print(json.dumps({k: out[k] for k in ("step_to_link6", "checks", "camera_prior", "camera_B_minus_prior", "v1_phone_scan_lens_midpoint_minus_prior_front_mm")}, indent=1))
