"""Refine the static arm's six joints so the CAD's lime shells (link2, link3, link5, gripper body) match the arm's lime
silhouette in ALLOWED phone frames (photos are used for fitting only). The base pose (calibration) and the gripper
opening stay. Per frame: the real mask = lime pixels near the arm's projection; the sim mask = the lime meshes' faces at
joint vector q; loss = median over frames of 1 - IoU. Writes align/arm_photo.json (joints, frame list, IoU before and
after).  usage: fit_arm_photo.py [STEP]"""
import json, os, sys, numpy as np, cv2, mujoco
from pathlib import Path
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import held
import bucketsim_v2 as b

ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); DS = 4; STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 3
cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]; M, t, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
plate = -cal["desk_top_z"]
cams = json.load(open(ROOT / "colmap/pinhole_cameras.json"))
robot = b.make_robot(0); b.matched_state(robot); m, d = robot._sim.model, robot._sim.data
jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]
q0 = np.array([d.qpos[a] for a in jadr])
lime = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "lime")
geoms = [g for g in range(m.ngeom) if m.geom_matid[g] == lime and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
meshes = []
for g in geoms:
    mid = m.geom_dataid[g]
    V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    F = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    meshes.append((g, V, F))


def world_meshes(q):
    for a, v in zip(jadr, q): d.qpos[a] = v
    mujoco.mj_kinematics(m, d)
    out = []
    for g, V, F in meshes:
        W = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]; W[:, 2] -= plate
        out.append(((W - t) @ M / sc, F))
    return out


def sil(q, cam, w, h):
    T = np.array(cam["cam_from_world"]); K = np.array(cam["K"]) / [[DS], [DS], [1]]
    out = np.zeros((h, w), np.uint8)
    for P, F in world_meshes(q):
        Xc = P @ T[:, :3].T + T[:, 3]; z = Xc[:, 2]
        if (z <= 0.05).any(): return None
        uv = (Xc[:, :2] / z[:, None]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        cv2.fillPoly(out, list(np.round(uv[F] * 8).astype(np.int32)), 1, shift=3)
    return out


names = sorted(n for n in cams if not held(n))[::STEP]
data = []
for n in names:
    cam = cams[n]; w, h = cam["w"] // DS, cam["h"] // DS
    s = sil(q0, cam, w, h)
    if s is None or s.sum() < 400: continue
    ys, xs = np.nonzero(s)
    if xs.min() < 2 or ys.min() < 2 or xs.max() > w - 3 or ys.max() > h - 3: continue   # the lime parts wholly in view
    img = cv2.resize(cv2.imread(str(ROOT / "colmap/pinhole/images" / n)), (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    near = cv2.dilate(s, np.ones((31, 31), np.uint8)) > 0
    real = ((hsv[..., 0] >= 28) & (hsv[..., 0] <= 45) & (hsv[..., 1] > 120) & (hsv[..., 2] > 110) & near).astype(np.uint8)
    real = cv2.morphologyEx(real, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    iou = (real & s).sum() / max((real | s).sum(), 1)
    if real.sum() < 200 or iou < 0.25: continue   # the arm hidden (hand, out of focus) or another lime object
    data.append((n, cam, w, h, real))
print("frames", len(data), flush=True)


def per(q):
    L = []
    for n, cam, w, h, real in data:
        s = sil(q, cam, w, h)
        L.append(1.0 if s is None else 1 - (real & s).sum() / max((real | s).sum(), 1))
    return np.array(L)


loss = lambda q: float(np.median(per(q)) * 0.6 + np.mean(np.sort(per(q))[: max(1, int(0.8 * len(data)))]) * 0.4)
r = minimize(loss, q0, method="Nelder-Mead", options={"xatol": 2e-4, "fatol": 1e-4, "maxiter": 1500,
             "initial_simplex": q0 + np.r_[np.zeros((1, 6)), np.eye(6) * np.radians(2.0)]})
before, after = per(q0), per(r.x)
out = {"method": __doc__.split("\n")[0], "joints_rad_before": q0.tolist(), "joints_rad": r.x.tolist(), "delta_deg": np.degrees(r.x - q0).round(2).tolist(),
       "frames": len(data), "frame_list": [n for n, *_ in data], "iou_median_before": float(1 - np.median(before)), "iou_median_after": float(1 - np.median(after))}
json.dump(out, open(ROOT / "align/arm_photo.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("delta_deg", "frames", "iou_median_before", "iou_median_after")}))
