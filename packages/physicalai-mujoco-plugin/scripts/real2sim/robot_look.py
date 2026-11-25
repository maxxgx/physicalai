"""Calibrate the sim arm's material colours to the real arm in training frames (matched pose): per material, the
median photo colour under its eroded segmentation vs the median sim-rendered colour, as an RGB gain on the material's
rgba (3 rounds). Writes align/robot_look.json {material: rgba} and align/robot_look.jpg (photo | before | after)."""
import json, sys, cv2, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, held, pinhole_cameras
from physicalai_mujoco_plugin.robot import MuJoCoRobot

cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]
M, tw, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]; HP = -cal["desk_top_z"]
cams = pinhole_cameras(); names = [n for n in sorted(cams) if not held(n)][::15]
robot = MuJoCoRobot("rebot_b601", scene=__import__("os").environ.get("SCENE", "bucket_sort_desk"), cameras=[], http_port=0, seed=0); robot.connect()
m, d = robot._sim.model, robot._sim.data
from compositor import setup_lighting
setup_lighting(m)
base_root = m.body_rootid[m.body("base_link").id]
arm = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == base_root and m.geom_group[g] < 3]
# the base mesh (with its square flange) and the scene's base plate get their own colours: geom_rgba, no material
OWN = [g for g in arm if m.geom_bodyid[g] == m.body("base_link").id] + [m.geom("base_plate").id]
arm = sorted(set(arm) | set(OWN))
for g in OWN:
    if m.geom_matid[g] >= 0: m.geom_rgba[g] = m.mat_rgba[m.geom_matid[g]]
    m.geom_matid[g] = -1
for g in range(m.ngeom):
    if g not in arm: m.geom_group[g] = 5
for i, v in enumerate(cal["robot"]["joints_rad"]): d.qpos[m.jnt_qposadr[m.joint(f"joint{i + 1}").id]] = v
mujoco.mj_forward(m, d)
rig, tilt, cam = m.body("overview_camera_rig").id, m.body("overview_camera_tilt").id, m.camera("overview").id
m.body_pos[tilt] = 0; m.body_quat[tilt] = [1, 0, 0, 0]; m.cam_pos[cam] = 0; m.cam_quat[cam] = [1, 0, 0, 0]
T0, K0, W0, H0 = cams[names[0]]; W, H = W0 // 2, H0 // 2; K = K0.copy(); K[:2] /= 2
m.cam_fovy[cam] = np.degrees(2 * np.arctan(H / 2 / K[1, 1]))
m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
ren = mujoco.Renderer(m, H, W); seg = mujoco.Renderer(m, H, W); seg.enable_segmentation_rendering()
opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
arm = [g for g in arm if g in OWN or m.geom_matid[g] >= 0]   # the wrist camera's housing and mount keep their own colours
mat_of = {g: (int(m.geom_matid[g]) if g not in OWN else 1000 + g) for g in arm}
mats = sorted(set(mat_of.values()))
def name(k):  # a material, or "geom:<geom name or its body's name>"
    if k < 1000: return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k)
    g = k - 1000; return "geom:" + (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "body:" + m.body(m.geom_bodyid[g]).name)
def rgba(k): return m.mat_rgba[k] if k < 1000 else m.geom_rgba[k - 1000]
print("materials", [(name(k), rgba(k).round(2).tolist()) for k in mats])
def set_cam(T):
    Rw = M @ T[:3, :3].T; cw = sc * M @ (-T[:3, :3].T @ T[:3, 3]) + tw + [0, 0, HP]
    q = np.zeros(4); mujoco.mju_mat2Quat(q, (Rw @ np.diag([1, -1, -1])).flatten()); m.body_pos[rig] = cw; m.body_quat[rig] = q
    mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
def views():
    for n in names:
        T = cams[n][0]; set_cam(T)
        ren.update_scene(d, camera=cam, scene_option=opt); rgb = ren.render()
        seg.update_scene(d, camera=cam, scene_option=opt); ids = seg.render()[..., 0]
        photo = cv2.cvtColor(cv2.resize(cv2.imread(str(ROOT / "colmap/pinhole/images" / n)), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        yield n, rgb, ids, photo
def stats():
    real, sim = {k: [] for k in mats}, {k: [] for k in mats}
    for n, rgb, ids, photo in views():
        for k in mats:
            mask = np.isin(ids, [g for g in arm if mat_of[g] == k]).astype(np.uint8)
            mask = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
            if mask.sum() > 150: real[k].append(np.median(photo[mask], 0)); sim[k].append(np.median(rgb[mask], 0))
    return {k: (np.median(real[k], 0), np.median(sim[k], 0), len(real[k])) for k in mats if real[k]}
before = [cv2.cvtColor(v[1], cv2.COLOR_RGB2BGR) for v in list(views())[:4]]
for rnd in range(3):
    st = stats()
    for k, (r, s_, cnt) in st.items():
        gain = np.clip((r + 2) / (s_ + 2), 0.25, 4.0)
        rgba(k)[:3] = np.clip(rgba(k)[:3] * gain, 0.0, 1.0)
    print("round", rnd, {name(k): (r.round(0).tolist(), s_.round(0).tolist(), cnt) for k, (r, s_, cnt) in st.items()})
after = list(views())[:4]
rows = [np.hstack([cv2.cvtColor(v[3], cv2.COLOR_RGB2BGR), b, cv2.cvtColor(v[1], cv2.COLOR_RGB2BGR)]) for v, b in zip(after, before)]
cv2.imwrite(str(ROOT / "align/robot_look.jpg"), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
json.dump({name(k): rgba(k).tolist() for k in mats}, open(ROOT / "align/robot_look.json", "w"), indent=1)
