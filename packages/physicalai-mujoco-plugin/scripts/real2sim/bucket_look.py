"""Fit the four bucket materials' colours to the real buckets in training frames (median photo colour under each sim
bucket's eroded segmentation vs the sim render's, 3 rounds), under the compositor's lighting. Adds them to
align/robot_look.json (the wrist camera draws the sim buckets)."""
import json, sys, cv2, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, held, pinhole_cameras
import bucketsim
from compositor import Calibration, setup_lighting
cal = bucketsim.calibration(); robot = bucketsim.make_robot(0); bucketsim.matched_state(robot)
m, d = robot._sim.model, robot._sim.data; setup_lighting(m)
COL = ("black", "green", "red", "blue")
geoms = {c: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"bucket_{c}").id and m.geom_group[g] < 3 and "_item" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")] for c in COL}
keep = set(g for v in geoms.values() for g in v)
for g in range(m.ngeom):
    if g not in keep: m.geom_group[g] = 5
mat = {c: m.geom_matid[geoms[c][0]] for c in COL}
cams = pinhole_cameras(); names = [n for n in sorted(cams) if not held(n)][::15]
T0, K0, W0, H0 = cams[names[0]]; W, H = W0 // 2, H0 // 2
m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
rig, cam = m.body("render_camera_rig").id, m.camera("render").id
ren = mujoco.Renderer(m, H, W); seg = mujoco.Renderer(m, H, W); seg.enable_segmentation_rendering()
opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
def stats():
    real, sim = {c: [] for c in COL}, {c: [] for c in COL}
    for n in names:
        T, K, w, h = cams[n]; K2 = K.copy(); K2[:2] /= 2
        c_, R = cal.camera_to_scene(T); q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten()); m.body_pos[rig] = c_; m.body_quat[rig] = q
        m.cam_fovy[cam] = np.degrees(2 * np.arctan(H / 2 / K2[1, 1])); mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
        ren.update_scene(d, camera=cam, scene_option=opt); rgb = ren.render(); seg.update_scene(d, camera=cam, scene_option=opt); ids = seg.render()[..., 0]
        photo = cv2.cvtColor(cv2.resize(cv2.imread(str(ROOT / "colmap/brush_all/images" / n)), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        for c in COL:
            mk = cv2.erode(np.isin(ids, geoms[c]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            if mk.sum() > 300: real[c].append(np.median(photo[mk], 0)); sim[c].append(np.median(rgb[mk], 0))
    return {c: (np.median(real[c], 0), np.median(sim[c], 0), len(real[c])) for c in COL if real[c]}
for rnd in range(3):
    st = stats()
    for c, (r, s_, n_) in st.items(): m.mat_rgba[mat[c], :3] = np.clip(m.mat_rgba[mat[c], :3] * np.clip((r + 2) / (s_ + 2), 0.25, 4), 0, 1)
    print("round", rnd, {c: (r.round(0).tolist(), s_.round(0).tolist(), n_) for c, (r, s_, n_) in st.items()})
look = json.load(open(ROOT / "align/robot_look.json"))
for c in COL: look[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat[c])] = m.mat_rgba[mat[c]].tolist()
json.dump(look, open(ROOT / "align/robot_look.json", "w"), indent=1); print("saved", [k for k in look])
