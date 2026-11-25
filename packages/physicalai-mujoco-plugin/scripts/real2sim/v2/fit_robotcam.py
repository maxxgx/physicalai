"""Fit a robot camera's intrinsics and pose from recorded frames (ALLOWED real episode frames, extracted by
real_episodes.py), against the reconstruction: per frame, the phone-path render (splat + CAD arm at the frame's joint
state) from the camera's current calibrated pose; SIFT matches real <-> render; the render's keypoints lifted to 3D
with the splat's median depth; then cv2.calibrateCamera over all frames (fx = fy, principal point free, k1 and k2,
no tangential). The overview is static: one pose (median of the per-frame poses). The wrist rides on link6: each
frame's pose is expressed in link6's frame (forward kinematics at the frame's joints) and the median taken.
Writes OUT.json: K, distortion, RMS, the pose (scene frame for the overview, link6 frame for the wrist) and the
change from the current calibration, and the exact frames used.
usage: fit_robotcam.py CAMERA FRAMES_DIR STATES.json OUT.json [--max 40]
  FRAMES_DIR: PNGs <camera>_e<episode>_t<ms>.png; STATES.json: {"<episode>/<ms>": [six joint angles, rad]}"""
import argparse, json, os, sys, numpy as np, cv2, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b

ap = argparse.ArgumentParser(); ap.add_argument("camera"); ap.add_argument("frames"); ap.add_argument("states"); ap.add_argument("out")
ap.add_argument("--max", type=int, default=40); a = ap.parse_args()
states = json.loads(Path(a.states).read_text()); states = states.get("states", states)
files = sorted(Path(a.frames).glob(f"{a.camera}_e*_t*.png"))
files = [f for f in files if f"{int(f.stem.split('_e')[1][:3])}/{int(f.stem.split('_t')[1])}" in states]
if len(files) > a.max: files = [files[i] for i in np.linspace(0, len(files) - 1, a.max).astype(int)]
robot = b.make_robot(0); b.matched_state(robot); m, d = robot._sim.model, robot._sim.data
from physicalai_mujoco_plugin.real2sim.desk_compositor import Compositor
H, W = 480, 640
comp = Compositor(m, d, b.kit(), width=W, height=H, show_objects=False)
cal = comp.cal; cam_name = {"gripper": "wrist"}.get(a.camera, a.camera); cid = m.camera(cam_name).id
jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]
sift = cv2.SIFT_create(4000); bf = cv2.BFMatcher(cv2.NORM_L2)
obj_pts, img_pts, used, link_T = [], [], [], []
K0 = comp.camera_K(cid)
for f in files:
    ep, ms = int(f.stem.split("_e")[1][:3]), int(f.stem.split("_t")[1])
    q = states[f"{ep}/{ms}"]
    for adr, v in zip(jadr, q): d.qpos[adr] = v
    mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    T = cal.scene_to_camera(d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3))
    rend = comp.render_sfm(T, K0)["rgb"]
    sp = comp.splat.render(T, K0, W, H, bg=(1.0, 1.0, 1.0), near=0.06 * cal.units_per_m)
    zs = sp["median_depth"]   # SfM units, camera z
    real = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE); ren = cv2.cvtColor(rend, cv2.COLOR_RGB2GRAY)
    k1, d1 = sift.detectAndCompute(real, None); k2, d2 = sift.detectAndCompute(ren, None)
    if d1 is None or d2 is None: continue
    good = [m_[0] for m_ in bf.knnMatch(d1, d2, k=2) if len(m_) == 2 and m_[0].distance < 0.75 * m_[1].distance]
    P, U = [], []
    for g in good:
        u, v = k2[g.trainIdx].pt; z = float(zs[int(v), int(u)])
        if not (z > 0 and np.isfinite(z)): continue
        Xc = np.array([(u - K0[0, 2]) / K0[0, 0] * z, (v - K0[1, 2]) / K0[1, 1] * z, z])
        Xw = np.linalg.inv(T[:3, :3]) @ (Xc - T[:3, 3])   # SfM frame
        P.append(Xw); U.append(k1[g.queryIdx].pt)
    if len(P) < 30: continue
    P, U = np.float32(P), np.float32(U)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(P, U, K0, None, reprojectionError=4.0, iterationsCount=2000)
    if not ok or inl is None or len(inl) < 25: continue
    obj_pts.append(P[inl[:, 0]]); img_pts.append(U[inl[:, 0]]); used.append(f.name)
    bid = m.body("link6").id; link_T.append((d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()))
print(a.camera, "frames with a pose:", len(used), "of", len(files), flush=True)
if len(used) < 3: raise SystemExit("too few frames matched")
flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (W, H), K0.copy(), np.zeros(5), flags=flags)
# poses: SfM camera-from-world per frame -> camera in the scene frame
poses = []
for rv, tv, (lp, lR) in zip(rvecs, tvecs, link_T):
    R, _ = cv2.Rodrigues(rv); Tsc = np.eye(4); Tsc[:3, :3] = R; Tsc[:3, 3] = tv[:, 0]
    c_sfm = -R.T @ tv[:, 0]; c_scene = cal.scale * c_sfm @ cal.M.T + cal.t + [0, 0, cal.plate]
    R_scene = cal.M @ R.T   # camera axes (OpenCV) in the scene frame
    if cam_name == "wrist":
        poses.append((lR.T @ (c_scene - lp), lR.T @ R_scene))   # in link6's frame
    else:
        poses.append((c_scene, R_scene))
cs = np.array([p for p, _ in poses]); c_med = np.median(cs, 0)
from scipy.spatial.transform import Rotation
Rm = Rotation.from_matrix(np.array([r for _, r in poses])).mean().as_matrix()
mujoco.mj_kinematics(m, d)
out = {"camera": a.camera, "K": K.tolist(), "fovy_deg": float(np.degrees(2 * np.arctan(H / 2 / K[1, 1]))), "dist": dist.ravel().tolist(),
       "rms_px": float(rms), "frames": len(used), "frame_list": used, "frame": "link6" if cam_name == "wrist" else "scene",
       "position_m": c_med.tolist(), "rotation_opencv_axes": Rm.tolist(), "position_spread_mm": (np.std(cs, 0) * 1000).round(1).tolist(),
       "K_before": K0.tolist()}
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("fovy_deg", "rms_px", "frames", "position_m", "position_spread_mm")}))
