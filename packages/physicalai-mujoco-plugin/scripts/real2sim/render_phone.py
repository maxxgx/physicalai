"""Render the phone's path through the real2sim scene: the splat (robot removed) with the MuJoCo arm and task objects
composited by depth, at each phone pose (SfM, interpolated to the video's 15 fps). Real | sim side by side; the real
half is black inside the reviewer's held-out windows (the sim half there is rendered at their registered poses).

usage: render_phone.py OUT.mp4 --splat PLY [--demo demo.npz | --matched] [--scale 0.5] [--frames a:b] [--names f00001,...]
  --matched: the arm at the calibrated static joints, no task objects (the real scene's state).
  --names:   render just these SfM frames (no video), to OUT_DIR/<name>.png: the HELDOUT_RENDER entry point."""
import argparse, json, subprocess, sys, time, cv2, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, held_t, pinhole_cameras
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--splat", required=True)
ap.add_argument("--demo"); ap.add_argument("--matched", action="store_true"); ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--frames"); ap.add_argument("--names"); ap.add_argument("--no-real", action="store_true"); a = ap.parse_args()

cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]
M, tw, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]; H_PLATE = -cal["desk_top_z"]
S_UNITS = 1.0 / sc
cams = pinhole_cameras(); names = sorted(cams); W0, H0 = cams[names[0]][2], cams[names[0]][3]; K0 = cams[names[0]][1]
W, H = int(round(W0 * a.scale)), int(round(H0 * a.scale)); K = K0.copy(); K[:2] *= a.scale
frames = json.load(open(ROOT / "colmap/frames.json")); t_of = {f"f{f['idx']:05d}.jpg": f["t"] for f in frames["frames"]}

# camera poses in the SfM frame: centres and camera-to-world rotations, interpolated over time
reg = sorted((t_of[n], n) for n in names if n in t_of)
ts = np.array([t for t, _ in reg]); C = np.array([-cams[n][0][:3, :3].T @ cams[n][0][:3, 3] for _, n in reg])
Rc2w = Rotation.from_matrix(np.array([cams[n][0][:3, :3].T for _, n in reg])); slerp = Slerp(ts, Rc2w)
def pose_at(t):
    t = float(np.clip(t, ts[0], ts[-1])); c = np.array([np.interp(t, ts, C[:, i]) for i in range(3)])
    R = slerp([t]).as_matrix()[0]; T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ c; return T

# the sim: the desk scene with only the robot and the task objects drawn (the splat shows the rest)
robot = MuJoCoRobot("rebot_b601", scene="bucket_sort_desk", cameras=[], http_port=0, seed=0); robot.connect()
m, d = robot._sim.model, robot._sim.data
root = m.body("base_link").rootid if hasattr(m.body("base_link"), "rootid") else m.body_rootid[m.body("base_link").id]
objects = {m.body(n).id for n in ("resistor", "battery", "cpu")}
# the arm's look, calibrated to the real arm in training frames (robot_look.py): material colours, the base mesh's own
LOOK = json.load(open(ROOT / "align/robot_look.json"))
for key, rgba in LOOK.items():
    if key == "geom:body:base_link":
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == m.body("base_link").id and m.geom_group[g] < 3: m.geom_matid[g] = -1; m.geom_rgba[g] = rgba
    elif not key.startswith("geom:"):
        m.mat_rgba[m.material(key).id] = rgba
for g in range(m.ngeom):
    b = m.geom_bodyid[g]
    if not (m.body_rootid[b] == m.body_rootid[m.body("base_link").id] or b in objects): m.geom_group[g] = 5
m.geom_group[m.geom("table_top").id] = 4  # the shadow catcher: drawn only in the shadow passes
opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
opt_desk = mujoco.MjvOption(); opt_desk.geomgroup[:] = [1, 1, 1, 0, 1, 0]
opt_bare = mujoco.MjvOption(); opt_bare.geomgroup[:] = [0, 0, 0, 0, 1, 0]
SHADOW = 0.75   # how dark a full sim shadow makes the desk (the real ones under the buckets are soft and light)
BLUR = 9        # px at the output scale: soft edges like the office's diffuse ceiling light
if a.matched:
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in objects: m.geom_group[g] = 5
rig, tilt, cam = m.body("overview_camera_rig").id, m.body("overview_camera_tilt").id, m.camera("overview").id
m.body_pos[tilt] = 0; m.body_quat[tilt] = [1, 0, 0, 0]; m.cam_pos[cam] = 0; m.cam_quat[cam] = [1, 0, 0, 0]
m.cam_fovy[cam] = np.degrees(2 * np.arctan(H / 2 / K[1, 1]))
m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
ren = mujoco.Renderer(m, H, W); dren = mujoco.Renderer(m, H, W); dren.enable_depth_rendering()
sren = mujoco.Renderer(m, H // 2, W // 2)
# one overhead light for the shadow passes, like the ceiling panels above the desk
m.light_castshadow[:] = 0; m.light_castshadow[0] = 1; m.light_pos[0] = [0.35, 0.10, 2.4]; m.light_dir[0] = [0, 0, -1]
m.light_cutoff[0] = 70; m.vis.quality.shadowsize = 4096
def set_camera(T):  # OpenCV world_to_cam in SfM -> the rig body's pose in the scene
    Rw = M @ T[:3, :3].T; cw = sc * M @ (-T[:3, :3].T @ T[:3, 3]) + tw + [0, 0, H_PLATE]
    q = np.zeros(4); mujoco.mju_mat2Quat(q, (Rw @ np.diag([1, -1, -1])).flatten()); m.body_pos[rig] = cw; m.body_quat[rig] = q
joint_q = {f"joint{i + 1}": v for i, v in enumerate(cal["robot"]["joints_rad"])}
demo = np.load(a.demo) if a.demo else None
def set_state(k):
    if demo is not None: d.qpos[:] = demo["qpos"][min(k, len(demo["qpos"]) - 1)]
    else:
        for n, v in joint_q.items(): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_forward(m, d)
splat = MetalSplatRenderer(a.splat)
def render(T, k):
    set_state(k); set_camera(T); mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    ren.update_scene(d, camera=cam, scene_option=opt); rgb = ren.render()
    dren.update_scene(d, camera=cam, scene_option=opt); z = dren.render()
    sp = splat.render(T, K, W, H, bg=(1.0, 1.0, 1.0)); bg = (np.clip(sp["rgb"], 0, 1) * 255).astype(np.uint8)
    zs = sp["median_depth"] / S_UNITS; zs[zs <= 0] = 1e9
    near = m.vis.map.znear * m.stat.extent; sim = (z < 0.999 * m.vis.map.zfar * m.stat.extent) & (z > near) & (z < zs + 0.01)
    # shadows: the desk with the sim drawn over the desk alone, at half resolution, blurred
    sren.update_scene(d, camera=cam, scene_option=opt_desk); a1 = sren.render().astype(np.float32).mean(2)
    sren.update_scene(d, camera=cam, scene_option=opt_bare); a0 = sren.render().astype(np.float32).mean(2)
    ratio = np.where(a0 > 20, np.clip(a1 / np.maximum(a0, 1), 0, 1), 1.0)
    ratio = cv2.GaussianBlur(cv2.resize(ratio, (W, H)), (0, 0), BLUR)
    shade = 1 - SHADOW * (1 - ratio)
    out = (bg.astype(np.float32) * shade[..., None]).astype(np.uint8); out[sim] = rgb[sim]; return out[:, :, ::-1]

if a.names:
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    for n in a.names.split(","): cv2.imwrite(str(od / (Path(n).stem + ".png")), render(cams[n][0], 0)); print("wrote", n)
    sys.exit(0)

# the real video at 15 fps, undistorted to the SfM's PINHOLE model (SIMPLE_RADIAL k1 from the model, 1920x1080)
K1 = 0.05045002030184394; F0 = 1684.0649756089904
yy, xx = np.mgrid[0:H0, 0:W0].astype(np.float32); xn, yn = (xx - K0[0, 2]) / K0[0, 0], (yy - K0[1, 2]) / K0[1, 1]
r2 = xn ** 2 + yn ** 2; mx = (F0 * xn * (1 + K1 * r2) + 960).astype(np.float32); my = (F0 * yn * (1 + K1 * r2) + 540).astype(np.float32)
n_frames = int(ts[-1] * 15)
lo, hi = (map(int, a.frames.split(":")) if a.frames else (0, n_frames))
dec = None if a.no_real else subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{lo / 15:.4f}", "-i", str(ROOT / "video/sdr_4k.mov"), "-vf", "fps=15,scale=1920:1080",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{(1 if a.no_real else 2) * W}x{H}", "-r", "15", "-i", "-",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
t0 = time.time()
for k in range(lo, hi):
    t = k / 15.0; sim = render(pose_at(t), k)
    if dec is not None:
        raw = dec.stdout.read(1920 * 1080 * 3)
        real = np.zeros_like(sim)
        if len(raw) == 1920 * 1080 * 3 and not held_t(t):
            real = cv2.resize(cv2.remap(np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3), mx, my, cv2.INTER_LINEAR), (W, H), interpolation=cv2.INTER_AREA)
        frame = np.hstack([real, sim])
    else: frame = sim
    cv2.putText(frame, f"t {t:5.1f}s", (10, 24), 0, 0.7, (255, 255, 255), 2)
    enc.stdin.write(np.ascontiguousarray(frame).tobytes())
    if k % 100 == 0: print(k, f"{(time.time() - t0) / max(1, k - lo + 1):.2f} s/frame", flush=True)
enc.stdin.close(); enc.wait(); print("wrote", a.out)
