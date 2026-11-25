"""D2 (r10): the fingers in the wrist camera, real against sim, on allowed frames (the reviewer's export only).

Measures, per image row (330-470), the four finger edges (left outer, left inner, right inner, right outer) in the real
frame (dark pixels connected to the bottom edge) and in the sim (segmentation of the finger geoms, the kit's wrist
camera, the state of t - latency from the export's states). Then fits a small change of the wrist camera's mount
(translation in mm and rotation in degrees, in its parent body) to those edges on the TRAIN episodes (0, 2, 6, 7, 9),
reports the edge error on the DEV episodes (1, 3, 10) before and after, and the desk alignment cost: the reprojection of
the camera's own SIFT fit (real/fit_gripper2.json correspondences) under the changed mount.
usage: fit_finger_mount.py STATES_PARQUET FRAMES_X5 OUT_JSON [--apply-thickness K]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b

ap = argparse.ArgumentParser(); ap.add_argument("states"); ap.add_argument("frames"); ap.add_argument("out")
ap.add_argument("--per-episode", type=int, default=6); a = ap.parse_args()
F = Path(a.frames); ROWS = np.arange(330, 471, 10)
TRAIN, DEV = (0, 2, 6, 7, 9), (1, 3, 10)
st = pd.read_parquet(a.states); lat = float(json.load(open(b.KIT))["camera_latency_s"]["wrist"])
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot)
# the drawn finger meshes (group < 3), taken before the compositor moves the groups (its hidden group holds the collision)
FINGERS = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("gripper_left", "gripper_right") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] < 3]
comp = b.compositor(robot, show_objects=False); ch = robot._sim.channels[0]
cam = m.camera("wrist").id; pos0, quat0 = m.cam_pos[cam].copy(), m.cam_quat[cam].copy()
seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering()
vopt = mujoco.MjvOption(); vopt.geomgroup[:] = 0; vopt.geomgroup[0] = 1

def edges(mask):
    out = np.full((len(ROWS), 4), np.nan)
    for i, y in enumerate(ROWS):
        xs = np.flatnonzero(mask[y]); L = xs[xs < 320]; R = xs[xs >= 320]
        if len(L): out[i, 0], out[i, 1] = L.min(), L.max()
        if len(R): out[i, 2], out[i, 3] = R.min(), R.max()
    return out

def real_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); mk = (hsv[..., 2] < 70) & (hsv[..., 1] < 90)
    n, lab, s, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8))
    return np.isin(lab, [i for i in range(1, n) if s[i, 1] + s[i, 3] >= 470 and s[i, 4] > 800])

def sim_mask():
    keep = m.geom_rgba.copy(); grp = m.geom_group.copy()
    m.geom_rgba[:, 3] = 0; m.geom_rgba[FINGERS, 3] = 1; m.geom_group[FINGERS] = 0
    try:
        mujoco.mj_forward(m, d); seg.update_scene(d, camera=cam, scene_option=vopt); s = seg.render()
    finally:
        m.geom_rgba[:] = keep; m.geom_group[:] = grp
    return (s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(s[..., 0], FINGERS)

# frames: per episode, ones where the gripper holds still (open or closed) and the bottom rows show both fingers
def pick(eps):
    out = []
    for ep in eps:
        rows = st[st["episode_index"] == ep].sort_values("timestamp"); ts = rows["timestamp"].to_numpy(float)
        S = np.stack(rows["observation.state"].to_numpy()).astype(float)
        files = sorted(F.glob(f"gripper_e{ep:03d}_t*.png")); cand = []
        for f in files:
            t = int(f.stem.split("_t")[1]) / 1000.0; tl = t - lat
            if not ts[0] <= tl <= ts[-1]: continue
            k = np.searchsorted(ts, tl); win = S[max(0, k - 6):k + 6]
            if np.ptp(win[:, -1]) > 1.0 or np.ptp(win[:, :-1], axis=0).max() > 1.0: continue   # gripper and arm still (deg)
            cand.append((f, np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])])))
        for i in np.linspace(0, len(cand) - 1, min(a.per_episode, len(cand))).astype(int): out.append(cand[i])
    return out

def measure(frames):
    sims = []
    for f, state in frames:
        ch.set_positions(state); sims.append(edges(sim_mask()))
    return np.array(sims)

real = {}
def set_mount(p):
    m.cam_pos[cam] = pos0 + np.asarray(p[:3]) / 1000.0
    q = np.zeros(4); r = np.radians(p[3:]); ang = np.linalg.norm(r)
    dq = np.r_[np.cos(ang / 2), (np.sin(ang / 2) * r / ang) if ang > 0 else np.zeros(3)]
    mujoco.mju_mulQuat(q, quat0, dq); m.cam_quat[cam] = q

def err(frames, p, per_edge=False):
    set_mount(p); sim = measure(frames); R = np.array([real[f.name] for f, _ in frames])
    e = np.abs(sim - R); ok = ~np.isnan(e)
    return (np.nanmean(np.where(ok, e, np.nan), axis=(0, 1)) if per_edge else float(np.nanmean(np.minimum(e, 60))))

train, dev = pick(TRAIN), pick(DEV)
for f, _ in train + dev: real[f.name] = edges(real_mask(cv2.imread(str(f))))
p0 = np.zeros(6)
if os.environ.get("FINGER_DEBUG"):
    set_mount(p0); sim = measure(dev + train)
    for (f, state), se in zip(dev + train, sim):
        e = np.abs(se - real[f.name]); print(f.name, "grip", round(float(state[-1]), 1), "median |err| per edge", np.nanmedian(e, axis=0).round(1), "row 410 real", real[f.name][8], "sim", se[8])
    sys.exit()
before = {"train": err(train, p0), "dev": err(dev, p0), "dev_per_edge_px": err(dev, p0, True).round(1).tolist()}
print("before", before, flush=True)
res = minimize(lambda p: err(train, p), p0, method="Nelder-Mead", options={"xatol": 0.05, "fatol": 0.02, "maxfev": 400, "initial_simplex": np.vstack([p0, np.eye(6) * np.r_[3, 3, 3, 2, 2, 2]])})
after = {"train": err(train, res.x), "dev": err(dev, res.x), "dev_per_edge_px": err(dev, res.x, True).round(1).tolist()}
print("after", after, "mount change (mm, deg)", res.x.round(2), flush=True)
json.dump({"source": "review/export/robotcam_allowed (frames_x5 gripper, states_allowed.parquet)", "rows": ROWS.tolist(),
           "edges": ["left outer", "left inner", "right inner", "right outer"], "latency_s": lat,
           "train_frames": [f.name for f, _ in train], "dev_frames": [f.name for f, _ in dev],
           "before_px": before, "after_px": after, "mount_change_mm_deg": res.x.round(3).tolist()}, open(a.out, "w"), indent=1)
