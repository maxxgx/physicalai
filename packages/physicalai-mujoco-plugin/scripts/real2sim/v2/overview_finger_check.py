"""r16 (D3a, the coordinator's independent check): the fingers in the OVERVIEW (which does not depend on the wrist camera),
old gripper map against the new one, DEV 1, 3, 10 (TRAIN 0, 2, 6, 7, 9 reported too).

Allowed overview frames (frames_x5, states at t - latency) where the sim fingers show (>= 150 px). A window around the
sim fingers (both maps' masks, dilated 15 px, minus the other drawn robot parts dilated 3 px); in it the real fingers
are the dark pixels (HSV value < 70, saturation < 90: fit_finger_mount.py's rule), the sim fingers the segmentation of
the drawn finger geoms (every drawn geom occluding). Per map: IoU and the symmetric contour distance (px), medians.
usage: overview_finger_check.py STATES_PARQUET FRAMES_X5 OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b

ST, F, OUT = pd.read_parquet(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
kit = json.load(open(b.KIT)); lat = float(kit["camera_latency_s"]["overview"])
FIT = json.load(open(ROOT / "real/r16/fit_gripper_camera.json"))["fit"]
MAPS = {"old_r12": (4.64, 0.12125), "new_r16": (FIT["g0_mm"], FIT["s_mm_per_deg_per_finger"])}
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot); ch = robot._sim.channels[0]
cam = m.camera("overview").id; fadr = [m.jnt_qposadr[m.joint(n).id] for n in ("joint_left", "joint_right")]
seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering(); opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
fingers = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("gripper_left", "gripper_right") and m.geom_contype[g] == 0 and m.geom_group[g] < 3]
robot_drawn = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == m.body_rootid[m.body("gripper_end").id] and m.geom_contype[g] == 0 and m.geom_group[g] < 3 and g not in fingers]

def masks(state, opening):
    ch.set_positions(state)
    for a in fadr: d.qpos[a] = opening / 2000.0
    mujoco.mj_forward(m, d); seg.update_scene(d, camera=cam, scene_option=opt); s = seg.render()
    isg = s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return isg & np.isin(s[..., 0], fingers), isg & np.isin(s[..., 0], robot_drawn)

def contour_dist(a, b_):
    ca = [c[:, 0] for c in cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]]
    cb = [c[:, 0] for c in cv2.findContours(b_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]]
    if not ca or not cb: return np.nan
    from scipy.spatial import cKDTree
    pa, pb = np.vstack(ca), np.vstack(cb)
    return float((cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean()) / 2)

res = {"source": "review/export/robotcam_allowed (frames_x5 overview, states_allowed.parquet)", "maps": MAPS}
rng = np.random.default_rng(0)
for split, eps in (("train", (0, 2, 6, 7, 9)), ("dev", (1, 3, 10))):
    rows = {k: [] for k in MAPS}
    for ep in eps:
        g = ST[ST.episode_index == ep].sort_values("timestamp"); ts = g.timestamp.to_numpy(float); S = np.stack(g["observation.state"].to_numpy()).astype(float)
        files = sorted(F.glob(f"overview_e{ep:03d}_t*.png"))
        for f in [files[i] for i in sorted(rng.choice(len(files), size=min(30, len(files)), replace=False))]:
            t = int(f.stem.split("_t")[1]) / 1000.0; tl = float(np.clip(t - lat, ts[0], ts[-1]))
            state = np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])])
            mk = {k: masks(state, g0 + 2 * s * abs(state[-1])) for k, (g0, s) in MAPS.items()}
            union = np.logical_or.reduce([v[0] for v in mk.values()])
            if union.sum() < 150: continue
            other = cv2.dilate(np.logical_or.reduce([v[1] for v in mk.values()]).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            win = (cv2.dilate(union.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~other
            hsv = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2HSV); real = (hsv[..., 2] < 70) & (hsv[..., 1] < 90) & win
            for k, (fm, _) in mk.items():
                fm = fm & win; u = (fm | real).sum()
                rows[k].append((float((fm & real).sum() / u) if u else np.nan, contour_dist(fm, real)))
    res[split] = {k: {"frames": len(v), "iou_median": round(float(np.nanmedian([x[0] for x in v])), 3) if v else None,
                      "contour_px_median": round(float(np.nanmedian([x[1] for x in v])), 2) if v else None} for k, v in rows.items()}
    print(split, json.dumps(res[split]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
