"""r16 (D3a / D2, coordinator's conditions): the gripper's reading-to-opening map and the wrist camera, fitted together.

The r12/r13 map (opening = 4.64 mm + 0.2425 mm/deg x |reading|, 70.1 mm at -270) and camera B were fitted to each other;
from the camera pose the mount's STEP gives, the real fingers at -270 deg are 82-87 mm apart (open_width_check.py). Here,
on allowed wrist frames where the gripper holds still (fit_finger_mount.py's pick, open and closed), TRAIN 0, 2, 6, 7, 9
fitted, DEV 1, 3, 10 reported: opening(r) = g0 + 2 s |r| (each finger moves s mm per degree; s's prior the BOM's
module-1 16-tooth pinion 0.1396 mm/deg, bounded +-15 %; g0 = 4.64 +- 5 mm), and the wrist camera from the STEP seat
(mount_from_step.py) within 10 mm and 3 deg per axis, fovy 65-71 deg (Gemini 305 datasheet). Cost: the four finger edges
per row (330-470) and both tips (fit_wrist_joint.py's), truncated at 60 px. Reported: each parameter against its bound,
DEV edges and tips for (a) camera B with the old map, (b) the fit. usage: fit_gripper_camera.py STATES FRAMES_X5 OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent
OUT = sys.argv[3]; sys.argv = sys.argv[:3] + ["/dev/null"]
src = (HERE / "fit_finger_mount.py").read_text().split("p0 = np.zeros(6)")[0]
ns = {"__file__": str(HERE / "fit_finger_mount.py"), "__name__": "ffm"}; exec(compile(src, "fit_finger_mount.py", "exec"), ns)
tsrc = (HERE / "finger_tips.py").read_text(); exec(compile(tsrc[tsrc.index("def tips(mask):"):tsrc.index("def measure(frames, spec):")], "finger_tips.py", "exec"), ns)
m, d, cam = ns["m"], ns["d"], ns["cam"]
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
KIT = json.load(open(os.environ["KIT"])); B = KIT["cameras"]["wrist"]; PRI = json.load(open(ROOT / "mount/mount_link6.json"))["camera_prior"]
fadr = [m.jnt_qposadr[m.joint(n).id] for n in ("joint_left", "joint_right")]
S0, G0 = 0.1396, 4.64
train, dev = ns["train"], ns["dev"]
real = {f.name: (ns["edges"](mk), ns["tips"](mk)) for f, _ in train + dev for mk in [ns["real_mask"](cv2.imread(str(f)))]}

FIX_S = os.environ.get("FIX_S") == "1"   # variant: the slope held at the BOM value
S_VALUE = float(os.environ["S_MM_PER_DEG"]) if os.environ.get("S_MM_PER_DEG") else None
OPEN270 = float(os.environ["OPEN270_MM"]) if os.environ.get("OPEN270_MM") else None   # r16: a measured opening at -270 deg (mm)

def unpack(x):
    s = S0 if FIX_S else S0 * (1 + 0.15 * np.tanh(x[0])); g0 = G0 + 5.0 * np.tanh(x[1])
    if S_VALUE is not None:   # r16: the slope chosen on DEV holds (the camera then fitted with it)
        s = S_VALUE
    if OPEN270 is not None:   # r16: the user's ruler reading fixes the opening at -270 deg; the slope's split from the holds
        g0 = OPEN270 - 540.0 * s
    dp = 10.0 * np.tanh(x[2:5]); dr = 3.0 * np.tanh(x[5:8]); fovy = 68.0 + 3.0 * np.tanh(x[8])
    return s, g0, dp, dr, fovy

def set_camera(pos, quat, fovy, dp=np.zeros(3), dr=np.zeros(3)):
    m.cam_pos[cam] = np.asarray(pos) + dp / 1000.0
    r = np.radians(dr); ang = np.linalg.norm(r); dq = np.r_[np.cos(ang / 2), (np.sin(ang / 2) * r / ang) if ang > 0 else np.zeros(3)]
    q = np.zeros(4); mujoco.mju_mulQuat(q, np.asarray(quat, float), dq); m.cam_quat[cam] = q; m.cam_fovy[cam] = fovy

def measure(frames, opening):
    E, T = [], []
    for f, state in frames:
        ns["ch"].set_positions(state)
        if opening is not None:
            for a in fadr: d.qpos[a] = opening(state[-1]) / 2000.0
        mk = ns["sim_mask"](); re_, rt = real[f.name]
        E.append(ns["edges"](mk) - re_); T.append(ns["tips"](mk) - rt)
    return np.array(E), np.array(T)

def cost(E, T):
    return float(np.nanmean(np.minimum(np.abs(E), 60))) + float(np.nanmean(np.minimum(np.hypot(T[..., 0], T[..., 1]), 60)))

def report(E, T):
    return {"edge_median_abs_px": np.nanmedian(np.abs(E), axis=(0, 1)).round(1).tolist(), "tip_median_px": np.nanmedian(np.hypot(T[..., 0], T[..., 1]), axis=0).round(1).tolist(), "cost": round(cost(E, T), 2)}

res = {"source": "review/export/robotcam_allowed (frames_x5 gripper, states_allowed.parquet); still gripper frames",
       "train_frames": [f.name for f, _ in train], "dev_frames": [f.name for f, _ in dev], "readings_train": sorted({round(float(s[-1])) for _, s in train})}
set_camera(B["pos"], B["quat"], B["fovy"])
res["old_cameraB_oldmap"] = {"train": report(*measure(train, None)), "dev": report(*measure(dev, None))}
print("old (camera B, the profile's map)", json.dumps(res["old_cameraB_oldmap"]), flush=True)

def f(x):
    s, g0, dp, dr, fovy = unpack(x); set_camera(PRI["pos_m"], PRI["quat_wxyz"], fovy, dp, dr)
    return cost(*measure(train, lambda r: g0 + 2 * s * abs(r)))
fit = minimize(f, np.zeros(9), method="Nelder-Mead", options={"xatol": 0.005, "fatol": 0.02, "maxfev": 900,
               "initial_simplex": np.vstack([np.zeros(9)] + [np.eye(9)[i] * 0.4 for i in range(9)])})
s, g0, dp, dr, fovy = unpack(fit.x)
set_camera(PRI["pos_m"], PRI["quat_wxyz"], fovy, dp, dr); opening = lambda r: g0 + 2 * s * abs(r)
bounds = {"s_mm_per_deg": [S0 * 0.85, S0 * 1.15], "g0_mm": [G0 - 5, G0 + 5], "dpos_mm": [-10, 10], "drot_deg": [-3, 3], "fovy": [65, 71]}
frac = lambda v, lo, hi: float(abs(2 * (v - lo) / (hi - lo) - 1))   # 1 = on the bound
res["fit"] = {"s_mm_per_deg_per_finger": round(float(s), 5), "s_vs_bom_pct": round(100 * (s / S0 - 1), 2), "g0_mm": round(float(g0), 3),
              "opening_at_-270_mm": round(float(opening(-270)), 2), "opening_at_-278_mm": round(float(opening(-278)), 2),
              "camera_dpos_mm": dp.round(2).tolist(), "camera_drot_deg": dr.round(2).tolist(), "fovy": round(float(fovy), 3),
              "camera_pos_m": m.cam_pos[cam].round(6).tolist(), "camera_quat": m.cam_quat[cam].round(6).tolist(),
              "bound_use_0_centre_1_bound": {"s": round(frac(s, *bounds["s_mm_per_deg"]), 3), "g0": round(frac(g0, *bounds["g0_mm"]), 3),
                                             "dpos": [round(abs(v) / 10, 3) for v in dp], "drot": [round(abs(v) / 3, 3) for v in dr], "fovy": round(frac(fovy, *bounds["fovy"]), 3)},
              "train": report(*measure(train, opening)), "dev": report(*measure(dev, opening)), "evals": int(fit.nfev)}
print("fit", json.dumps(res["fit"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
