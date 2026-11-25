"""r15 (D2, the user's mount): the wrist camera at the mount STEP's pose against camera B, on allowed frames only.

Poses (link6; the arm at each frame's recorded state, t - latency; the unit's joint zeros as the kit has them):
  B     the kit's camera B (r13 finger fit)
  P     the STEP prior (mount_from_step.py: the camera seat, the Gemini 305's colour origin), fovy of camera B
  P68   the STEP prior, the datasheet's fovy (68 deg)
  PF    the STEP prior plus a small fit: translation (mm), rotation (deg) and fovy, fitted on TRAIN (0, 2, 6, 7, 9) to the
        fingers (four edges per row, both tips; fit_wrist_joint.py's measure) AND the buckets (contour distance; each
        episode's stored layout); DEV (1, 3, 10) reported, never fitted
Measures per pose and split: the finger edges and tips (median px), and per bucket (red, green, blue) the IoU, the
contour distance (px) and the signed centroid offset sim - real (px, dx, dy) with its spread: one direction for every
bucket and frame points at the camera, a per-bucket direction at the layout.
usage: wrist_step_eval.py STATES_PARQUET FRAMES_X5 OUT_JSON [--maxfev N]"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
HERE = Path(__file__).resolve().parent
OUT = sys.argv[3]; MAXFEV = int(sys.argv[sys.argv.index("--maxfev") + 1]) if "--maxfev" in sys.argv else 300
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); DATASHEET = "--datasheet" in sys.argv
sys.argv = sys.argv[:3] + ["/dev/null"]
src = (HERE / "fit_finger_mount.py").read_text().split("p0 = np.zeros(6)")[0]
ns = {"__file__": str(HERE / "fit_finger_mount.py"), "__name__": "ffm"}; exec(compile(src, "fit_finger_mount.py", "exec"), ns)
tsrc = (HERE / "finger_tips.py").read_text(); exec(compile(tsrc[tsrc.index("def tips(mask):"):tsrc.index("def measure(frames, spec):")], "finger_tips.py", "exec"), ns)
wsrc = (HERE / "wrist_bucket_edges.py").read_text()
wns = dict(ns); wns["__name__"] = "wbe"; import re as _re; wns["PROXY"] = _re.compile(r"_plan\d+$|_pc\d+_\d+$")
wns["LAY"] = json.load(open(Path(os.environ["REAL2SIM_ROOT"]).expanduser() / "align/bucket_layouts_episodes.json"))["episodes"]
exec(compile(wsrc[wsrc.index("COL = "):wsrc.index("jadr4 = ")], "wrist_bucket_edges.py", "exec"), wns)
sys.path.insert(0, str(HERE)); import bucketsim_v2 as b
from scipy.optimize import minimize
m, d, cam, st, F, lat = ns["m"], ns["d"], ns["cam"], ns["st"], ns["F"], ns["lat"]
COL = wns["COL"]; LAY = wns["LAY"]
KIT = json.load(open(os.environ["KIT"])); camB = KIT["cameras"]["wrist"]
PRI = json.load(open(ROOT / "mount/mount_link6.json"))["camera_prior"]

def set_pose(pos_m, quat, fovy, dx=np.zeros(7)):
    m.cam_pos[cam] = np.asarray(pos_m) + dx[:3] / 1000.0
    r = np.radians(dx[3:6]); ang = np.linalg.norm(r)
    dq = np.r_[np.cos(ang / 2), (np.sin(ang / 2) * r / ang) if ang > 0 else np.zeros(3)]
    q = np.zeros(4); mujoco.mju_mulQuat(q, np.asarray(quat, float), dq); m.cam_quat[cam] = q; m.cam_fovy[cam] = fovy + dx[6]

# frames: fingers (free, still gripper: fit_finger_mount.py's pick) and buckets (>= 1500 px of a bucket colour)
fing = {"train": ns["train"], "dev": ns["dev"]}
real_f = {}
for f, _ in fing["train"] + fing["dev"]:
    mk = ns["real_mask"](cv2.imread(str(f))); real_f[f.name] = (ns["edges"](mk), ns["tips"](mk))
buck = {"train": wns["frames"]((0, 2, 6, 7, 9)), "dev": wns["frames"]((1, 3, 10))}
real_b = {f.name: wns["real_masks"](cv2.imread(str(f))) for s in buck.values() for _, f, _ in s}

def finger_meas(frames):
    E, T = [], []
    for f, state in frames:
        ns["ch"].set_positions(state); mk = ns["sim_mask"](); re, rt = real_f[f.name]
        E.append(ns["edges"](mk) - re); T.append(ns["tips"](mk) - rt)
    return np.array(E), np.array(T)

def bucket_meas(frames):
    rows = []
    for ep, f, state in frames:
        b.set_bucket_poses(ns["robot"], LAY[str(ep)]["poses"]); ns["ch"].set_positions(state)
        sm = wns["sim_masks"](); rm = real_b[f.name]
        for c in COL:
            if rm[c].sum() < 1500 and sm[c].sum() < 1500: continue
            u = (rm[c] | sm[c]).sum(); iou = (rm[c] & sm[c]).sum() / u
            cen = lambda x: np.array(np.nonzero(x)[::-1]).mean(1) if x.sum() else np.full(2, np.nan)
            both = rm[c].sum() >= 1500 and sm[c].sum() >= 1500
            rows.append((c, f.name, float(iou), wns["contour_dist"](rm[c], sm[c]), (cen(sm[c]) - cen(rm[c])) if both else np.full(2, np.nan)))
    return rows

def report(fr_f, fr_b):
    E, T = finger_meas(fr_f); rows = bucket_meas(fr_b)
    out = {"finger_edge_median_abs_px": np.nanmedian(np.abs(E), axis=(0, 1)).round(1).tolist(),
           "finger_tip_median_dist_px": np.nanmedian(np.hypot(T[..., 0], T[..., 1]), axis=0).round(1).tolist(), "buckets": {}}
    for c in COL:
        r = [x for x in rows if x[0] == c]
        if not r: continue
        off = np.array([x[4] for x in r])
        out["buckets"][c] = {"n": len(r), "iou_median": round(float(np.median([x[2] for x in r])), 3),
                             "edge_px_median": round(float(np.nanmedian([x[3] for x in r])), 1),
                             "centroid_offset_median_px": np.nanmedian(off, 0).round(1).tolist(), "centroid_offset_iqr_px": (np.nanpercentile(off, 75, 0) - np.nanpercentile(off, 25, 0)).round(1).tolist()}
    allb = [x[3] for x in rows]; out["bucket_edge_px_median_all"] = round(float(np.nanmedian(allb)), 1)
    return out

def cost(fr_f, fr_b):
    E, T = finger_meas(fr_f); rows = bucket_meas(fr_b)
    cf = float(np.nanmean(np.minimum(np.abs(E), 60))) + float(np.nanmean(np.minimum(np.hypot(T[..., 0], T[..., 1]), 60)))
    cb = float(np.nanmean(np.minimum([x[3] for x in rows], 80)))
    return cf + cb

res = {"source": "review/export/robotcam_allowed (frames_x5 gripper, states_allowed.parquet); allowed episodes only",
       "frames": {"fingers": {k: [f.name for f, _ in v] for k, v in fing.items()}, "buckets": {k: [f.name for _, f, _ in v] for k, v in buck.items()}},
       "camera_B": camB | {"r12": None}, "prior": PRI, "variants": {}}
poses = {"B": (camB["pos"], camB["quat"], camB["fovy"]), "P": (PRI["pos_m"], PRI["quat_wxyz"], camB["fovy"]), "P68": (PRI["pos_m"], PRI["quat_wxyz"], 68.0)}
if DATASHEET:
    # r15 (coordinator): the Gemini 305 datasheet's colour FOV (4:3 640x480: V 68 +- 3 deg) as a bound
    OUT2 = OUT.replace(".json", "_datasheet.json"); res2 = {"source": res["source"], "variants": {}}
    def rep(name, extra):
        res2["variants"][name] = {**extra, "train": report(fing["train"], buck["train"]), "dev": report(fing["dev"], buck["dev"])}
        print(name, json.dumps({k: v for k, v in res2["variants"][name].items() if k != "train"}), flush=True); json.dump(res2, open(OUT2, "w"), indent=1)
    # PB: the STEP prior, a bounded move (<= 10 mm per axis, <= 3 deg per axis) and fovy in 65-71, fitted on TRAIN
    bound = lambda x: np.r_[10 * np.tanh(x[:3]), 3 * np.tanh(x[3:6]), 0.0]
    def pb(x):
        set_pose(PRI["pos_m"], PRI["quat_wxyz"], 68.0 + 3 * np.tanh(x[6]), bound(x))
    fit = minimize(lambda x: (pb(x), cost(fing["train"], buck["train"]))[1], np.zeros(7), method="Nelder-Mead",
                   options={"xatol": 0.01, "fatol": 0.05, "maxfev": MAXFEV, "initial_simplex": np.vstack([np.zeros(7)] + [np.eye(7)[i] * 0.4 for i in range(7)])})
    pb(fit.x); rep("PB_prior_bounded_fovy65_71", {"delta_mm_deg": bound(fit.x)[:6].round(2).tolist(), "fovy": round(float(m.cam_fovy[cam]), 2)})
    # KA: camera B's pose, the intrinsics fx, fy, cx, cy (px) fitted on TRAIN (MuJoCo sensor model: sensorsize = resolution = 640 x 480)
    set_pose(camB["pos"], camB["quat"], camB["fovy"]); f0 = 240.0 / np.tan(np.radians(camB["fovy"]) / 2)
    m.cam_resolution[cam] = [640, 480]; m.cam_sensorsize[cam] = [640.0, 480.0]
    def ka(x):
        m.cam_intrinsic[cam] = [f0 * (1 + x[0]), f0 * (1 + x[1]), x[2], x[3]]
    ka(np.zeros(4)); same = report(fing["dev"], buck["dev"])   # must equal B (the sensor model at fx = fy = f, no offset)
    print("sensor model at B's fovy (should equal B):", json.dumps(same), flush=True)
    fit = minimize(lambda x: (ka(x), cost(fing["train"], buck["train"]))[1], np.zeros(4), method="Nelder-Mead",
                   options={"xatol": 0.001, "fatol": 0.05, "maxfev": MAXFEV, "initial_simplex": np.vstack([np.zeros(4), [0.03, 0, 0, 0], [0, 0.03, 0, 0], [0, 0, 8, 0], [0, 0, 0, 8]])})
    ka(fit.x); fx, fy = f0 * (1 + fit.x[0]), f0 * (1 + fit.x[1])
    rep("KA_cameraB_fx_fy_cx_cy", {"f0_px": round(f0, 2), "fx_px": round(fx, 2), "fy_px": round(fy, 2), "fx_over_fy": round(fx / fy, 4), "principal_offset_px": np.round(fit.x[2:], 2).tolist(),
                                   "hfov_deg": round(float(np.degrees(2 * np.arctan(320 / fx))), 2), "vfov_deg": round(float(np.degrees(2 * np.arctan(240 / fy))), 2)})
    m.cam_sensorsize[cam] = [0.0, 0.0]
    sys.exit()
for name, (p, q, fv) in poses.items():
    set_pose(p, q, fv); res["variants"][name] = {"pos_m": list(map(float, p)), "quat": list(map(float, q)), "fovy": fv,
                                                  "train": report(fing["train"], buck["train"]), "dev": report(fing["dev"], buck["dev"])}
    print(name, json.dumps(res["variants"][name]["dev"]), flush=True); json.dump(res, open(OUT, "w"), indent=1)
def train_cost(k):
    set_pose(*poses[k]); return cost(fing["train"], buck["train"])
best = min(("P", "P68"), key=train_cost)
p0, q0, f0 = poses[best]
fit = minimize(lambda x: (set_pose(p0, q0, f0, x), cost(fing["train"], buck["train"]))[1], np.zeros(7), method="Nelder-Mead",
               options={"xatol": 0.05, "fatol": 0.05, "maxfev": MAXFEV, "initial_simplex": np.vstack([np.zeros(7)] + [np.eye(7)[i] * s for i, s in enumerate((3, 3, 3, 1.5, 1.5, 1.5, 1.5))])})
set_pose(p0, q0, f0, fit.x)
res["variants"]["PF"] = {"from": best, "delta_mm_deg_dfovy": fit.x.round(3).tolist(), "pos_m": m.cam_pos[cam].round(6).tolist(), "quat": m.cam_quat[cam].round(6).tolist(),
                         "fovy": float(m.cam_fovy[cam]), "train": report(fing["train"], buck["train"]), "dev": report(fing["dev"], buck["dev"])}
print("PF", fit.x.round(2).tolist(), json.dumps(res["variants"]["PF"]["dev"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
