"""D2 (r13): is the fingers' left/right asymmetry in the wrist view the camera or the fingers? (export only)

The real fingers are one part, mirrored. On allowed wrist frames where the gripper is free and still (open or closed,
nothing held: fit_finger_mount.py's pick), per image row 330-470 the four finger edges and the two fingertips, real
against the sim's segmentation of the drawn fingers. Variants, each fitted on TRAIN (0, 2, 6, 7, 9), reported on DEV
(1, 3, 10):
  A  the CAD fingers, the kit's camera (no fit)
  B  the CAD fingers, the wrist camera's mount (3 mm + 3 deg in link6) and fovy fitted
  C  as B plus ONE symmetric finger shape: the jaw thinned about its inner face (scale) and its length beyond the jaw
     start (length), the same for both fingers
  R  the r12 kit drawing (per-finger length and lean; reference only)
Per variant: edge and tip errors (median px, per edge / per finger with sign: sim - real), and the parameters.
usage: fit_wrist_joint.py STATES_PARQUET FRAMES_X5 OUT_JSON [--maxfev N]"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
HERE = Path(__file__).resolve().parent
OUT = sys.argv[3]; MAXFEV = int(sys.argv[sys.argv.index("--maxfev") + 1]) if "--maxfev" in sys.argv else 500
sys.argv = sys.argv[:3] + ["/dev/null"]
src = (HERE / "fit_finger_mount.py").read_text().split("p0 = np.zeros(6)")[0]
ns = {"__file__": str(HERE / "fit_finger_mount.py"), "__name__": "ffm"}; exec(compile(src, "fit_finger_mount.py", "exec"), ns)
tsrc = (HERE / "finger_tips.py").read_text(); exec(compile(tsrc[tsrc.index("def tips(mask):"):tsrc.index("def measure(frames, spec):")], "finger_tips.py", "exec"), ns)
from finger_shape_probe import apply_finger_visual
from scipy.optimize import minimize
m, d, cam = ns["m"], ns["d"], ns["cam"]; KIT = json.load(open(os.environ["KIT"])); r12 = KIT["finger_visual"]
fovy0 = float(m.cam_fovy[cam])
real = {}
for f, _ in ns["train"] + ns["dev"]:
    mk = ns["real_mask"](cv2.imread(str(f))); real[f.name] = (ns["edges"](mk), ns["tips"](mk))

def shape_spec(s, L):
    return {"tip_axis": r12["tip_axis"], "ramp_m": r12.get("ramp_m", 0.01), "jaw_m": r12["jaw_m"], "scale": float(s),
            "tip_scale": {"gripper_left": float(L), "gripper_right": float(L)}}

def setup(mount6, dfovy, spec):
    ns["set_mount"](mount6); m.cam_fovy[cam] = fovy0 + dfovy
    apply_finger_visual(m, spec or shape_spec(1.0, 1.0))
    ns["seg"] = mujoco.Renderer(m, 480, 640); ns["seg"].enable_segmentation_rendering()

def measure(frames):
    E, T = [], []
    for f, state in frames:
        ns["ch"].set_positions(state); mk = ns["sim_mask"](); re, rt = real[f.name]
        E.append(ns["edges"](mk) - re); T.append(ns["tips"](mk) - rt)
    return np.array(E), np.array(T)   # (frames, rows, 4) signed px; (frames, 2 fingers, (dy, dx))

def cost(E, T):
    return float(np.nanmean(np.minimum(np.abs(E), 60))) + float(np.nanmean(np.minimum(np.hypot(T[..., 0], T[..., 1]), 60)))

def report(E, T):
    return {"edge_median_abs_px": np.nanmedian(np.abs(E), axis=(0, 1)).round(1).tolist(),
            "edge_median_signed_px": np.nanmedian(E, axis=(0, 1)).round(1).tolist(),
            "tip_median_dist_px": np.nanmedian(np.hypot(T[..., 0], T[..., 1]), axis=0).round(1).tolist(),
            "tip_median_dy_dx_px": np.nanmedian(T, axis=0).round(1).tolist(), "cost": round(cost(E, T), 2)}

out = {"source": "review/export/robotcam_allowed (frames_x5 gripper, states_allowed.parquet); free, still gripper frames",
       "edges": ["image-left outer", "image-left inner", "image-right inner", "image-right outer"], "tips": ["image-left", "image-right"],
       "train_frames": [f.name for f, _ in ns["train"]], "dev_frames": [f.name for f, _ in ns["dev"]], "fovy_kit": fovy0, "variants": {}}

def run(name, x_to_setup, x0, step):
    def c(x):
        x_to_setup(x); return cost(*measure(ns["train"]))
    if len(x0):
        fit = minimize(c, x0, method="Nelder-Mead", options={"xatol": 0.01, "fatol": 0.02, "maxfev": MAXFEV,
                       "initial_simplex": np.vstack([x0] + [x0 + np.eye(len(x0))[i] * step[i] for i in range(len(x0))])})
        x = fit.x
    else:
        x = x0
    x_to_setup(x); tr = report(*measure(ns["train"])); dv = report(*measure(ns["dev"]))
    out["variants"][name] = {"x": np.round(x, 4).tolist(), "train": tr, "dev": dv}
    print(name, "x", np.round(x, 3).tolist(), "\n  TRAIN", tr, "\n  DEV  ", dv, flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    return x

run("A_cad_kit_camera", lambda x: setup(np.zeros(6), 0.0, None), np.zeros(0), [])
setup(np.zeros(6), 0.0, r12); tr = report(*measure(ns["train"])); dv = report(*measure(ns["dev"]))
out["variants"]["R_r12_drawing"] = {"train": tr, "dev": dv}; print("R_r12_drawing\n  TRAIN", tr, "\n  DEV  ", dv, flush=True)
xb = run("B_cad_fitted_camera", lambda x: setup(x[:6], x[6], None), np.zeros(7), [3, 3, 3, 2, 2, 2, 2])
out["variants"]["B_cad_fitted_camera"]["params"] = "mount dx, dy, dz (mm, link6 frame), rotation vector (deg), dfovy (deg)"
xc = run("C_symmetric_shape_fitted_camera", lambda x: setup(x[:6], x[6], shape_spec(x[7], x[8])), np.r_[xb, 1.0, 1.0], [2, 2, 2, 1.5, 1.5, 1.5, 1.5, 0.1, 0.1])
out["variants"]["C_symmetric_shape_fitted_camera"]["params"] = "as B, then the jaw's thickness scale about its inner face and its length scale (both fingers)"
json.dump(out, open(OUT, "w"), indent=1)
