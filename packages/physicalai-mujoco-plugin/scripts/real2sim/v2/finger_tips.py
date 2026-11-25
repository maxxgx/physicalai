"""D2 (r11): the fingertips, real against sim, on allowed wrist frames (export only). The frames and states as
fit_finger_mount.py picks them (gripper and arm still), the sim with the kit's finger_visual. Per finger (left: the
component left of the image centre), its tip = the topmost row of the finger mask and the mean x of the mask in that
row's 4-px band. usage: finger_tips.py STATES_PARQUET FRAMES_X5 OUT_JSON [--scale-len L --jaw J]"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
HERE = Path(__file__).resolve().parent
OUT = sys.argv[3]; extra = sys.argv[4:]; sys.argv = sys.argv[:3] + ["/dev/null"]
src = (HERE / "fit_finger_mount.py").read_text().split("p0 = np.zeros(6)")[0]
ns = {"__file__": str(HERE / "fit_finger_mount.py"), "__name__": "ffm"}; exec(compile(src, "fit_finger_mount.py", "exec"), ns)
from finger_shape_probe import apply_finger_visual
m, d = ns["m"], ns["d"]; kit = json.load(open(os.environ["KIT"]))["finger_visual"]

def tips(mask):
    out = []
    for side in (0, 1):
        mk = mask.copy(); mk[:, 320:] = False if side == 0 else mk[:, 320:]; mk[:, :320] = mk[:, :320] if side == 0 else False
        mk[:250] = False   # the fingers rise from the bottom; ignore far objects
        n, lab, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8))
        keep = [i for i in range(1, n) if st[i, 1] + st[i, 3] >= 470]
        if not keep: out.append((np.nan, np.nan)); continue
        mk = np.isin(lab, keep); rows = np.flatnonzero(mk.any(1)); y = rows[0]
        xs = np.flatnonzero(mk[y:y + 4].any(0)); out.append((float(y), float(xs.mean())))
    return np.array(out)

def measure(frames, spec):
    apply_finger_visual(m, spec); ns["seg"] = mujoco.Renderer(m, 480, 640); ns["seg"].enable_segmentation_rendering()
    res = []
    for f, state in frames:
        ns["ch"].set_positions(state); sim = tips(ns["sim_mask"]()); real = tips(ns["real_mask"](cv2.imread(str(f)))); res.append((f.name, float(state[-1]), real, sim))
    return res

def summary(res):
    d_ = np.array([r[3] - r[2] for r in res])   # sim - real, (frame, side, (dy, dx))
    return {"left_dy_med": float(np.nanmedian(d_[:, 0, 0])), "left_dx_med": float(np.nanmedian(d_[:, 0, 1])), "right_dy_med": float(np.nanmedian(d_[:, 1, 0])),
            "right_dx_med": float(np.nanmedian(d_[:, 1, 1])), "tip_dist_med": float(np.nanmedian(np.hypot(d_[..., 0], d_[..., 1])))}

from scipy.optimize import minimize
LEFT, RIGHT = "gripper_left", "gripper_right"
# which body is drawn on the image's left: the one whose tip is left of the centre
def spec_of(x):
    return {**kit, "tip_scale": {SIDE[0]: float(x[0]), SIDE[1]: float(x[2])}, "tip_shear": {SIDE[0]: float(x[1]), SIDE[1]: float(x[3])}}
def side_of_bodies():
    import mujoco as mj
    mj.mj_forward(m, d); c = m.camera("wrist").id; R = d.cam_xmat[c].reshape(3, 3)
    xl = (d.xpos[m.body(LEFT).id] - d.cam_xpos[c]) @ R[:, 0]
    return (LEFT, RIGHT) if xl < 0 else (RIGHT, LEFT)
SIDE = side_of_bodies()
def cost(x, frames):
    res = measure(frames, spec_of(x)); d_ = np.array([r[3] - r[2] for r in res])
    return float(np.nanmean(np.minimum(np.hypot(d_[..., 0], d_[..., 1]), 60)))
x0 = np.array([1.0, 0.0, 1.0, 0.0])
fit = minimize(lambda x: cost(x, ns["train"]), x0, method="Nelder-Mead", options={"xatol": 0.003, "fatol": 0.1, "maxfev": 120,
               "initial_simplex": np.vstack([x0, x0 + [-0.15, 0, 0, 0], x0 + [0, 0.1, 0, 0], x0 + [0, 0, -0.15, 0], x0 + [0, 0, 0, 0.1]])})
best = spec_of(fit.x); print("fit (image-left body, image-right body)", SIDE, np.round(fit.x, 3).tolist(), flush=True)
base = measure(ns["train"], best); dev = measure(ns["dev"], best)
b0 = measure(ns["dev"], kit)
print("DEV before", {k: round(v, 1) for k, v in summary(b0).items()})
kit = best
print("TRAIN", {k: round(v, 1) for k, v in summary(base).items()}); print("DEV", {k: round(v, 1) for k, v in summary(dev).items()})
for name, g, r, s in base[:8]: print(name, "grip", round(g), "real tips (y,x)", r.round(0).tolist(), "sim", s.round(0).tolist())
json.dump({"finger_visual": {k: v for k, v in best.items()}, "dev_before": summary(b0), "train": summary(base), "dev": summary(dev), "train_frames": [r[0] for r in base], "dev_frames": [r[0] for r in dev]}, open(OUT, "w"), indent=1)
