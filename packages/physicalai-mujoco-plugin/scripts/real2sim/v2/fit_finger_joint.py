"""D2 (r11; r12: + holding frames, no lean): the drawn finger jaws fitted to the real wrist frames jointly (export only): thickness across (scale) and
jaw extent (jaw_m) as in r10, plus per finger the jaw's length (tip_scale) and lean (tip_shear), on TRAIN episodes
(0, 2, 6, 7, 9; the frames fit_finger_mount.py picks: gripper and arm still); DEV (1, 3, 10) only reported. Cost per
frame: the mean |error| of the four finger edges per row (330-470) plus the two fingertip errors (topmost point,
px). Drawing only (apply_finger_visual). usage: fit_finger_joint.py STATES_PARQUET FRAMES_X5 OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
HERE = Path(__file__).resolve().parent
OUT = sys.argv[3]; sys.argv = sys.argv[:3] + ["/dev/null"]
src = (HERE / "fit_finger_mount.py").read_text().split("p0 = np.zeros(6)")[0]
ns = {"__file__": str(HERE / "fit_finger_mount.py"), "__name__": "ffm"}; exec(compile(src, "fit_finger_mount.py", "exec"), ns)
tsrc = (HERE / "finger_tips.py").read_text(); tips_src = tsrc[tsrc.index("def tips(mask):"):tsrc.index("def measure(frames, spec):")]
exec(compile(tips_src, "finger_tips.py", "exec"), ns)
from finger_shape_probe import apply_finger_visual
from scipy.optimize import minimize
m, d = ns["m"], ns["d"]; kit = json.load(open(os.environ["KIT"]))["finger_visual"]
mujoco.mj_forward(m, d); c = m.camera("wrist").id; Rc = d.cam_xmat[c].reshape(3, 3)
SIDE = ("gripper_left", "gripper_right") if (d.xpos[m.body("gripper_left").id] - d.cam_xpos[c]) @ Rc[:, 0] < 0 else ("gripper_right", "gripper_left")
# r12: frames where the real gripper holds a resistor (beige: the finger edges stay clean), from the carry plateaus
# of real/grip_widths.json; TRAIN episodes 0, 6, 7 fitted, DEV 1, 10 reported; only rows where both real fingers show apart
def holds(eps):
    g = json.load(open(Path.home() / "real2sim-data/bucket-v2/real/grip_widths.json")); out = []
    for ep in eps:
        rows = ns["st"][ns["st"].episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
        for pl in g["episodes"][str(ep)]["carry_plateaus"]:
            if pl["opening_mm"] > 22: continue   # resistors only (a CPU or battery is dark: it merges with the fingers)
            for fr in np.linspace(pl["frames"][0] + 5, pl["frames"][1] - 5, 4).astype(int):
                ms = int(round(fr / 30 * 1000)); f = sorted(ns["F"].glob(f"gripper_e{ep:03d}_t*.png"), key=lambda p: abs(int(p.stem.split("_t")[1]) - ms))[0]
                t = int(f.stem.split("_t")[1]) / 1000; tl = float(np.clip(t - ns["lat"], ts[0], ts[-1]))
                out.append((f, np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])])))
    return out
hold_train, hold_dev = holds((0, 6, 7)), holds((1, 10))
real = {}
for f, _ in hold_train + hold_dev:
    e = ns["edges"](ns["real_mask"](cv2.imread(str(f)))); bad = ~((e[:, 2] - e[:, 1]) > 10); e[bad] = np.nan; real[f.name] = (e, np.full((2, 2), np.nan))
for f, _ in ns["train"] + ns["dev"]:
    mk = ns["real_mask"](cv2.imread(str(f))); real[f.name] = (ns["edges"](mk), ns["tips"](mk))

def spec_of(x):
    return {"tip_axis": kit["tip_axis"], "ramp_m": kit.get("ramp_m", 0.01), "scale": float(x[0]), "jaw_m": float(x[1]),
            "tip_scale": {SIDE[0]: float(x[2]), SIDE[1]: float(x[4])}, "tip_shear": {SIDE[0]: float(x[3]), SIDE[1]: float(x[5])}}
            # r12: the lean fitted with the holding frames too (r11's, fitted on open frames, closed the tips on a held object)

def errors(frames, x):
    apply_finger_visual(m, spec_of(x)); ns["seg"] = mujoco.Renderer(m, 480, 640); ns["seg"].enable_segmentation_rendering()
    E, T = [], []
    for f, state in frames:
        ns["ch"].set_positions(state); mk = ns["sim_mask"](); re, rt = real[f.name]
        E.append(np.abs(ns["edges"](mk) - re)); T.append(np.hypot(*(ns["tips"](mk) - rt).T) if np.isfinite(rt).any() else np.full(2, np.nan))
    return np.array(E), np.array(T)

def cost(x):
    E, T = errors(ns["train"], x); Eh, _ = errors(hold_train, x)
    return float(np.nanmean(np.minimum(E, 60))) + float(np.nanmean(np.minimum(T, 60))) + float(np.nanmean(np.minimum(Eh, 60)))

x0 = np.array([kit["scale"], kit["jaw_m"], kit["tip_scale"][SIDE[0]], kit["tip_shear"][SIDE[0]], kit["tip_scale"][SIDE[1]], kit["tip_shear"][SIDE[1]]])
step = np.array([0.05, 0.008, 0.08, 0.05, 0.08, 0.05])
fit = minimize(cost, x0, method="Nelder-Mead", options={"xatol": 0.002, "fatol": 0.05, "maxfev": 220, "initial_simplex": np.vstack([x0] + [x0 + np.eye(6)[i] * step[i] for i in range(6)])})
x = fit.x
def report(frames, x):
    E, T = errors(frames, x); return {"edge_median_px": np.nanmedian(E, axis=(0, 1)).round(1).tolist(), "tip_median_px": np.nanmedian(T, axis=0).round(1).tolist()}
r11 = json.load(open(os.environ["KIT"]))["finger_visual"]
def report_spec(frames, spec):
    apply_finger_visual(m, spec); ns["seg"] = mujoco.Renderer(m, 480, 640); ns["seg"].enable_segmentation_rendering()
    E = []
    for f, state in frames:
        ns["ch"].set_positions(state); E.append(np.abs(ns["edges"](ns["sim_mask"]()) - real[f.name][0]))
    return np.nanmedian(np.array(E), axis=(0, 1)).round(1).tolist()
res = {"source": "review/export/robotcam_allowed (frames_x5 gripper, states_allowed.parquet)", "image_left_right_bodies": SIDE,
       "dev_hold_r11": report_spec(hold_dev, r11), "dev_hold": None, "hold_train_frames": [f.name for f, _ in hold_train], "hold_dev_frames": [f.name for f, _ in hold_dev],
       "train": report(ns["train"], x), "dev": report(ns["dev"], x), "finger_visual": spec_of(x),
       "train_frames": [f.name for f, _ in ns["train"]], "dev_frames": [f.name for f, _ in ns["dev"]]}
res["dev_hold"] = report_spec(hold_dev, spec_of(x))
json.dump(res, open(OUT, "w"), indent=1)
for k in ("dev_hold_r11", "dev_hold", "train", "dev"): print(k, res[k])
print("x", np.round(x, 4).tolist())
