"""r17 item 2 (coordinator): a bounded refit of the overview camera with the 120 mm buckets (the user's statement), the
pitch included. Parameters: the camera's position +-10 mm and rotation +-1.5 deg about its own axes, from the kit pose
(tanh-bounded). Per candidate, every TRAIN episode's bucket layout is refitted with the builder's own fitter
(layout_fitter.LayoutFitter, FIT_LAYOUT's method: the camera moves the silhouettes, so a layout fitted with another camera
would bias the comparison) and the cost is
    J = 100 x (1 - mean fitted bucket IoU over TRAIN episodes and colours) + the overview arm cost (px; fit_overview_arm.py:
        the symmetric mean distance between the real and the sim lime shells, truncated at 60 px, at the recorded states)
Nelder-Mead on TRAIN (0, 2, 6, 7, 9); the kit pose and the TRAIN top 3 are then evaluated on DEV (1, 3, 10) with their own
DEV layout refits. Bound use is reported. Writes OUT_JSON as it goes (resumable reading only).
usage: overview_refit.py STATES_PARQUET FRAMES_X5 STATES_X5_JSON OUT_JSON [--maxfev 70]"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
OUT = sys.argv[4]; S5 = json.load(open(sys.argv[3])); S5 = S5.get("states", S5)
MAXFEV = int(sys.argv[sys.argv.index("--maxfev") + 1]) if "--maxfev" in sys.argv else 70
# the arm cost and its frames: fit_overview_arm.py's own code up to its results (8 frames per episode)
src = (HERE / "fit_overview_arm.py").read_text().split("res = {")[0]
sys.argv = [str(HERE / "fit_overview_arm.py"), sys.argv[1], sys.argv[2], "/dev/null"]
ns = {"__file__": str(HERE / "fit_overview_arm.py"), "__name__": "foa"}; exec(compile(src, "fit_overview_arm.py", "exec"), ns)
m, robot, cam, sets, evaluate, LAY, F = ns["m"], ns["robot"], ns["cam"], ns["sets"], ns["evaluate"], ns["LAY"], ns["F"]
from layout_fitter import LayoutFitter
fitter = LayoutFitter(robot)
P0, Q0 = m.cam_pos[cam].copy(), m.cam_quat[cam].copy()

def episode_inputs(ep):
    files = sorted(F.glob(f"overview_e{ep:03d}_t*.png"))
    pick = [files[i] for i in np.linspace(0, len(files) - 1, min(25, len(files))).astype(int)]
    keys = [f"{ep}/{int(f.stem.split('_t')[1])}" for f in pick]
    return [cv2.imread(str(f)) for f in pick], np.median(np.array([S5[k] for k in keys if k in S5]), axis=0)
TRAIN, DEV = (0, 2, 6, 7, 9), (1, 3, 10)
INP = {ep: episode_inputs(ep) for ep in TRAIN + DEV}

def unpack(x):
    return 0.010 * np.tanh(x[:3]), np.radians(1.5) * np.tanh(x[3:6])

def set_cam(x):
    dp, dr = unpack(x); q = np.zeros(4); mujoco.mju_euler2Quat(q, dr, "XYZ"); out = np.zeros(4); mujoco.mju_mulQuat(out, Q0, q)
    m.cam_pos[cam] = P0 + dp; m.cam_quat[cam] = out

def score(x, eps, split):
    set_cam(x); ious = []
    for ep in eps:
        res, _ = fitter.fit(*INP[ep]); LAY[str(ep)] = {"poses": res["poses"]}; ious += list(res["iou_fitted"].values())
    arm = evaluate(sets[split])["cost_px"]; iou = float(np.mean(ious))
    return {"J": round(100 * (1 - iou) + arm, 3), "iou_mean": round(iou, 4), "arm_px": arm}

log = {"method": __doc__.split("usage")[0].strip(), "kit_pose": {"pos": P0.tolist(), "quat": Q0.tolist()}, "evals": []}
def J(x):
    r = score(x, TRAIN, "train"); dp, dr = unpack(x)
    log["evals"].append({"x": np.round(x, 4).tolist(), "dpos_mm": (dp * 1000).round(2).tolist(), "drot_deg": np.degrees(dr).round(3).tolist(), **r})
    print(len(log["evals"]), log["evals"][-1], flush=True); json.dump(log, open(OUT, "w"), indent=1)
    return r["J"]

x0 = np.zeros(6); J(x0)
fit = minimize(J, x0, method="Nelder-Mead", options={"maxfev": MAXFEV, "xatol": 0.02, "fatol": 0.05,
                                                      "initial_simplex": np.vstack([x0] + [np.eye(6)[i] * 0.5 for i in range(6)])})
top = sorted(log["evals"], key=lambda e: e["J"])[:3]
log["dev"] = {"kit": score(x0, DEV, "dev")}
for k, e in enumerate(top):
    x = np.array(e["x"]); dp, dr = unpack(x)
    log["dev"][f"top{k + 1}"] = {**score(x, DEV, "dev"), "train": {kk: e[kk] for kk in ("J", "iou_mean", "arm_px")},
                                 "dpos_mm": (dp * 1000).round(2).tolist(), "drot_deg": np.degrees(dr).round(3).tolist(),
                                 "bound_use": [round(abs(v) / 10, 3) for v in dp * 1000] + [round(abs(v) / 1.5, 3) for v in np.degrees(dr)]}
    print("dev", k + 1, log["dev"][f"top{k + 1}"], flush=True)
print("dev kit", log["dev"]["kit"]); set_cam(x0); json.dump(log, open(OUT, "w"), indent=1)
