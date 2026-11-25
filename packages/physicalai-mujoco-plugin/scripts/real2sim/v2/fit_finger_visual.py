"""D2 (r10): fit the drawn finger jaws' thickness to the real wrist frames (the reviewer's export only).

The edges per row (330-470) as fit_finger_mount.py measures them (it found the inner edges within 0-2 px and the outer
ones ~40 px out: the jaws are too thick, the mount is right). Fits ``scale`` (across) and ``jaw_m`` (how far back from
the tip it applies) of desk_compositor.apply_finger_visual on the TRAIN episodes (0, 2, 6, 7, 9), checks the DEV ones
(1, 3, 10). The tip axis per finger is the wrist camera's forward direction in the finger's frame (the camera looks
along the jaws). Writes OUT_JSON with the frames, the edges before/after and the kit's ``finger_visual``.
usage: fit_finger_visual.py STATES_PARQUET FRAMES_X5 OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np, mujoco
sys.argv += [] ; HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
OUT = sys.argv[3]; sys.argv = sys.argv[:3] + ["/dev/null"]
os.environ.pop("FINGER_DEBUG", None)
import importlib.util
spec_ = importlib.util.spec_from_file_location("ffm", HERE / "fit_finger_mount.py")
src = (HERE / "fit_finger_mount.py").read_text().split("p0 = np.zeros(6)")[0]   # the measuring half only
ns = {"__file__": str(HERE / "fit_finger_mount.py"), "__name__": "ffm"}; exec(compile(src, "fit_finger_mount.py", "exec"), ns)
from finger_shape_probe import apply_finger_visual
from scipy.optimize import minimize
m, d, cam = ns["m"], ns["d"], ns["cam"]; train, dev, real = ns["train"], ns["dev"], ns["real"]
for f, _ in train + dev: real[f.name] = ns["edges"](ns["real_mask"](ns["cv2"].imread(str(f))))
mujoco.mj_forward(m, d); fwd = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
tip_axis = {b: (d.xmat[m.body(b).id].reshape(3, 3).T @ fwd).round(5).tolist() for b in ("gripper_left", "gripper_right")}

def err(frames, x, per_edge=False):
    apply_finger_visual(m, {"tip_axis": tip_axis, "jaw_m": float(x[1]), "ramp_m": 0.01, "scale": float(x[0])})
    ns["seg"] = mujoco.Renderer(m, 480, 640); ns["seg"].enable_segmentation_rendering()   # re-upload the meshes
    sim = ns["measure"](frames); R = np.array([real[f.name] for f, _ in frames]); e = np.abs(sim - R)
    return np.nanmedian(e, axis=(0, 1)).round(1).tolist() if per_edge else float(np.nanmean(np.minimum(e, 60)))

x0 = np.array([1.0, 0.05])
before = {"train": err(train, x0), "dev": err(dev, x0), "dev_median_per_edge_px": err(dev, x0, True)}
print("before", before, flush=True)
best = min(((err(train, [k, j]), k, j) for k in (0.4, 0.5, 0.6, 0.7) for j in (0.03, 0.045, 0.06)), key=lambda t: t[0])
print("grid best", best, flush=True)
res = minimize(lambda x: err(train, x), np.array(best[1:]), method="Nelder-Mead", options={"xatol": 0.002, "fatol": 0.05, "maxfev": 60})
x = np.clip(res.x, [0.2, 0.01], [1.0, 0.12])
after = {"train": err(train, x), "dev": err(dev, x), "dev_median_per_edge_px": err(dev, x, True)}
print("after", after, "scale", round(float(x[0]), 3), "jaw_m", round(float(x[1]), 4), flush=True)
fv = {"tip_axis": tip_axis, "jaw_m": round(float(x[1]), 4), "ramp_m": 0.01, "scale": round(float(x[0]), 3),
      "fitted_from": "real/finger_visual.json: edges per row on allowed wrist frames of the export (TRAIN episodes 0, 2, 6, 7, 9; checked on DEV 1, 3, 10)"}
json.dump({"source": "review/export/robotcam_allowed (frames_x5 gripper, states_allowed.parquet)", "edges": ["left outer", "left inner", "right inner", "right outer"],
           "rows": ns["ROWS"].tolist(), "train_frames": [f.name for f, _ in train], "dev_frames": [f.name for f, _ in dev],
           "before_px": before, "after_px": after, "finger_visual": fv}, open(OUT, "w"), indent=1)
