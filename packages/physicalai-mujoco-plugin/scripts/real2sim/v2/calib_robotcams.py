"""Record the robot cameras' fitted stream calibration in calibration.json (L9): for overview and wrist, a "stream"
section with the pose (overview: world; wrist: link6 frame), the MuJoCo quaternion, fovy, the full K and distortion,
rms, frames and sigma from real/fit_overview.json and real/fit_gripper2.json (export only). The camera entries' pos/quat
stay what the scene's mount and housing geometry use (phone path, D3); the streams use the kit overrides = this section.
usage: calib_robotcams.py"""
import json, os, numpy as np, mujoco
from pathlib import Path
R = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); cal = json.load(open(R / "calibration.json")); kit = json.load(open(R / "kit/real2sim.json"))
for cam, fit in (("overview", "fit_overview.json"), ("wrist", "fit_gripper2.json")):
    f = json.load(open(R / "real" / fit)); Rmj = np.array(f["rotation_opencv_axes"]) @ np.diag([1.0, -1.0, -1.0]); q = np.zeros(4); mujoco.mju_mat2Quat(q, Rmj.ravel())
    cal[cam]["stream"] = {"source": f"real/{fit} (reviewer's export, allowed episodes; fit_robotcam.py: SIFT real<->render, PnP, calibrateCamera)",
                          "frame": f["frame"], "pos": [round(v, 5) for v in f["position_m"]], "quat_mujoco": q.round(6).tolist(),
                          "fovy_deg": round(f["fovy_deg"], 3), "K": [[round(v, 3) for v in row] for row in f["K"]], "dist_k1_k2": [round(v, 4) for v in f["dist"][:2]],
                          "distortion_applied": False, "rms_px": round(f["rms_px"], 3), "frames": f["frames"], "position_sigma_mm": f["position_spread_mm"],
                          "kit_override": kit["cameras"].get(cam, {}),
                          "note": "the streams use this (kit camera override); the entry's pos/quat/fovy above are the mount and housing geometry of the scene (phone path)"}
json.dump(cal, open(R / "calibration.json", "w"), indent=1); print("ok")
