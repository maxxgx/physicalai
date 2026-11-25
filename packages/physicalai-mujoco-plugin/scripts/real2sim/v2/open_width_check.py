"""r16 (D3a, coordinator): the real gripper's opening at the maximum open command (-270 deg), from allowed wrist frames.

Frames: TRAIN (0, 2, 6, 7, 9) wrist frames where the gripper reading is <= -265 deg and still (fit_finger_mount.py's
pick restricted to open frames), DEV (1, 3, 10) reported. Per frame, the opening of the CAD fingers (both, symmetric) is
searched 50-100 mm so that the sim fingers' four edges per row (330-470) match the real ones best (median |error|),
with the wrist camera at (a) camera B (the kit) and (b) the mount STEP's seat prior (mount_from_step.py) with fovy 68 deg
(the datasheet) and with B's fovy. The profile maps -270 deg to 70.1 mm; the BOM (module 1, 16 teeth: 0.1396 mm of rack
per degree, per finger) gives 75.4 mm of change over 270 deg (79.9 mm with the 4.64 mm closed gap).
usage: open_width_check.py STATES_PARQUET FRAMES_X5 OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
HERE = Path(__file__).resolve().parent
OUT = sys.argv[3]; sys.argv = sys.argv[:3] + ["/dev/null"]
src = (HERE / "fit_finger_mount.py").read_text().split("p0 = np.zeros(6)")[0]
ns = {"__file__": str(HERE / "fit_finger_mount.py"), "__name__": "ffm"}; exec(compile(src, "fit_finger_mount.py", "exec"), ns)
m, d, cam, F, st, lat = ns["m"], ns["d"], ns["cam"], ns["F"], ns["st"], ns["lat"]
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
KIT = json.load(open(os.environ["KIT"])); B = KIT["cameras"]["wrist"]; PRI = json.load(open(ROOT / "mount/mount_link6.json"))["camera_prior"]
fadr = [m.jnt_qposadr[m.joint(n).id] for n in ("joint_left", "joint_right")]

HOLDS = os.environ.get("HOLDS") == "1"   # r16: closed-on-object frames (reading -100..-20) instead of the open ones

def open_frames(eps, per=8):
    out = []
    for f, state in ns["pick"](eps):
        if (-100 <= state[-1] <= -20) if HOLDS else (state[-1] <= -265): out.append((f, state))
    return out[: per * len(eps)]

def best_opening(frames):
    rows = []
    for f, state in frames:
        real = ns["edges"](ns["real_mask"](cv2.imread(str(f)))); ok = np.isfinite(real)
        if ok.sum() < 8: continue
        best = None
        for w in (np.arange(5.0, 45.01, 0.5) if HOLDS else np.arange(50.0, 100.01, 0.5)):
            ns["ch"].set_positions(state)
            for a in fadr: d.qpos[a] = w / 2000.0
            sim = ns["edges"](ns["sim_mask"]()); e = np.abs(sim - real)[ok & np.isfinite(sim)]
            if len(e) < 8: continue
            c = float(np.median(e))
            if best is None or c < best[1]: best = (w, c)
        if best: rows.append({"frame": f.name, "reading": round(float(state[-1]), 1), "opening_mm": best[0], "edge_err_px": round(best[1], 1)})
    W = np.array([r["opening_mm"] for r in rows]); E = np.array([r["edge_err_px"] for r in rows])
    return {"n": len(rows), "opening_mm_median": float(np.median(W)) if len(W) else None, "iqr": np.percentile(W, [25, 75]).tolist() if len(W) else None,
            "edge_err_px_median": float(np.median(E)) if len(E) else None, "frames": rows}

def set_cam(pos, quat, fovy):
    m.cam_pos[cam] = pos; m.cam_quat[cam] = quat; m.cam_fovy[cam] = fovy

res = {"source": "review/export/robotcam_allowed (frames_x5 gripper, states_allowed.parquet); open frames (reading <= -265)", "profile_mm_at_-270": 70.1,
       "bom_mm_at_-270": round(4.64 + 270 * 0.1396 * 2, 1)}
FIT = json.load(open(ROOT / "real/r16/fit_gripper_camera.json"))["fit"] if (ROOT / "real/r16/fit_gripper_camera.json").exists() else None
CAMS = [("camera_B", (B["pos"], B["quat"], B["fovy"])), ("step_prior_fovy68", (PRI["pos_m"], PRI["quat_wxyz"], 68.0)),
        ("step_prior_fovyB", (PRI["pos_m"], PRI["quat_wxyz"], B["fovy"]))]
if FIT: CAMS.append(("joint_fit_camera", (FIT["camera_pos_m"], FIT["camera_quat"], FIT["fovy"])))
for name, pose in CAMS:
    set_cam(*pose); res[name] = {}
    for split, eps in (("train", (0, 2, 6, 7, 9)), ("dev", (1, 3, 10))):
        r = best_opening(open_frames(eps)); res[name][split] = r
        print(name, split, {k: v for k, v in r.items() if k != "frames"}, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
