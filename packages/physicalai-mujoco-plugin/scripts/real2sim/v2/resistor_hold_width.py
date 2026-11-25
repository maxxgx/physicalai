"""r14 (T3, R-46): how far apart the real fingers are held by the real resistor (same-frame evidence: the CAD fingers are
the ruler; export only). Frames: the resistor carry plateaus of real/grip_widths.json (opening by reading <= 22 mm), 4 per
plateau, allowed episodes; TRAIN 0, 2, 6, 7, 9 and DEV 1, 3, 10 reported separately. Per frame: the real fingers' inner
edges per row (330-470, fit_finger_mount.py's dark mask; rows where the two fingers show apart), against the sim's at
the frame's recorded state with the finger opening (both fingers, symmetric) searched 10-45 mm (the r13 wrist camera,
CAD fingers, the unit's joint zeros); the opening whose inner edges match best (median |error| per row). The held
opening is the grasped section's width where the fingers touch it.
usage: resistor_hold_width.py STATES_PARQUET FRAMES_X5 OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
HERE = Path(__file__).resolve().parent
OUT = sys.argv[3]; sys.argv = sys.argv[:3] + ["/dev/null"]
src = (HERE / "fit_finger_mount.py").read_text().split("p0 = np.zeros(6)")[0]
ns = {"__file__": str(HERE / "fit_finger_mount.py"), "__name__": "ffm"}; exec(compile(src, "fit_finger_mount.py", "exec"), ns)
m, d, st, F, lat = ns["m"], ns["d"], ns["st"], ns["F"], ns["lat"]
fadr = [m.jnt_qposadr[m.joint(n).id] for n in ("joint_left", "joint_right")]
G = json.load(open(Path(os.environ["REAL2SIM_ROOT"]).expanduser() / "real/grip_widths.json"))
# r16: CAMERA=fit renders from the joint fit's wrist camera (fit_gripper_camera.py); ALL=1 takes every carry plateau (CPU too)
if os.environ.get("CAMERA") == "fit":
    _fit = json.load(open(Path(os.environ["REAL2SIM_ROOT"]).expanduser() / "real/r16/fit_gripper_camera.json"))["fit"]
    m.cam_pos[ns["cam"]] = _fit["camera_pos_m"]; m.cam_quat[ns["cam"]] = _fit["camera_quat"]; m.cam_fovy[ns["cam"]] = _fit["fovy"]
ALL = os.environ.get("ALL") == "1"

def frames(eps):
    out = []
    for ep in eps:
        if str(ep) not in G["episodes"]: continue
        rows = st[st.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
        for pl in G["episodes"][str(ep)]["carry_plateaus"]:
            if pl["opening_mm"] > 22 and not ALL: continue   # resistors (CPU, battery hold wider)
            for fr in np.linspace(pl["frames"][0] + 5, pl["frames"][1] - 5, 4).astype(int):
                ms = int(round(fr / 30 * 1000)); f = sorted(F.glob(f"gripper_e{ep:03d}_t*.png"), key=lambda p: abs(int(p.stem.split("_t")[1]) - ms))[0]
                t = int(f.stem.split("_t")[1]) / 1000; tl = float(np.clip(t - lat, ts[0], ts[-1]))
                out.append((ep, f, np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])]), pl["opening_mm"]))
    return out

def inner(e):   # rows where both fingers show apart: (left inner, right inner)
    ok = np.isfinite(e[:, 1]) & np.isfinite(e[:, 2]) & ((e[:, 2] - e[:, 1]) > 10)
    return e[:, 1:3], ok

res = {"source": "review/export/robotcam_allowed (frames_x5 gripper, states_allowed.parquet, real/grip_widths.json)", "frames": {}}
for split, eps in (("train", (0, 2, 6, 7, 9)), ("dev", (1, 3, 10))):
    rows = []
    for ep, f, state, reading_mm in frames(eps):
        re_, ok = inner(ns["edges"](ns["real_mask"](cv2.imread(str(f)))))
        if ok.sum() < 3: continue
        best = None
        for w in np.arange(10.0, 45.01, 0.5):
            ns["ch"].set_positions(state)
            for adr in fadr: d.qpos[adr] = w / 2000.0
            se, _ = inner(ns["edges"](ns["sim_mask"]()))
            err = np.abs(se - re_)[ok]; err = err[np.isfinite(err)]
            if len(err) < 4: continue
            c = float(np.median(err))
            if best is None or c < best[1]: best = (w, c)
        if best: rows.append({"frame": f.name, "episode": ep, "reading_mm": round(float(reading_mm), 1), "reading_deg": round(float(state[-1]), 1), "held_opening_mm": best[0], "inner_err_px": round(best[1], 1)})
    W = np.array([r["held_opening_mm"] for r in rows])
    res["frames"][split] = rows
    res[split] = {"n": len(rows), "median_mm": float(np.median(W)) if len(W) else None, "p25_p75_mm": np.percentile(W, [25, 75]).round(1).tolist() if len(W) else None}
    print(split, res[split], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
