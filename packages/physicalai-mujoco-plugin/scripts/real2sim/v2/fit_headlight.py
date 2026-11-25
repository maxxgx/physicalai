"""D1/D2 (r12): the robot cameras' headlight (kit robotcam_headlight: ambient, diffuse) fitted so the sim-drawn buckets
match the real ones' brightness (the office light is diffuse: real bucket walls are lit on every face; the sim's one
overhead light leaves vertical faces dark). Allowed frames only: ROBOTCAM_REPLAY-style renders (segment layout, the
kit, objects parked) at frames of TRAIN episodes (0, 2, 6, 7, 9), bucket pixels from the sim's segmentation (labels
1-4, eroded 3 px, the arm's pixels excluded), per bucket the median colour against the real frame's (after the camera
model, as streamed), mean |diff| per channel; grid over (ambient, diffuse); DEV episodes (1, 3, 10) reported. usage: fit_headlight.py STATES FRAMES OUT"""
import json, sys
from pathlib import Path
import numpy as np, cv2, mujoco, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
STATES, F, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
st = pd.read_parquet(STATES); lat = float(json.load(open(b.KIT))["camera_latency_s"]["wrist"])
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot)
comp = b.compositor(robot, show_objects=False); ch = robot._sim.channels[0]
def frames(eps, per=6):
    out = []
    for ep in eps:
        for cam, real in (("overview", "overview"), ("wrist", "gripper")):
            fs = sorted(F.glob(f"{real}_e{ep:03d}_t*.png")); out += [(ep, cam, f) for f in [fs[i] for i in np.linspace(0, len(fs) - 1, per).astype(int)]]
    return out
TRAIN, DEV = frames((0, 2, 6, 7, 9)), frames((1, 3, 10))
def state_at(ep, t):
    rows = st[st.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
    tl = float(np.clip(t - lat, ts[0], ts[-1])); return np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])])
def err(items, amb, dif):
    object.__setattr__(comp.kit, "robotcam_headlight", {"ambient": amb, "diffuse": dif}); per = []   # (a frozen dataclass)
    for ep, cam, f in items:
        _, poses = b.episode_layout(ep); b.set_bucket_poses(robot, poses)
        t = int(f.stem.split("_t")[1]) / 1000; ch.set_positions(state_at(ep, t)); d.time = t; mujoco.mj_forward(m, d)
        r = comp.render_camera(cam, outputs=True); real = cv2.imread(str(f))[:, :, ::-1].astype(np.float32); sim = r["rgb"].astype(np.float32)
        for lab in (1, 2, 3, 4):   # per bucket, the median colour (robust to a few px of misregistration and to contents)
            mk = cv2.erode((r["seg"] == lab).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            if mk.sum() < 400: continue
            per.append(float(np.abs(np.median(sim[mk], axis=0) - np.median(real[mk], axis=0)).mean()))
    return float(np.mean(per)), len(per)
res = {"source": "review/export/robotcam_allowed (frames_x5, states_allowed.parquet)", "train_frames": [f.name for _, _, f in TRAIN], "dev_frames": [f.name for _, _, f in DEV], "grid": []}
base = err(TRAIN, 0.28, 0.42); print("r11 headlight (0.28, 0.42): TRAIN bucket |diff|", round(base[0], 2), "levels over", base[1], "frames", flush=True)
for amb in (0.28, 0.45, 0.6, 0.75, 0.9):
    for dif in (0.2, 0.42, 0.6):
        e, n = err(TRAIN, amb, dif); res["grid"].append([amb, dif, round(e, 3)]); print(amb, dif, round(e, 2), flush=True)
best = min(res["grid"], key=lambda g: g[2]); res["best"] = {"ambient": best[0], "diffuse": best[1], "train": best[2], "train_r11": round(base[0], 3)}
res["best"]["dev"] = round(err(DEV, best[0], best[1])[0], 3); res["best"]["dev_r11"] = round(err(DEV, 0.28, 0.42)[0], 3)
print("best", res["best"]); json.dump(res, open(OUT, "w"), indent=1)
