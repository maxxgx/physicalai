"""r15 (D2 / D1, the user's mount): the wrist mount's drawn colour matched to the real (brown) print in the overview.

Allowed overview frames only (the reviewer's export: frames_x5 and states_allowed.parquet); TRAIN episodes 0, 2, 6, 7, 9
fit, DEV 1, 3, 10 reported. Per frame: the arm at t - latency, the episode's bucket layout, the overview composite
(the delivery's compositor), and the mount's visible pixels (a segmentation render of every drawn geom from the same
camera: pixels whose front geom is the mount, eroded 2 px; frames with >= 150 such pixels). The mount's RGB is scaled
per channel (3 rounds) so the composite's median over the TRAIN pixels matches the real one; the error reported is the
median |composite - real| per channel (0-255) over each split's pixels, before and after.
Writes $REAL2SIM_ROOT/real/r15/mount_colour.json; with --write, the kit look file's "mount:wrist" entry.
usage: mount_colour_fit.py STATES_PARQUET FRAMES_X5 [--write]"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b

ST, F = pd.read_parquet(sys.argv[1]), Path(sys.argv[2]); ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
kit = json.load(open(b.KIT)); lat = float(kit["camera_latency_s"]["overview"])
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot)
comp = b.compositor(robot, show_objects=False); ch = robot._sim.channels[0]
cam = m.camera("overview").id; mount = m.geom("wrist_mount").id
seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering()
opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
rng = np.random.default_rng(0)

def frames(eps, per=12):
    out = []
    for ep in eps:
        rows = ST[ST.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
        files = sorted(F.glob(f"overview_e{ep:03d}_t*.png"))
        for f in [files[i] for i in sorted(rng.choice(len(files), size=min(per * 3, len(files)), replace=False))]:
            t = int(f.stem.split("_t")[1]) / 1000.0; tl = float(np.clip(t - lat, ts[0], ts[-1]))
            out.append((ep, f, t, np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])])))
    return out

def pose(ep, t, state):
    _, poses = b.episode_layout(ep)
    if poses: b.set_bucket_poses(robot, poses)
    ch.set_positions(state); d.time = t; mujoco.mj_forward(m, d)

def mask():
    seg.update_scene(d, camera=cam, scene_option=opt); s = seg.render()
    mk = (s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (s[..., 0] == mount)
    return cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

sets = {"train": frames((0, 2, 6, 7, 9)), "dev": frames((1, 3, 10))}
use = {}
for split, fr in sets.items():
    keep = []
    for ep, f, t, state in fr:
        pose(ep, t, state); mk = mask()
        if mk.sum() >= 150: keep.append((ep, f, t, state, mk))
    use[split] = keep[:40]; print(split, "frames with the mount visible:", len(use[split]), flush=True)

def measure(split):
    sim, real = [], []
    for ep, f, t, state, mk in use[split]:
        pose(ep, t, state); rgb = comp.render_camera("overview")["rgb"]; img = cv2.imread(str(f))[:, :, ::-1]
        sim.append(rgb[mk].astype(float)); real.append(img[mk].astype(float))
    S, R = np.vstack(sim), np.vstack(real)
    return np.median(S, 0), np.median(R, 0), np.median(np.abs(S - R), 0)

rgba0 = m.geom_rgba[mount].copy(); res = {"source": "review/export/robotcam_allowed (frames_x5 overview, states_allowed.parquet)", "frames": {k: [f.name for _, f, *_ in v] for k, v in use.items()},
                                          "rgba_profile": rgba0.round(4).tolist()}
before = {k: measure(k) for k in use}
for _ in range(3):
    s_med, r_med, _ = measure("train")
    m.geom_rgba[mount, :3] = np.clip(m.geom_rgba[mount, :3] * np.clip(r_med / np.maximum(s_med, 1.0), 0.5, 2.0), 0.0, 1.0)
after = {k: measure(k) for k in use}
res["rgba_fitted"] = m.geom_rgba[mount].round(4).tolist()
for k in use:
    res[k] = {"n_frames": len(use[k]), "real_median_rgb": before[k][1].round(1).tolist(), "sim_median_rgb_before": before[k][0].round(1).tolist(), "sim_median_rgb_after": after[k][0].round(1).tolist(),
              "median_abs_err_before": before[k][2].round(1).tolist(), "median_abs_err_after": after[k][2].round(1).tolist()}
    print(k, json.dumps(res[k]), flush=True)
(ROOT / "real/r15").mkdir(parents=True, exist_ok=True); json.dump(res, open(ROOT / "real/r15/mount_colour.json", "w"), indent=1)
print("rgba", res["rgba_profile"], "->", res["rgba_fitted"])
if "--write" in sys.argv and np.mean(res["dev"]["median_abs_err_after"]) < np.mean(res["dev"]["median_abs_err_before"]):
    look_path = (b.KIT.parent / kit["look"]).resolve(); look = json.load(open(look_path))
    look["mount:wrist"] = res["rgba_fitted"]; json.dump(look, open(look_path, "w"), indent=1); print("look", look_path, "mount:wrist written")
