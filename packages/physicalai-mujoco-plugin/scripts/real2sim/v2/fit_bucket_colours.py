"""Fit the buckets' material colours (the sim buckets the robot cameras draw) to the real robot cameras, export only:
at allowed frames (frames_x5, states_allowed.parquet; the camera latency of the kit), render both cameras through the
full pipeline (fitted pose, tone curve; noise off), take each bucket's wall pixels (its seg label, eroded, minus pixels
whose sim colour is far from the wall's own, i.e. the sim pile) and compare robust medians with the real frame there;
scale the material colour in linear light by real/sim and iterate. One colour per bucket (a fitted colour, H2).
Writes align/bucket_colors_robotcam.json (colours, before/after per camera, the frame list).
usage: fit_bucket_colours.py EXPORT FRAMES_X5 [--n 24] [--iters 3]"""
import argparse, json, os, sys, numpy as np, cv2, mujoco, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["R2S_SENSOR_NOISE"] = "0"
import bucketsim_v2 as b
from physicalai_mujoco_plugin.real2sim.desk_compositor import COLORS, LABEL_BUCKET
ap = argparse.ArgumentParser(); ap.add_argument("export"); ap.add_argument("frames"); ap.add_argument("--n", type=int, default=24); ap.add_argument("--iters", type=int, default=3)
a = ap.parse_args(); F = Path(a.frames); ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
st = pd.read_parquet(Path(a.export) / "states_allowed.parquet")
names = sorted(p.name for p in F.glob("overview_e*_t*.png")); pick = [names[i] for i in np.linspace(0, len(names) - 1, a.n).astype(int)]
r = b.make_robot(0); b.reset(r, 0); m, d = r._sim.model, r._sim.data; b.matched_state(r)
comp = b.compositor(r, show_objects=False); ch = r._sim.channels[0]; lat = json.load(open(b.KIT)).get("camera_latency_s", {})
mat = {c: m.material(f"bucket_{c}").id for c in COLORS}
lin = lambda v: np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
def state(ep, t):
    rows = st[st.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
    return np.array([np.interp(np.clip(t, ts[0], ts[-1]), ts, S[:, j]) for j in range(S.shape[1])])
def measure():
    acc = {(c, cam): ([], []) for c in COLORS for cam in ("overview", "gripper")}
    for n in pick:
        ep, ms = int(n.split("_e")[1][:3]), int(n.split("_t")[1][:7]); t = ms / 1000
        for cam, real_name in (("overview", "overview"), ("wrist", "gripper")):
            ch.set_positions(state(ep, t - float(lat.get(cam, 0.0)))); d.time = t; mujoco.mj_forward(m, d)
            res = comp.render_camera(cam, outputs=True); sim = res["rgb"].astype(np.float32) / 255; seg = res["seg"]
            real = cv2.imread(str(F / n.replace("overview_", f"{real_name}_")))[:, :, ::-1].astype(np.float32) / 255
            for c in COLORS:
                mk = cv2.erode((seg == LABEL_BUCKET[c]).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                if mk.sum() < 300: continue
                med = np.median(sim[mk], 0); wall = mk & (np.abs(sim - med).max(-1) < 0.12)
                if wall.sum() < 200: continue
                acc[(c, real_name)][0].append(sim[wall]); acc[(c, real_name)][1].append(real[wall])
    return {k: (np.median(np.concatenate(s), 0), np.median(np.concatenate(rr), 0), int(sum(len(x) for x in s))) for k, (s, rr) in acc.items() if s}
before = {c: m.mat_rgba[mat[c], :3].copy().tolist() for c in COLORS}; hist = []
for it in range(a.iters + 1):
    res = measure(); hist.append({f"{c}/{cam}": {"sim": (v[0] * 255).round(1).tolist(), "real": (v[1] * 255).round(1).tolist(), "px": v[2]} for (c, cam), v in res.items()})
    print(f"iter {it}:", {k: (v['sim'], v['real']) for k, v in hist[-1].items() if k.endswith('/overview')}, flush=True)
    if it == a.iters: break
    for c in COLORS:   # both cameras together, weighted by pixels; ratio in linear light, damped
        num = np.zeros(3); den = 0
        for cam in ("overview", "gripper"):
            if (c, cam) in res:
                s_, r_, px = res[(c, cam)]; num += px * np.log(np.clip(lin(r_), 1e-4, 1) / np.clip(lin(s_), 1e-4, 1)); den += px
        if den: m.mat_rgba[mat[c], :3] = np.clip(m.mat_rgba[mat[c], :3] * np.exp(0.8 * num / den), 0.005, 1.0)
after = {c: m.mat_rgba[mat[c], :3].round(4).tolist() for c in COLORS}
out = {"source": "review/export/robotcam_allowed (frames_x5, states_allowed.parquet)", "method": "wall pixels of each sim bucket vs the real frame there, robust medians, material rgb scaled in linear light, damped 0.8, both cameras weighted by pixels",
       "material_rgb_before": before, "material_rgb_after": after, "iterations": hist, "frame_list": pick}
json.dump(out, open(ROOT / "align/bucket_colors_robotcam.json", "w"), indent=1); print("after", after)
