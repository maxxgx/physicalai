"""r16 (D7, coordinator-approved camera-model fit): a slow global exposure drift for a still scene, fitted on TRAIN.

Real still content (still_temporal.py on the allowed export) wanders in global brightness far more than the sim's: the
median |change of the frame mean| at lags 2-16 frames is 0.08-0.15 luma levels (TRAIN), the sim's 0.02. The term: a gain
g(t) = sigma * (a stationary Gaussian process of unit variance with a Lorentzian spectrum, corner f_c), a deterministic
function of sim time per camera (32 sinusoids, frequencies log-spaced 0.05-15 Hz, amplitudes from the spectrum, phases
drawn once per camera): out = img * (1 + g(t)). Evaluated offline through the recording path's codec: a still scene
(the allowed episodes' still states) rendered by the delivery's compositor at successive sim times (1/30 s), encoded
with Studio's settings (libsvtav1, crf 30, preset 12, g 2, yuv420p), decoded, and the same statistics computed. Grid of
(sigma, f_c); the cost: the mean |log| ratio of D(k), k = 2, 4, 8, 16 against TRAIN; DEV reported per candidate.
usage: drift_term_fit.py STATES_PARQUET REAL_STILL_JSON OUT_JSON [--camera overview]"""
import json, os, sys, io
from pathlib import Path
import numpy as np, cv2, mujoco, pandas as pd, av
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
from physicalai_mujoco_plugin.real2sim import desk_compositor as dc

ST, REAL, OUT = pd.read_parquet(sys.argv[1]), json.load(open(sys.argv[2])), sys.argv[3]
CAM = sys.argv[sys.argv.index("--camera") + 1] if "--camera" in sys.argv else "overview"
LAGS = (2, 4, 8, 16)
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot)
comp = b.compositor(robot, show_objects=False); ch = robot._sim.channels[0]

def drift_gain(t, sigma, fc, seed):
    if sigma <= 0: return 0.0
    rng = np.random.default_rng(seed); f = np.geomspace(0.05, 15.0, 32); ph = rng.uniform(0, 2 * np.pi, 32)
    w = np.gradient(f) / (1 + (f / fc) ** 2); a = np.sqrt(2 * w / w.sum())   # unit variance
    return float(sigma * np.sum(a * np.sin(2 * np.pi * f * t + ph)))

def still_states(eps, n=5):
    out = []
    for ep in eps:
        g = ST[ST.episode_index == ep].sort_values("timestamp"); S = np.stack(g["observation.state"].to_numpy()).astype(float)
        dq = np.r_[np.inf, np.abs(np.diff(S, axis=0)).max(1)]; idx = np.flatnonzero(dq < 0.05)
        if len(idx): out.append(S[idx[len(idx) // 2]])
    return out[:n]

def encode_decode(frames):
    buf = io.BytesIO(); c = av.open(buf, "w", format="mp4"); s = c.add_stream("libsvtav1", rate=30)
    s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"; s.options = {"crf": "30", "preset": "12", "g": "2"}
    for f in frames:
        for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")): c.mux(pkt)
    for pkt in s.encode(): c.mux(pkt)
    c.close(); buf.seek(0); dec = av.open(buf)
    return [fr.to_ndarray(format="gray").astype(np.float32) for fr in dec.decode(video=0)]

def stats(gray):
    F = np.stack(gray); mval = F.reshape(len(F), -1).mean(1)
    return {k: float(np.median(np.abs(mval[k:] - mval[:-k]))) for k in LAGS}

def run(sigma, fc, states, n=64):
    D = {k: [] for k in LAGS}
    for i, state in enumerate(states):
        ch.set_positions(state); frames = []
        for k in range(n):
            t = 100.0 + i * 50 + k / 30.0; d.time = t; mujoco.mj_forward(m, d)
            img = comp.render_camera(CAM)["rgb"].astype(np.float32) * (1.0 + drift_gain(t, sigma, fc, 11 if CAM == "overview" else 12))
            frames.append(np.clip(np.rint(img), 0, 255).astype(np.uint8))   # rounded, as the compositor's convertScaleAbs
        for k, v in stats(encode_decode(frames)).items(): D[k].append(v)
    return {k: float(np.median(v)) for k, v in D.items()}

tr = {int(k): v for k, v in REAL["train"]["global_mean_D"].items()}; dv = {int(k): v for k, v in REAL["dev"]["global_mean_D"].items()}
cost = lambda D, ref: float(np.mean([abs(np.log(max(D[k], 1e-4) / max(ref[k], 1e-4))) for k in LAGS]))
states_tr, states_dv = still_states((0, 2, 6, 7, 9)), still_states((1, 3, 10))
res = {"camera": CAM, "targets": {"train": tr, "dev": dv}, "grid": []}
base = run(0.0, 1.0, states_tr); res["off"] = {"D": base, "cost_train": cost(base, tr)}; print("off", base, round(cost(base, tr), 3), flush=True)
GRID = [(0.0005, 0.1), (0.001, 0.3), (0.002, 0.1), (0.002, 0.3), (0.002, 1.0), (0.0025, 0.2), (0.0025, 0.3), (0.0025, 0.5),
        (0.003, 0.2), (0.003, 0.3), (0.003, 0.5), (0.004, 0.1), (0.004, 0.3)]   # r16: refined around the codec's threshold
for sigma, fc in GRID:
    if True:
        Dt = run(sigma, fc, states_tr); row = {"sigma": sigma, "fc_hz": fc, "D_train": Dt, "cost_train": cost(Dt, tr)}
        res["grid"].append(row); print(row, flush=True)
best = sorted(res["grid"], key=lambda r: r["cost_train"])[:3]
for r in best:
    Dd = run(r["sigma"], r["fc_hz"], states_dv); r["D_dev"] = Dd; r["cost_dev"] = cost(Dd, dv); print("dev", r["sigma"], r["fc_hz"], Dd, round(r["cost_dev"], 3), flush=True)
res["chosen_on_dev"] = min(best, key=lambda r: r["cost_dev"])
json.dump(res, open(OUT, "w"), indent=1); print("chosen", res["chosen_on_dev"]["sigma"], res["chosen_on_dev"]["fc_hz"])
