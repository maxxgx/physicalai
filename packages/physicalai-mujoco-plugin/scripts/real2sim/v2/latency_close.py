"""Latency between the real camera frames and the joint states (export only): at allowed frames where the arm moves
fast, render the overview at the state of t + delta for a range of delta and score psnr_cc over a fixed region around
the arm (the union of the sim arm's silhouettes over all deltas, dilated); the best delta is the camera's latency
relative to the states. usage: latency_scan.py EXPORT FRAMES_X5 OUT.json [--n 40]"""
import sys, json, argparse, numpy as np, cv2, mujoco, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
ap = argparse.ArgumentParser(); ap.add_argument("export"); ap.add_argument("frames"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=40); ap.add_argument("--cam", default="overview")
a = ap.parse_args(); F = Path(a.frames)
st = pd.read_parquet(Path(a.export) / "states_allowed.parquet")
r = b.make_robot(0); b.reset(r, 0); m, d = r._sim.model, r._sim.data; b.matched_state(r)
comp = b.compositor(r, show_objects=False); ch = r._sim.channels[0]
def state_at(ep, t):
    rows = st[st.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
    return np.array([np.interp(t, ts, S[:, j]) for j in range(S.shape[1])]), ts, S
cands = []
for p in sorted(F.glob(f"{'gripper' if a.cam == 'gripper' else 'overview'}_e*_t*.png")):
    ep, ms = int(p.stem.split("_e")[1][:3]), int(p.stem.split("_t")[1]); t = ms / 1000
    s0, ts, S = state_at(ep, t); s1, _, _ = state_at(ep, t + 0.1)
    if ts[0] + 0.3 < t < ts[-1] - 0.3: cands.append((float(np.abs(s1[:6] - s0[:6]).max()) / 0.1, ep, t, p))
# r12: close views only (the arm large in the overview: its sim silhouette over CLOSE_PX at the frame's own state),
# where a latency error moves the silhouette the most
CLOSE_PX = 15000
close = []
for speed, ep, t, p in sorted(cands, reverse=True):
    s_, _, _ = state_at(ep, t); ch.set_positions(s_); d.time = t; mujoco.mj_forward(m, d)
    res_ = comp.render_camera("overview", outputs=True); area = int((res_["seg"] == 10).sum())
    if area > CLOSE_PX: close.append((speed, ep, t, p))
    if len(close) >= a.n: break
cands = close
deltas = np.round(np.arange(-0.30, 0.301, 1 / 30), 4); cam = "wrist" if a.cam == "gripper" else "overview"
scores = {float(x): [] for x in deltas}; used, skipped = [], []
def colour_fit(ren, real, mk):
    o = ren.copy()
    for c in range(3):
        A = np.stack([ren[..., c][mk], np.ones(int(mk.sum()))], 1); g, bb = np.linalg.lstsq(A, real[..., c][mk], rcond=None)[0]; o[..., c] = np.clip(g * ren[..., c] + bb, 0, 1)
    return o
for speed, ep, t, p in cands:
    real = cv2.imread(str(p))[:, :, ::-1].astype(np.float32) / 255; rend, segs = {}, []
    for dl in deltas:
        s, _, _ = state_at(ep, t + dl); ch.set_positions(s); d.time = t; mujoco.mj_forward(m, d)
        res = comp.render_camera(cam, outputs=True); rend[float(dl)] = res["rgb"].astype(np.float32) / 255; segs.append(res["seg"] == 10)
    mk = cv2.dilate(np.any(segs, 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if a.cam == "gripper":   # the view itself moves with the arm: score everything but the fingers
        mk = ~(cv2.dilate(np.all(segs, 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    if mk.sum() < 2000: skipped.append(p.name); continue
    used.append(p.name)
    for dl, ren in rend.items():
        cc = colour_fit(ren, real, mk); scores[dl].append(float(-10 * np.log10(np.mean((cc[mk] - real[mk]) ** 2) + 1e-12)))
med = {dl: float(np.median(v)) for dl, v in scores.items() if v}
best = max(med, key=med.get)
json.dump({"source": "review/export/robotcam_allowed", "camera": a.cam, "frames": len(used), "frame_list": used, "candidates_skipped_small_mask": skipped,
           "selection": f"r12 close views: of the allowed frames by fastest joint motion (max joint speed over the next 0.1 s, at least 0.3 s from either end), the first {a.n} whose sim arm silhouette in the overview exceeds {CLOSE_PX} px", "median_psnr_cc_by_delta_s": med, "best_delta_s": best,
           "meaning": "the real frame at time t shows the arm state at t + best_delta (negative: the camera lags the states)"}, open(a.out, "w"), indent=1)
print("best delta", best, {k: round(v, 2) for k, v in med.items()})
