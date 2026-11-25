"""Temporal noise of a recorded LeRobot v3 dataset (as Studio writes it), measured on flat static patches during still
runs, so it can be compared between the REAL episodes and a SIM recording made through the same Studio pipeline.

Real data: pass --lock review/HELDOUT_EPISODES.json. Only the lock's allowed episodes are used, and only inside their
usable (guarded) file ranges; each segment is decoded from the first keyframe inside that range as a raw stream
(real_episodes.allowed_segments), so no held-out frame is ever produced. Rows of data/*.parquet are filtered to the
allowed episodes before anything else is read.

Still runs: every joint changes less than STILL (public units) per frame for >= 8 frames. Per run and camera the
temporal mean/std per pixel; flat pixels: Sobel magnitude of the temporal mean < FLAT levels/px (edges flicker under
sub-pixel coding drift and would read as noise). Reported per camera:
  - sigma(level) per channel and for luma (BT.601 Y) in 16-level bins, and a fit sigma^2 = a + b * level;
  - chroma/luma ratio (Cb, Cr std over Y std);
  - spatial structure: correlation of the temporal residual at x/y lags 1, 2, 4 px (flat pixels);
  - temporal structure: lag-1 correlation of the residual across consecutive frames;
  - coding traces: residual energy on 8-px block boundaries over interior columns/rows (1 = none), and the fraction
    of flat pixels whose value repeats exactly between consecutive frames (copied/skip blocks).
usage: noise_real.py DATASET OUT.json [--lock LOCK] [--cams overview,gripper] [--max-runs 40]"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_episodes as R

ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("out"); ap.add_argument("--lock")
ap.add_argument("--cams", default="overview,gripper"); ap.add_argument("--max-runs", type=int, default=40)
ap.add_argument("--still", type=float, default=0.05); ap.add_argument("--flat", type=float, default=2.0)
ap.add_argument("--export", action="store_true", help="DATASET is review/export/robotcam_allowed (H3)")
a = ap.parse_args(); D = Path(a.dataset)
info = json.loads((D / ("info.json" if a.export else "meta/info.json")).read_text()); fps = info["fps"]
W, H = 640, 480

if a.export:   # per-episode files, frame k = frame_first + k; decoded from their first frame (file time 0)
    df = pd.read_parquet(D / "states_allowed.parquet", columns=["episode_index", "frame_index", "observation.state"])
    segs = [{**s_, "file_start": 0.0, "file_end": (s_["frames"] + 0.5) / fps} for s_ in R.export_segments(D)]
elif a.lock:
    lock = json.loads(Path(a.lock).read_text()); allowed = [int(e) for e in lock["allowed"]]
    files = sorted({lock["episodes"][str(e)]["data_file"] for e in allowed})
    df = pd.concat([pd.read_parquet(D / f, columns=["episode_index", "frame_index", "observation.state"]) for f in files])
    df = df[df["episode_index"].isin(allowed)]
    segs = R.allowed_segments(D, a.lock)
else:   # own sim recording: every episode; segments = whole episodes from meta/episodes (keyframe starts)
    df = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state"]) for p in sorted((D / "data").rglob("*.parquet"))])
    segs = []
    for ep, per in R.episode_videos(D).items():
        for cam, (path, f0, f1) in per.items():
            start = next((k for k in R.keyframes(path) if f0 - 1e-6 <= k < f1), None)
            if start is not None: segs.append({"episode": ep, "camera": cam, "video": str(path), "file_start": start, "file_end": f1, "episode_start": start - f0, "episode_end": f1 - f0})
df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

runs = {}   # episode -> [(t0, t1)] episode time of still runs
for ep, g in df.groupby("episode_index"):
    st = np.stack(g["observation.state"].values); fi = g["frame_index"].to_numpy()
    still = np.r_[False, np.abs(np.diff(st, axis=0)).max(1) < a.still]
    i = 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]: j += 1
            if j - i >= 8: runs.setdefault(int(ep), []).append((fi[i - 1] / fps, fi[j - 1] / fps))
            i = j
        else: i += 1

def ycc(x):   # BT.601 full range, float
    x = x.astype(np.float32); r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.stack([y, 0.564 * (b - y) + 128, 0.713 * (r - y) + 128], -1)

def corr(res, mask, dx, dy):
    a_ = res[:, dy:, dx:] if True else None
    h, w = res.shape[1:3]
    A = res[:, :h - dy, :w - dx]; B = res[:, dy:, dx:]; mk = mask[:h - dy, :w - dx] & mask[dy:, dx:]
    A, B = A[:, mk], B[:, mk]
    return float((A * B).mean() / np.sqrt((A * A).mean() * (B * B).mean() + 1e-12))

decoded: list = []
out = {"dataset": str(D), "lock": a.lock, "export": a.export, "still_deg_per_frame": a.still, "flat_levels_per_px": a.flat, "cameras": {}}
for cam in a.cams.split(","):
    lv, sd, sdy, sdc, cx, cy, tl, blk, rep, used = [], [], [], [], {}, {}, [], [], [], []
    for s in [s for s in segs if s["camera"] == cam]:
        rr = [(t0, t1) for t0, t1 in runs.get(s["episode"], []) if t0 >= s["episode_start"] - 1e-6 and t1 <= s["episode_end"] + 1e-6]
        if not rr: continue
        # decode this allowed segment once as raw RGB, keep frames in still runs
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{s['file_start']:.6f}", "-i", s["video"], "-t", f"{s['file_end'] - s['file_start'] - 1e-3:.6f}",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "-"], stdout=subprocess.PIPE)
        buf = {k: [] for k in range(len(rr))}; n = 0
        while True:
            raw = p.stdout.read(W * H * 3)
            if len(raw) < W * H * 3: break
            t = s["episode_start"] + n / fps; n += 1
            for k, (t0, t1) in enumerate(rr):
                if t0 - 1e-6 <= t <= t1 + 1e-6: buf[k].append(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
        p.wait()
        decoded.append({"episode": s["episode"], "camera": cam, "video": s["video"], "decoded_frames": n,
                         "file_time_s": [round(s["file_start"], 4), round(s["file_end"], 4)], "still_runs_s": [list(map(lambda v: round(v, 4), r_)) for r_ in rr]})
        for k, fr in buf.items():
            if len(fr) < 8 or len(used) >= a.max_runs: continue
            F = np.stack(fr).astype(np.float32); used.append(f"e{s['episode']} {rr[k][0]:.2f}-{rr[k][1]:.2f}s ({len(fr)} frames)")
            mean = F.mean(0); res = F - mean
            g = cv2.cvtColor(mean.astype(np.float32), cv2.COLOR_RGB2GRAY)
            mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8)
            flat = cv2.erode((mag < a.flat).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            flat[:8], flat[-8:], flat[:, :8], flat[:, -8:] = False, False, False, False
            if flat.sum() < 2000: continue
            std = F.std(0, ddof=1)
            Y = ycc(F); Yr = Y - Y.mean(0); ystd = Yr.std(0, ddof=1)
            lv.append(mean[flat]); sd.append(std[flat]); sdy.append(np.c_[Y.mean(0)[..., 0][flat], ystd[..., 0][flat]])
            sdc.append(np.c_[ystd[..., 1][flat], ystd[..., 2][flat], ystd[..., 0][flat]])
            yr = Yr[..., 0]
            for L in (1, 2, 4):
                cx.setdefault(L, []).append(corr(yr, flat, L, 0)); cy.setdefault(L, []).append(corr(yr, flat, 0, L))
            A, B = yr[:-1][:, flat], yr[1:][:, flat]; tl.append(float((A * B).mean() / np.sqrt((A * A).mean() * (B * B).mean() + 1e-12)))
            e2 = (yr ** 2).mean(0); cols = np.arange(8, W - 8); rows = np.arange(8, H - 8)
            colE = np.array([e2[:, c][flat[:, c]].mean() if flat[:, c].any() else np.nan for c in range(W)])
            rowE = np.array([e2[r][flat[r]].mean() if flat[r].any() else np.nan for r in range(H)])
            bc = np.nanmean(colE[cols][cols % 8 == 0]) / np.nanmean(colE[cols][cols % 8 != 0])
            br = np.nanmean(rowE[rows][rows % 8 == 0]) / np.nanmean(rowE[rows][rows % 8 != 0])
            blk.append((float(bc), float(br)))
            u8 = F.astype(np.uint8); rep.append(float((u8[1:] == u8[:-1]).all(-1)[:, flat].mean()))
    if not lv:
        out["cameras"][cam] = {"note": "no still run with enough flat area"}; continue
    lv, sd = np.concatenate(lv), np.concatenate(sd); sdy = np.concatenate(sdy); sdc = np.concatenate(sdc)
    bins = np.arange(0, 257, 16); table = []
    for i in range(len(bins) - 1):
        mk = (sdy[:, 0] >= bins[i]) & (sdy[:, 0] < bins[i + 1])
        if mk.sum() < 500: continue
        ch = [float(np.median(sd[(lv[:, c] >= bins[i]) & (lv[:, c] < bins[i + 1]), c])) if ((lv[:, c] >= bins[i]) & (lv[:, c] < bins[i + 1])).sum() > 200 else None for c in range(3)]
        table.append({"level": int(bins[i] + 8), "pixels": int(mk.sum()), "luma_sigma": round(float(np.median(sdy[mk, 1])), 3), "rgb_sigma": [round(x, 3) if x else None for x in ch]})
    Lc = np.array([t["level"] for t in table], float); S2 = np.array([t["luma_sigma"] for t in table]) ** 2
    fit = np.polyfit(Lc, S2, 1) if len(Lc) >= 2 else [np.nan, np.nan]
    out["cameras"][cam] = {
        "runs": used, "flat_pixels": int(len(lv)), "luma_sigma_by_level": table,
        "luma_fit_sigma2": {"a_read": round(float(fit[1]), 4), "b_shot_per_level": round(float(fit[0]), 5)},
        "chroma_over_luma": {"cb": round(float(np.median(sdc[:, 0]) / np.median(sdc[:, 2])), 3), "cr": round(float(np.median(sdc[:, 1]) / np.median(sdc[:, 2])), 3)},
        "spatial_corr_x": {L: round(float(np.mean(v)), 3) for L, v in cx.items()}, "spatial_corr_y": {L: round(float(np.mean(v)), 3) for L, v in cy.items()},
        "temporal_lag1_corr": round(float(np.mean(tl)), 3),
        "block8_boundary_energy_ratio": {"cols": round(float(np.mean([b[0] for b in blk])), 3), "rows": round(float(np.mean([b[1] for b in blk])), 3)},
        "exact_repeat_fraction": round(float(np.mean(rep)), 3)}
    c = out["cameras"][cam]
    print(cam, "runs", len(used), "| luma sigma by level", [(t["level"], t["luma_sigma"]) for t in table], "| fit", c["luma_fit_sigma2"],
          "| chroma/luma", c["chroma_over_luma"], "| corr x", c["spatial_corr_x"], "y", c["spatial_corr_y"], "| t-lag1", c["temporal_lag1_corr"],
          "| block8", c["block8_boundary_energy_ratio"], "| repeat", c["exact_repeat_fraction"], flush=True)
out["decode_record"] = decoded
Path(a.out).write_text(json.dumps(out, indent=1))
