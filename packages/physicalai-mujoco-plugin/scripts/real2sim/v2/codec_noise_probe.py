"""D7 (b), CPU only: does the kit's sensor-noise model survive Studio's AV1 settings on a still scene? A still base frame
(the temporal median of an allowed episode's overview frames: no noise of its own), N frames of the kit's noise model
(the same code path as Compositor._sensor_noise: luma half-res + per-pixel fields, chroma quarter-res, the kit's
levels), optional variants, encoded with ffmpeg libsvtav1 crf 30 preset 12 g 2 yuv420p (Studio's encoder settings,
as the real dataset's), decoded; per frame parity: share of exact copies (< 0.02 at 160x120 grey) and the median
|diff|. usage: codec_noise_probe.py FRAMES_X5 EPISODE OUTDIR [--n 60]"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("frames"); ap.add_argument("episode", type=int); ap.add_argument("out"); ap.add_argument("--n", type=int, default=60)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
files = sorted(Path(a.frames).glob(f"overview_e{a.episode:03d}_t*.png"))
base = np.median(np.stack([cv2.imread(str(f)) for f in files[:: max(1, len(files) // 25)]]), axis=0).astype(np.uint8)[:, :, ::-1]
kit = json.load(open(Path.home() / "real2sim-data/bucket-v2/kit/real2sim.json"))["sensor_noise"]; spec = {**kit, **kit["cameras"]["overview"]}
v = np.arange(256) / 255.0; L = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
Lp, Lm = np.minimum(L + 1e-4, 1), np.maximum(L - 1e-4, 0); enc = lambda x: np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
lut = np.clip(np.sqrt(spec["read_lin"] ** 2 + spec["shot_lin"] * L) * 255.0 * (enc(Lp) - enc(Lm)) / (Lp - Lm), 0, 4.0).astype(np.float32)
probe = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
k2 = 1 / float(cv2.resize(probe, (128, 128), interpolation=cv2.INTER_LINEAR).std()); k4 = 1 / float(cv2.resize(probe, (256, 256), interpolation=cv2.INTER_LINEAR).std())
h, w = base.shape[:2]

def noisy(rng, gain=1.0, low=0.0):
    half = rng.standard_normal((h // 2, w // 2)).astype(np.float32); full = rng.standard_normal((h, w)).astype(np.float32)
    quarter = rng.standard_normal((h // 4, w // 4, 2)).astype(np.float32); q8 = rng.standard_normal((h // 8, w // 8)).astype(np.float32)
    ny = cv2.addWeighted(cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR), k2 * 0.8, full, 0.6, 0.0)
    if low: ny = (ny * np.sqrt(1 - low) + np.sqrt(low) * cv2.resize(q8, (w, h), interpolation=cv2.INTER_LINEAR) / float(cv2.resize(probe, (512, 512), interpolation=cv2.INTER_LINEAR).std())).astype(np.float32)
    nc = cv2.resize(quarter, (w, h), interpolation=cv2.INTER_LINEAR)
    ycc = cv2.cvtColor(np.ascontiguousarray(base), cv2.COLOR_RGB2YCrCb); sy = cv2.LUT(ycc[..., 0], lut.reshape(1, 256)) * gain
    noise = cv2.merge([ny * sy, nc[..., 0] * sy * k4 * spec["chroma"], nc[..., 1] * sy * k4 * spec["chroma_cb"]])
    return cv2.cvtColor(cv2.convertScaleAbs(ycc.astype(np.float32) + noise), cv2.COLOR_YCrCb2RGB)

def run(tag, **kw):
    rng = np.random.default_rng(1); raw = out / f"{tag}.yuv"; mp4 = out / f"{tag}.mp4"
    with open(raw, "wb") as fh:
        for _ in range(a.n): fh.write(np.ascontiguousarray(noisy(rng, **kw)).tobytes())
    subprocess.run(["nice", "-n", "15", "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30", "-i", str(raw),
                    "-c:v", "libsvtav1", "-crf", "30", "-preset", "12", "-g", "2", "-pix_fmt", "yuv420p", str(mp4)], check=True); raw.unlink()
    cap = cv2.VideoCapture(str(mp4)); prev = None; k = 0; md = {0: [], 1: []}; lf = {0: [], 1: []}
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            md[k % 2].append(float(np.abs(g - prev).mean()))
            lf[k % 2].append(float(np.abs(cv2.resize(g, (20, 15), interpolation=cv2.INTER_AREA) - cv2.resize(prev, (20, 15), interpolation=cv2.INTER_AREA)).mean()))
        prev = g; k += 1
    r = {p: (round(100 * float((np.array(md[p]) < 0.02).mean()), 1), round(float(np.median(md[p])), 3), round(float(np.median(lf[p])), 3)) for p in (0, 1)}
    print(f"{tag:22s} intra pairs (copies%, med|diff|, low-freq) {r[0]}   predicted pairs {r[1]}", flush=True); mp4.unlink()
    return r

for low in (0.5, 0.65, 0.8):
    for gain in (1.0, 1.15, 1.3):
        run(f"low{low}_gain{gain}", low=low, gain=gain)
