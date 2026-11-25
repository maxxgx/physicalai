"""D7 (r11): fit the sensor-noise model's low-frequency luma field (parametric noise only; no photo content at render
time) to real still pairs through the codec. Per camera: a still base frame (the temporal median of allowed frames of
a TRAIN episode, used only here to give the encoder realistic content), N frames of the kit's noise model with a share
`low` of the luma variance moved into a 1/8-resolution field and the level scaled by `gain`, encoded with libsvtav1
crf 30 preset 12 g 2 yuv420p (the real dataset's and Studio's settings), decoded; intra pairs give med|diff| and its
low-frequency part (fitted to the TRAIN targets of real/still_codec_real.json), predicted pairs the copy share (a
check, not fitted). The chosen values are then run on a DEV episode's base and compared with the DEV targets.
usage: fit_noise_lowfreq.py FRAMES_X5 OUT_JSON"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, cv2
F = Path(sys.argv[1]); OUT = sys.argv[2]
ROOT = Path.home() / "real2sim-data/bucket-v2"; kit = json.load(open(ROOT / "kit/real2sim.json"))["sensor_noise"]
target = json.load(open(ROOT / "real/still_codec_real.json"))
probe = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
up = lambda s: 1.0 / float(cv2.resize(probe, (64 * s, 64 * s), interpolation=cv2.INTER_LINEAR).std())
k2, k4, k8 = up(2), up(4), up(8)

def base_of(cam, ep, n=25):
    files = sorted(F.glob(f"{'overview' if cam == 'overview' else 'gripper'}_e{ep:03d}_t*.png"))
    if cam == "gripper": files = files[:6]   # the arm at rest at the episode's start (first ~0.8 s)
    files = files[:: max(1, len(files) // n)]
    return np.median(np.stack([cv2.imread(str(f)) for f in files]), axis=0).astype(np.uint8)[:, :, ::-1], [f.name for f in files]

def noise_frames(base, cam, low, gain, n, seed=1):
    spec = {**kit, **(kit.get("cameras") or {}).get("wrist" if cam == "gripper" else "overview", {})}
    v = np.arange(256) / 255.0; L = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    Lp, Lm = np.minimum(L + 1e-4, 1), np.maximum(L - 1e-4, 0); enc = lambda x: np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    lut = np.clip(np.sqrt(spec["read_lin"] ** 2 + spec["shot_lin"] * L) * 255.0 * (enc(Lp) - enc(Lm)) / (Lp - Lm), 0, 4.0).astype(np.float32)
    h, w = base.shape[:2]; rng = np.random.default_rng(seed); ycc = cv2.cvtColor(np.ascontiguousarray(base), cv2.COLOR_RGB2YCrCb)
    sy = cv2.LUT(ycc[..., 0], lut.reshape(1, 256)) * gain
    for _ in range(n):
        half = rng.standard_normal((h // 2, w // 2)).astype(np.float32); full = rng.standard_normal((h, w)).astype(np.float32)
        quarter = rng.standard_normal((h // 4, w // 4, 2)).astype(np.float32); eighth = rng.standard_normal((h // 8, w // 8)).astype(np.float32)
        ny = cv2.addWeighted(cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR), k2 * 0.8, full, 0.6, 0.0)
        ny = (ny * np.sqrt(1 - low) + np.sqrt(low) * k8 * cv2.resize(eighth, (w, h), interpolation=cv2.INTER_LINEAR)).astype(np.float32)
        nc = cv2.resize(quarter, (w, h), interpolation=cv2.INTER_LINEAR)
        noise = cv2.merge([ny * sy, nc[..., 0] * sy * k4 * spec["chroma"], nc[..., 1] * sy * k4 * spec.get("chroma_cb", spec["chroma"])])
        yield cv2.cvtColor(cv2.convertScaleAbs(ycc.astype(np.float32) + noise), cv2.COLOR_YCrCb2RGB)

def stats(base, cam, low, gain, n=60):
    h, w = base.shape[:2]
    with tempfile.TemporaryDirectory() as td:
        mp4 = Path(td) / "c.mp4"
        p = subprocess.Popen(["nice", "-n", "15", "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30", "-i", "-",
                              "-c:v", "libsvtav1", "-crf", "30", "-preset", "12", "-g", "2", "-pix_fmt", "yuv420p", str(mp4)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for fr in noise_frames(base, cam, low, gain, n): p.stdin.write(np.ascontiguousarray(fr).tobytes())
        p.stdin.close(); p.wait()
        cap = cv2.VideoCapture(str(mp4)); prev = None; k = 0; intra, lf, pred = [], [], []
        while True:
            ok, f = cap.read()
            if not ok: break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev is not None:
                dm = float(np.abs(g - prev).mean())
                if k % 2 == 0: intra.append(dm); lf.append(float(np.abs(cv2.resize(g, (20, 15), interpolation=cv2.INTER_AREA) - cv2.resize(prev, (20, 15), interpolation=cv2.INTER_AREA)).mean()))
                else: pred.append(dm)
            prev = g; k += 1
    return {"intra_med_diff": float(np.median(intra)), "intra_lowfreq_med": float(np.median(lf)), "pred_copies_pct": float(100 * (np.array(pred) < 0.02).mean())}

res = {"method": __doc__.split("\n\n")[0], "cameras": {}}
for cam, ep_train, ep_dev in (("overview", 0, 1), ("gripper", 0, 1)):
    tr = target[f"{cam}_train"]; dv = target[f"{cam}_dev"]; base, names = base_of(cam, ep_train)
    def cost(s):
        return abs(s["intra_med_diff"] - tr["intra_med_diff"]) / tr["intra_med_diff"] + abs(s["intra_lowfreq_med"] - tr["intra_lowfreq_med"]) / tr["intra_lowfreq_med"]
    rows = []
    for low in (0.0, 0.4, 0.6, 0.75, 0.9):
        for gain in (0.8, 1.0, 1.2, 1.4, 1.7):
            s = stats(base, cam, low, gain); rows.append((cost(s), low, gain, s))
    rows.sort(key=lambda r: r[0]); c, low, gain, s = rows[0]
    dbase, dnames = base_of(cam, ep_dev); sd = stats(dbase, cam, low, gain)
    kit_now = stats(dbase, cam, 0.0, 1.0)
    res["cameras"][cam] = {"low": low, "gain": gain, "train_fit": s, "train_target": {k: tr[k] for k in ("intra_med_diff", "intra_lowfreq_med", "pred_copies_pct")},
                           "dev_check": sd, "dev_target": {k: dv[k] for k in ("intra_med_diff", "intra_lowfreq_med", "pred_copies_pct")}, "dev_kit_r10": kit_now,
                           "train_base_frames": names, "dev_base_frames": dnames, "grid": [(round(r[0], 3), r[1], r[2], {k: round(v, 3) for k, v in r[3].items()}) for r in rows[:6]]}
    print(cam, "best low", low, "gain", gain, "TRAIN fit", {k: round(v, 3) for k, v in s.items()}, "target", {k: round(tr[k], 3) for k in ("intra_med_diff", "intra_lowfreq_med", "pred_copies_pct")}, flush=True)
    print(cam, "DEV check", {k: round(v, 3) for k, v in sd.items()}, "target", {k: round(dv[k], 3) for k in ("intra_med_diff", "intra_lowfreq_med", "pred_copies_pct")}, "| r10 kit on DEV", {k: round(v, 3) for k, v in kit_now.items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
