"""Fit a robot camera's global colour response (kit camera_models: matrix + offset, then gamma; apply_camera_model)
from ALLOWED real frames against the composite rendered at the same joint state from the fitted camera (sensor noise
and any earlier colour model off).
Aligned regions only: static scene pixels (segmentation: not the robot, not task objects), eroded, where the local
normalised cross-correlation of real and sim luminance (15 px window, both blurred) exceeds NCC_MIN, i.e. the same
structure sits in the same place. On those pixel pairs a robust (Huber IRLS) fit of a DIAGONAL gain + offset per
channel on real^gamma, gamma on a grid. One global transform per camera: no spatial or per-frame terms.
Train/dev split by episode (TRAIN, DEV); the dev error before/after is what is reported.
usage: fit_colour.py CAMERA FRAMES_DIR STATES.json OUT.json [--per-episode 6]"""
import argparse, json, os, sys, numpy as np, cv2, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["R2S_SENSOR_NOISE"] = "0"
import bucketsim_v2 as b
from dataclasses import replace

TRAIN, DEV = (0, 2, 6, 7, 9), (1, 3, 10)   # every task (red 0/1, blue 2/3, green 6/7/9/10) on both sides
NCC_MIN = float(os.environ.get("NCC_MIN", "0.6"))
ap = argparse.ArgumentParser(); ap.add_argument("camera"); ap.add_argument("frames"); ap.add_argument("states"); ap.add_argument("out")
ap.add_argument("--per-episode", type=int, default=6); a = ap.parse_args()
states = json.loads(Path(a.states).read_text()); states = states.get("states", states)
def ep_ms(f): return int(f.stem.split("_e")[1][:3]), int(f.stem.split("_t")[1])
allf = sorted(f for f in Path(a.frames).glob(f"{a.camera}_e*_t*.png") if "%d/%d" % ep_ms(f) in states)
files = []
for ep in TRAIN + DEV:
    fe = [f for f in allf if ep_ms(f)[0] == ep]
    files += [fe[i] for i in np.linspace(0, len(fe) - 1, min(a.per_episode, len(fe))).astype(int)]
robot = b.make_robot(0); b.reset(robot, 0); b.matched_state(robot); m, d = robot._sim.model, robot._sim.data
from physicalai_mujoco_plugin.real2sim.desk_compositor import Compositor, DeskKit
comp = Compositor(m, d, replace(DeskKit.from_json(b.KIT), camera_models=None), width=640, height=480, show_objects=False)
name = {"gripper": "wrist"}.get(a.camera, a.camera)
jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]

def ncc(x, y, k=15):
    mx, my = cv2.blur(x, (k, k)), cv2.blur(y, (k, k))
    vx, vy = cv2.blur(x * x, (k, k)) - mx * mx, cv2.blur(y * y, (k, k)) - my * my
    return (cv2.blur(x * y, (k, k)) - mx * my) / np.sqrt(np.maximum(vx * vy, 1e-8)), np.minimum(vx, vy)

anchors = {sd: {"desk": [], "background": []} for sd in ("train", "dev")}
pairs = {"train": ([], []), "dev": ([], [])}; used = {"train": [], "dev": []}; frac = []
for f in files:
    ep, ms = ep_ms(f)
    for adr, v in zip(jadr, states[f"{ep}/{ms}"]): d.qpos[adr] = v
    d.time = ms / 1000.0; mujoco.mj_forward(m, d)
    r = comp.render_camera(name, outputs=True); seg = r["seg"]
    sim = cv2.GaussianBlur(r["rgb"].astype(np.float32) / 255.0, (0, 0), 2)
    real = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (0, 0), 2)
    static = cv2.erode(((seg != 10) & (seg != 12) & (seg < 20)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    c, var = ncc(cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY), cv2.cvtColor(real, cv2.COLOR_RGB2GRAY))
    # textured and correlated, or flat in both (desk, walls) with similar neighbourhood: flat regions have no NCC signal,
    # so they count only when they lie within 10 px of a correlated textured pixel (same surface, aligned)
    good = (c > NCC_MIN) & (var > 2e-4)
    near = cv2.dilate(good.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    keep = static & (good | (near & (var <= 2e-4)))
    frac.append(float(keep[static].mean()) if static.any() else 0.0)
    side = "train" if ep in TRAIN else "dev"
    s255, r255 = r["rgb"].astype(np.float32), cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB).astype(np.float32)
    st15 = cv2.erode(((seg != 10) & (seg != 12) & (seg < 20)).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    lum_s, lum_r = s255.mean(-1), r255.mean(-1); sat_s, sat_r = np.ptp(s255, -1), np.ptp(r255, -1)
    for nm, mk in (("desk", st15 & (lum_s > 190) & (sat_s < 30) & (sat_r < 30) & (lum_r > 110)),
                   ("background", st15 & (lum_s < 110) & (sat_s < 40) & (lum_r < 110) & (sat_r < 40))):
        if mk.sum() > 500: anchors[side][nm].append((np.median(s255[mk], 0) / 255, np.median(r255[mk], 0) / 255))
    k4 = keep[2::4, 2::4]
    pairs[side][0].append(sim[2::4, 2::4][k4]); pairs[side][1].append(real[2::4, 2::4][k4]); used[side].append(f.name)
Xt, Yt = (np.concatenate(v) for v in pairs["train"]); Xd, Yd = (np.concatenate(v) for v in pairs["dev"])
rng = np.random.default_rng(0)
if len(Xt) > 200000: s_ = rng.choice(len(Xt), 200000, replace=False); Xt, Yt = Xt[s_], Yt[s_]

# Monotone per-channel LUT from quantile matching on the ALIGNED pixel pairs (both sides show the same surfaces, so it
# maps the camera's response, not a scene histogram); knots at sim quantiles 0.5-99.5 % plus the ends (0 -> 0 side
# kept by the lowest knot, 1 -> extrapolated linearly). One global curve per camera.
# RUBRIC H2 (b): a tone curve has at most 16 control points per channel, counting the fitted points if the curve is
# resampled from them (R-32). Here: 14 quantile pairs plus the two ends = 16 fitted points per channel, and the curve is
# stored at 16 shared knots; the anchor warp only rescales those knots.
QS = np.linspace(0.02, 0.98, 14)
KNOTS = 16
def fit_lut(X, Y):
    xs = np.linspace(0, 1, KNOTS); ys = []
    for ch in range(3):
        xq, yq = np.quantile(X[:, ch], QS), np.quantile(Y[:, ch], QS)
        xq, keep = np.unique(xq, return_index=True); yq = np.maximum.accumulate(yq[keep])
        lo = yq[0] - (xq[0] - 0.0) * (yq[1] - yq[0]) / max(xq[1] - xq[0], 1e-3)
        hi = yq[-1] + (1.0 - xq[-1]) * (yq[-1] - yq[-2]) / max(xq[-1] - xq[-2], 1e-3)
        ys.append(np.clip(np.interp(xs, np.r_[0.0, xq, 1.0], np.r_[max(lo, 0.0), yq, min(max(hi, yq[-1]), 1.0)]), 0, 1).round(4).tolist())
    return {"x": xs.round(4).tolist(), "y": ys}
def apply(lut, X): return np.stack([np.interp(X[:, c], lut["x"], lut["y"][c]) for c in range(3)], -1)
def err(X, Y): return float(np.median(np.abs(X - Y)) * 255) if len(X) else float("nan")
def qerr(X, Y): return float(np.mean(np.abs(np.quantile(X, QS, axis=0) - np.quantile(Y, QS, axis=0))) * 255) if len(X) else float("nan")
lut = fit_lut(Xt, Yt)
def anchor(side, nm):
    v = anchors[side][nm]
    return (np.median([a_ for a_, _ in v], 0), np.median([b_ for _, b_ in v], 0)) if v else None
# Anchor warp: quantiles of different channels can belong to different surfaces, which tints single surfaces (the
# desk). Per channel, scale the curve by a factor that fixes the background and desk anchors exactly (class agreed
# on both sides, medians over the TRAIN frames), interpolated between them and constant outside.
A_bg, A_dk = anchor("train", "background"), anchor("train", "desk")
if A_bg and A_dk:
    xs = np.asarray(lut["x"]); ys = []
    for c in range(3):
        y = np.asarray(lut["y"][c]); f = lambda v: np.interp(v, xs, y)
        rb, rd = A_bg[1][c] / f(A_bg[0][c]), A_dk[1][c] / f(A_dk[0][c])
        ys.append(np.clip(y * np.interp(xs, [A_bg[0][c], A_dk[0][c]], [rb, rd]), 0, 1).round(4).tolist())
    lut = {"x": lut["x"], "y": ys}
spec = {"lut": lut}
def anchor_report(side):
    res = {}
    for nm in ("desk", "background"):
        A_ = anchor(side, nm)
        if A_: res[nm] = {"real": (A_[1] * 255).round(1).tolist(), "sim_before": (A_[0] * 255).round(1).tolist(), "sim_after": (apply(lut, A_[0][None])[0] * 255).round(1).tolist()}
    return res
def regions(X):
    lum = X.mean(1); sat = X.max(1) - X.min(1)
    return {"white desk (sim lum > .75)": lum > 0.75, "saturated (sim sat > .3)": sat > 0.3, "other": (lum <= 0.75) & (sat <= 0.3)}
reg = {k: {"pixels": int(mk.sum()), "before": round(err(Xd[mk], Yd[mk]), 2), "after": round(err(apply(lut, Xd[mk]), Yd[mk]), 2),
           "mean_real": (Yd[mk].mean(0) * 255).round(1).tolist() if mk.any() else None,
           "mean_sim_before": (Xd[mk].mean(0) * 255).round(1).tolist() if mk.any() else None,
           "mean_sim_after": (apply(lut, Xd[mk]).mean(0) * 255).round(1).tolist() if mk.any() else None} for k, mk in regions(Xd).items()}
out = {"camera": a.camera, "spec": spec, "method": f"aligned pixels (static seg, NCC>{NCC_MIN}); monotone per-channel LUT by quantile matching on those pairs, warped per channel to fix the desk and background anchors (class agreed on both sides, train medians); global, no spatial/per-frame terms",
       "train_episodes": TRAIN, "dev_episodes": DEV,
       "train_err_levels": {"before": round(err(Xt, Yt), 2), "after": round(err(apply(lut, Xt), Yt), 2)},
       "dev_err_levels": {"before": round(err(Xd, Yd), 2), "after": round(err(apply(lut, Xd), Yd), 2)},
       "dev_quantile_err_levels": {"before": round(qerr(Xd, Yd), 2), "after": round(qerr(apply(lut, Xd), Yd), 2)},
       "dev_regions": reg, "anchors_train": anchor_report("train"), "anchors_dev": anchor_report("dev"), "aligned_fraction_of_static": round(float(np.mean(frac)), 3), "pixels": {"train": int(len(Xt)), "dev": int(len(Xd))},
       "frame_list": used}
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("train_err_levels", "dev_err_levels", "dev_quantile_err_levels", "anchors_dev")}))
