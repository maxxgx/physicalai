"""Reviewer's frozen evaluation tool for the bucket real2sim project (see ../RUBRIC.md).

Run with the reviewer venv and REAL2SIM_ROOT set:  <venv>/bin/python -I tools/phone_eval.py <cmd> ...
(it has numpy, opencv, scikit-image, torch+lpips with the AlexNet weights cached, pycolmap 4).

Commands
  frames                         list held-out frames (evaluated) and guard frames (excluded, not evaluated)
  refs  --model D --pinhole P --out O
                                 undistort the frozen held-out references (heldout/phone/*.jpg, original
                                 1920x1080 SfM camera of model D) to the PINHOLE camera that model P gives
                                 the same image name; writes O/fNNNNN.png, O/fNNNNN_valid.png, O/poses.json
  metrics --renders R --refs O --out J [--scale 0.5] [--mask-dir M]
                                 psnr_raw, psnr_cc, ssim_cc, lpips_cc per frame + summary (RUBRIC D1)
  colour --renders R --refs O --out J
                                 CIEDE2000 of the four buckets, real vs render, from R/fNNNNN_seg.png (RUBRIC D2)
  overlay --renders R --refs O --out V
                                 draws R/fNNNNN_outline.png over the reference with a 10 px grid (RUBRIC D3)
  pairs --renders R --refs O --spec f00150,f00151:960:540,... --out V [--size 640] [--seed N]
                                 (a bare name centres the crop on the median of seg labels 1-4 and 10)
                                 blind real-vs-render crop pairs; key hidden in V/.key.json (RUBRIC D2)
  prep --src S --out O           robot-camera references: copy real frames S/* to O/fNNNNN.png with all-valid
                                 masks (names in sorted source order, O/map.json); then use metrics/overlay/pairs
  leakcheck --train T [--out J]  match every training image (a dir, or a text file of paths) to the 377
                                 phone frames by thumbnail; flags matches inside held-out windows + guard

Conventions
  Render file names = reference names with .png: fNNNNN.png (composite), fNNNNN_seg.png (uint8 labels:
  0 other, 1 bucket_black, 2 bucket_green, 3 bucket_red, 4 bucket_blue, 10 robot incl. gripper and wrist
  camera, 11 sim overview camera/pole, 20-99 manipulated objects), fNNNNN_outline.png (RGBA, alpha>0 on
  outline pixels; RGB colour per element: robot 255,255,0; buckets 255,0,255; desk edges 0,255,255; overview
  camera and pole 255,128,0; wrist camera housing 0,255,0; manipulated objects 255,255,255), cameras.json {name: {"width","height","K":[fx,fy,cx,cy],"cam_from_world":[[3x4]]}}.
  Image size of every render = the PINHOLE camera's size in model P.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent
HELDOUT = json.loads((REVIEW / "HELDOUT_PHONE.json").read_text())
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
FRAMES_JSON = ROOT / "colmap/frames.json"
ALL_IMAGES = ROOT / "colmap/images"
REF_DIR = REVIEW / "heldout/phone"
FPS = 15.0


def _windows():
    return [tuple(w) for w in HELDOUT["windows_s"]]


def window_of(t):
    for i, (a, b) in enumerate(_windows()):
        if a <= t <= b:
            return i
    return None


def in_guard(t):
    g = HELDOUT["guard_s"]
    return any(a - g <= t <= b + g for a, b in _windows())


def heldout_names():
    return sorted(p.stem for p in REF_DIR.glob("f*.jpg"))


def t_of(name):
    return int(name[1:6]) / FPS


# ---------------------------------------------------------------- frames
def cmd_frames(a):
    fr = json.loads(FRAMES_JSON.read_text())["frames"]
    ev = [x for x in fr if window_of(x["t"]) is not None]
    gd = [x for x in fr if window_of(x["t"]) is None and in_guard(x["t"])]
    print(f"{len(ev)} evaluated held-out frames, {len(gd)} guard frames, {len(fr)} total")
    for x in ev:
        print(f"heldout f{x['idx']:05d} t={x['t']:.3f} window={window_of(x['t'])}")
    for x in gd:
        print(f"guard   f{x['idx']:05d} t={x['t']:.3f}")
    print("KEY FRAMES (RUBRIC: sharpest frame of each half of each window):", ",".join(key_frames()))


def key_frames():
    fr = json.loads(FRAMES_JSON.read_text())["frames"]
    keys = []
    for a, b in _windows():
        mid = (a + b) / 2
        for lo, hi in ((a, mid), (mid, b)):
            c = [x for x in fr if lo <= x["t"] <= hi]
            keys.append(f"f{max(c, key=lambda x: x['sharp'])['idx']:05d}")
    return keys


# ---------------------------------------------------------------- refs
def _pose(img):
    p = img.cam_from_world
    p = p() if callable(p) else p
    return np.asarray(p.matrix(), dtype=np.float64)


def _registered(rec, img):
    hp = getattr(img, "has_pose", None)
    if hp is None:
        return True
    return bool(hp() if callable(hp) else hp)


def cmd_refs(a):
    import pycolmap

    D = pycolmap.Reconstruction(a.model)
    P = pycolmap.Reconstruction(a.pinhole)
    dn = {im.name: im for im in D.images.values()}
    pn = {im.name: im for im in P.images.values()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    poses, missing = {}, []
    for name in heldout_names():
        fn = name + ".jpg"
        if fn not in dn or fn not in pn or not _registered(D, dn[fn]) or not _registered(P, pn[fn]):
            missing.append(name)
            continue
        cd, cp = D.cameras[dn[fn].camera_id], P.cameras[pn[fn].camera_id]
        if cp.model.name != "PINHOLE":
            sys.exit(f"{a.pinhole}: camera of {fn} is {cp.model.name}, expected PINHOLE")
        src = cv2.imread(str(REF_DIR / fn), cv2.IMREAD_COLOR)
        if (src.shape[1], src.shape[0]) != (cd.width, cd.height):
            sys.exit(f"reference {fn} is {src.shape[1]}x{src.shape[0]}, model D camera {cd.width}x{cd.height}")
        W, H = cp.width, cp.height
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        pix = np.stack([u.ravel(), v.ravel()], 1)
        nrm = cp.cam_from_img(pix)
        dpx = cd.img_from_cam(np.hstack([nrm, np.ones((len(nrm), 1))]), check_cheirality=False)
        mx = dpx[:, 0].reshape(H, W).astype(np.float32)
        my = dpx[:, 1].reshape(H, W).astype(np.float32)
        img = cv2.remap(src, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = ((mx >= 0.5) & (my >= 0.5) & (mx <= cd.width - 1.5) & (my <= cd.height - 1.5)).astype(np.uint8) * 255
        cv2.imwrite(str(out / f"{name}.png"), img)
        cv2.imwrite(str(out / f"{name}_valid.png"), valid)
        Td, Tp = _pose(dn[fn]), _pose(pn[fn])
        poses[name] = {
            "cam_from_world": Tp.tolist(),
            "K": [float(x) for x in cp.params[:4]],
            "width": W,
            "height": H,
            "pose_diff_D_vs_P": float(np.abs(Td - Tp).max()),
        }
    (out / "poses.json").write_text(json.dumps({"poses": poses, "unregistered": missing}, indent=1))
    n = len(heldout_names())
    print(f"{len(poses)}/{n} held-out frames registered in both models ({100 * len(poses) / n:.1f} %); unregistered: {missing}")
    dmax = max((p["pose_diff_D_vs_P"] for p in poses.values()), default=0)
    print(f"max |pose(D) - pose(P)| = {dmax:.2e} (must be ~0: the pinhole model is the same SfM)")


# ---------------------------------------------------------------- metrics
def load_rgb(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return im[:, :, ::-1].astype(np.float32) / 255.0


def colour_fit(render, real, m):
    out = render.copy()
    for c in range(3):
        A = np.stack([render[..., c][m], np.ones(int(m.sum()))], 1)
        g, b = np.linalg.lstsq(A, real[..., c][m], rcond=None)[0]
        out[..., c] = np.clip(g * render[..., c] + b, 0, 1)
    return out


def psnr(a, b, m):
    return float(-10 * np.log10(np.mean((a[m] - b[m]) ** 2) + 1e-12))


def _pairs_of(a):
    refs = Path(a.refs)
    names = sorted(p.stem for p in refs.glob("f?????.png"))
    return refs, Path(a.renders), names


def _check_cameras(renders, refs):
    cj, pj = renders / "cameras.json", refs / "poses.json"
    if not cj.exists():
        return "renders/cameras.json missing"
    cams, poses = json.loads(cj.read_text()), json.loads(pj.read_text())["poses"]
    worst = 0.0
    for n, p in poses.items():
        c = cams.get(n)
        if c is None:
            continue
        d = max(np.abs(np.asarray(c["cam_from_world"]) - np.asarray(p["cam_from_world"])).max(),
                np.abs(np.asarray(c["K"]) - np.asarray(p["K"])).max() / max(p["K"][0], 1))
        worst = max(worst, float(d))
    return f"max deviation of delivered cameras.json from the SfM model: {worst:.2e} (rubric: <= 1e-4)"


def cmd_metrics(a):
    import lpips
    import torch
    from skimage.metrics import structural_similarity

    refs, renders, names = _pairs_of(a)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
    rows, missing = {}, []
    for n in names:
        rp = renders / f"{n}.png"
        if not rp.exists():
            missing.append(n)
            continue
        real, ren = load_rgb(refs / f"{n}.png"), load_rgb(rp)
        if real.shape != ren.shape:
            sys.exit(f"{n}: render {ren.shape} != reference {real.shape}")
        m = cv2.imread(str(refs / f"{n}_valid.png"), cv2.IMREAD_GRAYSCALE) > 127
        if a.mask_dir:
            mm = cv2.imread(str(Path(a.mask_dir) / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
            m &= mm > 127
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if a.scale != 1.0:
            sz = (round(real.shape[1] * a.scale), round(real.shape[0] * a.scale))
            real = cv2.resize(real, sz, interpolation=cv2.INTER_AREA)
            ren = cv2.resize(ren, sz, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
        cc = colour_fit(ren, real, m)
        _, smap = structural_similarity(real, cc, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                        sigma=1.5, use_sample_covariance=False, full=True)
        with torch.no_grad():
            t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].to(dev) * 2 - 1  # noqa: E731
            lmap = net(t(real), t(cc))[0, 0].cpu().numpy()
        rows[n] = {
            "t": t_of(n),
            "window": window_of(t_of(n)),
            "psnr_raw": psnr(ren, real, m),
            "psnr_cc": psnr(cc, real, m),
            "ssim_cc": float(smap.mean(axis=2)[m].mean()),
            "lpips_cc": float(lmap[m].mean()),
        }
        print(f"{n} t={t_of(n):6.2f} psnr_raw {rows[n]['psnr_raw']:.2f} psnr_cc {rows[n]['psnr_cc']:.2f} "
              f"ssim_cc {rows[n]['ssim_cc']:.3f} lpips_cc {rows[n]['lpips_cc']:.3f}", flush=True)

    def summ(sel):
        out = {}
        for k, worst in (("psnr_raw", min), ("psnr_cc", min), ("ssim_cc", min), ("lpips_cc", max)):
            v = [r[k] for r in sel]
            out[k] = {"median": float(np.median(v)), "worst": float(worst(v))} if v else None
        return out

    allr = list(rows.values())
    res = {
        "scale": a.scale,
        "n_refs": len(names),
        "n_scored": len(rows),
        "missing_renders": missing,
        "summary": summ(allr),
        "per_window": {str(i): summ([r for r in allr if r["window"] == i]) for i in range(len(_windows()))},
        "frames": rows,
    }
    if (refs / "poses.json").exists():
        res["camera_check"] = _check_cameras(renders, refs)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_refs", "n_scored", "missing_renders", "summary")}, indent=1))
    if "camera_check" in res:
        print(res["camera_check"])


# ---------------------------------------------------------------- colour
BUCKETS = {1: "black", 2: "green", 3: "red", 4: "blue"}


def cmd_colour(a):
    from skimage.color import deltaE_ciede2000, rgb2lab

    refs, renders, names = _pairs_of(a)
    per = {c: [] for c in BUCKETS.values()}
    for n in names:
        sp = renders / f"{n}_seg.png"
        if not (renders / f"{n}.png").exists() or not sp.exists():
            continue
        real, ren = load_rgb(refs / f"{n}.png"), load_rgb(renders / f"{n}.png")
        seg = cv2.imread(str(sp), cv2.IMREAD_UNCHANGED)
        if seg.ndim == 3:
            seg = seg[..., 0]
        valid = cv2.imread(str(refs / f"{n}_valid.png"), cv2.IMREAD_GRAYSCALE) > 127
        cc = colour_fit(ren, real, valid)
        for lab_id, col in BUCKETS.items():
            m = cv2.erode((seg == lab_id).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            m &= valid
            if m.sum() < 400:
                continue
            lr = rgb2lab(real[m].reshape(-1, 1, 3)).reshape(-1, 3).mean(0)
            ls = rgb2lab(cc[m].reshape(-1, 1, 3)).reshape(-1, 3).mean(0)
            per[col].append({"frame": n, "px": int(m.sum()), "dE": float(deltaE_ciede2000(lr, ls))})
    res = {c: {"n_frames": len(v), "mean_dE": float(np.mean([x["dE"] for x in v])) if v else None, "frames": v}
           for c, v in per.items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for c, v in res.items():
        print(f"{c:6s} frames {v['n_frames']:3d} mean dE2000 {v['mean_dE']}")


# ---------------------------------------------------------------- overlay
def cmd_overlay(a):
    refs, renders, names = _pairs_of(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        op = renders / f"{n}_outline.png"
        if not op.exists():
            print(f"{n}: no outline")
            continue
        real = cv2.imread(str(refs / f"{n}.png"), cv2.IMREAD_COLOR)
        ol = cv2.imread(str(op), cv2.IMREAD_UNCHANGED)
        if ol.shape[:2] != real.shape[:2] or ol.shape[2] != 4:
            sys.exit(f"{n}: outline must be RGBA of the reference size")
        vis = real.copy()
        vis[::10, :] = (vis[::10, :] * 0.75).astype(np.uint8)
        vis[:, ::10] = (vis[:, ::10] * 0.75).astype(np.uint8)
        vis[::50, :] = (vis[::50, :] * 0.5).astype(np.uint8)
        vis[:, ::50] = (vis[:, ::50] * 0.5).astype(np.uint8)
        m = ol[..., 3] > 0
        vis[m] = ol[..., :3][m]
        cv2.imwrite(str(out / f"{n}_overlay.png"), vis)
        H, W = vis.shape[:2]
        for q, (y, x) in enumerate(((0, 0), (0, W // 2), (H // 2, 0), (H // 2, W // 2))):
            crop = vis[y:y + H // 2, x:x + W // 2]
            if (ol[y:y + H // 2, x:x + W // 2, 3] > 0).any():
                cv2.imwrite(str(out / f"{n}_q{q}_x2.png"), cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    print(f"overlays in {out} (grid: thin line every 10 px, dark line every 50 px, in reference pixels; "
          f"*_qK_x2.png = quadrant K (0 TL, 1 TR, 2 BL, 3 BR) magnified 2x, written only where outlines exist)")


# ---------------------------------------------------------------- pairs
def cmd_pairs(a):
    refs, renders, _ = _pairs_of(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    key, s = {}, a.size // 2
    for i, item in enumerate(a.spec.split(",")):
        if ":" in item:
            n, cx, cy = item.split(":")
            cx, cy = int(cx), int(cy)
        else:  # centre on the robot and the buckets (seg labels 1-4 and 10) of the render
            n = item
            seg = cv2.imread(str(renders / f"{n}_seg.png"), cv2.IMREAD_UNCHANGED)
            seg = seg[..., 0] if seg.ndim == 3 else seg
            ys, xs = np.nonzero(np.isin(seg, [1, 2, 3, 4, 10]))
            if len(xs) == 0:
                sys.exit(f"{n}: no robot/bucket pixels in the seg; give an explicit centre")
            cx, cy = int(np.median(xs)), int(np.median(ys))
        real = cv2.imread(str(refs / f"{n}.png"), cv2.IMREAD_COLOR)
        ren = cv2.imread(str(renders / f"{n}.png"), cv2.IMREAD_COLOR)
        H, W = real.shape[:2]
        x0, y0 = min(max(cx - s, 0), W - a.size), min(max(cy - s, 0), H - a.size)
        cr, cs = real[y0:y0 + a.size, x0:x0 + a.size], ren[y0:y0 + a.size, x0:x0 + a.size]
        left_real = rng.random() < 0.5
        pair = np.hstack([cr if left_real else cs, np.full((a.size, 16, 3), 255, np.uint8), cs if left_real else cr])
        cv2.imwrite(str(out / f"pair_{i:02d}.png"), pair)
        key[f"pair_{i:02d}"] = {"frame": n, "crop": [x0, y0, a.size], "real": "left" if left_real else "right"}
    (out / ".key.json").write_text(json.dumps(key, indent=1))
    (out / "answers.txt").write_text("".join(f"pair_{i:02d} ?\n" for i in range(len(key))))
    print(f"{len(key)} pairs in {out}. Fill answers.txt with left/right = which side is REAL, record "
          f"`shasum -a 256 answers.txt` in the report, and only then open .key.json.")


# ---------------------------------------------------------------- prep (robot-camera frames)
def cmd_prep(a):
    src = sorted(p for p in Path(a.src).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mp = {}
    for i, p in enumerate(src):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        n = f"f{i:05d}"
        cv2.imwrite(str(out / f"{n}.png"), im)
        cv2.imwrite(str(out / f"{n}_valid.png"), np.full(im.shape[:2], 255, np.uint8))
        mp[n] = p.name
    (out / "map.json").write_text(json.dumps(mp, indent=1))
    print(f"{len(mp)} frames -> {out} (names f00000.. in sorted order of the source names; see map.json)")


# ---------------------------------------------------------------- leakcheck
def _thumb(p):
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape
    # undistorted/cropped variants keep the centre: compare the central 80 % only
    im = im[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)]
    t = cv2.resize(im, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (t - t.mean()) / (t.std() + 1e-6)


def cmd_leakcheck(a):
    tp = Path(a.train)
    if tp.is_dir():
        files = sorted(p for p in tp.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    else:
        files = [Path(os.path.expanduser(line.strip())) for line in tp.read_text().splitlines() if line.strip()]
    fr = json.loads(FRAMES_JSON.read_text())["frames"]
    bank = [(x, _thumb(ALL_IMAGES / f"f{x['idx']:05d}.jpg")) for x in fr]
    flags, rows = [], []
    for f in files:
        t = _thumb(f)
        if t is None:
            rows.append({"file": str(f), "error": "unreadable"})
            flags.append(str(f))
            continue
        scores = [(float((t * b).mean()), x) for x, b in bank]
        scores.sort(key=lambda s: -s[0])
        (c1, x1), (c2, _) = scores[0], scores[1]
        bad = in_guard(x1["t"])
        stem_t = None
        if f.stem.startswith("f") and f.stem[1:6].isdigit():
            stem_t = int(f.stem[1:6]) / FPS
            bad |= in_guard(stem_t)
        rows.append({"file": str(f), "nearest": f"f{x1['idx']:05d}", "t": x1["t"], "corr": c1, "corr2": c2,
                     "name_t": stem_t, "flag": bool(bad)})
        if bad:
            flags.append(str(f))
    res = {"n_files": len(files), "n_flagged": len(flags), "flagged": flags, "rows": rows}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    lowc = [r for r in rows if "corr" in r and r["corr"] < 0.8]
    print(f"{len(files)} training images, {len(flags)} flagged (inside a held-out window + guard, by name or by "
          f"nearest phone frame); {len(lowc)} with weak matches (corr < 0.8, inspect: they may be non-phone images)")
    for f in flags[:50]:
        print("FLAG", f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("frames")
    p = sp.add_parser("refs")
    p.add_argument("--model", required=True)
    p.add_argument("--pinhole", required=True)
    p.add_argument("--out", required=True)
    for c in ("metrics", "colour", "overlay", "pairs"):
        p = sp.add_parser(c)
        p.add_argument("--renders", required=True)
        p.add_argument("--refs", required=True)
        p.add_argument("--out", required=True)
        if c == "metrics":
            p.add_argument("--scale", type=float, default=0.5)
            p.add_argument("--mask-dir", default=None)
        if c == "pairs":
            p.add_argument("--spec", required=True)
            p.add_argument("--size", type=int, default=640)
            p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("prep")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("leakcheck")
    p.add_argument("--train", required=True)
    p.add_argument("--out", default=None)
    a = ap.parse_args()
    {"frames": cmd_frames, "refs": cmd_refs, "prep": cmd_prep, "metrics": cmd_metrics, "colour": cmd_colour, "overlay": cmd_overlay,
     "pairs": cmd_pairs, "leakcheck": cmd_leakcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
