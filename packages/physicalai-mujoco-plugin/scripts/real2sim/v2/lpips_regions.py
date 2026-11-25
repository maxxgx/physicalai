"""r18 item 4 (D2): which regions make the gripper camera's lpips_cc. On a dev_replay output (its renders and seg; the
real frames_x5), the same colour fit, the same metric mask and the same spatial AlexNet LPIPS map as dev_replay.py;
the map's mean split by the seg label under each pixel: robot/fingers (10) and the wrist mount (12), buckets (1-4), desk
(5), the rest (0: the splat background). Per region: its share of the mask, its mean LPIPS, and its contribution to the
frame's lpips_cc (share x mean), medians over TRAIN / DEV frames. usage: lpips_regions.py DEV_REPLAY_DIR FRAMES_X5 OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np, cv2
HERE = Path(__file__).resolve().parent
D, F, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
src = (HERE / "dev_replay.py").read_text(); head = src[:src.index("k = np.ones((25, 25)")]
head = head[head.index("import lpips"):] if "import lpips" in head else head
g = {"__name__": "dr"}; exec(compile("import numpy as np, cv2, json, sys\nfrom pathlib import Path\n" + head[head.index("import lpips"):], "dr", "exec"), g) if False else None
import lpips, torch
dev = "mps" if torch.backends.mps.is_available() else "cpu"; net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
fit_src = src[src.index("def colour_fit"):src.index("def grad")]; ns = {"np": np, "cv2": cv2}; exec(fit_src, ns); colour_fit = ns["colour_fit"]
load = lambda p: cv2.imread(str(p))[:, :, ::-1].astype(np.float32) / 255.0
REG = {"fingers/mount": (10, 12), "buckets": (1, 2, 3, 4), "desk": (5,), "rest": (0,)}
meta = json.load(open(D / "metrics.json")); out = {}
for split, eps in (("train", {0, 2, 6, 7, 9}), ("dev", {1, 3, 10})):
    rows = []
    for r in meta["frames"]["gripper"]:
        if r["episode"] not in eps: continue
        stem = f"{r['episode']:03d}_{r['t']:08.4f}"
        real = load(F / r["frame"].replace("overview_", "gripper_")); ren = load(D / "renders" / f"{stem}_gripper.png")
        seg = cv2.imread(str(D / "renders" / f"{stem}_gripper_seg.png"), cv2.IMREAD_UNCHANGED); seg = seg[..., 0] if seg.ndim == 3 else seg
        e = ((seg >= 20) & (seg <= 99)).astype(np.uint8); m = cv2.dilate(e, np.ones((25, 25), np.uint8)) == 0
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        cc = colour_fit(ren, real, m)
        with torch.no_grad():
            tt = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(dev) * 2 - 1
            lmap = net(tt(real), tt(cc))[0, 0].cpu().numpy()
        row = {"lpips": float(lmap[m].mean())}
        for name, labs in REG.items():
            sel = m & np.isin(seg, labs); share = float(sel.sum() / m.sum())
            row[name] = (share, float(lmap[sel].mean()) if sel.any() else np.nan)
        rows.append(row)
    out[split] = {"frames": len(rows), "lpips_median": round(float(np.median([x["lpips"] for x in rows])), 3), "regions": {
        name: {"share": round(float(np.median([x[name][0] for x in rows])), 3), "mean_lpips": round(float(np.nanmedian([x[name][1] for x in rows])), 3),
               "contribution": round(float(np.nanmedian([x[name][0] * x[name][1] for x in rows if np.isfinite(x[name][1])] or [0])), 3)} for name in REG}}
    print(split, json.dumps(out[split]), flush=True)
json.dump(out, open(OUT, "w"), indent=1)
