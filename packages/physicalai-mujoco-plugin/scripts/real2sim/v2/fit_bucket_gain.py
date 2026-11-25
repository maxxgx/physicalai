"""D1/D2 (r12): per-bucket colour gains for the sim-drawn buckets (their real-look material colours), after the
robot-camera headlight: on allowed TRAIN replay frames (episodes 0, 2, 6, 7, 9; segment layout; both cameras), per
bucket the median over frames of real / sim median colour per channel (sRGB levels, as streamed); DEV (1, 3, 10)
reported before and after applying it to the model's bucket materials. usage: fit_bucket_gain.py STATES FRAMES OUT"""
import json, sys
from pathlib import Path
import numpy as np, cv2, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
src = (Path(__file__).resolve().parent / "fit_headlight.py").read_text().split("res = {")[0]
ns = {"__file__": str(Path(__file__).resolve().parent / "fit_headlight.py"), "__name__": "fh"}; exec(compile(src, "fit_headlight.py", "exec"), ns)
b, m, d, comp, ch = ns["b"], ns["m"], ns["d"], ns["comp"], ns["ch"]; OUT = sys.argv[3]
from physicalai_mujoco_plugin.real2sim.desk_compositor import LABEL_BUCKET
LAB = {v: k for k, v in LABEL_BUCKET.items()}
mats = {c: sorted({int(m.geom_matid[g]) for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"bucket_{c}").id and m.geom_matid[g] >= 0 and m.geom_group[g] != 3}) for c in LAB.values()}
def medians(items):
    out = {c: [] for c in LAB.values()}
    for ep, cam, f in items:
        _, poses = b.episode_layout(ep); b.set_bucket_poses(ns["robot"], poses)
        t = int(f.stem.split("_t")[1]) / 1000; ch.set_positions(ns["state_at"](ep, t)); d.time = t; mujoco.mj_forward(m, d)
        r = comp.render_camera(cam, outputs=True); real = cv2.imread(str(f))[:, :, ::-1].astype(np.float32); sim = r["rgb"].astype(np.float32)
        for lab, c in LAB.items():
            mk = cv2.erode((r["seg"] == lab).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            if mk.sum() >= 400: out[c].append((np.median(sim[mk], axis=0), np.median(real[mk], axis=0)))
    return out
def err(med): return float(np.mean([np.abs(s - r).mean() for v in med.values() for s, r in v]))
tr = medians(ns["TRAIN"]); dv0 = medians(ns["DEV"])
gain = {c: np.clip(np.median([r / np.maximum(s, 1) for s, r in v], axis=0), 0.5, 2.0).round(3).tolist() for c, v in tr.items() if v}
print("gains (real/sim, sRGB)", gain, "| mats", mats, flush=True)
# apply as a material colour change (linear light ~ gain^2.2 on the base colour), then re-measure DEV
for c, g in gain.items():
    for k in mats[c]:
        m.mat_rgba[k, :3] = np.clip(m.mat_rgba[k, :3] * np.power(np.array(g), 2.2), 0, 1)
dv1 = medians(ns["DEV"])
res = {"source": "review/export/robotcam_allowed", "train_frames": [f.name for _, _, f in ns["TRAIN"]], "dev_frames": [f.name for _, _, f in ns["DEV"]],
       "gain_srgb": gain, "material_factor_linear": {c: np.power(np.array(g), 2.2).round(3).tolist() for c, g in gain.items()},
       "dev_err_before": round(err(dv0), 2), "dev_err_after": round(err(dv1), 2), "train_err_before": round(err(tr), 2)}
print({k: v for k, v in res.items() if "frames" not in k}); json.dump(res, open(OUT, "w"), indent=1)
