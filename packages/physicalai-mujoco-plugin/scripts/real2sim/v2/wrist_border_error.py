"""r18 item 4 (D2): does the gripper camera's bucket misfit grow toward the image border (a lens model) or not? On allowed
wrist frames where a red/green/blue bucket shows (fit_wrist_world.py's pick and masks, the kit as is: r17 camera, map,
joint zeros, layouts), every real contour point of each bucket's largest component: its distance to the nearest sim
contour point (px), binned by its distance to the nearest image border (0-30, 30-80, 80-160, >= 160 px) and by radius
from the image centre (0-150, 150-250, 250-320, >= 320 px); median and p90 per bin, TRAIN and DEV.
usage: wrist_border_error.py STATES_PARQUET FRAMES_X5 OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
ARGS, OUT = sys.argv[1:3], sys.argv[3]
src = (HERE / "fit_wrist_world.py").read_text().split('res = {"method"')[0]
sys.argv = [str(HERE / "fit_wrist_world.py"), *ARGS, "/dev/null"]
W = {"__file__": str(HERE / "fit_wrist_world.py"), "__name__": "fww"}; exec(compile(src, "fww", "exec"), W)
KIT = json.load(open(os.environ["KIT"])); wc = KIT["cameras"]["wrist"]; W["set_camera"](wc["pos"], wc["quat"], wc["fovy"]); W["set_finger_offset"](np.zeros(2))
op = lambda r: W["G0_MAP"] + 2 * W["S_MAP"] * abs(r)
def pts(mask):
    cs = cv2.findContours(W["largest"](mask).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return np.vstack([c[:, 0] for c in cs]) if cs else np.zeros((0, 2), int)
BB = [(0, 30), (30, 80), (80, 160), (160, 10 ** 6)]; RB = [(0, 150), (150, 250), (250, 320), (320, 10 ** 6)]
res = {}
for split in ("train", "dev"):
    by_b, by_r = {b: [] for b in BB}, {r: [] for r in RB}
    for ep, f, state, rm in W["WF"][split]:
        W["ch"].set_positions(state); W["set_layout"](W["LAY"][str(ep)]["poses"])
        for a in W["G"]["fadr"]: W["d"].qpos[a] = op(state[-1]) / 2000.0
        sm = W["sim_world"]()
        for c in W["COL"]:
            if rm[c].sum() < 1500 or sm[c].sum() < 300: continue
            pr, ps = pts(rm[c]), pts(sm[c])
            if not len(pr) or not len(ps): continue
            dist = cKDTree(ps).query(pr)[0]
            border = np.minimum.reduce([pr[:, 0], 639 - pr[:, 0], pr[:, 1], 479 - pr[:, 1]])
            rad = np.hypot(pr[:, 0] - 320, pr[:, 1] - 240)
            for b in BB: by_b[b] += dist[(border >= b[0]) & (border < b[1])].tolist()
            for r in RB: by_r[r] += dist[(rad >= r[0]) & (rad < r[1])].tolist()
    q = lambda v: {"n": len(v), "median": round(float(np.median(v)), 1), "p90": round(float(np.percentile(v, 90)), 1)} if v else None
    res[split] = {"by_border_px": {f"{a}-{b}": q(v) for (a, b), v in by_b.items()}, "by_radius_px": {f"{a}-{b}": q(v) for (a, b), v in by_r.items()}}
    print(split, json.dumps(res[split]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
