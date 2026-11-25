"""r13: what a wrist camera change does to the buckets in the wrist view (export only). On allowed wrist frames where
the red, green or blue bucket shows (>= 1500 px of its colour), the real bucket masks by colour (the layout fitter's
thresholds, whole frame) against the sim buckets' segmentation, each episode's buckets at its own stored fit
(align/bucket_layouts_episodes.json), the arm at the frame's state (t - latency). Per camera variant: IoU per bucket
and the symmetric mean contour distance (px), TRAIN (0, 2, 6, 7, 9) and DEV (1, 3, 10).
usage: wrist_bucket_edges.py STATES_PARQUET FRAMES_X5 FIT_JSON OUT_JSON [--joff a,b,c,d]   (FIT_JSON: fit_wrist_joint.py's
output; --joff: joint-zero offsets of joints 1-4, deg, added to the recorded state, as fit_overview_arm.py fits them)"""
import re as _re; PROXY = _re.compile(r"_plan\d+$|_pc\d+_\d+$")   # r14 planning-only proxies
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco, pandas as pd
HERE = Path(__file__).resolve().parent
FIT, OUT = sys.argv[3], sys.argv[4]
JOFF = np.array([float(v) for v in sys.argv[sys.argv.index("--joff") + 1].split(",")]) if "--joff" in sys.argv else np.zeros(4)
sys.argv = sys.argv[:3] + ["/dev/null"]
src = (HERE / "fit_finger_mount.py").read_text().split("p0 = np.zeros(6)")[0]
ns = {"__file__": str(HERE / "fit_finger_mount.py"), "__name__": "ffm"}; exec(compile(src, "fit_finger_mount.py", "exec"), ns)
sys.path.insert(0, str(HERE)); import bucketsim_v2 as b
m, d, cam, st, F, lat = ns["m"], ns["d"], ns["cam"], ns["st"], ns["F"], ns["lat"]
fovy0 = float(m.cam_fovy[cam]); fit = json.load(open(FIT))["variants"]
LAY = json.load(open(Path(os.environ["REAL2SIM_ROOT"]).expanduser() / "align/bucket_layouts_episodes.json"))["episodes"]
COL = ("red", "green", "blue")
seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering()
vopt = mujoco.MjvOption(); vopt.geomgroup[:] = 0; vopt.geomgroup[0] = 1
bucket_mesh = {}
for c in COL:
    bid = m.body(f"bucket_{c}").id
    bucket_mesh[c] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] == 0 and m.mesh(m.geom_dataid[g]).name == "bucket"]
ARM = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("link5", "link6", "gripper_end", "gripper_left", "gripper_right") and m.geom_contype[g] == 0 and not PROXY.search(m.geom(g).name or "")]

def sim_masks():
    geoms = [g for c in COL for g in bucket_mesh[c]] + ARM
    keep = m.geom_rgba.copy(); grp = m.geom_group.copy()
    m.geom_rgba[:, 3] = 0; m.geom_rgba[geoms, 3] = 1; m.geom_group[geoms] = 0
    try:
        mujoco.mj_forward(m, d); seg.update_scene(d, camera=cam, scene_option=vopt); s = seg.render()
    finally:
        m.geom_rgba[:] = keep; m.geom_group[:] = grp
    isg = s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return {c: isg & np.isin(s[..., 0], bucket_mesh[c]) for c in COL}

def real_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); h, s_, v = (hsv[..., i].astype(int) for i in range(3))
    mk = {"red": ((h < 10) | (h > 168)) & (s_ > 110) & (v > 60), "green": (h >= 60) & (h <= 95) & (s_ > 70) & (v > 60), "blue": (h >= 100) & (h <= 130) & (s_ > 110) & (v > 50)}
    return {c: cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0 for c, x in mk.items()}

def contour_dist(a_, b_):
    ca = np.vstack([c[:, 0] for c in cv2.findContours(a_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]] or [np.zeros((0, 2))])
    cb = np.vstack([c[:, 0] for c in cv2.findContours(b_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]] or [np.zeros((0, 2))])
    if not len(ca) or not len(cb): return np.nan
    from scipy.spatial import cKDTree
    inner = lambda p: (p[:, 0] > 2) & (p[:, 0] < 637) & (p[:, 1] > 2) & (p[:, 1] < 477)   # frame borders are not edges
    ca, cb = ca[inner(ca)], cb[inner(cb)]
    if not len(ca) or not len(cb): return np.nan
    return float((cKDTree(cb).query(ca)[0].mean() + cKDTree(ca).query(cb)[0].mean()) / 2)

def frames(eps, per=8):
    out = []
    for ep in eps:
        rows = st[st.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
        cand = []
        for f in sorted(F.glob(f"gripper_e{ep:03d}_t*.png")):
            t = int(f.stem.split("_t")[1]) / 1000; tl = t - lat
            if not ts[0] <= tl <= ts[-1]: continue
            rm = real_masks(cv2.imread(str(f)))
            if max(x.sum() for x in rm.values()) >= 1500: cand.append((ep, f, np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])])))
        out += [cand[i] for i in np.linspace(0, len(cand) - 1, min(per, len(cand))).astype(int)] if cand else []
    return out

jadr4 = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 5)]
sets = {"train": frames((0, 2, 6, 7, 9)), "dev": frames((1, 3, 10))}
reals = {f.name: real_masks(cv2.imread(str(f))) for s in sets.values() for _, f, _ in s}
variants = {"kit_camera": (np.zeros(6), 0.0), "B_finger_fit_camera": (np.array(fit["B_cad_fitted_camera"]["x"][:6]), fit["B_cad_fitted_camera"]["x"][6]),
            "C_finger_fit_camera": (np.array(fit["C_symmetric_shape_fitted_camera"]["x"][:6]), fit["C_symmetric_shape_fitted_camera"]["x"][6])}
res = {"frames": {k: [f.name for _, f, _ in v] for k, v in sets.items()}, "joint_offsets_deg": JOFF.tolist(), "variants": {}}
for name, (mount, dfov) in variants.items():
    ns["set_mount"](mount); m.cam_fovy[cam] = fovy0 + dfov; res["variants"][name] = {}
    for split, fr in sets.items():
        iou, dist = {c: [] for c in COL}, {c: [] for c in COL}
        for ep, f, state in fr:
            b.set_bucket_poses(ns["robot"], LAY[str(ep)]["poses"]); ns["ch"].set_positions(state)
            for j_, off in enumerate(JOFF): d.qpos[jadr4[j_]] += np.radians(off)
            sm = sim_masks(); rm = reals[f.name]
            for c in COL:
                if rm[c].sum() < 1500 and sm[c].sum() < 1500: continue
                u = (rm[c] | sm[c]).sum(); iou[c].append((rm[c] & sm[c]).sum() / u); dist[c].append(contour_dist(rm[c], sm[c]))
        res["variants"][name][split] = {c: {"n": len(iou[c]), "iou_median": round(float(np.median(iou[c])), 3) if iou[c] else None,
                                            "edge_px_median": round(float(np.nanmedian(dist[c])), 1) if dist[c] else None} for c in COL}
    print(name, json.dumps(res["variants"][name]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
