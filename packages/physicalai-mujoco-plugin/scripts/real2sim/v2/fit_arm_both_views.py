"""r13 (D1 binding item, cross-checked): joint-zero offsets of joints 1-4 against BOTH robot cameras, allowed frames only.

fit_overview_arm.py found the real gripper ~21 px below the sim one in the overview, explained by elbow +6.7 / wrist
flex -3.2 deg; the same offsets made the wrist camera's buckets worse (wrist_bucket_edges.py --joff). Here both views
are scored together with the wrist camera at its r13 mount (kit; fitted to the fingers, independent of the joints):
  overview: the lime-shell mask distance (fit_overview_arm.py's cost, px)
  wrist:    the red/green/blue bucket masks' mean contour distance (wrist_bucket_edges.py's, px; each episode's own
            stored bucket fit)
Cost = overview + wrist (both px). Variants: none; the overview-only offsets; offsets fitted to the sum (TRAIN
0, 2, 6, 7, 9), each reported on TRAIN and DEV (1, 3, 10) per view, with the overview housing offset.
r14: R2S_FIT_JOINTS=6 fits all six joint zeros (wrist yaw and roll too), as corrections on top of whatever zeros the
kit applies (bucketsim_v2 sets the unit's zeros).  usage: fit_arm_both_views.py STATES_PARQUET FRAMES_X5 OUT_JSON"""
import re as _re; PROXY = _re.compile(r"_plan\d+$|_pc\d+_\d+$")   # r14 planning-only proxies
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco, pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b

STATES, F, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
st = pd.read_parquet(STATES); LATK = json.load(open(b.KIT))["camera_latency_s"]
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot)
comp = b.compositor(robot, show_objects=False); ch = robot._sim.channels[0]
LAY = json.load(open(Path(os.environ["REAL2SIM_ROOT"]).expanduser() / "align/bucket_layouts_episodes.json"))["episodes"]
seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering()
vopt = mujoco.MjvOption(); vopt.geomgroup[:] = 0; vopt.geomgroup[0] = 1
ARM = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right") and m.geom_contype[g] == 0 and not PROXY.search(m.geom(g).name or "")]
LIME = [g for g in ARM if m.geom_matid[g] >= 0 and m.material(m.geom_matid[g]).name == "lime"]
HOUSING = [g for g in LIME if m.body(m.geom_bodyid[g]).name == "gripper_end"]
COL = ("red", "green", "blue")
BMESH = {c: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"bucket_{c}").id and m.geom_contype[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name == "bucket"] for c in COL}
ALLB = [g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.body(m.geom_bodyid[g]).name.startswith("bucket_") and m.mesh(m.geom_dataid[g]).name == "bucket"]
NJ = int(os.environ.get("R2S_FIT_JOINTS", "4"))   # r14: 6 = also wrist yaw and roll
jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, NJ + 1)]

def render(cam, geoms):
    keep = m.geom_rgba.copy(); grp = m.geom_group.copy()
    m.geom_rgba[:, 3] = 0; m.geom_rgba[geoms, 3] = 1; m.geom_group[geoms] = 0
    try:
        mujoco.mj_forward(m, d); seg.update_scene(d, camera=cam, scene_option=vopt); s = seg.render()
    finally:
        m.geom_rgba[:] = keep; m.geom_group[:] = grp
    return s[..., 0], s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)

def hsv_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); h, s_, v = (hsv[..., i].astype(int) for i in range(3))
    mk = {"lime": (h >= 25) & (h <= 45) & (s_ > 90) & (v > 70), "red": ((h < 10) | (h > 168)) & (s_ > 110) & (v > 60),
          "green": (h >= 60) & (h <= 95) & (s_ > 70) & (v > 60), "blue": (h >= 100) & (h <= 130) & (s_ > 110) & (v > 50)}
    k = {"lime": 3, "red": 5, "green": 5, "blue": 5}
    return {c: cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k[c], k[c]), np.uint8)) > 0 for c, x in mk.items()}

def cdist(a_, b_):
    pts = lambda x: np.vstack([c[:, 0] for c in cv2.findContours(x.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]] or [np.zeros((0, 2))])
    ca, cb = pts(a_), pts(b_)
    inner = lambda p: (p[:, 0] > 2) & (p[:, 0] < 637) & (p[:, 1] > 2) & (p[:, 1] < 477)
    ca, cb = ca[inner(ca)], cb[inner(cb)]
    if not len(ca) or not len(cb): return np.nan
    return float((cKDTree(cb).query(ca)[0].mean() + cKDTree(ca).query(cb)[0].mean()) / 2)

def frames(cam_real, eps, per, test):
    out = []
    for ep in eps:
        rows = st[st.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
        cand = [f for f in sorted(F.glob(f"{cam_real}_e{ep:03d}_t*.png")) if ts[0] + 0.3 <= int(f.stem.split("_t")[1]) / 1000 <= ts[-1]]
        cand = [(f, hsv_masks(cv2.imread(str(f)))) for f in cand]; cand = [(f, mk) for f, mk in cand if test(mk)]
        for i in (np.linspace(0, len(cand) - 1, min(per, len(cand))).astype(int) if cand else []):
            out.append((ep, cand[i][0], cand[i][1], ts, S))
    return out

SPL = {"train": (0, 2, 6, 7, 9), "dev": (1, 3, 10)}
OV = {k: frames("overview", e, 8, lambda mk: mk["lime"].sum() > 800) for k, e in SPL.items()}
WR = {k: frames("gripper", e, 8, lambda mk: max(mk[c].sum() for c in COL) >= 1500) for k, e in SPL.items()}

def set_state(ep, f, ts, S, lat, joff):
    b.set_bucket_poses(robot, LAY[str(ep)]["poses"])
    t = int(f.stem.split("_t")[1]) / 1000; tl = float(np.clip(t - lat, ts[0], ts[-1]))
    ch.set_positions(np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])]))
    for k in range(len(joff)): d.qpos[jadr[k]] += np.radians(joff[k])

def score(split, joff):
    cam_o, cam_w = m.camera("overview").id, m.camera("wrist").id; oc, offs, wc = [], [], []
    for ep, f, mk, ts, S in OV[split]:
        set_state(ep, f, ts, S, LATK["overview"], joff); ids, isg = render(cam_o, ARM + ALLB)
        sl = isg & np.isin(ids, LIME); rl = mk["lime"]
        if sl.sum() < 50: continue
        dts = cv2.distanceTransform((~sl).astype(np.uint8), cv2.DIST_L2, 3); dtr = cv2.distanceTransform((~rl).astype(np.uint8), cv2.DIST_L2, 3)
        oc.append((np.minimum(dts[rl], 60).mean() + np.minimum(dtr[sl], 60).mean()) / 2)
        sh = isg & np.isin(ids, HOUSING)
        if sh.sum() > 30:
            ys, xs = np.nonzero(sh); c = np.array([xs.mean(), ys.mean()]); ry, rx = np.nonzero(rl); near = np.hypot(rx - c[0], ry - c[1]) < 60
            if near.sum() > 30: offs.append([rx[near].mean() - c[0], ry[near].mean() - c[1]])
    for ep, f, mk, ts, S in WR[split]:
        set_state(ep, f, ts, S, LATK["wrist"], joff); ids, isg = render(cam_w, ALLB + ARM)
        for c in COL:
            sm = isg & np.isin(ids, BMESH[c])
            if mk[c].sum() < 1500 and sm.sum() < 1500: continue
            v = cdist(mk[c], sm)
            if np.isfinite(v): wc.append(min(v, 80.0))
    return {"overview_px": round(float(np.mean(oc)), 2), "wrist_buckets_px": round(float(np.mean(wc)), 2),
            "overview_housing_offset_px": np.median(offs, 0).round(1).tolist() if offs else None, "n": [len(oc), len(wc)]}

res = {"source": "review/export/robotcam_allowed (frames_x5, states_allowed.parquet)", "frames": {k: {"overview": [f.name for _, f, *_ in OV[k]], "wrist": [f.name for _, f, *_ in WR[k]]} for k in SPL}, "variants": {}}
def rec(name, joff):
    res["variants"][name] = {"joint_offsets_deg": np.round(joff, 3).tolist(), "train": score("train", joff), "dev": score("dev", joff)}
    print(name, json.dumps(res["variants"][name]), flush=True); json.dump(res, open(OUT, "w"), indent=1)
rec("none", np.zeros(NJ))
if NJ == 4: rec("overview_only", np.array([-0.592, 0.33, 6.66, -3.208]))
def cost(x):
    s = score("train", x); return s["overview_px"] + s["wrist_buckets_px"]
fit = minimize(cost, np.zeros(NJ), method="Nelder-Mead", options={"xatol": 0.05, "fatol": 0.05, "maxfev": 40 * NJ,
               "initial_simplex": np.vstack([np.zeros(NJ)] + [np.eye(NJ)[i] * 2.0 for i in range(NJ)])})
rec("both_views", fit.x)
