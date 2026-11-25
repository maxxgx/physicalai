"""r13 (D1 binding item): where the real arm is in the overview against the sim arm, on allowed frames (export only).

The real arm's lime shells (link2, link3, link5, the gripper housing) by colour (HSV hue 25-45, saturation > 90,
value > 70), against the sim's lime-material geoms in the segmentation (the arm at the recorded state of t - latency,
the kit's overview camera; the rest of the arm and the buckets occlude as drawn). Cost: the symmetric mean distance
between the two masks (distance transform, truncated at 60 px). Reported per variant on TRAIN (0, 2, 6, 7, 9) and DEV
(1, 3, 10): the cost, and the gripper housing's offset (centroid of the lime pixels within 60 px of the sim housing,
real - sim, px; +y = real lower). Variants:
  K  the kit (no change)
  J  joint-zero offsets of joints 1-4 (deg) fitted
  JB as J plus the arm base's height and x, y (mm)
  L  the latency scanned (-0.2..0.1 s) at the kit's geometry
usage: fit_overview_arm.py STATES_PARQUET FRAMES_X5 OUT_JSON [--per 8]"""
import re as _re; PROXY = _re.compile(r"_plan\d+$|_pc\d+_\d+$")   # r14 planning-only proxies
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b

STATES, F, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]; PER = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 8
st = pd.read_parquet(STATES); lat0 = float(json.load(open(b.KIT))["camera_latency_s"]["overview"])
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot)
comp = b.compositor(robot, show_objects=False); ch = robot._sim.channels[0]; cam = m.camera("overview").id
LAY = json.load(open(Path(os.environ["REAL2SIM_ROOT"]).expanduser() / "align/bucket_layouts_episodes.json"))["episodes"]
seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering()
vopt = mujoco.MjvOption(); vopt.geomgroup[:] = 0; vopt.geomgroup[0] = 1
ARM_BODIES = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right")
ARM = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ARM_BODIES and m.geom_contype[g] == 0 and not PROXY.search(m.geom(g).name or "")]
LIME = [g for g in ARM if m.geom_matid[g] >= 0 and m.material(m.geom_matid[g]).name == "lime"]
HOUSING = [g for g in LIME if m.body(m.geom_bodyid[g]).name == "gripper_end"]
BUCKETS = [g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.body(m.geom_bodyid[g]).name.startswith("bucket_")]
jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]
base = m.body("base_link").id; base_pos0 = m.body_pos[base].copy()

def sim_lime():
    geoms = ARM + BUCKETS; keep = m.geom_rgba.copy(); grp = m.geom_group.copy()
    m.geom_rgba[:, 3] = 0; m.geom_rgba[geoms, 3] = 1; m.geom_group[geoms] = 0
    try:
        mujoco.mj_forward(m, d); seg.update_scene(d, camera=cam, scene_option=vopt); s = seg.render()
    finally:
        m.geom_rgba[:] = keep; m.geom_group[:] = grp
    isg = s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return isg & np.isin(s[..., 0], LIME), isg & np.isin(s[..., 0], HOUSING)

def real_lime(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mk = (hsv[..., 0] >= 25) & (hsv[..., 0] <= 45) & (hsv[..., 1] > 90) & (hsv[..., 2] > 70)
    return cv2.morphologyEx(mk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

def frames(eps):
    out = []
    for ep in eps:
        rows = st[st.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
        cand = [f for f in sorted(F.glob(f"overview_e{ep:03d}_t*.png"))]
        cand = [f for f in cand if ts[0] + 0.3 <= int(f.stem.split("_t")[1]) / 1000 <= ts[-1]]
        cand = [f for f in cand if real_lime(cv2.imread(str(f))).sum() > 800]
        for i in np.linspace(0, len(cand) - 1, min(PER, len(cand))).astype(int):
            f = cand[i]; out.append((ep, f, ts, S))
    return out

sets = {"train": frames((0, 2, 6, 7, 9)), "dev": frames((1, 3, 10))}
real = {f.name: real_lime(cv2.imread(str(f))) for s in sets.values() for _, f, _, _ in s}
dt_real = {k: cv2.distanceTransform((~v).astype(np.uint8), cv2.DIST_L2, 3) for k, v in real.items()}

def evaluate(fr, joff=np.zeros(4), dbase=np.zeros(3), lat=lat0):
    m.body_pos[base] = base_pos0 + dbase / 1000.0; costs, offs = [], []
    for ep, f, ts, S in fr:
        b.set_bucket_poses(robot, LAY[str(ep)]["poses"])
        t = int(f.stem.split("_t")[1]) / 1000; tl = float(np.clip(t - lat, ts[0], ts[-1]))
        ch.set_positions(np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])]))
        for k in range(4): d.qpos[jadr[k]] += np.radians(joff[k])
        sl, sh = sim_lime(); rl = real[f.name]
        if sl.sum() < 50: continue
        dts = cv2.distanceTransform((~sl).astype(np.uint8), cv2.DIST_L2, 3)
        costs.append((np.minimum(dts[rl], 60).mean() + np.minimum(dt_real[f.name][sl], 60).mean()) / 2)
        if sh.sum() > 30:
            ys, xs = np.nonzero(sh); c = np.array([xs.mean(), ys.mean()])
            ry, rx = np.nonzero(rl); near = np.hypot(rx - c[0], ry - c[1]) < 60
            if near.sum() > 30: offs.append([rx[near].mean() - c[0], ry[near].mean() - c[1]])
    m.body_pos[base] = base_pos0
    offs = np.array(offs) if offs else np.zeros((0, 2))
    return {"cost_px": round(float(np.mean(costs)), 2), "n": len(costs), "housing_offset_median_px": np.median(offs, 0).round(1).tolist() if len(offs) else None,
            "housing_dy_positive_share": round(float((offs[:, 1] > 5).mean()), 2) if len(offs) else None}

res = {"source": "review/export/robotcam_allowed (frames_x5 overview, states_allowed.parquet)", "latency_kit_s": lat0,
       "frames": {k: [f.name for _, f, _, _ in v] for k, v in sets.items()}, "variants": {}}
def rec(name, **kw):
    res["variants"][name] = {"params": {k: (np.round(v, 3).tolist() if hasattr(v, "__len__") else v) for k, v in kw.items()},
                             "train": evaluate(sets["train"], **kw), "dev": evaluate(sets["dev"], **kw)}
    print(name, json.dumps(res["variants"][name]), flush=True); json.dump(res, open(OUT, "w"), indent=1)
rec("K_kit")
fit = minimize(lambda x: evaluate(sets["train"], joff=x)["cost_px"], np.zeros(4), method="Nelder-Mead",
               options={"xatol": 0.05, "fatol": 0.02, "maxfev": 160, "initial_simplex": np.vstack([np.zeros(4)] + [np.eye(4)[i] * 2.0 for i in range(4)])})
rec("J_joint_offsets_deg", joff=fit.x)
x0 = np.r_[fit.x, 0, 0, 0]
fit2 = minimize(lambda x: evaluate(sets["train"], joff=x[:4], dbase=x[4:])["cost_px"], x0, method="Nelder-Mead",
                options={"xatol": 0.05, "fatol": 0.02, "maxfev": 240, "initial_simplex": np.vstack([x0] + [x0 + np.eye(7)[i] * (1.5 if i < 4 else 8.0) for i in range(7)])})
rec("JB_joint_offsets_deg_base_mm", joff=fit2.x[:4], dbase=fit2.x[4:])
best = min(np.arange(-0.2, 0.101, 0.0333), key=lambda l: evaluate(sets["train"], lat=l)["cost_px"])
rec("L_latency_s", lat=float(best))
