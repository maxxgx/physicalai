"""r17 item 1 (coordinator; D2): the wrist camera fitted to the fingers AND the world together (allowed frames only).

r16's camera was fitted to the fingers alone (fit_gripper_camera.py) and misplaced buckets and the desk by 20-50 px;
camera B lined up the world but not the fingers. Cost on TRAIN (0, 2, 6, 7, 9), choice on DEV (1, 3, 10):
  fingers: fit_gripper_camera.py's (four finger edges per row and both tips, truncated at 60 px; still-gripper frames),
           with the r16 map (85 mm at -270);
  world:   on wrist frames where a red/green/blue bucket shows (wrist_bucket_edges.py's pick and masks; each episode's
           stored 120 mm layout; the arm at t - latency), the symmetric mean contour distance of each bucket (px), and
           (largest component of each mask). The desk's edge (white S < 45, V > 165) was tried and is off: its largest
           white component is the wall, not the desk (60 px, the truncation, for every camera);
  J = finger cost + world cost (both px).
Hypotheses (each bounded; the camera always from the mount STEP's seat within 10 mm, 3 deg, fovy 65-71):
  H0 the camera only;
  H1 + a finger/jaw mesh offset in the finger bodies' x-y (perpendicular to the rail), +-5 mm (along the rail is g0);
  H2 + the camera seat moved along the bracket (the camera's optical axis), +-30 mm (camera B sat ~21-26 mm away);
  H3 + g0 within the user's 82-88 mm at -270 (the slope kept);
  H4 + the bucket's footprint (drawn mesh length and width) scaled +-5 % (the height stays the user's 120 mm).
LAYOUT_KIT=kit.json: the bucket layouts from that kit (a candidate overview camera's FIT_LAYOUT fits).
Reported per hypothesis: TRAIN / DEV cost (fingers, world, J), parameters, bound use. Also camera B and the r16 camera.
usage: fit_wrist_world.py STATES_PARQUET FRAMES_X5 OUT_JSON [--hyp H0,H1,H2,H3]"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
from scipy.optimize import minimize
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
OUT = sys.argv[3]; HYP = sys.argv[sys.argv.index("--hyp") + 1].split(",") if "--hyp" in sys.argv else ["H0", "H1", "H2", "H3"]
ARGS = sys.argv[1:3]
# the finger cost: fit_gripper_camera.py's code up to its results (it execs fit_finger_mount.py: model, frames, masks)
src = (HERE / "fit_gripper_camera.py").read_text().split("res = {")[0]
sys.argv = [str(HERE / "fit_gripper_camera.py"), *ARGS, "/dev/null"]
G = {"__file__": str(HERE / "fit_gripper_camera.py"), "__name__": "fgc"}; exec(compile(src, "fit_gripper_camera.py", "exec"), G)
m, d, cam, ns = G["m"], G["d"], G["cam"], G["ns"]; st, F, lat, ch = ns["st"], ns["F"], ns["lat"], ns["ch"]
measure, fcost, set_camera, PRI, B = G["measure"], G["cost"], G["set_camera"], G["PRI"], G["B"]
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
LAY = json.load(open(ROOT / "align/bucket_layouts_episodes.json"))["episodes"]
if os.environ.get("LAYOUT_KIT"):   # r17: the layouts a candidate overview camera's FIT_LAYOUT gave (its kit's episodes)
    LAY = json.load(open(os.environ["LAYOUT_KIT"]))["bucket_layouts_episodes"]["episodes"]
KIT = json.load(open(os.environ["KIT"])); RULE = KIT["bucket_layouts_episodes"]["episodes"]
from physicalai_mujoco_plugin.profiles import rebot_b601 as prof
S_MAP = prof.REBOT_B601_GRIPPER_OPENING_PER_DEG * 1000 / 2; G0_MAP = prof.REBOT_B601_GRIPPER_OPENING_AT_ZERO * 1000
import bucketsim_v2 as b
robot = ns.get("robot")

# --- the world: buckets (wrist_bucket_edges.py's masks) and the desk ---------------------------------------------
COL = ("red", "green", "blue"); PROXY = __import__("re").compile(r"_plan\d+$|_pc\d+_\d+$")
seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering()
vopt = mujoco.MjvOption(); vopt.geomgroup[:] = 0; vopt.geomgroup[0] = 1
bucket_mesh = {c: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"bucket_{c}").id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                   and m.geom_contype[g] == 0 and m.mesh(m.geom_dataid[g]).name == "bucket"] for c in COL}
ARM = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("link5", "link6", "gripper_end", "gripper_left", "gripper_right")
       and m.geom_contype[g] == 0 and not PROXY.search(m.geom(g).name or "")]
TABLE = [m.geom("table_top").id]
bucket_adr = {c: m.jnt_qposadr[m.joint(f"bucket_{c}:joint").id] for c in ("black", "green", "red", "blue")}

def set_layout(poses):
    for c, p in poses.items():
        a = bucket_adr[c]; d.qpos[a:a + 3] = [p["x"], p["y"], 0.0]; d.qpos[a + 3:a + 7] = [np.cos(p["yaw"] / 2), 0, 0, np.sin(p["yaw"] / 2)]

def sim_world():
    geoms = [g for c in COL for g in bucket_mesh[c]] + ARM + TABLE
    keep = m.geom_rgba.copy(); grp = m.geom_group.copy()
    m.geom_rgba[:, 3] = 0; m.geom_rgba[geoms, 3] = 1; m.geom_group[geoms] = 0
    try:
        mujoco.mj_forward(m, d); seg.update_scene(d, camera=cam, scene_option=vopt); s = seg.render()
    finally:
        m.geom_rgba[:] = keep; m.geom_group[:] = grp
    isg = s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    out = {c: isg & np.isin(s[..., 0], bucket_mesh[c]) for c in COL}; out["desk"] = isg & np.isin(s[..., 0], TABLE)
    return out

def real_world(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); h, s_, v = (hsv[..., i].astype(int) for i in range(3))
    mk = {"red": ((h < 10) | (h > 168)) & (s_ > 110) & (v > 60), "green": (h >= 60) & (h <= 95) & (s_ > 70) & (v > 60),
          "blue": (h >= 100) & (h <= 130) & (s_ > 110) & (v > 50), "desk": (s_ < 45) & (v > 165)}
    return {c: cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0 for c, x in mk.items()}

def largest(x):
    n, lab, stt, _ = cv2.connectedComponentsWithStats(x.astype(np.uint8))
    return lab == 1 + int(np.argmax(stt[1:, 4])) if n > 1 else x

def contour_dist(a_, b_):
    a_, b_ = largest(a_), largest(b_)   # r17: the largest component of each (the real masks are fragmented)
    def pts(x):
        cs = cv2.findContours(x.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        p = np.vstack([c[:, 0] for c in cs]) if cs else np.zeros((0, 2), int)
        return p[(p[:, 0] > 2) & (p[:, 0] < 637) & (p[:, 1] > 2) & (p[:, 1] < 477)]
    ca, cb = pts(a_), pts(b_)
    if not len(ca) or not len(cb): return np.nan
    return float(min(60.0, (cKDTree(cb).query(ca)[0].mean() + cKDTree(ca).query(cb)[0].mean()) / 2))

def world_frames(eps, per=8):
    out = []
    for ep in eps:
        rows = st[st.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
        cand = []
        for f in sorted(F.glob(f"gripper_e{ep:03d}_t*.png")):
            tl = int(f.stem.split("_t")[1]) / 1000 - lat
            if not ts[0] <= tl <= ts[-1]: continue
            rm = real_world(cv2.imread(str(f)))
            if max(rm[c].sum() for c in COL) >= 1500: cand.append((ep, f, np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])]), rm))
        out += [cand[i] for i in np.linspace(0, len(cand) - 1, min(per, len(cand))).astype(int)] if cand else []
    return out
USE_DESK = os.environ.get("WRIST_DESK") == "1"
WF = {"train": world_frames((0, 2, 6, 7, 9)), "dev": world_frames((1, 3, 10))}
FF = {"train": G["train"], "dev": G["dev"]}

def world_cost(split, opening):
    dist = []
    for ep, f, state, rm in WF[split]:
        ch.set_positions(state); set_layout(LAY[str(ep)]["poses"])
        for a in G["fadr"]: d.qpos[a] = opening(state[-1]) / 2000.0
        sm = sim_world()
        for c in COL:
            if rm[c].sum() >= 1500 and sm[c].sum() >= 300:
                v = contour_dist(rm[c], sm[c])
                if np.isfinite(v): dist.append(v)
        share = rm["desk"].mean()
        if USE_DESK and 0.1 <= share <= 0.9:   # r17: off (the largest white component is the wall, not the desk: 60 px for every camera)
            v = contour_dist(rm["desk"], sm["desk"])
            if np.isfinite(v): dist.append(v)
    return float(np.mean(dist)) if dist else 60.0

BMESH = m.mesh("bucket").id; BV0 = m.mesh_vert[m.mesh_vertadr[BMESH]:m.mesh_vertadr[BMESH] + m.mesh_vertnum[BMESH]].copy()
_foot = [None]
def set_footprint(sxy):   # r17 H4: the drawn bucket mesh scaled in its x and y (the height stays the user's 120 mm)
    sxy = np.asarray(sxy, float)
    if _foot[0] is not None and np.allclose(_foot[0], sxy): return
    a = m.mesh_vertadr[BMESH]; m.mesh_vert[a:a + len(BV0)] = BV0 * np.r_[sxy, 1.0]
    mujoco.mjr_uploadMesh(m, seg._mjr_context, BMESH); _foot[0] = sxy.copy()

fl, fr_ = m.body("gripper_left").id, m.body("gripper_right").id; FPOS0 = {k: m.body_pos[k].copy() for k in (fl, fr_)}
def set_finger_offset(o):
    for k in (fl, fr_):
        R = np.zeros(9); mujoco.mju_quat2Mat(R, m.body_quat[k]); m.body_pos[k] = FPOS0[k] + R.reshape(3, 3) @ np.array([o[0], o[1], 0.0]) / 1000.0

def unpack(h, x):
    dp = 10.0 * np.tanh(x[0:3]); dr = 3.0 * np.tanh(x[3:6]); fovy = 68.0 + 3.0 * np.tanh(x[6]); extra = {}
    k = 7
    if h == "H1": extra["finger_offset_mm"] = 5.0 * np.tanh(x[k:k + 2]); k += 2
    if h == "H2": extra["seat_along_axis_mm"] = 30.0 * np.tanh(x[k]); k += 1
    if h == "H3": extra["open270_mm"] = 85.0 + 3.0 * np.tanh(x[k]); k += 1
    if h == "H4": extra["footprint_scale"] = 1.0 + 0.05 * np.tanh(x[k:k + 2]); k += 2   # the bucket's length and width
    return dp, dr, fovy, extra, k

def apply(h, x):
    dp, dr, fovy, extra, _ = unpack(h, x)
    if "seat_along_axis_mm" in extra:   # along the camera's optical axis (-z of the MuJoCo camera frame), in link6's frame
        R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(PRI["quat_wxyz"], float)); dp = dp + R.reshape(3, 3) @ np.array([0, 0, -extra["seat_along_axis_mm"]])
    set_camera(PRI["pos_m"], PRI["quat_wxyz"], fovy, dp, dr)
    set_finger_offset(extra.get("finger_offset_mm", np.zeros(2)))
    set_footprint(extra.get("footprint_scale", np.ones(2)))
    o270 = extra.get("open270_mm", 270 * 2 * S_MAP + G0_MAP); g0 = o270 - 540 * S_MAP
    return lambda r: g0 + 2 * S_MAP * abs(r)

def evaluate(opening, split):
    fc = fcost(*measure(FF[split], opening)); wc = world_cost(split, opening)
    return {"fingers": round(fc, 2), "world": round(wc, 2), "J": round(fc + wc, 2)}

res = {"method": __doc__.split("usage")[0].strip(), "frames": {"fingers": {k: [f.name for f, _ in v] for k, v in FF.items()},
       "world": {k: [f.name for _, f, _, _ in v] for k, v in WF.items()}}, "baselines": {}, "hypotheses": {}}
r16 = lambda r: G0_MAP + 2 * S_MAP * abs(r)
set_camera(KIT["cameras"]["wrist"]["pos"], KIT["cameras"]["wrist"]["quat"], KIT["cameras"]["wrist"]["fovy"]); set_finger_offset(np.zeros(2))
res["baselines"]["r16_camera"] = {s: evaluate(r16, s) for s in ("train", "dev")}
cb = KIT["cameras"]["wrist"].get("r15_camera_B") or B
set_camera(cb["pos"], cb["quat"], cb["fovy"]); res["baselines"]["camera_B"] = {s: evaluate(r16, s) for s in ("train", "dev")}
print("baselines", json.dumps(res["baselines"]), flush=True); json.dump(res, open(OUT, "w"), indent=1)
for h in HYP:
    n = unpack(h, np.zeros(12))[4]
    fit = minimize(lambda x: evaluate(apply(h, x), "train")["J"], np.zeros(n), method="Nelder-Mead",
                   options={"xatol": 0.01, "fatol": 0.02, "maxfev": 120 * n, "initial_simplex": np.vstack([np.zeros(n)] + [np.eye(n)[i] * 0.4 for i in range(n)])})
    op = apply(h, fit.x); dp, dr, fovy, extra, _ = unpack(h, fit.x)
    bu = {"dpos": [round(abs(v) / 10, 3) for v in dp], "drot": [round(abs(v) / 3, 3) for v in dr], "fovy": round(abs(fovy - 68) / 3, 3)}
    if "finger_offset_mm" in extra: bu["finger_offset"] = [round(abs(v) / 5, 3) for v in extra["finger_offset_mm"]]
    if "seat_along_axis_mm" in extra: bu["seat"] = round(abs(float(extra["seat_along_axis_mm"])) / 30, 3)
    if "open270_mm" in extra: bu["open270"] = round(abs(float(extra["open270_mm"]) - 85) / 3, 3)
    if "footprint_scale" in extra: bu["footprint"] = [round(abs(float(v) - 1) / 0.05, 3) for v in extra["footprint_scale"]]
    res["hypotheses"][h] = {"train": evaluate(op, "train"), "dev": evaluate(op, "dev"), "evals": int(fit.nfev),
                            "dpos_mm": dp.round(2).tolist(), "drot_deg": dr.round(3).tolist(), "fovy": round(float(fovy), 3),
                            **{k: (np.round(v, 2).tolist() if hasattr(v, "__len__") else round(float(v), 2)) for k, v in extra.items()},
                            "camera_pos_m": m.cam_pos[cam].round(6).tolist(), "camera_quat": m.cam_quat[cam].round(6).tolist(), "bound_use": bu}
    print(h, json.dumps(res["hypotheses"][h]), flush=True); json.dump(res, open(OUT, "w"), indent=1)
    set_finger_offset(np.zeros(2)); set_footprint(np.ones(2))
