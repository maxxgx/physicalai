"""bucket_sort_desk_v2 real2sim: the MANIFEST commands (review RUBRIC section 7).

  bucketsim_v2.py heldout_render --frames f00004,f00008 --out DIR          matched state at the registered PINHOLE poses
  bucketsim_v2.py video --t0 T0 --t1 T1 --seed S --out FILE [--sbs]         the phone path, sim running the demonstrator
  bucketsim_v2.py demo --seeds 1,2 --pairs cube:green,cpu:red --variant default|rand --out JSON [--dump DIR]
  bucketsim_v2.py robotcam --seed S --variant default|rand --out DIR        overview.mp4, wrist.mp4, frames/events/poses
  bucketsim_v2.py rand --seeds 1,2,3 --out DIR                               both cameras at t=0 and first grasp + variant.json
  bucketsim_v2.py speed [--frames 300]
  bucketsim_v2.py cal
  bucketsim_v2.py stability --episodes 10 [--port 8941]                      through the plugin CLI (subprocess)

REAL2SIM_ROOT names the project folder (calibration.json, colmap/pinhole_cameras.json); KIT the kit file (default
REAL2SIM_ROOT/kit/real2sim.json; a delivery passes its own). Run from a delivery: it imports the delivery's frozen plugin."""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent / "plugin" if (HERE.parent / "plugin").exists() else HERE.parents[2]
sys.path.insert(0, str(PLUGIN / "src"))
sys.path.insert(0, str(HERE))
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser().resolve()
KIT = Path(os.environ.get("KIT", ROOT / "kit/real2sim.json")).expanduser().resolve()
# r13: this desk's arm unit: its joint zeros (kit rebot_b601_joint_zero_deg) apply through the profile's per-unit setting,
# before the plugin is imported (read at import); R2S_JOINT_ZERO=0 turns them off (r12's mapping)
if os.environ.get("R2S_JOINT_ZERO", "1") != "0" and KIT.exists() and "rebot_b601_joint_zero_deg" in json.loads(KIT.read_text()):
    os.environ.setdefault("PHYSICALAI_REBOT_B601_JOINT_ZERO", str(KIT))
from heldout import held, in_heldout  # noqa: E402

CAM_W, CAM_H, CAM_FPS = 640, 480, 30
SCENE = "bucket_sort_desk_v2"
DT = 0.02


def kit():
    from physicalai_mujoco_plugin.real2sim.desk_compositor import DeskKit
    return DeskKit.from_json(KIT)


def calibration() -> dict:
    return json.load(open(kit().calibration))


def pinhole_cameras() -> dict:
    out = {}
    rel = json.load(open(KIT)).get("pinhole_cameras")
    path = (KIT.parent / rel) if rel else ROOT / "colmap/pinhole_cameras.json"
    for n, c in json.load(open(path)).items():
        T = np.eye(4); T[:3] = np.array(c["cam_from_world"]); out[n] = (T, np.array(c["K"]), c["w"], c["h"])
    return out


def layout_for(variant: str):
    from dataclasses import replace
    from physicalai_mujoco_plugin.bucket_sort_arms import REBOT_DESK_V2_DEFAULT, REBOT_DESK_V2_LAYOUT, REBOT_DESK_V2_RAND
    # record: the Studio recording mix (REBOT_DESK_V2_LAYOUT: real look 3 in 10, stratified retry after a failure)
    return replace({"default": REBOT_DESK_V2_DEFAULT, "rand": REBOT_DESK_V2_RAND, "record": REBOT_DESK_V2_LAYOUT}[variant])


def make_robot(seed: int, variant: str = "default"):
    from physicalai_mujoco_plugin.robot import MuJoCoRobot
    robot = MuJoCoRobot("rebot_b601", scene=SCENE, cameras=[], http_port=0, seed=seed)
    robot.connect(); robot._automation.set_autopilot("off")
    robot._sim.on_reset.area = layout_for(variant)
    return robot


def reset(robot, seed: int, required: str | None = None, bucket: str | None = None) -> None:
    from physicalai_mujoco_plugin.bucket_sort_objects import real_kind
    task = robot._sim.on_reset
    task.required = required
    # a kind-level task's pairs name kinds (r09 D4: passing the body raised for every pair)
    named = (real_kind(required) or required) if (required is not None and getattr(task, "_kind_mode", False)) else required
    task.set_allowed_pairs(None if required is None or bucket is None else [(named, bucket)])
    robot._set_seed(seed); robot._reset(); robot._set_seed(None)


def compositor(robot, w=CAM_W, h=CAM_H, show_objects=True):
    from physicalai_mujoco_plugin.real2sim.desk_compositor import Compositor
    return Compositor(robot._sim.model, robot._sim.data, kit(), width=w, height=h, show_objects=show_objects)


def tick(robot, demo) -> None:
    """One control tick; while the demonstrator plans on its thread, wait without stepping, so runs depend on the seed only."""
    robot.get_observation()
    while demo.planner_running:
        time.sleep(0.0005)


def object_names(m) -> tuple[str, ...]:
    from physicalai_mujoco_plugin.bucket_sort import scene_objects
    return scene_objects(m)


def type_body(m, kind: str) -> str:
    """The body for a declared object type: resistor, battery, cpu, or a pool kind (its first body)."""
    names = object_names(m)
    if kind not in declared_types(m):
        raise SystemExit(f"unknown object type {kind!r}; declared: {declared_types(m)}")
    if kind in names: return kind
    for n in names:
        if n.startswith(f"obj_{kind}_"): return n
    raise SystemExit(f"unknown object type {kind!r}; declared: {declared_types(m)}")


def declared_types(m) -> list[str]:
    """The object types DEMO accepts. A kind-level desk (the real objects, TaskLayout.kind_counts) names only its real
    kinds (resistor, cpu, battery); its generic shapes are unnamed distractors, never a target."""
    from physicalai_mujoco_plugin.bucket_sort_objects import REAL_KINDS, kind_of
    kinds_only = layout_for("default").kind_counts is not None
    out = []
    for n in object_names(m):
        k = kind_of(n) or n
        if kinds_only and k not in REAL_KINDS: continue
        if k not in out: out.append(k)
    return out


# --- matched state ------------------------------------------------------------------------------------------
def bucket_pose_at(cal: dict, colour: str, t: float | None) -> dict:
    """A bucket's measured pose at video time *t* (s): the calibration's, unless a declared move (``bucket_moves``) had
    happened by then (the person moved it while handling its contents)."""
    pose = cal["buckets"][colour]
    for mv in cal.get("bucket_moves", []):
        if mv["bucket"] == colour and t is not None and t >= mv["from_s"]:
            pose = mv
    return pose


def matched_state(robot, t: float | None = None) -> None:
    """The scene as in the video at time *t* (s; None = its start): the arm at the calibrated joints, buckets as
    measured then and full, every task object parked."""
    from physicalai_mujoco_plugin.bucket_sort import park_objects, set_bucket_fill, bucket_colors
    cal = calibration(); m, d = robot._sim.model, robot._sim.data
    for i, v in enumerate(cal["robot"]["joints_rad"]):
        d.qpos[m.jnt_qposadr[m.joint(f"joint{i + 1}").id]] = v
    for n in ("joint_left", "joint_right"):
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = cal["robot"].get("finger_opening_m", 0.0) / 2
    for c in bucket_colors(m):
        b = bucket_pose_at(cal, c, t); adr = m.jnt_qposadr[m.joint(f"bucket_{c}:joint").id]
        d.qpos[adr:adr + 3] = [b["x"], b["y"], 0.0]; d.qpos[adr + 3:adr + 7] = [np.cos(b["yaw"] / 2), 0, 0, np.sin(b["yaw"] / 2)]
    park_objects(m, d, object_names(m))
    set_bucket_fill(m, {c: 1.0 for c in bucket_colors(m)})
    for name in ("r2s_table_look", "r2s_bucket_rgb", "r2s_cam_jitter", "r2s_image"):
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
        if k >= 0: m.numeric_data[m.numeric_adr[k]:m.numeric_adr[k] + m.numeric_size[k]] = 0
    for name, v in (("r2s_exposure", [1]), ("r2s_wb", [1, 1, 1]), ("r2s_light", [0, 0, 1])):
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
        if k >= 0: m.numeric_data[m.numeric_adr[k]:m.numeric_adr[k] + len(v)] = v
    mujoco.mj_forward(m, d)


def cmd_heldout_render(a) -> None:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cams = pinhole_cameras(); names = [n if n.endswith(".jpg") else n + ".jpg" for n in a.frames.split(",")]
    robot = make_robot(0); matched_state(robot)
    _, _, W, H = cams[names[0]]
    comp = compositor(robot, W, H, show_objects=False); meta = {}
    comp.phone_finger_handover = True   # r17 (R-56): matched-state phone views only (the arm at its calibrated pose)
    cal = calibration()
    for n in names:
        if n not in cams: print("not registered:", n, flush=True); continue
        T, K, w, h = cams[n]; assert (w, h) == (W, H)
        t_n = int(n[1:6]) / 30.0
        if cal.get("bucket_moves"): matched_state(robot, t_n)   # the scene at this frame's time
        # (tried at r07: drawing a moved bucket from the sim in these views - lower PSNR, its CG contents, and its ghost
        # zone tore the neighbour's edge; the splat's piece, moved, stays)
        r = comp.render_sfm(T, K, outputs=True); stem = Path(n).stem
        cv2.imwrite(str(out / f"{stem}.png"), r["rgb"][:, :, ::-1]); cv2.imwrite(str(out / f"{stem}_seg.png"), r["seg"])
        cv2.imwrite(str(out / f"{stem}_outline.png"), cv2.cvtColor(r["outline"], cv2.COLOR_RGBA2BGRA))
        meta[stem] = {"width": w, "height": h, "K": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], "cam_from_world": T[:3].tolist()}
        print("wrote", stem, flush=True)
    json.dump(meta, open(out / "cameras.json", "w"), indent=1)


# --- the demonstrator over time -----------------------------------------------------------------------------
def run_demo(robot, seconds: float, fps: float, on_frame=None) -> list[dict]:
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator
    t, k, events, last = 0.0, 0, [], None
    while t < seconds:
        tick(robot, demo); t += DT
        while k / fps <= t and k / fps < seconds:
            if on_frame: on_frame(k)
            k += 1
        ep = robot._http_status()["episode"]; key = ep["episode_count"]
        if key != last:
            events.append({"t": round(t, 2), "episode": key, "instruction": (ep.get("instruction") or {}).get("text")}); last = key
    return events


def cmd_video(a) -> None:
    from scipy.spatial.transform import Rotation, Slerp
    cams = pinhole_cameras()
    reg = sorted((int(n[1:6]) / 30.0, n) for n in cams); ts = np.array([t for t, _ in reg])
    C = np.array([-cams[n][0][:3, :3].T @ cams[n][0][:3, 3] for _, n in reg])
    slerp = Slerp(ts, Rotation.from_matrix(np.array([cams[n][0][:3, :3].T for _, n in reg])))
    def pose_at(t):
        t = float(np.clip(t, ts[0], ts[-1])); c = np.array([np.interp(t, ts, C[:, i]) for i in range(3)])
        R = slerp([t]).as_matrix()[0]; T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ c; return T
    _, K0, W0, H0 = cams[reg[0][1]]
    W, H = 1280, 720   # the phone's 1884x1059 scaled by 1280/1884 (719.5 rows), the principal point moved to the centre row
    s_ = W / W0; K = K0.copy(); K[:2] *= s_; K[1, 2] += (H - H0 * s_) / 2
    robot = make_robot(a.seed); reset(robot, a.seed)
    comp = compositor(robot, W, H)
    assert not comp.phone_finger_handover   # r17: the finger hand-over never acts in the VIDEO (the CAD fingers always drawn)
    fps, k0 = 15.0, int(round(a.t0 * 15))
    dec = None
    if a.sbs:   # reads the MOV: not for the sandbox; the real half is black inside the held-out windows and guard
        SDR_RECIPE = ("zscale=tin=arib-std-b67:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=100:p=bt2020,format=gbrpf32le,"
                      "zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=pc,format=rgb24")   # RUBRIC 3.1
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a.t0:.4f}", "-i", os.path.expanduser("~/Downloads/rebot_setup.MOV"),
                                "-vf", f"fps=15,{SDR_RECIPE},scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{(2 if a.sbs else 1) * W}x{H}", "-r", "15",
                            "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    def frame(k):
        if k < k0: return
        img = comp.render_sfm(pose_at(k / fps), K)["rgb"][:, :, ::-1]
        if dec is not None:   # NB: the raw real frame is decoded (to keep the stream in step) but never shown when held out
            raw = dec.stdout.read(W * H * 3); real = np.zeros_like(img)
            if len(raw) == W * H * 3 and not in_heldout(k / fps): real = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
            img = np.hstack([real, img])
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
    events = run_demo(robot, a.t1, fps, frame)
    enc.stdin.close(); enc.wait()
    json.dump(events, open(str(a.out) + ".events.json", "w"), indent=1); print("wrote", a.out)


# --- DEMO -----------------------------------------------------------------------------------------------------
def physics_flags(m, d) -> list[str]:
    flags = []
    for e in range(m.neq):
        if d.eq_active[e]:
            kind = mujoco.mjtEq(m.eq_type[e]).name
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT if kind == "mjEQ_JOINT" else mujoco.mjtObj.mjOBJ_BODY, i) for i in (m.eq_obj1id[e], m.eq_obj2id[e])]
            if not (kind == "mjEQ_JOINT" and all(n and n.startswith("joint_") for n in names)):
                flags.append(f"{kind} {names}")
    if m.nmocap: flags.append(f"{m.nmocap} mocap bodies")
    return flags


def max_penetration(m, d, bodies: set[int]) -> float:
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if m.geom_bodyid[c.geom1] in bodies or m.geom_bodyid[c.geom2] in bodies:
            worst = max(worst, -float(c.dist))
    return worst


_DRAWN_TREES: dict = {}


def _drawn_trees(m) -> dict:
    """Per finger and pool object body: a KD-tree of its drawn surface (body frame; ~0.3 mm samples), cached per model."""
    if id(m) in _DRAWN_TREES: return _DRAWN_TREES[id(m)]
    from scipy.spatial import cKDTree
    from collision_decomp import sample_surface
    import re
    proxy = re.compile(r"_plan\d+$|_pc\d+_\d+$"); out = {}; rng = np.random.default_rng(0)
    bodies = [m.body(n).id for n in ("gripper_left", "gripper_right")] + [b for b in range(m.nbody) if m.body(b).name.startswith("obj_")]
    for b in bodies:
        Vs, Fs, off = [], [], 0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] or m.geom_conaffinity[g] or proxy.search(m.geom(g).name or ""): continue
            if re.search(r"_hull\d+$|_tight\d+$", m.geom(g).name or ""): continue   # collision pieces (contacts off while parked)
            r_ = np.zeros(9); mujoco.mju_quat2Mat(r_, m.geom_quat[g]); mid = int(m.geom_dataid[g])
            a_, n_ = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            Vs.append(m.mesh_vert[a_:a_ + n_].astype(float) @ r_.reshape(3, 3).T + m.geom_pos[g]); Fs.append(m.mesh_face[fa:fa + fn] + off); off += n_
        if Vs: out[b] = cKDTree(sample_surface(np.vstack(Vs), np.vstack(Fs), 0.0003, rng, cap=200_000))
    _DRAWN_TREES[id(m)] = out
    return out


def run_episode(robot, seed: int, max_s: float = 45.0, on_tick=None) -> dict:
    """One demonstrator episode from the current reset; the task's own success, held for 1 s of sim time."""
    m, d = robot._sim.model, robot._sim.data; task = robot._sim.on_reset; ins = task.instruction
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator; demo._rng = np.random.default_rng(seed)
    count = robot._http_status()["episode"]["episode_count"]; ok, t, wall, held_since = False, 0.0, time.time(), None
    names = object_names(m); placed = [n for n in names if n in task.layout.objects]
    bodies = {m.body(n).id for n in placed}   # the parked ones rest on the hidden shelf, off stage
    ids = np.array([m.body(n).id for n in placed]); dofs = np.array([m.jnt_dofadr[m.joint(f"{n}:joint").id] for n in placed])
    pen, settle_s, first_grasp, calm = 0.0, None, None, 0
    others = np.array([m.body(n).id for n in placed if n != ins.object], dtype=int); rest, nudge = None, 0.0
    nudge_v, nudge_who, nudge_t = 0.0, None, None
    pen_at = None
    # r13 grasp measures: an object is held while geoms of BOTH fingers touch it; per held tick the finger opening and
    # the object's position in the gripper's frame (its change = sliding in the grip; the first 0.2 s after the grip
    # forms = snapping into it)
    fl, fr, ge, l6 = m.body("gripper_left").id, m.body("gripper_right").id, m.body("gripper_end").id, m.body("link6").id
    fadr = [m.jnt_qposadr[m.joint(n).id] for n in ("joint_left", "joint_right")]
    held_log: dict[str, list] = {}
    drawn_tree = _drawn_trees(m)   # r14 (R-44): the drawn surfaces per body, for the gap between the drawings at a contact
    # r13: the gripper housing (whose collision reaches up to ~8 mm past its drawing) touching an object or a bucket
    bucket_bodies = {m.body(f"bucket_{c}").id for c in ("black", "green", "red", "blue")}; housing_ticks = 0
    while t < max_s and time.time() - wall < 900:
        tick(robot, demo); t += DT
        if robot._http_status()["episode"]["episode_count"] != count: break
        _p = max_penetration(m, d, bodies)
        if _p > pen:
            pen = _p
            for _i in range(d.ncon):   # which contact (for the report)
                _c = d.contact[_i]
                if -_c.dist >= _p - 1e-9 and (m.geom_bodyid[_c.geom1] in bodies or m.geom_bodyid[_c.geom2] in bodies):
                    _nm = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
                    pen_at = (_nm(_c.geom1), _nm(_c.geom2), round(t, 2), demo.phase); break
        speeds = np.linalg.norm(d.qvel[dofs[:, None] + np.arange(3)], axis=1) if len(dofs) else np.zeros(0)
        # settled: every placed object below 0.01 m/s for 0.1 s on end (a single slow tick mid-bounce is not settling)
        calm = calm + 1 if (speeds < 0.01).all() else 0
        if settle_s is None and calm >= 5: settle_s = round(t - 4 * DT, 2)
        if first_grasp is None and demo.phase == "close": first_grasp = round(t, 2)
        # T6: once settled, before the first grasp, how far any OTHER placed object is pushed from where it settled
        if settle_s is not None and first_grasp is None and len(others):
            if rest is None: rest = d.xpos[others].copy()
            nudge = max(nudge, float(np.linalg.norm(d.xpos[others] - rest, axis=1).max()))
            od = np.array([m.jnt_dofadr[m.body_jntadr[bid]] for bid in others])
            sp = np.linalg.norm(d.qvel[od[:, None] + np.arange(3)], axis=1)
            if sp.max() > nudge_v: nudge_v, nudge_who, nudge_t = float(sp.max()), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(others[sp.argmax()])), round(t, 2)
        touch: dict[int, set] = {}
        for _i in range(d.ncon):
            _c = d.contact[_i]; b1, b2 = m.geom_bodyid[_c.geom1], m.geom_bodyid[_c.geom2]
            for fb, ob in ((b1, b2), (b2, b1)):
                if fb in (fl, fr) and ob in bodies: touch.setdefault(int(ob), set()).add(int(fb))
        # (r15: link6 too: the wrist mount and the camera's body collide since r15, under the same rule)
        if any((m.geom_bodyid[d.contact[_i].geom1] in (ge, l6) and m.geom_bodyid[d.contact[_i].geom2] in bodies | bucket_bodies) or
               (m.geom_bodyid[d.contact[_i].geom2] in (ge, l6) and m.geom_bodyid[d.contact[_i].geom1] in bodies | bucket_bodies) for _i in range(d.ncon)):
            housing_ticks += 1
        for ob, fs in touch.items():
            if len(fs) == 2:
                rel = d.xmat[ge].reshape(3, 3).T @ (d.xpos[ob] - d.xpos[ge])
                gap = 0.0
                for _i in range(d.ncon):   # R-44: at each finger-object contact, the drawn finger's and the drawn object's distance
                    _c = d.contact[_i]; b1, b2 = int(m.geom_bodyid[_c.geom1]), int(m.geom_bodyid[_c.geom2])
                    if {b1, b2} & {fl, fr} and ob in (b1, b2):
                        g_ = 0.0
                        for bb in (b1, b2):
                            if bb in drawn_tree:
                                local = d.xmat[bb].reshape(3, 3).T @ (_c.pos - d.xpos[bb]); g_ += float(drawn_tree[bb].query(local)[0])
                        gap = max(gap, g_)
                held_log.setdefault(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, ob), []).append((round(t, 3), sum(float(d.qpos[a_]) for a_ in fadr) * 1000, rel.copy(), gap))
        if on_tick: on_tick(t, d.xpos[ids].copy() if len(ids) else np.zeros((0, 3)))
        ep = robot._http_status()["episode"]
        if ep["success"] and ep["in_bucket"][ins.object] == ins.bucket:
            held_since = t if held_since is None else held_since
            if t - held_since >= 1.0: ok = True; break
        else:
            held_since = None
        if demo.status()["outcome"] in {"failed", "stalled"}: break
    st = demo.status(); grasps = {}
    for name, log in held_log.items():
        T = np.array([x[0] for x in log]); W = np.array([x[1] for x in log]); P = np.array([x[2] for x in log]); GAP = np.array([x[3] for x in log])
        runs = np.split(np.arange(len(T)), np.flatnonzero(np.diff(T) > 2.5 * DT) + 1)   # each continuous hold
        g = []
        for r_ in runs:
            if len(r_) < 10: continue
            tt, pp = T[r_], P[r_]; v = np.linalg.norm(np.diff(pp, axis=0), axis=1) / np.maximum(np.diff(tt), 1e-9)
            early = tt <= tt[0] + 0.2
            g.append({"t": [float(tt[0]), float(tt[-1])], "opening_mm": round(float(np.median(W[r_])), 2),
                      "slide_after_0p2s_mm": round(float(np.linalg.norm(pp[-1] - pp[early][-1]) * 1000), 2),
                      "slide_speed_max_mmps_after_0p2s": round(float(v[~early[1:]].max() * 1000), 1) if (~early[1:]).any() else 0.0,
                      "snap_first_0p2s_mm": round(float(np.linalg.norm(pp[early][-1] - pp[0]) * 1000), 2),
                      "drawn_gap_mm_p95_max": [round(float(np.percentile(GAP[r_], 95)) * 1000, 2), round(float(GAP[r_].max()) * 1000, 2)]})
        if g: grasps[name] = g
    return {"seed": seed, "grasps": grasps, "housing_contact_ticks": housing_ticks, "unclean_reasons": list(st.get("reasons") or []), "clean": st.get("clean"), "object": ins.object, "object_label": ins.label or ins.object, "bucket": ins.bucket, "bucket_shown": ins.shown_bucket,
            "instruction": ins.text, "success": ok, "failure": None if ok else (st["failure"] or f"no settled success in {max_s:.0f} s"),
            # a glitch (rubric T2): any contact deeper than 3 mm, or a physics warning flag, during the episode
            "glitch": bool(pen > 0.003 or physics_flags(m, d)), "sim_s": round(t, 2), "physics_flags": physics_flags(m, d), "max_penetration_m": round(pen, 5), "max_penetration_at": pen_at,
            "settle_s": settle_s, "first_grasp_s": first_grasp, "pre_grasp_nudge_m": round(nudge, 4), "pre_grasp_speed_mps": round(nudge_v, 4), "pre_grasp_pushed": nudge_who, "pre_grasp_push_t": nudge_t, "placed_objects": placed, "pile": task.layout.pile,
            "variant": task.look, "bucket_shown_map": task.bucket_names,
            "buckets": {c: dict(task.layout.buckets[c].as_dict()) for c in task.layout.buckets}}


def cmd_replay(a) -> None:
    """ROBOTCAM_REPLAY (RUBRIC 7, 6.3): both robot cameras at recorded joint states and times. --states: a parquet of a
    LeRobot episode's rows (episode_index, timestamp, observation.state in the driver's public units, as Studio writes
    it; e.g. the reviewer's states_E.parquet); --times: "E:t" (episode, seconds) or "t" (a file of one episode), the
    state linearly interpolated in time between the rows around t. --layout episode (default, r10): the buckets as
    fitted for that episode (episode_layout: allowed episodes their own, any other the nearest allowed by index, ties
    to the earlier); --layout matched: the r09 behaviour below. The scene is otherwise the matched state (calibrated buckets,
    the real look, every task object parked: the sim does not know where the real objects lay) at --video-time on the
    phone capture's clock (the black bucket moved at 81.5 s); the joints and the gripper go through the rebot_b601
    profile's own channel conversion (ArmChannels.set_positions); sim time = t (sensor noise draws per time).
    Writes <E>_<t>_{overview,gripper}.png with _seg.png and _outline.png (the sim "wrist" camera is the real "gripper")
    and replay.json (the state used per time). Reads nothing but the states file and this delivery."""
    import pandas as pd
    st = pd.read_parquet(a.states)
    need = {"episode_index", "timestamp", "observation.state"}
    if not need <= set(st.columns):
        raise SystemExit(f"--states needs columns {sorted(need)}; has {sorted(st.columns)}")
    eps = sorted(int(e) for e in st["episode_index"].unique())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    robot = make_robot(0); reset(robot, 0); m, d = robot._sim.model, robot._sim.data
    matched_state(robot, a.video_time)
    comp = compositor(robot, show_objects=False); ch = robot._sim.channels[0]
    lat = json.load(open(KIT)).get("camera_latency_s", {}) if a.latency is None else {"overview": a.latency, "wrist": a.latency}
    record = []
    layout_file = layout_sha = None
    if a.layout not in ("episode", "matched"):
        import hashlib
        raw = Path(a.layout).read_bytes(); layout_file = json.loads(raw); layout_sha = hashlib.sha256(raw).hexdigest()
        if set(layout_file.get("poses", {})) != {"red", "green", "blue", "black"}:
            raise SystemExit(f"--layout {a.layout}: needs \"poses\" for red, green, blue and black")
    for item in [x for x in a.times.split(",") if x.strip()]:
        if ":" in item:
            e_s, t_s = item.split(":"); ep, t = int(e_s), float(t_s)
        elif len(eps) == 1:
            ep, t = eps[0], float(item)
        else:
            raise SystemExit(f"--times {item!r}: give E:t, the file holds episodes {eps}")
        rows = st[st["episode_index"] == ep].sort_values("timestamp")
        if rows.empty: raise SystemExit(f"episode {ep} not in --states")
        ts = rows["timestamp"].to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
        if not ts[0] - 1e-6 <= t <= ts[-1] + 1e-6:
            raise SystemExit(f"t = {t} outside episode {ep}'s rows ({ts[0]:.3f}-{ts[-1]:.3f} s)")
        stem = f"{ep:03d}_{t:08.4f}"; used = {}
        layout_src = None
        if a.layout == "episode":
            layout_src, poses = episode_layout(ep)
            if poses: set_bucket_poses(robot, poses)
        elif a.layout != "matched":   # r12 (U-5): a FIT_LAYOUT file (or any {"poses": {colour: {x, y, yaw}}})
            layout_src = {"file": str(Path(a.layout).resolve()), "sha256": layout_sha}
            set_bucket_poses(robot, layout_file["poses"])
        for cam, real in (("overview", "overview"), ("wrist", "gripper")):
            # the real camera shows the arm late by its latency (kit camera_latency_s): the state of t - latency
            tl = float(np.clip(t - float(lat.get(cam, 0.0)), ts[0], ts[-1]))
            state = np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])]); used[real] = [round(float(v), 5) for v in state]
            ch.set_positions(state)
            # r13: no replay-only drawing rule: the fingers are drawn where the recorded reading puts them, as a recording would
            d.time = t; mujoco.mj_forward(m, d)
            r = comp.render_camera(cam, outputs=True)
            if a.contents_mask:   # builder's diagnostic: where the sim bucket contents are (render once more, buckets empty)
                from physicalai_mujoco_plugin.bucket_sort import set_bucket_fill, bucket_colors
                set_bucket_fill(m, {c: 0.0 for c in bucket_colors(m)}); empty = comp.render_camera(cam, outputs=True)
                set_bucket_fill(m, {c: 1.0 for c in bucket_colors(m)})
                diff = (np.abs(empty["rgb"].astype(np.int16) - r["rgb"].astype(np.int16)).max(-1) > 12) & np.isin(r["seg"], [1, 2, 3, 4])
                cv2.imwrite(str(out / f"{stem}_{real}_contents.png"), (diff * 255).astype(np.uint8))
            cv2.imwrite(str(out / f"{stem}_{real}.png"), r["rgb"][:, :, ::-1]); cv2.imwrite(str(out / f"{stem}_{real}_seg.png"), r["seg"])
            cv2.imwrite(str(out / f"{stem}_{real}_outline.png"), cv2.cvtColor(r["outline"], cv2.COLOR_RGBA2BGRA))
        record.append({"episode": ep, "t": t, "bucket_layout_from_episode": layout_src, "state_public_used": used, "latency_s": {k: float(v) for k, v in lat.items() if k != "fitted_from"}, "files": f"{stem}_{{overview,gripper}}[_seg|_outline].png"})
    json.dump({"states": str(a.states), "video_time": a.video_time,
               "layout": {"episode": "the default: the per-bucket statement rule (kit bucket_layouts_episodes.bucket_segments)",
                          "matched": "the phone-matched layout (r09)"}.get(a.layout, f"file {Path(a.layout).resolve()} (sha256 {layout_sha})"), "cameras": {"overview": "overview", "gripper": "sim wrist"},
               "noise": KIT_NOISE_NOTE, "frames": record}, open(out / "replay.json", "w"), indent=1)
    print("replayed", len(record), "times ->", out)


def cmd_fit_layout(a) -> None:
    """FIT_LAYOUT (r12, U-5): the four bucket poses of one real episode from its overview video, by the builder's own
    per-episode fitter (layout_fitter.py, the method of fit_bucket_layouts.py unchanged). Frames: the video's decoded
    frames k with t0 <= k/30 <= t1 (file time, the video's own clock from 0), every 5th from the first of them, of which
    25 spread evenly by index; their per-pixel median. The arm's joints for its mask: --states rows (episode-relative
    timestamp = --start + k/30, the nearest row within half a frame) through the profile's channel conversion, the
    median over the 25 frames. Writes --out (poses, IoUs, the frame list, the inputs' sha256) and <out stem>_overlay.jpg
    (the median frame | the fitted sim outlines per bucket colour, the real colour masks in magenta, the excluded arm
    region hatched grey). Reads only --video, --states and this delivery."""
    import hashlib, subprocess
    import pandas as pd
    from layout_fitter import LayoutFitter, STEP, pick
    W, H = CAM_W, CAM_H
    st = pd.read_parquet(a.states)
    if a.episode is not None: st = st[st["episode_index"] == a.episode]
    if st.empty or st["episode_index"].nunique() != 1:
        raise SystemExit("--states: give a file of one episode, or --episode E")
    st = st.sort_values("timestamp"); ts = st["timestamp"].to_numpy(float); S = np.stack(st["observation.state"].to_numpy()).astype(float)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(a.video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "-"], stdout=subprocess.PIPE)
    cand, k = [], 0
    while True:
        raw = proc.stdout.read(W * H * 3)
        if len(raw) < W * H * 3: break
        if a.t0 - 1e-6 <= k / 30.0 <= a.t1 + 1e-6: cand.append((k, raw))
        k += 1
    proc.wait()
    if proc.returncode: raise SystemExit(f"ffmpeg could not decode {a.video}")
    cand = cand[::STEP]
    if not cand: raise SystemExit(f"no frame of {a.video} ({k} frames) in [{a.t0}, {a.t1}] s")
    chosen = pick(cand)
    robot = make_robot(0); reset(robot, 0); m, d = robot._sim.model, robot._sim.data
    matched_state(robot); compositor(robot, show_objects=False)   # applies the kit's fitted overview camera
    fitter = LayoutFitter(robot); ch = robot._sim.channels[0]
    keep = d.qpos.copy(); qs = []
    for kk, _ in chosen:
        te = a.start + kk / 30.0; i = int(np.argmin(np.abs(ts - te)))
        if abs(ts[i] - te) > 0.5 / 30 + 1e-6: raise SystemExit(f"--states has no row at {te:.3f} s (video frame {kk}; --start {a.start})")
        ch.set_positions(S[i]); qs.append([float(d.qpos[j]) for j in fitter.jadr]); d.qpos[:] = keep
    frames = [cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(H, W, 3), cv2.COLOR_RGB2BGR) for _, raw in chosen]
    res, overlay = fitter.fit(frames, np.median(np.array(qs), axis=0))
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res.update({"command": "FIT_LAYOUT", "video": str(Path(a.video).resolve()), "video_sha256": sha(a.video), "video_frames": k,
                "states": str(Path(a.states).resolve()), "states_sha256": sha(a.states), "episode": int(st["episode_index"].iloc[0]),
                "t0": a.t0, "t1": a.t1, "start": a.start, "method": LayoutFitter.__module__ + ": " + __import__("layout_fitter").__doc__.split("\n\n")[0],
                "frame_list": [{"video_frame": kk, "episode_time_ms": int(round((a.start + kk / 30.0) * 1000))} for kk, _ in chosen],
                "overlay": out.stem + "_overlay.jpg"})
    out.write_text(json.dumps(res, indent=1)); cv2.imwrite(str(out.with_name(out.stem + "_overlay.jpg")), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("fit_layout", json.dumps({c: (res["iou_fitted"][c], res["moved_mm_deg"][c]) for c in res["poses"]}), "->", out)


def episode_layout(ep: int) -> tuple[int | str | dict | None, dict | None]:
    """The bucket layout for real episode *ep*. r12 (the graded rule, `bucket_segments`): per bucket, the median of the
    allowed episodes' unchanged fits in the bucket's own segment: red {0}, {1-5}, {6-10} (the user: "red moved on ep2
    onwards and again on ep7", counting from 1); green and black {0-3}, {4-10}; blue one pool over every allowed episode.
    The source is then a dict per bucket (segment and the allowed episodes pooled), recorded in replay.json. r11 (U-4 as refined by the user: episodes 0-3 one layout, 4-10 another,
    the green and slightly the black bucket touched in between): the kit's segment layout, the per-bucket median of the
    allowed episodes' fits in the episode's segment (A: 0-3 from 0, 1, 2, 3; B: 4-10 from 6, 7, 9, 10). r10, without
    segments: the kit's `bucket_layouts_episodes`, fitted per
    allowed episode from the reviewer's export (the overview frames' temporal median, fit_bucket_layouts.py). Any other
    episode (held-out or unknown) uses the nearest allowed episode by index, ties to the earlier one; the rule was fixed
    and declared before any held-out comparison and is not tuned on held-out results. Returns (source episode, poses)."""
    lay = json.load(open(KIT)).get("bucket_layouts_episodes")
    if not lay: return None, None
    if lay.get("bucket_segments"):   # r12 graded rule (U-4 red, the user: "red moved on ep2 onwards and again on ep7")
        src, poses = {}, {}
        for c, segs in lay["bucket_segments"].items():
            seg = next(s for s in segs if ep in s["episodes"])
            src[c] = {"segment": f"{seg['episodes'][0]}-{seg['episodes'][-1]}", "allowed_pooled": seg["allowed_pooled"]}
            poses[c] = seg["pose"]
        return src, poses
    if lay.get("segments"):   # r11 graded rule (U-4 refined by the user): the pooled allowed fits of the episode's segment
        for name, seg in lay["segments"].items():
            if ep in seg["episodes"]:
                return f"segment {name} (allowed {','.join(str(e) for e in seg['allowed_pooled'])})", seg["poses"]
    allowed = sorted(int(e) for e in lay["episodes"])
    if os.environ.get("R2S_LAYOUT_LEAVE_OUT"):   # builder's check of the rule on allowed episodes: never an episode's own fit
        allowed = [k for k in allowed if k != ep]
    src = min(allowed, key=lambda k: (abs(k - ep), k))
    return src, lay["episodes"][str(src)]["poses"]


def set_bucket_poses(robot, poses: dict) -> None:
    m, d = robot._sim.model, robot._sim.data
    for c, b in poses.items():
        adr = m.jnt_qposadr[m.joint(f"bucket_{c}:joint").id]
        d.qpos[adr:adr + 3] = [b["x"], b["y"], 0.0]; d.qpos[adr + 3:adr + 7] = [np.cos(b["yaw"] / 2), 0, 0, np.sin(b["yaw"] / 2)]
    mujoco.mj_forward(m, d)


KIT_NOISE_NOTE = "sensor noise as the kit (R2S_SENSOR_NOISE=0 turns it off); drawn per sim time"


def cmd_kinds(a) -> None:
    """Kind-level episodes (r09) drawn from the task's own distribution (no forced pair): per seed, the instruction,
    the targets and whether the demonstrator sorted them all (the task's success held 1 s)."""
    from physicalai_mujoco_plugin.bucket_sort_objects import real_kind
    seeds = range(a.start, a.start + a.count)
    robot = make_robot(a.start, a.variant); out = []
    for seed in seeds:
        robot._automation.set_autopilot("off"); reset(robot, seed)
        task = robot._sim.on_reset; ins = task.instruction
        r = run_episode(robot, seed, max_s=25.0 + 18.0 * len(ins.all_targets))
        r.update({"kind": ins.kind, "targets": list(ins.all_targets), "canonical": ins.canonical_text,
                  "counts": {k: sum(real_kind(n) == k for n in task.layout.objects) for k in ("resistor", "cpu", "battery")},
                  "stacked": len(task.layout.heights)})
        out.append(r); print(json.dumps({k: r[k] for k in ("seed", "instruction", "kind", "success", "failure", "sim_s", "glitch", "max_penetration_m", "pile", "stacked", "counts")}), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    n = len(out); s = sum(r["success"] for r in out)
    print(f"success {s} / {n}; glitches {sum(r['glitch'] for r in out)}; by targets",
          {k: f"{sum(r['success'] for r in out if len(r['targets']) == k)}/{sum(len(r['targets']) == k for r in out)}" for k in range(1, 5)})


def cmd_demo(a) -> None:
    seeds = [int(s) for s in a.seeds.split(",")]; pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
    assert len(seeds) == len(pairs), "one pair per seed"
    robot = make_robot(seeds[0], a.variant); m = robot._sim.model; out = []
    if a.dump: Path(a.dump).mkdir(parents=True, exist_ok=True)
    for seed, (kind, bucket) in zip(seeds, pairs):
        robot._automation.set_autopilot("off"); body = type_body(m, kind); reset(robot, seed, body, bucket)
        rec_t, rec_p = [], []
        n_targets = len(getattr(robot._sim.on_reset.instruction, "all_targets", (None,)))
        r = run_episode(robot, seed, max_s=25.0 + 18.0 * n_targets, on_tick=lambda t, p: (rec_t.append(t), rec_p.append(p)))
        r["pair"] = f"{kind}:{bucket}"
        if a.dump:
            np.savez_compressed(Path(a.dump) / f"{seed}.npz", t=np.array(rec_t), obj_pos=np.array(rec_p), obj_names=np.array(r["placed_objects"]))
        out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("success", sum(r["success"] for r in out), "/", len(out))


# --- ROBOTCAM -------------------------------------------------------------------------------------------------
def robotcam_state(a) -> None:
    """S2: render the states a plugin grab saved (``grab``), offline, with the same compositor class, and compare."""
    from physicalai_mujoco_plugin.real2sim.desk_compositor import Compositor, DeskKit, _apply_sync
    src, out = Path(a.state), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    robot = make_robot(0); m, d = robot._sim.model, robot._sim.data; res = {}
    comp = Compositor(m, d, DeskKit.from_json(KIT), width=CAM_W, height=CAM_H)   # ONE per model: it rewrites the geom groups
    for cam in ("overview", "wrist"):
        st = np.load(src / f"{cam}_state.npz")
        sync = {k[5:]: st[k] for k in st.files if k.startswith("sync_")}
        d.qpos[:] = st["qpos"]; d.qvel[:] = 0.0; d.time = float(st["sim_t"]); _apply_sync(m, sync); mujoco.mj_forward(m, d)
        img = comp.render_camera(cam)["rgb"][:, :, ::-1]
        cv2.imwrite(str(out / f"{cam}_robotcam.png"), img)
        live = cv2.imread(str(src / f"{cam}.png")).astype(np.float64); mse = float(((live - img.astype(np.float64)) ** 2).mean())
        res[cam] = {"psnr_db": round(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse), 2), "max_abs_diff": int(np.abs(live - img).max()), "sim_t": float(st["sim_t"])}
        print(cam, res[cam], flush=True)
    json.dump(res, open(out / "s2_psnr.json", "w"), indent=1)


def cmd_grab(a) -> None:
    """S2: the plugin CLI (as PLUGIN_CLI, a subprocess) with R2S_GRAB_DIR; autopilot drive for --after s, then both
    streams' next frames are saved exactly as streamed with the exact states they were rendered from; the plugin is
    always shut down. Reproduce with ``robotcam --state DIR``."""
    import urllib.error, urllib.request
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"): f.unlink()
    env = dict(os.environ, PYTHONPATH=str(PLUGIN / "src"), PYTHONDONTWRITEBYTECODE="1", R2S_GRAB_DIR=str(out))
    p = subprocess.Popen([sys.executable, "-B", "-m", "physicalai_mujoco_plugin", "start", "--profile", "rebot_b601", "--scene", SCENE, "--no-gui",
                          "--seed", str(a.seed), "--http-port", str(a.port), "--name", f"mujoco-rebot_b601-grab{a.port}", "--appearance", str(KIT)],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{a.port}"
    try:
        t0 = time.time()
        while True:   # up, and a first composited frame of each camera served
            try:
                for cam in ("overview", "wrist"): urllib.request.urlopen(f"{url}/cameras/{cam}/frame.jpg", timeout=10).read()
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                if p.poll() is not None or time.time() - t0 > 180: raise SystemExit("GRAB FAILED: the plugin did not serve frames")
                time.sleep(1)
        urllib.request.urlopen(urllib.request.Request(url + "/autopilot", data=b'{"mode":"drive"}', headers={"content-type": "application/json"}), timeout=10).read()
        time.sleep(a.after)
        (out / "grab").touch(); t1 = time.time()
        while (out / "grab").exists():
            if time.time() - t1 > 30 or p.poll() is not None: raise SystemExit("GRAB FAILED: no frames saved")
            time.sleep(0.1)
        print("grabbed", sorted(f.name for f in out.iterdir()), flush=True)
    finally:
        try: urllib.request.urlopen(urllib.request.Request(url + "/shutdown", data=b"", method="POST"), timeout=10)
        except Exception: pass
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.terminate()
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill(); p.wait()


def cmd_robotcam(a) -> None:
    if a.state:
        robotcam_state(a); return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    robot = make_robot(a.seed, a.variant); reset(robot, a.seed); comp = compositor(robot); m, d = robot._sim.model, robot._sim.data
    encs = {n: subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{CAM_W}x{CAM_H}", "-r", str(CAM_FPS), "-i", "-",
                                 "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out / f"{n}.mp4")], stdin=subprocess.PIPE) for n in ("overview", "wrist")}
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator; demo._rng = np.random.default_rng(a.seed)
    count = robot._http_status()["episode"]["episode_count"]; t, k = 0.0, 0
    frames, poses, events, prev_phase = [], {"overview": [], "wrist": []}, {"grasp": [], "release": []}, None
    joints = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)] + [m.jnt_qposadr[m.joint("joint_left").id]]
    pending = []
    def save(kk, n, r):
        img = r["rgb"][:, :, ::-1]
        cv2.imwrite(str(out / f"{n}_{kk:05d}.png"), img); cv2.imwrite(str(out / f"{n}_{kk:05d}_seg.png"), r["seg"])
        cv2.imwrite(str(out / f"{n}_{kk:05d}_outline.png"), cv2.cvtColor(r["outline"], cv2.COLOR_RGBA2BGRA))
    while t < 45.0:
        tick(robot, demo); t += DT
        phase = demo.phase
        while k / CAM_FPS <= t:
            if phase != prev_phase and phase in ("close", "release"):
                events["grasp" if phase == "close" else "release"].append(k)
            prev_phase = phase
            ev = any(abs(k - e) <= 1 for es in events.values() for e in es) or (k + 1) in [e + 1 for es in events.values() for e in es]
            for n in ("overview", "wrist"):
                want = (k % 10 == 0) or ev
                r = comp.render_camera(n, outputs=want); encs[n].stdin.write(np.ascontiguousarray(r["rgb"][:, :, ::-1]).tobytes())
                if want: save(k, n, r)
                cam = m.camera(n).id
                poses[n].append(comp.cal.scene_to_camera(d.cam_xpos[cam], d.cam_xmat[cam].reshape(3, 3))[:3].tolist())
            frames.append({"k": k, "t": round(k / CAM_FPS, 4), "sim_time": round(float(d.time), 4), "joints": [float(d.qpos[j]) for j in joints]})
            k += 1
        if robot._http_status()["episode"]["episode_count"] != count or demo.status()["outcome"] in {"success", "failed", "stalled"}:
            # one more second after the outcome, for the release frames
            if not pending: pending = [t + 1.0]
            if t >= pending[0]: break
    # frame e+1 of the last events may not have been saved as an event frame: render nothing more (the episode ended)
    for e in encs.values(): e.stdin.close(); e.wait()
    task = robot._sim.on_reset
    json.dump({"fps": CAM_FPS, "size": [CAM_W, CAM_H], "frames": frames}, open(out / "frames.json", "w"))
    json.dump(events, open(out / "events.json", "w"), indent=1)
    json.dump({"convention": "OpenCV cam_from_world, SfM frame", "_metres_per_sfm_unit": comp.cal.scale, **poses}, open(out / "poses_sfm.json", "w"))
    json.dump({"seed": a.seed, "variant": a.variant, "instruction": task.instruction.as_dict(), "look": task.look,
               "bucket_shown": task.bucket_names, "objects": list(task.layout.objects), "pile": task.layout.pile,
               "outcome": demo.status()}, open(out / "episode.json", "w"), indent=1)
    print("frames", k, demo.status()["outcome"], "events", events)


# --- RAND -------------------------------------------------------------------------------------------------------
def cmd_rand(a) -> None:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); seeds = [int(s) for s in a.seeds.split(",")]
    from physicalai_mujoco_plugin.bucket_sort import LOOK_KNOBS, TABLE_LOOKS, TABLE_PATTERNS, BUCKET_PALETTE
    robot = make_robot(seeds[0], "rand"); comp = compositor(robot); tiles = []
    for seed in seeds:
        robot._automation.set_autopilot("off"); reset(robot, seed); task = robot._sim.on_reset
        shots = {"t0": {n: comp.render_camera(n)["rgb"] for n in ("overview", "wrist")}}
        demo = robot._automation.autopilot.demonstrator
        robot._automation.set_autopilot("drive"); demo._rng = np.random.default_rng(seed); t = 0.0
        while t < 30.0:
            tick(robot, demo); t += DT
            if demo.phase == "close": break
        shots["grasp"] = {n: comp.render_camera(n)["rgb"] for n in ("overview", "wrist")}
        for when, imgs in shots.items():
            for n, img in imgs.items(): cv2.imwrite(str(out / f"{seed}_{when}_{n}.png"), img[:, :, ::-1])
        json.dump({"seed": seed, "look": task.look, "bucket_shown": task.bucket_names, "buckets": {c: p.as_dict() for c, p in task.layout.buckets.items()},
                   "objects": list(task.layout.objects), "object_colors": task.layout.object_colors, "pile": task.layout.pile,
                   "instruction": task.instruction.as_dict(), "grasp_t": round(t, 2)}, open(out / f"{seed}_variant.json", "w"), indent=1)
        row = np.hstack([shots["t0"]["overview"], shots["t0"]["wrist"], shots["grasp"]["overview"], shots["grasp"]["wrist"]])[:, :, ::-1].copy()
        cv2.putText(row, f"seed {seed}: {task.instruction.text}", (8, 24), 0, 0.7, (255, 255, 255), 2); tiles.append(cv2.resize(row, (1280, 240)))
    cv2.imwrite(str(out / "rand_grid.jpg"), np.vstack(tiles))
    lay = layout_for("rand")
    json.dump({"look": LOOK_KNOBS, "table_looks": TABLE_LOOKS, "table_patterns": TABLE_PATTERNS, "bucket_palette": BUCKET_PALETTE,
               "real_bucket_probability": lay.real_bucket_probability, "row_jitter_m_m_rad": lay.row_jitter, "shuffle_row": lay.shuffle_row,
               "active_objects": lay.active_objects, "pile_probability": lay.pile_probability}, open(out / "knobs.json", "w"), indent=1)
    print("wrote", out / "rand_grid.jpg")


# --- SPEED, CAL, STABILITY --------------------------------------------------------------------------------------------
def cmd_speed(a) -> None:
    """r15 (R-50): NEW images per wall second, per camera; a call that returns no new frame is not counted.

    --path service (default, the recording path): the plugin as it streams to Studio. MuJoCoRobot with its default
    cameras and the kit, its camera thread polling the compositor, physics stepped at real time (one control tick per
    DT of wall time) with the demonstrator driving; new images are counted from each camera's frame buffer.
    --path loop: r14's single-threaded loop (physics, then one call per camera, as fast as it runs), counting new
    images only. With --seeds, one measurement per seed (each from its own reset)."""
    from physicalai_mujoco_plugin.real2sim.desk_compositor import DeskCameraCompositor
    seeds = [int(x) for x in str(a.seeds).split(",") if x.strip()]
    cams = ("overview", "wrist"); window = a.frames / CAM_FPS; rows = []
    if a.path == "service":
        from physicalai_mujoco_plugin.robot import MuJoCoRobot
        robot = MuJoCoRobot("rebot_b601", scene=SCENE, cameras=None, http_port=0, seed=seeds[0], appearance=str(KIT))
        robot.connect(); robot._automation.set_autopilot("off"); robot._sim.on_reset.area = layout_for("default")
        bufs = robot._cameras.frame_buffers
        for seed in seeds:
            robot._automation.set_autopilot("off"); reset(robot, seed); task = robot._sim.on_reset
            pile = bool(getattr(task.layout, "pile", False)); count = len(task.layout.objects)
            robot._automation.set_autopilot("drive")
            def run(seconds):
                t0 = time.perf_counter(); k = 0; sim0 = float(robot._sim.data.time)
                while time.perf_counter() - t0 < seconds:
                    robot.get_observation(); k += 1
                    lag = t0 + k * DT - time.perf_counter()
                    if lag > 0: time.sleep(lag)
                return (float(robot._sim.data.time) - sim0) / (time.perf_counter() - t0)
            w0 = time.perf_counter()   # warm-up: the camera processes start (first time) and frames of this episode arrive
            first = {c: bufs[c]._seq for c in cams}
            while any(bufs[c]._seq <= first[c] + 2 for c in cams if c in bufs) or not all(c in bufs for c in cams):
                run(0.5)
                if time.perf_counter() - w0 > 240: raise RuntimeError("no camera frames within 240 s")
            seq0 = {c: bufs[c]._seq for c in cams}; t0 = time.perf_counter(); rtf = run(window); el = time.perf_counter() - t0
            fps = {c: (bufs[c]._seq - seq0[c]) / el for c in cams}
            rows.append((seed, pile, count, fps))
            if os.environ.get("R2S_TIMING"):
                from physicalai_mujoco_plugin.real2sim import desk_compositor as _dcm
                for c in cams:
                    v = _dcm._TIMES.pop(("free", c), [])
                    ww = _dcm._TIMES.pop(("free_w", c), []); tr = _dcm._TIMES.pop(("free_try", c), [])
                    if ww: print(f"  {c}: jobs per worker {np.bincount(ww).tolist()}; start attempts {len(tr)}, of which no idle worker {sum(1 for x in tr if x < 0)}", flush=True)
                    g = _dcm._TIMES.pop(("free_gate", c), None); ks_ = _dcm._TIMES.pop(("free_keys", c), [0])[0]
                    if g: print(f"  {c}: polls {g[2]}, distinct states {ks_} ({ks_ / el:.1f}/s), polls with a new state {g[0]}, of which held by the stagger gap {g[1]}", flush=True)
                    gp = _dcm._TIMES.pop(("free_gap", c), [])
                    if gp: print(f"  {c}: stagger gap {1000 * np.median(gp):.1f} ms median, p90 {1000 * np.percentile(gp, 90):.1f}", flush=True)
                    cl = _dcm._TIMES.pop(("call", c), [])
                    if cl: print(f"  {c}: {len(cl) / el:.1f} compositor calls/s, state copy {1000 * np.median(cl):.2f} ms median, p90 {1000 * np.percentile(cl, 90):.2f}", flush=True)
                    if v: print(f"  {c}: render job {1000 * np.median(v):.1f} ms median, p90 {1000 * np.percentile(v, 90):.1f} ({len(v)} jobs)", flush=True)
            print(f"SPEED seed {seed} ({'pile' if pile else 'scattered'}, {count} objects), service path: new images/s "
                  + ", ".join(f"{c} {fps[c]:.2f}" for c in cams) + f" (physics {rtf:.2f} x real time)", flush=True)
        robot.disconnect()
    else:
        robot = make_robot(seeds[0]); reset(robot, seeds[0]); m, d = robot._sim.model, robot._sim.data
        dc = DeskCameraCompositor(kit(), m, str(KIT))
        class R: height, width = CAM_H, CAM_W
        for seed in seeds:
            robot._automation.set_autopilot("off"); reset(robot, seed); task = robot._sim.on_reset
            pile = bool(getattr(task.layout, "pile", False)); count = len(task.layout.objects)
            robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator
            n, t0, new, calls = 0, None, {c: 0 for c in cams}, 0
            while n < a.frames + 10:
                for _ in range(max(1, int(round(1 / CAM_FPS / DT)))): tick(robot, demo)
                for c in cams:
                    got = dc(R, d, c, None) is not None
                    if t0 is not None: new[c] += got
                n += 1
                if n == 10: t0 = time.time()
                elif t0 is not None: calls += 1
            el = time.time() - t0; fps = {c: new[c] / el for c in cams}; rows.append((seed, pile, count, fps))
            print(f"SPEED seed {seed} ({'pile' if pile else 'scattered'}, {count} objects), loop path: new images/s "
                  + ", ".join(f"{c} {fps[c]:.2f}" for c in cams) + f" ({calls / el:.2f} loop iterations/s)", flush=True)
    for name, sel in (("piles", [r for r in rows if r[1]]), ("scattered", [r for r in rows if not r[1]])):
        if sel: print(f"SPEED {a.path} {name}: median new images/s " + ", ".join(f"{c} {np.median([r[3][c] for r in sel]):.2f}" for c in cams)
                      + f" over {len(sel)} seed(s) (min pair {min(min(r[3].values()) for r in sel):.2f}; {CAM_W}x{CAM_H})")


def cmd_cal(a) -> None:
    print(json.dumps(calibration(), indent=1))


def cmd_stability(a) -> None:
    """S4: the plugin CLI as a subprocess, autopilot drive, N episodes; no crash, no hang, RSS growth. A camera snapshot
    may answer 503 (no frame yet) while the cameras restart at an episode reset: transient HTTP errors are retried for
    up to 30 s and counted. The simulation is always shut down at the end, also when the check fails."""
    import urllib.error, urllib.request
    env = dict(os.environ, PYTHONPATH=str(PLUGIN / "src"), PYTHONDONTWRITEBYTECODE="1")
    log = open(Path(a.out) if a.out else os.devnull, "w")
    p = subprocess.Popen([sys.executable, "-B", "-m", "physicalai_mujoco_plugin", "start", "--profile", "rebot_b601", "--scene", SCENE, "--no-gui",
                          "--seed", "1", "--http-port", str(a.port), "--appearance", str(KIT)], env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{a.port}"; transient = {"count": 0}

    def fetch(path, data=None, *, patience=30.0):
        """GET (or POST with data) with retries over transient errors; fails if the process died or after *patience* s."""
        t_end = time.time() + patience
        while True:
            try:
                req = urllib.request.Request(url + path, data=data, headers={"content-type": "application/json"}, method="POST" if data is not None else "GET")
                return urllib.request.urlopen(req, timeout=10).read()
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                transient["count"] += 1
                if p.poll() is not None: raise SystemExit(f"STABILITY FAILED: the plugin exited ({p.returncode}) during {path}")
                if time.time() > t_end: raise SystemExit(f"STABILITY FAILED: {path} kept failing for {patience:.0f} s ({exc})")
                time.sleep(0.5)

    def rss():   # the whole process tree: the launcher, the simulation owner and its compositor processes (MB)
        tree, frontier = [p.pid], [p.pid]
        while frontier:
            kids = [int(k) for pid in frontier for k in subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()]
            tree += kids; frontier = kids
        out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(map(str, tree))], capture_output=True, text=True).stdout.split()
        return sum(int(v) for v in out) / 1024
    try:
        t0 = time.time(); fetch("/health", patience=120); print(f"started in {time.time() - t0:.1f} s", flush=True)
        transient["count"] = 0
        fetch("/autopilot", b'{"mode":"drive"}')
        start_count, rss0, t1 = json.loads(fetch("/health"))["episode"]["episode_count"], rss(), time.time()
        last, t_progress = start_count, time.time()
        while last - start_count < a.episodes:
            if p.poll() is not None: raise SystemExit("STABILITY FAILED: crashed")
            if time.time() - t_progress > 120: raise SystemExit("STABILITY FAILED: hang (no new episode for 120 s)")
            time.sleep(2)
            count = json.loads(fetch("/health"))["episode"]["episode_count"]
            if count != last: last, t_progress = count, time.time()
            fetch("/cameras/overview/frame.jpg")
        print(f"STABILITY {a.episodes} episodes in {time.time() - t1:.0f} s, RSS of the whole process tree (launcher, owner, compositor processes) {rss0:.0f} -> {rss():.0f} MB, "
              f"{transient['count']} transient HTTP errors retried", flush=True)
    finally:   # never leave the simulation running
        try: urllib.request.urlopen(urllib.request.Request(url + "/shutdown", data=b"", method="POST"), timeout=10)
        except Exception: pass
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.terminate()
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill(); p.wait()


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("heldout_render"); p.add_argument("--frames", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("video"); p.add_argument("--t0", type=float, default=3.0); p.add_argument("--t1", type=float, default=154.0)
    p.add_argument("--seed", type=int, default=7); p.add_argument("--sbs", action="store_true"); p.add_argument("--out", required=True)
    p = sub.add_parser("replay"); p.add_argument("--states", required=True); p.add_argument("--times", required=True); p.add_argument("--out", required=True)
    p.add_argument("--video-time", type=float, default=None, help="phone-capture time of the bucket arrangement (default: its start)")
    p.add_argument("--latency", type=float, default=None, help="camera latency in s (default: the kit's camera_latency_s)")
    p.add_argument("--contents-mask", action="store_true", help="builder's diagnostic: also write <stem>_<camera>_contents.png")
    p.add_argument("--layout", default="episode", help="FILE: the poses of a FIT_LAYOUT output (r12, U-5) for every time; episode (default): per bucket, the median of the allowed fits in the bucket's segment (r12, U-4: red {0},{1-5},{6-10}; green, black {0-3},{4-10}; blue one pool); matched: the phone-matched layout (r09)")
    p = sub.add_parser("fit_layout"); p.add_argument("--video", required=True); p.add_argument("--t0", type=float, required=True); p.add_argument("--t1", type=float, required=True)
    p.add_argument("--states", required=True, help="the episode's rows (as ROBOTCAM_REPLAY's --states): the arm's joints for its mask"); p.add_argument("--episode", type=int)
    p.add_argument("--start", type=float, default=0.0, help="episode time of the video's first frame (s)"); p.add_argument("--out", required=True)
    p = sub.add_parser("kinds"); p.add_argument("--start", type=int, default=1000); p.add_argument("--count", type=int, default=10)
    p.add_argument("--variant", default="default", choices=["default", "rand", "record"]); p.add_argument("--out", required=True)
    p = sub.add_parser("demo"); p.add_argument("--seeds", required=True); p.add_argument("--pairs", required=True)
    p.add_argument("--variant", default="default", choices=["default", "rand"]); p.add_argument("--out", required=True); p.add_argument("--dump")
    p = sub.add_parser("grab"); p.add_argument("--seed", type=int, default=1); p.add_argument("--port", type=int, default=8942); p.add_argument("--after", type=float, default=6.0); p.add_argument("--out", required=True)
    p = sub.add_parser("robotcam"); p.add_argument("--state", help="reproduce a plugin grab (S2) instead of running an episode"); p.add_argument("--seed", type=int, default=0); p.add_argument("--variant", default="default", choices=["default", "rand"]); p.add_argument("--out", required=True)
    p = sub.add_parser("speed"); p.add_argument("--frames", type=int, default=300); p.add_argument("--seeds", default="1"); p.add_argument("--path", default="service", choices=("service", "loop"))
    p = sub.add_parser("rand"); p.add_argument("--seeds", default="1,2,3,4,5,6"); p.add_argument("--out", required=True)
    p = sub.add_parser("cal")
    p = sub.add_parser("stability"); p.add_argument("--episodes", type=int, default=10); p.add_argument("--port", type=int, default=8941); p.add_argument("--out")
    a = ap.parse_args()
    {"heldout_render": cmd_heldout_render, "video": cmd_video, "demo": cmd_demo, "robotcam": cmd_robotcam, "speed": cmd_speed, "kinds": cmd_kinds, "replay": cmd_replay, "fit_layout": cmd_fit_layout,
     "rand": cmd_rand, "cal": cmd_cal, "stability": cmd_stability, "grab": cmd_grab}[a.cmd](a)


if __name__ == "__main__":
    main()
