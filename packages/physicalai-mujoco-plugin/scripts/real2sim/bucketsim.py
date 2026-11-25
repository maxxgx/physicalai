"""bucket_sort_desk real2sim: the MANIFEST commands (review/RUBRIC.md, section 3).

  bucketsim.py heldout_render --frames f00001,f00002 --out DIR     matched state at registered phone poses
  bucketsim.py video [--t0 0 --t1 75.3] [--seed 7] [--sbs] --out FILE
  bucketsim.py demo --seeds 1,2 --pairs cpu:green,resistor:blue --out JSON
  bucketsim.py robotcam --seed S --out DIR                       overview.mp4, wrist.mp4, every 10th frame as PNG + _seg
  bucketsim.py speed [--frames 300]
  bucketsim.py rand --seeds 1,2,3,4,5,6 --out DIR
  bucketsim.py cal

Run with REAL2SIM_ROOT set to the project folder: python -I scripts/real2sim/bucketsim.py ... (see a delivery's MANIFEST)"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

import cv2
import mujoco
import numpy as np

# this checkout's plugin first (a venv may have another checkout's plugin installed in editable mode); a delivery's
# frozen copy when run from one
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PLUGIN, ROOT, held_t, pinhole_cameras  # noqa: E402
sys.path.insert(0, str(PLUGIN / "src"))
from compositor import COLORS, Calibration, Compositor, calibration  # noqa: E402

CAM_W, CAM_H, CAM_FPS = 640, 480, 30
"""Robot-camera streams: the policy's dataset config is not known yet; this is the assumption (D7 X6)."""
WRIST_FOVY = 58.0


def make_robot(seed: int):
    from physicalai_mujoco_plugin.robot import MuJoCoRobot
    robot = MuJoCoRobot("rebot_b601", scene="bucket_sort_desk", cameras=[], http_port=0, seed=seed)
    robot.connect(); robot._automation.set_autopilot("off")
    return robot


def reset(robot, seed: int, pair: tuple[str, str] | None = None) -> None:
    task = robot._sim.on_reset
    task.set_allowed_pairs([pair] if pair else None)
    robot._set_seed(seed); robot._reset(); robot._set_seed(None)


def matched_state(robot) -> None:
    """The scene as in the video: the arm at the calibrated static joints, objects off the desk, buckets as measured."""
    cal = calibration(); m, d = robot._sim.model, robot._sim.data
    from physicalai_mujoco_plugin.bucket_sort import set_bucket_fill
    for i, v in enumerate(cal.data["robot"]["joints_rad"]):
        d.qpos[m.jnt_qposadr[m.joint(f"joint{i + 1}").id]] = v
    for n in ("joint_left", "joint_right"):
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = cal.data["robot"].get("finger_opening_m", 0.0)
    for c in COLORS:
        b = cal.data["buckets"][c]; adr = m.jnt_qposadr[m.joint(f"bucket_{c}:joint").id]
        d.qpos[adr:adr + 3] = [b["x"], b["y"], 0.0]; d.qpos[adr + 3:adr + 7] = [np.cos(b["yaw"] / 2), 0, 0, np.sin(b["yaw"] / 2)]
    for k, n in enumerate(("resistor", "battery", "cpu")):  # parked under the desk, out of every view
        adr = m.jnt_qposadr[m.joint(f"{n}:joint").id]; d.qpos[adr:adr + 3] = [0.3 + 0.1 * k, 0.0, -0.5]
    set_bucket_fill(m, {c: 1.0 for c in COLORS})
    mujoco.mj_forward(m, d)


# --- HELDOUT_RENDER ------------------------------------------------------------------------------------
def cmd_heldout_render(a) -> None:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cams = pinhole_cameras(); names = [n if n.endswith(".jpg") else n + ".jpg" for n in a.frames.split(",")]
    robot = make_robot(0); matched_state(robot)
    T0, K0, W, H = cams[names[0]]
    comp = Compositor(robot._sim.model, robot._sim.data, calibration(), width=W, height=H, show_objects=False)
    meta = {}
    for n in names:
        T, K, w, h = cams[n]; assert (w, h) == (W, H)
        r = comp.render_sfm(T, K, outputs=True); stem = Path(n).stem
        cv2.imwrite(str(out / f"{stem}.png"), r["rgb"][:, :, ::-1]); cv2.imwrite(str(out / f"{stem}_seg.png"), r["seg"])
        cv2.imwrite(str(out / f"{stem}_outline.png"), cv2.cvtColor(r["outline"], cv2.COLOR_RGBA2BGRA))
        meta[stem] = {"width": w, "height": h, "K": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], "cam_from_world": T[:3].tolist()}
        print("wrote", stem, flush=True)
    json.dump(meta, open(out / "cameras.json", "w"), indent=1)



def tick(robot, demo) -> None:
    """One control tick; while the demonstrator plans (on its own thread), wait without stepping the sim, so a run
    depends on the seed only, not on how long planning takes on this machine."""
    robot.get_observation()
    while demo.planner_running:   # the finished plan is picked up by the next tick
        time.sleep(0.0005)

# --- the demonstrator over time -------------------------------------------------------------------------
def run_demo(robot, seconds: float, fps: float, on_frame=None) -> list[dict]:
    """Episodes back to back for *seconds* of sim time; calls on_frame(k) every 1/fps s. Returns the events."""
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator
    t, k, events, last = 0.0, 0, [], None
    while t < seconds:
        tick(robot, demo)
        t += 0.02
        while k / fps <= t and k / fps < seconds:
            if on_frame: on_frame(k)
            k += 1
        ep = robot._http_status()["episode"]; key = ep["episode_count"]
        if key != last:
            events.append({"t": round(t, 2), "episode": key, "instruction": (ep.get("instruction") or {}).get("text")}); last = key
    return events


# --- VIDEO ------------------------------------------------------------------------------------------------
def cmd_video(a) -> None:
    from scipy.spatial.transform import Rotation, Slerp
    cams = pinhole_cameras(); frames = json.load(open(ROOT / "colmap/frames.json"))
    t_of = {f"f{f['idx']:05d}.jpg": f["t"] for f in frames["frames"]}
    reg = sorted((t_of[n], n) for n in cams if n in t_of); ts = np.array([t for t, _ in reg])
    C = np.array([-cams[n][0][:3, :3].T @ cams[n][0][:3, 3] for _, n in reg])
    slerp = Slerp(ts, Rotation.from_matrix(np.array([cams[n][0][:3, :3].T for _, n in reg])))
    def pose_at(t):
        t = float(np.clip(t, ts[0], ts[-1])); c = np.array([np.interp(t, ts, C[:, i]) for i in range(3)])
        R = slerp([t]).as_matrix()[0]; T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ c; return T
    _, K0, W0, H0 = cams[reg[0][1]]
    W, H = (W0, H0) if a.full else (1280, int(round(H0 * 1280 / W0)) // 2 * 2)
    K = K0.copy(); K[0] *= W / W0; K[1] *= H / H0
    robot = make_robot(a.seed); reset(robot, a.seed)
    comp = Compositor(robot._sim.model, robot._sim.data, calibration(), width=W, height=H)
    fps, k0, k1 = 15.0, int(round(a.t0 * 15)), int(round(a.t1 * 15))
    # real frames (side by side), undistorted from the SfM's SIMPLE_RADIAL to its PINHOLE model; black in held-out time
    dec = None
    if a.sbs:
        k1r, f0 = 0.05045002030184394, 1684.0649756089904
        yy, xx = np.mgrid[0:H0, 0:W0].astype(np.float32); xn, yn = (xx - K0[0, 2]) / K0[0, 0], (yy - K0[1, 2]) / K0[1, 1]
        r2 = xn ** 2 + yn ** 2; mx = (f0 * xn * (1 + k1r * r2) + 960).astype(np.float32); my = (f0 * yn * (1 + k1r * r2) + 540).astype(np.float32)
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{k0 / fps:.4f}", "-i", str(ROOT / "video/sdr_4k.mov"), "-vf", "fps=15,scale=1920:1080",
                                "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{(2 if a.sbs else 1) * W}x{H}", "-r", "15",
                            "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    t0 = time.time()
    def frame(k):
        if k < k0: return
        img = comp.render_sfm(pose_at(k / fps), K)["rgb"][:, :, ::-1]
        if dec is not None:
            raw = dec.stdout.read(1920 * 1080 * 3); real = np.zeros_like(img)
            if len(raw) == 1920 * 1080 * 3 and not held_t(k / fps):
                real = cv2.resize(cv2.remap(np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3), mx, my, cv2.INTER_LINEAR), (W, H), interpolation=cv2.INTER_AREA)
            img = np.hstack([real, img])
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
        if k % 150 == 0: print(f"t {k / fps:.0f}s, {(time.time() - t0) / max(1, k - k0 + 1):.2f} s/frame", flush=True)
    events = run_demo(robot, a.t1, fps, frame)
    enc.stdin.close(); enc.wait()
    json.dump(events, open(str(a.out) + ".events.json", "w"), indent=1); print("wrote", a.out)


# --- DEMO -------------------------------------------------------------------------------------------------
def physics_flags(m, d) -> list[str]:
    flags = []
    for e in range(m.neq):
        if d.eq_active[e]:
            j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]; kind = mujoco.mjtEq(m.eq_type[e]).name
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT if kind == "mjEQ_JOINT" else mujoco.mjtObj.mjOBJ_BODY, i) for i in (j1, j2)]
            flags.append(f"{kind} {names[0]}-{names[1]} (robot model)" if all(n and n.startswith('joint_') for n in names) else f"{kind} {names}")
    if m.nmocap: flags.append(f"{m.nmocap} mocap bodies")
    return flags


def cmd_demo(a) -> None:
    seeds = [int(s) for s in a.seeds.split(",")]; pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
    assert len(seeds) == len(pairs), "one pair per seed"
    robot = make_robot(seeds[0]); m, d = robot._sim.model, robot._sim.data; out = []
    for seed, (obj, bucket) in zip(seeds, pairs):
        robot._automation.set_autopilot("off"); reset(robot, seed, (obj, bucket))
        demo_qpos_writes = []
        robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator; demo._rng = np.random.default_rng(seed)
        count = robot._http_status()["episode"]["episode_count"]; ok, t, wall, held_since = False, 0.0, time.time(), None
        objs = {n: (m.body(n).id, m.jnt_dofadr[m.joint(f"{n}:joint").id]) for n in ("resistor", "battery", "cpu")}
        prev = {n: d.xpos[b].copy() for n, (b, _) in objs.items()}; vprev = dict.fromkeys(objs, 0.0)
        while t < 45.0 and time.time() - wall < 600:
            tick(robot, demo)
            t += 0.02
            if robot._http_status()["episode"]["episode_count"] != count: break   # the next episode's reset moved them
            for n, (b, dof) in objs.items():   # a teleport: moved further in a tick than its speed allows
                step = float(np.linalg.norm(d.xpos[b] - prev[n])); v = max(vprev[n], float(np.linalg.norm(d.qvel[dof:dof + 3])))
                vprev[n] = float(np.linalg.norm(d.qvel[dof:dof + 3]))
                if step > 0.005 + 2 * v * 0.02 and f"teleport {n}" not in demo_qpos_writes:
                    demo_qpos_writes.append(f"teleport {n}"); print(f"  jump {n} t={t:.2f} step={step * 1000:.1f}mm v={v:.3f} z={d.xpos[b][2]:.3f}", flush=True)
                prev[n] = d.xpos[b].copy()
            ep = robot._http_status()["episode"]
            if ep["episode_count"] != count: break
            # success: the task's own test (the named object resting in the named bucket, out of the gripper) holding
            # for a full second of sim time; the demonstrator's own report does not count
            if ep["success"] and ep["in_bucket"][obj] == bucket:
                held_since = t if held_since is None else held_since
                if t - held_since >= 1.0: ok = True; break
            else:
                held_since = None
            if demo.status()["outcome"] in {"failed", "stalled"}: break
        st = demo.status()
        out.append({"seed": seed, "pair": f"{obj}:{bucket}", "success": ok, "failure": None if ok else (st["failure"] or "no settled success in 45 s"),
                    "sim_s": round(t, 2), "physics_flags": physics_flags(m, d) + demo_qpos_writes})
        print(json.dumps(out[-1]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("success", sum(r["success"] for r in out), "/", len(out))


# --- ROBOTCAM -------------------------------------------------------------------------------------------
def robot_cam_compositors(robot):
    m, d = robot._sim.model, robot._sim.data; cal = calibration()
    comp = Compositor(m, d, cal, width=CAM_W, height=CAM_H)
    return comp


def cmd_robotcam(a) -> None:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    robot = make_robot(a.seed); reset(robot, a.seed); comp = robot_cam_compositors(robot)
    encs = {n: subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{CAM_W}x{CAM_H}", "-r", str(CAM_FPS), "-i", "-",
                                 "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out / f"{n}.mp4")], stdin=subprocess.PIPE) for n in ("overview", "wrist")}
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator; demo._rng = np.random.default_rng(a.seed)
    count = robot._http_status()["episode"]["episode_count"]; t, k = 0.0, 0
    while t < 45.0:
        tick(robot, demo)
        t += 0.02
        while k / CAM_FPS <= t:
            for n in ("overview", "wrist"):
                r = comp.render_camera(n, outputs=(k % 10 == 0)); img = r["rgb"][:, :, ::-1]
                encs[n].stdin.write(np.ascontiguousarray(img).tobytes())
                if k % 10 == 0:
                    cv2.imwrite(str(out / f"{n}_{k:05d}.png"), img); cv2.imwrite(str(out / f"{n}_{k:05d}_seg.png"), r["seg"])
            k += 1
        if robot._http_status()["episode"]["episode_count"] != count or demo.status()["outcome"] in {"success", "failed", "stalled"}:
            break
    for e in encs.values(): e.stdin.close(); e.wait()
    json.dump({"frames": k, "fps": CAM_FPS, "size": [CAM_W, CAM_H], "outcome": demo.status()}, open(out / "robotcam.json", "w"), indent=1)
    print("frames", k, demo.status()["outcome"])


# --- SPEED ----------------------------------------------------------------------------------------------
def cmd_speed(a) -> None:
    robot = make_robot(1); reset(robot, 1); comp = robot_cam_compositors(robot)
    robot._automation.set_autopilot("drive"); n, t0 = 0, None
    while n < a.frames + 10:
        for _ in range(int(round(1 / CAM_FPS / 0.02)) or 1): tick(robot, robot._automation.autopilot.demonstrator)
        comp.render_camera("overview"); comp.render_camera("wrist"); n += 1
        if n == 10: t0 = time.time()
    print(f"SPEED {a.frames / (time.time() - t0):.2f} frames/s (overview + wrist, {CAM_W}x{CAM_H}, physics and demonstrator running)")


# --- RAND -----------------------------------------------------------------------------------------------
KNOBS = {
    "light_intensity": {"range": [0.7, 1.3], "default": 1.0, "what": "sim light diffuse scale (arm, objects, shadow strength)"},
    "light_kelvin": {"range": [3500, 6500], "default": 5000, "what": "sim light colour temperature"},
    "light_direction_deg": {"range": [0, 25], "default": 0, "what": "shadow light tilt from vertical, random azimuth"},
    "overview_jitter": {"range": {"pos_m": 0.02, "rot_deg": 3.0}, "default": 0, "what": "overview camera extrinsics"},
    "wrist_jitter": {"range": {"pos_m": 0.003, "rot_deg": 2.0}, "default": 0, "what": "wrist camera mount"},
    "objects": {"range": "SIX_JOINT_LAYOUT object area (at least 0.29 m from the base), any heading", "default": "per seed", "what": "task object poses (TaskLayout)"},
    "object_count": {"range": [1, 3], "default": 3, "what": "objects on the desk: the named one plus 0-2 others (the rest parked off the desk)"},
    "bucket_row": {"range": {"pos_m": 0.015, "yaw_deg": 6.0}, "default": "measured row", "what": "bucket poses, contents move with them (REBOT_DESK_LAYOUT)"},
    "bucket_fill": {"range": [0.0, 1.0], "default": "per seed", "what": "fraction of each bucket's contents shown (TaskLayout.fill)"},
    "exposure_ev": {"range": [-0.4, 0.4], "default": 0.0, "what": "image exposure"},
    "white_balance": {"range": [0.92, 1.08], "default": 1.0, "what": "per-channel gain on R and B"},
    "noise_sigma": {"range": [0, 4], "default": 0, "what": "Gaussian sensor noise, /255"},
}


def kelvin_rgb(k: float) -> np.ndarray:
    """Rough blackbody colour (Tanner Helland's fit), normalised to its maximum."""
    t = k / 100.0
    r = 255.0 if t <= 66 else 329.7 * (t - 60) ** -0.1332
    g = 99.47 * np.log(t) - 161.12 if t <= 66 else 288.12 * (t - 60) ** -0.0755
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.52 * np.log(t - 10) - 305.04)
    c = np.clip([r, g, b], 0, 255); return c / c.max()


def image_aug(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ev = rng.uniform(*KNOBS["exposure_ev"]["range"]); wb = rng.uniform(*KNOBS["white_balance"]["range"], 2)
    x = img.astype(np.float32) * 2 ** ev; x[..., 0] *= wb[0]; x[..., 2] *= wb[1]
    x += rng.normal(0, rng.uniform(*KNOBS["noise_sigma"]["range"]), x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def jitter_body(m, body: int, pos_m: float, rot_deg: float, rng: np.random.Generator) -> None:
    from scipy.spatial.transform import Rotation
    m.body_pos[body] += rng.uniform(-pos_m, pos_m, 3)
    q = m.body_quat[body]; r = Rotation.from_quat([q[1], q[2], q[3], q[0]]) * Rotation.from_rotvec(np.radians(rng.uniform(-rot_deg, rot_deg, 3)))
    x, y, z, w = r.as_quat(); m.body_quat[body] = [w, x, y, z]


def cmd_rand(a) -> None:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); seeds = [int(s) for s in a.seeds.split(",")]
    robot = make_robot(seeds[0]); m, d = robot._sim.model, robot._sim.data; comp = robot_cam_compositors(robot)
    light0 = (m.light_diffuse.copy(), m.light_pos.copy(), m.light_dir.copy()); rig = m.body("overview_camera_rig").id
    wrist = m.camera("wrist").id; cam0 = (m.body_pos[rig].copy(), m.body_quat[rig].copy(), m.cam_pos[wrist].copy(), m.cam_quat[wrist].copy())
    tiles = []
    for seed in seeds:
        rng = np.random.default_rng(seed); reset(robot, seed)
        m.light_diffuse[:], m.light_pos[:], m.light_dir[:] = light0
        m.light_diffuse[:] = light0[0] * rng.uniform(*KNOBS["light_intensity"]["range"]) * kelvin_rgb(rng.uniform(*KNOBS["light_kelvin"]["range"]))
        tilt, az = np.radians(rng.uniform(*KNOBS["light_direction_deg"]["range"])), rng.uniform(0, 2 * np.pi)
        m.light_dir[0] = [np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)]
        m.body_pos[rig], m.body_quat[rig] = cam0[0].copy(), cam0[1].copy()
        jitter_body(m, rig, KNOBS["overview_jitter"]["range"]["pos_m"], KNOBS["overview_jitter"]["range"]["rot_deg"], rng)
        m.cam_pos[wrist] = cam0[2] + rng.uniform(-KNOBS["wrist_jitter"]["range"]["pos_m"], KNOBS["wrist_jitter"]["range"]["pos_m"], 3)
        from scipy.spatial.transform import Rotation
        q = cam0[3]; r = Rotation.from_quat([q[1], q[2], q[3], q[0]]) * Rotation.from_rotvec(np.radians(rng.uniform(-1, 1, 3) * KNOBS["wrist_jitter"]["range"]["rot_deg"]))
        x, y, z, w = r.as_quat(); m.cam_quat[wrist] = [w, x, y, z]
        # object count: the named object always, 0-2 of the others parked under the desk
        ins = robot._sim.on_reset.instruction; others = [n for n in ("resistor", "battery", "cpu") if n != ins.object]
        for n in rng.permutation(others)[:int(rng.integers(0, 3))]:
            adr = m.jnt_qposadr[m.joint(f"{n}:joint").id]; d.qpos[adr:adr + 3] = [0.3, 0.0, -0.5]
        mujoco.mj_forward(m, d)
        row = [image_aug(comp.render_camera(n)["rgb"], rng) for n in ("overview", "wrist")]
        img = np.hstack(row)[:, :, ::-1].copy(); cv2.putText(img, f"seed {seed}", (8, 22), 0, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"rand_{seed}.png"), img); tiles.append(cv2.resize(img, (640, 240)))
    cv2.imwrite(str(out / "rand_grid.jpg"), np.vstack(tiles)); json.dump(KNOBS, open(out / "knobs.json", "w"), indent=1)
    print("wrote", out / "rand_grid.jpg")


# --- CAL ----------------------------------------------------------------------------------------------------
def cmd_cal(a) -> None:
    print(json.dumps(json.load(open(ROOT / "calibration.json")), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("heldout_render"); p.add_argument("--frames", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("video"); p.add_argument("--t0", type=float, default=0.0); p.add_argument("--t1", type=float, default=75.3)
    p.add_argument("--seed", type=int, default=7); p.add_argument("--sbs", action="store_true"); p.add_argument("--full", action="store_true"); p.add_argument("--out", required=True)
    p = sub.add_parser("demo"); p.add_argument("--seeds", required=True); p.add_argument("--pairs", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("robotcam"); p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("speed"); p.add_argument("--frames", type=int, default=300)
    p = sub.add_parser("rand"); p.add_argument("--seeds", default="1,2,3,4,5,6"); p.add_argument("--out", required=True)
    p = sub.add_parser("cal")
    a = ap.parse_args()
    {"heldout_render": cmd_heldout_render, "video": cmd_video, "demo": cmd_demo, "robotcam": cmd_robotcam, "speed": cmd_speed, "rand": cmd_rand, "cal": cmd_cal}[a.cmd](a)


if __name__ == "__main__":
    main()
