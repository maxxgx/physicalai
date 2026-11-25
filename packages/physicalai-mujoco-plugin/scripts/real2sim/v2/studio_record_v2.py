"""Record bucket_sort_desk_v2 episodes through Physical AI Studio's own recording pipeline (STUDIO_RECORD).

Studio (upstream, runtime-session protocol) drives the simulated reBot follower by teleoperation from the generic
MuJoCo virtual leader; the plugin's autopilot runs in ``leader`` mode, so the demonstrator's commands are the leader
(and the dataset's actions: the gripper closes past the object). The script:
  1. creates (or reuses) a Studio project, the follower (MuJoCo_reBotB601_Follower), the leader (MuJoCo_Virtual_Leader),
     two IP cameras on the plugin's composited MJPEG streams (overview, gripper = the plugin's wrist stream, named as on the real dataset; 640x480), an environment and a
     dataset (REST);
  2. opens the project's runtime session (websocket): load_environment, set_control teleop, load_dataset at --fps;
  3. switches the plugin's autopilot to ``leader`` and lets the plugin reset episodes itself after each success
     (POST /episode/auto-reset; a success may go on from the same table: a sorting sequence);
  4. per plugin episode (a new ``reset_count``): start_episode with the episode's worded instruction as the task; on the
     demonstrator's ``success`` and the task's success held 1 s: save_episode (clean ones only unless --keep all); on
     ``failed``/``stalled``: discard_episode and POST /reset;
  5. after N saved episodes: unload the dataset (Studio copies the episodes into it) and print the dataset's folder.
usage: studio_record_v2.py --episodes N [--studio http://127.0.0.1:7871] [--sim http://127.0.0.1:8931]
       [--project bucket_sort_sim_v2] [--dataset NAME] [--fps 30] [--seed S] [--keep perfect|all] [--max-tries M]"""
from __future__ import annotations

import argparse, json, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, required=True); ap.add_argument("--studio", default="http://127.0.0.1:7871")
ap.add_argument("--sim", default="http://127.0.0.1:8931"); ap.add_argument("--project", default="bucket_sort_sim_v2")
ap.add_argument("--dataset", default=None); ap.add_argument("--fps", type=int, default=30); ap.add_argument("--seed", type=int)
ap.add_argument("--keep", default="perfect", choices=["perfect", "all"]); ap.add_argument("--max-tries", type=int, default=0)
ap.add_argument("--start-settle", type=float, default=0.5, help="seconds after a reset before start_episode, so the streams show the new scene (r13)")
ap.add_argument("--camera-base", default=None, help="base URL of the camera streams (default: the simulation's); r14 test: a fixed-rate fake source")
ap.add_argument("--owner", default="mujoco-rebot_b601-follow"); ap.add_argument("--log", default=None)
ap.add_argument("--no-plan-before-record", dest="plan_before_record", action="store_false",
                help="r16: by default the demonstrator plans each episode's first attempt BEFORE the Studio episode starts "
                     "(the simulation's start gate): the recording starts at 'ready', then the go, the real operator's start "
                     "pause and the motion; planning time is not recorded. A try that fails before 'ready' counts as a try.")
ap.add_argument("--camera-fps", type=int, default=60, help="the IP cameras capture rate in Studio: above the recording fps, so its 30 Hz recorder finds a new frame each tick")
a = ap.parse_args()
LOG = open(a.log, "a") if a.log else None


def log(*x):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(v) for v in x); print(msg, flush=True)
    if LOG: LOG.write(msg + "\n"); LOG.flush()


def http(method, url, body=None, timeout=30):
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


S, SIM = a.studio.rstrip("/"), a.sim.rstrip("/")
CAMS = (a.camera_base or a.sim).rstrip("/")
port = int(SIM.rsplit(":", 1)[1])
# 1. the project and its parts
projects = http("GET", f"{S}/api/projects")
proj = next((p for p in projects if p["name"] == a.project), None)
if proj is None:
    proj = http("POST", f"{S}/api/projects", {"id": str(uuid.uuid4()), "name": a.project, "datasets": []})
    log("created project", a.project, proj["id"])
pid = proj["id"]
robots = http("GET", f"{S}/api/projects/{pid}/robots")
fol = next((r for r in robots if r["type"] == "MuJoCo_reBotB601_Follower"), None)
if fol is None:
    fol = http("POST", f"{S}/api/projects/{pid}/robots", {"id": str(uuid.uuid4()), "name": "reBot B601 (sim, bucket desk v2)",
               "type": "MuJoCo_reBotB601_Follower", "payload": {"name": a.owner, "http_url": SIM}})
lead = next((r for r in robots if r["type"] == "MuJoCo_Virtual_Leader"), None)
if lead is None:
    lead = http("POST", f"{S}/api/projects/{pid}/robots", {"id": str(uuid.uuid4()), "name": "MuJoCo virtual leader",
                "type": "MuJoCo_Virtual_Leader", "payload": {"http_port": port}})
cams = {c["name"]: c for c in http("GET", f"{S}/api/projects/{pid}/cameras")}
# Studio camera name -> the plugin's stream. Named as on the real reBot dataset (observation.images.overview /
# observation.images.gripper), so sim and real recordings share their feature names; the plugin's camera is "wrist".
STREAMS = {"overview": "overview", "gripper": "wrist"}
for name, stream in STREAMS.items():
    if name not in cams:
        cams[name] = http("POST", f"{S}/api/projects/{pid}/cameras", {"id": str(uuid.uuid4()), "driver": "ipcam", "name": name,
                          "fingerprint": {"url": f"{CAMS}/cameras/{stream}/mjpeg"}, "hardware_name": f"MuJoCo {stream} (composited)",
                          "payload": {"url": f"{CAMS}/cameras/{stream}/mjpeg", "width": 640, "height": 480, "fps": a.camera_fps}})
env_robots = [{"robot_id": fol["id"], "tele_operator": {"type": "robot", "robot_id": lead["id"]}}]   # a session drives one robot
env_cams = [{"camera_id": cams[n]["id"]} for n in STREAMS]
envs = http("GET", f"{S}/api/projects/{pid}/environments")
env = next((e for e in envs if e["name"] == "bucket desk v2 (gripper)"), None)
if env is None:
    env = http("POST", f"{S}/api/projects/{pid}/environments", {"id": str(uuid.uuid4()), "name": "bucket desk v2 (gripper)", "robots": env_robots, "cameras": env_cams})
ds_name = a.dataset or f"bucket_sort_v2_{time.strftime('%Y%m%d_%H%M')}"
proj = http("GET", f"{S}/api/projects/{pid}")
ds = next((d for d in proj.get("datasets", []) if d.get("name") == ds_name), None)
if ds is None:
    ds = http("POST", f"{S}/api/dataset", {"id": str(uuid.uuid4()), "name": ds_name, "default_task": "put the object in the bucket",
              "project_id": pid, "environment_id": env["id"]})
log("project", pid, "dataset", ds_name, ds["id"])

# 2. the runtime session
from websockets.sync.client import connect  # noqa: E402
ws = connect(S.replace("http", "ws", 1) + f"/api/projects/{pid}/runtime/ws", open_timeout=10, max_size=2**24)
state: dict = {}


def pump(timeout=0.05):
    """Read messages for up to *timeout* s; keep the last state; return acks by request id."""
    acks = {}
    end = time.time() + timeout
    while True:
        try:
            raw = ws.recv(timeout=max(0.0, end - time.time()))
        except TimeoutError:
            return acks
        if isinstance(raw, bytes):
            continue   # observation frames
        m = json.loads(raw); t = m.get("type")
        if t == "state": state.update(m["data"])
        elif t == "ack": acks[m["request_id"]] = m
        elif t == "error": log("studio error:", m.get("component"), m.get("message"))


def command(cmd: str, timeout=60.0, **kw) -> dict:
    rid = uuid.uuid4().hex; ws.send(json.dumps({"command": cmd, "request_id": rid, **kw}))
    end = time.time() + timeout
    while time.time() < end:
        acks = pump(0.1)
        if rid in acks:
            if not acks[rid].get("ok"): raise SystemExit(f"{cmd} refused: {acks[rid].get('error')}")
            return acks[rid]
    raise SystemExit(f"{cmd}: no ack in {timeout} s")


pump(1.0)
# r16: the simulation's camera processes (two per camera) take a while to deliver a first frame; Studio's IP cameras
# fail to start on a stream without one (the r16 check's first try: "Failed to open IP camera stream")
import urllib.request as _ur
t_cam = time.time()
for cam_name in ("overview", "wrist"):
    while True:
        try:
            with _ur.urlopen(f"{CAMS}/cameras/{cam_name}/frame.jpg", timeout=5) as r_:
                if r_.status == 200 and len(r_.read()) > 1000: break
        except Exception:
            pass
        if time.time() - t_cam > 180: raise SystemExit(f"the simulation's {cam_name} stream delivers no frame")
        time.sleep(1.0)
log(f"camera streams ready after {time.time() - t_cam:.1f} s")
command("load_environment", timeout=120, environment={"id": env["id"], "name": env["name"], "robots": env_robots, "cameras": env_cams}, unload_incompatible=True)
log("environment loaded:", json.dumps(state.get("environment"))[:300])
command("set_control", control={"type": "teleop", "hz": 100})
command("load_dataset", dataset_id=ds["id"], fps=a.fps, timeout=120)
log("dataset loaded:", json.dumps(state.get("dataset")))

# 3. the plugin: autopilot as the leader, episodes reset by the plugin after each success
# r14: one scene per episode. The plugin's seed is FIXED (every reset repeats it), so a simulation started with
# `--seed` would record one scene over and over (r13 review). The recorder sets the seed itself: with --seed S the
# reset before try k uses S + k - 1 (a varied run that repeats exactly), without it no seed (random, and any seed the
# simulation was started with is cleared).
def episode_seed(k: int) -> int | None:
    return None if a.seed is None else (a.seed + k - 1) % 2_147_483_647
http("POST", f"{SIM}/seed", {"seed": episode_seed(1)})
http("POST", f"{SIM}/episode/auto-reset", {"enabled": True, "dwell_s": 4.0})   # longer than the 1 s success hold below
http("POST", f"{SIM}/reset")
time.sleep(1.0)
http("POST", f"{SIM}/autopilot", {"mode": "leader"})
http("POST", f"{SIM}/autopilot/start-gate", {"enabled": bool(a.plan_before_record)})

# 4. the episodes
# resumable: a dataset that already holds episodes (an interrupted run, same --dataset) continues its count
saved, tries, last_gen = int((state.get("dataset") or {}).get("episodes_recorded") or 0), 0, None
if saved: log(f"resuming: the dataset already holds {saved} episodes")
def health():
    return http("GET", f"{SIM}/health", timeout=10)
while saved < a.episodes and (not a.max_tries or tries < a.max_tries):
    # a new episode: a new layout or sequence step
    t0 = time.time()
    while True:
        pump(0.05); h = health(); ep = h["episode"]; gen = (ep.get("reset_count"), ep.get("episode_count"))
        if ep.get("instruction") and gen != last_gen: break
        if time.time() - t0 > 60: raise SystemExit("the simulation does not start a new episode")
    last_gen = gen; task = ep["instruction"]["text"]; tries += 1
    if a.plan_before_record:   # r16: wait until the first attempt is planned (not recorded); a failure before counts
        t_ready, early = time.time(), None
        while True:
            pump(0.05); h = health(); ap_ = h.get("autopilot") or {}
            if ap_.get("ready"): break
            if ap_.get("outcome") in ("failed", "stalled"): early = ap_["outcome"]; break
            if (h["episode"].get("reset_count"), h["episode"].get("episode_count")) != last_gen: early = "reset"; break
            if time.time() - t_ready > 120: early = "timeout"; break
        if early is not None:
            log(f"episode try {tries}: {task!r}: {early} before recording ({ap_.get('failure')}; reasons {ap_.get('reasons')})")
            if a.seed is not None: http("POST", f"{SIM}/seed", {"seed": episode_seed(tries + 1)})
            if early != "reset": http("POST", f"{SIM}/reset")
            continue
        log(f"  planned before recording in {time.time() - t_ready:.1f} s")
    # r13: let Studio's camera streams and follower state catch up with the reset before recording (the pilot's first
    # 1-6 frames showed the previous scene; first_frame_check.py checks every saved episode)
    pump(a.start_settle)
    if a.plan_before_record:   # the go just before the recording starts: the start pause covers Studio's start latency
        http("POST", f"{SIM}/autopilot/start-gate", {"go": True})
    command("start_episode", task=task)
    log(f"episode try {tries}: {task!r} (sequence step {ep.get('sequence_index')}, look {ep.get('look', {}).get('variant')})")
    outcome, held_since, t_start = None, None, time.time()
    while True:
        pump(0.05); h = health(); ap_ = h.get("autopilot") or {}; ep = h["episode"]
        if ap_.get("outcome") in ("failed", "stalled"): outcome = ap_["outcome"]; break
        if ap_.get("outcome") == "success" and ep.get("success"):
            held_since = held_since or time.time()
            if time.time() - held_since >= 1.0: outcome = "success"; break
        else:
            held_since = None
        if (ep.get("reset_count"), ep.get("episode_count")) != last_gen: outcome = "reset"; break
        if time.time() - t_start > 90: outcome = "timeout"; break
    if a.seed is not None:   # the next reset (the plugin's after a success, ours after a failure) draws the next scene
        http("POST", f"{SIM}/seed", {"seed": episode_seed(tries + 1)})
    clean = bool(ap_.get("clean")) if outcome == "success" else False
    if outcome == "success" and (clean or a.keep == "all"):
        command("save_episode", timeout=60); saved += 1
        log(f"  saved ({saved}/{a.episodes}) in {time.time() - t_start:.1f} s")
    else:
        command("discard_episode", timeout=60)
        log(f"  discarded: {outcome}{'' if outcome != 'success' else ' not clean'} ({ap_.get('failure')}; reasons {ap_.get('reasons')})")
        if outcome != "success":
            http("POST", f"{SIM}/reset")

# 5. done
http("POST", f"{SIM}/autopilot", {"mode": "off"})
command("unload_dataset", timeout=600)
ds_info = http("GET", f"{S}/api/dataset/{ds['id']}")
log("done:", saved, "episodes saved in", tries, "tries; dataset", json.dumps(ds_info)[:400])
ws.close()
