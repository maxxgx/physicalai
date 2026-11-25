"""(--matrix: one episode per bucket x object, seeds from first_seed.) Run the scripted demonstrator on bucket_sort_desk for several seeds; report outcome per (object, bucket) and save
overview-camera frames (start | grasp | end) as a contact sheet. usage: desk_episodes.py N [first_seed]"""
import sys, time, json, numpy as np, mujoco, cv2
from pathlib import Path
from physicalai_mujoco_plugin.robot import MuJoCoRobot
sys.path.insert(0, str(Path(__file__).parent)); from common import ROOT
MATRIX = '--matrix' in sys.argv; sys.argv = [a for a in sys.argv if a != '--matrix']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8; first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
PAIRS = [(o, c) for c in ('black', 'green', 'red', 'blue') for o in ('resistor', 'battery', 'cpu')]
robot = MuJoCoRobot("rebot_b601", scene="bucket_sort_desk", cameras=[], http_port=0, seed=first)
robot.connect()
m, d = robot._sim.model, robot._sim.data
ren = mujoco.Renderer(m, 240, 320)
def snap():
    ren.update_scene(d, camera="overview"); return ren.render()[:, :, ::-1].copy()
rows, results = [], []
for i, seed in enumerate(range(first, first + (len(PAIRS) if MATRIX else N))):
    if MATRIX: robot._sim.on_reset.set_allowed_pairs([PAIRS[i]])
    robot._automation.set_autopilot("off"); robot._set_seed(seed); robot._reset(); robot._set_seed(None)
    task = robot._sim.on_reset; ins = task.instruction
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator; demo._rng = np.random.default_rng(seed)
    count = robot._http_status()["episode"]["episode_count"]; frames = [snap()]; t0 = time.time(); ok = False; ticks = 0
    while time.time() - t0 < 240:
        robot.get_observation()
        if not demo.planning: ticks += 1
        ep = robot._http_status()["episode"]
        if ep["episode_count"] != count: break
        if ticks == 600: frames.append(snap())
        if ep["success"]: ok = ep["in_bucket"][ins.object] == ins.bucket
        if demo.status()["outcome"] in {"success", "failed", "stalled"} or ticks > 45 / 0.02: break
    frames.append(snap())
    r = {"seed": seed, "object": ins.object, "bucket": ins.bucket, "success": ok, "outcome": demo.status()["outcome"],
         "failure": demo.status()["failure"], "fallback_layout": task.layout.fallback, "sim_s": round(ticks * 0.02, 1)}
    results.append(r); print(json.dumps(r), flush=True)
    row = np.hstack(frames + [np.zeros_like(frames[0])] * (3 - len(frames)))
    cv2.putText(row, f"{seed} {ins.object}->{ins.bucket} {'OK' if ok else 'FAIL'}", (6, 18), 0, 0.5, (0, 255, 0) if ok else (0, 0, 255), 1)
    rows.append(row)
cv2.imwrite(str(ROOT / f"sim/desk_episodes_{first}{'_matrix' if MATRIX else ''}.jpg"), np.vstack(rows))
print("success", sum(r["success"] for r in results), "/", len(results))
robot.disconnect()
