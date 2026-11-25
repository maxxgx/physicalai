"""Record the scripted demonstrator on bucket_sort_desk for DURATION seconds of sim time (episodes back to back),
qpos at FPS. usage: record_demo.py OUT.npz [--seconds 76] [--fps 15] [--seed 7]"""
import argparse, json, numpy as np
from physicalai_mujoco_plugin.robot import MuJoCoRobot
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seconds", type=float, default=76.0)
ap.add_argument("--fps", type=float, default=15.0); ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
robot = MuJoCoRobot("rebot_b601", scene="bucket_sort_desk", cameras=[], http_port=0, seed=a.seed); robot.connect()
robot._automation.set_autopilot("off"); robot._set_seed(a.seed); robot._reset(); robot._set_seed(None)
robot._automation.set_autopilot("drive")
d = robot._sim.data; dt = 0.02; demo = robot._automation.autopilot.demonstrator
qs, events, t, next_t, last = [], [], 0.0, 0.0, None
while t < a.seconds:
    robot.get_observation()
    if demo.planning: continue  # planning happens off the clock; the arm holds still meanwhile in the video
    t += dt
    while next_t <= t:
        qs.append(d.qpos.copy()); next_t += 1.0 / a.fps
    ep = robot._http_status()["episode"]; ins = ep.get("instruction") or {}
    key = (ep["episode_count"], ins.get("text"))
    if key != last: events.append({"t": round(t, 2), "episode": ep["episode_count"], "instruction": ins.get("text")}); last = key
np.savez_compressed(a.out, qpos=np.array(qs), fps=a.fps)
json.dump(events, open(a.out.replace(".npz", "_events.json"), "w"), indent=1)
print("frames", len(qs), "episodes", len({e["episode"] for e in events}))
robot.disconnect()
