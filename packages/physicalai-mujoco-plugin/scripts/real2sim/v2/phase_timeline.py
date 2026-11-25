"""r16 (D4): an episode's phase timeline (sim time of every demonstrator phase change; per phase total), for slow
multi-object episodes. usage: phase_timeline.py SEED KIND:BUCKET VARIANT"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
seed, (kind, bucket), variant = int(sys.argv[1]), sys.argv[2].split(":"), sys.argv[3]
robot = b.make_robot(seed, variant); m = robot._sim.model
robot._automation.set_autopilot("off"); b.reset(robot, seed, b.type_body(m, kind), bucket); robot._automation.set_autopilot("drive")
demo = robot._automation.autopilot.demonstrator; count = robot._http_status()["episode"]["episode_count"]
t, last, since, tot, events = 0.0, None, 0.0, defaultdict(float), []
while t < 80:
    b.tick(robot, demo); t += b.DT
    ph = demo._phase + (f"/{demo._plan.kind}" if demo._plan is not None and demo._phase not in ("settle",) else "") + (" (planning)" if demo._job is not None else "")
    if ph != last:
        if last is not None: tot[last] += t - since; events.append((round(since, 2), last, round(t - since, 2)))
        last, since = ph, t
    if robot._http_status()["episode"]["episode_count"] != count: break
tot[last] += t - since; events.append((round(since, 2), last, round(t - since, 2)))
for e in events: print(*e)
print("per phase:", json.dumps({k: round(v, 2) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}))
