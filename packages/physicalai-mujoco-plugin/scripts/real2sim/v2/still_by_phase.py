"""r17 (D7 analysis, coordinator): near-still time by demonstrator phase, in process (sim time; live planning stalls come
on top, measured in the Studio check). Per control tick the phase (the settle before an episode's first object is the
start pause) and whether the arm is near-still: every joint (and the gripper's finger travel) slower than 0.1 deg per
1/30 s (3 deg/s; the finger: 0.1 mm per frame). Per phase: its share of all time, and of the near-still time.
usage: still_by_phase.py START COUNT VARIANT OUT_JSON"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
start, count, variant, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
robot = b.make_robot(start, variant); m, d = robot._sim.model, robot._sim.data
jdof = [m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 7)]; fdof = m.jnt_dofadr[m.joint("joint_left").id]
tot, still = defaultdict(float), defaultdict(float)
for seed in range(start, start + count):
    robot._automation.set_autopilot("off"); b.reset(robot, seed); robot._automation.set_autopilot("drive")
    demo = robot._automation.autopilot.demonstrator; n0 = robot._http_status()["episode"]["episode_count"]; t = 0.0
    while t < 60:
        b.tick(robot, demo); t += b.DT
        if robot._http_status()["episode"]["episode_count"] != n0: break
        ep = demo._episode; ph = demo._phase
        if ph == "settle" and ep is not None and ep.targets_done == 0 and ep.attempt == 0: ph = "start pause"
        if ph == "approach" and demo._plan is not None and demo._plan.kind == "transit": ph = "transit"
        q = np.degrees(np.abs(d.qvel[jdof])).max(); f = abs(float(d.qvel[fdof])) * 1000
        tot[ph] += b.DT
        if q < 3.0 and f < 3.0: still[ph] += b.DT
T, S = sum(tot.values()), sum(still.values())
res = {"seeds": [start, start + count - 1], "variant": variant, "sim_s": round(T, 1), "near_still_pct_of_time": round(100 * S / T, 1),
       "by_phase": {k: {"pct_of_time": round(100 * tot[k] / T, 1), "near_still_pct_of_time": round(100 * still[k] / T, 1),
                        "share_of_near_still": round(100 * still[k] / S, 1)} for k in sorted(tot, key=lambda k: -still[k])}}
print(json.dumps(res, indent=1)); json.dump(res, open(OUT, "w"), indent=1)
