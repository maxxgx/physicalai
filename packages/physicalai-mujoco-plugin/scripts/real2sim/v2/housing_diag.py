"""r16: where a housing-bucket touch happens: at each first contact of a hand body (not the fingers) with a bucket, the
sim time, the demonstrator's phase, the running plan's kind and its planned clearances (mm), the contact's geoms and
depth, and the hand body's height over the bucket's rim. usage: housing_diag.py SEED VARIANT"""
import sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
seed, variant = int(sys.argv[1]), sys.argv[2]
robot = b.make_robot(seed, variant); m, d = robot._sim.model, robot._sim.data
robot._automation.set_autopilot("off"); b.reset(robot, seed); robot._automation.set_autopilot("drive")
demo = robot._automation.autopilot.demonstrator; count = robot._http_status()["episode"]["episode_count"]
hand = {m.body(n).id for n in ("link6", "gripper_end")}; buckets = {m.body(f"bucket_{c}").id: c for c in ("black", "green", "red", "blue")}
t, seen = 0.0, set()
while t < 60:
    b.tick(robot, demo); t += b.DT
    for i in range(d.ncon):
        c = d.contact[i]; g1, g2 = c.geom1, c.geom2; b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        for hb, bb, hg, bg in ((b1, b2, g1, g2), (b2, b1, g2, g1)):
            if hb in hand and bb in buckets:
                key = (round(t, 1), m.geom(hg).name)
                if key in seen: continue
                seen.add(key); p = demo._plan
                print(f"t {t:.2f} phase {demo._phase} plan {getattr(p, 'kind', None)} ok {getattr(p, 'ok', None)} clearance { {k: round(v * 1000, 1) for k, v in (getattr(p, 'clearance', {}) or {}).items()} } "
                      f"| {m.geom(hg).name} ({m.body(hb).name}) x {m.geom(bg).name} depth {c.dist * 1000:.2f} mm | hand z {d.geom_xpos[hg][2] * 1000:.0f} mm", flush=True)
    if robot._http_status()["episode"]["episode_count"] != count: break
