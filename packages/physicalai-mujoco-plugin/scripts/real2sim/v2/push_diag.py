"""r17 (D4 T6): what touches a non-target object before the first close: every first contact of a hand body (link6,
gripper_end, the fingers) with an object the instruction does not name, until the first close: sim time, phase, the
plan's kind and planned clearances, the geoms and depth. usage: push_diag.py SEED KIND:BUCKET VARIANT"""
import sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
seed, (kind, bucket), variant = int(sys.argv[1]), sys.argv[2].split(":"), sys.argv[3]
robot = b.make_robot(seed, variant); m, d = robot._sim.model, robot._sim.data
robot._automation.set_autopilot("off"); b.reset(robot, seed, b.type_body(m, kind), bucket); robot._automation.set_autopilot("drive")
demo = robot._automation.autopilot.demonstrator; ins = robot._sim.on_reset.instruction
hand = {m.body(n).id for n in ("link6", "gripper_end", "gripper_left", "gripper_right")}
targets = set(getattr(ins, "all_targets", None) or (ins.object,)); objs = {m.body(n).id: n for n in robot._sim.on_reset.layout.objects if n not in targets}
tgt = {m.body(n).id: n for n in targets}
t, seen = 0.0, set(); fast = {}
oadr = {n: m.jnt_dofadr[m.joint(f"{n}:joint").id] for n in objs.values()}
while t < 20:
    b.tick(robot, demo); t += b.DT
    if demo._phase == "close": break
    for n, a in oadr.items():
        v = float(np.linalg.norm(d.qvel[a:a + 3]))
        if v > 0.01 and v > fast.get(n, (0,))[0]:
            touch = sorted({m.body(m.geom_bodyid[c.geom2 if m.geom_bodyid[c.geom1] == m.body(n).id else c.geom1]).name for c in d.contact[:d.ncon]
                            if m.body(n).id in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])})
            fast[n] = (v, round(t, 2), demo._phase, touch)
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        for hb, ob, hg in ((b1, b2, c.geom1), (b2, b1, c.geom2)):
            if hb in hand and ob in tgt and ("T", hb) not in seen and t > 1.0:
                seen.add(("T", hb)); print(f"t {t:.2f} {demo._phase}: {m.geom(hg).name} ({m.body(hb).name}) touches the TARGET {tgt[ob]} depth {c.dist * 1000:.2f} mm", flush=True)
            if hb in hand and ob in objs and (hb, ob) not in seen:
                seen.add((hb, ob)); p = demo._plan
                print(f"t {t:.2f} {demo._phase} plan {getattr(p, 'kind', None)} ok {getattr(p, 'ok', None)} clearance { {k: round(v * 1000, 1) for k, v in (getattr(p, 'clearance', {}) or {}).items()} } | {m.geom(hg).name} ({m.body(hb).name}) x {objs[ob]} depth {c.dist * 1000:.2f} mm", flush=True)
print("fastest non-target motion before the close (m/s, t, phase, touching):", {n: (round(v[0], 3),) + v[1:] for n, v in fast.items()})
print("first close at", round(t, 2), "target", ins.object)
