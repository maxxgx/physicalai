"""Which contacts go deeper than 3 mm in a DEMO episode: per (geom1, geom2) pair the worst depth, when, and the
demonstrator phase. usage: pen_probe.py VARIANT SEED:TYPE:BUCKET [...]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
variant = sys.argv[1]
for spec in sys.argv[2:]:
    seed, kind, bucket = spec.split(":"); seed = int(seed)
    robot = b.make_robot(seed, variant); m, d = robot._sim.model, robot._sim.data
    import os
    if os.environ.get("FINGER_SOLREF"):   # experiment: stiffer finger contacts
        fing = [g for g in range(m.ngeom) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) in ("gripper_left", "gripper_right") and m.geom_contype[g]]
        m.geom_solref[fing] = [float(v) for v in os.environ["FINGER_SOLREF"].split(",")]
        if os.environ.get("FINGER_SOLIMP"): m.geom_solimp[fing, :3] = [float(v) for v in os.environ["FINGER_SOLIMP"].split(",")]
        m.geom_solmix[fing] = 100.0
    robot._automation.set_autopilot("off"); b.reset(robot, seed, b.type_body(m, kind), bucket)
    task = robot._sim.on_reset; placed = {m.body(n).id for n in b.object_names(m) if n in task.layout.objects}
    worst = {}; demo_ref = {}
    def on_tick(t, p):
        demo = robot._automation.autopilot.demonstrator
        for i in range(d.ncon):
            c = d.contact[i]
            if m.geom_bodyid[c.geom1] in placed or m.geom_bodyid[c.geom2] in placed:
                if -c.dist > 0.0025:
                    key = tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}:{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1])}",
                                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}:{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])}")))
                    if -c.dist > worst.get(key, (0,))[0]: worst[key] = (round(-c.dist * 1000, 2), round(t, 2), demo.phase)
    r = b.run_episode(robot, seed, on_tick=on_tick)
    print(json.dumps({"seed": seed, "pair": f"{kind}:{bucket}", "success": r["success"], "max_pen_mm": round(r["max_penetration_m"] * 1000, 2),
                      "deep": sorted(([k[0], k[1], *v] for k, v in worst.items()), key=lambda x: -x[2])[:6]}), flush=True)
    robot.disconnect() if hasattr(robot, "disconnect") else None
