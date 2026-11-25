"""r16 (D3a): the fingers' collision at the map's poses: closed (reading 0), held (-50), open (-270) and the readings'
overshoot (-278), through the profile's gripper channel (PHYSICALAI_REBOT_GRIPPER_MAP picks the map). Per pose: each
finger's travel (mm, against the model's joint range) and the signed least distance (mm, mj_geomDistance; < 0 = overlap)
from its collision geoms to the other collision geoms of the arm (link6, the housing, the mount, the other finger).
usage: finger_pose_audit.py OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
from physicalai_mujoco_plugin.profiles import rebot_b601 as prof
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data
fl, fr = m.body("gripper_left").id, m.body("gripper_right").id; root = m.body_rootid[fl]
col = lambda g: m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0
fing = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and col(g)] for n, bid in (("left", fl), ("right", fr))}
others = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root and col(g) and m.geom_bodyid[g] not in (fl, fr)]
jr = {n: m.jnt_range[m.joint(f"joint_{n}").id].tolist() for n in ("left", "right")}
res = {"map": prof.REBOT_B601_GRIPPER_MAP, "finger_range_m": prof.REBOT_B601_FINGER_RANGE, "joint_range_m": jr, "poses": {}}
fromto = np.zeros(6)
for reading in (0.0, -50.0, -270.0, -278.0):
    travel = (reading - prof.REBOT_B601_GRIPPER_OFFSET) / prof.REBOT_B601_GRIPPER_SCALE
    for n in ("left", "right"): d.qpos[m.jnt_qposadr[m.joint(f"joint_{n}").id]] = travel
    mujoco.mj_forward(m, d); row = {"travel_mm": round(travel * 1000, 2), "opening_mm": round(2 * travel * 1000, 2)}
    for n, gs in fing.items():
        best = (np.inf, None)
        for g in gs:
            for o in others + (fing["right"] if n == "left" else []):
                dist = mujoco.mj_geomDistance(m, d, g, o, 0.05, fromto)
                if dist < best[0]: best = (dist, m.body(m.geom_bodyid[o]).name)
        row[f"{n}_least_mm"] = [round(best[0] * 1000, 2), best[1]]
    res["poses"][str(reading)] = row; print(reading, row, flush=True)
print("joint range (m)", jr); json.dump(res, open(sys.argv[1], "w"), indent=1)
