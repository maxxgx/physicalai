"""Settle piles of task-object copies inside a bucket with physics, for the desk scene's bucket contents.
Writes align/clutter.json: per pile, each item's type and pose in the bucket's body frame (resting, sorted bottom up).
usage: settle_clutter.py [N_PER_PILE] [PILES_PER_TYPE]"""
import json, sys, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); from common import ROOT, PLUGIN
ASSETS = PLUGIN / "urdf/scenes/bucket_sort/assets"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10; PILES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
HULL = {"resistor": ("resistor_hull", 0.040), "battery": ("battery_hull", 0.045), "cpu": ("cpu_hull", 0.030)}
def xml(kind, poses):
    items = "\n".join(f'<body name="i{k}" pos="{x:.4f} {y:.4f} {z:.4f}" euler="{a:.3f} {b:.3f} {c:.3f}"><freejoint/>'
                      f'<geom type="mesh" mesh="{HULL[kind][0]}" mass="{HULL[kind][1]}" condim="4" friction="0.9 0.01 0.0001"/></body>'
                      for k, (x, y, z, a, b, c) in enumerate(poses))
    return f'''<mujoco><compiler meshdir="{ASSETS}"/><option timestep="0.002" cone="elliptic" impratio="10"/>
<asset>{"".join(f'<mesh name="{h}" file="{h}.msh"/>' for h, _ in HULL.values())}</asset>
<worldbody><geom type="plane" size="1 1 .01"/>
<body name="bucket"><geom type="box" pos="0 -0.0125 0.003" size="0.075 0.08875 0.003"/>
<geom type="box" pos="0.072 0 0.063" size="0.003 0.10125 0.057"/><geom type="box" pos="-0.072 0 0.063" size="0.003 0.10125 0.057"/>
<geom type="box" pos="0 -0.09825 0.063" size="0.069 0.003 0.057"/><geom type="box" pos="0 0.0867 0.0195" euler="0.8789 0 0" size="0.069 0.0227 0.003"/></body>
{items}</worldbody></mujoco>'''
out = {}
for kind in HULL:
    for p in range(PILES):
        rng = np.random.default_rng(100 * p + len(kind))
        poses = [(rng.uniform(-0.045, 0.045), rng.uniform(-0.07, 0.05), 0.03 + 0.03 * k, *rng.uniform(-np.pi, np.pi, 3)) for k in range(N)]
        m = mujoco.MjModel.from_xml_string(xml(kind, poses)); d = mujoco.MjData(m)
        for _ in range(int(4.0 / m.opt.timestep)): mujoco.mj_step(m, d)
        items = []
        for k in range(N):
            b = m.body(f"i{k}").id; pos = d.xpos[b].copy(); q = d.xquat[b].copy()
            inside = abs(pos[0]) < 0.07 and abs(pos[1]) < 0.095 and 0 < pos[2] < 0.12
            if inside: items.append({"type": kind, "pos": pos.round(5).tolist(), "quat": q.round(5).tolist()})
        items.sort(key=lambda it: it["pos"][2])
        out[f"{kind}_{p}"] = items
        print(kind, p, "inside", len(items), "/", N, "top z mm", round(max(i["pos"][2] for i in items) * 1000, 1), "max speed", round(float(np.abs(d.qvel).max()), 4))
json.dump(out, open(ROOT / "align/clutter.json", "w"), indent=1)
