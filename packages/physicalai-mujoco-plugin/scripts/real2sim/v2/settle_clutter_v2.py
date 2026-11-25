"""Settle the v2 buckets' real contents (mixed piles of the task-object copies) with physics, in each bucket's frame.
Contents as seen in allowed frames (e.g. f00972, f01358, f01806 region / old sheets 30-36 s, 118-124 s): black full of
resistors and 9 V batteries, blue full of CPUs, green four resistors and one battery, red empty.
Writes REAL2SIM_ROOT/align/clutter_v2.json {colour: [{type, pos, quat}], bottom up}.  usage: settle_clutter_v2.py"""
import json, os, sys, numpy as np, mujoco
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
ASSETS = Path(__file__).resolve().parents[3] / "urdf/scenes/bucket_sort/assets"
_SIZE = json.load(open(ROOT / "align/object_size.json")) if (ROOT / "align/object_size.json").exists() else {}
SCALE = {k: float(_SIZE.get(k, {}).get("scale", 1.0)) for k in ("resistor", "battery", "cpu")}   # r09: measured object sizes
HULL = {"resistor": ("resistor_hull", 0.040 * SCALE["resistor"] ** 3), "battery": ("battery_hull", 0.045 * SCALE["battery"] ** 3), "cpu": ("cpu_hull", 0.030 * SCALE["cpu"] ** 3)}
# with the resistors scaled to the measured size (r09 0.57, r10 0.65) the black bucket needs more of them to look as full as it is
PILES = {"black": ["resistor"] * (24 if SCALE["resistor"] < 0.8 else 10) + ["battery"] * 4, "blue": ["cpu"] * min(26, round(11 / SCALE["cpu"] ** 2)),   # r11: CPUs 0.644 x: as many as fill the bucket as before
         "green": ["resistor"] * 4 + ["battery"], "red": []}
def xml(kinds, poses):
    items = "\n".join(f'<body name="i{k}" pos="{x:.4f} {y:.4f} {z:.4f}" euler="{a:.3f} {b:.3f} {c:.3f}"><freejoint/>'
                      f'<geom type="mesh" mesh="{HULL[kind][0]}" mass="{HULL[kind][1]}" condim="4" friction="0.9 0.01 0.0001"/></body>'
                      for k, (kind, (x, y, z, a, b, c)) in enumerate(zip(kinds, poses)))
    return f'''<mujoco><compiler meshdir="{ASSETS}"/><option timestep="0.002" cone="elliptic" impratio="10"/>
<asset>{"".join(f'<mesh name="{h}" file="{h}.msh" scale="{SCALE[k]} {SCALE[k]} {SCALE[k]}"/>' for k, (h, _) in HULL.items())}</asset>
<worldbody><geom type="plane" size="1 1 .01"/>
<body name="bucket"><geom type="box" pos="0 -0.0125 0.003" size="0.075 0.08875 0.003"/>
<geom type="box" pos="0.072 0 0.063" size="0.003 0.10125 0.057"/><geom type="box" pos="-0.072 0 0.063" size="0.003 0.10125 0.057"/>
<geom type="box" pos="0 -0.09825 0.063" size="0.069 0.003 0.057"/><geom type="box" pos="0 0.0867 0.0195" euler="0.8789 0 0" size="0.069 0.0227 0.003"/></body>
{items}</worldbody></mujoco>'''
out = {}
for col, kinds in PILES.items():
    if not kinds: out[col] = []; continue
    rng = np.random.default_rng(len(col)); kinds = [kinds[i] for i in rng.permutation(len(kinds))]
    poses = [(rng.uniform(-0.04, 0.04), rng.uniform(-0.065, 0.045), 0.03 + (0.03 if len(kinds) < 16 else 0.018) * k, *rng.uniform(-np.pi, np.pi, 3)) for k in range(len(kinds))]
    m = mujoco.MjModel.from_xml_string(xml(kinds, poses)); d = mujoco.MjData(m)
    for _ in range(int(10.0 / m.opt.timestep)): mujoco.mj_step(m, d)
    items = []
    for k, kind in enumerate(kinds):
        b = m.body(f"i{k}").id; pos = d.xpos[b].copy(); q = d.xquat[b].copy()
        if abs(pos[0]) < 0.07 and abs(pos[1]) < 0.095 and 0 < pos[2] < 0.12: items.append({"type": kind, "pos": pos.round(5).tolist(), "quat": q.round(5).tolist()})
    items.sort(key=lambda it: it["pos"][2]); out[col] = items
    print(col, "inside", len(items), "/", len(kinds), "top z mm", round(max(i["pos"][2] for i in items) * 1000, 1), "max speed", round(float(np.abs(d.qvel).max()), 4))
(ROOT / "align").mkdir(exist_ok=True); json.dump(out, open(ROOT / "align/clutter_v2.json", "w"), indent=1)
