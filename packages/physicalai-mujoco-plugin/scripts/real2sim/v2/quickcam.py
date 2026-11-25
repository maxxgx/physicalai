"""Builder check: overview + wrist composites at reset for a few seeds (and the arm moved over the object area).
usage: quickcam.py OUT.jpg SEED [SEED...]"""
import sys, numpy as np, cv2, mujoco
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv, seeds = sys.argv[:2], [int(s) for s in sys.argv[2:]]
import bucketsim_v2 as B
robot = B.make_robot(seeds[0]); comp = B.compositor(robot); rows = []
m, d = robot._sim.model, robot._sim.data
for s in seeds:
    B.reset(robot, s)
    imgs = [comp.render_camera(n)["rgb"] for n in ("overview", "wrist")]
    # the wrist over the object area: joints that put the hand ~25 cm above the desk in front of the base
    q0 = d.qpos.copy()
    for j, v in zip(range(1, 7), (0.0, -0.9, -0.6, 1.2, 0.0, 0.0)): d.qpos[m.jnt_qposadr[m.joint(f"joint{j}").id]] = v
    mujoco.mj_forward(m, d); imgs.append(comp.render_camera("wrist")["rgb"]); d.qpos[:] = q0; mujoco.mj_forward(m, d)
    rows.append(np.hstack(imgs))
cv2.imwrite(sys.argv[1], cv2.resize(np.vstack(rows)[:, :, ::-1], None, fx=0.75, fy=0.75)); print("ok")
