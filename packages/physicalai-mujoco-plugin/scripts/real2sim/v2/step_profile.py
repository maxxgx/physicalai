"""r17 item 7: physics cost per step on the SPEED seeds, the finger collision as delivered (256 pieces per finger) against
an alternative file (ALT_COLLISION, e.g. r16's 96-piece fingers): the demo episode run for SECONDS of sim time, mj_step
timed alone (the demonstrator and cameras excluded), median / p90 us, and the contacts per step.
usage: step_profile.py SEED [SECONDS]"""
import os, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physicalai_mujoco_plugin.profiles import rebot_b601 as prof
if os.environ.get("ALT_COLLISION"): prof.REBOT_B601_TIGHT_COLLISION = Path(os.environ["ALT_COLLISION"])
import bucketsim_v2 as b
seed = int(sys.argv[1]); secs = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
robot = b.make_robot(seed); b.reset(robot, seed); m, d = robot._sim.model, robot._sim.data
demo = robot._automation.autopilot.demonstrator; robot._automation.set_autopilot("drive")
orig = mujoco.mj_step; times, ncon = [], []
def timed(model, data, *a, **k):
    t0 = time.perf_counter(); orig(model, data, *a, **k); times.append(time.perf_counter() - t0); ncon.append(data.ncon)
mujoco.mj_step = timed
try:
    t = 0.0
    while t < secs:
        b.tick(robot, demo); t += b.DT
finally:
    mujoco.mj_step = orig
T = np.array(times) * 1e6
print(f"seed {seed} {'ALT' if os.environ.get('ALT_COLLISION') else 'delivered'}: steps {len(T)}, step us median {np.median(T):.0f} p90 {np.percentile(T, 90):.0f}, contacts median {np.median(ncon):.0f} max {max(ncon)}, ngeom {m.ngeom}")
