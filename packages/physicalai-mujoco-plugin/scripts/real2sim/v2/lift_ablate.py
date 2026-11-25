"""r16: the "grasp did not lift" episodes, as adopted against the old finger collision (ALT_COLLISION=path) or the r12
map (PHYSICALAI_REBOT_GRIPPER_MAP=r12): per seed success, the lift failures and the grasps' openings.
usage: lift_ablate.py VARIANT SEED [SEED ...]"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physicalai_mujoco_plugin.profiles import rebot_b601 as prof
if os.environ.get("ALT_COLLISION"): prof.REBOT_B601_TIGHT_COLLISION = Path(os.environ["ALT_COLLISION"])
import bucketsim_v2 as b
variant, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
for seed in seeds:
    robot = b.make_robot(seed, variant); robot._automation.set_autopilot("off"); b.reset(robot, seed); r = b.run_episode(robot, seed)
    lift = sum("did not lift" in x for x in r["unclean_reasons"]) + ("did not lift" in (r["failure"] or ""))
    print(seed, "success", r["success"], "no-lift", lift, "grasps", [(o, x["opening_mm"]) for o, xs in r["grasps"].items() for x in xs], flush=True)
