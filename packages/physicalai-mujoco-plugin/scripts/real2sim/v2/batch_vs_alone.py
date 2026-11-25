"""r16 (coordinator, the integer-ticks option): does a seed give the same episode alone and as the last of a batch?
Runs SEEDS[:-1] then SEEDS[-1] in one process (one robot), and SEEDS[-1] alone in a fresh robot; compares the last
episode (sim_s, first_grasp_s, settle_s, success, max penetration, grasp openings). TICK=0 runs the r15 float clock; VARIANT the layout (default).
usage: batch_vs_alone.py OUT_JSON SEED [SEED ...]"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
from physicalai_mujoco_plugin import bucket_sort_demo as bsd
bsd.TICK_CLOCK = os.environ.get("TICK", "1") == "1"
OUT, SEEDS = sys.argv[1], [int(s) for s in sys.argv[2:]]
KEYS = ("success", "sim_s", "first_grasp_s", "settle_s", "max_penetration_m", "failure")

def episode(robot, seed):
    robot._automation.set_autopilot("off"); b.reset(robot, seed); r = b.run_episode(robot, seed)
    return {k: r[k] for k in KEYS} | {"openings": {n: [g["opening_mm"] for g in v] for n, v in r["grasps"].items()}}

V = os.environ.get("VARIANT", "default")
robot = b.make_robot(SEEDS[0], V); batch = [episode(robot, s) for s in SEEDS]
alone = episode(b.make_robot(SEEDS[-1], V), SEEDS[-1])
res = {"tick_clock": bsd.TICK_CLOCK, "seeds": SEEDS, "batch_last": batch[-1], "alone": alone, "identical": batch[-1] == alone}
print(json.dumps(res)); json.dump(res, open(OUT, "w"), indent=1)
