"""r16 (coordinator: the trimmed finger collision must not let flat holds slip, r15's multi-contact lesson): the record
seeds whose flat holds slipped in r15's ablation (7418 battery, 7425 CPU), as delivered (multi-contact on, the drift on):
per grasp the opening, the hold's span and the slide after the first 0.2 s. usage: slip_check_r16.py [SEED ...]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
seeds = [int(s) for s in sys.argv[1:]] or [7418, 7425]
robot = b.make_robot(seeds[0], "record")
for seed in seeds:
    robot._automation.set_autopilot("off"); b.reset(robot, seed); r = b.run_episode(robot, seed)
    g = [(o, x["opening_mm"], x["t"], x["slide_after_0p2s_mm"]) for o, xs in (r.get("grasps") or {}).items() for x in xs]
    print(seed, "success", r["success"], "max pen", round(r["max_penetration_m"] * 1000, 2), "unclean", r["unclean_reasons"], "grasps", g, flush=True)
