"""r16 (D4 / stalls, coordinator): outcomes against the planner's clearance margins, on fresh seeds (not the reviewer's).

Runs `kinds`-style episodes and records, per episode, every executed attempt plan's clearances (CLEARANCE keys: bucket,
objects, table, held_bucket, held_objects; metres, the plan's minimum) and whether it ran although it failed a check
(best effort), with the episode's outcome: success, glitch, pre-grasp push (> 1 cm/s), housing contact, collateral
(the demonstrator's unclean reasons). usage: margin_study.py START COUNT VARIANT OUT_JSON"""
import sys, json, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import bucketsim_v2 as b

if __name__ == "__main__":
    start, count, variant, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    robot = b.make_robot(start, variant); rows = []
    for seed in range(start, start + count):
        robot._automation.set_autopilot("off"); b.reset(robot, seed)
        r = b.run_episode(robot, seed)
        demo = robot._automation.autopilot.demonstrator
        plans = [{"kind": p.kind, "ok": p.ok, "problems": list(p.problems), "clearance_mm": {k: round(v * 1000, 2) for k, v in p.clearance.items()}}
                 for p in getattr(demo, "plans", []) if p.kind in ("attempt", "place")]
        reasons = list(r.get("unclean_reasons") or [])
        rows.append({"seed": seed, "success": r["success"], "failure": r.get("failure"), "glitch": r["glitch"], "max_penetration_mm": round(r["max_penetration_m"] * 1000, 2),
                     "pushed": (r.get("pre_grasp_speed_mps") or 0) > 0.01, "housing": r.get("housing_contact_ticks", 0) > 0,
                     "collateral": any("carried along" in x for x in reasons), "plans": plans})
        json.dump(rows, open(out, "w"), indent=1)
        print(seed, r["success"], len(plans), flush=True)
