"""r16 final baselines: bucketsim_v2.py kinds (the task's own draws) plus the collateral-arming audit the coordinator
asked for. Per episode, every placed object that is not a target: its first rise of more than 5 mm above its lowest
point (the demonstrator's COLLATERAL_LIFT_M) counted from the reset (r15's arming) and from the demonstrator's first plan
(r16's arming, after _pile_still); a flag only the old arming raises is listed with its time, the demonstrator's phase
then and the arm's state (no plan yet / moving). Output: kinds' list of episodes, each with "collateral_audit".
usage: baseline_r16.py START COUNT VARIANT OUT_JSON"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
import os
from physicalai_mujoco_plugin import bucket_sort_demo as _bsd
if os.environ.get("BUCKET_HIT_MM"): _bsd.BUCKET_HIT_M = float(os.environ["BUCKET_HIT_MM"]) / 1000   # r16 comparison only
if os.environ.get("AHEAD_BE") == "0": _bsd.AHEAD_BEST_EFFORT = False   # r17 comparison: no best-effort reuse
if os.environ.get("WAITS_OLD") == "1":   # r16 ablation: r15's fixed between-object pause and 0.3 s pile hold
    _bsd.BETWEEN_PAUSE_S = 0.2; _bsd.BucketSortDemonstrator.PILE_STILL_S = 0.3
if os.environ.get("TIMING") is not None:   # r18 comparison: force the scene's r2s_operator_timing (0 off, 18 on)
    _bsd.BucketSortDemonstrator._operator_timing = lambda self: int(os.environ["TIMING"])
for _k, _v in json.loads(os.environ.get("TIMING_SET", "{}")).items():   # r18 ablations of OPERATOR_TIMING_R18
    _bsd.OPERATOR_TIMING_R18[_k] = tuple(_v) if isinstance(_v, list) else _v
from physicalai_mujoco_plugin.bucket_sort_objects import real_kind
def main() -> None:
    start, count, variant, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    LIFT = 0.005
    robot = b.make_robot(start, variant); m = robot._sim.model; out = []

    def first_rise(T, Z, i0):
        low, k0 = np.inf, None
        for k in range(i0, len(T)):
            low = min(low, Z[k])
            if Z[k] > low + LIFT: return k
        return None

    SEEDS = [int(x) for x in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS") else range(start, start + count)
    for seed in SEEDS:
        robot._automation.set_autopilot("off"); b.reset(robot, seed)
        task = robot._sim.on_reset; ins = task.instruction; demo = robot._automation.autopilot.demonstrator
        placed = [n for n in b.object_names(m) if n in task.layout.objects]; rec = []
        hook = lambda t, p: rec.append((t, p[:, 2].copy(), demo._plan is not None, demo._phase))   # noqa: E731
        w0 = time.perf_counter()
        r = b.run_episode(robot, seed, max_s=25.0 + 18.0 * len(ins.all_targets), on_tick=hook)
        r["wall_s"] = round(time.perf_counter() - w0, 2)   # r18 SPEED: the episode's wall time (planning waits included)
        r.update({"kind": ins.kind, "targets": list(ins.all_targets), "canonical": ins.canonical_text,
                  "counts": {k: sum(real_kind(n) == k for n in task.layout.objects) for k in ("resistor", "cpu", "battery")},
                  "stacked": len(task.layout.heights)})
        T = np.array([x[0] for x in rec]); Z = np.array([x[1] for x in rec]) if rec else np.zeros((0, len(placed)))
        armed = np.array([x[2] for x in rec]); i_arm = int(np.argmax(armed)) if armed.any() else len(rec)
        audit = {"first_plan_s": round(float(T[i_arm]), 2) if i_arm < len(T) else None, "old_only": [], "both": [], "new_only": []}
        for j, n in enumerate(placed):
            if n in ins.all_targets: continue
            ko, kn = first_rise(T, Z[:, j], 0), first_rise(T, Z[:, j], i_arm)
            if ko is None and kn is None: continue
            k = ko if ko is not None else kn
            row = {"object": n, "t": round(float(T[k]), 2), "phase": rec[k][3], "arm": "moving" if rec[k][2] else "no plan yet"}
            audit["both" if (ko is not None and kn is not None) else ("old_only" if ko is not None else "new_only")].append(row)
        r["collateral_audit"] = audit; out.append(r)
        print(json.dumps({k: r[k] for k in ("seed", "instruction", "success", "failure", "sim_s", "glitch", "max_penetration_m", "unclean_reasons", "wall_s")} | {"audit": audit}), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    n = len(out); print(f"success {sum(r['success'] for r in out)} / {n}; glitches {sum(r['glitch'] for r in out)}")


if __name__ == "__main__":
    main()
