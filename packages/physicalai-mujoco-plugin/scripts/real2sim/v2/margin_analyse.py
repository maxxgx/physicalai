"""r16: margin_study.py's output, per clearance margin: the episodes binned by their worst executed attempt-plan clearance
(mm; the planner's budget in brackets), with success, glitches, pre-grasp pushes, housing contacts and collateral per bin.
usage: margin_analyse.py STUDY_JSON [STUDY_JSON ...]"""
import json, sys
import numpy as np

BUDGET = {"bucket": 12, "objects": 8, "table": 2, "held_bucket": 10, "held_objects": 8}
BINS = [(-1e9, 0), (0, 3), (3, 6), (6, 9), (9, 12), (12, 1e9)]
rows = [r for p in sys.argv[1:] for r in json.load(open(p))]
print("episodes", len(rows), "success", sum(r["success"] for r in rows), "glitch", sum(r["glitch"] for r in rows),
      "pushed", sum(r["pushed"] for r in rows), "housing", sum(r["housing"] for r in rows), "collateral", sum(r["collateral"] for r in rows))
out = {}
for key, budget in BUDGET.items():
    vals = []
    for r in rows:
        v = [p["clearance_mm"][key] for p in r["plans"] if p["kind"] == "attempt" and key in p["clearance_mm"]]
        if v: vals.append((min(v), r))
    print(f"\n{key} (budget {budget} mm): {len(vals)} episodes with the measure")
    out[key] = []
    for lo, hi in BINS:
        sel = [r for v, r in vals if lo <= v < hi]
        if not sel: continue
        n = len(sel); row = {"bin_mm": [lo, hi], "n": n, "success": sum(r["success"] for r in sel), "glitch": sum(r["glitch"] for r in sel),
                             "pushed": sum(r["pushed"] for r in sel), "housing": sum(r["housing"] for r in sel), "collateral": sum(r["collateral"] for r in sel),
                             "failures": [(r["seed"], (r["failure"] or "")[:50]) for r in sel if not r["success"]][:6]}
        out[key].append(row)
        lab = f"{'<0' if lo < -1e8 else lo}..{'' if hi > 1e8 else hi}"
        print(f"  {lab:>7} mm: n {n:3d} success {row['success']:3d} ({100 * row['success'] / n:5.1f} %) glitch {row['glitch']} pushed {row['pushed']} housing {row['housing']} collateral {row['collateral']}  {row['failures'][:3]}")
json.dump(out, open(sys.argv[1].replace(".json", "_analysis.json"), "w"), indent=1)
