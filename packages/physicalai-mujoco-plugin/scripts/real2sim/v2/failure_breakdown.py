"""r16: failure causes per baseline set (kinds-style episode lists): bucket refusal, other planner refusal (cuts), reach
(no lean / off the path), no settle in time, no lift, dropped (slipped, landed outside, wrong bucket), other; glitches;
hangs by the literal rule (sim time > 3 x the set's median) and by U-8 (sim time / targets > 3 x the set's median per
target, every target object counted). usage: failure_breakdown.py NAME=FILE[,FILE...] ...   (prints a table, JSON)"""
import json, sys
import numpy as np

def cause(r):
    f = (r.get("failure") or "")
    if r["success"]: return None
    for key, label in (("into a bucket", "bucket refusal"), ("cuts into another object", "other planner refusal"), ("no lean", "reach"),
                       ("misses its path", "reach"), ("no settled success", "no settle / 45 s"), ("did not lift", "no lift"),
                       ("slipped", "dropped"), ("landed outside", "dropped"), ("ended up in", "dropped")):
        if key in f: return label
    return "other: " + f[:40]

out = {}
for arg in sys.argv[1:]:
    name, files = arg.split("=", 1); rows = [r for f in files.split(",") for r in json.load(open(f))]
    med = float(np.median([r["sim_s"] for r in rows])); n_t = [max(1, len(r.get("targets") or [1])) for r in rows]
    med_t = float(np.median([r["sim_s"] / k for r, k in zip(rows, n_t)]))
    causes = {}
    for r in rows:
        c = cause(r)
        if c: causes.setdefault(c, []).append(r["seed"])
    hang_lit = [r["seed"] for r in rows if r["success"] and r["sim_s"] > 3 * med]
    hang_u8 = [r["seed"] for r, k in zip(rows, n_t) if r["success"] and r["sim_s"] > 3 * med_t * k]
    out[name] = {"n": len(rows), "success": sum(r["success"] for r in rows), "clean": sum(r["success"] and not r.get("unclean_reasons") for r in rows),
                 "glitch": [r["seed"] for r in rows if r["glitch"]], "causes": causes, "median_s": round(med, 2), "hang_literal": hang_lit,
                 "median_per_target_s": round(med_t, 2), "hang_u8": hang_u8,
                 "success_under_literal": sum(r["success"] for r in rows) - len(hang_lit), "success_under_u8": sum(r["success"] for r in rows) - len(hang_u8)}
    o = out[name]
    print(f"{name}: n {o['n']} success {o['success']} clean {o['clean']} glitch {len(o['glitch'])} | " + ", ".join(f"{k} {len(v)}" for k, v in sorted(causes.items()))
          + f" | hangs literal {len(hang_lit)} (median {o['median_s']} s), U-8 {len(hang_u8)} (per target {o['median_per_target_s']} s)")
json.dump(out, open("/dev/stdout", "w") if False else open(sys.argv[0].replace(".py", "_last.json"), "w"), indent=1)
