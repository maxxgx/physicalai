"""r15: grasp quality of two `kinds` runs on the same seeds (e.g. r14 against r15): success, glitches, max penetration,
pre-grasp pushes, housing contacts, clean, median episode time, and per grasp the snap (first 0.2 s) and slide (after),
and the drawn gap of resistor grasps on the caps (opening > 19 mm) and at the waist.
usage: kinds_compare.py A.json B.json [LABEL_A LABEL_B]"""
import json, sys
import numpy as np

def load(p):
    r = json.load(open(p)); return {e["seed"]: e for e in (r if isinstance(r, list) else r["episodes"])}

def summary(E):
    g = [x | {"obj": o} for e in E.values() for o, xs in (e.get("grasps") or {}).items() for x in xs]
    caps = [x["drawn_gap_mm_p95_max"][1] for x in g if "resistor" in x["obj"] and x["opening_mm"] > 19 and x.get("drawn_gap_mm_p95_max")]
    waist = [x["drawn_gap_mm_p95_max"][1] for x in g if "resistor" in x["obj"] and x["opening_mm"] <= 19 and x.get("drawn_gap_mm_p95_max")]
    q = lambda v: [round(float(np.median(v)), 2), round(float(np.percentile(v, 95)), 2), round(float(np.max(v)), 2)] if len(v) else None
    return {"episodes": len(E), "success": sum(e["success"] for e in E.values()), "clean": sum(bool(e.get("clean")) for e in E.values()),
            "glitches": sum(e["glitch"] for e in E.values()), "max_penetration_mm": round(max(e["max_penetration_m"] for e in E.values()) * 1000, 2),
            "pre_grasp_pushes_over_1cm_s": sum((e.get("pre_grasp_speed_mps") or 0) > 0.01 for e in E.values()), "housing_contact_episodes": sum(e.get("housing_contact_ticks", 0) > 0 for e in E.values()),
            "median_sim_s": round(float(np.median([e["sim_s"] for e in E.values()])), 2), "grasps": len(g),
            "snap_mm_median_p95_max": q([x["snap_first_0p2s_mm"] for x in g if x.get("snap_first_0p2s_mm") is not None]),
            "slide_mm_median_p95_max": q([x["slide_after_0p2s_mm"] for x in g if x.get("slide_after_0p2s_mm") is not None]),
            "resistor_cap_drawn_gap_max_mm_median_p95_max": q(caps), "n_cap_grasps": len(caps),
            "resistor_waist_drawn_gap_max_mm_median_p95_max": q(waist), "failed": sorted((s, e.get("failure")) for s, e in E.items() if not e["success"])}

A, B = load(sys.argv[1]), load(sys.argv[2]); la, lb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
common = sorted(set(A) & set(B))
out = {la: summary({s: A[s] for s in common}), lb: summary({s: B[s] for s in common}), "seeds": [common[0], common[-1], len(common)]}
for k in (la, lb): print(k, json.dumps(out[k]))
print(json.dumps(out), file=sys.stderr)
