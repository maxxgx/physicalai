"""Selection bias of recording successes only: the distribution of all drawn episodes (the specification as sampled)
against the distribution of the successful ones, by arrangement, named kind and number of targets.
usage: selection_bias.py RUN.json [RUN.json ...]"""
import sys, json, collections
rows = [x for p in sys.argv[1:] for x in json.load(open(p))]
def arrangement(x): return "scattered" if not x["pile"] else ("pile, stacked" if x["stacked"] else "pile")
def targets(x): n = len(x["targets"]); return "4" if n >= 4 else str(n)
out = {}
for name, key in (("arrangement", arrangement), ("kind", lambda x: x["kind"]), ("targets", targets)):
    alls = collections.Counter(key(x) for x in rows); succ = collections.Counter(key(x) for x in rows if x["success"])
    na, ns = sum(alls.values()), sum(succ.values())
    out[name] = {k: {"drawn": round(alls[k] / na, 3), "successful": round(succ[k] / ns, 3) if ns else 0, "success_rate": round(succ[k] / alls[k], 2)} for k in sorted(alls)}
print(json.dumps({"episodes": len(rows), "successes": sum(x["success"] for x in rows), **out}, indent=1))
