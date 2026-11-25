"""K4 (r10): how many of a pile's objects end up lying on another, over fresh layout draws of the default variant
(the declared stack_probability is the attempt rate per object after the first). usage: stacking_rate.py N OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
n = int(sys.argv[1]); robot = b.make_robot(0); task = robot._sim.on_reset
piles = stacked = objs = 0; per = []
for seed in range(820000, 820000 + n):
    robot._set_seed(seed); robot._reset()
    lay = task.layout
    if lay.pile:
        piles += 1; k = len(lay.objects); s = len(lay.heights); objs += k; stacked += s; per.append(s / k)
res = {"draws": n, "piles": piles, "pile_share": piles / n, "pile_objects": objs, "stacked_objects": stacked,
       "stacked_share_of_pile_objects": stacked / max(objs, 1), "attempt_rate_declared": 0.5,
       "note": "an object counts as stacked when the layout drops it onto another (Layout.heights)"}
json.dump(res, open(sys.argv[2], "w"), indent=1); print(res)
