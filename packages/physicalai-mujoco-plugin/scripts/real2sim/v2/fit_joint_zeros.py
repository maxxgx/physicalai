"""r17 item 1, H5 (coordinator, R-47: per unit): joint-zero offsets of joints 1-6 (+-3 deg, added to every recorded state)
fitted on TRAIN (0, 2, 6, 7, 9) over three terms together, chosen on DEV (1, 3, 10):
  fingers      fit_gripper_camera.py's finger edges and tips in the wrist view (px);
  wrist world  fit_wrist_world.py's bucket contours in the wrist view (px; the stored r17 layouts);
  overview arm fit_overview_arm.py's lime-shell distance in the overview (px).
J = the sum. The cameras stay as the kit has them (the wrist camera fitted to the fingers, the r17 overview camera).
Adopted only if all three terms hold or improve on DEV (a fit buying the wrist world with the overview arm or the fingers
is the degeneracy again). Bound use per joint reported. usage: fit_joint_zeros.py STATES_PARQUET FRAMES_X5 OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
ARGS, OUT = sys.argv[1:3], sys.argv[3]
def load(script, split_at):
    src = (HERE / script).read_text().split(split_at)[0]
    sys.argv = [str(HERE / script), *ARGS, "/dev/null"]
    g = {"__file__": str(HERE / script), "__name__": script}; exec(compile(src, script, "exec"), g); return g
W = load("fit_wrist_world.py", 'res = {"method"')
O = load("fit_overview_arm.py", "res = {")
KIT = json.load(open(os.environ["KIT"])); wc = KIT["cameras"]["wrist"]
W["set_camera"](wc["pos"], wc["quat"], wc["fovy"]); W["set_finger_offset"](np.zeros(2))
OFF = np.zeros(6)
def patch(g, ch, m, d):   # every recorded state set in that namespace gets the offsets
    jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]; orig = ch.set_positions
    def set_positions(state, *a, **k):
        out = orig(state, *a, **k)
        for i, adr in enumerate(jadr): d.qpos[adr] += np.radians(OFF[i])
        return out
    ch.set_positions = set_positions
patch(W, W["ch"], W["m"], W["d"]); patch(O, O["ch"], O["m"], O["d"])
r16 = lambda r: W["G0_MAP"] + 2 * W["S_MAP"] * abs(r)

def terms(split):
    f = W["fcost"](*W["measure"](W["FF"][split], r16)); w = W["world_cost"](split, r16)
    o = O["evaluate"](O["sets"][split])["cost_px"]
    return {"fingers": round(f, 2), "wrist_world": round(w, 2), "overview_arm": round(float(o), 2), "J": round(f + w + float(o), 2)}

def J(x):
    OFF[:] = 3.0 * np.tanh(x); return terms("train")["J"]
res = {"method": __doc__.split("usage")[0].strip(), "zero": {s: terms(s) for s in ("train", "dev")}}
print("zero", json.dumps(res["zero"]), flush=True)
fit = minimize(J, np.zeros(6), method="Nelder-Mead", options={"xatol": 0.01, "fatol": 0.02, "maxfev": 600,
               "initial_simplex": np.vstack([np.zeros(6)] + [np.eye(6)[i] * 0.3 for i in range(6)])})
OFF[:] = 3.0 * np.tanh(fit.x)
res["fit"] = {"offsets_deg": OFF.round(3).tolist(), "bound_use": [round(abs(v) / 3.0, 3) for v in OFF], "evals": int(fit.nfev),
              "train": terms("train"), "dev": terms("dev")}
z, f = res["zero"]["dev"], res["fit"]["dev"]
res["adopt"] = bool(all(f[k] <= z[k] + 0.05 for k in ("fingers", "wrist_world", "overview_arm")))
print("fit", json.dumps(res["fit"]), "adopt", res["adopt"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
