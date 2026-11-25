"""Robot fit error with the scale factor k held at fixed values (everything else re-fitted from the best fit)."""
import json, sys, numpy as np
from pathlib import Path
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
import fit_robot as R
from common import ROOT
b = json.load(open(ROOT / "align/robot_fit.json"))
p0 = np.r_[b["base_xy_m"], b["base_yaw"], b["base_dz_m"], np.log(b["k_robot_over_bucket"]), b["joints_rad"]]
for k in (0.97, 1.0, 1.03, 1.064, 1.10):
    f = lambda q: R.loss(np.r_[q[:4], np.log(k), q[4:]])
    r = minimize(f, np.r_[p0[:4], p0[5:]], method="Powell", options={"maxiter": 8000, "xtol": 1e-5, "ftol": 1e-10})
    P = R.model_pts(np.r_[r.x[:4], np.log(k), r.x[4:]])
    a = R.treeA.query(P)[0]; lime = R.cKDTree(P[R.is_lime]).query(R.L)[0]
    print(f"k {k:.3f} rms mm {np.sqrt(r.fun / 2) * 1000:.2f} | model->splat median mm {np.median(a) * 1000:.1f} | lime->model median mm {np.median(lime) * 1000:.1f} | dz {r.x[3] * 1000:.1f} mm", flush=True)
