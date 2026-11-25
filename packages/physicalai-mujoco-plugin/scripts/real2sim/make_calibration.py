"""Compose align/*.json into calibration.json: the SfM -> sim-world similarity (robot base at the origin, facing +x,
z up, metres), the robot's static joints and plate, the buckets, the desk edges, with the evidence for the scale.

usage: make_calibration.py [S_UNITS_PER_M]   (default: the bucket long sides, 3.255)"""
import json, sys, numpy as np
from pathlib import Path
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT

S = float(sys.argv[1]) if len(sys.argv) > 1 else 3.255
B = json.load(open(ROOT / "align/buckets3d.json")); sb = B["s_units_per_m"]
pl = B["plane"]; o, u, v, n = (np.array(pl[k]) for k in "ouvn"); A = np.stack([u, v, n])
k = S / sb
# the robot base pose at this scale: refit base and joints with k held (from the free-k fit)
import fit_robot as R
rb = json.load(open(ROOT / "align/robot_fit.json"))
p0 = np.r_[rb["base_xy_m"], rb["base_yaw"], rb["base_dz_m"], rb["joints_rad"]]
r = minimize(lambda q: R.loss(np.r_[q[:4], np.log(k), q[4:]]), p0, method="Powell", options={"maxiter": 8000, "xtol": 1e-6, "ftol": 1e-11})
bx, by, yaw, dz = r.x[:4]; joints = r.x[4:]
c, s_ = np.cos(-yaw), np.sin(-yaw); Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
# world = Rz (q - b) / k with q = A (P - o) / sb  =>  world = (1/S) Rz A P + t
M = Rz @ A; t = -(M @ o) / S - Rz @ np.array([bx, by, dz]) / k
def to_world_q(q): return (Rz @ (np.asarray(q, float) - [bx, by, dz])) / k
buckets = {}
for col, b in B["buckets"].items():
    w = to_world_q([b["x"], b["y"], 0.0]); buckets[col] = {"x": float(w[0]), "y": float(w[1]), "yaw": float(np.arctan2(np.sin(b["yaw"] - yaw), np.cos(b["yaw"] - yaw)))}
E = json.load(open(ROOT / "align/desk_edges.json"))
def edge(a, b0):  # y = a x + b0 in plane metres -> a point and direction in world
    p = to_world_q([0.0, b0, 0.0]); d = Rz @ np.array([1.0, a, 0.0]); d /= np.linalg.norm(d); return p[:2], d[:2]
fp, fd = edge(*E["front"]); bp, bd = edge(*E["back"])
nrm = np.array([-fd[1], fd[0]])
out = {
    "units": "metres, sim world: robot base origin, +x forward, z up; desk top at z = desk_top_z",
    "sfm_to_world": {"scale": 1.0 / S, "rotation": M.tolist(), "translation": t.tolist(), "note": "world = scale * rotation @ p_sfm + translation"},
    "scale": {"chosen_units_per_m": S, "source": "bucket long sides (202.5 mm CAD), mid-wall extents, 3 buckets",
              "bucket_length_estimates": [3.244, 3.261, 3.262], "bucket_rim_height_estimates": [3.141, 3.186, 3.131],
              "bucket_cad_fit_shared": sb, "robot_cad_fit_free_k": rb["k_robot_over_bucket"] * sb,
              "robot_rms_mm_by_k": "see align/robot_scale_profile.log (shallow: 9.4 / 8.9 / 8.5 mm at k 0.97 / 1.00 / 1.064)"},
    "robot": {"profile": "rebot_b601", "joints_rad": joints.tolist(), "joints_deg": np.degrees(joints).round(2).tolist(),
              "plate_thickness_m": float(dz / k), "fit_rms_mm": float(np.sqrt(r.fun / 2) * 1000), "base_yaw_in_plane_rad": float(yaw)},
    "desk_top_z": float(-dz / k),
    "desk": {"front_edge_point": fp.tolist(), "front_edge_dir": fd.tolist(), "back_edge_point": bp.tolist(),
             "depth_m": float(abs((bp - fp) @ nrm)), "front_edge_distance_from_base_m": float(abs(fp @ nrm))},
    "buckets": buckets,
}
json.dump(out, open(ROOT / "calibration.json", "w"), indent=1)
print(json.dumps({k_: out[k_] for k_ in ("robot", "desk_top_z", "desk", "buckets")}, indent=1))
