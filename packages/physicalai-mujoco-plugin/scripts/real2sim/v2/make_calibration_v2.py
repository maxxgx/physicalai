"""Compose REAL2SIM_ROOT/calibration.json (v2 schema; RUBRIC L9) from the fits:
  align/buckets3d.json  desk plane + four buckets (bucket CAD, shared scale)      fit_desk_buckets.py
  align/robot_fit.json  robot base, plate, joints, free robot/bucket scale k       ../fit_robot.py
  align/lidar_reg.json  scan registration: scale ratio, desk outline and height   lidar_register.py   (optional)
  align/overview_cam.json  overview pose in the SfM frame                          overview_pose.py    (optional)
The scene frame: robot base at the origin facing +x, z up, metres, desk top at z = 0 (the plate under the robot).
usage: make_calibration_v2.py [--scale-source bucket|lidar|mean] [--write-world-v0]"""
import argparse, json, os, sys, numpy as np
from pathlib import Path
from scipy.optimize import minimize
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
ap = argparse.ArgumentParser(); ap.add_argument("--scale-source", default="mean"); ap.add_argument("--write-world-v0", action="store_true")
a = ap.parse_args()
B = json.load(open(ROOT / "align/buckets3d.json")); sb = B["s_units_per_m"]
pl = B["plane"]; o, u, v, n = (np.array(pl[k]) for k in "ouvn"); A = np.stack([u, v, n])
rb = json.load(open(ROOT / "align/robot_fit.json")); k_robot = rb["k_robot_over_bucket"]
lid = json.load(open(ROOT / "align/lidar_reg.json")) if (ROOT / "align/lidar_reg.json").exists() and not a.write_world_v0 else None
# scale references, all as SfM units per metre (sb is only the frame of the fits: plane metres at sb units/m)
refs = {"bucket_cad": B.get("bucket_scale_units_per_m", sb), "robot_cad": sb * k_robot}
lidar_note = None
if lid:
    # the scan registration ran in align/world_v0.json (4.0 units/m): scan metres = ratio x those metres
    lidar_note = {"units_per_m": 4.0 / lid["scale_ratio_scan_over_cad"], "icp_inlier_frac": lid["icp_inlier_frac"],
                  "used": False, "why": "weak registration (inlier fraction and scale drift with the truncation); reported, not used"}
S = float(np.mean([refs["bucket_cad"], refs["robot_cad"]])) if a.scale_source == "mean" else refs[a.scale_source + "_cad"]
spread = (max(refs.values()) - min(refs.values())) / S * 100
k = S / sb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_robot as R  # noqa: E402  (loads align/robot_pts.npz)
p0 = np.r_[rb["base_xy_m"], rb["base_yaw"], rb["base_dz_m"], rb["joints_rad"]]
r = minimize(lambda q: R.loss(np.r_[q[:4], np.log(k), q[4:]]), p0, method="Powell", options={"maxiter": 8000, "xtol": 1e-6, "ftol": 1e-11})
bx, by, yaw, dz = r.x[:4]; joints = r.x[4:]
c, s_ = np.cos(-yaw), np.sin(-yaw); Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
Mrot = Rz @ A; t = -(Mrot @ o) / S - Rz @ np.array([bx, by, dz]) / k
plate = float(dz / k)
def to_world_q(q): return (Rz @ (np.asarray(q, float) - [bx, by, dz])) / k
buckets = {}
for col, b in B["buckets"].items():
    w = to_world_q([b["x"], b["y"], 0.0])
    buckets[col] = {"x": float(w[0]), "y": float(w[1]), "yaw": float(np.arctan2(np.sin(b["yaw"] - yaw), np.cos(b["yaw"] - yaw))),
                    "sigma_m": 0.012, "sigma_yaw_rad": float(np.radians(4.0)), "fit_rms_mm": B["per_bucket"][col]["rms_mm"],
                    "method": "rim in a straight-down splat render (contents smear over it)"}
# refined by silhouettes in allowed phone frames (fit_buckets_photo.py), already in the scene frame
_bh = ROOT / "align/bucket_height.json"
bucket_height = json.load(open(_bh)) if _bh.exists() else None
_bp = ROOT / "align/buckets_photo.json"
if _bp.exists():
    for col, _r in json.load(open(_bp))["buckets"].items():
        buckets[col].update({"x": _r["x"], "y": _r["y"], "yaw": _r["yaw"], "sigma_m": 0.010 if col == "black" else 0.006, "sigma_yaw_rad": float(np.radians(3.0 if col == "black" else 2.0)),
                             "method": f"CAD silhouette vs colour silhouette in {_r['frames']} allowed phone frames (median IoU {_r['iou_median_after']:.2f} at 1/4 size), started from the straight-down rim fit; black on its dark silhouette)"})
# real buckets stand touching: fitted CAD footprints may overlap by a few mm (within their sigmas), which the simulation
# would resolve by shoving them apart at every reset. Separate each overlapping pair symmetrically along the line
# between their centres until 1 mm apart.
from physicalai_mujoco_plugin.bucket_sort import Pose2D as _P, bucket_footprint as _fp, polygon_gap as _gap
import itertools as _it
for _a, _b in _it.combinations(sorted(buckets), 2):
    A, B = buckets[_a], buckets[_b]; d_ = np.array([B["x"] - A["x"], B["y"] - A["y"]]); d_ /= np.linalg.norm(d_); moved = 0.0
    while _gap(_fp(_P(A["x"], A["y"], A["yaw"])), _fp(_P(B["x"], B["y"], B["yaw"]))) < 0.001 and moved < 0.01:
        A["x"] -= d_[0] * 0.00025; A["y"] -= d_[1] * 0.00025; B["x"] += d_[0] * 0.00025; B["y"] += d_[1] * 0.00025; moved += 0.00025
    if moved: print(f"buckets {_a} and {_b} overlapped: each moved {moved * 1000:.2f} mm apart"); A["separated_mm"] = B["separated_mm"] = round(moved * 1000, 2)
world = {"scale": 1.0 / S, "rotation": Mrot.tolist(), "translation": t.tolist(), "plate_m": plate}
if a.write_world_v0:
    json.dump(world, open(ROOT / "align/world_v0.json", "w"), indent=1); print("wrote align/world_v0.json; plate", round(plate, 4)); sys.exit()

def sfm_to_scene(P):
    return (1.0 / S) * np.asarray(P) @ Mrot.T + t + [0, 0, plate]

# desk: the scan's top outline (scene frame), a minimum-area rectangle; height from the scan
desk = {"method": "fallback: 1.55 x 0.79 m centred on the bucket row", "center": [0.30, 0.0], "half": [0.40, 0.775], "yaw": 0.0, "height_m": 0.75}
if os.environ.get("DESK_JSON"):
    desk = json.load(open(os.environ["DESK_JSON"]))
elif lid and False:
    import cv2
    xy = np.array(lid["desk"]["outline_scene_xy"], np.float32)
    (cx, cy), (w_, h_), ang = cv2.minAreaRect(xy)
    # the scan outline is in provisional (bucket-scale) metres: rescale to the chosen scale
    f = sb / S
    desk = {"method": "LiDAR scan (2026-11-03) top outline, 5 cm cells, minimum-area rectangle; height above the floor from the scan",
            "center": [float(cx * f), float(cy * f)], "half": [float(w_ / 2 * f), float(h_ / 2 * f)], "yaw": float(np.radians(ang)),
            "height_m": float(lid["desk"]["height_m"] * f), "sigma_m": 0.025}
ov = None
if (ROOT / "align/overview_cam.json").exists():
    OC = json.load(open(ROOT / "align/overview_cam.json"))
    Rsfm = np.array(OC["R_world_from_cv"]); c_sfm = np.array(OC["centre_sfm"])
    Rw = Mrot @ Rsfm                                   # camera axes (OpenCV) in the scene frame
    Rmj = Rw @ np.diag([1.0, -1.0, -1.0])             # MuJoCo camera: x right, y up, -z forward
    import mujoco
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rmj.flatten())
    pos = sfm_to_scene(c_sfm)
    b = OC["lens_spacing_sfm"] / S
    mt = {k_: sfm_to_scene(np.array(v_)).tolist() for k_, v_ in OC["mount_sfm"].items()}
    ov = {"device": "Orbbec Gemini 305 (colour module, UVC 640x480)", "pos": pos.tolist(), "quat": q.tolist(),
          "fovy": 68.0, "intrinsics": {"model": "pinhole, no distortion (datasheet: < 1.5 %)", "width": 640, "height": 480,
                                         "fx": 355.8, "fy": 355.8, "cx": 320.0, "cy": 240.0,
                                         "derivation": "datasheet V1.0 table 4-7-3: colour 640x480 V68 deg -> fy = 240/tan(34 deg) = 355.8 px; 4:3 crop of the 1280x800 sensor, square pixels"},
          "lens_spacing_m": float(b), "housing_pos_cam": [float(b / 2), 0.0, 0.0115],
          "sigma_pos_m": OC.get("sigma_pos_m", 0.006), "sigma_rot_deg": OC.get("sigma_rot_deg", 3.0), "method": OC["method"],
          "mount": {"clamp": mt["clamp"], "pole_top": mt["pole_top"], "ball_head": mt["ball_head"], "pole_radius": 0.011, "ball_radius": 0.018}}
wrist = {"device": "Orbbec Gemini 305 in the printed brown mount (Seeed D405_305_Mount)", "body": "link6",
         "pos": [0.1017, 0.0031, 0.0602], "quat": [0.18301270189221933, 0.6830127018922193, 0.6830127018922193, 0.18301270189221933],
         "fovy": 68.0, "intrinsics": {"model": "pinhole, no distortion", "width": 640, "height": 480, "fx": 355.8, "fy": 355.8, "cx": 320.0, "cy": 240.0,
                                      "derivation": "as the overview (same device)"},
         "sigma_pos_m": 0.005, "sigma_rot_deg": 3.0,
         "method": "the mount STEP pose corrected by the v1 phone measurement (+21 mm along the roll axis; same printed mount, the Gemini 305 has the D405's 42 x 42 x 23 mm body and 18 mm stereo baseline, colour = left module like the D405's left imager); not re-measured in v2 yet"}
_ap = ROOT / "align/arm_photo.json"
joints_method = "reBot CAD fitted to the splat's arm points (fit_robot.py)"
if _ap.exists():   # refined on the arm's lime shells in allowed phone frames (fit_arm_photo.py)
    _a = json.load(open(_ap)); joints = np.asarray(_a["joints_rad"], float)
    joints_method += f"; then refined on the lime shells' silhouettes in {_a['frames']} allowed frames (fit_arm_photo.py, IoU {_a['iou_median_before']:.3f} -> {_a['iou_median_after']:.3f})"
out = {
    "units": "metres, scene frame: robot base origin, +x forward, z up; desk top at z = 0, robot on a plate of plate_m",
    "sfm_to_world": {"scale": 1.0 / S, "rotation": Mrot.tolist(), "translation": t.tolist(),
                     "note": "robot-base frame = scale * rotation @ p_sfm + translation (desk top at z = desk_top_z); scene = that + [0, 0, plate]",
                     "sigma": {"rotation_deg": 0.3, "translation_m": 0.003, "scale_pct": round(max(0.3, spread / 2), 3)},
                     "method": "desk plane by RANSAC on the splat's white Gaussians; scale and buckets from the bucket CAD; robot base, yaw, plate and joints from the reBot CAD fitted to the splat's arm points (fit_robot.py)"},
    "scale": {"chosen_units_per_m": S, "source": a.scale_source, "references_units_per_m": refs, "spread_pct": round(spread, 3),
              "sigma_pct": round(max(0.3, spread / 2), 3),
              "lidar": lidar_note,
              "notes": "bucket_cad: outer rims of the green, red and blue buckets (150 x 202.5 mm) in a straight-down splat render, 6 sides; robot_cad: free scale of the reBot CAD fitted to the splat's arm points (fit_robot.py)"},
    "robot": {"profile": "rebot_b601", "joints_rad": joints.tolist(), "joints_deg": np.degrees(joints).round(2).tolist(), "joints_sigma_rad": 0.03,
              "joints_method": joints_method,
              "finger_opening_m": 0.09, "finger_opening_sigma_m": 0.01, "plate_thickness_m": plate, "fit_rms_mm": float(np.sqrt(r.fun / 2) * 1000),
              "base_sigma_m": 0.004, "base_yaw_sigma_deg": 1.0, "method": "reBot CAD (Menagerie seeed_rebot_devarm) fitted to splat points: lime shells + all non-white points near them; static through the video"},
    "desk_top_z": -plate, "desk": desk, "buckets": buckets,
    "bucket_moves": json.load(open(ROOT / "align/bucket_moves.json"))["moves"] if (ROOT / "align/bucket_moves.json").exists() else [],
    "bucket_height_scale": bucket_height["scale"] if bucket_height else 1.0, "bucket_height_scale_sigma": bucket_height["sigma"] if bucket_height else None, "overview": ov, "wrist": wrist,
}
json.dump(out, open(ROOT / "calibration.json", "w"), indent=1)
print(json.dumps({k_: out[k_] for k_ in ("scale", "robot", "desk", "buckets")}, indent=1)[:2500])
