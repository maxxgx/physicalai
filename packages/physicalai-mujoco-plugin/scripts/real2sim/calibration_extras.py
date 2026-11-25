"""Add the 1-sigma uncertainties, methods and the robot cameras to calibration.json (after make_calibration.py)."""
import json, sys, numpy as np, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import PLUGIN, ROOT
p = ROOT / "calibration.json"; cal = json.load(open(p))
xml = (PLUGIN / "urdf/scenes/bucket_sort_desk/scene.xml").read_text()
rig = re.search(r'<body name="overview_camera_rig" pos="([^"]+)" xyaxes="([^"]+)"', xml)
cal["uncertainty_1sigma"] = {
    "scale": {"value_units_per_m": cal["scale"]["chosen_units_per_m"], "sigma_rel": 0.010,
              "method": "bucket long sides (3 buckets, 3.244/3.261/3.262, spread 0.3%) plus 1% for the splat surface bias and print vs CAD; "
                        "robot CAD fit gives 3.30 with a flat loss over k 1.03-1.10 (align/robot_scale_profile.log): 1.4% apart"},
    "rotation_deg": {"sigma": 0.5, "method": "desk plane from splat depth at white-desk pixels (SfM points agree to 2 mm over 0.5 m) and the robot yaw fit"},
    "translation_m": {"sigma": 0.004, "method": "robot base position from the CAD fit (8.5 mm RMS over ~100k points)"},
    "robot_base": {"xy_m": 0.004, "yaw_deg": 0.8, "plate_thickness_m": 0.002,
                   "method": "CAD fit; the yaw agrees with the desk-edge normal to 0.8 deg"},
    "robot_joints_deg": {"sigma": [4.0, 4.0, 4.0, 6.0, 8.0, 8.0], "finger_opening_m": 0.003,
                         "method": "CAD fit at the zero rest pose (joints 1-6 and the finger opening); the wrist joints 5 and 6 are the least "
                                   "constrained (the lime shells end before the wrist; joint 5 fits 8.5 deg off zero)"},
    "buckets_m": {"xy": 0.005, "yaw_deg": 3.0, "method": "3D CAD fit to bucket-coloured splat points, 5.7 mm RMS (black: 7.6 mm, dark points)"},
}
cal["overview_camera"] = {
    "device": "Intel RealSense D4xx (the RealSense label is legible on the housing in allowed frame f00993; four-element front "
              "window seen in f01016); modelled as a D435: 90 x 25 x 25 mm bar, the upstream schematic's camera",
    "intrinsics": {"fovy_deg": 42.0, "sigma_deg": 3.0, "distortion": "none modelled", "resolution": [640, 480],
                   "derivation": "D435 RGB sensor (OV2740) datasheet: 69 x 42 deg (H x V) at 16:9 (1920x1080); the 640x480 stream is "
                                 "a 4:3 crop of the same height, so 42 deg vertical, 2 atan(tan(21 deg) 4/3) = 54 deg horizontal"},
    "extrinsics_scene": {"pos": [float(v) for v in rig.group(1).split()], "xyaxes": [float(v) for v in rig.group(2).split()]},
    "uncertainty_1sigma": {"pos_m": 0.015, "orientation_deg": 15.0},
    "method": "position: the bar's back face (RealSense label between its two mounting holes) triangulated from hand marks in allowed "
              "frames f00236 and f00991 (50 s apart, 1-2.5 px reprojection), plus half the bar's depth. Yaw: the back face's normal points "
              "within 3 deg of the schematic's task point. Pitch: NOT measured reliably; the marks suggest a nearly level bar (7 deg down, "
              "from one edge mark with 8-19 px reprojection), which would leave the task below the 42 deg view; the camera is aimed at "
              "the task point (50 deg down) as set up to the schematic. Housing: D435 bar 90 x 25 x 25 mm",
}
cal["wrist_camera"] = {
    "device": "Intel RealSense D405 on Seeed's D405_305_Mount (link6)",
    "mount": "REBOT_B601_WRIST_CAMERA: pos (0.1017, 0.0031, 0.0602) m in link6, measured (the lens midpoint triangulated from hand "
             "marks in allowed frames f00762, f00743, f00603; 2-5 px reprojection; the mount STEP's (0.10414, 0, 0.03915) was 21 mm "
             "short along the roll axis); looking along (-0.5, 0, 0.866) from the STEP",
    "intrinsics": {"fovy_deg": 58.0, "resolution": [640, 480], "distortion": "none modelled",
                   "derivation": "D405 colour (OV9782) datasheet: 87 x 58 deg (H x V) over the 1280x800 (16:10) stream. The 640x480 "
                                 "(4:3) stream keeps the full sensor height and crops the width (1067 of 1280 columns), so the vertical "
                                 "FOV stays 58 deg and the horizontal becomes 2 atan(tan(29 deg) 4/3) = 73.0 deg; fy = 240 / tan(29 deg) "
                                 "= 433 px. An SfM measurement on another D405 at 640x480 gave about 62 deg vertical (dataset-room project), "
                                 "so the datasheet value may be 2-4 deg low; to be measured on real wrist frames"},
    "uncertainty_1sigma": {"pos_m": 0.004, "orientation_deg": 5.0, "fovy_deg": 2.0},
    "bracket": "the printed brown mount, a box from 3 to 12 cm out along link6 +x, 4 to 8 cm along the roll axis, measured from the "
               "splat's brown Gaussians (624) at the calibrated pose; 94% of them lie on link6 +x, which confirms the side",
    "method": "camera position triangulated (see mount), orientation from the mount STEP; bracket and side measured from the phone splat",
}
json.dump(cal, open(p, "w"), indent=1); print("ok")
