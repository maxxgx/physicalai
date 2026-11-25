"""The overview camera's pose in the SfM frame from the triangulated lens marks (align/overview_tri.json) and the splat's
housing Gaussians (the 42 x 42 x 23 mm white body around them).

- x axis (OpenCV, image right): from the colour lens (the camera's own left module = the bezel on the image's right when
  the phone faces the camera) to the other lens.
- z axis (forward): the housing's thin axis (PCA of its white Gaussians within 3 lens spacings), pointing from the body
  centre to the front face, made orthogonal to x. y = z x x.
- centre: the colour lens.
- mount: the ball head below the body and the pole down to the desk (dark Gaussians), for the scene's geoms.
Writes align/overview_cam.json.  usage: overview_pose.py SPLAT.ply"""
import json, os, sys, numpy as np, cv2
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
from physicalai_mujoco_plugin.real2sim.splats import read_ply

T = json.load(open(ROOT / "align/overview_tri.json")); B = json.load(open(ROOT / "align/buckets3d.json"))
n = np.array(B["plane"]["n"]); o = np.array(B["plane"]["o"]); s_u = B["s_units_per_m"]
Cl = np.array(T["lens_img_right"]["X"]); Ol = np.array(T["lens_img_left"]["X"]); b = float(np.linalg.norm(Ol - Cl))
x = (Ol - Cl) / b; mid = (Cl + Ol) / 2
s = read_ply(sys.argv[1]); keep = s.opacities > 0.4; P = s.centers[keep].astype(np.float64); C = np.clip(s.colors[keep], 0, 1)
hsv = cv2.cvtColor((C.reshape(-1, 1, 3) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float); S_, V = hsv[:, 1] / 255, hsv[:, 2] / 255
near = np.linalg.norm(P - mid, axis=1) < 0.045 * s_u
white = near & (S_ < 0.2) & (V > 0.55)
W = P[white]; Hc = np.median(W, 0)
ev, evec = np.linalg.eigh(np.cov((W - W.mean(0)).T)); thin = evec[:, 0]
if thin @ (mid - Hc) < 0: thin = -thin
z = thin - (thin @ x) * x; z /= np.linalg.norm(z); y = np.cross(z, x)
R = np.stack([x, y, z], 1)   # world_from_cv columns
depth = float((mid - Hc) @ z) / s_u
print(f"lens spacing {b / s_u * 1000:.1f} mm, {white.sum()} white housing Gaussians, face {depth * 1000:.1f} mm in front of their median "
      f"(box: 11.5), PCA sd mm {np.sqrt(ev) / s_u * 1000}", flush=True)
# camera tilt and heading in the desk frame
up = n; fwd_h = z - (z @ up) * up
pitch = float(np.degrees(np.arcsin(-z @ up)))
print(f"looks {pitch:.1f} deg below the desk plane", flush=True)
# mount: the pole under the body: dark Gaussians below the housing within 3 cm of its vertical axis, down to the desk
h = (P - o) @ n; hc = (Hc - o) @ n
lat = (P - Hc) - np.outer((P - Hc) @ n, n)
dark = (V < 0.3) & (np.linalg.norm(lat, axis=1) < 0.03 * s_u) & (h < hc) & (h > -0.02 * s_u)
D = P[dark]; dh = (D - o) @ n
ball = D[(dh > hc - 0.08 * s_u)].mean(0) if (dh > hc - 0.08 * s_u).any() else Hc - 0.04 * s_u * n
pole_top = ball - 0.02 * s_u * n
foot = Hc - hc * n + 0.0 * n
clamp = foot - 0.0 * n
print(f"pole: {dark.sum()} dark Gaussians; housing {hc / s_u * 1000:.0f} mm above the desk", flush=True)
json.dump({"centre_sfm": Cl.tolist(), "R_world_from_cv": R.tolist(), "lens_spacing_sfm": b, "housing_median_sfm": Hc.tolist(),
           "face_depth_m": depth, "pitch_below_desk_deg": pitch,
           "mount_sfm": {"ball_head": ball.tolist(), "pole_top": pole_top.tolist(), "clamp": (clamp - 0.02 * s_u * n).tolist()},
           "sigma_pos_m": 0.004, "sigma_rot_deg": 3.0,
           "method": "lens bezels triangulated in 3 allowed frames (f01128, f01142, f01160; 0.2-1.2 px reprojection) for the centre and the x axis; forward = thin axis of the splat's white housing Gaussians; Orbbec datasheet intrinsics"},
          open(ROOT / "align/overview_cam.json", "w"), indent=1)
