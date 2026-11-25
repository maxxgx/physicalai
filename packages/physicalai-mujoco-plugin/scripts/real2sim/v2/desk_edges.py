"""The white desk's edges near the robot from a straight-down splat render in the scene frame (no photo): white pixels
(S < 45, V > 165), then per scan line from the base outward the last white run: the far edge (+x), the right (-y) and
left (+y) edges as robust lines; the back edge from the calibration's desk (behind the base). Writes align/desk.json
(the scene's desk box) and notes/sheets/desk_edges.jpg.  usage: desk_edges.py SPLAT.ply"""
import json, os, sys, numpy as np, cv2
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer
cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]; M, t, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
plate = -cal["desk_top_z"]
X0, X1, Y0, Y1, RES = -0.3, 1.2, -1.2, 1.2, 0.002
W, H = int((Y1 - Y0) / RES), int((X1 - X0) / RES)
Hc = 6.0   # camera height (m): near-orthographic
def s2sfm(p): return M.T @ (np.asarray(p, float) - [0, 0, plate] - t) / sc
c = s2sfm([(X0 + X1) / 2, (Y0 + Y1) / 2, Hc]); z = M.T @ np.array([0, 0, -1.0]); x = M.T @ np.array([0, -1.0, 0]); y = np.cross(z, x)
T = np.eye(4); T[:3, :3] = np.stack([x, y, z]); T[:3, 3] = -T[:3, :3] @ c
F = Hc / RES; K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
r = MetalSplatRenderer(sys.argv[1]).render(T, K, W, H, bg=(0.0, 0.0, 0.0))
img = (np.clip(r["rgb"], 0, 1) * 255).astype(np.uint8); bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
white = ((hsv[..., 1] < 45) & (hsv[..., 2] > 165)).astype(np.uint8)
white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
def px(X, Y): return int(W / 2 - (Y - (Y0 + Y1) / 2) / RES), int(H / 2 - (X - (X0 + X1) / 2) / RES)   # image x = -scene y, image y = -scene x
def scan(points):   # last white pixel along each ray (list of (X, Y) samples from inside out) -> the edge point
    out = []
    for ray in points:
        vals = [white[min(max(py, 0), H - 1), min(max(px_, 0), W - 1)] for px_, py in (px(X, Y) for X, Y in ray)]
        vals = np.array(vals); run = np.convolve(vals.astype(float), np.ones(10), "valid") == 10
        if not run.any(): continue
        # from the start, follow the white until a gap of 2 cm
        i = 0; last = None
        while i < len(vals):
            if vals[i]: last = i; i += 1
            elif last is not None and i - last > 10: break
            else: i += 1
        if last is not None: out.append(ray[last])
    return np.array(out)
def robust_line(P, axis):   # fit coord[axis] = a * coord[1-axis] + b
    P = np.asarray(P); keep = np.ones(len(P), bool)
    for _ in range(6):
        a, b = np.polyfit(P[keep, 1 - axis], P[keep, axis], 1); res = np.abs(P[:, axis] - (a * P[:, 1 - axis] + b))
        keep = res < max(0.005, 2.5 * np.median(res[keep]))
    return a, b, int(keep.sum()), len(P)
far = scan([[(X, Yv) for X in np.arange(0.35, 1.2, RES)] for Yv in [*np.arange(-0.72, -0.42, 0.02), *np.arange(0.44, 0.72, 0.02)]])   # beside the bucket row
right = scan([[(Xv, Y) for Y in np.arange(-0.05, -1.2, -RES)] for Xv in np.arange(0.0, 0.28, 0.01)])   # in front of the buckets
left = scan([[(Xv, Y) for Y in np.arange(0.05, 1.2, RES)] for Xv in np.arange(0.0, 0.6, 0.02)])
af, bf, kf, nf = robust_line(far, 0); ar, br, kr, nr = robust_line(right, 1); al, bl, kl, nl = (0, 0, 0, 0) if len(left) < 3 else robust_line(left, 1)
print(f"far edge: x = {af:.4f} y + {bf:.4f} ({kf}/{nf}); right edge: y = {ar:.4f} x + {br:.4f} ({kr}/{nr}); left edge: y = {al:.4f} x + {bl:.4f} ({kl}/{nl})")
# the back edge: behind the base as the calibration had it (its x at y = 0)
d0 = cal["desk"]; yaw0 = d0["yaw"]; cy0, sy0 = np.cos(yaw0), np.sin(yaw0)
corners0 = np.array(d0["center"]) + np.array([[sx * d0["half"][0], sy * d0["half"][1]] for sx in (-1, 1) for sy in (-1, 1)]) @ np.array([[cy0, -sy0], [sy0, cy0]]).T
back = float(min(corners0[:, 0]))
# the back edge: the white run behind the base, beside the plate (scans from x = 0.3 backward)
bk = scan([[(X, Yv) for X in np.arange(0.3, -0.3, -RES)] for Yv in [*np.arange(-0.6, -0.15, 0.03), *np.arange(0.15, 0.6, 0.03)]])
if len(bk) >= 5: back = float(np.median(bk[:, 0])); print("back edge scans:", len(bk), "x", np.percentile(bk[:, 0], [10, 50, 90]).round(3))
yaw = float(np.arctan(-af))   # the far edge's direction (x = af y + bf): the desk turned by atan(-af) about z
x_far = bf; y_r = br + ar * 0.15
# the left edge runs into a neighbouring white desk (no white gap): placed from the desk's length measured on this same
# desk in v1 (1.550 m), from the right edge
y_l = y_r + 1.550
out = {"method": "white desk in a straight-down splat render (scene frame, 2 mm/px): far, right and left edges as robust lines from scans outward from the base (within 1.2 m); back edge behind the base from the earlier fit",
       "center": [float((back + x_far) / 2), float((y_r + y_l) / 2)], "half": [float((x_far - back) / 2), float((y_l - y_r) / 2)],
       "yaw": yaw, "height_m": d0.get("height_m", 0.7275), "sigma_m": 0.005,
       "edges": {"far": [af, bf, kf, nf], "right": [ar, br, kr, nr], "left_from_length_m": 1.550, "back_x": back}}
json.dump(out, open(ROOT / "align/desk.json", "w"), indent=1); print(json.dumps({k: out[k] for k in ("center", "half", "yaw")}))
vis = bgr.copy()
for P, col in ((far, (0, 0, 255)), (right, (0, 255, 0)), (left, (255, 0, 0))):
    for X, Y in P: cv2.circle(vis, px(X, Y), 4, col, -1)
cv2.rectangle(vis, px(x_far, y_l), px(back, y_r), (0, 255, 255), 3)
cv2.imwrite(str(ROOT / "notes/sheets/desk_edges.jpg"), cv2.resize(vis, (W // 2, H // 2)))
