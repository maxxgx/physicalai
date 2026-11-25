"""Straight-down render of the splat over the task (desk-plane frame), with a 10 cm grid, the buckets and the robot base."""
import json, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer

f = json.load(open(ROOT / "align/buckets3d.json")); s = f["s_units_per_m"]
pl = f["plane"]; o, u, v, n = (np.array(pl[k]) for k in "ouvn")
cx, cy, H, W, F = 0.05, 0.0, 1.6, 1400, 1100.0  # look at (cx, cy) from H metres up
c = o + s * (cx * u + cy * v + H * n)
z = -n; x = u; y = np.cross(z, x)
T = np.eye(4); T[:3, :3] = np.stack([x, y, z]); T[:3, 3] = -T[:3, :3] @ c
K = np.array([[F, 0, W / 2], [0, F, W / 2], [0, 0, 1.0]])
out = MetalSplatRenderer(ROOT / "splat/v0/out/export_30000.ply").render(T, K, W, W, bg=(0.15, 0.15, 0.15))
img = cv2.cvtColor((np.clip(out["rgb"], 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR); raw = img.copy()
def px(X, Y):  # plane metres -> pixel (on the desk, z = 0)
    P = o + s * (X * u + Y * v); q = K @ (T[:3, :3] @ P + T[:3, 3]); return int(q[0] / q[2]), int(q[1] / q[2])
for t in np.arange(-1.0, 1.01, 0.1):
    a, b = px(t, -1.0), px(t, 1.0); cv2.line(img, a, b, (90, 90, 200) if abs(t) > 1e-6 else (0, 0, 255), 1)
    a, b = px(-1.0, t), px(1.0, t); cv2.line(img, a, b, (90, 90, 200) if abs(t) > 1e-6 else (0, 0, 255), 1)
for t in np.arange(-0.5, 1.01, 0.5):
    cv2.putText(img, f"x{t:+.1f}", px(t, -0.95), 0, 0.5, (0, 0, 255), 1); cv2.putText(img, f"y{t:+.1f}", px(-0.95, t), 0, 0.5, (0, 0, 255), 1)
r = json.load(open(ROOT / "align/robot_fit.json")) if (ROOT / "align/robot_fit.json").exists() else None
if r: cv2.circle(img, px(*r["base_xy_m"]), 8, (0, 255, 0), 2)
cv2.imwrite(str(ROOT / "align/topdown.jpg"), img); print("ok")

# the desk's front and back edges: per column, the white run through the bucket row, as lines in plane metres
hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
white = ((hsv[..., 1] < 50) & (hsv[..., 2] > 165)).astype(np.uint8)
white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
inv = np.linalg.inv(K)
def to_plane(xp, yp):  # pixel -> plane metres at z = 0
    d = T[:3, :3].T @ (inv @ np.array([xp, yp, 1.0])); t = -(c - o) @ n / (d @ n); P = c + t * d
    return float((P - o) @ u / s), float((P - o) @ v / s)
front, back = [], []
for X in np.arange(-0.45, 0.85, 0.02):
    col = [px(X, Y) for Y in np.arange(-0.6, 0.8, 0.002)]
    vals = np.array([white[min(max(q[1], 0), W - 1), min(max(q[0], 0), W - 1)] for q in col])
    Ys = np.arange(-0.6, 0.8, 0.002)
    run = np.convolve(vals.astype(float), np.ones(15), "valid") == 15  # 3 cm of white in a row
    if not run.any(): continue
    lo, hi = int(np.argmax(run)), int(len(run) - 1 - np.argmax(run[::-1]) + 14)
    front.append((X, Ys[lo])); back.append((X, Ys[hi]))
def robust_line(pts):
    pts = np.array(pts); keep = np.ones(len(pts), bool)
    for _ in range(5):
        a, b = np.polyfit(pts[keep, 0], pts[keep, 1], 1); r = np.abs(pts[:, 1] - (a * pts[:, 0] + b))
        keep = r < max(0.01, 2.5 * np.median(r[keep]))
    return a, b, keep.sum(), len(pts)
ba, bb, bk, bn = robust_line(back)
fr = np.array([q for q in front if (-0.35 < q[0] < -0.12 or 0.22 < q[0] < 0.6) and -0.6 < q[1] < 0.0])
fa = ba; fb = float(np.median(fr[:, 1] - ba * fr[:, 0])); fk, fn = int(np.sum(np.abs(fr[:, 1] - (fa * fr[:, 0] + fb)) < 0.01)), len(fr)
depth = abs(bb - fb) / np.hypot(1, fa)
print(f"front y = {fa:.4f} x + {fb:.4f} ({fk}/{fn}); back y = {ba:.4f} x + {bb:.4f} ({bk}/{bn})")
print("angles deg", round(np.degrees(np.arctan(fa)), 2), round(np.degrees(np.arctan(ba)), 2), "| desk depth m", round(depth, 4))
json.dump({"front": [fa, fb], "back": [ba, bb], "depth_m": depth}, open(ROOT / "align/desk_edges.json", "w"))
for (a, b, col) in ((fa, fb, (255, 0, 255)), (ba, bb, (255, 0, 255))):
    cv2.line(img, px(-0.9, a * -0.9 + b), px(1.0, a * 1.0 + b), col, 2)
cv2.imwrite(str(ROOT / "align/topdown.jpg"), img)
