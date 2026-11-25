"""Buckets from a straight-down render of the splat (no photo): a pinhole camera far above the desk plane (12 SfM units,
long focal, ~0.25 mm/px at the rim), each bucket's rim segmented by its colour around its seed, the outer contour's
minimum-area rectangle = the rim's outer outline at rim height (CAD: 150 x 202.5 mm at 120 mm). Gives each bucket's
centre and long-side direction in the plane frame and a scale (8 sides). Writes align/buckets_topdown.json and
notes/sheets/buckets_topdown.jpg.  usage: fit_buckets_topdown.py SPLAT.ply"""
import json, os, sys, numpy as np, cv2
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer
B = json.load(open(ROOT / "align/buckets3d.json")); pl = B["plane"]; o, u, v, n = (np.array(pl[k]) for k in "ouvn")
SEEDS = json.load(open(ROOT / "align/bucket_seeds.json"))["seeds_sfm"]
seeds = {c: np.array([(np.array(X) - o) @ u, (np.array(X) - o) @ v]) for c, X in SEEDS.items()}
ctr = np.mean(list(seeds.values()), 0)
S0 = 3.9; RIM = 0.120 * S0                    # rim height in units (approximate scale is enough: the camera is far)
Hc = 12.0; W = 2400; span = 1.5                # image covers 1.5 units either way
F = W / 2 / (span / (Hc - RIM))
cam = o + ctr[0] * u + ctr[1] * v + Hc * n; z = -n; x = u; y = np.cross(z, x)
T = np.eye(4); T[:3, :3] = np.stack([x, y, z]); T[:3, 3] = -T[:3, :3] @ cam
K = np.array([[F, 0, W / 2], [0, F, W / 2], [0, 0, 1.0]])
r = MetalSplatRenderer(sys.argv[1]).render(T, K, W, W, bg=(0.0, 0.0, 0.0))
img = (np.clip(r["rgb"], 0, 1) * 255).astype(np.uint8); bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(int); Hh, Ss, Vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
def to_px(p): return np.array([W / 2 + F * (p[0] - ctr[0]) / (Hc - RIM), W / 2 - F * (p[1] - ctr[1]) / (Hc - RIM)])   # image y = -v
def to_plane(px, h=RIM): px = np.asarray(px, float); return ctr + np.array([px[0] - W / 2, -(px[1] - W / 2)]) * (Hc - h) / F
desk = (Ss < 45) & (Vv > 150)
masks = {"green": (Hh >= 45) & (Hh <= 85) & (Ss > 80), "red": ((Hh <= 12) | (Hh >= 160)) & (Ss > 70) & (Vv > 50),
         "blue": (Hh >= 98) & (Hh <= 125) & (Ss > 90)}
colour_any = masks["green"] | masks["red"] | masks["blue"]
yy, xx = np.mgrid[0:W, 0:W]
sb = to_px(seeds["black"]); roi = np.hypot(xx - sb[0], yy - sb[1]) < 0.40 * F / (Hc - RIM)
masks["black"] = roi & ~desk & ~(masks["green"] | masks["blue"]) & ~(masks["red"] & (Vv > 120) & (Ss > 150))
out, sides, vis = {}, [], bgr.copy()
for col, m in masks.items():
    m = m.astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    k, lab, st, cents = cv2.connectedComponentsWithStats(m)
    sp = to_px(seeds[col]); big = [i for i in range(1, k) if st[i, cv2.CC_STAT_AREA] > 2000]
    i = min(big, key=lambda i: np.hypot(*(cents[i] - sp)))
    print(col, "seed px", sp.round(0), "components", [(int(st[j, cv2.CC_STAT_AREA]), cents[j].round(0).tolist()) for j in big][:6], flush=True)
    comp = (lab == i).astype(np.uint8)
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))   # fill the interior: the outer outline
    if col == "black":   # the contents are colourful: open away thin bridges to the desk's smears
        comp = cv2.morphologyEx(comp, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    short, long_ = sorted((w, h)); yaw_img = np.radians(ang) + (0 if w >= h else np.pi / 2)   # OpenCV: side w runs along ang
    yaw_long = -yaw_img   # the image's y axis is the plane's -v: angles flip
    s_short, s_long = short * (Hc - RIM) / F / 0.150, long_ * (Hc - RIM) / F / 0.2025
    out[col] = {"centre_plane_units": to_plane([cx, cy]).tolist(), "yaw_long": float(yaw_long), "s_short": float(s_short), "s_long": float(s_long),
                "fill": float(cv2.contourArea(cnt) / max(w * h, 1))}
    sides += [s_short, s_long]
    cv2.drawContours(vis, [np.int32(cv2.boxPoints(((cx, cy), (w, h), ang)))], -1, (0, 255, 255), 3)
    print(col, "s from short/long sides", round(s_short, 3), round(s_long, 3), "rect fill", round(out[col]["fill"], 3), flush=True)
s = float(np.median(sides))
json.dump({"buckets": out, "s_units_per_m": s, "sides": sides, "spread_pct": float(np.std(sides) / s * 100), "camera_height_units": Hc,
           "rim_height_m": 0.120}, open(ROOT / "align/buckets_topdown.json", "w"), indent=1)
cv2.imwrite(str(ROOT / "notes/sheets/buckets_topdown.jpg"), cv2.resize(vis, (1200, 1200)))
print("scale", round(s, 4), "spread %", round(float(np.std(sides) / s * 100), 2))
