"""Register the office LiDAR scan (lidar/room_pts.npz, Z up, metres, floor at 0) to the scene frame of a provisional
calibration (align/world_v0.json: SfM -> world similarity from the bucket/robot CAD fits), with a similarity (yaw,
translation, scale) so the scan's metric scale is a second, independent scale reference.

Source points: the splat's opaque centres in the world frame (static structure only: below 0.02 m under the desk top,
or more than 0.9 m from the robot base; the desk top itself). Target: the scan's points. Search: each scan desk
(horizontal patch 0.6-0.95 m high) x yaw every 10 deg x scale 0.97/1.0/1.03, then point-to-plane ICP.

Writes align/lidar_reg.json: world_from_room (4x4 incl. scale), scale ratio (scan metres per CAD metre) with its
spread over bootstrap subsets, the desk outline and height from the scan, residuals.
usage: lidar_register.py SPLAT.ply"""
import json, os, sys, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
from physicalai_mujoco_plugin.real2sim.splats import read_ply

W0 = json.load(open(ROOT / "align/world_v0.json")); M, t, sc = np.array(W0["rotation"]), np.array(W0["translation"]), W0["scale"]
plate = W0["plate_m"]
Zd = np.load(ROOT / "align/dense_pts.npz"); P0 = Zd["P"][np.random.default_rng(2).choice(len(Zd["P"]), 2_000_000, replace=False)].astype(np.float64)
P = sc * P0 @ M.T + t; P[:, 2] += plate   # scene frame: desk top at z = 0 (dense splat-depth points of allowed views)
r = np.hypot(P[:, 0], P[:, 1])
src = P[((P[:, 2] < -0.02) | (r > 0.9) | (np.abs(P[:, 2]) < 0.004)) & (r < 4.0) & (P[:, 2] > -1.2) & (P[:, 2] < 2.2)]
rng = np.random.default_rng(0); src = src[rng.choice(len(src), min(60000, len(src)), replace=False)]
L = np.load(ROOT / "lidar/room_pts.npz"); Q, N = L["P"].astype(np.float64), L["N"].astype(np.float64)
tree = cKDTree(Q)

# scan desks: horizontal points 0.6-0.95 m high, 5 cm grid, connected components
h = (N[:, 2] > 0.95) & (Q[:, 2] > 0.6) & (Q[:, 2] < 0.95)
cells = {}
for p in Q[h]:
    cells.setdefault((int(np.floor(p[0] / 0.05)), int(np.floor(p[1] / 0.05))), []).append(p[2])
seen, desks = set(), []
for c0 in cells:
    if c0 in seen: continue
    stack, comp = [c0], []
    while stack:
        c = stack.pop()
        if c in seen or c not in cells: continue
        seen.add(c); comp.append(c)
        stack += [(c[0] + dx, c[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    if len(comp) < 150: continue   # < 0.375 m^2
    xy = (np.array(comp) + 0.5) * 0.05; z = float(np.median(np.concatenate([cells[c] for c in comp])))
    desks.append({"centre": xy.mean(0), "cells": len(comp), "z": z, "xy": xy})
print("scan desks:", [(d["centre"].round(2).tolist(), round(d["z"], 3), d["cells"]) for d in desks], flush=True)

def transform(x, yaw, scale, tr):
    c, s_ = np.cos(yaw), np.sin(yaw); R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    return scale * x @ R.T + tr, R

def score(x):
    d, _ = tree.query(x, distance_upper_bound=0.15); d = np.where(np.isfinite(d), d, 0.15); return float(np.mean(d ** 2))

# the splat desk's centre in the scene frame (white points at desk height)
deskpts = P[np.abs(P[:, 2]) < 0.004]; dc = np.median(deskpts[:, :2], 0)
best = None; sub = src[rng.choice(len(src), 8000, replace=False)]
for D in desks:
    for yaw in np.radians(np.arange(0, 360, 10)):
        for scale in (0.97, 1.0, 1.03):
            c, s_ = np.cos(yaw), np.sin(yaw); R = np.array([[c, -s_], [s_, c]])
            tr = np.r_[D["centre"] - scale * R @ dc, D["z"]]
            x, _ = transform(sub, yaw, scale, tr); f = score(x)
            if best is None or f < best[0]: best = (f, yaw, scale, tr)
print("coarse", round(np.sqrt(best[0]) * 1000, 1), "mm rms; yaw", round(np.degrees(best[1]), 1), "scale", best[2], flush=True)

def icp(x0, yaw, scale, tr, iters=40, trunc=0.05):
    for it in range(iters):
        x, R = transform(x0, yaw, scale, tr)
        d, i = tree.query(x, distance_upper_bound=trunc); ok = np.isfinite(d)
        q, n = Q[i[ok]], N[i[ok]]; p0 = x0[ok]
        # linearised point-to-plane in (dyaw, dlogscale, dtx, dty, dtz) around the current estimate
        xr = x[ok]; rel = xr - tr
        Jn = np.stack([(np.cross([0, 0, 1.0], rel) * n).sum(1), (rel * n).sum(1), n[:, 0], n[:, 1], n[:, 2]], 1)
        res = ((q - xr) * n).sum(1)
        w = 1.0 / np.maximum(np.abs(res), 0.005)
        delta = np.linalg.lstsq(Jn * w[:, None], res * w, rcond=None)[0]
        yaw += delta[0]; scale *= np.exp(delta[1]); tr = tr + delta[2:]
        if np.abs(delta).max() < 1e-6: break
    x, R = transform(x0, yaw, scale, tr); d, _ = tree.query(x, distance_upper_bound=trunc)
    return yaw, scale, tr, float(np.median(d[np.isfinite(d)])), float(np.isfinite(d).mean())

_, yaw, scale, tr = best
for trunc in (0.10, 0.05, 0.025):
    yaw, scale, tr, med, frac = icp(src, yaw, scale, tr, trunc=trunc)
    print(f"icp trunc {trunc}: scale {scale:.5f} yaw {np.degrees(yaw):.2f} median {med * 1000:.1f} mm inliers {frac:.2f}", flush=True)
boot = []
for b in range(8):
    idx = rng.choice(len(src), len(src) // 2, replace=False)
    boot.append(icp(src[idx], yaw, scale, tr, iters=15, trunc=0.025)[1])
sd = float(np.std(boot))
# the scan's desk under the robot: the component nearest the transformed scene origin
c, s_ = np.cos(yaw), np.sin(yaw); Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
origin_room = tr; D = min(desks, key=lambda D: np.linalg.norm(D["centre"] - origin_room[:2]))
xy_scene = ((np.c_[D["xy"], np.full(len(D["xy"]), D["z"])] - tr) @ Rz) / scale
out = {"room_from_scene": {"yaw": float(yaw), "scale": float(scale), "translation": tr.tolist(),
                           "note": "room = scale * Rz(yaw) @ scene + translation; scene = robot-base frame with the desk top at z = 0"},
       "scale_ratio_scan_over_cad": float(scale), "scale_ratio_sd_bootstrap": sd, "icp_median_mm": med * 1000, "icp_inlier_frac": frac,
       "desk": {"height_m": float(D["z"] / scale), "cells_5cm": D["cells"], "outline_scene_xy": xy_scene[:, :2].round(4).tolist()},
       "scan_desks": [{"centre": d_["centre"].tolist(), "z": d_["z"], "cells": d_["cells"]} for d_ in desks]}
json.dump(out, open(ROOT / "align/lidar_reg.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("desk", "scan_desks")}, indent=1), "desk height", round(D["z"] / scale, 4))
