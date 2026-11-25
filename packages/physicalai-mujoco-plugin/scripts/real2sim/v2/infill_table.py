"""Fill the desk under each real bucket with real desk Gaussians, so a bucket the randomisation moves leaves no hole.

The phone never saw the desk under the buckets. For each bucket's calibrated footprint (1.5 cm wider), a patch of the
same shape is copied from the nearest clean desk area: a shifted footprint whose Gaussians all lie within 6 mm of the
desk plane (nothing standing on it) and that keeps 8 cm from every bucket, the robot base and the desk edge. The copies
are translated (and given the desk's height) into the hole, under the bucket's own Gaussians, which hide them while the
bucket stands there. Writes OUT.ply (input + copies) and OUT_infill.json.  usage: infill_table.py IN.ply OUT.ply"""
import json, os, sys, numpy as np
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]; M, t, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
plate = -cal["desk_top_z"]
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_bytes(); end = raw.index(b"end_header\n") + len(b"end_header\n"); header = raw[:end].decode()
props = [l.split()[-1] for l in header.splitlines() if l.startswith("property")]
n = int(next(l for l in header.splitlines() if l.startswith("element vertex")).split()[-1])
V = np.frombuffer(raw[end:end + n * 4 * len(props)], np.float32).reshape(n, len(props)).copy()
xyz = [props.index(k) for k in ("x", "y", "z")]
P = V[:, xyz].astype(np.float64); Pw = sc * P @ M.T + t; Pw[:, 2] += plate     # scene frame, desk top z = 0
d = cal["desk"]; cy, sy = np.cos(d["yaw"]), np.sin(d["yaw"]); Rd = np.array([[cy, -sy], [sy, cy]])
def in_desk(xy, margin):
    L = (xy - d["center"]) @ Rd; return (np.abs(L[:, 0]) < d["half"][0] - margin) & (np.abs(L[:, 1]) < d["half"][1] - margin)
HALF = (0.075 + 0.015, 0.10125 + 0.015)
def footprint(xy, b, shift=np.zeros(2)):
    c, s_ = np.cos(b["yaw"]), np.sin(b["yaw"]); R = np.array([[c, -s_], [s_, c]])
    L = (xy - [b["x"], b["y"]] - shift) @ R; return (np.abs(L[:, 0]) < HALF[0]) & (np.abs(L[:, 1]) < HALF[1])
near_xy = Pw[:, :2]; low = np.abs(Pw[:, 2]) < 0.006; tall = (Pw[:, 2] > 0.006) & (Pw[:, 2] < 0.4)
copies, log = [], {}
for col, b in ([] if True else cal["buckets"].items()):   # replaced by the synthetic desk below
    best = None
    for dx in np.arange(-0.4, 0.41, 0.02):
        for dy in np.arange(-0.6, 0.61, 0.02):
            sh = np.array([dx, dy])
            if np.hypot(*sh) < 0.25: continue
            fp = footprint(near_xy, b, sh)
            if not fp.any() or tall[fp].any(): continue
            corners = np.array([[b["x"], b["y"]]]) + sh
            if not in_desk(corners, 0.10).all(): continue
            if any(np.hypot(*(corners[0] - [o["x"], o["y"]])) < 0.30 for o in cal["buckets"].values()): continue
            if np.hypot(*corners[0]) < 0.25: continue
            k = int((fp & low).sum())
            if best is None or abs(np.hypot(*sh)) < best[0]: best = (float(np.hypot(*sh)), sh, k)
    if best is None:
        print(col, "no clean patch found"); continue
    _, sh, k = best; fp = footprint(near_xy, b, sh) & low
    C = V[fp].copy(); Q = Pw[fp].copy(); Q[:, :2] -= sh; Q[:, 2] = np.clip(Q[:, 2], -0.004, 0.0015)
    Q[:, 2] -= plate; C[:, xyz] = (((Q - t) @ M) / sc).astype(np.float32)   # back to the SfM frame (M orthonormal)
    copies.append(C); log[col] = {"shift_m": sh.round(3).tolist(), "gaussians": int(len(C))}
    print(col, "patch from", sh.round(2), len(C), "Gaussians", flush=True)
# the desk under the real arm: the phone never saw it (the arm lay flat over the object area). Everything over the
# arm's footprint (its CAD at the calibrated joints projected on the desk, 3 cm wider; not the base plate, not the
# buckets) from 1 cm below the desk to 30 cm above it is deleted and the footprint refilled from a clean desk patch.
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import cad  # noqa: E402
Warm = cad.arm_points(np.r_[cal["robot"]["joints_rad"], 0.0], cad.geom_local_points(per_geom=400))
g = 0.005; lo = np.array([-0.2, -0.6]); N = int(1.4 / g)
occ = np.zeros((N, N), np.uint8); ij = np.floor((Warm[:, :2] - lo) / g).astype(int); ok = (ij >= 0).all(1) & (ij < N).all(1)
occ[ij[ok, 1], ij[ok, 0]] = 1
occ = cv2.dilate(cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
# the whole workspace desk in front of the base (where the arm lay, its cable and shadow, and the objects go): rebuilt
# from clean desk, outside the buckets (1 cm wider) and the base plate
WS = (0.06, 0.52, -0.44, 0.44)
yy_, xx_ = np.mgrid[0:N, 0:N]; wx, wy = xx_ * g + lo[0] + g / 2, yy_ * g + lo[1] + g / 2
ws = (wx > WS[0]) & (wx < WS[1]) & (wy > WS[2]) & (wy < WS[3])
for b_ in cal["buckets"].values():
    c_, s__ = np.cos(b_["yaw"]), np.sin(b_["yaw"]); Lb = np.stack([wx - b_["x"], wy - b_["y"]], -1) @ np.array([[c_, -s__], [s__, c_]])
    ws &= ~((np.abs(Lb[..., 0]) < 0.075 + 0.01) & (np.abs(Lb[..., 1]) < 0.10125 + 0.01))
occ = np.maximum(occ, ws.astype(np.uint8))
def in_arm(xy):
    ij = np.floor((xy - lo) / g).astype(int); ok = (ij >= 0).all(1) & (ij < N).all(1); out = np.zeros(len(xy), bool)
    out[ok] = occ[ij[ok, 1], ij[ok, 0]] > 0; return out
plate_xy = (np.abs(near_xy[:, 0]) < 0.08) & (np.abs(near_xy[:, 1]) < 0.11)
under = in_arm(near_xy) & ~plate_xy & (Pw[:, 2] > -0.08) & (Pw[:, 2] < 0.30)
for b in cal["buckets"].values():
    under &= ~footprint(near_xy, b, np.zeros(2))
# the base plate: the sim draws it (the robot model's plate, L4 within ~14 px), and the splat's own plate and the dark
# Gaussians at its edges doubled it as a dark band and streaks in the wrist view (r04 report, R-10). The robot cameras'
# splat drops everything low over the plate's footprint plus 1.5 cm (desk_compositor); the desk under it is laid here.
PLATE = (0.08 + 0.015, 0.11 + 0.015)
# (the robot cameras' splat drops them in the compositor; the phone path keeps the splat's plate)
keepV = ~under
print("under the arm: deleted", int(under.sum()), "Gaussians", flush=True)
# refill with synthetic desk: flat Gaussians every 1 cm at the desk's height over the footprint (and under the buckets),
# each coloured by the median of the bright real desk Gaussians within 8 cm around it (outside the deleted set and the
# dark smears), so the desk's soft shading carries on and nothing of the arm, its cable or shadow is left
from scipy.spatial import cKDTree
lumV = 0.5 + 0.2820948 * V[:, [props.index(f"f_dc_{i}") for i in range(3)]].mean(1)
colV = np.clip(0.5 + 0.2820948 * V[:, [props.index(f"f_dc_{i}") for i in range(3)]], 0, 1)
neutral = (colV.max(1) - colV.min(1)) < 0.05
deskref = low & ~under & neutral & (lumV > 0.72) & (np.hypot(near_xy[:, 0], near_xy[:, 1]) < 1.2)
tree = cKDTree(near_xy[deskref]); dcs = V[deskref][:, [props.index(f"f_dc_{i}") for i in range(3)]]
fill = occ.copy(); robot_fill = np.zeros_like(occ)
for b_ in cal["buckets"].values():
    c_, s__ = np.cos(b_["yaw"]), np.sin(b_["yaw"]); Lb = np.stack([wx - b_["x"], wy - b_["y"]], -1) @ np.array([[c_, -s__], [s__, c_]])
    # robot cameras only (tagged, dropped from the phone path): under it and 7 cm around, where the robot cameras draw
    # the sim bucket over the splat without the bucket and its ghosts (desk_compositor GHOST_MARGIN, 6 cm)
    fill |= ((np.abs(Lb[..., 0]) < 0.075 - 0.008) & (np.abs(Lb[..., 1]) < 0.10125 - 0.008)).astype(np.uint8)   # inside: hidden while it stands
    robot_fill |= ((np.abs(Lb[..., 0]) < 0.075 + 0.07) & (np.abs(Lb[..., 1]) < 0.10125 + 0.07)).astype(np.uint8)
fill &= ~((np.abs(wx) < 0.075) & (np.abs(wy) < 0.105))
robot_fill |= ((np.abs(wx) < PLATE[0] + 0.01) & (np.abs(wy) < PLATE[1] + 0.01)).astype(np.uint8)   # under and around the plate
robot_fill &= ~fill   # one desk Gaussian per cell
step = 2   # 1 cm grid on the 5 mm occupancy grid
def grid(mask, seed):
    ys_, xs_ = np.nonzero(mask[::step, ::step]); q = np.c_[xs_ * step * g + lo[0] + g / 2, ys_ * step * g + lo[1] + g / 2]
    q = q + np.random.default_rng(seed).uniform(-0.002, 0.002, q.shape)
    return q[in_desk(q, 0.012)]   # never past the desk's edges
pts_main, pts_robot = grid(fill, 3), grid(robot_fill, 4)
pts = np.r_[pts_main, pts_robot]
_, nn = tree.query(pts, k=64); col = np.median(dcs[nn], axis=1)
S = np.zeros((len(pts), len(props)), np.float32)
Q3 = np.c_[pts, np.full(len(pts), -0.0005 - plate)]
S[:, xyz] = (((Q3 - t) @ M) / sc).astype(np.float32)
for i in range(3): S[:, props.index(f"f_dc_{i}")] = col[:, i]
S[:, props.index("opacity")] = 4.0
S[len(pts_main):, props.index("opacity")] = 3.99   # the robot-camera-only desk: the compositor's phone path drops this tag
unit = 1.0 / sc   # SfM units per metre
for i, sz in enumerate((0.0085, 0.0085, 0.0006)): S[:, props.index(f"scale_{i}")] = np.log(sz * unit)
# flat in the desk plane: the rotation taking the Gaussian's z to the desk normal (the scene's z in the SfM frame)
nz = M.T @ np.array([0.0, 0.0, 1.0]); ax = np.cross([0, 0, 1.0], nz); ang = np.arccos(np.clip(nz[2], -1, 1))
qv = np.r_[np.cos(ang / 2), (ax / (np.linalg.norm(ax) + 1e-12)) * np.sin(ang / 2)]
for i in range(4): S[:, props.index(f"rot_{i}")] = qv[i]
copies.append(S); log["synthetic_desk"] = {"gaussians": int(len(pts_main)), "robot_camera_only": int(len(pts_robot)), "deleted": int(under.sum()), "workspace_m": list(WS)}
print("synthetic desk:", len(S), "Gaussians over the workspace and under the buckets; deleted", int(under.sum()), flush=True)
V = V[keepV]
out = np.vstack([V, *copies]).astype(np.float32)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(header.replace(f"element vertex {n}", f"element vertex {len(out)}").encode() + out.tobytes())
json.dump(log, open(dst.with_name(dst.stem + "_infill.json"), "w"), indent=1); print("wrote", dst, len(out))
