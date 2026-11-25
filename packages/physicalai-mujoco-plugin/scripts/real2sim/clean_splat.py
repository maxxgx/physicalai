"""Delete the real arm's leftovers from a splat: Gaussians within R of the arm CAD at the calibrated static pose
(more around the wrist camera mount), except on the desk surface. usage: clean_splat.py IN.ply OUT.ply [R_m] [WRIST_m] [UNMASKED.ply]
With UNMASKED.ply (a splat of the same frames trained with the robot visible, splat/v0): the desk and plate around the base
(within 15 cm, up to 1.2 cm above the desk) come from it instead: the masked splat never saw them well (a pale smear), and
there the real cable lies on the desk."""
import json, sys, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
import cad

src, dst = Path(sys.argv[1]), Path(sys.argv[2]); R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.04; RW = float(sys.argv[4]) if len(sys.argv) > 4 else 0.08
cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]; M, t, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
local = cad.geom_local_points(per_geom=1500)
W = cad.arm_points(np.r_[cal["robot"]["joints_rad"], 0.0], local)
m, _ = cad.model(); gid = np.concatenate([np.full(len(p), g) for g, (_, p) in local.items()])
wrist = np.isin([m.geom_bodyid[g] for g in gid], [m.body(b).id for b in ("link6", "gripper_end", "gripper_left", "gripper_right")])
raw = src.read_bytes(); end = raw.index(b"end_header\n") + len(b"end_header\n"); header = raw[:end].decode()
props = [l.split()[-1] for l in header.splitlines() if l.startswith("property")]
assert all(l.split()[1] == "float" for l in header.splitlines() if l.startswith("property")), "float properties only"
n = int(next(l for l in header.splitlines() if l.startswith("element vertex")).split()[-1])
V = np.frombuffer(raw[end:end + n * 4 * len(props)], np.float32).reshape(n, len(props))
P = V[:, [props.index(k) for k in ("x", "y", "z")]].astype(np.float64)
Pw = sc * P @ M.T + t                                   # SfM -> robot-base frame (desk top at z = -plate)
plate = -cal["desk_top_z"]
d_arm = cKDTree(W).query(Pw)[0]; d_wr = cKDTree(W[wrist]).query(Pw)[0]
dc0 = V[:, [props.index(f"f_dc_{i}") for i in range(3)]]; lum0 = 0.5 + 0.2820948 * dc0.mean(1)
# the arm's leftovers, and its dark cable within 10 cm of the arm (robot_gaussians.py carries the same ones on the links);
# the desk top and the plate stay
drop = ((d_arm < 0.20) | (d_wr < RW)) & (Pw[:, 2] > -plate + 0.012)   # the same set robot_gaussians.py carries
drop &= ~((Pw[:, 0] < -0.04) & (d_arm >= 0.012) & (Pw[:, 2] + plate < 0.03))   # the clamp and the edge cables stay
# the real arm's cable lying on the desk in front of the base (scene frame: x 0.07-0.21 m, y -0.03-0.04 m)
z_desk = Pw[:, 2] + plate                                # height above the desk top
dc = V[:, [props.index(f"f_dc_{i}") for i in range(3)]]; lum = 0.5 + 0.2820948 * dc.mean(1)   # SH DC -> colour
cable = (Pw[:, 0] > 0.07) & (Pw[:, 0] < 0.21) & (Pw[:, 1] > -0.03) & (Pw[:, 1] < 0.04) & (z_desk < 0.25) & (lum < 0.35)
drop |= cable
# the real arm's shadow baked into the desk: desk-level Gaussians in front of the base darker than the desk around
# them get the desk's median colour (SH DC and the view-dependent terms)
desk = (np.abs(z_desk) < 0.006) & ~drop
area = desk & (Pw[:, 0] > 0.05) & (Pw[:, 0] < 0.40) & (np.abs(Pw[:, 1]) < 0.25)
ring = desk & (Pw[:, 0] > 0.05) & (Pw[:, 0] < 0.40) & (np.abs(Pw[:, 1]) > 0.25) & (np.abs(Pw[:, 1]) < 0.40)
ref_lum = np.median(lum[ring]); shaded = area & (lum < ref_lum - 0.04) & (lum > 0.45)
rest = [i for i, k in enumerate(props) if k.startswith("f_rest_")]
V = V.copy(); V[np.ix_(shaded, [props.index(f"f_dc_{i}") for i in range(3)])] = np.median(dc[ring], 0)
V[np.ix_(shaded, rest)] = np.median(V[np.ix_(ring, rest)], 0)
print("cable", int(cable.sum()), "| desk de-shadowed", int(shaded.sum()), "ref lum", round(float(ref_lum), 3))
keep = V[~drop]
if len(sys.argv) > 5:
    raw0 = Path(sys.argv[5]).read_bytes(); end0 = raw0.index(b"end_header\n") + len(b"end_header\n"); h0 = raw0[:end0].decode()
    assert [l.split()[-1] for l in h0.splitlines() if l.startswith("property")] == props, "same PLY layout"
    n0 = int(next(l for l in h0.splitlines() if l.startswith("element vertex")).split()[-1])
    V0 = np.frombuffer(raw0[end0:end0 + n0 * 4 * len(props)], np.float32).reshape(n0, len(props))
    P0 = sc * V0[:, [props.index(k) for k in ("x", "y", "z")]].astype(np.float64) @ M.T + t
    near_base = lambda Q: (np.hypot(Q[:, 0], Q[:, 1]) < 0.15) & (Q[:, 2] + plate < 0.012) & (Q[:, 2] + plate > -0.01)
    da0 = cKDTree(W).query(P0)[0]
    take = near_base(P0) & ~((da0 < 0.012) & (P0[:, 2] + plate > 0.012))
    kmask = ~near_base(Pw[~drop])
    keep = np.concatenate([keep[kmask], V0[take]])
    print("desk around the base from the unmasked splat:", int(take.sum()), "in,", int((~kmask).sum()), "out")
# the cable where it lies on the desk rides with the arm (robot_gaussians.py): here those Gaussians take the desk's colour
Pk = sc * keep[:, [props.index(k) for k in ("x", "y", "z")]].astype(np.float64) @ M.T + t; zk = Pk[:, 2] + plate
dck = keep[:, [props.index(f"f_dc_{i}") for i in range(3)]]; lumk = 0.5 + 0.2820948 * dck.mean(1)
dk = cKDTree(W).query(Pk)[0]; offp = (np.abs(Pk[:, 0]) > 0.075) | (np.abs(Pk[:, 1]) > 0.105)
cab = (lumk < 0.3) & (dk < 0.20) & (zk > -0.005) & (zk <= 0.012) & offp & (Pk[:, 0] >= -0.04)
deskk = (np.abs(zk) < 0.006) & (lumk > 0.6) & (dk < 0.40) & offp
# deleted, not re-coloured: a re-coloured cable stands 5 mm proud of the desk and hides resting objects behind it; the
# white desk around closes the gap
keep = keep[~cab]
print("cable on the desk deleted:", int(cab.sum()))
# desk-coloured Gaussians standing above the desk where the task objects lie (the real desk there is empty): they hide
# the bottoms of resting objects
raised = (Pk[:, 0] > 0.10) & (Pk[:, 0] < 0.42) & (np.abs(Pk[:, 1]) < 0.30) & (zk > 0.003) & (zk < 0.05) & (lumk > 0.55) & ~cab
keep = keep[~raised[~cab]]
print("raised desk-coloured Gaussians in the object area deleted:", int(raised.sum()))
dst.write_bytes(header.replace(f"element vertex {n}", f"element vertex {len(keep)}").encode() + keep.tobytes())
print("dropped", int(drop.sum()), "of", n, "->", dst)
