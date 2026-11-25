"""The real arm's Gaussians from the splat that still holds it (v0), each attached to the arm link nearest to it at the
calibrated static pose: within 1.2 cm of the arm CAD, or within 8 cm of the wrist (the camera and its bracket), above the
desk. Writes align/robot_gs.ply (a 3DGS PLY with those Gaussians, SfM frame) and align/robot_gs_links.json."""
import json, sys, numpy as np, mujoco
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
import cad
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "splat/v0/out/export_30000.ply"
OUT = sys.argv[2] if len(sys.argv) > 2 else "robot_gs"
cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]; M, t, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
plate = -cal["desk_top_z"]
m, d = cad.model(); local = cad.geom_local_points(per_geom=2500)
q = np.r_[cal["robot"]["joints_rad"], cal["robot"].get("finger_opening_m", 0.0)]
W = cad.arm_points(q, local)                                   # robot-base frame
body = np.concatenate([np.full(len(p), m.geom_bodyid[g]) for g, (_, p) in local.items()])
raw = SRC.read_bytes(); end = raw.index(b"end_header\n") + len(b"end_header\n"); header = raw[:end].decode()
props = [l.split()[-1] for l in header.splitlines() if l.startswith("property")]
n = int(next(l for l in header.splitlines() if l.startswith("element vertex")).split()[-1])
V = np.frombuffer(raw[end:end + n * 4 * len(props)], np.float32).reshape(n, len(props))
P = V[:, [props.index(k) for k in ("x", "y", "z")]].astype(np.float64); Pw = sc * P @ M.T + t
dist, nn = cKDTree(W).query(Pw)
wrist_ids = [m.body(b).id for b in ("link6", "gripper_end", "gripper_left", "gripper_right")]
wb = np.isin(body, wrist_ids); dw, _ = cKDTree(W[wb]).query(Pw)
dc = V[:, [props.index(f"f_dc_{i}") for i in range(3)]]; lum = 0.5 + 0.2820948 * dc.mean(1)
# the arm, its wrist camera and bracket, and its cable: dark Gaussians within 10 cm of the arm above the desk ride along
# too (clean_splat.py deletes the same ones from the scene, so no cut piece of the cable stays behind)
# everything within 10 cm of the arm (the cable's highlights too) and the dark cable up to 20 cm, above the desk;
# clean_splat.py deletes the same set from the scene
sel = ((dist < 0.20) | (dw < 0.08)) & (Pw[:, 2] > -plate + 0.012)   # one radius for every colour: no splinters left
# and the cable where it lies on the desk (dark, at desk level, outside the base plate): it belongs with the arm
zd = Pw[:, 2] + plate; off_plate = (np.abs(Pw[:, 0]) > 0.075) | (np.abs(Pw[:, 1]) > 0.105)
sel |= (lum < 0.3) & (dist < 0.20) & (zd > -0.005) & (zd <= 0.012) & off_plate
# the clamp and the cables at the desk's front edge and behind it stay in the scene (they never move)
sel &= ~((Pw[:, 0] < -0.04) & (dist >= 0.012) & (Pw[:, 2] + plate < 0.03))
link = body[nn[sel]]
# the camera and bracket points near the wrist belong to link6
link[(dw[sel] < 0.08) & (dist[sel] >= 0.012) & (lum[sel] >= 0.3) & (dist[sel] < dw[sel] + 0.03)] = m.body("link6").id
# and so does everything inside the bracket's box (link6 frame: 2-13 cm out along x, |y| < 3 cm, 2-9 cm along z)
b6 = m.body("link6").id; L6 = (Pw[sel] - d.xpos[b6]) @ d.xmat[b6].reshape(3, 3)
link[(L6[:, 0] > 0.02) & (L6[:, 0] < 0.13) & (np.abs(L6[:, 1]) < 0.03) & (L6[:, 2] > 0.02) & (L6[:, 2] < 0.09)] = b6
out = V[sel]
(ROOT / f"align/{OUT}.ply").write_bytes(header.replace(f"element vertex {n}", f"element vertex {len(out)}").encode() + out.tobytes())
# only the turret rides on base_link (drawn at full weight always); the cable at the base rides on link1 with the arm
link[(link == m.body("base_link").id) & (dist[sel] >= 0.012)] = m.body("link1").id
names = [m.body(int(b)).name for b in link]
json.dump({"links": names, "joints_rad": q.tolist(), "source": str(SRC)}, open(ROOT / f"align/{OUT}_links.json", "w"))
from collections import Counter
print("robot Gaussians", len(out), Counter(names).most_common())
