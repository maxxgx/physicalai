"""Bucket heights from the splat (no photo): the scene-frame z of the opaque Gaussians inside each bucket's calibrated
footprint, split into the half toward the closed end (tall end wall) and the open half (low lip); top = 99th
percentile. Compared with the CAD mesh's own tops. usage: bucket_heights.py SPLAT.ply"""
import json, os, sys, numpy as np, mujoco
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); PLUG = Path(__file__).resolve().parents[3]
cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]; M, t, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
plate = -cal["desk_top_z"]
raw = Path(sys.argv[1]).read_bytes(); end = raw.index(b"end_header\n") + 11; header = raw[:end].decode()
props = [l.split()[-1] for l in header.splitlines() if l.startswith("property")]; n = int(next(l for l in header.splitlines() if l.startswith("element vertex")).split()[-1])
V = np.frombuffer(raw[end:end + n * 4 * len(props)], np.float32).reshape(n, len(props))
P = V[:, [props.index(k) for k in "xyz"]].astype(float) @ M.T * sc + t; P[:, 2] += plate
op = 1 / (1 + np.exp(-V[:, props.index("opacity")])); P = P[op > 0.5]
m = mujoco.MjModel.from_xml_path(str(PLUG / "urdf/scenes/bucket_sort_desk_v2/scene.xml")); mid = m.mesh("bucket").id
Vm = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
gid = [g for g in range(m.ngeom) if m.geom_dataid[g] == mid][0]; R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[gid]); Vb = Vm @ R.reshape(3, 3).T + m.geom_pos[gid]
cad_closed = np.percentile(Vb[Vb[:, 1] < -0.05, 2], 99.5); cad_open = np.percentile(Vb[Vb[:, 1] > 0.05, 2], 99.5)
print(f"CAD: closed-end top {cad_closed * 1000:.1f} mm, open-end top {cad_open * 1000:.1f} mm, footprint {np.ptp(Vb[:, 0]) * 1000:.1f} x {np.ptp(Vb[:, 1]) * 1000:.1f} mm")
for c, b in cal["buckets"].items():
    cy, sy = np.cos(b["yaw"]), np.sin(b["yaw"]); L = (P[:, :2] - [b["x"], b["y"]]) @ np.array([[cy, -sy], [sy, cy]])
    inside = (np.abs(L[:, 0]) < 0.065) & (np.abs(L[:, 1]) < 0.09) & (P[:, 2] > 0.02) & (P[:, 2] < 0.25)
    closed = inside & (L[:, 1] < -0.05); opn = inside & (L[:, 1] > 0.05)
    f = lambda s: f"{np.percentile(P[s, 2], 99) * 1000:.1f}" if s.sum() > 20 else "n/a"
    print(f"{c}: closed-end top {f(closed)} mm ({closed.sum()} G), open-end top {f(opn)} mm ({opn.sum()} G)")
