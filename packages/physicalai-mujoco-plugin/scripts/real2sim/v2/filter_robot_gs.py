"""Drop desk-coloured Gaussians from the arm's own set (robot_gs.ply): bright, neutral ones within 2 cm of the desk top
are the desk the cut caught, not the arm; drawn on the moving arm they float over the desk as pale blobs.
usage: filter_robot_gs.py IN.ply OUT.ply   (the _links.json next to IN is filtered to OUT's name too)"""
import json, os, sys, numpy as np
from pathlib import Path
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]; M, t, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
plate = -cal["desk_top_z"]; src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_bytes(); end = raw.index(b"end_header\n") + len(b"end_header\n"); header = raw[:end].decode()
props = [l.split()[-1] for l in header.splitlines() if l.startswith("property")]
n = int(next(l for l in header.splitlines() if l.startswith("element vertex")).split()[-1])
V = np.frombuffer(raw[end:end + n * 4 * len(props)], np.float32).reshape(n, len(props))
P = V[:, [props.index(k) for k in ("x", "y", "z")]].astype(np.float64); Pw = sc * P @ M.T + t; z = Pw[:, 2] + plate
col = np.clip(0.5 + 0.2820948 * V[:, [props.index(f"f_dc_{i}") for i in range(3)]], 0, 1); lum = col.mean(1)
drop = (z < 0.02) & (lum > 0.62) & ((col.max(1) - col.min(1)) < 0.08)
links = json.load(open(src.with_name(src.stem + "_links.json")))
keep = ~drop
dst.write_bytes(header.replace(f"element vertex {n}", f"element vertex {int(keep.sum())}").encode() + V[keep].tobytes())
links["links"] = [l for l, k in zip(links["links"], keep) if k]
json.dump(links, open(dst.with_name(dst.stem + "_links.json"), "w")); print("dropped", int(drop.sum()), "of", n)
