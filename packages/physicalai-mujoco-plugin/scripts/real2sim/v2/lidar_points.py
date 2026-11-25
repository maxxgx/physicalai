"""Sample the office LiDAR mesh (room frame: metres, +Y up, floor at 0) into a Z-up point cloud with normals and colours.
usage: lidar_points.py MESH.ply OUT.npz [N]   (private capture: OUT stays under REAL2SIM_ROOT/lidar, never in Git)"""
import sys, numpy as np, trimesh
m = trimesh.load(sys.argv[1], process=False); n = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000
P, fi = trimesh.sample.sample_surface(m, n, seed=0)
N = m.face_normals[fi]
C = m.visual.vertex_colors[m.faces[fi, 0], :3] if hasattr(m.visual, "vertex_colors") else np.zeros((n, 3), np.uint8)
Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # room (x, y up, z) -> (x, -z, y): Z up, right-handed
np.savez_compressed(sys.argv[2], P=(P @ Z.T).astype(np.float32), N=(N @ Z.T).astype(np.float32), C=C.astype(np.uint8))
print(len(P), (P @ Z.T).min(0).round(2), (P @ Z.T).max(0).round(2))
