"""r15 (D2, the user's mount): the wrist mount's drawing, collision and planning proxies, from its STEP geometry.

Input: $REAL2SIM_ROOT/mount/mount_link6.stl and mount_link6.json (mount_from_step.py: the STEP placed on the CAD gripper's
bolt holes, link6 frame, metres). Output: the plugin profile's data file profiles/rebot_b601_wrist_mount.json:
  mount         the drawn mesh (vertices, faces; link6), its collision as convex pieces (collision_decomp.decompose, the
                r13 tolerance 1.2 mm, signed audit collision_decomp.report) and planning proxies (build_plan_proxies.py's
                greedy merge, 6 mm as the gripper housing), the default colour (brown: the user's print)
  camera_body   the Gemini 305 (42 x 42 x 23 mm) as a box on the cradle seat: drawn and collided by the same box
usage: build_mount_collision.py"""
import json, os, sys, time
from pathlib import Path
import numpy as np, trimesh
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from collision_decomp import decompose, report, oriented_samples, inside_hulls
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); MD = ROOT / "mount"
OUT = HERE.parents[2] / "src/physicalai_mujoco_plugin/profiles/rebot_b601_wrist_mount.json"
src = (HERE / "build_plan_proxies.py").read_text()
ns = {"np": np}; exec("from scipy.spatial import ConvexHull\nfrom collision_decomp import inside_hulls, thicken\n" + src[src.index("def proxies("):src.index("def drawn_samples(")], ns)
mesh = trimesh.load(MD / "mount_link6.stl", process=True); V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
info_json = json.load(open(MD / "mount_link6.json")); pri = info_json["camera_prior"]
t = time.time(); PIECES = int(sys.argv[sys.argv.index("--pieces") + 1]) if "--pieces" in sys.argv else 128
hulls, info = decompose(V, F, tol=0.0012, max_pieces=PIECES); rep = report(V, F, hulls)
print("mount", info, rep, "%.0fs" % (time.time() - t), flush=True)
S, N = oriented_samples(V, F, 0.0008, np.random.default_rng(0))
px = ns["proxies"]([h for h in hulls], 0.006, S, N)
assert inside_hulls(np.vstack(hulls), px, eps=1e-6).all()
print("proxies", len(px), flush=True)
# the camera body on the seat: its back on the cradle plate, centred; front glass = the prior's front centre
fwd = np.array(pri["forward_link6"]); up = np.array(pri["image_up_link6"]); x = np.array(pri["image_x_link6"])
front = np.array(pri["front_centre_link6_mm"]) / 1000.0; centre = front - 0.0115 * fwd
Rb = np.c_[x, up, -fwd]; q = Rot.from_matrix(Rb).as_quat()
out = {"source": "inputs/camera_mount/D405_305_Mount.step (the user's mount; Seeed reBot-DevArm hardware/camera-mounts/b601-camera-mounts), "
                 "tessellated by step_to_mesh.py, placed by mount_from_step.py (bolt holes on the CAD gripper's +x pair, clamp gap on the cnc7 + pla7 stack)",
       "frame": "link6, metres", "step_to_link6": info_json["step_to_link6"], "placement_checks": info_json["checks"],
       "mount": {"vertices": V.round(6).tolist(), "faces": F.tolist(), "rgba": [0.36, 0.20, 0.12, 1.0],
                 "pieces": [h.round(6).tolist() for h in hulls], "proxies": [p.round(6).tolist() for p in px], "decomposition": info, "report": rep},
       "camera_body": {"pos": centre.round(6).tolist(), "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])], "half": [0.021, 0.021, 0.0115],
                       "rgba": [0.12, 0.12, 0.13, 1.0], "note": "Gemini 305, 42 x 42 x 23 mm, its back on the cradle plate (box: drawn = collision)"}}
OUT.write_text(json.dumps(out)); print("wrote", OUT, OUT.stat().st_size // 1024, "KB")
