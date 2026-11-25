"""Dense coloured points (SfM frame): the splat's median depth at ALLOWED training views (half resolution), back-projected,
coloured by the real photo at that pixel. Every STEP-th training view, every 2nd pixel where alpha > 0.95.
Writes align/dense_pts.npz {P (N,3) float32, C (N,3) uint8 RGB, view (N,) int16}.  usage: dense_points.py SPLAT.ply [STEP]"""
import json, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import held, ROOT
from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer
step = int(sys.argv[2]) if len(sys.argv) > 2 else 3
cams = json.load(open(ROOT / "colmap/pinhole_cameras.json")); train = json.load(open(ROOT / "splat/v0/train_list.json"))["train"]
names = [n for n in train if not held(n)][::step]
r = MetalSplatRenderer(sys.argv[1]); Ps, Cs, Vs = [], [], []
for i, n in enumerate(names):
    c = cams[n]; T = np.eye(4); T[:3] = np.array(c["cam_from_world"]); K = np.array(c["K"]); w, h = c["w"] // 2, c["h"] // 2; K2 = K.copy(); K2[:2] /= 2
    out = r.render(T, K2, w, h); D = out["median_depth"]; ok = (out["alpha"] > 0.95) & (D > 0)
    ok[1::2, :] = False; ok[:, 1::2] = False
    img = cv2.cvtColor(cv2.resize(cv2.imread(str(ROOT / "colmap/pinhole/images" / n)), (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    ys, xs = np.nonzero(ok); z = D[ys, xs]
    Xc = np.c_[(xs + 0.5 - K2[0, 2]) / K2[0, 0] * z, (ys + 0.5 - K2[1, 2]) / K2[1, 1] * z, z]
    Ps.append(((Xc - T[:3, 3]) @ T[:3, :3]).astype(np.float32)); Cs.append(img[ys, xs]); Vs.append(np.full(len(z), i, np.int16))
np.savez_compressed(ROOT / "align/dense_pts.npz", P=np.concatenate(Ps), C=np.concatenate(Cs), view=np.concatenate(Vs), names=np.array(names))
print(len(names), "views,", sum(len(p) for p in Ps), "points")
