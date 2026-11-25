"""Real vs sim resistor size in the overview, measured the same way on both (biases cancel): beige-and-banded blobs on
the desk (desk pixels by the sim segmentation; buckets, robot and their surroundings excluded), the long and short sides
of each blob's minimum-area rectangle, converted to mm at the depth where the blob centre's ray meets z = 1 cm. Real:
the export's overview frames (allowed episodes, every 5th frame); sim: overview renders of resistor episodes with the
scene's current resistor scale. Reports the upper quartile of the long side (least foreshortened) and the median short
side. usage: resistor_ratio_overview.py FRAMES_X5 STATES_X5 OUT.json"""
import sys, json, numpy as np, cv2, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
from physicalai_mujoco_plugin.bucket_sort_objects import real_kind
F, S, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())["states"], Path(sys.argv[3])
r = b.make_robot(0); m, d = r._sim.model, r._sim.data
from physicalai_mujoco_plugin.real2sim.desk_compositor import Compositor
b.reset(r, 0); comp = Compositor(m, d, b.kit(), width=640, height=480, show_objects=True)
cid = m.camera("overview").id; jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]

def blobs(img, desk):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    beige = (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 35) & (hsv[..., 2] > 100)
    band = ((hsv[..., 1] > 110) & ((hsv[..., 0] < 10) | (hsv[..., 0] > 165))) | (hsv[..., 2] < 70)
    mk = cv2.morphologyEx(((beige | band) & desk).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(mk); res = []
    K = comp.camera_K(cid); R = d.cam_xmat[cid].reshape(3, 3); c = d.cam_xpos[cid]
    for i in range(1, n):
        if st[i][4] < 25 or st[i][4] > 900 or (beige & (lab == i)).sum() < 0.4 * st[i][4]: continue
        (u, v), (w, h), _ = cv2.minAreaRect(np.column_stack(np.where(lab == i))[:, ::-1].astype(np.float32))
        ray = R @ np.array([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -1.0]); t = (0.01 - c[2]) / ray[2]
        depth = float(-(R.T @ (t * ray))[2]); mm = depth / K[0, 0] * 1000
        res.append((max(w, h) * mm, min(w, h) * mm))
    return res

def desk_mask():
    seg = comp.render_camera("overview", outputs=True)["seg"]
    desk = (seg == 5).astype(np.uint8)
    return cv2.erode(desk, np.ones((9, 9), np.uint8)) > 0, seg

real = []
for key in sorted(S):
    ep, ms = map(int, key.split("/"))
    if ms % 2000: continue   # every 2 s
    for a_, v in zip(jadr, S[key]): d.qpos[a_] = v
    mujoco.mj_forward(m, d); desk, _ = desk_mask()
    real += blobs(cv2.imread(str(F / f"overview_e{ep:03d}_t{ms:07d}.png")), desk)
sim = []
for seed in range(8000, 8040):
    r._automation.set_autopilot("off"); b.reset(r, seed)
    for _ in range(500): mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    res = comp.render_camera("overview", outputs=True); seg = res["seg"]
    desk = cv2.erode((seg == 5).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    # sim: only resistor pixels count as the object (the segmentation knows them): restrict to their labels
    sim += blobs(res["rgb"][:, :, ::-1].copy(), desk | (seg >= 20))
def summ(v):
    v = [x for x in v if x[0] < 100 and 1.2 <= x[0] / max(x[1], 1e-6) <= 4.0]   # resistor-like blobs only, both sides alike
    L = [x[0] for x in v]; W = [x[1] for x in v]
    return {"n": len(v), "long_p75_mm": float(np.percentile(L, 75)), "long_median_mm": float(np.median(L)), "short_median_mm": float(np.median(W))} if v else {}
o = {"real": summ(real), "sim": summ(sim), "raw": {"real": [list(map(float, x)) for x in real], "sim": [list(map(float, x)) for x in sim]}, "sim_scale_now": json.load(open(Path.home() / "real2sim-data/bucket-v2/align/object_size.json"))["resistor"]["scale"]}
if o["real"] and o["sim"]:
    o["ratio_long_p75"] = o["real"]["long_p75_mm"] / o["sim"]["long_p75_mm"]; o["ratio_short_median"] = o["real"]["short_median_mm"] / o["sim"]["short_median_mm"]
    o["implied_scale"] = o["sim_scale_now"] * float(np.mean([o["ratio_long_p75"], o["ratio_short_median"]]))
out.write_text(json.dumps(o, indent=1)); print(json.dumps({k: v for k, v in o.items() if k != "raw"}))
