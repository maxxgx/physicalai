"""Real object size from the gripper camera (export only): approach frames of the resistor tasks (gripper open, the
camera looking down at the desk), the beige resistor segmented (not white desk, not the dark fingers from the sim
render), its minimum-area rectangle, and pixels -> mm through the fitted gripper camera (kit) at the depth where its
centre ray meets the plane z = D/2 (D its diameter, iterated). usage: object_size_wrist.py EXPORT FRAMES STATES_X5 OUT.json SHEET.jpg"""
import sys, json, numpy as np, cv2, mujoco, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
ex, F, S, out, sheet = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())["states"], Path(sys.argv[4]), sys.argv[5]
st = pd.read_parquet(ex / "states_allowed.parquet", columns=["episode_index", "frame_index", "observation.state", "task_index"])
grip = {(int(e), int(f)): float(np.asarray(s)[6]) for e, f, s in zip(st.episode_index, st.frame_index, st["observation.state"])}
task = {int(e): int(t) for e, t in zip(st.episode_index, st.task_index)}
r = b.make_robot(0); b.reset(r, 0); b.matched_state(r); m, d = r._sim.model, r._sim.data
from physicalai_mujoco_plugin.real2sim.desk_compositor import Compositor
comp = Compositor(m, d, b.kit(), width=640, height=480, show_objects=False)
cid = m.camera("wrist").id; jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]
fadr = [m.jnt_qposadr[m.joint(j).id] for j in ("joint_left", "joint_right")]
res, tiles = [], []
for key in sorted(S):
    ep, ms = map(int, key.split("/")); fi = int(round(ms * 30 / 1000)); g = grip.get((ep, fi))
    if g is None or g > -260 or task[ep] not in (0, 3): continue
    for a_, v in zip(jadr, S[key]): d.qpos[a_] = v
    for a_ in fadr: d.qpos[a_] = 0.05
    mujoco.mj_forward(m, d); seg = comp.render_camera("wrist", outputs=True)["seg"]
    R = d.cam_xmat[cid].reshape(3, 3); c = d.cam_xpos[cid].copy(); K = comp.camera_K(cid)
    if R[2, 2] < 0.5 or c[2] < 0.08: continue   # looking clearly down (view direction -R[:, 2] has z < -0.5), above the desk
    img = cv2.imread(str(F / f"gripper_e{ep:03d}_t{ms:07d}.png")); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    beige = (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 40) & (hsv[..., 1] < 170) & (hsv[..., 2] > 110)
    bands = (hsv[..., 2] < 90) | ((hsv[..., 1] > 120) & ((hsv[..., 0] < 8) | (hsv[..., 0] > 165)))
    mask = (beige | bands) & (seg != 10)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 300 or x < 5 or y < 5 or x + w > 635 or y + h > 380: continue
        if (beige & (lab == i)).sum() < 0.4 * area: continue
        rect = cv2.minAreaRect(np.column_stack(np.where(lab == i))[:, ::-1].astype(np.float32))
        lw = sorted(rect[1]); asp = lw[1] / max(lw[0], 1)
        if 1.3 < asp < 3.5 and (best is None or area > best[1]): best = (rect, area, lw)
    if best is None: continue
    rect, area, (wpx, lpx) = best; u, v = rect[0]; D = 0.02
    for _ in range(4):
        ray = R @ np.array([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -1.0])
        t = (D / 2 - c[2]) / ray[2]; p = c + t * ray; depth = float(-(R.T @ (p - c))[2])
        D = wpx * depth / K[0, 0]
    L = lpx * depth / K[0, 0]
    res.append({"frame": f"gripper_e{ep:03d}_t{ms:07d}.png", "diameter_mm": round(D * 1000, 1), "length_mm": round(L * 1000, 1), "depth_m": round(depth, 3), "px": [round(wpx, 1), round(lpx, 1)]})
    if len(tiles) < 8:
        box = cv2.boxPoints(rect).astype(int); t_ = img.copy(); cv2.drawContours(t_, [box], 0, (0, 255, 255), 2)
        cv2.putText(t_, f"D {D*1000:.1f} L {L*1000:.1f} mm (depth {depth:.2f})", (5, 20), 0, 0.6, (255, 0, 255), 2); tiles.append(cv2.resize(t_, (320, 240)))
summ = {k: {"median": float(np.median([x[k] for x in res])), "iqr": [float(np.percentile([x[k] for x in res], 25)), float(np.percentile([x[k] for x in res], 75))]} for k in ("diameter_mm", "length_mm")} if res else {}
out.write_text(json.dumps({"source": "review/export/robotcam_allowed", "n": len(res), "summary": summ, "frames": res}, indent=1))
while tiles and len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
if tiles: cv2.imwrite(sheet, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
print(len(res), summ)
