"""Check the gripper scale (rebot_b601: -270 driver deg = 0.05 m per finger, provisional) against the real gripper
camera: the sim at each frame's recorded joint and gripper state, rendered from the fitted gripper camera (kit), with
the robot's outline over the real frame. Frames: open, and closed on an object (grip_widths.json plateaus).
Source: the reviewer's export only (frames_x5 written from it, states_allowed.parquet).
usage: finger_check.py EXPORT_DIR FRAMES_DIR STATES_X5.json GRIP_WIDTHS.json OUT.jpg"""
import sys, json, numpy as np, cv2, mujoco, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
ex, F, S, G, out = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())["states"], json.loads(Path(sys.argv[4]).read_text()), sys.argv[5]
st = pd.read_parquet(ex / "states_allowed.parquet", columns=["episode_index", "frame_index", "observation.state"])
grip = {(int(e), int(f)): float(np.asarray(s)[6]) for e, f, s in zip(st.episode_index, st.frame_index, st["observation.state"])}
picks = []   # (episode, frame_index, why)
for ep, e in G["episodes"].items():
    for run in e["carry_plateaus"][:1]:
        a, z = run["frames"]; mid = (a + z) // 2; picks.append((int(ep), mid - mid % 5 + (1 if int(ep) in (6, 9) else 0), f"closed {run['opening_mm']} mm"))
for ep in (0, 7):
    picks.append((ep, 30 + (0), "open"))
r = b.make_robot(0); b.reset(r, 0); b.matched_state(r); m, d = r._sim.model, r._sim.data
from physicalai_mujoco_plugin.real2sim.desk_compositor import Compositor
comp = Compositor(m, d, b.kit(), width=640, height=480, show_objects=False)
jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]; fadr = [m.jnt_qposadr[m.joint(j).id] for j in ("joint_left", "joint_right")]
tiles = []
for ep, fi, why in picks[:8]:
    key = f"{ep}/{int(round(fi / 30 * 1000))}"; png = F / f"gripper_e{ep:03d}_t{int(round(fi / 30 * 1000)):07d}.png"
    if key not in S or not png.exists() or (ep, fi) not in grip: continue
    for a_, v in zip(jadr, S[key]): d.qpos[a_] = v
    g = grip[(ep, fi)]; finger = min(max(abs(g) / 5400.0, 0.0), 0.05)
    for a_ in fadr: d.qpos[a_] = finger
    mujoco.mj_forward(m, d); res = comp.render_camera("wrist", outputs=True)
    edge = cv2.Canny(((res["seg"] == 10) * 255).astype(np.uint8), 50, 150) > 0
    real = cv2.imread(str(png)); real[cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (0, 255, 255)
    t = cv2.resize(real, (400, 300)); cv2.putText(t, f"e{ep} f{fi} gripper {g:.0f} deg -> {2 * finger * 1000:.0f} mm ({why})", (4, 14), 0, 0.4, (255, 0, 255), 1); tiles.append(t)
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])); print("ok", len(tiles))
