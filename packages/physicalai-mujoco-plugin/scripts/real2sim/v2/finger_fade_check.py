"""r17 (R-56 condition, DF8): the phone path's finger hand-over as the arm leaves its calibrated pose. A demo episode with
the phone camera held at an allowed registered view (asserted outside the held-out windows); every 1/15 s frame while
the arm splat's weight w goes from 1 to 0 (and two before / after) rendered and cropped around the gripper; a strip
with w per frame, and the mean |change| inside the crop between consecutive frames (a pop would stand out).
usage: finger_fade_check.py SEED FRAME OUT_JPG"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
seed, name, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); H = json.load(open(ROOT / "review/HELDOUT_PHONE.json"))
assert not any(a - H["guard_s"] <= int(name[1:6]) / 30 <= bb + H["guard_s"] for a, bb in H["windows_s"]), name
cams = b.pinhole_cameras(); T, K, W, Hh = cams[name if name.endswith(".jpg") else name + ".jpg"]
robot = b.make_robot(seed); b.reset(robot, seed); m, d = robot._sim.model, robot._sim.data
comp = b.compositor(robot, W, Hh); comp.phone_finger_handover = True; ge = m.body("gripper_end").id; cal = comp.cal
# the demo never comes near the calibrated pose (w = 0 throughout a VIDEO episode), so the hand-over is driven here: the
# arm from the calibrated pose (matched state) to home in 16 equal joint steps, the phone camera fixed
b.matched_state(robot); jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]
q0 = d.qpos[jadr].copy(); q1 = np.array(demo_home) if (demo_home := getattr(robot._automation.autopilot.demonstrator.active_arm, "home", None)) is not None else np.zeros(6)
frames = []
for k in range(17):
    d.qpos[jadr] = q0 + (q1 - q0) * k / 16; mujoco.mj_forward(m, d); w = comp._robot_splat_weight(); img = comp.render_sfm(T, K)["rgb"]
    x = cal.M.T @ (d.xpos[ge] - cal.t - [0, 0, cal.plate]) / cal.scale; pc = T[:3, :3] @ x + T[:3, 3]
    u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]; h = 300
    x0, y0 = int(np.clip(u - h, 0, W - 2 * h)), int(np.clip(v - h, 0, Hh - 2 * h))
    frames.append((k / 16, w, cv2.resize(img[y0:y0 + 2 * h, x0:x0 + 2 * h], (240, 240))))
shots = [f for f in frames if len(f) == 3]
diffs = [round(float(np.abs(shots[i][2].astype(int) - shots[i - 1][2].astype(int)).mean()), 2) for i in range(1, len(shots))]
tiles = [cv2.putText(s[2][:, :, ::-1].copy(), f"step {s[0]:.2f} w {s[1]:.2f}", (6, 18), 0, 0.5, (255, 255, 255), 1) for s in shots]
while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(OUT, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
print("frames", len(shots), "w", [round(s[1], 2) for s in shots], "mean |change| between consecutive", diffs)
