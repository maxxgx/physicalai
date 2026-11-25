"""Preview stills for a round: real ALLOWED robot-camera frames beside the sim render at the same joint state (full kit:
fitted camera, colour model, sensor noise; task objects hidden: the real objects' positions are not in the state).
usage: preview_r08.py OUT_DIR"""
import sys, json, re, numpy as np, cv2, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
from physicalai_mujoco_plugin.real2sim.desk_compositor import Compositor
R = Path.home() / "real2sim-data/bucket-v2"; out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
lock = json.loads((R / "review/HELDOUT_EPISODES.json").read_text()); allowed = set(lock["allowed"])
S = json.loads((R / "real/states_x5.json").read_text())["states"]; F = R / "real/frames_x5"
r = b.make_robot(0); b.reset(r, 0); b.matched_state(r); m, d = r._sim.model, r._sim.data
comp = Compositor(m, d, b.kit(), width=640, height=480, show_objects=False)
jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]
def label(img, text):
    img = img.copy(); cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); return img
for cam, name, keys in (("overview", "overview", ("1/20000", "10/5000", "3/30000")), ("gripper", "wrist", ("3/12500", "1/15000", "10/12000"))):
    rows = []
    for key in keys:
        ep, ms = map(int, key.split("/")); assert ep in allowed
        if key not in S: key = min((k for k in S if k.startswith(f"{ep}/")), key=lambda k: abs(int(k.split("/")[1]) - ms)); ms = int(key.split("/")[1])
        for a_, v in zip(jadr, S[key]): d.qpos[a_] = v
        d.time = ms / 1000; mujoco.mj_forward(m, d)
        sim = comp.render_camera(name)["rgb"][:, :, ::-1]; real = cv2.imread(str(F / f"{cam}_e{ep:03d}_t{ms:07d}.png"))
        rows.append(np.hstack([label(cv2.resize(real, (480, 360)), f"real {cam}, allowed episode {ep}, t={ms/1000:.1f}s"),
                               label(cv2.resize(sim, (480, 360)), f"sim {name}, same joint state (r08 kit)")]))
    cv2.imwrite(str(out / f"r08_{name}_real_vs_sim.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("ok")
