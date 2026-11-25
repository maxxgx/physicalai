"""DF10 preview (r10): sim robot-camera frames with resting objects (t = 0) and the gripper near the desk (the first
close), both cameras, through the kit's compositor. Sim only (no photo). usage: contact_shadow_preview.py OUT SEED..."""
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for seed in [int(s) for s in sys.argv[2:]]:
    robot = b.make_robot(seed); b.reset(robot, seed); comp = b.compositor(robot)
    def shot(tag):
        for cam in ("wrist", "overview"):
            r = comp.render_camera(cam, outputs=False) if False else comp.render_camera(cam)
            img = r["rgb"] if isinstance(r, dict) else r
            cv2.imwrite(str(out / f"s{seed}_{tag}_{cam}.png"), img[:, :, ::-1])
    shot("rest")
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator; T = 0.0
    while T < 20:
        b.tick(robot, demo); T += b.DT
        if demo.phase == "close": shot("close"); break
    robot.disconnect()
print("wrote", out)
