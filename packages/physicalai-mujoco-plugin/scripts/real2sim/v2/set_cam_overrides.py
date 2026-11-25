"""Write fitted robot-camera poses (fit_robotcam.py outputs) into the kit as camera overrides (pose in the camera's
parent body, fovy): they move only the compositor's cameras (the robot cameras), not the scene's housing geometry.
usage: set_cam_overrides.py KIT FIT_overview.json [FIT_gripper.json]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
kit_path = Path(sys.argv[1]); kit = json.loads(kit_path.read_text())
r = b.make_robot(0); b.matched_state(r); m, d = r._sim.model, r._sim.data; mujoco.mj_kinematics(m, d)
kit.setdefault("cameras", {})
for fp in sys.argv[2:]:
    f = json.loads(Path(fp).read_text()); name = {"gripper": "wrist"}.get(f["camera"], f["camera"]); cid = m.camera(name).id
    R_cv = np.array(f["rotation_opencv_axes"]); R_mj = R_cv @ np.diag([1.0, -1.0, -1.0]); p = np.array(f["position_m"])
    if f["frame"] == "link6":   # to world at the matched state, via link6
        bid = m.body("link6").id; R6 = d.xmat[bid].reshape(3, 3); p = d.xpos[bid] + R6 @ p; R_mj = R6 @ R_mj
    pb = m.cam_bodyid[cid]; Rb = d.xmat[pb].reshape(3, 3); pos = Rb.T @ (p - d.xpos[pb]); Rl = Rb.T @ R_mj
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rl.ravel())
    kit["cameras"][name] = {**kit["cameras"].get(name, {}), "pos": pos.round(6).tolist(), "quat": q.round(7).tolist(),
                            "fovy": round(float(f["fovy_deg"]), 3), "fitted_from": f"{Path(fp).name}: {f['frames']} allowed real frames, rms {f['rms_px']:.2f} px"}
    print(name, kit["cameras"][name])
kit_path.write_text(json.dumps(kit, indent=1))
