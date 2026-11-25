"""T3 (r11): object lengths on the desk from allowed wrist frames (export only). For a frame of an allowed episode, the
fitted wrist camera (the kit's pose and fovy) at the recorded state of t - latency; each marked end point (pixel, read
off a zoomed crop) is cast onto the horizontal plane at the object's mid height; length = distance between the two
points. usage: t3_measure.py STATES_PARQUET   (the points are listed below, with their frames)"""
import json, sys
from pathlib import Path
import numpy as np, mujoco, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
POINTS = [   # frame, episode, t (s), camera, object, height of its ends (m), end A (u, v), end B (u, v)
    ("overview_e001_t0000000.png", 1, 0.0, "overview", "resistor", 0.012, (381.6, 243.0), (418.0, 253.0)),
    ("overview_e001_t0000000.png", 1, 0.0, "overview", "cpu a", 0.010, (398.0, 275.0), (440.0, 275.0)),
    ("overview_e001_t0000000.png", 1, 0.0, "overview", "cpu b", 0.010, (329.0, 290.0), (370.0, 290.0)),
    ("overview_e003_t0000000.png", 3, 0.0, "overview", "battery 9 V (48.5 mm incl. terminals)", 0.009, (443.0, 239.0), (476.0, 230.0)),
    ("overview_e003_t0000000.png", 3, 0.0, "overview", "cpu c", 0.010, (314.0, 241.0), (308.0, 272.0)),
    ("overview_e003_t0000000.png", 3, 0.0, "overview", "cpu d", 0.010, (382.0, 258.0), (380.0, 292.0)),
    ("overview_e003_t0000000.png", 3, 0.0, "overview", "cpu e", 0.010, (393.0, 217.0), (423.6, 234.0)),
    ("gripper_e000_t0000000.png", 0, 0.0, "wrist", "cpu (right)", 0.010, (122, 207), (202, 220)),
    ("gripper_e000_t0000000.png", 0, 0.0, "wrist", "cpu (left)", 0.010, (36, 234), (110, 199)),
    ("gripper_e000_t0000000.png", 0, 0.0, "wrist", "resistor", 0.012, (344, 161), (327, 231)),
    ("gripper_e003_t0000000.png", 3, 0.0, "wrist", "battery 9 V (48.5 mm incl. terminals)", 0.009, (144, 171), (228, 181)),
    ("gripper_e003_t0000000.png", 3, 0.0, "wrist", "cpu (9808, same frame as the battery)", 0.010, (405, 215), (485, 172.5)),
]
st = pd.read_parquet(sys.argv[1]); lat = float(json.load(open(b.KIT))["camera_latency_s"]["wrist"])
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot)
comp = b.compositor(robot, show_objects=False); ch = robot._sim.channels[0]; cam = m.camera("wrist").id
H, W = 480, 640; f = H / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2)
out = []
for frame, ep, t, camname, name, z, A, B in POINTS:
    cam = m.camera(camname).id; f = H / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2)
    rows = st[st.episode_index == ep].sort_values("timestamp"); ts = rows.timestamp.to_numpy(float); S = np.stack(rows["observation.state"].to_numpy()).astype(float)
    tl = float(np.clip(t - lat, ts[0], ts[-1])); ch.set_positions(np.array([np.interp(tl, ts, S[:, j]) for j in range(S.shape[1])])); mujoco.mj_forward(m, d)
    R = d.cam_xmat[cam].reshape(3, 3); c = d.cam_xpos[cam]
    def hit(uv):
        u, v = uv; ray = R @ np.array([(u + 0.5 - W / 2) / f, -(v + 0.5 - H / 2) / f, -1.0]); s = (z - c[2]) / ray[2]
        return c + s * ray
    pa, pb = hit(A), hit(B); L = float(np.linalg.norm(pa - pb)); out.append({"frame": frame, "object": name, "end_height_m": z, "A": A, "B": B, "length_mm": round(L * 1000, 1), "camera_height_m": round(float(c[2]), 3)})
    print(f"{frame} {name}: {L * 1000:.1f} mm (camera {c[2]:.3f} m above the desk)")
    out[-1]["camera"] = camname
json.dump({"source": "review/export/robotcam_allowed (frames_x5 gripper, states_allowed.parquet)", "method": __doc__.split("\n")[0], "measurements": out},
          open(Path(b.ROOT) / "real/t3_measure.json" if hasattr(b, "ROOT") else "t3_measure.json", "w"), indent=1)
