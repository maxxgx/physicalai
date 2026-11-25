"""r16 (D3b): where do the doubled real fingers in the close phone views come from? ALLOWED phone frames only (outside
every held-out window +- guard, review/HELDOUT_PHONE.json; asserted). Per frame: the real photo around the gripper, the
phone-path render as delivered (CAL opening), at other openings, and with the arm splat's finger Gaussians left out
(KIT_NOFINGERS) / the desk splat's Gaussians near the fingers left out; a sheet per frame.
usage: d3b_fingers.py OUT_DIR FRAME[,FRAME...] OPENING_MM[,OPENING_MM...]"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
H = json.load(open(ROOT / "review/HELDOUT_PHONE.json"))
def allowed(n):
    t = int(n[1:6]) / 30.0
    return not any(a - H["guard_s"] <= t <= bb + H["guard_s"] for a, bb in H["windows_s"])
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
names = [n if n.endswith(".jpg") else n + ".jpg" for n in sys.argv[2].split(",")]
assert all(allowed(n) for n in names), [n for n in names if not allowed(n)]
openings = [float(v) / 1000 for v in sys.argv[3].split(",")]
cams = b.pinhole_cameras(); robot = b.make_robot(0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot)
_, _, W, Hh = cams[names[0]]; comp = b.compositor(robot, W, Hh, show_objects=False); comp.phone_finger_handover = True
fadr = [m.jnt_qposadr[m.joint(n).id] for n in ("joint_left", "joint_right")]; ge = m.body("gripper_end").id
IMG = ROOT / "colmap/images"
for n in names:
    T, K, w, h = cams[n]; b.matched_state(robot, int(n[1:6]) / 30.0)
    p = T[:3, :3] @ (np.asarray(d.xpos[ge]) * 1.0) + T[:3, 3]
    # the phone cameras are in SfM units: project via the compositor's own world -> sfm map
    cal = comp.cal; xs = cal.M.T @ (d.xpos[ge] - cal.t - [0, 0, cal.plate]) / cal.scale
    pc = T[:3, :3] @ xs + T[:3, 3]; u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
    half = int(np.clip(0.12 / cal.scale * K[0, 0] / pc[2] / max(cal.units_per_m * cal.scale, 1e-9), 120, 420)) if False else 260
    x0, y0 = int(np.clip(u - half, 0, w - 2 * half)), int(np.clip(v - half, 0, h - 2 * half))
    crop = lambda im: cv2.resize(im[y0:y0 + 2 * half, x0:x0 + 2 * half], (400, 400))
    tiles = [cv2.putText(crop(cv2.imread(str(IMG / n))), "real " + n[:-4], (8, 24), 0, 0.7, (255, 255, 255), 2)]
    for op in openings:
        for a in fadr: d.qpos[a] = op / 2
        mujoco.mj_forward(m, d); r = comp.render_sfm(T, K)
        tiles.append(cv2.putText(crop(r["rgb"][:, :, ::-1].copy()), f"opening {op * 1000:.0f} mm", (8, 24), 0, 0.7, (255, 255, 255), 2))
    if os.environ.get("HIDE_CAD_FINGERS") == "1":   # r17: the first opening again, the CAD fingers not drawn (the splat's only)
        for a in fadr: d.qpos[a] = openings[0] / 2
        mujoco.mj_forward(m, d); fg = [k for k in range(m.ngeom) if m.body(m.geom_bodyid[k]).name in ("gripper_left", "gripper_right")]
        keep = m.geom_rgba[fg].copy(); m.geom_rgba[fg, 3] = 0.0
        try: r = comp.render_sfm(T, K)
        finally: m.geom_rgba[fg] = keep
        tiles.append(cv2.putText(crop(r["rgb"][:, :, ::-1].copy()), f"{openings[0] * 1000:.0f} mm, no CAD fingers", (8, 24), 0, 0.7, (255, 255, 255), 2))
    if os.environ.get("NO_ARM_SPLAT") == "1":   # the first opening again without the arm splat's moving Gaussians
        for a in fadr: d.qpos[a] = openings[0] / 2
        mujoco.mj_forward(m, d); keep = comp.rs; comp.rs = None
        try: r = comp.render_sfm(T, K)
        finally: comp.rs = keep
        tiles.append(cv2.putText(crop(r["rgb"][:, :, ::-1].copy()), f"{openings[0] * 1000:.0f} mm, no arm splat", (8, 24), 0, 0.7, (255, 255, 255), 2))
    cv2.imwrite(str(OUT / f"{n[:-4]}_sheet.jpg"), np.hstack(tiles)); print("wrote", n, (round(u), round(v)), flush=True)
