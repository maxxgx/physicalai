"""r17 (R-56 condition): for every D3 key frame (held out: never opened), the nearest ALLOWED registered phone frame in
time whose view holds a finger (asserted outside every window +- guard): real | the finger hand-over off (r16) | on (r17),
whole frame scaled to 480 px wide, one row per key frame. usage: keyframe_handover_sheet.py OUT_JPG KEYFRAME[,...]"""
import json, os, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); H = json.load(open(ROOT / "review/HELDOUT_PHONE.json"))
allowed = lambda n: not any(a - H["guard_s"] <= int(n[1:6]) / 30 <= bb + H["guard_s"] for a, bb in H["windows_s"])
OUT, KEYS = sys.argv[1], sys.argv[2].split(",")
cams = b.pinhole_cameras(); robot = b.make_robot(0); b.matched_state(robot); m, d = robot._sim.model, robot._sim.data
_, _, W, Hh = next(iter(cams.values())); comp = b.compositor(robot, W, Hh, show_objects=False); cal = comp.cal
def finger_in_view(T, K):
    for body in ("gripper_left", "gripper_right"):
        x = cal.M.T @ (d.xpos[m.body(body).id] - cal.t - [0, 0, cal.plate]) / cal.scale; pc = T[:3, :3] @ x + T[:3, 3]
        if pc[2] > 0:
            u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
            if 0.05 * W < u < 0.95 * W and 0.05 * Hh < v < 0.95 * Hh: return True
    return False
rows, picked = [], {}
for k in KEYS:
    tk = int(k[1:6]) / 30
    for n in sorted((n for n in cams if allowed(n)), key=lambda n: abs(int(n[1:6]) / 30 - tk)):
        T, K, _, _ = cams[n]; b.matched_state(robot, int(n[1:6]) / 30)
        if not finger_in_view(T, K): continue
        assert allowed(n); picked[k] = n
        real = cv2.resize(cv2.imread(str(ROOT / "colmap/images" / n)), (480, int(480 * Hh / W)))
        comp.phone_finger_handover = False; off = comp.render_sfm(T, K)["rgb"][:, :, ::-1]
        comp.phone_finger_handover = True; on = comp.render_sfm(T, K)["rgb"][:, :, ::-1]; comp.phone_finger_handover = False
        tiles = [real] + [cv2.resize(np.ascontiguousarray(x), (480, real.shape[0])) for x in (off, on)]
        for t_, lab in zip(tiles, (f"real {n} (key {k})", "r16 (hand-over off)", "r17 (hand-over on)")): cv2.putText(t_, lab, (6, 18), 0, 0.5, (255, 255, 255), 1)
        rows.append(np.hstack(tiles)); break
cv2.imwrite(OUT, np.vstack(rows)); print("picked", picked)
