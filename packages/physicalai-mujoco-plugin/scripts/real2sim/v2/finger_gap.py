"""Finger gap, real vs sim, in the gripper camera: for each frame (open, or closed on a resistor; export only) and image
rows through the finger pads, the inner edges of the two sim fingers (segmentation, at the recorded gripper value through
the provisional scale) and of the two real fingers (dark pixels, luminance < 60, nearest the sim edges). The ratio of the
gaps (real / sim) at the same depth turns the model's opening into the real one.
usage: finger_gap.py EXPORT FRAMES STATES_X5 OUT.json"""
import sys, json, numpy as np, cv2, mujoco, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
ex, F, S, out = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())["states"], Path(sys.argv[4])
st = pd.read_parquet(ex / "states_allowed.parquet", columns=["episode_index", "frame_index", "observation.state", "task_index"])
grip = {(int(e), int(f)): float(np.asarray(s)[6]) for e, f, s in zip(st.episode_index, st.frame_index, st["observation.state"])}
task = {int(e): int(t) for e, t in zip(st.episode_index, st.task_index)}
r = b.make_robot(0); b.reset(r, 0); b.matched_state(r); m, d = r._sim.model, r._sim.data
from physicalai_mujoco_plugin.real2sim.desk_compositor import Compositor
comp = Compositor(m, d, b.kit(), width=640, height=480, show_objects=False)
jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]; fadr = [m.jnt_qposadr[m.joint(j).id] for j in ("joint_left", "joint_right")]
rows_out = []
for key in sorted(S):
    ep, ms = map(int, key.split("/")); fi = int(round(ms * 30 / 1000)); g = grip.get((ep, fi))
    if g is None: continue
    kind = "open" if g < -260 else ("closed" if -90 < g < -30 and task[ep] in (0, 3) else None)   # resistor tasks only when closed
    if kind is None: continue
    png = F / f"gripper_e{ep:03d}_t{ms:07d}.png"
    if not png.exists(): continue
    for a_, v in zip(jadr, S[key]): d.qpos[a_] = v
    finger = min(abs(g) / 5400.0, 0.05)
    for a_ in fadr: d.qpos[a_] = finger
    mujoco.mj_forward(m, d); seg = comp.render_camera("wrist", outputs=True)["seg"]
    real = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2GRAY)
    gaps = []
    for y in range(300, 470, 6):
        cols = np.where(seg[y] == 10)[0]
        if len(cols) < 4: continue
        left, right = cols[cols < 320], cols[cols >= 320]
        if not len(left) or not len(right): continue
        ls, rs = left.max(), right.min()   # sim inner edges
        dark = real[y] < 60
        # real inner edges: the dark run nearest each sim edge, searched within 60 px
        lr = [x for x in range(max(ls - 60, 0), min(ls + 60, 639)) if dark[x] and not dark[x + 1]]
        rr = [x for x in range(max(rs - 60, 1), min(rs + 60, 639)) if dark[x] and not dark[x - 1]]
        if not lr or not rr: continue
        lreal = min(lr, key=lambda x: abs(x - ls)); rreal = min(rr, key=lambda x: abs(x - rs))
        if rreal - lreal > 5 and rs - ls > 5: gaps.append(((rreal - lreal) / (rs - ls), rs - ls))
    if len(gaps) >= 5:
        ratio = float(np.median([g_[0] for g_ in gaps]))
        rows_out.append({"frame": png.name, "state": kind, "gripper_deg": round(g, 1), "model_opening_mm": round(2 * finger * 1000, 1),
                         "gap_ratio_real_over_sim": round(ratio, 3), "real_opening_mm": round(ratio * 2 * finger * 1000, 1), "rows": len(gaps)})
res = {"source": "review/export/robotcam_allowed (frames_x5, states_allowed.parquet)", "frames": rows_out}
for k in ("open", "closed"):
    v = [x["gap_ratio_real_over_sim"] for x in rows_out if x["state"] == k]; w = [x["real_opening_mm"] for x in rows_out if x["state"] == k]
    res[k] = {"n": len(v), "ratio_median": float(np.median(v)) if v else None, "ratio_iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if v else None,
              "real_opening_mm_median": float(np.median(w)) if w else None}
out.write_text(json.dumps(res, indent=1)); print({k: res[k] for k in ("open", "closed")})
