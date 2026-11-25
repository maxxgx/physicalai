"""r13: grasp clips from a recorded Studio dataset (LeRobot v3) for the grasp-defect check (review/NOTES.md addendum).

Per grasp (from the recorded gripper channel: a close from open, a hold, an open): frames from 1.5 s before the close
starts to 0.8 s after the open ends, at native resolution: overview | wrist | the overview cropped around the gripper
(240 x 180 around the projected tool point between the fingertips, x2), 30 fps H.264 (ffmpeg). The crop follows the sim's forward
kinematics of the recorded state through the kit's overview camera. Also a key-frame sheet per clip (approach, close,
lift, carry, release, at 2/3 size) for a still check.  usage: grasp_clips.py DATASET_DIR OUT_DIR [--n 6] [--seed 0]"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b

D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
info = json.load(open(D / "meta/info.json")); fps = info["fps"]
df = pd.concat([pd.read_parquet(p) for p in sorted((D / "data").rglob("*.parquet"))]).sort_values("index")
eps = pd.concat([pd.read_parquet(p) for p in sorted((D / "meta/episodes").rglob("*.parquet"))]).sort_values("episode_index")
tasks = pd.read_parquet(D / "meta/tasks.parquet").reset_index()
task_of = {int(r.task_index): r["task"] if "task" in r else r.iloc[0] for _, r in tasks.iterrows()}

def frames_of(cam: str, row) -> list:
    pre = f"videos/observation.images.{cam}"
    f = D / f"videos/observation.images.{cam}/chunk-{int(row[pre + '/chunk_index']):03d}/file-{int(row[pre + '/file_index']):03d}.mp4"
    t0 = float(row[pre + "/from_timestamp"]); cap = cv2.VideoCapture(str(f)); cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    out = []
    for _ in range(int(row["length"])):
        ok, im = cap.read()
        if not ok: break
        out.append(im)
    return out

def _runs(mask: np.ndarray, gap: int = 3) -> list:
    idx = np.flatnonzero(mask)
    if not len(idx): return []
    out = [[idx[0], idx[0]]]
    for k in idx[1:]:
        if k - out[-1][1] <= gap + 1: out[-1][1] = k
        else: out.append([k, k])
    return [(a_, b_ + 1) for a_, b_ in out if b_ - a_ >= 2]

def grasps(g: np.ndarray) -> list:
    """(close start, close end, open start, open end) frame indices from the gripper reading (-270 open .. 0 closed):
    a hold is a run of at least 1 s with the reading above -110 deg (the fingers nearer than ~31 mm: on an object; a
    pre-shape stays wider); the close is the closing run (reading rising) into it, the release the opening run out of it."""
    v = np.r_[0, np.diff(g)]; out = []
    for hs, he in _runs(g > -110, gap=2):
        if he - hs < fps: continue
        cs = hs
        while cs > 0 and v[cs - 1] > 0.05: cs -= 1
        oe = he
        while oe < len(g) - 1 and v[oe] < -0.05: oe += 1
        out.append((cs, hs, he, oe))
    return out

robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot); b.compositor(robot, show_objects=False)
ch = robot._sim.channels[0]; cam = m.camera("overview").id; ge = m.body("link6").id   # the tool point: link6 + 163 mm (the profile end effector)
def project(state):
    ch.set_positions(state); mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    R = d.cam_xmat[cam].reshape(3, 3); f = 0.5 * 480 / np.tan(np.radians(m.cam_fovy[cam]) / 2)
    q = (d.xpos[ge] + d.xmat[ge].reshape(3, 3) @ np.array([0.0, 0.0, 0.163]) - d.cam_xpos[cam]) @ R
    return int(320 + f * q[0] / -q[2]), int(240 - f * q[1] / -q[2])

rng = np.random.default_rng(SEED); order = rng.permutation(len(eps)); made = []
for k in order:
    if len(made) >= N: break
    row = eps.iloc[k]; e = int(row.episode_index); rows = df[df.episode_index == e].sort_values("frame_index")
    S = np.stack(rows["observation.state"].to_numpy()).astype(float); gs = grasps(S[:, -1])
    if not gs: continue
    cs, ce, os_, oe = gs[0]; a_, z_ = max(0, cs - int(1.5 * fps)), min(len(S) - 1, oe + int(0.8 * fps))
    ov, wr = frames_of("overview", row), frames_of("gripper", row)
    name = f"ep{e:03d}_grasp"; tmp = OUT / f"{name}_frames"; tmp.mkdir(exist_ok=True)
    keys = {"approach": max(a_, cs - 10), "close start": cs, "closed": ce, "closed +0.2 s": min(ce + 6, os_), "lift": min(ce + fps // 2, os_),
            "carry": (ce + os_) // 2, "release start": os_, "release end": oe, "after": min(z_, oe + 10)}
    sheet = []
    for i in range(a_, z_ + 1):
        if i >= len(ov) or i >= len(wr): break
        u, v = project(S[i]); x0, y0 = int(np.clip(u - 120, 0, 400)), int(np.clip(v - 90, 0, 300))
        crop = cv2.resize(ov[i][y0:y0 + 180, x0:x0 + 240], (480, 360), interpolation=cv2.INTER_NEAREST)
        crop = np.vstack([crop, np.zeros((120, 480, 3), np.uint8)])
        cv2.putText(crop, f"ep {e} f{i}  grip {S[i, -1]:.0f}", (8, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        frame = np.hstack([ov[i], wr[i], crop]); cv2.imwrite(str(tmp / f"{i - a_:05d}.png"), frame)
        if i in keys.values():
            lab = frame.copy(); cv2.putText(lab, ", ".join(k for k, v in keys.items() if v == i), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            sheet.append(cv2.resize(lab, (lab.shape[1] * 2 // 3, lab.shape[0] * 2 // 3)))
    clip = OUT / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(clip)], check=True)
    cv2.imwrite(str(OUT / f"{name}_keys.jpg"), np.vstack(sheet) if sheet else np.zeros((10, 10, 3), np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
    for p in tmp.iterdir(): p.unlink()
    tmp.rmdir()
    made.append({"episode": e, "task": task_of.get(int(rows.task_index.iloc[0])), "frames": [a_, z_], "close": [cs, ce], "open": [os_, oe], "keys": keys, "clip": clip.name})
    print(json.dumps(made[-1], default=int), flush=True)
json.dump({"dataset": str(D), "clips": made}, open(OUT / "clips.json", "w"), indent=1, default=int)
