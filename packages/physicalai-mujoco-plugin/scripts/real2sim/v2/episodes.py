"""Run the scripted demonstrator on bucket_sort_desk_v2 for N seeds (sim only, no splat) and report each outcome.
usage: episodes.py N [first_seed] [--scene S] [--out JSONL] [--sheet JPG] [--sequence]
--sequence: after a success keep the table (task.offer_continue) for the next seed, as recording will."""
import argparse, json, sys, time
import numpy as np, mujoco
from physicalai_mujoco_plugin.robot import MuJoCoRobot

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int); ap.add_argument("first", type=int, nargs="?", default=0)
ap.add_argument("--scene", default="bucket_sort_desk_v2"); ap.add_argument("--out"); ap.add_argument("--sheet")
ap.add_argument("--sequence", action="store_true"); ap.add_argument("--max-sim-s", type=float, default=45.0)
ap.add_argument("--zone", action="store_true", help="objects 0.23-0.32 m ahead, 0.13-0.23 m to a random side (v1's misses)")
ap.add_argument("--objects", help="active object range lo,hi"); ap.add_argument("--pile", type=float)
a = ap.parse_args()
robot = MuJoCoRobot("rebot_b601", scene=a.scene, cameras=[], http_port=0, seed=a.first)
robot.connect()
m, d = robot._sim.model, robot._sim.data
from dataclasses import replace as _replace
task0 = robot._sim.on_reset; base_area = task0.area
if a.objects: base_area = _replace(base_area, active_objects=tuple(int(v) for v in a.objects.split(",")))
if a.pile is not None: base_area = _replace(base_area, pile_probability=a.pile)
task0.area = base_area
ren = mujoco.Renderer(m, 240, 320) if a.sheet else None
def snap():
    ren.update_scene(d, camera="overview"); return ren.render()[:, :, ::-1].copy()
rows, results = [], []
out = open(a.out, "a") if a.out else None
keep = False
for seed in range(a.first, a.first + a.n):
    task = robot._sim.on_reset
    if a.zone:
        side = 1 if np.random.default_rng(seed + 999).random() < 0.5 else -1
        task.area = _replace(base_area, object_x=(0.23, 0.32), object_y=(0.13, 0.23) if side > 0 else (-0.23, -0.13))
    robot._automation.set_autopilot("off")
    if keep: task.offer_continue()
    robot._set_seed(seed); robot._reset(); robot._set_seed(None)
    ins = task.instruction; lay = task.layout
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator
    demo._rng = np.random.default_rng(seed)
    count = robot._http_status()["episode"]["episode_count"]
    frames = [snap()] if ren else []; t0 = time.time(); ok = False; ticks = 0
    p = d.xpos[m.body(ins.object).id].copy()
    while time.time() - t0 < 300:
        robot.get_observation()
        if not demo.planning: ticks += 1
        ep = robot._http_status()["episode"]
        if ep["episode_count"] != count: break
        if ren and ticks == 600: frames.append(snap())
        if ep["success"]: ok = ep["in_bucket"][ins.object] == ins.bucket
        if demo.status()["outcome"] in {"success", "failed", "stalled"} or ticks > a.max_sim_s / 0.02: break
    if ren: frames.append(snap())
    st = demo.status()
    r = {"seed": seed, "object": ins.object, "label": ins.label, "bucket": ins.bucket, "text": ins.text, "success": bool(ok),
         "outcome": st["outcome"], "failure": st.get("failure"), "attempts": st.get("attempt"),
         "n_objects": len(lay.objects), "pile": lay.pile, "seq": task.sequence_index, "fallback": lay.fallback,
         "r": round(float(np.hypot(p[0], p[1])), 3), "angle_deg": round(float(np.degrees(np.arctan2(p[1], p[0]))), 1),
         "sim_s": round(ticks * 0.02, 1), "wall_s": round(time.time() - t0, 1)}
    results.append(r); print(json.dumps(r), flush=True)
    if out: out.write(json.dumps(r) + "\n"); out.flush()
    keep = a.sequence and ok and task.layout is not None
    if ren:
        import cv2
        row = np.hstack(frames + [np.zeros_like(frames[0])] * (3 - len(frames)))
        cv2.putText(row, f"{seed} {ins.label or ins.object}->{ins.bucket} {'OK' if ok else 'FAIL'}", (6, 18), 0, 0.5,
                    (0, 255, 0) if ok else (0, 0, 255), 1)
        rows.append(row)
if ren and rows:
    import cv2; cv2.imwrite(a.sheet, np.vstack(rows[:40]))
print("success", sum(r["success"] for r in results), "/", len(results))
robot.disconnect()
