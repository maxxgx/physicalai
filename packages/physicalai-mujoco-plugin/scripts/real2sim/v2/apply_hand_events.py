"""r14 (D3, coordinator ruling): bucket poses in the phone capture change only at hand events (align/hand_events.json).

Per bucket, in time order, from its calibrated pose: for each of its hand events, the pose fitted (fit_buckets_photo.py,
the same settings for every event: every allowed registered frame, partial views allowed) on the allowed frames from
the event's end to the bucket's next event (or the capture's end), starting from (and compared against) the pose carried
over; adopted from the event's START if it changes by > 3 mm or > 2 deg AND the median IoU on those frames improves,
else the pose carries over. Writes align/hand_events_applied.json (every event: frames, pose change, IoU before / after,
adopted y/n) and, with --write, the calibration's `bucket_moves` (r13's kept as `bucket_moves_r13`).
usage: apply_hand_events.py [--write]"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser(); HERE = Path(__file__).resolve().parent
EV = json.load(open(ROOT / "align/hand_events.json")); cal_path = ROOT / "calibration.json"; cal = json.load(open(cal_path))
fno = lambda f: int(f[1:6]); LAST = 4716
out, moves = [], []
for c in ("black", "green", "red", "blue"):
    evs = sorted([e for e in EV["events"] if e["bucket"] == c], key=lambda e: fno(e["start_frame"]))
    pose = np.array([cal["buckets"][c]["x"], cal["buckets"][c]["y"], cal["buckets"][c]["yaw"]])
    for i, e in enumerate(evs):
        lo = fno(e["end_frame"]) + 1; hi = (fno(evs[i + 1]["start_frame"]) - 1) if i + 1 < len(evs) else LAST
        tmp = ROOT / f"sim/r14/hand_fit_{e['id']}.json"
        env = dict(os.environ, FRAME_RANGE=f"{lo},{hi}", FRAME_STEP="1", PARTIAL="1", FIT_OUT=str(tmp), INIT_POSES=json.dumps({c: pose.tolist()}))
        run = subprocess.run([sys.executable, "-I", "-B", str(HERE / "fit_buckets_photo.py"), c], env=env, capture_output=True, text=True)
        row = {**e, "start_s": round(fno(e["start_frame"]) / 30, 2), "end_s": round(fno(e["end_frame"]) / 30, 2),
               "fit_frames_range_s": [round(lo / 30, 2), round(hi / 30, 2)], "pose_before": pose.round(5).tolist()}
        if run.returncode or not tmp.exists():
            row.update(adopted=False, reason="no fit (no usable allowed frame in the range)", frames=0, stderr=run.stderr[-300:])
            out.append(row); print(c, e["id"], "no fit", flush=True); continue
        r = json.load(open(tmp))["buckets"].get(c)
        if not r or not r.get("frames"):
            row.update(adopted=False, reason="no usable allowed frame in the range", frames=0); out.append(row); print(c, e["id"], "no frames", flush=True); continue
        new = np.array([r["x"], r["y"], r["yaw"]]); shift = float(np.hypot(*(new[:2] - pose[:2]))) * 1000; dyaw = float(np.degrees(new[2] - pose[2]))
        better = r["iou_median_after"] > r["iou_median_before"]; big = shift > 3.0 or abs(dyaw) > 2.0
        row.update(frames=r["frames"], frame_list=r["frame_list"], pose_fitted=new.round(5).tolist(), shift_mm=round(shift, 1), dyaw_deg=round(dyaw, 2),
                   iou_median_before=r["iou_median_before"], iou_median_after=r["iou_median_after"], adopted=bool(better and big),
                   reason="adopted" if better and big else ("change <= 3 mm and <= 2 deg" if not big else "no IoU improvement"))
        out.append(row); print(c, e["id"], row["frames"], "frames", f"{shift:.1f} mm {dyaw:+.2f} deg", "IoU", r["iou_median_before"], "->", r["iou_median_after"], "ADOPTED" if row["adopted"] else "carried over", flush=True)
        if row["adopted"]:
            pose = new; moves.append({"bucket": c, "from_s": row["start_s"], "x": float(new[0]), "y": float(new[1]), "yaw": float(new[2]),
                                      "method": f"hand-event rule (r14): event {e['id']} {row['start_s']}-{row['end_s']} s; fitted on {r['frames']} allowed frames {lo / 30:.1f}-{hi / 30:.1f} s (frame list in align/hand_events_applied.json); IoU {r['iou_median_before']} -> {r['iou_median_after']}"})
json.dump({"rule": EV["rule"], "events": out, "moves": moves}, open(ROOT / "align/hand_events_applied.json", "w"), indent=1)
if "--write" in sys.argv:
    if "bucket_moves_r13" not in cal: cal["bucket_moves_r13"] = cal.get("bucket_moves", [])
    cal["bucket_moves"] = moves; json.dump(cal, open(cal_path, "w"), indent=1); print("calibration bucket_moves:", len(moves))
