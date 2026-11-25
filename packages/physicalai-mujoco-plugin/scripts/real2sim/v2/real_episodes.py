"""Access to the real robot episodes that never touches a held-out frame.

The reviewer's lock (review/HELDOUT_EPISODES.json) names, per episode and camera, the held-out time ranges and their
guards. LeRobot v3 stores each camera's episodes concatenated in shared video files, so an ordinary seek would decode
from the keyframe before the start - possibly inside a held-out range. Here every allowed segment starts at the first
KEYFRAME inside it (found from packet headers with ffprobe, which reads no pixels) and ends before the next held-out
range's guard, so the decoder never produces a held-out frame.

usage (library): segments = allowed_segments(DATASET, LOCK); extract(seg, OUT_DIR, step=1)
       (CLI):     real_episodes.py DATASET LOCK OUT_DIR [--step N] [--cams overview,gripper] [--max-per-episode K]"""
from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path


def load_lock(path) -> dict:
    """{episode_index: {camera: [(t0, t1), ...]}}: held-out ranges WITH their guards, in episode time (s).

    Tolerant of the lock's layout: a top-level "episodes" list or mapping; per episode either whole-episode
    ("held_out": true) or per-camera lists of ranges given as [t0, t1] or {"start"/"t0", "end"/"t1", "guard"}."""
    raw = json.loads(Path(path).read_text())
    eps = raw.get("episodes", raw)
    items = eps.items() if isinstance(eps, dict) else ((e.get("episode_index", e.get("episode")), e) for e in eps)
    out: dict = {}
    for k, v in items:
        if k is None: continue
        k = int(k)
        if isinstance(v, dict) and (v.get("held_out") is True or v.get("whole") is True):
            out[k] = {"*": [(float("-inf"), float("inf"))]}; continue
        cams = v.get("cameras", v) if isinstance(v, dict) else {"*": v}
        guard_default = float(v.get("guard_s", raw.get("guard_s", 0.0))) if isinstance(v, dict) else float(raw.get("guard_s", 0.0))
        per = {}
        for cam, ranges in cams.items():
            if not isinstance(ranges, list): continue
            rr = []
            for r in ranges:
                if isinstance(r, dict):
                    t0 = float(r.get("start", r.get("t0"))); t1 = float(r.get("end", r.get("t1"))); g = float(r.get("guard", r.get("guard_s", guard_default)))
                else:
                    t0, t1 = map(float, r[:2]); g = guard_default
                rr.append((t0 - g, t1 + g))
            per[cam.replace("observation.images.", "")] = rr
        out[k] = per
    return out


def episode_videos(dataset) -> dict:
    """{episode: {camera: (video file, from_s, to_s)}} from LeRobot v3's meta/episodes (call only after the lock)."""
    import pandas as pd
    ds = Path(dataset); info = json.loads((ds / "meta/info.json").read_text())
    tmpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    e = pd.concat([pd.read_parquet(p) for p in sorted((ds / "meta/episodes").rglob("*.parquet"))])
    keys = [k for k in info["features"] if k.startswith("observation.images.")]
    out = {}
    for _, r in e.iterrows():
        ep = int(r["episode_index"]); out[ep] = {}
        for k in keys:
            ci, fi = int(r[f"videos/{k}/chunk_index"]), int(r[f"videos/{k}/file_index"])
            path = ds / tmpl.format(video_key=k, chunk_index=ci, file_index=fi)
            out[ep][k.replace("observation.images.", "")] = (path, float(r[f"videos/{k}/from_timestamp"]), float(r[f"videos/{k}/to_timestamp"]))
    return out


_KEYS: dict = {}


def keyframes(video) -> list[float]:
    """Keyframe times (s) of a video from its packet headers: ffprobe reads packets, decodes no frame."""
    v = str(video)
    if v not in _KEYS:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                              "-of", "csv=p=0", v], capture_output=True, text=True, check=True).stdout
        _KEYS[v] = sorted(float(t) for t, f in (l.split(",")[:2] for l in out.splitlines() if l.strip()) if "K" in f and t not in ("", "N/A"))
    return _KEYS[v]


def allowed_segments(dataset, lock_path, cams=None) -> list[dict]:
    """Allowed (episode, camera) segments in file time, each starting at a keyframe and ending before any held-out
    range (with guard). Whole-episode locks give nothing for that episode."""
    raw = json.loads(Path(lock_path).read_text())
    if "allowed" in raw and isinstance(raw.get("episodes"), dict) and "usable_file_range_with_guard" in json.dumps(raw["episodes"]):
        # the reviewer's episode lock (RUBRIC 6.2): only the allowed episodes, inside their usable (guarded) ranges
        segs = []
        for ep in raw["allowed"]:
            e = raw["episodes"][str(ep)]
            for key, v in e["videos"].items():
                cam = key.replace("observation.images.", "")
                if cams and cam not in cams: continue
                path = Path(dataset) / v["file"]; a, b_ = v["usable_file_range_with_guard"]; f0 = v["from_timestamp"]
                start = next((k for k in keyframes(path) if a - 1e-6 <= k < b_), None)
                if start is None or start >= b_ - 1.0 / 30: continue
                segs.append({"episode": int(ep), "camera": cam, "video": str(path), "file_start": start, "file_end": b_,
                             "episode_start": start - f0, "episode_end": b_ - f0})
        return segs
    lock = load_lock(lock_path); vids = episode_videos(dataset); segs = []
    for ep, per_cam in sorted(vids.items()):
        for cam, (path, f0, f1) in per_cam.items():
            if cams and cam not in cams: continue
            held = sorted(lock.get(ep, {}).get(cam, []) + lock.get(ep, {}).get("*", []))
            free, t = [], 0.0; dur = f1 - f0
            for a, b in held:
                if a > t: free.append((t, min(a, dur)))
                t = max(t, b)
            if t < dur: free.append((t, dur))
            ks = keyframes(path)
            for a, b in free:
                start = next((k for k in ks if f0 + a <= k < f0 + b), None)   # first keyframe inside, in file time
                if start is None or start >= f0 + b - 1.0 / 30: continue
                segs.append({"episode": ep, "camera": cam, "video": str(path), "file_start": start, "file_end": f0 + b,
                             "episode_start": start - f0, "episode_end": b})
    return segs


def extract(seg: dict, out_dir, step: int = 1) -> list[Path]:
    """Decode one allowed segment (from its keyframe) to PNGs named <camera>_e<episode>_t<episode time ms>.png."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pattern = out / f"_tmp_{seg['camera']}_{seg['episode']}_%06d.png"
    vf = f"select=not(mod(n\\,{step}))" if step > 1 else "null"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{seg['file_start']:.6f}", "-i", seg["video"], "-t", f"{seg['file_end'] - seg['file_start'] - 1e-3:.6f}",
                    "-vf", vf, "-vsync", "0", str(pattern)], check=True)
    files = []
    for i, p in enumerate(sorted(out.glob(f"_tmp_{seg['camera']}_{seg['episode']}_*.png"))):
        t_ms = int(round((seg["episode_start"] + i * step / 30.0) * 1000))
        q = out / f"{seg['camera']}_e{seg['episode']:03d}_t{t_ms:07d}.png"; p.rename(q); files.append(q)
    return files


# --- the reviewer's export (RUBRIC H3: the ONLY robot-camera material the builder may use) -----------------------
# review/export/robotcam_allowed/: ep{E:03d}_{camera}.mp4 (each allowed episode cut at its guarded range, lossless,
# frame k = episode frame_index start + k) and states_allowed.parquet (the same frame ranges).

def export_segments(export_dir, cams=None) -> list[dict]:
    """One segment per export file: episode, camera, the file, the episode frame_index range it holds (from
    states_allowed.parquet) and its episode time range."""
    import pandas as pd
    ex = Path(export_dir); st = pd.read_parquet(ex / "states_allowed.parquet", columns=["episode_index", "frame_index"])
    rng = st.groupby("episode_index")["frame_index"].agg(["min", "max", "count"])
    segs = []
    for f in sorted(ex.glob("ep*_*.mp4")):
        ep, cam = int(f.stem[2:5]), f.stem.split("_", 1)[1]
        if cams and cam not in cams: continue
        lo, hi, n = (int(v) for v in rng.loc[ep])
        segs.append({"episode": ep, "camera": cam, "video": str(f), "frame_first": lo, "frame_last": hi, "frames": n,
                     "episode_start": lo / 30.0, "episode_end": (hi + 1) / 30.0, "source": "review/export/robotcam_allowed"})
    return segs


def decode_export(seg: dict, fmt: str = "rgb24"):
    """Yield (frame_index, frame) for every frame of an export file (raw decode through ffmpeg)."""
    import numpy as np
    ch = {"rgb24": 3, "gray": 1}[fmt]; W, H = 640, 480
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", seg["video"], "-f", "rawvideo", "-pix_fmt", fmt, "-vsync", "0", "-"], stdout=subprocess.PIPE)
    k = 0
    while True:
        raw = p.stdout.read(W * H * ch)
        if len(raw) < W * H * ch: break
        frame = np.frombuffer(raw, np.uint8).reshape(H, W, ch) if ch == 3 else np.frombuffer(raw, np.uint8).reshape(H, W)
        yield seg["frame_first"] + k, frame; k += 1
    p.wait()


def extract_export(seg: dict, out_dir, step: int = 1, phase: int | None = None) -> list[Path]:
    """Write every *step*-th frame of an export file (frame_index % step == phase; default phase: the file's first
    frame_index % step) as <camera>_e<episode>_t<episode time ms>.png."""
    import cv2
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); files = []
    ph = seg["frame_first"] % step if phase is None else phase
    for fi, frame in decode_export(seg):
        if fi % step != ph: continue
        q = out / f"{seg['camera']}_e{seg['episode']:03d}_t{int(round(fi / 30.0 * 1000)):07d}.png"
        cv2.imwrite(str(q), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)); files.append(q)
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("lock", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--step", type=int, default=1); ap.add_argument("--cams"); ap.add_argument("--max-per-episode", type=int, default=0)
    ap.add_argument("--export", action="store_true", help="DATASET is review/export/robotcam_allowed (H3); usage: --export EXPORT OUT")
    ap.add_argument("--phase", default="", help="E:p,... frame_index %% step for episode E (default: its first frame)")
    a = ap.parse_args()
    if a.export:
        out = Path(a.lock); out.mkdir(parents=True, exist_ok=True)
        phases = {int(k): int(v) for k, v in (x.split(":") for x in a.phase.split(",") if x)}
        segs = export_segments(a.dataset, a.cams.split(",") if a.cams else None); record = []
        for s in segs:
            fs = extract_export(s, out, a.step, phases.get(s["episode"]))
            record.append({**s, "decoded_frames": s["frames"], "written_every": a.step, "phase": phases.get(s["episode"], s["frame_first"] % a.step), "written": len(fs)})
            print(f"episode {s['episode']} {s['camera']}: frames {s['frame_first']}-{s['frame_last']} decoded {s['frames']}, wrote {len(fs)}")
        (out / "decode_record.json").write_text(json.dumps({"script": "real_episodes.py --export", "segments": record}, indent=1))
        raise SystemExit(0)
    segs = allowed_segments(a.dataset, a.lock, a.cams.split(",") if a.cams else None)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "segments.json").write_text(json.dumps(segs, indent=1))
    for s in segs:
        fs = extract(s, a.out, a.step)
        print(f"episode {s['episode']} {s['camera']}: {s['episode_start']:.2f}-{s['episode_end']:.2f} s -> {len(fs)} frames")
