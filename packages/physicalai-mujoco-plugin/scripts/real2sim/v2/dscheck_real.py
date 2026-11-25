"""dscheck_lite's duplicate measure (mean |diff| < 0.25 at 160x120 between consecutive frames, split by whether the arm
moved) on the REAL episodes, allowed ones only: each allowed segment decoded from the first keyframe inside the lock's
usable guarded range (real_episodes.allowed_segments), joint states from allowed rows only.
usage: dscheck_real.py DATASET LOCK"""
import sys, json, subprocess, numpy as np, pandas as pd, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_episodes as R
EXPORT = sys.argv[1] == "--export"
if EXPORT:   # review/export/robotcam_allowed (H3): usage dscheck_real.py --export EXPORT_DIR [RECORD.json]
    D = Path(sys.argv[2]); fps = 30
    df = pd.read_parquet(D / "states_allowed.parquet", columns=["episode_index", "frame_index", "observation.state"])
    SEGS = [{**s_, "file_start": 0.0, "file_end": (s_["frames"] + 0.5) / fps} for s_ in R.export_segments(D)]
    RECORD = Path(sys.argv[3]) if len(sys.argv) > 3 else None
else:
    D, LOCK = Path(sys.argv[1]), sys.argv[2]; lock = json.loads(Path(LOCK).read_text()); allowed = [int(e) for e in lock["allowed"]]; fps = lock["fps"]
    files = sorted({lock["episodes"][str(e)]["data_file"] for e in allowed})
    df = pd.concat([pd.read_parquet(D / f, columns=["episode_index", "frame_index", "observation.state"]) for f in files])
    df = df[df["episode_index"].isin(allowed)]
    SEGS = R.allowed_segments(D, LOCK); RECORD = None
st = {(int(e), int(f)): np.asarray(s, float) for e, f, s in zip(df["episode_index"], df["frame_index"], df["observation.state"])}
W, H = 640, 480; res = {}
decoded = []
for s in SEGS:
    if EXPORT:   # the rubric's dscheck decode: cv2.VideoCapture, BGR -> gray, INTER_AREA to 160x120 (R2S tools; H5 r08)
        cap = cv2.VideoCapture(s["video"]); frames = []
        while True:
            ok, im = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).tobytes())
        import io
        p = type("P", (), {"stdout": io.BytesIO(b"".join(frames)), "wait": lambda self: None})()
    else:
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{s['file_start']:.6f}", "-i", s["video"], "-t", f"{s['file_end'] - s['file_start'] - 1e-3:.6f}",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"], stdout=subprocess.PIPE)
    prev, n, r = None, 0, res.setdefault(s["camera"], {"n": 0, "dup": 0, "mv": 0, "dup_mv": 0, "mv1": 0, "dup_mv1": 0})
    f0 = int(round(s["episode_start"] * fps))
    while True:
        raw = p.stdout.read(W * H)
        if len(raw) < W * H: break
        g = cv2.resize(np.frombuffer(raw, np.uint8).reshape(H, W), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        fi = f0 + n; n += 1
        if prev is not None and (s["episode"], fi) in st and (s["episode"], fi - 1) in st:
            dq = float(np.abs(st[(s["episode"], fi)] - st[(s["episode"], fi - 1)]).max()); dup = float(np.abs(g - prev).mean()) < 0.25
            r["n"] += 1; r["dup"] += dup; r["mv"] += dq > 1e-4; r["dup_mv"] += dup and dq > 1e-4; r["mv1"] += dq > 1.0; r["dup_mv1"] += dup and dq > 1.0
        prev = g
    p.wait()
    decoded.append({"episode": s["episode"], "camera": s["camera"], "video": s["video"], "decoded_frames": n, "first_frame_index": f0})
for cam, r in res.items():
    print(f"{cam}: {r['n']} frame pairs, identical to the previous {r['dup']} ({100 * r['dup'] / max(r['n'], 1):.1f} %), while the arm moves "
          f"{r['dup_mv']} ({100 * r['dup_mv'] / max(r['mv'], 1):.1f} % of moving), while a joint moves > 1 deg {r['dup_mv1']} ({100 * r['dup_mv1'] / max(r['mv1'], 1):.1f} % of {r['mv1']})")
if RECORD is not None:
    RECORD.write_text(json.dumps({"script": "dscheck_real.py --export", "definition": "RUBRIC dscheck: cv2.VideoCapture decode, BGR2GRAY, INTER_AREA 160x120, mean |diff| < 0.25; moving = a joint changes > 1e-4", "result": res, "decode_record": decoded}, indent=1))
