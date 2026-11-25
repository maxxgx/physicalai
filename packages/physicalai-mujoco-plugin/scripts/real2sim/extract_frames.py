"""Pick the sharpest frame of every group of N from the SDR phone video (Laplacian variance), as JPEG at the source size,
scaled down to MAX_SIDE on the longest side (1920 for 4K). Names are fNNNNN.jpg, NNNNN the index at FPS.
usage: extract_frames.py VIDEO OUT_DIR [--fps 15] [--group 3] [--max-side 1920]"""
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out")
ap.add_argument("--fps", type=float, default=15); ap.add_argument("--group", type=int, default=3)
ap.add_argument("--max-side", type=int, default=1920)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
sw, sh_ = (int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
    "stream=width,height", "-of", "csv=p=0", a.video], capture_output=True, text=True, check=True).stdout.split(",")[:2])
k = min(1.0, a.max_side / max(sw, sh_)); w, h = 2 * round(sw * k / 2), 2 * round(sh_ * k / 2)
proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-vf", f"fps={a.fps},scale={w}:{h}:flags=lanczos",
                         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
group = []; kept = []; i = 0
def flush():
    if group:
        j, s, img = max(group, key=lambda g: g[1])
        cv2.imwrite(str(out / f"f{j:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95]); kept.append((j, s)); group.clear()
while True:
    buf = proc.stdout.read(w * h * 3)
    if len(buf) < w * h * 3: break
    img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    g = cv2.cvtColor(cv2.resize(img, (w // 2, h // 2)), cv2.COLOR_BGR2GRAY)
    group.append((i, float(cv2.Laplacian(g, cv2.CV_32F).var()), img.copy())); i += 1
    if len(group) == a.group: flush()
flush()
sh = np.array([s for _, s in kept])
json.dump({"fps": a.fps, "group": a.group, "frames": [{"idx": j, "t": j / a.fps, "sharp": s} for j, s in kept]},
          open(out.parent / "frames.json", "w"), indent=0)
print(f"{i} decoded, {len(kept)} kept; sharpness median {np.median(sh):.0f}, p10 {np.percentile(sh,10):.0f}")
