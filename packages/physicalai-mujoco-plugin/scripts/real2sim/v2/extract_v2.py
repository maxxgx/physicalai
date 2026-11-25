"""Decode the phone MOV with the review's frozen HLG->SDR recipe (RUBRIC 3.1) at 15 fps (even source frames), 1920 wide,
and keep: every WINDOW frame (even NNNNN inside a held-out window: joint-SfM registration only, never opened), and the
sharpest of every 3 consecutive even frames outside the windows and the guard (training / fitting). Guard frames are
dropped. Names are fNNNNN.jpg with NNNNN = round(pts_time * 30). Never displays anything.
usage: extract_v2.py VIDEO OUT_DIR [--width 1920]"""
import argparse, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import in_heldout, in_window

SDR_RECIPE = ("zscale=tin=arib-std-b67:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=100:p=bt2020,format=gbrpf32le,"
              "zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=pc,format=rgb24")
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("--width", type=int, default=1920)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
tmp = Path(tempfile.mkdtemp(prefix="extract_v2_", dir=out.parent))
vf = f"select='not(mod(n\\,2))',showinfo,{SDR_RECIPE},scale={a.width}:-2:flags=area"
cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", a.video, "-vf", vf, "-fps_mode", "passthrough", "-q:v", "2", str(tmp / "%06d.jpg")]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode: sys.exit(r.stderr[-2000:])
times = [float(m) for m in re.findall(r"showinfo.*?pts_time:\s*([-0-9.]+)", r.stderr)]
files = sorted(tmp.glob("*.jpg")); assert len(files) == len(times), (len(files), len(times))
idx = [round(t * 30) for t in times]
window, allowed = [], []
for f, n, t in zip(files, idx, times):
    if in_window(t): window.append((f, n))
    elif not in_heldout(t): allowed.append((f, n))
for f, n in window: shutil.move(str(f), out / f"f{n:05d}.jpg")
def sharp(f):
    g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE); g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
    return float(cv2.Laplacian(g, cv2.CV_32F).var())
kept = []; group = []
def flush():
    if group:
        f, n, s = max(group, key=lambda x: x[2]); shutil.move(str(f), out / f"f{n:05d}.jpg"); kept.append({"idx": n, "t": n / 30, "sharp": s}); group.clear()
prev = None
for f, n in allowed:
    if prev is not None and n - prev > 2: flush()   # a held-out gap ends the group
    group.append((f, n, sharp(f))); prev = n
    if len(group) == 3: flush()
flush(); shutil.rmtree(tmp)
json.dump({"recipe": SDR_RECIPE, "width": a.width, "decoded": len(files), "window": [n for _, n in window], "allowed": kept},
          open(out.parent / "frames.json", "w"), indent=0)
print(f"{len(files)} even frames decoded; {len(window)} window frames (SfM only); {len(kept)} allowed kept; guard dropped")
