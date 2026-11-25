"""Builder's check on ALLOWED frames: real (undistorted PINHOLE image) | matched-state composite | outline over real,
for frames chosen by time. Refuses held-out names. usage: dev_sbs.py OUT.jpg t1 t2 ... [--w 640]"""
import os, subprocess, sys, json, tempfile
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import held, ROOT
args = [a for a in sys.argv[2:] if not a.startswith("--")]; W = 640
cams = json.load(open(ROOT / "colmap/pinhole_cameras.json"))
names = []
for t in map(float, args):
    cand = sorted((abs(int(n[1:6]) / 30 - t), n) for n in cams if not held(n))
    names.append(cand[0][1])
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, "-I", str(Path(__file__).parent / "bucketsim_v2.py"), "heldout_render", "--frames", ",".join(names), "--out", str(tmp)],
               check=True, env=dict(os.environ), stdout=subprocess.DEVNULL)
rows = []
for n in names:
    assert not held(n)
    real = cv2.imread(str(ROOT / "colmap/pinhole/images" / n)); st = Path(n).stem
    ren = cv2.imread(str(tmp / f"{st}.png")); ol = cv2.imread(str(tmp / f"{st}_outline.png"), cv2.IMREAD_UNCHANGED)
    over = real.copy(); a = ol[..., 3] > 0; over[a] = ol[..., :3][a]
    h = int(real.shape[0] * W / real.shape[1]); row = np.hstack([cv2.resize(x, (W, h), interpolation=cv2.INTER_AREA) for x in (real, ren, over)])
    cv2.putText(row, st, (6, 22), 0, 0.7, (0, 255, 255), 2); rows.append(row)
cv2.imwrite(sys.argv[1], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(" ".join(names))
