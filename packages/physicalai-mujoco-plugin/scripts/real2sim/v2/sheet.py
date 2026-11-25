"""Contact sheet of ALLOWED frames only (held-out rule applied by name). usage: sheet.py DIR T0 T1 N OUT [--fps 30] [--w 400]
DIR names fNNNNN.jpg; --fps is the index rate of the names (30 for v2 names, 15 for colmap/old_v0)."""
import argparse, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import in_heldout
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float)
ap.add_argument("n", type=int); ap.add_argument("out"); ap.add_argument("--fps", type=float, default=30); ap.add_argument("--w", type=int, default=400)
ap.add_argument("--cols", type=int, default=4)
a = ap.parse_args(); d = Path(a.dir)
names = sorted(p.name for p in d.glob("f*.jpg"))
ok = [n for n in names if a.t0 <= int(n[1:6]) / a.fps <= a.t1 and not in_heldout(int(n[1:6]) / a.fps)]
pick = [ok[int(i)] for i in np.linspace(0, len(ok) - 1, min(a.n, len(ok)))]
tiles = []
for n in pick:
    im = cv2.imread(str(d / n)); h = int(im.shape[0] * a.w / im.shape[1]); im = cv2.resize(im, (a.w, h), interpolation=cv2.INTER_AREA)
    cv2.putText(im, f"{n[:6]} {int(n[1:6]) / a.fps:.1f}s", (5, 20), 0, 0.55, (0, 255, 255), 2); tiles.append(im)
while len(tiles) % a.cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(" ".join(pick))
