"""Gridded zoom of an ALLOWED frame for hand marks: crop (cx, cy, half) scaled by K with lines every 10 px (labels every
50 px, full-frame coordinates). Refuses held-out frames. usage: crop_grid.py IMG cx cy half K OUT"""
import sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import held
p, cx, cy, half, k, out = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
assert not held(Path(p).name), "held-out frame"
im = cv2.imread(p); x0, y0 = max(0, cx - half), max(0, cy - half); c = im[y0:cy + half, x0:cx + half]
z = cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
for x in range((x0 // 10 + 1) * 10, x0 + c.shape[1], 10):
    col = (0, 255, 255) if x % 50 == 0 else (90, 90, 90); cv2.line(z, ((x - x0) * k, 0), ((x - x0) * k, z.shape[0]), col, 1)
    if x % 50 == 0: cv2.putText(z, str(x), ((x - x0) * k + 2, 12), 0, 0.4, (0, 255, 255), 1)
for y in range((y0 // 10 + 1) * 10, y0 + c.shape[0], 10):
    col = (0, 255, 255) if y % 50 == 0 else (90, 90, 90); cv2.line(z, (0, (y - y0) * k), (z.shape[1], (y - y0) * k), col, 1)
    if y % 50 == 0: cv2.putText(z, str(y), (2, (y - y0) * k - 2), 0, 0.4, (0, 255, 255), 1)
cv2.imwrite(out, z)
