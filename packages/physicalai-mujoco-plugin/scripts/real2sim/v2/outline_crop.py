"""Builder check on ALLOWED frames: the matched-state outline over the real (PINHOLE) image, full resolution, cropped.
usage: outline_crop.py OUT.jpg FRAME:x0:y0:x1:y1 [...]"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import held, ROOT
specs = [s.split(":") for s in sys.argv[2:]]; names = [s[0] if s[0].endswith(".jpg") else s[0] + ".jpg" for s in specs]
for n in names: assert not held(n), n
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, "-I", str(Path(__file__).parent / "bucketsim_v2.py"), "heldout_render", "--frames", ",".join(names), "--out", str(tmp)],
               check=True, env=dict(os.environ), stdout=subprocess.DEVNULL)
tiles = []
for (n, x0, y0, x1, y1) in [(nm, *map(int, s[1:])) for nm, s in zip(names, specs)]:
    real = cv2.imread(str(ROOT / "colmap/pinhole/images" / n)); ol = cv2.imread(str(tmp / f"{Path(n).stem}_outline.png"), cv2.IMREAD_UNCHANGED)
    a = ol[..., 3] > 0; real[a] = ol[..., :3][a]; c = real[y0:y1, x0:x1]
    cv2.putText(c, n, (5, 22), 0, 0.7, (0, 255, 255), 2); tiles.append(cv2.resize(c, (800, int(800 * c.shape[0] / c.shape[1]))))
cv2.imwrite(sys.argv[1], np.vstack(tiles))
