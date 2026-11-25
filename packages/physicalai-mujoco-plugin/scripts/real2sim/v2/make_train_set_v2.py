"""Brush training folder from the undistorted joint SfM, without any held-out frame (window or guard; names only, never
opened). usage: make_train_set_v2.py PINHOLE_DIR OUT [--masks DIR] [--exclude DEV_SPLIT.json]"""
import argparse, json, shutil, sys
from pathlib import Path
import pycolmap
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import held as held_name
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--masks"); ap.add_argument("--exclude"); a = ap.parse_args()
EXCL = set(json.load(open(a.exclude))["frames"]) if a.exclude else set()
held = lambda n: n in EXCL or held_name(n)
src, out = Path(a.src), Path(a.out)
r = pycolmap.Reconstruction(src / "sparse")
if out.exists(): shutil.rmtree(out)
(out / "sparse/0").mkdir(parents=True); (out / "images").mkdir()
r.write_text(out / "sparse/0")
for f in (out / "sparse/0").iterdir():
    if f.name not in ("cameras.txt", "images.txt", "points3D.txt"): f.unlink()
lines = (out / "sparse/0/images.txt").read_text().splitlines(); body = []; i = 0
while i < len(lines):
    if lines[i].startswith("#"): body.append(lines[i]); i += 1; continue
    if not held(lines[i].split()[9]): body += lines[i:i + 2]
    i += 2
(out / "sparse/0/images.txt").write_text("\n".join(body) + "\n")
kept = sorted(im.name for im in r.images.values() if not held(im.name))
for n in kept: shutil.copy2(src / "images" / n, out / "images" / n)
if a.masks:
    (out / "masks").mkdir()
    for n in kept: shutil.copy(Path(a.masks) / (Path(n).stem + ".png"), out / "masks" / (Path(n).stem + ".png"))
json.dump({"held_out_dropped": len(r.images) - len(kept), "train": kept}, open(out / "train_list.json", "w"))
print(f"train {len(kept)} images, dropped {len(r.images) - len(kept)} (held out or dev)")
