"""Brush training folder from the undistorted model, without the reviewer's held-out phone frames (windows + guard).
Never opens held-out images: filters by file name (index at 15 fps).
usage: make_train_set.py SRC_UNDISTORTED OUT [--masks DIR] [--exclude DEV_SPLIT.json]   (--exclude: also leave out its "frames")"""
import argparse, json, shutil, sys
from pathlib import Path
import pycolmap
sys.path.insert(0, str(Path(__file__).parent))
from common import held as held_name
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--masks"); ap.add_argument("--exclude"); a = ap.parse_args()
EXCL = set(json.load(open(a.exclude))["frames"]) if a.exclude else set()
def held(name):
    return name in EXCL or held_name(name)
src, out = Path(a.src), Path(a.out)
r = pycolmap.Reconstruction(src / "sparse")
drop = [i for i, im in r.images.items() if held(im.name)]
if out.exists(): shutil.rmtree(out)
(out / "sparse/0").mkdir(parents=True); (out / "images").mkdir()
r.write_text(out / "sparse/0")
for f in (out / "sparse/0").iterdir():
    if f.name not in ("cameras.txt", "images.txt", "points3D.txt"): f.unlink()
lines = (out / "sparse/0/images.txt").read_text().splitlines(); body = []; i = 0
while i < len(lines):  # header comments, then two lines per image (pose line, 2D points line)
    if lines[i].startswith("#"): body.append(lines[i]); i += 1; continue
    if not held(lines[i].split()[9]): body += lines[i:i + 2]
    i += 2
(out / "sparse/0/images.txt").write_text("\n".join(body) + "\n")
kept = [im.name for im in r.images.values() if not held(im.name) and im.has_pose]
for n in kept:
    (out / "images" / n).symlink_to((src / "images" / n).resolve())
if a.masks:
    (out / "masks").mkdir()
    for n in kept: shutil.copy(Path(a.masks) / (Path(n).stem + ".png"), out / "masks" / (Path(n).stem + ".png"))
json.dump({"held_out_dropped": len(drop), "train": sorted(kept)}, open(out / "train_list.json", "w"))
print(f"train {len(kept)} images, held out {len(drop)}")
