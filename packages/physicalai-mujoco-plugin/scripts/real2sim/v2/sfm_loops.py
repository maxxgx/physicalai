"""Re-run COLMAP mapping on WORK_DIR's database with extra loop pairs: every frame against the next WINDOW frames and
against every KEY-th frame (loop closure). Features already in the database are reused.
usage: sfm_loops.py WORK_DIR [--window 30] [--key 8]"""
import argparse, shutil, time
from pathlib import Path
import pycolmap

ap = argparse.ArgumentParser(); ap.add_argument("w"); ap.add_argument("--window", type=int, default=30); ap.add_argument("--key", type=int, default=8)
a = ap.parse_args(); w = Path(a.w); db = w / "database.db"; t0 = time.time()
names = sorted(p.name for p in (w / "images").glob("*.jpg"))
pairs = set()
for i, n in enumerate(names):
    for m in names[i + 1:i + 1 + a.window]: pairs.add((n, m))
    for j in range(0, len(names), a.key):
        if abs(j - i) > a.window: pairs.add(tuple(sorted((n, names[j]))))
(w / "pairs_loops.txt").write_text("".join(f"{x} {y}\n" for x, y in sorted(pairs)))
print(len(pairs), "pairs", flush=True)
pm = pycolmap.ImportedPairingOptions(); pm.match_list_path = str(w / "pairs_loops.txt")
pycolmap.match_image_pairs(db, pairing_options=pm)
print(f"matching {time.time()-t0:.0f}s", flush=True)
out = w / "sfm2"
if out.exists(): shutil.rmtree(out)
out.mkdir()
opts = pycolmap.IncrementalPipelineOptions(); opts.min_model_size = 20
recs = pycolmap.incremental_mapping(db, w / "images", out, options=opts)
print(f"mapping {time.time()-t0:.0f}s", flush=True)
for k, r in recs.items():
    ns = sorted(im.name for im in r.images.values()); print(k, r.num_reg_images(), ns[0], ns[-1], r.num_points3D(), flush=True)
best = max(recs.values(), key=lambda r: r.num_reg_images())
(out / "best").mkdir(exist_ok=True); best.write(out / "best")
brush = w / "brush"
if brush.exists(): shutil.rmtree(brush)
pycolmap.undistort_images(brush, out / "best", w / "images")
print(f"undistorted {time.time()-t0:.0f}s", flush=True)
