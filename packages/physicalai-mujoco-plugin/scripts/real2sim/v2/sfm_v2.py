"""Joint COLMAP SfM of every extracted frame (allowed + window frames; one SIMPLE_RADIAL camera), then an undistorted
PINHOLE copy. Pairs: frames within --dt seconds of each other, plus every frame against key frames every --key s.
usage: sfm_v2.py WORK_DIR [--dt 4] [--key 1.5]   (WORK_DIR/images -> database.db, sfm/<k>, sfm/best, pinhole)"""
import argparse, shutil, time
from pathlib import Path
import pycolmap

ap = argparse.ArgumentParser(); ap.add_argument("w"); ap.add_argument("--dt", type=float, default=4.0); ap.add_argument("--key", type=float, default=1.5)
a = ap.parse_args(); w = Path(a.w); db = w / "database.db"; t0 = time.time()
names = sorted(p.name for p in (w / "images").glob("f*.jpg")); T = {n: int(n[1:6]) / 30 for n in names}
pairs = set(); keys = []; last = -1e9
for n in names:
    if T[n] - last >= a.key: keys.append(n); last = T[n]
for i, n in enumerate(names):
    for m in names[i + 1:]:
        if T[m] - T[n] > a.dt: break
        pairs.add((n, m))
    for k in keys:
        if abs(T[k] - T[n]) > a.dt: pairs.add(tuple(sorted((n, k))))
(w / "pairs.txt").write_text("".join(f"{x} {y}\n" for x, y in sorted(pairs)))
print(len(names), "images", len(pairs), "pairs", len(keys), "keys", flush=True)
if not (w / "matched.flag").exists():
    for f in w.glob("database.db*"): f.unlink()
    ex = pycolmap.FeatureExtractionOptions(); ex.sift.max_num_features = 8192
    reader = pycolmap.ImageReaderOptions(); reader.camera_model = "SIMPLE_RADIAL"
    pycolmap.extract_features(db, w / "images", camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ex)
    print(f"features {time.time()-t0:.0f}s", flush=True)
    pm = pycolmap.ImportedPairingOptions(); pm.match_list_path = str(w / "pairs.txt")
    pycolmap.match_image_pairs(db, pairing_options=pm)
    print(f"matching {time.time()-t0:.0f}s", flush=True); (w / "matched.flag").touch()
out = w / "sfm"
if out.exists(): shutil.rmtree(out)
out.mkdir()
opts = pycolmap.IncrementalPipelineOptions(); opts.min_model_size = 10
recs = pycolmap.incremental_mapping(db, w / "images", out, options=opts)
print(f"mapping {time.time()-t0:.0f}s", flush=True)
for k, r in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
    ns = sorted(im.name for im in r.images.values()); print("model", k, r.num_reg_images(), ns[0], ns[-1], r.num_points3D(), flush=True)
best = max(recs.values(), key=lambda r: r.num_reg_images())
(out / "best").mkdir(exist_ok=True); best.write(out / "best")
pin = w / "pinhole"
if pin.exists(): shutil.rmtree(pin)
pycolmap.undistort_images(pin, out / "best", w / "images")
print(f"undistorted {time.time()-t0:.0f}s", flush=True)
