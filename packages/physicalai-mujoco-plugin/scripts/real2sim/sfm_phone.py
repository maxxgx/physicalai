"""COLMAP SfM of the phone frames (one SIMPLE_RADIAL camera), then an undistorted PINHOLE copy for Brush.
usage: sfm_phone.py WORK_DIR [--window 20]   (WORK_DIR/images in; WORK_DIR/sfm, WORK_DIR/brush out)
Matches each frame with the next WINDOW frames in video order (WORK_DIR/pairs.txt, written when missing)."""
import sys, time, shutil
from pathlib import Path
import pycolmap

w = Path(sys.argv[1]); db = w / "database.db"; t0 = time.time()
win = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 20
if not (w / "pairs.txt").exists():
    names = sorted(p.name for p in (w / "images").glob("*.jpg"))
    (w / "pairs.txt").write_text("".join(f"{a} {b}\n" for i, a in enumerate(names) for b in names[i + 1:i + 1 + win]))
if not (w / "matched.flag").exists():
    if db.exists(): db.unlink()
    ex = pycolmap.FeatureExtractionOptions(); ex.sift.max_num_features = 8192
    reader = pycolmap.ImageReaderOptions(); reader.camera_model = "SIMPLE_RADIAL"
    pycolmap.extract_features(db, w / "images", camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ex)
    print(f"features {time.time()-t0:.0f}s", flush=True)
    pm = pycolmap.ImportedPairingOptions(); pm.match_list_path = str(w / "pairs.txt")
    pycolmap.match_image_pairs(db, pairing_options=pm)
    print(f"matching {time.time()-t0:.0f}s", flush=True); (w / "matched.flag").touch()
(w / "sfm").mkdir(exist_ok=True)
recs = pycolmap.incremental_mapping(db, w / "images", w / "sfm")
print(f"mapping {time.time()-t0:.0f}s", flush=True)
for k, r in recs.items():
    print(k, r.summary())
best = max(recs.values(), key=lambda r: r.num_reg_images())
best.write(w / "sfm" / "best")
out = w / "brush"
if out.exists(): shutil.rmtree(out)
pycolmap.undistort_images(out, w / "sfm" / "best", w / "images")
print(f"undistorted {time.time()-t0:.0f}s", flush=True)
