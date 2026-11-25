"""Register frames the joint SfM left out into it, keeping every registered pose and the camera fixed: extra pairs for
each unregistered frame (all frames within --dt s, key frames every --key s), then incremental mapping continued from
sfm/best with relaxed absolute-pose thresholds. Writes sfm/best2 and re-undistorts into pinhole/.
usage: sfm_more.py WORK_DIR [--dt 8] [--key 0.5]"""
import argparse, shutil, time
from pathlib import Path
import pycolmap
ap = argparse.ArgumentParser(); ap.add_argument("w"); ap.add_argument("--dt", type=float, default=8.0); ap.add_argument("--key", type=float, default=0.5)
a = ap.parse_args(); w = Path(a.w); db = w / "database.db"; t0 = time.time()
r0 = pycolmap.Reconstruction(w / "sfm/best"); reg = {im.name for im in r0.images.values()}
names = sorted(p.name for p in (w / "images").glob("f*.jpg")); T = {n: int(n[1:6]) / 30 for n in names}
miss = [n for n in names if n not in reg]
keys, last = [], -1e9
for n in names:
    if n in reg and T[n] - last >= a.key: keys.append(n); last = T[n]
pairs = set()
for n in miss:
    for m in names:
        if m != n and (abs(T[m] - T[n]) <= a.dt or m in keys): pairs.add(tuple(sorted((n, m))))
(w / "pairs_more.txt").write_text("".join(f"{x} {y}\n" for x, y in sorted(pairs)))
print(len(miss), "unregistered,", len(pairs), "extra pairs", flush=True)
pm = pycolmap.ImportedPairingOptions(); pm.match_list_path = str(w / "pairs_more.txt")
pycolmap.match_image_pairs(db, pairing_options=pm)
print(f"matching {time.time()-t0:.0f}s", flush=True)
out = w / "sfm_more"
if out.exists(): shutil.rmtree(out)
out.mkdir()
o = pycolmap.IncrementalPipelineOptions(); o.fix_existing_frames = True; o.constant_cameras = {1}; o.min_model_size = 10
o.mapper.abs_pose_min_num_inliers = 12; o.mapper.abs_pose_min_inlier_ratio = 0.12; o.mapper.abs_pose_max_error = 16.0; o.min_num_matches = 10
recs = pycolmap.incremental_mapping(db, w / "images", out, options=o, input_path=w / "sfm/best")
print(f"mapping {time.time()-t0:.0f}s", flush=True)
best = max(recs.values(), key=lambda r: r.num_reg_images())
moved = 0
for im in best.images.values():
    if im.name in reg:
        i0 = next(i for i in r0.images.values() if i.name == im.name)
        moved = max(moved, float(abs(im.cam_from_world().matrix() - i0.cam_from_world().matrix()).max()))
print("registered", best.num_reg_images(), "max pose change of the old frames", moved, flush=True)
(w / "sfm/best2").mkdir(exist_ok=True); best.write(w / "sfm/best2")
pin = w / "pinhole"
if pin.exists(): shutil.rmtree(pin)
pycolmap.undistort_images(pin, w / "sfm/best2", w / "images")
print(f"undistorted {time.time()-t0:.0f}s", flush=True)
