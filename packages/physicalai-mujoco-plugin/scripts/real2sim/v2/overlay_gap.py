"""Builder's dev-split overlay check (not the reviewer's): per bucket and ALLOWED frame (bucket wholly in view, its real
silhouette among the largest 50 %), the median distance in full-resolution pixels from the sim CAD silhouette's outline
to the real colour silhouette's outline (both ways, averaged), for a given calibration and bucket height scale.
usage: overlay_gap.py CALIBRATION.json KZ [COLOURS]"""
import json, os, sys, numpy as np, cv2
cal_path, kz = sys.argv[1], float(sys.argv[2]); cols = sys.argv[3] if len(sys.argv) > 3 else "green,red,blue,black"
os.environ["KZ"] = "1"; os.environ["FRAME_STEP"] = "2"; sys.argv = ["x", cols]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_buckets_photo.py")).read(); head = src[:src.index("res = {}")]
g = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_buckets_photo.py"), "__name__": "x"}; exec(compile(head, "fit", "exec"), g)
# the mesh as compiled may carry the scene's height scale: undo it, then apply this check's
scene_kz = float(np.ptp(g["Vb"][:, 2]) / 0.120); g["Vb"] = g["Vb"] * [1, 1, kz / scene_kz]
cal = json.load(open(cal_path)); DS = g["DS"]
def outline(mask): return cv2.Canny(mask * 255, 50, 150) > 0
rows = []
for c, D in g["data"].items():
    a = np.array([float(r[4].sum()) for r in D]); D = [r for r, a_ in zip(D, a) if a_ >= np.quantile(a, 0.5)]
    p = np.array([cal["buckets"][c]["x"], cal["buckets"][c]["y"], cal["buckets"][c]["yaw"]])
    for n, cam, w, h, real, valid in D:
        s = g["sil"](p, cam, w, h)
        if s is None: continue
        es, er = outline(s) & valid, outline(real) & valid
        if es.sum() < 10 or er.sum() < 10: continue
        dr = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3); ds = cv2.distanceTransform((~es).astype(np.uint8), cv2.DIST_L2, 3)
        rows.append((c, n, DS * 0.5 * (np.median(dr[es]) + np.median(ds[er]))))
gaps = np.array([r[2] for r in rows])
for c in cols.split(","):
    gc = np.array([r[2] for r in rows if r[0] == c]); print(f"  {c}: {len(gc)} frames, median {np.median(gc):.0f} px, 90th {np.percentile(gc, 90):.0f} px")
print(f"ALL {len(gaps)} bucket-frames: median {np.median(gaps):.1f} px, 90th {np.percentile(gaps, 90):.1f} px, worst {gaps.max():.0f} px")
