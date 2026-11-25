"""D7 (r10): what separates consecutive frames, real against sim: per consecutive pair (grey, INTER_AREA 160x120, the
dscheck test), mean |diff| and the global part |mean(diff)| (brightness flicker), for the overview camera.
usage: diff_stats.py VIDEO [VIDEO...]   (real: the export's allowed episode videos; sim: a Studio dataset's video)"""
import sys, numpy as np, cv2
for path in sys.argv[1:]:
    cap = cv2.VideoCapture(path); prev = None; md, gm = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None: d = g - prev; md.append(float(np.abs(d).mean())); gm.append(float(abs(d.mean())))
        prev = g
    md, gm = np.array(md), np.array(gm)
    print(path.split("/")[-1] if "ep0" in path else path.split("/")[-4][:8] + "/" + path.split("/")[-3], "pairs", len(md), "identical(<0.25) %.1f%%" % (100 * (md < 0.25).mean()),
          "mean|diff| p10 %.2f p50 %.2f" % tuple(np.percentile(md, [10, 50])), "global |mean diff| p50 %.3f p90 %.3f" % tuple(np.percentile(gm, [50, 90])))
