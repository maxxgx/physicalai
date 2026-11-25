"""D7 (b), CPU only: with AV1 g 2 (intra every 2nd frame) a predicted frame that codes nothing is a copy of the frame
before it. Per parity of the frame index: the mean |diff| to the previous frame (grey, 160x120, the dscheck test) and
the share of exact copies (< 0.02), and the temporal noise of a static background patch (rows 20-110, cols 20-200 at
160x120: the far desk and the room, no arm). usage: gop_parity.py VIDEO..."""
import sys, numpy as np, cv2
for path in sys.argv[1:]:
    cap = cv2.VideoCapture(path); prev = None; k = 0; md = {0: [], 1: []}; bg = {0: [], 1: []}
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            d = np.abs(g - prev); md[k % 2].append(float(d.mean())); bg[k % 2].append(float(d[20:110, 20:200].mean()))
        prev = g; k += 1
    out = []
    for p in (0, 1):
        a, b = np.array(md[p]), np.array(bg[p])
        out.append(f"parity {p}: copies {100 * (a < 0.02).mean():5.1f}%  <0.25 {100 * (a < 0.25).mean():5.1f}%  bg |diff| p50 {np.median(b):.3f}")
    print(path.split("/")[-1] if "ep0" in path else "/".join(path.split("/")[-5:-3])[:40], "|", " | ".join(out))
