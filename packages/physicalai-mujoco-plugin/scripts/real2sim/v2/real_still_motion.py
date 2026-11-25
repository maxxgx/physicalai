"""D7 (r11, user decision U-2): what makes real still frames differ in a way AV1 keeps. On allowed still pairs (every
joint within 0.05 deg between the two frames; the export only; frame lists recorded), per camera and parity
(intra / predicted under g 2): (1) micro-motion: the global sub-pixel shift between the frames (cv2.phaseCorrelate,
full resolution, Hann window) and its peak response; (2) exposure: the global gain (mean ratio of the second frame
to the first); (3) rolling flicker: the row-mean difference profile, its strongest period (FFT over 480 rows) and
amplitude; (4) the residual |diff| after removing the gain. Predicted pairs are split into copies (< 0.02 at 160x120,
the dscheck test) and survivors, to see which component the survivors carry. usage: real_still_motion.py EXPORT OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
E = Path(sys.argv[1]); st = pd.read_parquet(E / "states_allowed.parquet")
STARTS = {0: 0, 1: 0, 2: 0, 3: 0, 6: 15, 7: 0, 9: 15, 10: 0}
win = cv2.createHanningWindow((640, 480), cv2.CV_64F)
res = {"source": "review/export/robotcam_allowed (videos, states_allowed.parquet)", "method": __doc__.split("\n")[0], "cameras": {}}
for cam in ("overview", "gripper"):
    rows_out = []
    for ep, s0 in STARTS.items():
        rows = st[st.episode_index == ep]; S = dict(zip(rows.frame_index.to_numpy(), np.stack(rows["observation.state"].to_numpy()).astype(float)))
        cap = cv2.VideoCapture(str(E / f"ep{ep:03d}_{cam}.mp4")); prev = None; k = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64); j = s0 + k
            if prev is not None and j in S and j - 1 in S and np.abs(S[j] - S[j - 1]).max() <= 0.05:
                (dx, dy), resp = cv2.phaseCorrelate(prev.copy(), g.copy(), win)   # (it windows its inputs in place)
                gain = float(g.mean() / max(prev.mean(), 1e-6))
                rowdiff = (g - prev).mean(axis=1); spec = np.abs(np.fft.rfft(rowdiff - rowdiff.mean()))
                kmax = int(np.argmax(spec[2:]) + 2)
                small = lambda a: cv2.resize(a.astype(np.float32), (160, 120), interpolation=cv2.INTER_AREA)
                dm = float(np.abs(small(g) - small(prev)).mean()); resid = float(np.abs(small(g) - small(prev) * gain).mean())
                rows_out.append({"pair": f"e{ep}/f{j}", "parity": "intra" if j % 2 == 0 else "pred", "copy": dm < 0.02, "dmean": dm,
                                 "shift_px": float(np.hypot(dx, dy)), "corr_peak": float(resp), "gain_minus_1": gain - 1.0,
                                 "row_flicker_period_rows": 480.0 / kmax, "row_flicker_amp": float(spec[kmax] * 2 / len(rowdiff)), "resid_after_gain": resid})
            prev = g; k += 1
    def summ(sel):
        if not sel: return None
        a = lambda key: np.array([r[key] for r in sel])
        return {"n": len(sel), "shift_px_med": float(np.median(a("shift_px"))), "shift_px_p90": float(np.percentile(a("shift_px"), 90)),
                "abs_gain_change_med": float(np.median(np.abs(a("gain_minus_1")))), "abs_gain_change_p90": float(np.percentile(np.abs(a("gain_minus_1")), 90)),
                "row_flicker_amp_med": float(np.median(a("row_flicker_amp"))), "dmean_med": float(np.median(a("dmean"))), "resid_after_gain_med": float(np.median(a("resid_after_gain")))}
    out = {"intra": summ([r for r in rows_out if r["parity"] == "intra"]), "pred_copies": summ([r for r in rows_out if r["parity"] == "pred" and r["copy"]]),
           "pred_survivors": summ([r for r in rows_out if r["parity"] == "pred" and not r["copy"]]), "pairs": rows_out}
    res["cameras"][cam] = out
    for k_ in ("intra", "pred_copies", "pred_survivors"): print(cam, k_, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in (out[k_] or {}).items()})
json.dump(res, open(sys.argv[2], "w"), indent=1)
