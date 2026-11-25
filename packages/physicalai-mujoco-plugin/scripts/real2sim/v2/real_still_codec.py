"""D7 (r11), the noise target: real still pairs through the real dataset's own AV1 (the export's decoded frames), per
camera and split (TRAIN 0, 2, 6, 7, 9 / DEV 1, 3, 10). A still pair: consecutive frames with every joint within 0.05 deg.
Intra pairs (even frame_index: the second frame is an intra frame under g 2) give the sensor's temporal noise as the
codec keeps it: median mean |diff| at 160x120 grey and its low-frequency part (20x15). Predicted pairs (odd
frame_index) give the share of exact copies (< 0.02), reported as a check. usage: real_still_codec.py EXPORT OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
E = Path(sys.argv[1]); st = pd.read_parquet(E / "states_allowed.parquet")
STARTS = {0: 0, 1: 0, 2: 0, 3: 0, 6: 15, 7: 0, 9: 15, 10: 0}; SPLITS = {"train": (0, 2, 6, 7, 9), "dev": (1, 3, 10)}
res = {"source": "review/export/robotcam_allowed (videos, states_allowed.parquet)", "definition": __doc__.split("\n")[0]}
for cam in ("overview", "gripper"):
    for split, eps in SPLITS.items():
        intra, intra_lf, pred, frames = [], [], [], []
        for ep in eps:
            rows = st[st.episode_index == ep]; S = dict(zip(rows.frame_index.to_numpy(), np.stack(rows["observation.state"].to_numpy()).astype(float)))
            cap = cv2.VideoCapture(str(E / f"ep{ep:03d}_{cam}.mp4")); prev = None; k = 0
            while True:
                ok, f = cap.read()
                if not ok: break
                g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32); j = STARTS[ep] + k
                if prev is not None and j in S and j - 1 in S and np.abs(S[j] - S[j - 1]).max() <= 0.05:
                    dmean = float(np.abs(g - prev).mean()); frames.append(f"e{ep}/f{j}")
                    if j % 2 == 0:
                        intra.append(dmean); intra_lf.append(float(np.abs(cv2.resize(g, (20, 15), interpolation=cv2.INTER_AREA) - cv2.resize(prev, (20, 15), interpolation=cv2.INTER_AREA)).mean()))
                    else:
                        pred.append(dmean)
                prev = g; k += 1
        r = {"intra_pairs": len(intra), "intra_med_diff": float(np.median(intra)), "intra_lowfreq_med": float(np.median(intra_lf)),
             "pred_pairs": len(pred), "pred_copies_pct": float(100 * (np.array(pred) < 0.02).mean()), "pred_med_diff": float(np.median(pred)), "frames": frames}
        res[f"{cam}_{split}"] = r; print(cam, split, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "frames"})
json.dump(res, open(sys.argv[2], "w"), indent=1)
