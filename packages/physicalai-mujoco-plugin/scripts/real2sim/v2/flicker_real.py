"""D7 (r10): frame-to-frame global brightness change of the real cameras (export only), on still pairs: frames k-1, k
of an allowed episode where no joint changed (|delta| <= 1e-4 in observation.state between the matching rows). Per
pair: the change of the mean grey level (full frame, 640x480) and its relative size (over the mean level). Fits the
temporal gain jitter: relative change = g_k - g_{k-1} with g white, so sigma_g = std(rel) / sqrt(2) (also the robust
estimate 1.4826 * MAD / sqrt(2)). usage: flicker_real.py EXPORT OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
E = Path(sys.argv[1]); st = pd.read_parquet(E / "states_allowed.parquet")
STARTS = {0: 0, 1: 0, 2: 0, 3: 0, 6: 15, 7: 0, 9: 15, 10: 0}   # README: episode frame_index of file frame 0
out = {"source": "review/export/robotcam_allowed (videos, states_allowed.parquet)", "cameras": {}}
for cam in ("overview", "gripper"):
    rel, pairs = [], []
    for ep, s0 in STARTS.items():
        rows = st[st.episode_index == ep].sort_values("frame_index") if "frame_index" in st else st[st.episode_index == ep].sort_values("timestamp")
        fi = rows["frame_index"].to_numpy() if "frame_index" in rows else np.round(rows["timestamp"].to_numpy() * 30).astype(int)
        S = dict(zip(fi, np.stack(rows["observation.state"].to_numpy()).astype(float)))
        cap = cv2.VideoCapture(str(E / f"ep{ep:03d}_{cam}.mp4")); prev = None; k = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            m = float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()); j = s0 + k
            if prev is not None and (j in S) and (j - 1 in S) and np.abs(S[j] - S[j - 1]).max() <= 1e-4:
                rel.append((m - prev) / max(prev, 1.0)); pairs.append(f"e{ep}/f{j}")
            prev = m; k += 1
    rel = np.array(rel)
    out["cameras"][cam] = {"still_pairs": len(rel), "rel_change_std": float(rel.std()), "rel_change_abs_p50": float(np.median(np.abs(rel))),
                           "rel_change_abs_p90": float(np.percentile(np.abs(rel), 90)), "sigma_g": float(rel.std() / np.sqrt(2)),
                           "sigma_g_robust": float(1.4826 * np.median(np.abs(rel - np.median(rel))) / np.sqrt(2)),
                           "lag1_autocorr_of_change": float(np.corrcoef(rel[:-1], rel[1:])[0, 1]) if len(rel) > 3 else None, "pairs": pairs}
    c = out["cameras"][cam]; print(cam, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items() if k != "pairs"})
json.dump(out, open(sys.argv[2], "w"), indent=1)
