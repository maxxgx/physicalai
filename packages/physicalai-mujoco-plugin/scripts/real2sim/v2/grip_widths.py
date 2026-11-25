"""Object widths from the real gripper (evidence for the object sizes): in each allowed episode, plateaus of the
follower's gripper.pos (>= 15 frames changing < 0.3 per frame) below the open level are the fingers resting closed on a
carried object; the opening is 2 x |pos| / 5400 m (rebot_b601 REBOT_B601_GRIPPER_SCALE, provisional: -270 deg = 0.05 m
per finger). Source: the reviewer's export only (states_allowed.parquet, H3).  usage: grip_widths.py EXPORT_DIR OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ex, out = Path(sys.argv[1]), Path(sys.argv[2])
st = pd.read_parquet(ex / "states_allowed.parquet", columns=["episode_index", "frame_index", "observation.state", "task_index"])
tasks = pd.read_parquet(ex / "tasks.parquet"); task_of = {int(r): str(t) for t, r in zip(tasks.index, tasks["task_index"])}
res = {"source": "review/export/robotcam_allowed/states_allowed.parquet", "scale_m_per_deg_per_finger": 0.05 / 270, "episodes": {}}
for ep, g in st.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
    v = np.stack(g["observation.state"].values)[:, 6]; fi = g["frame_index"].to_numpy(); runs = []; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and abs(v[j + 1] - v[j]) < 0.3: j += 1
        med = float(np.median(v[i:j + 1]))
        if j - i >= 15 and med > -250: runs.append({"frames": [int(fi[i]), int(fi[j])], "gripper_deg": round(med, 1), "opening_mm": round(2 * abs(med) / 5400 * 1000, 1)})
        i = j + 1
    res["episodes"][int(ep)] = {"task": task_of[int(g["task_index"].iloc[0])], "frame_range": [int(fi.min()), int(fi.max())], "rows": int(len(g)), "carry_plateaus": runs}
for kind, key in (("resistor", "resistor"), ("cpu", "CPU")):
    w = [r["opening_mm"] for e in res["episodes"].values() if key in e["task"] for r in e["carry_plateaus"]]
    res[f"{kind}_opening_mm"] = {"values": w, "median": float(np.median(w)) if w else None}
out.write_text(json.dumps(res, indent=1))
print({k: res[k] for k in ("resistor_opening_mm", "cpu_opening_mm")})
