"""Joint states of the ALLOWED real episodes at the extracted frames, in sim joint angles (rad): the dataset's
observation.state (the driver's public units) through the reBot profile's own channel conversion (Channels.to_model).
Rows of held-out episodes are never read past the episode filter. usage: real_states.py DATASET LOCK FRAMES_DIR OUT.json | real_states.py --export EXPORT_DIR FRAMES_DIR OUT.json (H3: the only allowed source)"""
import json, sys, re
from pathlib import Path
import numpy as np, pandas as pd
if sys.argv[1] == "--export":   # review/export/robotcam_allowed (H3): states_allowed.parquet, info.json
    ex, frames, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); fps = 30
    names = json.loads((ex / "info.json").read_text())["features"]["observation.state"]["names"]
    df = pd.read_parquet(ex / "states_allowed.parquet", columns=["episode_index", "frame_index", "observation.state"])
    allowed = set(int(e) for e in df["episode_index"].unique())
else:
    ds, lock, frames, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]), Path(sys.argv[4])
    allowed = set(int(e) for e in lock["allowed"]); fps = lock["fps"]
    info = json.loads((ds / "meta/info.json").read_text()); names = info["features"]["observation.state"]["names"]
    files = sorted({lock["episodes"][str(e)]["data_file"] for e in allowed})
    df = pd.concat([pd.read_parquet(ds / f, columns=["episode_index", "frame_index", "observation.state"]) for f in files])
    df = df[df["episode_index"].isin(allowed)]
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.channels import ArmChannels as Channels
robot = MuJoCoRobot("rebot_b601", scene="bucket_sort_desk_v2", cameras=[], http_port=0, seed=0); robot.connect()
ch = robot._sim.channels[0]; by_name = {c.name: c for c in ch.channels}
arm = [n for n in names if n.replace(".pos", "") in by_name][:7]
idx = {n: names.index(n) for n in arm}
res, n_ok = {}, 0
for f in sorted(frames.glob("*_e*_t*.png")):
    m = re.match(r".*_e(\d+)_t(\d+)\.png", f.name); ep, ms = int(m[1]), int(m[2])
    if ep not in allowed: continue
    fi = int(round(ms / 1000 * fps)); row = df[(df["episode_index"] == ep) & (df["frame_index"] == fi)]
    if row.empty: continue
    st = np.asarray(row.iloc[0]["observation.state"], float)
    q = [Channels.to_model(by_name[n.replace(".pos", "")], float(st[idx[n]])) for n in arm[:6]]
    res[f"{ep}/{ms}"] = q; n_ok += 1
out.write_text(json.dumps({"names": arm, "states": res, "note": "sim joint angles (rad), joint1..6, from observation.state via the profile's channel conversion"}, indent=0))
print("states for", n_ok, "frames; channels", arm)
