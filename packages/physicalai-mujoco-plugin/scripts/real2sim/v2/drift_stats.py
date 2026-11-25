"""r15 (D7): the arm's speed shares and still runs, real (the reviewer's allowed export: TRAIN 0, 2, 6, 7, 9 and DEV
1, 3, 10) against the sim demonstrator at operator-drift scales (the scene's ``r2s_operator_drift``), own seeds.

Speed: the largest joint change of the six arm joints per 1/30 s (deg). Bins: < 0.1 (near-still), 0.1-0.3, 0.3-1,
1-3, >= 3. Still runs: consecutive near-still frames (lengths in 1/30 s frames). The sim is sampled at its 50 Hz ticks
and scaled to 1/30 s; each episode from the attempt's start to its outcome, as recorded. Per sim run also the share of
near-still by demonstrator phase. DEV chooses the scale (closest near-still share and still-run p90 to DEV's).
usage: drift_stats.py STATES_PARQUET OUT_JSON --scales 0,0.75,1,1.25 --seeds 9300:9324"""
import json, os, sys, time, collections
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b

ST, OUT = sys.argv[1], sys.argv[2]
SCALES = [float(x) for x in sys.argv[sys.argv.index("--scales") + 1].split(",")] if "--scales" in sys.argv else [0.0, 1.0]
a, z = (int(x) for x in (sys.argv[sys.argv.index("--seeds") + 1] if "--seeds" in sys.argv else "9300:9324").split(":"))
BINS = [0.1, 0.3, 1.0, 3.0]

def summary(speed, runs):
    h = np.histogram(speed, [0] + BINS + [1e9])[0] / max(len(speed), 1)
    r = np.array(runs) if runs else np.zeros(1)
    return {"near_still": round(float(h[0]), 3), "share_bins_<0.1_0.3_1_3_>3": h.round(3).tolist(), "still_runs_n": len(runs),
            "still_run_frames_p50_p90_max": [round(float(np.percentile(r, 50)), 1), round(float(np.percentile(r, 90)), 1), round(float(r.max()), 1)]}

def runs_of(still, unit=1.0):
    out, n = [], 0
    for s in still:
        if s: n += 1
        elif n: out.append(n * unit); n = 0
    if n: out.append(n * unit)
    return out

res = {"real": {}, "sim": {}}
st = pd.read_parquet(ST)
for split, eps in (("train", (0, 2, 6, 7, 9)), ("dev", (1, 3, 10))):
    sp, rn = [], []
    for ep in eps:
        rows = st[st.episode_index == ep].sort_values("timestamp"); S = np.stack(rows["observation.state"].to_numpy()).astype(float); ts = rows.timestamp.to_numpy(float)
        v = np.abs(np.diff(S[:, :6], axis=0)).max(1) / (np.diff(ts) * 30); sp += v.tolist(); rn += runs_of(v < 0.1)
    res["real"][split] = summary(sp, rn); print("real", split, json.dumps(res["real"][split]), flush=True)
robot = b.make_robot(a); m = robot._sim.model; adr = m.numeric_adr[m.numeric("r2s_operator_drift").id]
for k in SCALES:
    m.numeric_data[adr] = k
    sp, rn, ph_n, ph_s, outc = [], [], collections.Counter(), collections.Counter(), collections.Counter()
    for seed in range(a, z):
        robot._automation.set_autopilot("off"); b.reset(robot, seed); robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator
        prev, t, w0, v_ep = None, 0.0, time.time(), []
        while t < 40 and time.time() - w0 < 300:
            q = robot.get_observation().joint_positions.copy()
            while demo.planner_running: time.sleep(0.0005)
            t += b.DT; ep = demo._episode
            if ep is not None and ep.outcome != "running": break
            if prev is not None:
                v = np.abs(q[:6] - prev[:6]).max() / (b.DT * 30); v_ep.append(v); ph_n[demo.phase] += 1; ph_s[demo.phase] += v < 0.1
            prev = q
        outc[getattr(ep, "outcome", None)] += 1; sp += v_ep; rn += runs_of(np.array(v_ep) < 0.1, b.DT * 30)
    tot = sum(ph_s.values())
    res["sim"][str(k)] = {**summary(sp, rn), "outcomes": dict(outc),
                          "near_still_by_phase_share_of_all": {p: round(ph_s[p] / max(tot, 1), 3) for p in sorted(ph_s, key=lambda p: -ph_s[p]) if ph_s[p]}}
    print("sim scale", k, json.dumps(res["sim"][str(k)]), flush=True)
dev = res["real"]["dev"]
score = lambda r: abs(r["near_still"] - dev["near_still"]) / max(dev["near_still"], 1e-6) + abs(r["still_run_frames_p50_p90_max"][1] - dev["still_run_frames_p50_p90_max"][1]) / max(dev["still_run_frames_p50_p90_max"][1], 1)
res["dev_choice"] = min(res["sim"], key=lambda k: score(res["sim"][k])) if res["sim"] else None
res["seeds"] = [a, z]; json.dump(res, open(OUT, "w"), indent=1); print("DEV choice: scale", res["dev_choice"], flush=True)
