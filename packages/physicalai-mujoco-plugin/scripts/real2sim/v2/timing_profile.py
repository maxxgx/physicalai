"""r18 item 1 (D7, coordinator): the operator's timing, real against sim, measured the same way on 30 fps readings.

Per episode, the readings (degrees; the six arm joints and the gripper, -270 open .. 0 closed). Profile:
  still-run histogram   a frame is still when every channel changes < 0.1 deg from the previous one; % of frames in runs
                        of 1, 2-3, 4-9, 10-29 frames, 1-3 s, 3-7 s, >= 7 s;
  speed-bin shares      the arm's largest joint change per frame: < 0.1, 0.1-0.3, 0.3-1, >= 1 deg; % of frames;
  events (s)            close: from the gripper leaving open (> -250) to its plateau (|change| < 0.5 deg for 3 frames);
                        hold after close: plateau to the arm moving (> 0.3 deg per frame);
                        open: from the plateau's end (the gripper opening) to open (<= -250);
                        pause at release: the opening's start to the arm moving.
Real: the allowed export (TRAIN 0, 2, 6, 7, 9 / DEV 1, 3, 10). Sim: in-process episodes (bucketsim_v2), the readings
sampled every control tick and resampled to 30 fps; DEMO_OVERRIDES='{"field": value or ["x", factor]}' changes every
episode's final DemoParams (a value, ["x", factor] on it, ["u", lo, hi] or ["c", choices...] from a stream of its own); TIMING=0/18
forces the scene's r2s_operator_timing (r18: bucket_sort_demo.operator_timing).
usage: timing_profile.py real EXPORT OUT_JSON | timing_profile.py sim START COUNT VARIANT OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np

BINS = [(1, 1), (2, 3), (4, 9), (10, 29), (30, 89), (90, 209), (210, 10 ** 9)]
LAB = ["1", "2-3", "4-9", "10-29", "1-3 s", "3-7 s", ">=7 s"]
SB = [(0, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 1e9)]; SBL = ["<0.1", "0.1-0.3", "0.3-1", ">=1"]

def profile(episodes):
    tot, runs, sp = 0, np.zeros(len(BINS)), np.zeros(len(SB))
    ev = {"close_s": [], "hold_after_close_s": [], "open_s": [], "pause_at_release_s": []}
    for S in episodes:
        if len(S) < 10: continue
        dq = np.abs(np.diff(S, axis=0)); still = np.r_[False, dq.max(1) < 0.1]; arm = np.r_[0.0, dq[:, :6].max(1)]
        tot += len(S); st = None
        for i, s in enumerate(np.r_[still, False]):
            if s and st is None: st = i
            if not s and st is not None:
                n = i - st; runs[[k for k, (a, b) in enumerate(BINS) if a <= n <= b][0]] += n; st = None
        for k, (a, b) in enumerate(SB): sp[k] += ((arm >= a) & (arm < b)).sum()
        g = S[:, 6]; dg = np.r_[0.0, np.diff(g)]; i = 1
        while i < len(g):
            if g[i - 1] <= -250 < g[i]:   # closing starts
                j = i
                while j + 3 < len(g) and not (np.abs(dg[j:j + 3]) < 0.5).all(): j += 1
                if j + 3 >= len(g): break
                ev["close_s"].append((j - i) / 30); k = j
                while k < len(g) and arm[k] <= 0.3: k += 1
                ev["hold_after_close_s"].append((k - j) / 30)
                o = j   # the plateau's end: the gripper opening toward open
                while o < len(g) and not (dg[o] < -0.5 and g[o] < g[j] - 3): o += 1
                if o >= len(g): break
                e = o
                while e < len(g) and g[e] > -250: e += 1
                if e < len(g): ev["open_s"].append((e - o) / 30)
                m = o
                while m < len(g) and arm[m] <= 0.3: m += 1
                ev["pause_at_release_s"].append((m - o) / 30); i = max(e, o + 1)
            i += 1
    q = lambda v: [round(float(x), 2) for x in np.percentile(v, [10, 50, 90])] if v else None
    return {"frames": int(tot), "near_still_pct": round(100 * runs.sum() / max(tot, 1), 2),
            "still_runs_pct": dict(zip(LAB, (100 * runs / max(tot, 1)).round(2).tolist())),
            "speed_bins_pct": dict(zip(SBL, (100 * sp / max(tot, 1)).round(2).tolist())),
            "events_p10_p50_p90": {k: q(v) for k, v in ev.items()}, "events_n": {k: len(v) for k, v in ev.items()}}

def distance(a, b):
    """Histogram distance used for the choice: L1 between the still-run and speed-bin shares (percentage points)."""
    return round(sum(abs(a["still_runs_pct"][k] - b["still_runs_pct"][k]) for k in LAB) + sum(abs(a["speed_bins_pct"][k] - b["speed_bins_pct"][k]) for k in SBL), 2)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "real":
        import pandas as pd
        df = pd.read_parquet(Path(sys.argv[2]) / "states_allowed.parquet").sort_values(["episode_index", "timestamp"])
        eps = {int(e): np.stack(g["observation.state"].values).astype(float) for e, g in df.groupby("episode_index")}
        res = {"train": profile([eps[e] for e in (0, 2, 6, 7, 9)]), "dev": profile([eps[e] for e in (1, 3, 10)])}
        json.dump(res, open(sys.argv[3], "w"), indent=1); print(json.dumps(res, indent=1)); sys.exit(0)
    start, count, variant, OUT = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import bucketsim_v2 as b
    from dataclasses import replace
    from physicalai_mujoco_plugin import bucket_sort_demo as bsd
    OV = json.loads(os.environ.get("DEMO_OVERRIDES", "{}"))
    for _k, _v in json.loads(os.environ.get("TIMING_SET", "{}")).items():   # r18 ablations of OPERATOR_TIMING_R18
        bsd.OPERATOR_TIMING_R18[_k] = tuple(_v) if isinstance(_v, list) else _v
    if os.environ.get("TIMING") is not None:   # r18: force the scene's r2s_operator_timing (0: off, 18: operator_timing)
        bsd.BucketSortDemonstrator._operator_timing = lambda self: int(os.environ["TIMING"])
    if OV:   # r18: applied to the episode's final parameters (after the brisk, drift and timing draws), own stream
        orig_start = bsd.BucketSortDemonstrator._start_episode
        def _start_episode(self, data):
            orig_start(self, data); ep = self._episode; p = ep.params
            rng = np.random.default_rng([int(getattr(self.task, "episode_seed", 0) or 0), 99])
            def val(f, v):   # a value, ["x", factor], ["u", lo, hi] or ["c", choices...]
                if not isinstance(v, list): return v
                if v[0] == "c": return float(v[1:][int(rng.integers(len(v) - 1))])
                return getattr(p, f) * v[1] if v[0] == "x" else float(rng.uniform(v[1], v[2]))
            ep.params = replace(p, **{f: val(f, v) for f, v in OV.items()})
        bsd.BucketSortDemonstrator._start_episode = _start_episode
    robot = b.make_robot(start, variant); ch = robot._sim.channels[0]; episodes = []; phases = []
    for seed in range(start, start + count):
        robot._automation.set_autopilot("off"); b.reset(robot, seed); robot._automation.set_autopilot("drive")
        demo = robot._automation.autopilot.demonstrator; n0 = robot._http_status()["episode"]["episode_count"]; t, T, R, PH = 0.0, [], [], []
        while t < 60:
            b.tick(robot, demo); t += b.DT
            if robot._http_status()["episode"]["episode_count"] != n0 or demo._episode is None or demo._episode.outcome != "running": break
            T.append(t); R.append(np.asarray(ch.read_positions(), float)); PH.append(demo._phase)
        if len(T) < 10: continue
        T, R = np.array(T), np.array(R); tt = np.arange(T[0], T[-1], 1 / 30)
        episodes.append(np.stack([np.interp(tt, T, R[:, j]) for j in range(R.shape[1])], 1))
        phases.append([PH[min(int(np.searchsorted(T, x)), len(PH) - 1)] for x in tt])
    res = {"seeds": [start, start + count - 1], "variant": variant, "overrides": OV, "timing": os.environ.get("TIMING"), "timing_set": json.loads(os.environ.get("TIMING_SET", "{}")), "sim": profile(episodes)}
    # where the short still runs (4-29 frames) start: the phase at each run's first frame (share of all frames)
    from collections import Counter
    short, tot = Counter(), 0
    for S, P in zip(episodes, phases):
        still = np.r_[False, np.abs(np.diff(S, axis=0)).max(1) < 0.1]; tot += len(S); st = None
        for i, x in enumerate(np.r_[still, False]):
            if x and st is None: st = i
            if not x and st is not None:
                if 4 <= i - st <= 29: short[P[st]] += i - st
                st = None
    res["short_runs_by_phase_pct"] = {k: round(100 * v / tot, 2) for k, v in short.most_common()}
    slow = Counter()   # r18: the arm's < 0.1 deg/frame frames by phase (share of all frames)
    for S, P in zip(episodes, phases):
        a = np.r_[0.0, np.abs(np.diff(S[:, :6], axis=0)).max(1)]
        for i in np.flatnonzero(a < 0.1): slow[P[i]] += 1
    res["slow_by_phase_pct"] = {k: round(100 * v / tot, 2) for k, v in slow.most_common()}
    print("slow (<0.1) by phase", res["slow_by_phase_pct"])
    print("short runs by phase", res["short_runs_by_phase_pct"])
    if os.environ.get("DUMP_NPZ"): np.savez_compressed(os.environ["DUMP_NPZ"], *episodes)   # r18: the 30 fps readings
    json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res["sim"]))
