"""r15 R6 probe (usage: REAL2SIM_ROOT=... KIT=... python r6_probe.py): the same seed twice in-process, paced to real time, the planner in its own process (as live)."""
import sys, time, numpy as np
sys.path.insert(0, '/Users/xiangxi2/physicalai-wt-bucket-v2/packages/physicalai-mujoco-plugin/scripts/real2sim/v2')
import bucketsim_v2 as b
if __name__ == "__main__":
    robot = b.make_robot(9400); runs = []
    for rep in range(2):
        for seed in (9400, 9401):
            robot._automation.set_autopilot("off"); b.reset(robot, seed); robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator
            log = []; t0 = time.perf_counter(); k = 0
            while time.perf_counter() - t0 < 40:
                robot.get_observation(); k += 1
                ep = demo._episode; log.append((round(float(robot._sim.data.time), 4), demo.phase, robot._sim.data.qpos[:7].copy()))
                if ep is not None and ep.outcome != "running": break
                lag = t0 + k * b.DT - time.perf_counter()
                if lag > 0: time.sleep(lag)
            runs.append((rep, seed, log))
            planning = sum(1 for _, ph, _ in log if ph == "planning") * b.DT
            print(f"rep {rep} seed {seed}: ticks {len(log)}, outcome {getattr(demo._episode, 'outcome', None)}, sim s in 'planning' {planning:.2f}", flush=True)
    for seed in (9400, 9401):
        a, c = [r[2] for r in runs if r[1] == seed]
        n = min(len(a), len(c)); first = next((i for i in range(n) if np.abs(a[i][2] - c[i][2]).max() > 1e-9), None)
        print(f"seed {seed}: lengths {len(a)} vs {len(c)}; first differing tick {first} (sim t {a[first][0] if first is not None else None}, phase {a[first][1] if first is not None else None}); max |dq| {max(np.abs(a[i][2] - c[i][2]).max() for i in range(n)):.4f} rad", flush=True)
