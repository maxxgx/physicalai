"""Why the demonstrator cannot reach a grasp: for each seed (episodes.py's setup, same layouts) print every grasp pose no
allowed lean reaches, the lean the IK needs under wider limits and with the yaw turned by half a turn, and the
yaw candidates offered. usage: lean_probe.py SEED [...]"""
import sys, time, numpy as np
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin import bucket_sort_demo as bd
cls = [c for c in vars(bd).values() if isinstance(c, type) and "tilt_at" in vars(c)][0]
orig_tilt, orig_yc = cls.tilt_at, cls.yaw_candidates; done = set()
def tilt_at(self, pos, yaw, *, grasp=False):
    was = self.unreached; r = orig_tilt(self, pos, yaw, grasp=grasp)
    key = (round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(pos[2]), 2), round(float(yaw), 2), grasp)
    if self.unreached and not was and key not in done:
        done.add(key)
        wide = [(lim, None if (f := self.ik.least_tilt(self.ik.home, pos, yaw, np.radians(lim))) is None else round(float(np.degrees(f[0])))) for lim in (25, 40)]
        f = self.ik.least_tilt(self.ik.home, pos, yaw + np.pi, np.radians(15))
        print(f"   unreached {'GRASP' if grasp else 'other'} pos {np.round(pos, 3).tolist()} yaw {np.degrees(yaw):.0f}: wider limits {wide}, yaw+180 {None if f is None else round(float(np.degrees(f[0])))}", flush=True)
    return r
def yc(self, axis_yaw, current, xy, *, period=np.pi):
    r = orig_yc(self, axis_yaw, current, xy, period=period); print(f"   candidates axis {np.degrees(axis_yaw):.0f} period {np.degrees(period):.0f} -> {[round(float(np.degrees(y))) for y in r]}", flush=True); return r
cls.tilt_at, cls.yaw_candidates = tilt_at, yc
for seed in map(int, sys.argv[1:]):
    robot = MuJoCoRobot("rebot_b601", scene="bucket_sort_desk_v2", cameras=[], http_port=0, seed=seed); robot.connect()
    m, d = robot._sim.model, robot._sim.data; task = robot._sim.on_reset
    robot._automation.set_autopilot("off"); robot._set_seed(seed); robot._reset(); robot._set_seed(None)
    ins = task.instruction; lay = task.layout; p = d.xpos[m.body(ins.object).id]
    print(f"seed {seed}: {ins.object} -> {ins.bucket}, pile {lay.pile}, {len(lay.objects)} objects, object at {np.round(p, 3).tolist()}", flush=True)
    robot._automation.set_autopilot("drive"); demo = robot._automation.autopilot.demonstrator; demo._rng = np.random.default_rng(seed)
    t0 = time.time(); ticks = 0
    while time.time() - t0 < 120 and ticks < 2250:
        robot.get_observation(); ticks += 0 if demo.planning else 1
        if demo.status()["outcome"] in {"success", "failed", "stalled"}: break
    print("  ->", demo.status()["outcome"], demo.status()["failure"], flush=True); robot.disconnect(); done.clear()
