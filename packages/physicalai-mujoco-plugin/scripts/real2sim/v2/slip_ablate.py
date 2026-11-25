"""r15 (D4): the record-mix slips (7418, 7425) with multi-contact on / off and the drift on / off. usage: python slip_ablate.py"""
import sys, json, mujoco
sys.path.insert(0, '/Users/xiangxi2/physicalai-wt-bucket-v2/packages/physicalai-mujoco-plugin/scripts/real2sim/v2')
import bucketsim_v2 as b
if __name__ == "__main__":
    robot = b.make_robot(7418, "record"); m = robot._sim.model; adr = m.numeric_adr[m.numeric("r2s_operator_drift").id]
    MC = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD); base = m.opt.disableflags & ~MC
    for seed in (7418, 7425):
        for mc_off in (True, False):
            for drift in (1.0, 0.0):
                m.opt.disableflags = base | (MC if mc_off else 0); m.numeric_data[adr] = drift
                robot._automation.set_autopilot("off"); b.reset(robot, seed); r = b.run_episode(robot, seed)
                g = [(o, x["opening_mm"], x["t"], x["slide_after_0p2s_mm"]) for o, xs in (r.get("grasps") or {}).items() for x in xs]
                print(seed, "multiccd", "off" if mc_off else "on", "drift", drift, "success", r["success"], "max pen", round(r["max_penetration_m"] * 1000, 2), "grasps", g, flush=True)
