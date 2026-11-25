"""r18 item 4 (D2, coordinator): does FIT_LAYOUT place buckets partly outside the overview view on the allowed episodes?
Per allowed episode (the kit's FIT_LAYOUT layouts: TRAIN 0, 2, 6, 7, 9; DEV 1, 3, 10) and bucket: its drawn mesh's
vertices projected through the sim's overview camera (the kit's pose, 640x480); the share outside the image and the
mesh's image bounding box. Geometry only (no frames are read). usage: layout_view_check.py OUT_JSON"""
import json, os, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
if __name__ == "__main__":
    import bucketsim_v2 as b
    robot = b.make_robot(0); m, d = robot._sim.model, robot._sim.data
    KIT = json.load(open(os.environ["KIT"])); EPS = KIT["bucket_layouts_episodes"]["episodes"]
    cam = m.camera("overview").id; W, H = 640, 480; f = (H / 2) / np.tan(np.radians(m.cam_fovy[cam]) / 2)
    adr = {c: m.jnt_qposadr[m.joint(f"bucket_{c}:joint").id] for c in ("black", "green", "red", "blue")}
    meshes = {c: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"bucket_{c}").id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                  and m.geom_contype[g] == 0] for c in adr}
    res = {"method": __doc__.split("usage")[0].strip(), "fovy_deg": float(m.cam_fovy[cam]), "episodes": {}}
    for ep, rec in sorted(EPS.items(), key=lambda kv: int(kv[0])):
        if int(ep) in (4, 5, 8): continue   # held-out episodes are never touched
        for c, p in rec["poses"].items():
            a = adr[c]; d.qpos[a:a + 3] = [p["x"], p["y"], 0.0]; d.qpos[a + 3:a + 7] = [np.cos(p["yaw"] / 2), 0, 0, np.sin(p["yaw"] / 2)]
        mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
        R, C = d.cam_xmat[cam].reshape(3, 3), d.cam_xpos[cam]; out = {}
        for c, gs in meshes.items():
            pts = []
            for g in gs:
                mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
            P = (np.vstack(pts) - C) @ R; z = -P[:, 2]
            u = W / 2 + f * P[:, 0] / z; vv = H / 2 - f * P[:, 1] / z
            outside = (u < 0) | (u >= W) | (vv < 0) | (vv >= H) | (z <= 0)
            out[c] = {"outside_pct": round(100 * float(outside.mean()), 1), "u": [round(float(u.min()), 1), round(float(u.max()), 1)],
                      "v": [round(float(vv.min()), 1), round(float(vv.max()), 1)], "iou_fitted": rec.get("iou_fitted", {}).get(c)}
        res["episodes"][ep] = out
        print(ep, {c: (o["outside_pct"], o["iou_fitted"]) for c, o in out.items()}, flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
