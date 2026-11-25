"""CAD surface samples from the upstream bucket_sort scene (reBot B601): the bucket mesh and the arm at joints q."""
import numpy as np, mujoco
from physicalai_mujoco_plugin.robot import MuJoCoRobot

_cache = {}

def model():
    if "m" not in _cache:
        sim = MuJoCoRobot("rebot_b601", scene="bucket_sort", cameras=[], http_port=0, seed=0)
        sim.connect(); sim._automation.set_autopilot("off")
        m = sim._sim.model; d = mujoco.MjData(m); d.qpos[:] = sim._sim.data.qpos
        _cache.update(m=m, d=d, sim=sim)
    return _cache["m"], _cache["d"]

def _tri_samples(v, f, n, rng):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n); s = np.sqrt(r1)
    return (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]

def mesh_points(mesh_name, n=4000, seed=0):
    m, _ = model(); mid = m.mesh(mesh_name).id
    v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    v = m.mesh_vert[v0:v0 + nv].astype(float); f = m.mesh_face[f0:f0 + nf]
    return _tri_samples(v, f, n, np.random.default_rng(seed))

def geom_local_points(per_geom=600, seed=0, bodies=None):
    """{geom_id: (body_id, points in the geom frame)} for the arm's visual meshes."""
    m, _ = model(); rng = np.random.default_rng(seed); out = {}
    arm = {m.body(n).id for n in (bodies or ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right"])}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in arm and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] != 3:
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            out[g] = (m.geom_bodyid[g], _tri_samples(m.mesh_vert[v0:v0 + nv].astype(float), m.mesh_face[f0:f0 + nf], per_geom, rng))
    return out

def arm_points(q, local):
    """World points of the arm at arm joint angles q (radians, 6) plus gripper opening q[6] (m), base at the origin."""
    m, d = model()
    names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    for i, n in enumerate(names): d.qpos[m.jnt_qposadr[m.joint(n).id]] = q[i]
    if len(q) > 6:
        for n in ("joint_left", "joint_right"):
            try: d.qpos[m.jnt_qposadr[m.joint(n).id]] = q[6]
            except KeyError: pass
    mujoco.mj_kinematics(m, d)
    pts = [d.geom_xpos[g] + p @ d.geom_xmat[g].reshape(3, 3).T for g, (_, p) in local.items()]
    return np.vstack(pts)

def bucket_body_points(n=4000, seed=0):
    """Bucket surface in its body frame (origin at the footprint centre on the table, z up, open end at +y)."""
    m, _ = model(); g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body("bucket_black").id and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH][0]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return mesh_points("bucket", n, seed) @ R.reshape(3, 3).T + m.geom_pos[g]
