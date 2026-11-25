"""Per-episode bucket layouts from the real overview camera (the reviewer's export only; r10, the r09 D1 lever).

For each allowed episode: the per-pixel temporal median of its overview frames (frames_x5; the moving arm drops out),
each bucket's real mask by colour (red, green, blue; black = dark pixels on the desk near its prior place), and its pose
(x, y, yaw) fitted by maximising the IoU of the sim bucket's segmentation (MuJoCo, the kit's fitted overview camera)
with that mask: a grid around the phone-matched pose, then a local refinement, one bucket at a time with the others
fixed, two rounds. Writes align/bucket_layouts_episodes.json (per episode: poses, IoU before/after, the frame list) and
the held-out rule (fixed and declared before any held-out comparison): the nearest allowed episode by index, ties to
the earlier. usage: fit_bucket_layouts.py FRAMES_X5 STATES_X5 [--episodes 0,1,...]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, cv2, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bucketsim_v2 as b
from physicalai_mujoco_plugin.real2sim.desk_compositor import COLORS, LABEL_BUCKET

ap = argparse.ArgumentParser(); ap.add_argument("frames"); ap.add_argument("states"); ap.add_argument("--episodes", default="0,1,2,3,6,7,9,10")
a = ap.parse_args(); F = Path(a.frames); ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
S = json.loads(Path(a.states).read_text())["states"]
robot = b.make_robot(0); b.reset(robot, 0); m, d = robot._sim.model, robot._sim.data; b.matched_state(robot)
comp = b.compositor(robot, show_objects=False)            # applies the kit's fitted overview pose and fovy to the model
cam = m.camera("overview").id; seg_r = mujoco.Renderer(m, 480, 640); seg_r.enable_segmentation_rendering()
vopt = mujoco.MjvOption(); vopt.geomgroup[:] = 0; vopt.geomgroup[0] = 1   # _render_only moves what it draws to group 0
jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]
bucket_body = {c: m.body(f"bucket_{c}").id for c in COLORS}
bucket_adr = {c: m.jnt_qposadr[m.joint(f"bucket_{c}:joint").id] for c in COLORS}
def _subtree(root):
    out = [root]
    for i in range(m.nbody):
        k = i
        while k > 0 and k != root: k = m.body_parentid[k]
        if k == root and i != root: out.append(i)
    return out
SUBTREE = {c: _subtree(bid) for c, bid in bucket_body.items()}

def set_pose(c, x, y, yaw):
    adr = bucket_adr[c]; d.qpos[adr:adr + 3] = [x, y, 0.0]; d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

def get_pose(c):
    adr = bucket_adr[c]; q = d.qpos[adr + 3:adr + 7]; return float(d.qpos[adr]), float(d.qpos[adr + 1]), float(2 * np.arctan2(q[3], q[0]))

BUCKET_MESH = {c: [g for g in np.flatnonzero(np.isin(m.geom_bodyid, SUBTREE[c])) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH] for c in COLORS}
ROBOT_GEOMS = np.flatnonzero(np.isin(m.geom_bodyid, [i for i in range(m.nbody) if m.body(i).name in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_left", "gripper_right", "gripper_end")]))
ROBOT_GEOMS = [g for g in ROBOT_GEOMS if m.geom_group[g] in (0, 1, 2) or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]

def _render_only(geoms):
    """Segmentation with only *geoms* drawn (every other geom transparent, so nothing in the splat proxies occludes)."""
    keep = m.geom_rgba.copy(); grp = m.geom_group.copy()
    m.geom_rgba[:, 3] = 0.0; m.geom_rgba[geoms, 3] = 1.0; m.geom_group[geoms] = 0
    try:
        mujoco.mj_forward(m, d); seg_r.update_scene(d, camera=cam, scene_option=vopt); s = seg_r.render()
    finally:
        m.geom_rgba[:] = keep; m.geom_group[:] = grp
    return s[..., 0], s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)

def sim_masks(robot=False):
    objid, isgeom = _render_only([g for c in COLORS for g in BUCKET_MESH[c]])
    out = {c: isgeom & np.isin(objid, BUCKET_MESH[c]) for c in COLORS}
    if robot:
        objid, isgeom = _render_only(ROBOT_GEOMS); out["robot"] = isgeom & np.isin(objid, ROBOT_GEOMS)
    return out

def real_masks(img, prior):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); h, s_, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    near = {c: cv2.dilate(prior[c].astype(np.uint8), np.ones((61, 61), np.uint8)) > 0 for c in COLORS}
    masks = {"red": ((h < 10) | (h > 168)) & (s_ > 110) & (v > 60), "green": (h >= 60) & (h <= 95) & (s_ > 70) & (v > 60),
             "blue": (h >= 100) & (h <= 130) & (s_ > 110) & (v > 50)}
    colored = masks["red"] | masks["green"] | masks["blue"]
    masks["black"] = (v < 70) & ~colored
    out = {}
    for c in COLORS:
        mk = masks[c] & near[c]
        mk = cv2.morphologyEx(mk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mk)   # the largest blob near the prior
        out[c] = (lab == (1 + int(np.argmax(st[1:, 4])))) if n > 1 else mk.astype(bool)
    return out

def iou(x, y):
    u = (x | y).sum(); return float((x & y).sum() / u) if u else 0.0

result = {"source": "review/export/robotcam_allowed (frames_x5 overview, states)", "method": __doc__.split("\n\n")[0], "episodes": {}}
base = {c: get_pose(c) for c in COLORS}
for ep in [int(e) for e in a.episodes.split(",")]:
    files = sorted(F.glob(f"overview_e{ep:03d}_t*.png"))
    pick = [files[i] for i in np.linspace(0, len(files) - 1, min(25, len(files))).astype(int)]
    med = np.median(np.stack([cv2.imread(str(f)) for f in pick]), axis=0).astype(np.uint8)
    # the arm's typical pose for this episode (its median state) only to exclude the arm region from the masks
    for c, p in base.items(): set_pose(c, *p)
    keys = [f"{ep}/{int(f.stem.split('_t')[1])}" for f in pick]
    q = np.median(np.array([S[k] for k in keys if k in S]), axis=0)
    for adr, v in zip(jadr, q): d.qpos[adr] = v
    prior = sim_masks(robot=True); real = real_masks(med, prior)
    if os.environ.get("LAYOUT_DEBUG"):
        dbg = med.copy(); pal = {"black": (255, 255, 0), "green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 0, 0)}
        for c in COLORS:
            cs, _ = cv2.findContours(prior[c].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(dbg, cs, -1, pal[c], 1)
            cr, _ = cv2.findContours(real[c].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(dbg, cr, -1, (255, 0, 255), 2)
        cv2.imwrite(os.environ["LAYOUT_DEBUG"], dbg)
    arm = cv2.dilate(prior["robot"].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    before = {c: iou(prior[c] & ~arm, real[c] & ~arm) for c in COLORS}
    for _round in range(2):
        for c in COLORS:
            x0, y0, w0 = get_pose(c); best = (iou(sim_masks()[c] & ~arm, real[c] & ~arm), x0, y0, w0)
            for step_xy, step_w, n in ((0.01, np.radians(3), 3), (0.003, np.radians(1), 2)):
                cx, cy, cw = best[1:]
                for dx in np.arange(-n, n + 1) * step_xy:
                    for dy in np.arange(-n, n + 1) * step_xy:
                        for dw in np.arange(-1, 2) * step_w:
                            set_pose(c, cx + dx, cy + dy, cw + dw); sc = iou(sim_masks()[c] & ~arm, real[c] & ~arm)
                            if sc > best[0]: best = (sc, cx + dx, cy + dy, cw + dw)
            set_pose(c, *best[1:])
    after_masks = sim_masks()
    result["episodes"][str(ep)] = {
        "poses": {c: dict(zip(("x", "y", "yaw"), [round(v, 5) for v in get_pose(c)])) for c in COLORS},
        "moved_mm_deg": {c: [round((get_pose(c)[0] - base[c][0]) * 1000, 1), round((get_pose(c)[1] - base[c][1]) * 1000, 1), round(float(np.degrees(get_pose(c)[2] - base[c][2])), 1)] for c in COLORS},
        "iou_phone_layout": {c: round(before[c], 3) for c in COLORS}, "iou_fitted": {c: round(iou(after_masks[c] & ~arm, real[c] & ~arm), 3) for c in COLORS},
        "frame_list": [f.name for f in pick]}
    print(ep, {c: (result["episodes"][str(ep)]["iou_phone_layout"][c], result["episodes"][str(ep)]["iou_fitted"][c], result["episodes"][str(ep)]["moved_mm_deg"][c]) for c in COLORS}, flush=True)
    cv2.imwrite(str(ROOT / f"sim/r10/layout_e{ep:03d}.jpg"), np.hstack([med, cv2.cvtColor(np.stack([sum((after_masks[c] * (i + 1) * 60) for i, c in enumerate(COLORS))] * 3, -1).astype(np.uint8)[..., 0], cv2.COLOR_GRAY2BGR)]))
allowed = sorted(int(e) for e in result["episodes"])
def nearest(e):   # the rule: the nearest allowed episode by index, ties to the earlier
    return min(allowed, key=lambda k: (abs(k - e), k))
result["heldout_rule"] = "a held-out or unknown episode uses the layout of the nearest allowed episode by index, ties to the earlier one (coordinator note, NOTES; fixed before any held-out comparison)"
result["rule_table"] = {str(e): nearest(e) for e in range(0, 11)}
(ROOT / "align").mkdir(exist_ok=True); json.dump(result, open(ROOT / "align/bucket_layouts_episodes.json", "w"), indent=1)
print("rule", result["rule_table"])
