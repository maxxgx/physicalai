"""The per-episode bucket-pose fitter of fit_bucket_layouts.py (r10) as a reusable function, unchanged in method, for
FIT_LAYOUT (r12, U-5: the reviewer fits held-out layouts with the builder's fitter).

Given the per-pixel temporal median of 25 overview frames and the arm's median joint angles over those frames: each
bucket's real mask by colour (red, green, blue; black = dark pixels near its prior place), the sim arm's silhouette at
the median joints (dilated 15 px) excluded from both masks, and each bucket's pose (x, y, yaw) fitted by maximising the
IoU of the sim bucket's segmentation (the kit's fitted overview camera) with the real mask: a grid around the
phone-matched pose (+-3 x 10 mm, +-3 deg), then a local one (+-2 x 3 mm, +-1 deg), one bucket at a time with the
others fixed, two rounds. Frames are never skipped: the moving arm drops out of the median, the still arm is masked."""
from __future__ import annotations

import re
import numpy as np, cv2, mujoco
from physicalai_mujoco_plugin.real2sim.desk_compositor import COLORS

PROXY = re.compile(r"_plan\d+$|_pc\d+_\d+$")   # r14 planning-only proxies: neither drawn nor colliding
FRAMES = 25          # overview frames per fit, spread evenly by index over the candidates
STEP = 5             # candidates: every 5th decoded frame (the frames_x5 rule)
PAL = {"black": (255, 255, 0), "green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 0, 0)}


def pick(candidates: list) -> list:
    """The fixed frame rule: FRAMES of the candidates, spread evenly by index (np.linspace, truncated)."""
    return [candidates[i] for i in np.linspace(0, len(candidates) - 1, min(FRAMES, len(candidates))).astype(int)]


class LayoutFitter:
    def __init__(self, robot):
        """*robot*: a bucketsim_v2 robot after reset, matched_state and compositor (which applies the kit's overview pose)."""
        m, d = robot._sim.model, robot._sim.data; self.m, self.d = m, d
        self.cam = m.camera("overview").id; self.seg_r = mujoco.Renderer(m, 480, 640); self.seg_r.enable_segmentation_rendering()
        self.vopt = mujoco.MjvOption(); self.vopt.geomgroup[:] = 0; self.vopt.geomgroup[0] = 1
        self.jadr = [m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 7)]
        bucket_body = {c: m.body(f"bucket_{c}").id for c in COLORS}
        self.bucket_adr = {c: m.jnt_qposadr[m.joint(f"bucket_{c}:joint").id] for c in COLORS}
        def subtree(root):
            out = [root]
            for i in range(m.nbody):
                k = i
                while k > 0 and k != root: k = m.body_parentid[k]
                if k == root and i != root: out.append(i)
            return out
        sub = {c: subtree(bid) for c, bid in bucket_body.items()}
        # the drawn bucket and its drawn contents (r13: the bucket's and the contents' collision are mesh pieces too, and
        # reach up to ~3 mm past the drawing; r12's silhouette also held the contents' collision hulls, inside the drawing)
        self.bucket_mesh = {c: [g for g in np.flatnonzero(np.isin(m.geom_bodyid, sub[c])) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                                and not (m.geom_contype[g] or m.geom_conaffinity[g]) and not PROXY.search(m.geom(g).name or "")] for c in COLORS}
        rg = np.flatnonzero(np.isin(m.geom_bodyid, [i for i in range(m.nbody) if m.body(i).name in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_left", "gripper_right", "gripper_end")]))
        self.robot_geoms = [g for g in rg if m.geom_group[g] in (0, 1, 2) or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.base = {c: self.get_pose(c) for c in COLORS}   # the phone-matched poses: the grid's start

    def set_pose(self, c, x, y, yaw):
        adr = self.bucket_adr[c]; d = self.d
        d.qpos[adr:adr + 3] = [x, y, 0.0]; d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def get_pose(self, c):
        adr = self.bucket_adr[c]; q = self.d.qpos[adr + 3:adr + 7]
        return float(self.d.qpos[adr]), float(self.d.qpos[adr + 1]), float(2 * np.arctan2(q[3], q[0]))

    def _render_only(self, geoms):
        m, d = self.m, self.d; keep = m.geom_rgba.copy(); grp = m.geom_group.copy()
        m.geom_rgba[:, 3] = 0.0; m.geom_rgba[geoms, 3] = 1.0; m.geom_group[geoms] = 0
        try:
            mujoco.mj_forward(m, d); self.seg_r.update_scene(d, camera=self.cam, scene_option=self.vopt); s = self.seg_r.render()
        finally:
            m.geom_rgba[:] = keep; m.geom_group[:] = grp
        return s[..., 0], s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)

    def sim_masks(self, robot=False):
        objid, isgeom = self._render_only([g for c in COLORS for g in self.bucket_mesh[c]])
        out = {c: isgeom & np.isin(objid, self.bucket_mesh[c]) for c in COLORS}
        if robot:
            objid, isgeom = self._render_only(self.robot_geoms); out["robot"] = isgeom & np.isin(objid, self.robot_geoms)
        return out

    @staticmethod
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
            n, lab, st, _ = cv2.connectedComponentsWithStats(mk)
            out[c] = (lab == (1 + int(np.argmax(st[1:, 4])))) if n > 1 else mk.astype(bool)
        return out

    @staticmethod
    def iou(x, y):
        u = (x | y).sum(); return float((x & y).sum() / u) if u else 0.0

    def fit(self, frames_bgr: list, q_median) -> tuple[dict, np.ndarray]:
        """Fit the four bucket poses to *frames_bgr* (the picked overview frames, BGR uint8) with the arm at *q_median*
        (joint1..6, rad). Returns the result dict and an overlay image (the median frame; fitted sim outlines per bucket
        colour, the real masks in magenta, the excluded arm region hatched in grey)."""
        med = np.median(np.stack(frames_bgr), axis=0).astype(np.uint8)
        for c, p in self.base.items(): self.set_pose(c, *p)
        for adr, v in zip(self.jadr, q_median): self.d.qpos[adr] = v
        prior = self.sim_masks(robot=True); real = self.real_masks(med, prior)
        arm = cv2.dilate(prior["robot"].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        iou = self.iou
        before = {c: iou(prior[c] & ~arm, real[c] & ~arm) for c in COLORS}
        for _round in range(2):
            for c in COLORS:
                x0, y0, w0 = self.get_pose(c); best = (iou(self.sim_masks()[c] & ~arm, real[c] & ~arm), x0, y0, w0)
                for step_xy, step_w, n in ((0.01, np.radians(3), 3), (0.003, np.radians(1), 2)):
                    cx, cy, cw = best[1:]
                    for dx in np.arange(-n, n + 1) * step_xy:
                        for dy in np.arange(-n, n + 1) * step_xy:
                            for dw in np.arange(-1, 2) * step_w:
                                self.set_pose(c, cx + dx, cy + dy, cw + dw); sc = iou(self.sim_masks()[c] & ~arm, real[c] & ~arm)
                                if sc > best[0]: best = (sc, cx + dx, cy + dy, cw + dw)
                self.set_pose(c, *best[1:])
        after = self.sim_masks(); base = self.base; gp = self.get_pose
        res = {"poses": {c: dict(zip(("x", "y", "yaw"), [round(v, 5) for v in gp(c)])) for c in COLORS},
               "moved_mm_deg": {c: [round((gp(c)[0] - base[c][0]) * 1000, 1), round((gp(c)[1] - base[c][1]) * 1000, 1), round(float(np.degrees(gp(c)[2] - base[c][2])), 1)] for c in COLORS},
               "iou_phone_layout": {c: round(before[c], 3) for c in COLORS},
               "iou_fitted": {c: round(iou(after[c] & ~arm, real[c] & ~arm), 3) for c in COLORS}}
        ov = med.copy()
        ov[arm & ((np.indices(arm.shape).sum(0) // 4) % 2 == 0)] = (128, 128, 128)
        for c in COLORS:
            cr, _ = cv2.findContours(real[c].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cr, -1, (255, 0, 255), 2)
            cs, _ = cv2.findContours(after[c].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cs, -1, PAL[c], 1)
            ys, xs = np.nonzero(after[c])
            if len(xs): cv2.putText(ov, f"{c} IoU {res['iou_fitted'][c]:.2f}", (int(xs.min()), max(12, int(ys.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PAL[c], 1)
        return res, np.hstack([med, ov])
