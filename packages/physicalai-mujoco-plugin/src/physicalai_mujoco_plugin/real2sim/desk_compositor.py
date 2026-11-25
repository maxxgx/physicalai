"""Photoreal camera frames for the ``bucket_sort_desk`` scene: the phone splat of the real desk (real arm removed) with the
MuJoCo arm, wrist camera and task objects drawn over it by depth, the arm's own Gaussians riding its links near the pose
the phone saw, sim shadows darkening the splat desk, and segmentation and outlines from the sim's geometry.

It renders any camera: a pose from the phone's SfM, or the scene's ``overview`` and ``wrist`` cameras. A kit file
(``real2sim.json``, :class:`DeskKit`) names the assets. :func:`compositor_factory` plugs it into the plugin's camera
service (``physicalai-mujoco start --appearance real2sim.json``), so the streams Studio records are the composited ones.

Frames: the SfM frame (OpenCV cameras, SfM units) and the scene frame (MuJoCo world, metres, desk top at z = 0, robot base
on its plate). ``calibration.json``'s ``sfm_to_world`` maps SfM to the robot-base frame; the scene frame adds the plate."""
from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np


@dataclass(frozen=True)
class DeskKit:
    """The assets of a desk real2sim kit."""

    splat: Path
    """The scene splat (3DGS PLY, SfM frame), the real arm removed."""
    calibration: Path
    """``calibration.json``: the SfM-to-world similarity, the robot's static joints, the buckets."""
    look: Path | None = None
    """The arm's calibrated material colours."""
    robot_gs: Path | None = None
    """The real arm's Gaussians (PLY) with ``<stem>_links.json`` next to it: the link each rides on."""
    viewer_home: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    """The browser viewer's start view (camera position, look-at point; scene frame): on the phone's path."""

    @classmethod
    def from_json(cls, path: str | Path) -> DeskKit:
        """Read a kit file; relative paths are relative to it."""
        path = Path(path).expanduser().resolve(); meta = json.loads(path.read_text())
        rel = lambda key: (path.parent / meta[key]).resolve() if meta.get(key) else None  # noqa: E731
        home = meta.get("viewer_home")
        return cls(splat=rel("splat"), calibration=rel("calibration"), look=rel("look"), robot_gs=rel("robot_gs"),
                   viewer_home=(tuple(home["position"]), tuple(home["look_at"])) if home else None)


COLORS = ("black", "green", "red", "blue")
LABEL_BUCKET = {c: i + 1 for i, c in enumerate(COLORS)}
LABEL_ROBOT, LABEL_OVERVIEW = 10, 11
LABEL_OBJECT = {"resistor": 20, "battery": 21, "cpu": 22}
OUTLINE = {"robot": (255, 255, 0), "bucket": (255, 0, 255), "desk": (0, 255, 255), "overview": (255, 128, 0),
           "wrist": (0, 255, 0), "object": (255, 255, 255)}
G_DRAW, G_BRACKET, G_SEG, G_DESK, G_HIDE = 0, 1, 3, 4, 5   # geom groups used here: drawn, seg-only, shadow catcher, never


class Calibration:
    def __init__(self, path: Path) -> None:
        self.data = json.load(open(path)); g = self.data["sfm_to_world"]
        self.M, self.t, self.scale = np.array(g["rotation"]), np.array(g["translation"]), g["scale"]
        self.plate = -self.data["desk_top_z"]
        self.units_per_m = 1.0 / self.scale

    def camera_to_scene(self, world_to_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SfM OpenCV world_to_cam -> (camera centre, camera-to-world rotation with MuJoCo axes) in the scene frame."""
        R = self.M @ world_to_cam[:3, :3].T
        c = self.scale * self.M @ (-world_to_cam[:3, :3].T @ world_to_cam[:3, 3]) + self.t + [0, 0, self.plate]
        return c, R @ np.diag([1.0, -1.0, -1.0])

    def scene_to_camera(self, pos: np.ndarray, xmat: np.ndarray) -> np.ndarray:
        """A MuJoCo camera's pose in the scene -> SfM OpenCV world_to_cam."""
        R_cv = self.M.T @ (xmat @ np.diag([1.0, -1.0, -1.0]))           # camera-to-SfM rotation
        c = self.M.T @ (np.asarray(pos) - [0, 0, self.plate] - self.t) / self.scale
        T = np.eye(4); T[:3, :3] = R_cv.T; T[:3, 3] = -R_cv.T @ c
        return T


def setup_lighting(m: mujoco.MjModel) -> None:
    """The renderer's light, shared by the compositor and the colour fit (robot_look.py): one overhead shadow light like the
    ceiling panels, a dimmer headlight for shading contrast, and specular metal."""
    m.vis.headlight.ambient[:] = 0.28; m.vis.headlight.diffuse[:] = 0.42; m.vis.headlight.specular[:] = 0.15
    m.vis.quality.shadowsize = 4096
    m.light_castshadow[:] = 0; m.light_castshadow[0] = 1; m.light_pos[0] = [0.35, 0.10, 2.4]; m.light_dir[0] = [0, 0, -1]; m.light_cutoff[0] = 70
    for name, spec, shine in (("alum", 0.6, 0.7), ("black", 0.35, 0.5), ("lime", 0.25, 0.4)):
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if k >= 0: m.mat_specular[k] = spec; m.mat_shininess[k] = shine


def apply_look(m: mujoco.MjModel, path: Path) -> None:
    """The arm's material colours calibrated to the real arm in training frames (robot_look.py)."""
    for key, rgba in json.load(open(path)).items():
        if key == "geom:body:base_link":
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == m.body("base_link").id and m.geom_group[g] < 3:
                    m.geom_matid[g] = -1; m.geom_rgba[g] = rgba
        elif not key.startswith("geom:"):
            m.mat_rgba[m.material(key).id] = rgba


def bucket_pieces(cal: Calibration, means0: np.ndarray, m: mujoco.MjModel) -> dict:
    """Each bucket's Gaussians (with its contents): those inside its calibrated footprint, 1.5 cm wider, from
    2 mm above the desk to 13.5 cm. They move rigidly with the sim bucket when it leaves its calibrated pose."""
    Pw = cal.scale * means0.astype(np.float64) @ cal.M.T + cal.t + [0, 0, cal.plate]
    pieces = {}
    for c in COLORS:
        b = cal.data["buckets"][c]; cy, sy = np.cos(b["yaw"]), np.sin(b["yaw"])
        R0 = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]]); p0 = np.array([b["x"], b["y"], 0.0])
        L = (Pw - p0) @ R0
        idx = np.flatnonzero((np.abs(L[:, 0]) < 0.075 + 0.015) & (np.abs(L[:, 1]) < 0.10125 + 0.015) & (L[:, 2] > 0.002) & (L[:, 2] < 0.135))
        pieces[c] = (idx, R0, p0, m.body(f"bucket_{c}").id)
    return pieces


def moved_splat(cal: Calibration, pieces: dict, means0: np.ndarray, quats0: np.ndarray, d: mujoco.MjData):
    """The splat's centres and rotations with every moved bucket piece carried along; None when none moved."""
    from physicalai_mujoco_plugin.real2sim.splats import matrix_to_quat, quat_multiply
    means = quats = None
    for idx, R0, p0, body in pieces.values():
        Rs, ps = d.xmat[body].reshape(3, 3), d.xpos[body]
        Rd = Rs @ R0.T; pd = ps - Rd @ p0
        if np.linalg.norm(pd + Rd @ p0 - p0) < 0.001 and np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1))) < 0.2:
            continue
        if means is None: means, quats = means0.copy(), quats0.copy()
        # in the SfM frame: x_w = s M x + t' (t' = t + plate), so x -> M^T Rd M x + M^T (Rd t' + pd - t') / s
        tp = cal.t + [0, 0, cal.plate]; Rsfm = cal.M.T @ Rd @ cal.M; tsfm = cal.M.T @ (Rd @ tp + pd - tp) / cal.scale
        means[idx] = (means0[idx].astype(np.float64) @ Rsfm.T + tsfm).astype(np.float32)
        quats[idx] = quat_multiply(matrix_to_quat(Rsfm)[None], quats0[idx].astype(np.float64)).astype(np.float32)
    return means, quats


class Compositor:
    """Renders one camera of a running sim (model m, data d) over the splat."""

    SHADOW, BLUR = 0.5, 9   # the office's diffuse light casts light shadows
    RS_FADE_DEG = 25.0

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, kit: DeskKit, *,
                 width: int, height: int, show_objects: bool = True, robot_splat: bool = True) -> None:
        from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer
        self.m, self.d, self.kit, self.W, self.H = m, d, kit, width, height
        self.cal = cal = Calibration(kit.calibration)
        if kit.look is not None:
            apply_look(m, kit.look)
        base_root = m.body_rootid[m.body("base_link").id]
        objects = {m.body(n).id: n for n in LABEL_OBJECT}
        bucket_body = {m.body(f"bucket_{c}").id: c for c in COLORS}
        self.label = np.zeros(m.ngeom, np.uint8); self.part = {}; self.contents = []; self.bucket_geoms = []
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]; name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            visual = m.geom_group[g] < 3
            if m.body_rootid[b] == base_root and visual:
                self.label[g] = LABEL_ROBOT; self.part[g] = "wrist" if name.endswith("wrist_housing") else "robot"
                # the wrist bracket in a group of its own: the wrist camera, which sits against it, leaves it out
                # (with the housing; neither casts a sim shadow: next to the wrist camera theirs would fill its view)
                m.geom_group[g] = G_BRACKET if name.endswith(("wrist_mount", "wrist_housing")) else (2 if m.geom_group[g] == G_BRACKET else m.geom_group[g])
                continue
            if b in objects and visual:
                self.label[g] = LABEL_OBJECT[objects[b]]; self.part[g] = "object"
                m.geom_group[g] = G_DRAW if show_objects else G_HIDE
                continue
            if b in bucket_body and "_item" in name:
                self.contents.append(g); m.geom_group[g] = G_HIDE
                continue
            if b in bucket_body and visual and "_item" not in name:
                self.label[g] = LABEL_BUCKET[bucket_body[b]]; self.part[g] = "bucket"; m.geom_group[g] = G_SEG
                self.bucket_geoms.append(g)
            elif name in ("overview_pole", "overview_pole_upper", "overview_housing"):
                self.label[g] = LABEL_OVERVIEW; self.part[g] = "overview"; m.geom_group[g] = G_SEG
            elif name == "table_top":
                self.part[g] = "desk"; m.geom_group[g] = G_DESK
            else:
                m.geom_group[g] = G_HIDE    # collision geoms, and what the splat shows: plate, contents, floor
        # each content item: its collision geom (whose contacts say whether a reset filled it) and its visual parts
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in self.contents}
        self.items = {g: [names[n2] for n2 in names if n2.startswith(n + "_")] for n, g in names.items() if n.split("_item")[1].isdigit()}
        def opt(groups):
            o = mujoco.MjvOption(); o.geomgroup[:] = [1 if i in groups else 0 for i in range(6)]; return o
        self.o_draw, self.o_seg = opt({0, 1, 2}), opt({0, 1, 2, G_SEG}); self.o_draw_wrist = opt({0, 2})
        self.o_objects = opt({G_DRAW})   # the task objects alone (the arm is in groups 1 and 2)
        self.o_desk, self.o_bare, self.o_deskonly = opt({0, 2, G_DESK}), opt({G_DESK}), opt({G_DESK})
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width); m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        setup_lighting(m)
        self.rgb = mujoco.Renderer(m, height, width); self.dep = mujoco.Renderer(m, height, width); self.dep.enable_depth_rendering()
        self.seg = mujoco.Renderer(m, height, width); self.seg.enable_segmentation_rendering()
        self.sha = mujoco.Renderer(m, height // 2, width // 2)
        self.splat = MetalSplatRenderer(kit.splat)
        self._bucket_pieces()
        self._robot_splat(robot_splat)
        self.rig = m.body("render_camera_rig").id; self.cam = m.camera("render").id

    # --- buckets that move: their splat pieces follow the sim bodies ------------------------------------------
    def _bucket_pieces(self) -> None:
        self.means0 = self.splat.means.cpu().numpy().astype(np.float32); self.quats0 = self.splat.quats.cpu().numpy().astype(np.float32)
        self.pieces = bucket_pieces(self.cal, self.means0, self.m)

    def _moved_splat(self):
        return moved_splat(self.cal, self.pieces, self.means0, self.quats0, self.d)

    # --- the arm drawn by its own Gaussians, carried by the links ----------------------------------------------
    def _robot_splat(self, enabled: bool) -> None:
        """The real arm's Gaussians (robot_gaussians.py), each riding on its link: their poses relative to the link at the
        calibrated static joints stay fixed. Drawn over the CAD render; what the phone never saw stays CAD."""
        from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer
        path = self.kit.robot_gs; self.rs = None
        if not enabled or path is None or not path.exists():
            return
        from physicalai_mujoco_plugin.real2sim.splats import read_ply_fields
        meta = json.load(open(path.with_name(path.stem + "_links.json")))
        fields = read_ply_fields(path); m = self.m
        static = np.array([n == "base_link" for n in meta["links"]])
        # the base's Gaussians (the turret, the clamp, the cables at the base) never move: drawn at full weight always
        sub = lambda keep: {k: (v[keep] if hasattr(v, "shape") and len(v) == len(static) else v) for k, v in fields.items()}  # noqa: E731
        self.rs_static = MetalSplatRenderer(sub(static)) if static.any() else None
        self.rs = MetalSplatRenderer(sub(~static))
        meta = {**meta, "links": [n for n, s_ in zip(meta["links"], static) if not s_]}
        self.rs_means0 = self.rs.means.cpu().numpy().astype(np.float64); self.rs_quats0 = self.rs.quats.cpu().numpy().astype(np.float64)
        d0 = mujoco.MjData(m); d0.qpos[:] = self.d.qpos
        for i, v in enumerate(meta["joints_rad"][:6]): d0.qpos[m.jnt_qposadr[m.joint(f"joint{i + 1}").id]] = v
        for n in ("joint_left", "joint_right"): d0.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["joints_rad"][6]
        mujoco.mj_kinematics(m, d0)
        links = np.array([m.body(n).id for n in meta["links"]])
        self.rs_groups = [(b, np.flatnonzero(links == b), d0.xmat[b].reshape(3, 3).copy(), d0.xpos[b].copy()) for b in np.unique(links)]

    def _render_robot_splat(self, T: np.ndarray, K: np.ndarray, near_m: float, hide: tuple[int, ...] = ()):
        from physicalai_mujoco_plugin.real2sim.splats import matrix_to_quat, quat_multiply
        cal, d = self.cal, self.d; means, quats = self.rs_means0.copy(), self.rs_quats0.copy()
        tp = cal.t + [0, 0, cal.plate]; turn = 0.0
        for b, idx, R0, p0 in self.rs_groups:
            Rd = d.xmat[b].reshape(3, 3) @ R0.T; pd = d.xpos[b] - Rd @ p0
            turn = max(turn, float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))))
            Rsfm = cal.M.T @ Rd @ cal.M; tsfm = cal.M.T @ (Rd @ tp + pd - tp) / cal.scale
            means[idx] = self.rs_means0[idx] @ Rsfm.T + tsfm
            quats[idx] = quat_multiply(matrix_to_quat(Rsfm)[None], self.rs_quats0[idx])
            if b in hide:   # at the camera centre: closer than the near plane, so dropped
                means[idx] = -T[:3, :3].T @ T[:3, 3]
        # the Gaussians were fitted from views of the arm at rest: trust them near that pose only. Full weight at the
        # calibrated pose, none once any link has turned RS_FADE_DEG from it (continuous, so nothing pops)
        weight = float(np.clip(1 - turn / self.RS_FADE_DEG, 0, 1))
        if weight == 0.0:
            return None
        rr = self.rs.render(T, K, self.W, self.H, bg=(0.0, 0.0, 0.0), near=near_m * cal.units_per_m,
                            means=means.astype(np.float32), quats=quats.astype(np.float32))
        rr["alpha"] = rr["alpha"] * weight
        return rr

    # --- cameras --------------------------------------------------------------------------------------
    def _virtual(self, world_to_cam: np.ndarray, K: np.ndarray) -> int:
        """Put the scene's free render camera at an SfM pose."""
        m = self.m
        c, R = self.cal.camera_to_scene(world_to_cam); q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten())
        m.body_pos[self.rig] = c; m.body_quat[self.rig] = q; m.cam_fovy[self.cam] = np.degrees(2 * np.arctan(self.H / 2 / K[1, 1]))
        return self.cam

    def camera_K(self, cam: int) -> np.ndarray:
        f = self.H / 2 / np.tan(np.radians(self.m.cam_fovy[cam]) / 2)
        return np.array([[f, 0, self.W / 2], [0, f, self.H / 2], [0, 0, 1.0]])

    # --- rendering ------------------------------------------------------------------------------------
    def render_sfm(self, world_to_cam: np.ndarray, K: np.ndarray, *, outputs: bool = False) -> dict:
        cam = self._virtual(world_to_cam, K)
        return self._render(cam, world_to_cam, K, outputs, near_m=0.01)

    def render_camera(self, name: str, *, outputs: bool = False) -> dict:
        cam = self.m.camera(name).id; mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)
        T = self.cal.scene_to_camera(self.d.cam_xpos[cam], self.d.cam_xmat[cam].reshape(3, 3))
        # the splat holds the real overview head (and, at rest, the real wrist camera): drop what is within 6 cm
        # the wrist camera does not see its own bracket and housing (they ride on link6 with it)
        hide = (self.m.body("link6").id,) if self.m.cam_bodyid[cam] == self.m.body("link6").id else ()
        return self._render(cam, T, self.camera_K(cam), outputs, near_m=0.06, hide=hide)

    def _render(self, cam: int, T: np.ndarray, K: np.ndarray, outputs: bool, near_m: float, hide: tuple[int, ...] = ()) -> dict:
        m, d, W, H = self.m, self.d, self.W, self.H
        m.geom_group[self.contents] = G_HIDE   # a reset's fill restores their groups; the splat shows the contents
        mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
        o = self.o_draw_wrist if hide else self.o_draw
        shown = []
        if hide:   # the sim buckets, and the contents a reset filled (their contacts are on)
            shown = list(self.bucket_geoms) + [g for coll, parts in self.items.items() if m.geom_contype[coll] or m.geom_conaffinity[coll] for g in parts]
            m.geom_group[shown] = G_DRAW
        self.rgb.update_scene(d, camera=cam, scene_option=o); rgb = self.rgb.render()
        self.dep.update_scene(d, camera=cam, scene_option=o); z = self.dep.render()
        if shown:
            m.geom_group[self.bucket_geoms] = G_SEG; m.geom_group[self.contents] = G_HIDE
        means, quats = self._moved_splat()
        if hide:
            # the wrist camera: the phone never came this close to the buckets, and their splat smears and doubles there;
            # it sees the sim buckets and their contents instead (drawn below), and the splat's bucket pieces are dropped
            if means is None:
                means, quats = self.means0.copy(), self.quats0.copy()
            centre = -T[:3, :3].T @ T[:3, 3]
            for idx, *_ in self.pieces.values():
                means[idx] = centre
        sp = self.splat.render(T, K, W, H, bg=(1.0, 1.0, 1.0), near=near_m * self.cal.units_per_m, means=means, quats=quats)
        bg = np.clip(sp["rgb"], 0, 1) * 255
        zs = sp["median_depth"] / self.cal.units_per_m; zs[zs <= 0] = 1e9
        far = 0.999 * m.vis.map.zfar * m.stat.extent
        sim = (z < far) & (z < zs + 0.01)
        self.sha.update_scene(d, camera=cam, scene_option=self.o_desk); a1 = self.sha.render().astype(np.float32).mean(2)
        self.sha.update_scene(d, camera=cam, scene_option=self.o_bare); a0 = self.sha.render().astype(np.float32).mean(2)
        ratio = cv2.resize(np.where(a0 > 20, np.clip(a1 / np.maximum(a0, 1), 0, 1), 1.0), (W, H), interpolation=cv2.INTER_LINEAR)
        # soft like the office's diffuse ceiling light: a short contact term and a wide penumbra, scaled with the image
        k = max(W, H) / 1280.0
        dark = 0.4 * (1 - cv2.GaussianBlur(ratio, (0, 0), 3 * k)) + 0.6 * (1 - cv2.GaussianBlur(ratio, (0, 0), self.BLUR * 1.6 * k))
        # only where the splat shows the desk itself (its depth within 3.5 cm of the sim desk's), feathered
        self.dep.update_scene(d, camera=cam, scene_option=self.o_deskonly); zd = self.dep.render()
        on_desk = cv2.GaussianBlur((np.abs(zs - zd) < 0.035).astype(np.float32), (0, 0), 4 * k)
        shade = 1 - self.SHADOW * dark * on_desk
        # where the splat shows the desk itself, a sim surface in front of the sim desk is in front of it: the splat's
        # desk can stand a few mm proud and would bite into resting objects
        sim |= (z < far) & (np.abs(zs - zd) < 0.015) & (z < zd - 0.001)
        comp = (bg * shade[..., None]).astype(np.uint8); comp[sim] = rgb[sim]
        layers = []   # the arm's own Gaussians over the CAD, where they are in front of the scene and the sim
        moving = None
        if self.rs is not None:
            if self.rs_static is not None:
                layers.append(self.rs_static.render(T, K, W, H, bg=(0.0, 0.0, 0.0), near=near_m * self.cal.units_per_m))
            moving = self._render_robot_splat(T, K, near_m, hide)
            layers.append(moving)
        layers = [r for r in layers if r is not None]
        if layers:
            # the arm's Gaussians are hidden only by the scene and by sim objects, not by the CAD arm they stand for
            # (where the CAD sits a little in front of them, it would punch holes in them). Only the moving layer, near
            # the calibrated pose, needs the extra pass; the turret's CAD fits its Gaussians
            if moving is not None:
                self.dep.update_scene(d, camera=cam, scene_option=self.o_objects); zo = self.dep.render()
                zo = np.where(zo < far, zo, 1e9)
            else:
                zo = np.where(z < far, z, 1e9)
            total = np.zeros((H, W), np.float32)
            for rr in layers:
                zr = rr["median_depth"] / self.cal.units_per_m
                a = np.clip(rr["alpha"], 0, 1) * ((zr > 0) & (zr < zs + 0.01) & (zr < zo + 0.015))
                comp = (a[..., None] * np.clip(rr["rgb"], 0, 1) * 255 + (1 - a[..., None]) * comp).astype(np.uint8)
                total = np.maximum(total, a)
            # the CAD's thin rims around the Gaussians and small holes inside them: the scene behind instead (the CAD
            # silhouette is a few px wider than the real arm's, and white at its edges)
            if moving is not None:
                robot_px = sim & (z < zo - 1e-4)
                near = cv2.dilate((total > 0.6).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                rim = robot_px & near & (total < 0.5)
                comp[rim] = (bg * shade[..., None])[rim].astype(np.uint8)
        res = {"rgb": comp, "T": T, "K": K}
        if outputs:
            res.update(self._seg_outline(cam, sim, zs))
        return res

    def _seg_outline(self, cam: int, sim_vis: np.ndarray, zs: np.ndarray) -> dict:
        m, d = self.m, self.d
        self.seg.update_scene(d, camera=cam, scene_option=self.o_seg); ids = self.seg.render()[..., 0]
        self.dep.update_scene(d, camera=cam, scene_option=self.o_seg); z = self.dep.render()
        valid = ids >= 0
        geom = np.where(valid, ids, 0)
        lab = np.where(valid, self.label[geom], 0).astype(np.uint8)
        # a sim element counts where it is in front of the splat (robot, objects) or where it stands for a real
        # one the splat shows (buckets, overview camera): those are drawn by the splat, so they are not depth-tested
        robotlike = (lab == LABEL_ROBOT) | (lab >= 20)
        lab[robotlike & ~sim_vis] = 0
        outline = np.zeros((self.H, self.W, 4), np.uint8)
        k3 = np.ones((3, 3), np.uint8)
        part_of = np.full(m.ngeom, "", object)
        for g, p in self.part.items(): part_of[g] = p
        parts = np.where(valid, part_of[geom], "")
        for part, colour in OUTLINE.items():
            if part == "desk":
                continue
            mask = (parts == part) & ((~np.isin(parts, ["robot", "wrist", "object"])) | sim_vis)
            if not mask.any(): continue
            mk = mask.astype(np.uint8)
            edge = (mk - cv2.erode(mk, k3)) > 0
            # creases and rims inside the element: depth steps over 1 cm
            zz = np.where(mask, z, 0).astype(np.float32)
            step = np.maximum(np.abs(cv2.Sobel(zz, cv2.CV_32F, 1, 0, ksize=1)), np.abs(cv2.Sobel(zz, cv2.CV_32F, 0, 1, ksize=1)))
            edge |= cv2.erode(mk, k3).astype(bool) & (step > 0.01)
            outline[edge] = (*colour, 255)
        # desk edges: the top's outline as the desk alone shows it
        self.seg.update_scene(d, camera=cam, scene_option=self.o_deskonly); dk = (self.seg.render()[..., 0] >= 0).astype(np.uint8)
        dedge = (dk - cv2.erode(dk, k3)) > 0
        dedge[:2, :] = dedge[-2:, :] = False; dedge[:, :2] = dedge[:, -2:] = False
        outline[dedge & (outline[..., 3] == 0)] = (*OUTLINE["desk"], 255)
        return {"seg": lab, "outline": outline}


# --- the plugin's camera service ---------------------------------------------------------------------------------
class DeskCameraCompositor:
    """The camera service's compositor for one kit: called with the camera's plain MuJoCo frame, it returns the
    composited one. It works on a private copy of the model and data, so the live simulation is never written."""

    def __init__(self, kit: DeskKit, model: mujoco.MjModel) -> None:
        self.kit, self.source = kit, model
        self._by_size: dict[tuple[int, int], Compositor] = {}
        self._lock = threading.Lock()

    def __call__(self, renderer: object, data: mujoco.MjData, camera: str, frame: np.ndarray) -> np.ndarray:  # noqa: ARG002
        h, w = frame.shape[:2]
        with self._lock:
            comp = self._by_size.get((w, h))
            if comp is None:   # built on first use, on the thread that renders (its GL and Metal renderers live there)
                m = copy.deepcopy(self.source); d = mujoco.MjData(m)
                comp = self._by_size[(w, h)] = Compositor(m, d, self.kit, width=w, height=h)
            comp.d.qpos[:] = data.qpos; comp.d.qvel[:] = 0.0
            return comp.render_camera(camera)["rgb"]


def compositor_factory(kit_path: str | Path):
    """``CameraService.compositor_factory`` for a kit file: every camera of the model gets the shared compositor.

    Returns:
        A callable building ``{camera name: compositor}`` for a model.
    """
    kit = DeskKit.from_json(kit_path)

    def build(model: mujoco.MjModel) -> dict[str, DeskCameraCompositor]:
        shared = DeskCameraCompositor(kit, model)
        return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i): shared for i in range(model.ncam)}

    return build


def splat_shown_geoms(m: mujoco.MjModel) -> frozenset[int]:
    """The geoms the browser viewer leaves to the splat: the desk, plate and floor, the overview pole and head, the
    buckets and their contents. The arm and the task objects stay sim meshes."""
    bodies = {m.body(f"bucket_{c}").id for c in COLORS} | {0}
    bodies |= {i for n in ("overview_camera_rig", "overview_camera_tilt") if (i := mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)) >= 0}
    return frozenset(g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies)


class DeskViewerBackground:
    """The browser viewer over the desk splat: each browser's camera gets the splat rendered on Metal as its background,
    with a depth map, and viser draws the arm and the task objects over it (the splat's buckets follow the sim's).

    Built on the booth's :class:`~physicalai_mujoco_plugin.real2sim.viewer_background.SplatViewerBackground`: the desk,
    the pole and the buckets occlude by their sim depth (they match the splat's), elsewhere the splat's own depth,
    pushed 1.5 cm back so its slightly proud surfaces do not bite into what rests on them."""

    def __init__(self, server: object, model: mujoco.MjModel, data: mujoco.MjData, kit: DeskKit, *,
                 home: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None) -> None:
        from physicalai_mujoco_plugin.real2sim.geometry import Similarity
        from physicalai_mujoco_plugin.real2sim.viewer_background import SplatViewerBackground

        cal = Calibration(kit.calibration); tp = cal.t + [0, 0, cal.plate]
        transform = Similarity(rotation=cal.M, origin=-cal.M.T @ tp / cal.scale, scale=cal.scale)
        live, overrides, qpos = data, {}, data.qpos.copy()
        if home is not None:   # the splat looks right from near the phone's path: each browser starts there
            @server.on_client_connect
            def _home(client) -> None:
                client.camera.position, client.camera.look_at, client.camera.up_direction = home[0], home[1], (0.0, 0.0, 1.0)

        class _Background(SplatViewerBackground):
            def _splat_overrides(self) -> dict:
                return overrides

            def render(self, *args, **kwargs):
                self._data.qpos[:] = qpos   # the sim buckets' current poses, for their occluding depth
                return super().render(*args, **kwargs)

        self.bg = bg = _Background(server, model, kit.splat, transform, frozenset(), drawn_bodies=(), workspace=None,
                                   depth_margin=0.015)
        m = bg._model; shown = splat_shown_geoms(m)
        bg._holdout = np.array([g in shown and "_item" not in (m.geom(g).name or "") for g in range(m.ngeom)])
        means0 = bg._splat.means.cpu().numpy().astype(np.float32); quats0 = bg._splat.quats.cpu().numpy().astype(np.float32)
        pieces = bucket_pieces(cal, means0, m)
        bodies = [body for *_, body in pieces.values()]

        def watch() -> None:
            """Re-render every browser's background when a reset (or the arm) moves a bucket."""
            nonlocal overrides, qpos
            last = None
            while True:
                pos, mat = live.xpos.copy(), live.xmat.copy()
                now = np.concatenate([pos[bodies].ravel(), mat[bodies].ravel()])
                if last is None or np.abs(now - last).max() > 0.002:
                    means, quats = moved_splat(cal, pieces, means0, quats0, type("D", (), {"xpos": pos, "xmat": mat}))
                    overrides = {} if means is None else {"means": means, "quats": quats}
                    qpos = live.qpos.copy(); last = now
                    bg.refresh()
                threading.Event().wait(0.25)

        threading.Thread(target=watch, daemon=True, name="desk-viewer-buckets").start()
