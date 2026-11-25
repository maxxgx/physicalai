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
import sys
import threading
import time
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
    cameras: dict | None = None
    """Per camera name, a pose and field of view that replace the model's in the compositor's copy: ``pos`` and
    ``quat`` (wxyz) in the camera's parent body, ``fovy`` in degrees; any subset (a measured robot camera)."""
    camera_models: dict | None = None
    """Per camera name, the image model of the real camera applied to its composited frame (:func:`apply_camera_model`)."""
    sim_bucket_cameras: tuple[str, ...] | None = None
    """Robot cameras that draw the sim buckets and their contents (over the splat without the bucket pieces) instead of
    the splat's buckets; the wrist camera always does. The phone-path renders keep the splat's buckets."""
    free_running_cameras: bool = False
    """r14 (D7): each camera's worker renders on its own pace instead of in step with the others (the slowest camera
    no longer paces them all); a call returns a frame only when a new one is finished (else None: nothing to publish).
    A frame shows the state handed over when its render started (at most one frame older)."""
    workers_per_camera: int | dict | None = None
    free_running_workers: dict | None = None
    free_running_refresh_hz: float | None = None
    """r16: a free-running camera renders again at least this often although the scene did not change (a real camera
    keeps capturing: each frame its own sensor noise, which follows sim time); ``None``: only when the scene changed."""
    """r15 (D7): processes per camera in the free-running mode (``{"wrist": 2}``); 1 by default."""
    """Compositor processes per camera (default 1). More pipeline the frames: higher throughput, one frame of latency."""
    robotcam_headlight: dict | None = None
    """The robot cameras' headlight ``{ambient, diffuse}`` (:meth:`Compositor._headlight`); ``None`` keeps the phone path's."""
    sensor_noise: dict | None = None
    """The robot cameras' temporal sensor noise (:meth:`Compositor._sensor_noise`): ``read_lin`` and ``shot_lin`` (linear
    light, sigma_lin = sqrt(read_lin^2 + shot_lin * L)) and ``chroma`` (chroma/luma ratio). The environment variable
    ``R2S_SENSOR_NOISE=0`` turns it off (an audit switch); the phone-path renders never get it."""

    @classmethod
    def from_json(cls, path: str | Path) -> DeskKit:
        """Read a kit file; relative paths are relative to it."""
        path = Path(path).expanduser().resolve(); meta = json.loads(path.read_text())
        rel = lambda key: (path.parent / meta[key]).resolve() if meta.get(key) else None  # noqa: E731
        home = meta.get("viewer_home")
        return cls(splat=rel("splat"), calibration=rel("calibration"), look=rel("look"), robot_gs=rel("robot_gs"),
                   viewer_home=(tuple(home["position"]), tuple(home["look_at"])) if home else None,
                   cameras=meta.get("cameras"), camera_models=meta.get("camera_models"), sensor_noise=meta.get("sensor_noise"), robotcam_headlight=meta.get("robotcam_headlight"),
                   workers_per_camera=meta.get("workers_per_camera"), free_running_workers=meta.get("free_running_workers"), free_running_refresh_hz=meta.get("free_running_refresh_hz"), free_running_cameras=bool(meta.get("free_running_cameras", False)),
                   sim_bucket_cameras=tuple(meta.get("sim_bucket_cameras") or ()) or None)

    def __post_init__(self) -> None:
        """RUBRIC H2 (b): a camera model's tone curve has at most 16 control points per channel (R-32)."""
        for name, spec in (self.camera_models or {}).items():
            lut = (spec or {}).get("lut")
            if lut and (len(lut["x"]) > 16 or any(len(y) > 16 for y in lut["y"])):
                msg = f"camera_models.{name}.lut has more than 16 control points per channel"
                raise ValueError(msg)

    def noise_on(self) -> bool:
        import os
        return bool(self.sensor_noise) and os.environ.get("R2S_SENSOR_NOISE", "1") not in ("0", "off", "false")


_LUT_TABLES: dict = {}


def _lut_table(lut: dict) -> np.ndarray:
    """The kit's per-channel tone curve as a (1, 256, 3) uint8 table for cv2.LUT, cached per curve."""
    key = (tuple(lut["x"]), tuple(tuple(c) for c in lut["y"]))
    if key not in _LUT_TABLES:
        t = np.linspace(0.0, 1.0, 256); lx = np.asarray(lut["x"], float)
        _LUT_TABLES[key] = np.clip(np.stack([np.interp(t, lx, np.asarray(c, float)) for c in lut["y"]], -1) * 255.0 + 0.5, 0, 255).astype(np.uint8)[None]
    return _LUT_TABLES[key]


def apply_camera_model(img: np.ndarray, spec: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """A real camera's look on a composited frame: ``matrix`` (3x3, linear RGB in [0, 1]) and ``offset`` (3), then
    ``gamma``, ``lut`` (a monotone per-channel tone curve), ``vignette`` (fall-off at the corners, 0 for none), ``blur`` (Gaussian sigma in px) and ``noise``
    (sigma in 8-bit levels). Missing keys leave that step out."""
    if spec.get("lut") and not any(spec.get(k) for k in ("matrix", "offset", "vignette", "blur", "noise")) and spec.get("gamma", 1.0) == 1.0:
        return cv2.LUT(np.ascontiguousarray(img, dtype=np.uint8), _lut_table(spec["lut"]))   # the tone curve alone: one table lookup
    x = img.astype(np.float32) / 255.0
    if "matrix" in spec or "offset" in spec:
        M = np.asarray(spec.get("matrix", np.eye(3)), np.float32); off = np.asarray(spec.get("offset", np.zeros(3)), np.float32)
        x = np.clip(x @ M.T + off, 0, 1)
    if spec.get("gamma", 1.0) != 1.0:
        x = x ** (1.0 / float(spec["gamma"]))
    if spec.get("lut"):   # monotone per-channel tone curve: {"x": knots in [0, 1], "y": [[r...], [g...], [b...]]}
        lx = np.asarray(spec["lut"]["x"], np.float32); table = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        lut = np.stack([np.interp(table, lx, np.asarray(ch, np.float32)) for ch in spec["lut"]["y"]], -1)
        idx = np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
        x = np.stack([lut[..., c][idx[..., c]] for c in range(3)], -1)
    if spec.get("vignette", 0.0):
        h, w = x.shape[:2]; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        x = x * (1 - float(spec["vignette"]) * r2 / 2)[..., None]
    if spec.get("blur", 0.0):
        x = cv2.GaussianBlur(x, (0, 0), float(spec["blur"]))
    out = x * 255.0
    if spec.get("noise", 0.0):
        rng = rng or np.random.default_rng()
        out = out + rng.normal(0.0, float(spec["noise"]), out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


COLORS = ("black", "green", "red", "blue")
LABEL_BUCKET = {c: i + 1 for i, c in enumerate(COLORS)}
LABEL_ROBOT, LABEL_OVERVIEW, LABEL_DESK, LABEL_WRIST = 10, 11, 5, 12
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
    m.vis.quality.shadowsize = 1024
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
        elif key.startswith("mount:"):   # r15: a robot camera's drawn bracket (mount_colour_fit.py), e.g. "mount:wrist"
            suffix = key.split(":", 1)[1] + "_mount"
            for g in range(m.ngeom):
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(suffix) and m.geom_group[g] < 3:
                    m.geom_rgba[g] = rgba
        elif not key.startswith("geom:"):
            m.mat_rgba[m.material(key).id] = rgba


def _gain_drift(t: float, sigma: float, fc: float, seed: int) -> float:
    """r16 (D7, camera model; scripts/real2sim/v2/drift_term_fit.py): a camera's slow exposure drift at sim time *t*: a
    stationary Gaussian process of standard deviation *sigma* with a Lorentzian spectrum (corner *fc* Hz), as 32 sinusoids
    log-spaced 0.05-15 Hz with phases fixed per *seed* (the camera): reproducible, the same for every render of a time."""
    if sigma <= 0.0:
        return 0.0
    f = np.geomspace(0.05, 15.0, 32); ph = np.random.default_rng(seed).uniform(0.0, 2 * np.pi, 32)
    w = np.gradient(f) / (1.0 + (f / fc) ** 2); a = np.sqrt(2.0 * w / w.sum())
    return float(sigma * np.sum(a * np.sin(2 * np.pi * f * t + ph)))


PIECE_MARGIN = 0.025
GHOST_MARGIN = 0.06
"""The robot cameras' splat (without the buckets, which they draw from the sim) also drops everything this far around a
bucket: the splat's buckets are smeared well beyond their outline. infill_table.py lays the desk 7 cm around."""
"""How far beyond a bucket's CAD footprint its splat piece reaches: the splat's bucket edges are soft (r05: at 1.5 cm,
smudges of the real bucket stayed in front of the sim bucket in the robot cameras). The desk under it is synthetic."""


def bucket_pieces(cal: Calibration, means0: np.ndarray, m: mujoco.MjModel) -> dict:
    """Each bucket's Gaussians (with its contents): those inside its calibrated footprint, PIECE_MARGIN wider, from
    2 mm above the desk to 16 cm. They move rigidly with the sim bucket when it leaves its calibrated pose. Where two
    buckets' boxes overlap (buckets standing touching), a Gaussian goes to the bucket whose footprint it is nearer
    (r06: the moved black piece had carried part of the red bucket along)."""
    Pw = cal.scale * means0.astype(np.float64) @ cal.M.T + cal.t + [0, 0, cal.plate]
    boxes, dist = {}, {}
    for c in COLORS:
        b = cal.data["buckets"][c]; cy, sy = np.cos(b["yaw"]), np.sin(b["yaw"])
        R0 = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]]); p0 = np.array([b["x"], b["y"], 0.0])
        L = (Pw - p0) @ R0
        boxes[c] = (R0, p0, (np.abs(L[:, 0]) < 0.075 + PIECE_MARGIN) & (np.abs(L[:, 1]) < 0.10125 + PIECE_MARGIN) & (L[:, 2] > 0.002) & (L[:, 2] < 0.16))
        dist[c] = np.maximum(np.abs(L[:, 0]) - 0.075, np.abs(L[:, 1]) - 0.10125)   # signed distance to the footprint
    D = np.stack([dist[c] for c in COLORS], 1); nearest = np.array(COLORS)[np.argmin(D, axis=1)]
    pieces = {}
    for c in COLORS:
        R0, p0, inbox = boxes[c]
        pieces[c] = (np.flatnonzero(inbox & (nearest == c)), R0, p0, m.body(f"bucket_{c}").id)
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
    CONTACT, CONTACT_M, CONTACT_H = 0.33, 0.012, 0.02
    """Contact shadows (DF10, r10): under a resting object or a part near the desk, the office's diffuse light darkens
    the desk around its footprint (real allowed wrist frames: ~0.67 of the open desk at the contact edge, fading over
    ~2 cm in the resting wrist view). The darkening at the footprint's edge, the fade's width in metres (scaled to pixels by the desk's depth) and the height
    over which a part's shadow fades as it lifts."""
    SHADOW_EVERY = 1
    SHADOW_EVERY_STATIC = 2
    A0_WARP_FRAMES = 15
    """The moving wrist camera renders its shadow pass (the light's shadow map and the scene) every frame (r11: r10's
    warped reuse left ghost shapes in grasp views). A static camera (the overview) renders it every 2nd frame and
    reuses it in between only under :meth:`_shadow_reusable` (no reset, look change or jump since it was rendered)."""
    RS_FADE_DEG = 25.0
    phone_finger_handover = False
    """r17 (R-56): in render_sfm, the arm splat's fingers stand in for the CAD fingers' colour (weighted by the splat's
    weight). Off by default; only the matched-state phone render (bucketsim_v2.py heldout_render) turns it on."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, kit: DeskKit, *,
                 width: int, height: int, show_objects: bool = True, robot_splat: bool = True) -> None:
        from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer
        self.m, self.d, self.kit, self.W, self.H = m, d, kit, width, height
        self.cal = cal = Calibration(kit.calibration)
        if kit.look is not None:
            apply_look(m, kit.look)
        base_root = m.body_rootid[m.body("base_link").id]
        from physicalai_mujoco_plugin.bucket_sort import scene_objects
        object_names = scene_objects(m)
        objects = {m.body(n).id: n for n in object_names}
        label_of = {n: LABEL_OBJECT.get(n, 30 + i) for i, n in enumerate(object_names)}
        self.object_bodies = np.array(sorted(objects), dtype=int)
        self.object_geoms: list[int] = []
        bucket_body = {m.body(f"bucket_{c}").id: c for c in COLORS}
        self.label = np.zeros(m.ngeom, np.uint8); self.part = {}; self.contents = []; self.bucket_geoms = []
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]; name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            visual = m.geom_group[g] < 3
            if m.body_rootid[b] == base_root and visual:
                self.label[g] = LABEL_WRIST if name.endswith("wrist_housing") else LABEL_ROBOT
                self.part[g] = "wrist" if name.endswith("wrist_housing") else "robot"
                # the wrist bracket in a group of its own: the wrist camera, which sits against it, leaves it out
                # (with the housing; neither casts a sim shadow: next to the wrist camera theirs would fill its view)
                m.geom_group[g] = G_BRACKET if name.endswith(("wrist_mount", "wrist_housing")) else (2 if m.geom_group[g] == G_BRACKET else m.geom_group[g])
                continue
            if b in objects and visual:
                self.label[g] = label_of[objects[b]]; self.part[g] = "object"
                m.geom_group[g] = G_DRAW if show_objects else G_HIDE
                self.object_geoms.append(g)
                continue
            if b in bucket_body and "_item" in name:
                self.contents.append(g); m.geom_group[g] = G_HIDE
                continue
            if b in bucket_body and visual and "_item" not in name:
                self.label[g] = LABEL_BUCKET[bucket_body[b]]; self.part[g] = "bucket"; m.geom_group[g] = G_SEG
                self.bucket_geoms.append(g)
            elif name in ("overview_pole", "overview_pole_upper", "overview_housing", "overview_clamp"):
                self.label[g] = LABEL_OVERVIEW; self.part[g] = "overview"; m.geom_group[g] = G_SEG
            elif name == "table_top":
                self.label[g] = LABEL_DESK; self.part[g] = "desk"; m.geom_group[g] = G_DESK
            else:
                m.geom_group[g] = G_HIDE    # collision geoms, and what the splat shows: plate, contents, floor
        # each content item: its collision geom (whose contacts say whether a reset filled it) and its visual parts
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in self.contents}
        # (r13: an exposed item collides with several pieces, ``<item>`` and ``<item>_c<j>``: only its drawn parts are listed)
        self.items = {g: [names[n2] for n2 in names if n2.startswith(n + "_") and not (m.geom_contype[names[n2]] or m.geom_conaffinity[names[n2]])]
                      for n, g in names.items() if n.split("_item")[1].isdigit()}
        def opt(groups):
            o = mujoco.MjvOption(); o.geomgroup[:] = [1 if i in groups else 0 for i in range(6)]; return o
        self.o_draw, self.o_seg = opt({0, 1, 2}), opt({0, 1, 2, G_SEG, G_DESK}); self.o_draw_wrist = opt({0, 2})
        self.o_objects = opt({G_DRAW})   # the task objects alone (the arm is in groups 1 and 2)
        self.o_desk, self.o_bare, self.o_deskonly = opt({0, 2, G_DESK}), opt({G_DESK}), opt({G_DESK})
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width); m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        setup_lighting(m)
        self.rgb = mujoco.Renderer(m, height, width); self.dep = mujoco.Renderer(m, height, width); self.dep.enable_depth_rendering()
        self.seg = mujoco.Renderer(m, height, width); self.seg.enable_segmentation_rendering()
        self.sha = mujoco.Renderer(m, height // 2, width // 2)
        for r_ in (self.dep, self.seg, self.rgb):   # only the shadow-ratio renders (sha) need a shadow pass: desk shadows come from them
            r_.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        # the sim layer's colour and depth come from one pass: no skybox (its depth would mark every pixel as sim) and no
        # reflection pass (the splat shows the room)
        for r_ in (self.rgb, self.sha):
            r_.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0; r_.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        from physicalai_mujoco_plugin.real2sim.splats import read_ply_fields
        self._fields = read_ply_fields(kit.splat)
        ol = np.asarray(self._fields["opacity_logits"], np.float64).ravel()
        self._robot_only = np.abs(ol - 3.99) < 1e-5   # infill_table.py: desk laid for the robot cameras only
        self._phone_full = np.flatnonzero(~self._robot_only)   # phone-splat index -> full index
        self.splat = MetalSplatRenderer(self._subset(~self._robot_only))
        self._bucket_pieces()
        self._table_gaussians()
        self.show_objects = show_objects
        self._robot_splat(robot_splat)
        self.rig = m.body("render_camera_rig").id; self.cam = m.camera("render").id
        for name, spec in (kit.cameras or {}).items():   # measured robot cameras (the kit's), in this private copy
            k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if k < 0: continue
            if "pos" in spec: m.cam_pos[k] = spec["pos"]
            if "quat" in spec: m.cam_quat[k] = spec["quat"]
            if "fovy" in spec: m.cam_fovy[k] = spec["fovy"]
        self._noise_rng = np.random.default_rng(0)

    # --- buckets that move: their splat pieces follow the sim bodies ------------------------------------------
    def _subset(self, mask: np.ndarray) -> dict:
        return {k: (v[mask] if hasattr(v, "shape") and len(v) == len(mask) else v) for k, v in self._fields.items()}

    def _bucket_pieces(self) -> None:
        self.means0 = self.splat.means.cpu().numpy().astype(np.float32); self.quats0 = self.splat.quats.cpu().numpy().astype(np.float32)
        self.pieces = bucket_pieces(self.cal, self.means0, self.m)
        # the wrist camera never shows the splat's buckets (it sees the sim's): a copy of the splat without them, built once
        from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer
        from physicalai_mujoco_plugin.real2sim.splats import read_ply_fields
        # over the full set (phone splat + the robot-camera-only desk), indexed by full index
        fields = self._fields; N = len(np.asarray(fields["opacity_logits"]).ravel())
        keep = np.ones(N, bool)
        for idx, *_ in self.pieces.values(): keep[self._phone_full[idx]] = False
        # and the splat's soft ghosts of the buckets around them (stretched Gaussians whose centres lie outside the
        # pieces) and their mirror images under the glossy desk (the splat models a reflection as Gaussians below the
        # surface): everything within GHOST_MARGIN of a footprint, 30 cm below to 25 cm above the desk, except the
        # synthetic desk laid under and around the buckets (flat, at the desk's height; infill_table.py)
        cal = self.cal; Pw = cal.scale * np.asarray(fields["means"], np.float64) @ cal.M.T + cal.t + [0, 0, cal.plate]
        sc = np.exp(np.asarray(fields["log_scales"], np.float64)) * cal.scale   # metres
        ls = np.asarray(fields["log_scales"], np.float64); ol = np.asarray(fields["opacity_logits"], np.float64).ravel()
        synthetic = ((np.abs(ol - 4.0) < 1e-5) | self._robot_only) & (np.abs(ls[:, 0] - ls[:, 1]) < 1e-5) & (sc.min(1) < 0.001)   # infill_table.py's exact signature
        self._ghost, self._synthetic = {}, synthetic
        for c in COLORS:
            b = cal.data["buckets"][c]; cy, sy = np.cos(b["yaw"]), np.sin(b["yaw"])
            L = (Pw[:, :2] - [b["x"], b["y"]]) @ np.array([[cy, -sy], [sy, cy]])
            ghost = (np.abs(L[:, 0]) < 0.075 + GHOST_MARGIN) & (np.abs(L[:, 1]) < 0.10125 + GHOST_MARGIN) & (Pw[:, 2] > -0.30) & (Pw[:, 2] < 0.25)
            self._ghost[c] = ghost
            keep[ghost & ~synthetic] = False
        # the base plate: the sim draws it; the splat's plate and the dark Gaussians at its edges made a band (R-10)
        plate = (np.abs(Pw[:, 0]) < 0.095) & (np.abs(Pw[:, 1]) < 0.125) & (Pw[:, 2] > -0.03) & (Pw[:, 2] < 0.06)
        keep[plate & ~synthetic] = False
        self.splat_wrist = MetalSplatRenderer(self._subset(keep))
        full_to_wrist = np.where(keep, np.cumsum(keep) - 1, -1)
        self._wrist_index = full_to_wrist[self._phone_full]   # phone-splat index -> splat_wrist index, -1 if dropped
        # the robot-camera-only desk in splat_wrist: recoloured with the table look too (_apply_look)
        self._wrist_extra = full_to_wrist[np.flatnonzero(self._robot_only & keep)]
        self._wrist_extra_xy = Pw[self._robot_only & keep, :2]
        self._wrist_extra_sh0 = np.asarray(fields["sh"])[self._robot_only & keep].astype(np.float32).copy()

    def _phone_splat_without(self, colours: frozenset):
        """The phone path's splat without some buckets (their pieces and their ghosts), with the synthetic desk laid in
        their zones: for a bucket the phone saw in two places (the black one, moved during the capture), whose splat is
        the two superimposed. Built once per set. The other buckets stay the splat's and must stand at their calibrated
        poses (the matched state)."""
        cache = getattr(self, "_partial", {})
        if colours not in cache:
            from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer
            zone = np.zeros(len(self._robot_only), bool)
            for c in colours: zone |= self._ghost[c]
            keep = ~self._robot_only | (zone & self._robot_only)       # the robot-only desk inside these zones only
            for c in colours: keep[self._phone_full[self.pieces[c][0]]] = False
            keep[zone & ~self._synthetic] = False
            cache[colours] = MetalSplatRenderer(self._subset(keep)); self._partial = cache
        return cache[colours]

    def _moved_splat(self):
        """moved_splat, reused while no bucket piece moved more than 0.3 mm / ~0.06 deg since it was computed (the
        buckets stand still within an episode; recomputing every frame cost ~10 ms)."""
        poses = np.concatenate([np.r_[self.d.xpos[body], self.d.xmat[body]] for _, _, _, body in self.pieces.values()]) if self.pieces else np.zeros(0)
        cached = getattr(self, "_moved_cache", None)
        if cached is not None and cached[0].shape == poses.shape and np.abs(cached[0] - poses).max() < 3e-4:
            return cached[1]
        out = moved_splat(self.cal, self.pieces, self.means0, self.quats0, self.d)
        self._moved_cache = (poses.copy(), out)
        return out

    # --- the table's look: recolour its Gaussians (r2s_table_look), keeping the real shading -------------------------
    def _table_gaussians(self) -> None:
        """The splat's table surface: Gaussians over the sim table top (5 mm beyond its edges), from 60 cm below (its
        reflections) to 1 cm above it, away from the robot's base and from the buckets' pieces. Their colours are kept to recolour."""
        m, cal = self.m, self.cal
        self.table_idx = None; self._look = None
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "r2s_table_look")
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        if k < 0 or g < 0:
            return
        self._look_adr = int(m.numeric_adr[k])
        e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "r2s_exposure"); self._exposure_adr = int(m.numeric_adr[e]) if e >= 0 else None
        Pw = cal.scale * self.means0.astype(np.float64) @ cal.M.T + cal.t + [0, 0, cal.plate]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
        L = (Pw - m.geom_pos[g]) @ R; half = m.geom_size[g]
        top = half[2]
        # the top itself (2.5 cm below to 1 cm above) and what lies under it: the splat models the glossy top's
        # reflections (windows, ceiling lights) as Gaussians below the surface; left white, they showed through a dark
        # or saturated table look as whitish glare (r05 report, R-20)
        keep = (np.abs(L[:, 0]) < half[0] + 0.005) & (np.abs(L[:, 1]) < half[1] + 0.005) & (L[:, 2] > top - 0.60) & (L[:, 2] < top + 0.010)
        keep &= ~((np.abs(Pw[:, 0]) < 0.075) & (np.abs(Pw[:, 1]) < 0.105))   # the robot's base plate keeps its colour
        for idx, *_ in self.pieces.values():
            keep[idx] = False
        self.table_idx = np.flatnonzero(keep)
        self.table_xy = Pw[self.table_idx, :2]
        sh = self.splat.sh[self.table_idx].cpu().numpy().astype(np.float64)
        self.table_sh0 = sh
        colour = np.clip(0.28209479177387814 * sh[:, 0, :] + 0.5, 0.0, None)
        self.table_mean = np.median(colour, axis=0)

    def _num(self, name: str, n: int, default) -> np.ndarray:
        k = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
        if k < 0:
            return np.asarray(default, float)
        a = int(self.m.numeric_adr[k]); return np.asarray(self.m.numeric_data[a:a + n], float)

    def _bucket_shells(self) -> None:
        """Each bucket's shell Gaussians (walls, bottom, rim; not its contents): those of its piece outside the interior
        shrunk by 8 mm, or within 6 mm of the bottom. Their colours are kept to recolour (r2s_bucket_rgb)."""
        cal = self.cal; self.shells = {}; self._bucket_look = None
        Pw = cal.scale * self.means0.astype(np.float64) @ cal.M.T + cal.t + [0, 0, cal.plate]
        (x0, x1), (y0, y1), (z0, z1) = ((-0.069, 0.069), (-0.095, 0.095), (0.006, 0.120))
        for c, (idx, R0, p0, _) in self.pieces.items():
            L = (Pw[idx] - p0) @ R0   # bucket frame: the CAD's x across, y along (the yaw includes the quarter turn)
            inside = (L[:, 0] > x0 + 0.008) & (L[:, 0] < x1 - 0.008) & (L[:, 1] > y0 + 0.008) & (L[:, 1] < y1 - 0.008) & (L[:, 2] > z0 + 0.006)
            sidx = idx[~inside]
            sh = self.splat.sh[sidx].cpu().numpy().astype(np.float64)
            col = np.clip(0.28209479177387814 * sh[:, 0, :] + 0.5, 0, None); lum = col @ [0.299, 0.587, 0.114]
            self.shells[c] = (sidx, sh, float(np.median(lum)))

    def _apply_bucket_look(self) -> None:
        """Recolour the splat buckets' shells to the colours the model asks for (r2s_bucket_rgb), keeping their shading."""
        if not hasattr(self, "shells"):
            self._bucket_shells()
        from physicalai_mujoco_plugin.bucket_sort import bucket_colors
        cols = bucket_colors(self.m); v = self._num("r2s_bucket_rgb", 4 * len(cols), np.zeros(4 * len(cols)))
        key = tuple(np.round(v, 5))
        if key == self._bucket_look:
            return
        import torch
        self._bucket_look = key
        for i, c in enumerate(cols):
            if c not in self.shells: continue
            sidx, sh0, lum0 = self.shells[c]; sh = sh0.copy()
            if v[4 * i] > 0.5:
                target = v[4 * i + 1:4 * i + 4]
                col = np.clip(0.28209479177387814 * sh[:, 0, :] + 0.5, 0, None); lum = col @ [0.299, 0.587, 0.114]
                f = np.clip(lum / max(lum0, 1e-3), 0.25, 3.0)
                new = np.clip(target[None, :] * f[:, None], 0, 1.2)
                sh[:, 1:, :] *= (f / np.maximum(lum, 1e-3) * (target.mean()))[:, None, None].clip(0, 4)
                sh[:, 0, :] = (new - 0.5) / 0.28209479177387814
            self.splat.sh[torch.as_tensor(sidx, device=self.splat.device)] = torch.as_tensor(sh, dtype=torch.float32, device=self.splat.device)

    def _apply_light(self) -> None:
        m = self.m
        if not hasattr(self, "_light0"):
            self._light0 = (m.light_pos[0].copy(), m.light_diffuse[0].copy())
        dx, dy, k = self._num("r2s_light", 3, [0, 0, 1])
        m.light_pos[0] = self._light0[0] + [dx, dy, 0]; m.light_diffuse[0] = self._light0[1] * k

    def _jitter_camera(self, cam: int, name: str) -> None:
        m = self.m
        if not hasattr(self, "_cam0"):
            self._cam0 = {}
        base = self._cam0.setdefault(cam, (m.cam_pos[cam].copy(), m.cam_quat[cam].copy()))
        j = self._num("r2s_cam_jitter", 12, np.zeros(12))
        part = j[0:6] if name == "overview" else j[6:12] if name == "wrist" else np.zeros(6)
        q = np.zeros(4); mujoco.mju_euler2Quat(q, part[3:6], "xyz") if hasattr(mujoco, "mju_euler2Quat") else None
        if not q.any():
            rv = part[3:6]; ang = float(np.linalg.norm(rv)); q = np.r_[1.0, 0, 0, 0] if ang < 1e-12 else np.r_[np.cos(ang / 2), np.sin(ang / 2) * rv / ang]
        out = np.zeros(4); mujoco.mju_mulQuat(out, base[1], q)
        m.cam_pos[cam] = base[0] + part[:3]; m.cam_quat[cam] = out

    def _image_effects(self, img: np.ndarray, cam: int) -> np.ndarray:
        """White balance, noise and blur of the randomised look (r2s_wb, r2s_image); noise seeded by sim time and camera."""
        wb = self._num("r2s_wb", 3, [1, 1, 1]); noise, blur = self._num("r2s_image", 2, [0, 0])
        if np.allclose(wb, 1) and noise == 0 and blur == 0:
            return img
        # uint8 throughout (saturating cv2 ops); the noise comes from a bank of 16 frames drawn once per noise level and
        # camera (fixed seed), picked by sim time: same look as fresh Gaussian noise per frame at a fraction of the cost
        x = img if np.allclose(wb, 1) else cv2.multiply(np.ascontiguousarray(img), tuple(float(v) for v in wb) + (0.0,), dtype=cv2.CV_8U)
        if blur > 0: x = cv2.GaussianBlur(x, (0, 0), float(blur))
        if noise > 0:
            bank = getattr(self, "_noise_bank", {}); key = (round(float(noise), 3), cam, x.shape)
            if key not in bank:
                rng = np.random.default_rng(int(key[0] * 1000) * 131 + cam)
                n = np.rint(rng.normal(0, float(noise), (16, *x.shape))).clip(-255, 255).astype(np.int16)
                bank = {key: (np.clip(n, 0, 255).astype(np.uint8), np.clip(-n, 0, 255).astype(np.uint8))}; self._noise_bank = bank
            pos, neg = bank[key]; k = (int(round(float(self.d.time) * 1000)) * 131 + cam) % 16
            x = cv2.subtract(cv2.add(x, pos[k]), neg[k])
        return x

    def _recolour(self, sh0: np.ndarray, xy: np.ndarray, ids: np.ndarray, look: tuple) -> np.ndarray:
        """The table look on a set of desk Gaussians (SH, scene xy, a per-Gaussian id for the speckle): the colour and
        pattern of r2s_table_look, shading kept; the real desk when the look is off."""
        sh = sh0.copy()
        if look[0] > 0.5:
            target = np.asarray(look[1:4]); pattern = int(look[4]); scale, phase = look[5], look[6]
            u, v = xy[:, 0], xy[:, 1]
            hsh = np.sin(ids * 12.9898 + 78.233) * 43758.5453; noise = hsh - np.floor(hsh)
            if pattern == 1:   # wood grain along the table's y
                f = 1 + 0.10 * np.sin(2 * np.pi * (v * 22 * scale + 0.8 * np.sin(u * 6 * scale + phase))) + 0.05 * (noise - 0.5)
            elif pattern == 2:   # speckle
                f = 1 + 0.22 * (noise - 0.5)
            elif pattern == 3:   # a light grid, like a cutting mat
                cell = 0.05 / scale; du = np.abs(((u + phase * 0.01) / cell) % 1 - 0.5); dv = np.abs(((v + phase * 0.01) / cell) % 1 - 0.5)
                f = np.where((du > 0.46) | (dv > 0.46), 1.25, 1.0)
            else:
                f = np.ones(len(u))
            ratio = (target / np.maximum(self.table_mean, 1e-3))[None, :] * f[:, None]
            colour = np.clip(0.28209479177387814 * sh[:, 0, :] + 0.5, 0.0, None) * ratio
            sh[:, 1:, :] *= ratio[:, None, :]
            sh[:, 0, :] = (np.clip(colour, 0, 1.2) - 0.5) / 0.28209479177387814
        return sh

    def _apply_look(self) -> float:
        """Recolour the table Gaussians when the model's look changed; return the exposure gain."""
        m = self.m
        self._apply_bucket_look(); self._apply_light()
        if self.table_idx is None:
            return 1.0
        look = tuple(np.round(m.numeric_data[self._look_adr:self._look_adr + 7], 5))
        gain = float(m.numeric_data[self._exposure_adr]) if self._exposure_adr is not None else 1.0
        if look == self._look:
            return gain
        import torch
        self._look = look
        sh = self._recolour(self.table_sh0, self.table_xy, self.table_idx, look)
        self.splat.sh[torch.as_tensor(self.table_idx, device=self.splat.device)] = torch.as_tensor(sh, dtype=torch.float32, device=self.splat.device)
        if hasattr(self, "splat_wrist"):   # the same table look in the robot cameras' copy
            iw = self._wrist_index[self.table_idx]; ok_ = iw >= 0
            self.splat_wrist.sh[torch.as_tensor(iw[ok_], device=self.splat.device)] = torch.as_tensor(sh[ok_], dtype=torch.float32, device=self.splat.device)
            if len(self._wrist_extra):
                she = self._recolour(self._wrist_extra_sh0, self._wrist_extra_xy, np.arange(len(self._wrist_extra)) + 7919, look)
                self.splat_wrist.sh[torch.as_tensor(self._wrist_extra, device=self.splat.device)] = torch.as_tensor(she, dtype=torch.float32, device=self.splat.device)
        return gain

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
        names = np.array(meta["links"])
        # (tried at r07: dropping the fingers' Gaussians - the bare sim fingers looked worse than the real ones)
        fingers = names == "dropped"   # r17 (R-56): labelled out by relabel_finger_gaussians.py (DROP_MM), never drawn
        static = names == "base_link"
        # the base's Gaussians (the turret, the clamp, the cables at the base) never move: drawn at full weight always
        sub = lambda keep: {k: (v[keep] if hasattr(v, "shape") and len(v) == len(static) else v) for k, v in fields.items()}  # noqa: E731
        self.rs_static = MetalSplatRenderer(sub(static)) if static.any() else None
        moving = ~static & ~fingers
        self.rs = MetalSplatRenderer(sub(moving))
        meta = {**meta, "links": [str(n) for n in names[moving]]}
        self.rs_means0 = self.rs.means.cpu().numpy().astype(np.float64); self.rs_quats0 = self.rs.quats.cpu().numpy().astype(np.float64)
        d0 = mujoco.MjData(m); d0.qpos[:] = self.d.qpos
        for i, v in enumerate(meta["joints_rad"][:6]): d0.qpos[m.jnt_qposadr[m.joint(f"joint{i + 1}").id]] = v
        # [6] is the gripper's opening (m) when the Gaussians were cut; each finger joint moves half of it
        for n in ("joint_left", "joint_right"): d0.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["joints_rad"][6] / 2
        mujoco.mj_kinematics(m, d0)
        links = np.array([m.body(n).id for n in meta["links"]])
        self.rs_groups = [(b, np.flatnonzero(links == b), d0.xmat[b].reshape(3, 3).copy(), d0.xpos[b].copy()) for b in np.unique(links)]

    def _render_robot_splat(self, T: np.ndarray, K: np.ndarray, near_m: float, hide: tuple[int, ...] = ()):
        from physicalai_mujoco_plugin.real2sim.splats import matrix_to_quat, quat_multiply
        cal, d = self.cal, self.d
        turn = max(float(np.degrees(np.arccos(np.clip((np.trace(d.xmat[b].reshape(3, 3) @ R0.T) - 1) / 2, -1, 1))))
                   for b, _, R0, _ in self.rs_groups)
        if turn >= self.RS_FADE_DEG:
            return None   # faded out: skip the work
        means, quats = self.rs_means0.copy(), self.rs_quats0.copy()
        tp = cal.t + [0, 0, cal.plate]
        for b, idx, R0, p0 in self.rs_groups:
            Rd = d.xmat[b].reshape(3, 3) @ R0.T; pd = d.xpos[b] - Rd @ p0
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
    def render_sfm(self, world_to_cam: np.ndarray, K: np.ndarray, *, outputs: bool = False, sim_buckets: frozenset | None = None) -> dict:
        """A phone-path view; *sim_buckets*: buckets drawn from the sim instead of the splat (a bucket moved during the
        capture), the others from the splat at their calibrated poses."""
        self._headlight(False)
        cam = self._virtual(world_to_cam, K)
        self._phone = True
        try:
            return self._render(cam, world_to_cam, K, outputs, near_m=0.01, sim_buckets=frozenset(sim_buckets) if sim_buckets else None)
        finally:
            self._phone = False

    def _headlight(self, robot_camera: bool, camera: str | None = None) -> None:
        """The robot cameras' headlight (kit ``robotcam_headlight``: ambient, diffuse), fitted so the MuJoCo-lit parts
        (buckets, arm) sit at the real robot cameras' brightness after the tone curve; the phone path keeps
        :func:`setup_lighting`'s."""
        spec = getattr(self.kit, "robotcam_headlight", None) or {}
        if camera is not None and isinstance(spec.get(camera), dict):   # r12: per camera (the overview's buckets)
            spec = spec[camera]
        elif any(isinstance(v, dict) for v in spec.values()):
            spec = {}
        h = self.m.vis.headlight
        if robot_camera and spec:
            h.ambient[:] = float(spec.get("ambient", 0.28)); h.diffuse[:] = float(spec.get("diffuse", 0.42))
        else:
            h.ambient[:] = 0.28; h.diffuse[:] = 0.42

    OCC_M, OCC_NEAR, OCC_FAR, OCC_STRENGTH = 0.010, 0.002, 0.030, 0.5   # r12: 0.5 at full occlusion (r11 0.33: not visible in the overview)
    """Contact occlusion on the sim layer (:meth:`_sim_occlusion`): the sampling radius in metres (scaled to pixels by
    the scene's depth) and the depth range (m) within which a closer neighbour counts as touching (beyond it, e.g. a
    finger above the floor, it does not: no halo)."""

    def _sim_occlusion(self, cam: int, z: np.ndarray, sim: np.ndarray, far: float) -> np.ndarray | None:
        """A screen-space contact term on the sim layer at half resolution (0..1, as the desk's contact map): per pixel,
        the share of 16 neighbours (8 directions, 2 radii ~OCC_M apart in the scene) that are closer to the camera by
        OCC_NEAR..OCC_FAR, i.e. touching surfaces that rise next to it (an object on a bucket floor, the floor-wall
        crease, a pile's gaps); twice that share, clipped, so a pixel half enclosed is fully dark. Sim depth only."""
        if not sim.any():
            return None
        H, W = z.shape; h2, w2 = H // 2, W // 2
        zz = np.where(sim, z, np.float32(far)).astype(np.float32)
        z2 = cv2.resize(zz, (w2, h2), interpolation=cv2.INTER_NEAREST)
        valid = z2 < far * 0.5
        if not valid.any():
            return None
        f = (h2 / 2) / np.tan(np.radians(self.m.cam_fovy[cam]) / 2)
        zmed = float(np.median(z2[::4, ::4][valid[::4, ::4]])) if valid[::4, ::4].any() else 0.5
        r1 = max(1, int(round(f * self.OCC_M / zmed))); acc = np.zeros((h2, w2), np.float32); n = 0
        for r in (r1, 2 * r1):
            for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
                nb = np.roll(z2, (dy * r, dx * r), axis=(0, 1)); dz = z2 - nb
                acc += ((dz > self.OCC_NEAR) & (dz < self.OCC_FAR)).astype(np.float32); n += 1
        occ = np.clip(2.0 * acc / n, 0.0, 1.0) * valid
        return cv2.GaussianBlur(occ, (0, 0), 1.0)

    def _contact_parts(self) -> list[int]:
        """The geoms whose footprints cast contact shadows: the task objects' drawn geoms and the fingers' (cached)."""
        if getattr(self, "_contact_geoms", None) is None:
            m = self.m; fingers = {m.body(n).id for n in ("gripper_left", "gripper_right") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
            self._contact_geoms = list(self.object_geoms) + [g for g in range(m.ngeom) if m.geom_bodyid[g] in fingers and self.part.get(g) == "robot"]
            t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
            self._desk_top = None if t < 0 else ("table_top", t)
        return self._contact_geoms

    def _contact_shadow(self, cam: int, w2: int, h2: int, zd2: np.ndarray) -> np.ndarray:
        """Contact darkness at half resolution (0..1): each part within CONTACT_H of the desk, its lowest centimetre's
        footprint dropped onto the desk plane and projected into the camera (pinhole, the composite's own), weighted by
        exp(-height / CONTACT_H), then blurred by CONTACT_M at the desk's median depth. Sim geometry only."""
        m, d = self.m, self.d; geoms = self._contact_parts()
        if self._desk_top is None or not geoms:
            return np.zeros((h2, w2), np.float32)
        t = self._desk_top[1]; desk_z = float(d.geom_xpos[t][2] + m.geom_size[t][2])
        R = d.cam_xmat[cam].reshape(3, 3); pos = d.cam_xpos[cam]
        f = (h2 / 2) / np.tan(np.radians(m.cam_fovy[cam]) / 2); cx, cy = w2 / 2, h2 / 2
        acc = np.zeros((h2, w2), np.float32)
        local = getattr(self, "_contact_local", None)
        if local is None:   # per geom, its (subsampled) vertices in its own frame, and its body's reach: computed once
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float); local = {}
            for g in geoms:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = int(m.geom_dataid[g]); v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                    v = np.asarray(v[:: max(1, len(v) // 200)], float)
                else:
                    v = m.geom_aabb[g, :3] + corners * m.geom_aabb[g, 3:]
                local[g] = v
            self._contact_local = local
        by_body: dict[int, list[np.ndarray]] = {}
        for g in geoms:
            if self.part.get(g) == "object" and m.geom_group[g] == G_HIDE:   # objects not shown (show_objects=False)
                continue
            b = int(m.geom_bodyid[g]); zb = float(d.xpos[b][2]) - desk_z
            if zb > 0.15 or zb < -0.08:   # parked, or well above the desk: no contact shadow (cheap test first)
                continue
            by_body.setdefault(b, []).append(local[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        hw = float(getattr(self, "_handover_w", 0.0)); finger_bodies = {int(m.geom_bodyid[g]) for g in self._finger_geoms()} if hw > 0.0 else set()
        for b_, P in by_body.items():
            # r18 (R-60): a finger left out of a phone view's colour casts no shadow there (weighted by 1 - w)
            strength = (1.0 - hw) if b_ in finger_bodies else 1.0
            if strength <= 0.0:
                continue
            P = np.vstack(P); low = float(P[:, 2].min()); h = low - desk_z
            if h > 3 * self.CONTACT_H or h < -0.05:   # lifted away, or parked under the desk
                continue
            foot = P[P[:, 2] < low + 0.01].copy(); foot[:, 2] = desk_z
            c = (foot - pos) @ R; z = -c[:, 2]
            if (z <= 0.01).any():
                continue
            uv = np.stack([cx + f * c[:, 0] / z, cy - f * c[:, 1] / z], 1).astype(np.float32)
            hull = cv2.convexHull(uv).round().astype(np.int32)
            cv2.fillConvexPoly(acc, hull, strength * float(np.exp(-max(h, 0.0) / self.CONTACT_H)))   # overlapping parts: the stronger
        if not acc.any():
            return acc
        zs_ = zd2[::8, ::8]; on = np.isfinite(zs_) & (zs_ > 0) & (zs_ < 5)
        zmed = float(np.median(zs_[on])) if on.any() else 0.5
        sigma = max(0.6, f * self.CONTACT_M / zmed)
        # the footprint blurred, doubled and clipped: full (1) inside and at its edge, fading outward over ~CONTACT_M
        # (a wide blur at half this resolution again, then upsampled: the same soft field at a quarter of the cost)
        if sigma > 3.0:
            small = cv2.resize(acc, (w2 // 2, h2 // 2), interpolation=cv2.INTER_AREA)
            blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 2), (w2, h2), interpolation=cv2.INTER_LINEAR)
        else:
            blurred = cv2.GaussianBlur(acc, (0, 0), sigma)
        return np.clip(2.0 * blurred, 0.0, 1.0)

    def render_camera(self, name: str, *, outputs: bool = False) -> dict:
        self._headlight(True, name)
        cam = self.m.camera(name).id; self._jitter_camera(cam, name); mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)
        T = self.cal.scene_to_camera(self.d.cam_xpos[cam], self.d.cam_xmat[cam].reshape(3, 3))
        # the splat holds the real overview head (and, at rest, the real wrist camera): drop what is within 6 cm
        # the wrist camera does not see its own bracket and housing (they ride on link6 with it)
        hide = (self.m.body("link6").id,) if self.m.cam_bodyid[cam] == self.m.body("link6").id else ()
        # the 6 cm splat cut hides the real overview head at the fixed camera; the wrist camera moves with the gripper and
        # sees the desk and buckets down to a few cm (R-35): 2 cm there
        res = self._render(cam, T, self.camera_K(cam), outputs, near_m=0.02 if hide else 0.06, hide=hide,
                           sim_buckets=bool(hide) or name in (self.kit.sim_bucket_cameras or ()))
        spec = (self.kit.camera_models or {}).get(name)
        if spec:
            res["rgb"] = apply_camera_model(res["rgb"], spec, np.random.default_rng(int(round(float(self.d.time) * 1000)) * 7 + cam))
        contact = getattr(self, "_contact_full", None); self._contact_full = None
        if contact is not None and contact.shape == res["rgb"].shape[:2]:
            f = (1.0 - self.CONTACT * contact).astype(np.float32)
            res["rgb"] = cv2.multiply(res["rgb"], cv2.merge([f, f, f]), dtype=cv2.CV_8U)
        res["rgb"] = self._image_effects(res["rgb"], cam)
        if self.kit.noise_on():
            res["rgb"] = self._sensor_noise(self._frame_jitter(res["rgb"], cam), cam)
        return res

    def _frame_jitter(self, img: np.ndarray, cam: int) -> np.ndarray:
        """Frame-to-frame changes of a real camera that sensor noise is not (r11, D7, user decision U-2): a per-frame
        exposure gain jitter (kit ``sensor_noise.cameras.<name>.gain_jitter``: sigma of the gain) and, for a camera on
        the arm, a sub-pixel image shake (``shake_px``: [sigma, share of larger moves, their sigma], px per axis: a
        mixture). Fitted on still pairs of the allowed episodes (real/still_motion_real.json; frame lists there):
        median |change of mean level| 0.12 % (overview) / 0.36 % (wrist); the wrist's global shift median 0.06 px,
        90th percentile 0.69 px. Drawn per sim time and camera (reproducible), on with the sensor noise."""
        spec = (self.kit.sensor_noise or {}).get("cameras", {}).get(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_CAMERA, int(cam)) or "") or {}
        gj, shake, drift = spec.get("gain_jitter", 0.0), spec.get("shake_px"), spec.get("gain_drift")
        # a sigma, or a mixture [sigma, share of larger changes, their sigma] (the real changes are heavy-tailed)
        gj = [float(gj), 0.0, 0.0] if not isinstance(gj, (list, tuple)) else [float(v) for v in gj]
        if gj[0] <= 0.0 and not shake and not drift:
            return img
        rng = np.random.default_rng((int(round(float(self.d.time) * 1000)) * 92821 + int(cam) * 7 + 3) % (2**32))
        out = img
        if shake:
            s1, pb, s2 = (float(v) for v in shake)
            dx, dy = rng.normal(0.0, s2 if rng.random() < pb else s1, 2)
            out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (out.shape[1], out.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        alpha = 1.0
        if gj[0] > 0.0:
            sig = gj[2] if (gj[1] > 0.0 and rng.random() < gj[1]) else gj[0]
            alpha += sig * rng.standard_normal()
        if drift:   # r16: slow exposure drift, a stationary function of sim time (drift_term_fit.py; kit [sigma, corner Hz])
            alpha += _gain_drift(float(self.d.time), float(drift[0]), float(drift[1]), 11 + int(cam))
        if alpha != 1.0:
            out = cv2.convertScaleAbs(out, alpha=float(alpha))
        return out

    def _sensor_noise(self, img: np.ndarray, cam: int) -> np.ndarray:
        """Temporal sensor noise of a colour camera streaming over UVC (kit ``sensor_noise``), new in every frame.

        The noise is defined in linear light, sigma_lin(L) = sqrt(read_lin^2 + shot_lin * L), and carried through the
        sRGB curve to 8-bit levels per pixel value (gamma-aware: relatively more in the shadows, less in the highlights).
        It is added in YCbCr: luma noise part per pixel, part drawn at half resolution and upsampled (demosaicing and
        compression leave it mildly correlated), chroma noise at quarter resolution, ``chroma`` times weaker (4:2:2 / 4:2:0 streams). Each
        frame draws fresh fields from a generator seeded by the sim time (ms) and the camera: no field ever repeats,
        and a rerun of the same episode reproduces the same frames."""
        spec = self.kit.sensor_noise; h, w = img.shape[:2]
        per = (spec.get("cameras") or {}).get(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_CAMERA, int(cam)) or "")
        if per:   # per-camera values (kit ``sensor_noise.cameras.<name>``) over the shared ones
            spec = {**spec, **per}
        key = (img.shape, float(spec.get("read_lin", 2e-4)), float(spec.get("shot_lin", 1.15e-4)), float(spec.get("chroma", 0.5)),
               float(spec.get("chroma_cb", spec.get("chroma", 0.5))), float(spec.get("gain", 1.0)))
        low = float(spec.get("low_share", 0.0))   # r11: share of the luma variance in a 1/8-resolution field
        caches = self.__dict__.setdefault("_sn_caches", {}); cache = caches.get(key)
        if cache is None:
            v = np.arange(256) / 255.0
            L = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
            Lp = np.minimum(L + 1e-4, 1.0); Lm = np.maximum(L - 1e-4, 0.0)
            enc = lambda x: np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
            slope = 255.0 * (enc(Lp) - enc(Lm)) / (Lp - Lm)   # d(level)/d(linear)
            lut = np.sqrt(key[1] ** 2 + key[2] * L) * slope   # 8-bit levels of noise at each value
            # the variance an upsampled field keeps (bilinear): measured once, so the amplitude is the model's
            probe = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
            k2 = 1.0 / float(cv2.resize(probe, (128, 128), interpolation=cv2.INTER_LINEAR).std())
            k4 = 1.0 / float(cv2.resize(probe, (256, 256), interpolation=cv2.INTER_LINEAR).std())
            k8 = 1.0 / float(cv2.resize(probe, (512, 512), interpolation=cv2.INTER_LINEAR).std())
            cache = caches[key] = (key, (np.clip(lut, 0, 4.0) * key[5]).astype(np.float32), k2, k4, k8)
        _, lut, k2, k4, k8 = cache
        bufs = getattr(self, "_sn_bufs", None)
        if bufs is None or bufs[0].shape != (h // 2, w // 2):
            bufs = self._sn_bufs = (np.empty((h // 2, w // 2), np.float32), np.empty((h, w), np.float32), np.empty((h // 4, w // 4, 2), np.float32),
                                    np.empty((h // 8, w // 8), np.float32))
        half, full, quarter, eighth = bufs
        # fresh per frame, reproducible; OpenCV takes a C int, so the seed is reduced mod 2^31 - 1 (it overflowed
        # after ~271 s of sim time and every render failed)
        cv2.setRNGSeed((int(round(float(self.d.time) * 1000)) * 7919 + int(cam) * 104729 + 17) % 2147483647)
        cv2.randn(half, 0.0, 1.0); cv2.randn(full, 0.0, 1.0); cv2.randn(quarter, 0.0, 1.0)
        # luma: a half-resolution field (correlated over ~2 px) mixed with a per-pixel one, unit variance (lag-1 ~0.45)
        ny = cv2.addWeighted(cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR), k2 * 0.8, full, 0.6, 0.0)
        if low > 0.0:   # r11 (D7): the real sensor's blotchy part, which survives AV1 where the fine part does not
            cv2.randn(eighth, 0.0, 1.0)
            ny = cv2.addWeighted(ny, float(np.sqrt(1.0 - low)), cv2.resize(eighth, (w, h), interpolation=cv2.INTER_LINEAR), float(np.sqrt(low) * k8), 0.0)
        nc = cv2.resize(quarter, (w, h), interpolation=cv2.INTER_LINEAR)
        ycc = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2YCrCb)
        sy = cv2.LUT(ycc[..., 0], lut.reshape(1, 256))   # the noise level at each pixel's value (float32)
        noise = cv2.merge([cv2.multiply(ny, sy), cv2.multiply(nc[..., 0], sy, scale=k4 * key[3]), cv2.multiply(nc[..., 1], sy, scale=k4 * key[4])])   # YCrCb: [1] Cr (chroma), [2] Cb (chroma_cb)
        out = cv2.add(ycc.astype(np.float32), noise)
        return cv2.cvtColor(cv2.convertScaleAbs(out), cv2.COLOR_YCrCb2RGB)   # saturating (values stay >= 0 in YCrCb)

    def _robot_splat_weight(self) -> float:
        """The arm splat's weight now: 1 at its calibrated pose, 0 once any link has turned RS_FADE_DEG from it."""
        d = self.d
        turn = max(float(np.degrees(np.arccos(np.clip((np.trace(d.xmat[b].reshape(3, 3) @ R0.T) - 1) / 2, -1, 1))))
                   for b, _, R0, _ in self.rs_groups)
        return float(np.clip(1 - turn / self.RS_FADE_DEG, 0, 1))

    def _finger_geoms(self) -> list[int]:
        """The drawn geoms of the fingers (cached)."""
        if not hasattr(self, "_finger_geoms_cache"):
            m = self.m; bodies = {m.body(n).id for n in ("gripper_left", "gripper_right") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
            self._finger_geoms_cache = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_contype[g] == 0
                                        and m.geom_conaffinity[g] == 0 and m.geom_group[g] < 3]
        return self._finger_geoms_cache

    def _rgb_depth(self, cam: int, option) -> tuple[np.ndarray, np.ndarray]:
        """One render pass, both buffers: the colour and the metric depth (as mujoco.Renderer's depth mode computes it)."""
        r = self.rgb; r.update_scene(self.d, camera=cam, scene_option=option); rgb = r.render()
        depth = np.empty((self.H, self.W), np.float32)
        mujoco.mjr_readPixels(None, depth, r._rect, r._mjr_context)
        depth = depth[::-1]   # OpenGL reads bottom-up; Renderer.render() returned the colour already flipped upright
        extent = self.m.stat.extent; zn = np.float32(self.m.vis.map.znear * extent); zf = np.float32(self.m.vis.map.zfar * extent)
        c = -(zf + zn) / (zf - zn); dd = -(np.float32(2) * zf * zn) / (zf - zn); c = np.float32(-0.5) * c - np.float32(0.5); dd = np.float32(-0.5) * dd
        z = (dd / (depth.astype(np.float64) + c)).astype(np.float32)
        z[depth <= np.float32(1e-7)] = np.float32(2.0 * zf)   # nothing drawn (the cleared reverse-Z buffer): beyond far
        return rgb, z

    REUSE_QPOS_RAD, REUSE_POS_M = 0.03, 0.004
    """A static camera reuses its last shadow ratio for one frame only while no joint moved more than 0.03 rad and no
    free body more than 4 mm since that ratio was rendered (about what a 30 Hz frame of normal motion does), and the
    look (materials, numerics, light) is the same: a reset, a look change or a fast move renders afresh."""

    def _shadow_signature(self) -> tuple:
        m, d = self.m, self.d
        look = hash((m.mat_rgba.tobytes(), m.numeric_data.tobytes(), m.light_pos.tobytes(), m.light_diffuse.tobytes(), m.geom_rgba.tobytes()))
        return (look, d.qpos.copy())

    def _shadow_reusable(self, old: tuple, new: tuple) -> bool:
        if old[0] != new[0] or old[1].shape != new[1].shape:
            return False
        m = self.m; dq = np.abs(new[1] - old[1])
        if not hasattr(self, "_qpos_kind"):   # per qpos entry: 0 hinge/slide, 1 free position, 2 free quaternion
            kind = np.zeros(m.nq, np.int8)
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    a = m.jnt_qposadr[j]; kind[a:a + 3] = 1; kind[a + 3:a + 7] = 2
                elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                    a = m.jnt_qposadr[j]; kind[a:a + 4] = 2
            self._qpos_kind = kind
        k = self._qpos_kind
        return bool((dq[k == 0] <= self.REUSE_QPOS_RAD).all() and (dq[k == 1] <= self.REUSE_POS_M).all() and (dq[k == 2] <= 0.02).all())

    def _desk_homography(self, cam: int, h: int, w: int) -> np.ndarray:
        """The 3x3 map from desk-plane points (x, y, 1 on z = 0, world) to this camera's pixels (homogeneous, an h x w
        image with OpenCV's pixel-centre convention), for the camera's current pose and field of view."""
        m, d = self.m, self.d
        f = h / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2); cx, cy = w / 2 - 0.5, h / 2 - 0.5
        R = d.cam_xmat[cam].reshape(3, 3); c = d.cam_xpos[cam]
        Rt = R.T; P = np.column_stack([Rt[:, 0], Rt[:, 1], -Rt @ c])        # camera coordinates of (x, y, 1)
        A = np.array([[f, 0.0, -cx], [0.0, -f, -cy], [0.0, 0.0, -1.0]])      # MuJoCo camera: x right, y up, looking along -z
        return A @ P

    def _desk_depth(self, cam: int) -> np.ndarray:
        """The sim desk top's depth (camera z, metres) at every pixel, from the plane z = 0 and the desk box; far elsewhere.
        The inverse depth of a plane is affine in the image (an outer sum); the box mask is found at 1/8 resolution."""
        m, d = self.m, self.d
        f = self.H / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2)
        xn = ((np.arange(self.W, dtype=np.float32) + 0.5 - self.W / 2) / f); yn = -((np.arange(self.H, dtype=np.float32) + 0.5 - self.H / 2) / f)
        R = d.cam_xmat[cam].reshape(3, 3); c = d.cam_xpos[cam]; cz = float(c[2]) if abs(c[2]) > 1e-6 else 1e-6
        inv = (-R[2, 0] / cz) * xn[None, :] + (-R[2, 1] / cz) * yn[:, None] + np.float32(R[2, 2] / cz)
        far = np.float32(0.999 * m.vis.map.zfar * m.stat.extent * 2)
        depth = np.where(inv > 1e-6, 1.0 / np.maximum(inv, 1e-6), far).astype(np.float32)
        s8 = 8; xs, ys = xn[s8 // 2::s8], yn[s8 // 2::s8]
        dirs = np.stack(np.broadcast_arrays(xs[None, :], ys[:, None], -np.ones((1, 1), np.float32)), -1) @ R.T
        t = np.where(dirs[..., 2] < -1e-6, -c[2] / np.minimum(dirs[..., 2], -1e-6), np.inf)
        g = m.geom("table_top").id; p = c[:2] + dirs[..., :2] * t[..., None]
        L = (p - d.geom_xpos[g][:2]) @ d.geom_xmat[g].reshape(3, 3)[:2, :2]
        inside = ((np.abs(L[..., 0]) <= m.geom_size[g][0]) & (np.abs(L[..., 1]) <= m.geom_size[g][1]) & np.isfinite(t)).astype(np.uint8)
        inside = cv2.resize(inside, (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0
        return np.where(inside, depth, far)

    def _render(self, cam: int, T: np.ndarray, K: np.ndarray, outputs: bool, near_m: float, hide: tuple[int, ...] = (),
                sim_buckets: bool | None = None) -> dict:
        # sim_buckets: draw the sim buckets and their contents over the splat copy without the bucket pieces (the wrist
        # camera always; others when the kit lists them in sim_bucket_cameras)
        sim_buckets = bool(hide) if sim_buckets is None else sim_buckets
        some = sim_buckets if isinstance(sim_buckets, (set, frozenset)) else None   # only these buckets from the sim
        if some is not None and not some: sim_buckets, some = False, None
        m, d, W, H = self.m, self.d, self.W, self.H
        m.geom_group[self.contents] = G_HIDE   # a reset's fill restores their groups; the splat shows the contents
        mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
        if self.object_geoms and self.show_objects:   # pool objects parked below the floor are drawn nowhere
            og = np.asarray(self.object_geoms); parked = d.xpos[m.geom_bodyid[og], 2] < -1.0
            m.geom_group[og] = np.where(parked, G_HIDE, G_DRAW)
        gain = self._apply_look()
        # the look moved the key light (r2s_light): its world position must follow now; mj_camlight ran before it, so
        # the render used the previous call's light (r07 S2: a state reproduced offline differed in the sim's lighting)
        mujoco.mj_camlight(m, self.d)
        o = self.o_draw_wrist if hide else self.o_draw
        shown = []
        if sim_buckets:   # the sim buckets, and the contents a reset filled (their contacts are on)
            shown = list(self.bucket_geoms) + [g for coll, parts in self.items.items() if m.geom_contype[coll] or m.geom_conaffinity[coll] for g in parts]
            if some is not None:
                bodies = {m.body(f"bucket_{c}").id for c in some}
                shown = [g for g in shown if m.geom_bodyid[g] in bodies]
            m.geom_group[shown] = G_DRAW
        # r17 (R-56, D3 L8; coordinator ruling during r17): in a matched-state phone view (the held-out phone renders: the
        # arm at its calibrated pose) the arm splat draws the real fingers; the CAD fingers' faces the phone barely saw
        # (their lit tops) read as a second finger beside them. With ``phone_finger_handover`` on (only the HELDOUT
        # render sets it; VIDEO, ROBOTCAM, recordings and demos never), the CAD fingers' colour is weighted by 1 - w, w the
        # arm splat's own weight (_robot_splat_weight: 1 at the calibrated pose, 0 once any link has turned RS_FADE_DEG),
        # so they are left out at w = 1 and fade back with the splat. The seg and outline outputs keep them.
        w = self._robot_splat_weight() if (getattr(self, "_phone", False) and self.phone_finger_handover and self.rs is not None) else 0.0
        self._handover_w = w   # r18 (R-60): the fingers' contact shadow fades with their colour
        fingers = self._finger_geoms() if w > 0.0 else []
        z_seg = None   # the depth with the CAD fingers, for the seg and outline outputs
        if fingers:
            rgb_cad, z_seg = self._rgb_depth(cam, o); z = z_seg
            keep_groups = m.geom_group[fingers].copy(); m.geom_group[fingers] = G_HIDE
            try:
                rgb, z_bare = self._rgb_depth(cam, o)
            finally:
                m.geom_group[fingers] = keep_groups
            if w < 1.0:   # the CAD fingers' colour at 1 - w; the depth (the sim layer's mask) keeps them
                rgb = (w * rgb.astype(np.float32) + (1.0 - w) * rgb_cad.astype(np.float32) + 0.5).astype(np.uint8)
            else:
                z = z_bare
        else:
            rgb, z = self._rgb_depth(cam, o)
        if shown:
            m.geom_group[self.bucket_geoms] = G_SEG; m.geom_group[self.contents] = G_HIDE
        # the wrist camera: the phone never came this close to the buckets, and their splat smears and doubles there; it
        # sees the sim buckets and their contents instead (drawn below), from the splat copy without the bucket pieces
        means, quats = (None, None) if sim_buckets else self._moved_splat()
        splat = self._phone_splat_without(frozenset(some)) if some is not None else (self.splat_wrist if sim_buckets else self.splat)
        # the splat image of a camera that has not moved, with no bucket moved and the same look: from the cache (the
        # overview camera within an episode)
        key = (T.tobytes(), K.tobytes(), W, H, near_m, bool(hide), tuple(sorted(some)) if some is not None else bool(sim_buckets), None if means is None else hash(means[:: max(1, len(means) // 4096)].tobytes()),
               self._look, getattr(self, "_bucket_look", None))
        cached = getattr(self, "_sp_cache", {}).get(cam)
        if cached is not None and cached[0] == key:
            sp = cached[1]
        else:
            # (r12: queuing this before the sim's colour pass was tried; Metal and OpenGL contend, it was slower)
            sp = splat.render(T, K, W, H, bg=(1.0, 1.0, 1.0), near=near_m * self.cal.units_per_m, means=means, quats=quats)
            if not hasattr(self, "_sp_cache"): self._sp_cache = {}
            self._sp_cache[cam] = (key, sp)
        zs = sp["median_depth"] * np.float32(1.0 / self.cal.units_per_m); zs[zs <= 0] = 1e9
        far = 0.999 * m.vis.map.zfar * m.stat.extent
        sim = (z < far) & (z < zs + 0.01)
        sim_seg = sim if z_seg is None else (z_seg < far) & (z_seg < zs + 0.01)
        # the shadow passes (two half-resolution renders) every SHADOW_EVERY frames of a camera: soft shadows move little in
        # one frame; between them the last ratio is reused (deterministic: a frame counter per camera)
        if not hasattr(self, "_shadow_state"): self._shadow_state = {}
        # per camera: frame counter, the last rendered ratio, and the scene it was rendered from (r11, D5: a reused
        # ratio must never outlive a reset, a look change or a jump; r10 left the previous episode's arm on the desk)
        n_, rendered, sig_r = self._shadow_state.get(cam, (0, None, None))
        static = m.cam_bodyid[cam] == 0 or m.body_rootid[m.cam_bodyid[cam]] != m.body_rootid[m.body("base_link").id]
        sig = self._shadow_signature()
        reuse = (static and rendered is not None and n_ % self.SHADOW_EVERY_STATIC != 1 % self.SHADOW_EVERY_STATIC
                 and sig_r is not None and self._shadow_reusable(sig_r, sig))
        if reuse:
            ratio = rendered
        else:
            self.sha.update_scene(d, camera=cam, scene_option=self.o_desk); a1 = self.sha.render().astype(np.float32).mean(2)
            key0 = (d.cam_xpos[cam].round(6).tobytes(), d.cam_xmat[cam].round(6).tobytes(), m.light_pos[0].tobytes(), m.light_diffuse[0].tobytes(), float(m.cam_fovy[cam]))
            c0 = getattr(self, "_a0_cache", {}).get(cam)
            light = (m.light_pos[0].tobytes(), m.light_diffuse[0].tobytes(), float(m.cam_fovy[cam]), hash(m.geom_rgba.tobytes()), hash(m.mat_rgba.tobytes()))
            if c0 is not None and c0[0] == key0:
                a0 = c0[1]   # the bare desk under the same light from the same pose (the overview camera)
            elif c0 is not None and len(c0) > 3 and c0[3] == light and c0[4] < self.A0_WARP_FRAMES:
                # r11 (stream rate): a moving camera's bare-desk render carried to this pose over the desk plane. The
                # bare desk is smooth lighting on a plane (no shadow in it: those come from this frame's own pass), so
                # the warp is exact on the desk; re-rendered every A0_WARP_FRAMES frames and whenever the look changes
                hh, ww = c0[1].shape; G = self._desk_homography(cam, hh, ww)
                a0 = cv2.warpPerspective(c0[1], c0[2] @ np.linalg.inv(G), (ww, hh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                self._a0_cache[cam] = (c0[0], c0[1], c0[2], light, c0[4] + 1)
            else:
                self.sha.update_scene(d, camera=cam, scene_option=self.o_bare); a0 = self.sha.render().astype(np.float32).mean(2)
                if not hasattr(self, "_a0_cache"): self._a0_cache = {}
                self._a0_cache[cam] = (key0, a0, self._desk_homography(cam, *a0.shape), light, 0)
            ratio = rendered = np.where(a0 > 20, np.clip(a1 / np.maximum(a0, 1), 0, 1), 1.0).astype(np.float32)
            sig_r = sig
        self._shadow_state[cam] = (n_ + 1, rendered, sig_r)
        # (shading at the shadow renders' half resolution, then one upscale)
        h2, w2 = ratio.shape
        # soft like the office's diffuse ceiling light: a short contact term and a wide penumbra, scaled with the image
        k = max(W, H) / 1280.0
        dark = 0.4 * (1 - cv2.GaussianBlur(ratio, (0, 0), 1.5 * k)) + 0.6 * (1 - cv2.GaussianBlur(ratio, (0, 0), self.BLUR * 0.8 * k))
        # only where the splat shows the desk itself (its depth within 3.5 cm of the sim desk's), feathered
        zd = self._desk_depth(cam)
        zs2 = cv2.resize(zs, (w2, h2), interpolation=cv2.INTER_NEAREST); zd2 = cv2.resize(zd, (w2, h2), interpolation=cv2.INTER_NEAREST)
        contact = self._contact_shadow(cam, w2, h2, zd2)
        on_desk = cv2.GaussianBlur((np.abs(zs2 - zd2) < 0.035).astype(np.float32), (0, 0), 2 * k)
        shade = cv2.resize((1 - self.SHADOW * dark * on_desk).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        # where the splat shows the desk itself, a sim surface in front of the sim desk is in front of it: the splat's
        # desk can stand a few mm proud and would bite into resting objects
        sim |= (z < far) & (np.abs(zs - zd) < 0.015) & (z < zd - 0.001)
        shaded = cv2.multiply(sp["rgb"], cv2.merge([shade, shade, shade]), scale=255.0, dtype=cv2.CV_8U)   # saturating
        # the contact shadows go on after the camera model (render_camera): a neutral darkening of the camera's own
        # output, as measured on the real frames; shading the splat before the per-channel tone curve tints it
        self._contact_full = None if not contact.any() else cv2.resize(contact * on_desk, (W, H), interpolation=cv2.INTER_LINEAR)
        # the sim layer over the splat with its edge feathered over ~1.5 px (a 3 px box on the mask): the CG outlines
        # against the desk and the room are not stair-stepped (MSAA would also blend the segmentation renderer's labels)
        soft = cv2.blur(sim.astype(np.float32), (3, 3))
        edge = (soft > 0.0) & (soft < 1.0)
        comp = shaded.copy(); np.copyto(comp, rgb, where=sim[..., None])
        if self._contact_full is not None:   # not over the sim layer (the objects themselves); feathered with its edge
            self._contact_full *= (1.0 - soft)
        # r11 (DF10 in buckets): contact occlusion on the sim layer itself (bucket floors and walls, their contents, a
        # released object), from the sim depth; it joins the desk's contact map (the larger of the two per pixel)
        occ = self._sim_occlusion(cam, z, sim, far)
        if occ is not None:
            occ_full = cv2.resize(occ, (W, H), interpolation=cv2.INTER_LINEAR) * soft * (self.OCC_STRENGTH / self.CONTACT)
            self._contact_full = occ_full if self._contact_full is None else np.maximum(self._contact_full, occ_full)
        if edge.any():
            a = soft[edge][:, None]
            comp[edge] = (rgb[edge].astype(np.float32) * a + shaded[edge].astype(np.float32) * (1 - a) + 0.5).astype(np.uint8)
        layers = []   # the arm's own Gaussians over the CAD, where they are in front of the scene and the sim
        moving = None
        if self.rs is not None:
            if self.rs_static is not None:   # the base never moves: cached for a camera that has not moved
                ks = (T.tobytes(), K.tobytes(), W, H, near_m)
                cs = getattr(self, "_rs_cache", {}).get(cam)
                if cs is not None and cs[0] == ks:
                    layers.append(cs[1])
                else:
                    rr0 = self.rs_static.render(T, K, W, H, bg=(0.0, 0.0, 0.0), near=near_m * self.cal.units_per_m)
                    if not hasattr(self, "_rs_cache"): self._rs_cache = {}
                    self._rs_cache[cam] = (ks, rr0); layers.append(rr0)
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
                zo = z   # (far pixels are far in both)
            total = np.zeros((H, W), np.float32) if moving is not None else None
            for rr in layers:
                zr = rr["median_depth"] / self.cal.units_per_m
                a = np.clip(rr["alpha"], 0, 1) * ((zr > 0) & (zr < zs + 0.01) & (zr < zo + 0.015))
                px = a > 0.004   # blend only where the layer shows
                if px.any():
                    ap = a[px][:, None]
                    comp[px] = (ap * np.clip(rr["rgb"][px], 0, 1) * 255 + (1 - ap) * comp[px]).astype(np.uint8)
                if total is not None: total = np.maximum(total, a)
            # the CAD's thin rims around the Gaussians and small holes inside them: the scene behind instead (the CAD
            # silhouette is a few px wider than the real arm's, and white at its edges)
            if moving is not None:
                robot_px = sim & (z < zo - 1e-4)
                near = cv2.dilate((total > 0.6).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                rim = robot_px & near & (total < 0.5)
                comp[rim] = shaded[rim]
        if abs(gain - 1.0) > 1e-3:
            comp = np.clip(comp.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        res = {"rgb": comp, "T": T, "K": K}
        if outputs:
            res.update(self._seg_outline(cam, sim_seg, zs))
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
        robotlike = (lab == LABEL_ROBOT) | (lab == LABEL_WRIST) | (lab >= 20)
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
def _sync_arrays(src: mujoco.MjModel) -> dict:
    """What a reset changes on the live model and the compositor's copy must follow: the pool objects' and buckets'
    material colours, the contacts (bucket fill, parking) and the r2s numerics (table look, exposure)."""
    from physicalai_mujoco_plugin.bucket_sort import bucket_colors, scene_objects
    ids = [mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_MATERIAL, n) for n in (*scene_objects(src), *(f"bucket_{c}" for c in bucket_colors(src)))]
    mats = np.array([i for i in ids if i >= 0], dtype=int)
    return {"mats": mats, "mat_rgba": src.mat_rgba[mats].copy(), "contype": src.geom_contype.copy(),
            "conaffinity": src.geom_conaffinity.copy(), "numeric": src.numeric_data.copy()}


def _apply_sync(m: mujoco.MjModel, s: dict) -> None:
    if len(s["mats"]): m.mat_rgba[s["mats"]] = s["mat_rgba"]
    m.geom_contype[:] = s["contype"]; m.geom_conaffinity[:] = s["conaffinity"]
    if m.nnumericdata: m.numeric_data[:] = s["numeric"]


def _worker_main(model_bytes: bytes, kit_path: str, camera: str, w: int, h: int, conn, shm_name: str) -> None:
    """A camera's compositor in a process of its own: builds the model from bytes, then renders each state it is sent
    into the shared frame buffer."""
    import os, tempfile
    from multiprocessing import shared_memory
    fd, path = tempfile.mkstemp(suffix=".mjb"); os.write(fd, model_bytes); os.close(fd)
    m = mujoco.MjModel.from_binary_path(path); os.unlink(path); d = mujoco.MjData(m)
    comp = Compositor(m, d, DeskKit.from_json(kit_path), width=w, height=h)
    shm = shared_memory.SharedMemory(name=shm_name); frame = np.ndarray((h, w, 3), np.uint8, buffer=shm.buf)
    conn.send("ready")
    import sys, time
    timing = os.environ.get("R2S_TIMING"); times = []
    while True:
        msg = conn.recv()
        if msg is None: break
        qpos, sync, sim_t = msg; t0 = time.perf_counter()
        d.qpos[:] = qpos; d.qvel[:] = 0.0; d.time = sim_t; _apply_sync(m, sync)   # time: noise fields follow sim time
        try:
            frame[:] = comp.render_camera(camera)["rgb"]; conn.send("ok")
            if timing:
                times.append(time.perf_counter() - t0)
                if len(times) == 150:
                    t = np.array(times) * 1000; times.clear()
                    print(f"[r2s timing] {camera}: render ms median {np.median(t):.1f} p90 {np.percentile(t, 90):.1f} max {t.max():.1f}", file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            conn.send(f"error: {exc}")
    shm.close()


class _ProcessWorker:
    def __init__(self, kit_path: str, model: mujoco.MjModel, camera: str, size: tuple[int, int]) -> None:
        import multiprocessing as mp, os, tempfile
        from multiprocessing import shared_memory
        w, h = size; self.size = size; self.camera = camera
        fd, path = tempfile.mkstemp(suffix=".mjb"); os.close(fd)
        mujoco.mj_saveModel(model, path, None); model_bytes = open(path, "rb").read(); os.unlink(path)
        self.shm = shared_memory.SharedMemory(create=True, size=w * h * 3)
        self.frame = np.ndarray((h, w, 3), np.uint8, buffer=self.shm.buf)
        ctx = mp.get_context("spawn"); self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(model_bytes, kit_path, camera, w, h, child, self.shm.name), daemon=True,
                                name=f"desk-compositor-{camera}")
        self.proc.start(); self.pending = None; self.done_key = None
        import atexit
        atexit.register(self.close)
        if not self.conn.poll(180) or self.conn.recv() != "ready":
            raise RuntimeError(f"compositor process for {camera} did not start")

    def close(self) -> None:
        """Stop the process and free the shared frame (also at interpreter exit)."""
        if self.shm is None: return
        try:
            self.conn.send(None); self.proc.join(2)
        except (OSError, ValueError):
            pass
        if self.proc.is_alive(): self.proc.terminate()
        self.frame = None; self.shm.close(); self.shm.unlink(); self.shm = None

    # -- pipelined use (several workers per camera, DeskCameraCompositor) ------------------------------------------
    def start_job(self, seq: int, qpos: np.ndarray, sync: dict, sim_t: float) -> None:
        self.conn.send((qpos, sync, sim_t)); self.pending = ("job", seq); self.job_inputs = (qpos, sync, sim_t)

    def poll_done(self, block: bool = False) -> bool:
        """Collect a finished job: ``self.result = (seq, frame copy, inputs)``. Returns whether one finished."""
        if self.pending is None or not self.conn.poll(10 if block else 0):
            if block and self.pending is not None: raise RuntimeError(f"compositor for {self.camera} timed out")
            return False
        msg = self.conn.recv(); seq = self.pending[1]; self.pending = None
        if msg != "ok": raise RuntimeError(f"compositor for {self.camera}: {msg}")
        self.result = (seq, self.frame.copy(), self.job_inputs); return True

    def submit(self, key: bytes, qpos: np.ndarray, sync: dict, sim_t: float = 0.0) -> None:
        if self.pending is not None or self.done_key == key: return
        self.conn.send((qpos, sync, sim_t)); self.pending = key

    def get(self, key: bytes, qpos: np.ndarray, sync: dict, sim_t: float = 0.0) -> np.ndarray:
        if self.pending is not None and self.pending != key:
            self._wait()   # an older job: finish it first
        self.submit(key, qpos, sync, sim_t)
        if self.pending is not None: self._wait()
        return self.frame.copy()

    def _wait(self) -> None:
        if not self.conn.poll(10): raise RuntimeError(f"compositor for {self.camera} timed out")
        msg = self.conn.recv(); self.done_key, self.pending = self.pending, None
        if msg != "ok": raise RuntimeError(f"compositor for {self.camera}: {msg}")


_TIMING = bool(__import__("os").environ.get("R2S_TIMING")); _TIMES: dict = {}
_STAMP = bool(__import__("os").environ.get("R2S_STAMP")); _STAMPS: dict = {}


def _stamp(camera: str, out: np.ndarray) -> np.ndarray:
    """Diagnostics only (R2S_STAMP=1): a frame counter in the top-left 16x8 px, to count stale frames downstream."""
    if not _STAMP: return out
    _STAMPS[camera] = _STAMPS.get(camera, 0) + 1; n = _STAMPS[camera]; out = out.copy()
    for bit in range(8): out[0:8, bit * 2:bit * 2 + 2] = 255 if (n >> bit) & 1 else 0
    return out


def _maybe_grab(camera: str, frame: np.ndarray, qpos: np.ndarray, sync: dict, sim_t: float) -> None:
    """The state-matched grab (rubric S2): with ``R2S_GRAB_DIR`` set, while the file ``<dir>/grab`` exists, the next
    frame of each camera is written exactly as streamed (``<camera>.png``, lossless) with the exact inputs it was
    rendered from (``<camera>_state.npz``: qpos, the synced model arrays, sim time); once every camera has one, the
    trigger is removed. ``bucketsim_v2.py robotcam --state DIR`` renders the same states offline."""
    import os
    root = os.environ.get("R2S_GRAB_DIR")
    if not root or not os.path.exists(os.path.join(root, "grab")):
        return
    path = Path(root)
    if not (path / f"{camera}.png").exists():
        cv2.imwrite(str(path / f"{camera}.png"), np.ascontiguousarray(frame[:, :, ::-1]))
        np.savez(path / f"{camera}_state.npz", qpos=qpos, sim_t=np.float64(sim_t), **{f"sync_{k}": v for k, v in sync.items()})
    wanted = {name for name in os.environ.get("R2S_GRAB_CAMERAS", "overview,wrist").split(",")}
    if all((path / f"{c}.png").exists() for c in wanted):
        try: (path / "grab").unlink()
        except FileNotFoundError: pass


class DeskCameraCompositor:
    """The camera service's compositor for one kit: called with the camera's renderer (and its plain frame, or None),
    it returns the composited frame. Each camera renders in a process of its own (no GIL shared with the simulation, the
    demonstrator or the other camera), on its own copy of the model; a request for one camera starts the others on the
    same state, so the cameras of one tick render at the same time. The live simulation is never written."""

    def __init__(self, kit: DeskKit, model: mujoco.MjModel, kit_path: str | None = None) -> None:
        self.kit, self.source, self.kit_path = kit, model, kit_path
        self._workers: dict[str, _ProcessWorker] = {}
        self._lock = threading.Lock()
        # r14: an int (every camera) or a {camera: count} mapping (e.g. only the costlier wrist pipelined)
        wpc = kit.workers_per_camera
        self._pool_sizes = dict(wpc) if isinstance(wpc, dict) else {}
        self._free_dur: dict = {}; self._free_pub: dict = {}; self._free_last_start: dict = {}; self._free_seen: dict = {}
        self.pool_size = int(wpc or 1) if not isinstance(wpc, dict) else 1
        self._pools: dict[str, list[_ProcessWorker]] = {}; self._seq: dict[str, int] = {}
        self._last_key: dict[str, bytes] = {}; self._returned: dict[str, int] = {}

    def _pipelined(self, camera: str, size: tuple[int, int], key: bytes, qpos: np.ndarray, sync: dict, sim_t: float):
        """Several processes per camera, round robin: each call hands the new state to an idle one and returns the
        newest finished frame (waiting for the oldest running job only when none is newer than the last returned).
        Throughput rises with the pool; a frame is about one call older than the state passed in, like a real
        camera's latency. Returns ``(frame, (qpos, sync, sim_t) it was rendered from)``."""
        pool = self._pools.get(camera)
        if pool is None:
            pool = self._pools[camera] = [_ProcessWorker(self.kit_path, self.source, camera, size) for _ in range(int(self._pool_sizes.get(camera, self.pool_size)))]
            for wk in pool: wk.result = None
        for wk in pool: wk.poll_done()
        if key != self._last_key.get(camera):
            idle = [wk for wk in pool if wk.pending is None]
            if idle:
                self._seq[camera] = self._seq.get(camera, 0) + 1
                idle[0].start_job(self._seq[camera], qpos, sync, sim_t); self._last_key[camera] = key
        last = self._returned.get(camera, 0)
        done = [wk.result for wk in pool if wk.result is not None and wk.result[0] > last]
        if not done:
            running = sorted((wk for wk in pool if wk.pending is not None), key=lambda wk: wk.pending[1])
            if running:
                running[0].poll_done(block=True); done = [running[0].result]
            else:   # nothing new and nothing running (the state has not changed): the last frame again
                done = [max((wk.result for wk in pool if wk.result is not None), key=lambda r: r[0])]
        seq, frame, inputs = max(done, key=lambda r: r[0])
        self._returned[camera] = seq
        return frame, inputs

    def _free_run(self, camera: str, size: tuple[int, int], key: bytes, qpos: np.ndarray, sync: dict, sim_t: float):
        """Processes for the camera, never waited for: collect finished frames, hand an idle process the current state.
        With n > 1 processes (kit ``free_running_workers``, r15) their starts are staggered by the measured render time
        / n, and only a frame newer than the last one returned is returned. Returns ``(frame, inputs)`` for a NEW
        frame, else None."""
        import time as _t
        pool = self._pools.get(camera)
        if pool is None:
            n = int((self.kit.free_running_workers or {}).get(camera, 1))
            pool = self._pools[camera] = [_ProcessWorker(self.kit_path, self.source, camera, size) for _ in range(n)]
            for wk in pool: wk.result = None; wk.t_start = 0.0
            self._free_dur[camera] = 0.0; self._free_pub[camera] = 0; self._free_last_start[camera] = 0.0
        now = _t.perf_counter(); best = None
        for wk in pool:
            if wk.poll_done():
                d = now - wk.t_start
                if d < 0.25:   # a process's first job (its start-up, seconds) would hold the stagger for long
                    self._free_dur[camera] = d if not self._free_dur[camera] else 0.8 * self._free_dur[camera] + 0.2 * d
                if _TIMING: _TIMES.setdefault(("free", camera), []).append(d); _TIMES.setdefault(("free_w", camera), []).append(pool.index(wk))
                if wk.result[0] > self._free_pub[camera] and (best is None or wk.result[0] > best[0]): best = wk.result
        gap = self._free_dur[camera] / len(pool) if len(pool) > 1 else 0.0
        if _TIMING:
            st = _TIMES.setdefault(("free_gate", camera), [0, 0, 0])
            ch = key != self._last_key.get(camera); ok = now - self._free_last_start[camera] >= gap
            st[0] += ch; st[1] += ch and not ok; st[2] += 1; _TIMES.setdefault(("free_gap", camera), []).append(gap)
            if self._free_seen.get(camera) != key: self._free_seen[camera] = key; _TIMES.setdefault(("free_keys", camera), [0])[0] += 1
        refresh = self.kit.free_running_refresh_hz
        stale = refresh is not None and now - self._free_last_start[camera] >= 1.0 / float(refresh)
        if (key != self._last_key.get(camera) or stale) and now - self._free_last_start[camera] >= gap:
            idle = next((wk for wk in pool if wk.pending is None), None)
            if _TIMING: _TIMES.setdefault(("free_try", camera), []).append(-1 if idle is None else pool.index(idle))
            if idle is not None:
                self._seq[camera] = self._seq.get(camera, 0) + 1
                idle.start_job(self._seq[camera], qpos, sync, sim_t); idle.t_start = now; self._last_key[camera] = key
                self._free_last_start[camera] = now
        if best is None:
            return None
        self._free_pub[camera] = best[0]
        seq, frame, inputs = best
        return frame, inputs

    def _look_noise(self, sync: dict) -> bool:
        """Whether the randomised look adds image noise (r2s_image[0] > 0): then frames depend on sim time too."""
        k = mujoco.mj_name2id(self.source, mujoco.mjtObj.mjOBJ_NUMERIC, "r2s_image")
        return k >= 0 and float(sync["numeric"][self.source.numeric_adr[k]]) > 0

    def __call__(self, renderer: object, data: mujoco.MjData, camera: str, frame: np.ndarray | None) -> np.ndarray:
        h, w = (frame.shape[:2] if frame is not None else (renderer.height, renderer.width))
        if _TIMING: _t0 = time.perf_counter()
        qpos = np.array(data.qpos, copy=True); sync = _sync_arrays(self.source)
        if _TIMING: _TIMES.setdefault(("call", camera), []).append(time.perf_counter() - _t0)
        key = qpos.tobytes() + sync["numeric"].tobytes() + sync["mat_rgba"].tobytes() + sync["contype"].tobytes()
        sim_t = float(data.time)
        if self.kit.noise_on() or self._look_noise(sync):
            key += np.float64(sim_t).tobytes()   # a new noise field each tick, even with the arm still
        if self.kit.free_running_cameras:
            with self._lock:
                got = self._free_run(camera, (w, h), key, qpos, sync, sim_t)
            if got is None:
                return None
            out, (q_, s_, t_) = got
            _maybe_grab(camera, out, q_, s_, t_)
            return _stamp(camera, out)
        if int(self._pool_sizes.get(camera, self.pool_size)) > 1:
            with self._lock:
                out, (q_, s_, t_) = self._pipelined(camera, (w, h), key, qpos, sync, sim_t)
                _maybe_grab(camera, out, q_, s_, t_)
                return _stamp(camera, out)
        with self._lock:
            worker = self._workers.get(camera)
            if worker is None:
                worker = self._workers[camera] = _ProcessWorker(self.kit_path, self.source, camera, (w, h))
            others = [wk for name, wk in self._workers.items() if name != camera]
            for wk in others:
                wk.submit(key, qpos, sync, sim_t)   # the other cameras of this tick, at the same time
            t0 = time.perf_counter(); out = worker.get(key, qpos, sync, sim_t)
            if _TIMING:
                _TIMES.setdefault(camera, []).append(time.perf_counter() - t0)
                if len(_TIMES[camera]) == 150:
                    t = np.array(_TIMES.pop(camera)) * 1000
                    print(f"[r2s timing] {camera} call ms median {np.median(t):.1f} p90 {np.percentile(t, 90):.1f} max {t.max():.1f}", file=sys.stderr, flush=True)
            _maybe_grab(camera, out, qpos, sync, sim_t)
            return _stamp(camera, out)


def compositor_factory(kit_path: str | Path):
    """``CameraService.compositor_factory`` for a kit file: every camera of the model gets the shared compositor.

    Returns:
        A callable building ``{camera name: compositor}`` for a model.
    """
    kit = DeskKit.from_json(kit_path)

    def build(model: mujoco.MjModel) -> dict[str, DeskCameraCompositor]:
        shared = DeskCameraCompositor(kit, model, str(Path(kit_path).expanduser().resolve()))
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
