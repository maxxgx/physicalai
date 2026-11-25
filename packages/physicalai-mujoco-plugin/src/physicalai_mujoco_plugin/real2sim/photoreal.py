# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Photoreal compositing of a MuJoCo scene over a trained Gaussian splat, on the Mac (Metal).

Per frame, at a supersampled size:

1. MuJoCo renders the full scene (colour, depth, segmentation) and once more with the dynamic
   geoms (arms, manipulated objects) hidden. The ratio of the two on static surfaces is the
   shadow the dynamic geoms cast (differential rendering).
2. The splat renders colour, depth and alpha from the same camera.
3. A pixel shows the sim where a dynamic geom is the nearest sim surface and is nearer than the
   splat's surface (things only the splat has, such as lamps or posters, still occlude); every
   other pixel shows the splat, darkened by the shadow ratio.
4. The composite is downsampled (area filter), which anti-aliases the sim edges against the
   splat without halos.

Sim colours pass through a colour transform (``grade``) fitted to real frames.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import cv2
import mujoco
import numpy as np

from physicalai_mujoco_plugin.real2sim.appearance import camera_view, capture_camera_matrices

if TYPE_CHECKING:
    from collections.abc import Iterable

    from physicalai_mujoco_plugin.real2sim.geometry import Similarity
    from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer

_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


@dataclasses.dataclass
class Grade:
    """Colour transform for sim pixels: ``out = (gain @ rgb_linear + bias) ** (1 / gamma)``, all in [0, 1]."""

    gain: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    bias: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    gamma: float = 1.0

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32) / 255.0
        x = np.clip(x.reshape(-1, 3) @ self.gain.T.astype(np.float32) + self.bias.astype(np.float32), 0, 1)
        if self.gamma != 1.0:
            x = x ** (1.0 / self.gamma)
        return x.reshape(rgb.shape)


@dataclasses.dataclass
class CompositeSettings:
    supersample: int = 2
    shadow_strength: float = 0.8
    """How much of the sim's shadow darkens the splat (the real scene has fill light the sim lacks)."""
    shadow_blur_px: float = 2.0
    """Gaussian blur of the shadow ratio at the output size (soft penumbra)."""
    depth_margin_m: float = 0.02
    """A dynamic surface is shown when it is no more than this behind the splat's depth."""
    noise_sigma: float = 0.0
    """Sensor noise added to the final image (8-bit units)."""
    blur_px: float = 0.0
    """Gaussian blur of the final image (output pixels): the lens and compression softness of a real camera."""
    vignette: float = 0.0
    """Corner darkening: the image is scaled by ``1 - vignette * r**2``, ``r`` = 1 at the corners."""
    env_map: str = ""
    """Near-field cameras: an equirectangular sRGB panorama (Blender's convention, booth frame) drawn wherever the
    sim has no surface, instead of the splat (whose near field streaks from viewpoints the capture never had)."""
    near_field: bool = False
    """Draw every sim surface (fixtures and the textured table too) where it is nearer than the splat: for
    cameras whose viewpoints the capture never covered (wrist, overview), where the splat's near field streaks."""


class BoothCompositor:
    """Composite a MuJoCo scene's dynamic geoms over a splat, with shadows, occlusion and anti-aliasing."""

    def __init__(
        self,
        model: mujoco.MjModel,
        splat: MetalSplatRenderer,
        transform: Similarity,
        *,
        dynamic_bodies: Iterable[str] = ("box", "cover"),
        dynamic_prefixes: Iterable[str] = ("left_", "right_"),
        grade: Grade | None = None,
        settings: CompositeSettings | None = None,
    ) -> None:
        """Classify geoms by body name and keep one renderer per size.

        Args:
            model: The compiled scene.
            splat: The splat renderer (capture frame).
            transform: Capture frame to scene frame.
            dynamic_bodies: Bodies drawn by the sim (with their subtrees).
            dynamic_prefixes: Body-name prefixes drawn by the sim (the arms).
            grade: Colour transform for sim pixels.
            settings: Compositing settings.
        """
        self.model = model
        self.splat = splat
        self.transform = transform
        self.grade = grade or Grade()
        self.settings = settings or CompositeSettings()
        dyn_bodies = set()
        names = set(dynamic_bodies)
        prefixes = tuple(dynamic_prefixes)
        for b in range(model.nbody):
            name = model.body(b).name
            if name in names or name.startswith(prefixes):
                dyn_bodies.add(b)
        for b in range(model.nbody):  # subtrees of dynamic bodies are dynamic
            parent = b
            while parent > 0 and parent not in dyn_bodies:
                parent = model.body_parentid[parent]
            if parent in dyn_bodies:
                dyn_bodies.add(b)
        self.dynamic = np.zeros(model.ngeom, dtype=bool)
        self.dynamic[[g for g in range(model.ngeom) if model.geom_bodyid[g] in dyn_bodies]] = True
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._env = None
        if self.settings.env_map:
            env = cv2.imread(self.settings.env_map)
            self._env = cv2.cvtColor(env, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 if env is not None else None

    def _renderer(self, w: int, h: int) -> mujoco.Renderer:
        if (w, h) not in self._renderers:
            self._renderers[(w, h)] = mujoco.Renderer(self.model, h, w)
        return self._renderers[(w, h)]

    def _passes(self, data: mujoco.MjData, camera: int, w: int, h: int, *, static_pass: bool = True) -> dict[str, np.ndarray]:
        r = self._renderer(w, h)
        r.update_scene(data, camera=camera)
        full = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=camera)
        seg = r.render()[..., :2].copy()
        r.disable_segmentation_rendering()
        r.enable_depth_rendering()
        r.update_scene(data, camera=camera)
        depth = r.render().copy()
        r.disable_depth_rendering()
        if not static_pass:
            return {"full": full, "static": full, "seg": seg, "depth": depth}
        # Static-only pass: drop the dynamic geoms from the scene (they neither draw nor cast shadows).
        r.update_scene(data, camera=camera)
        scene = r.scene
        keep = []
        for i in range(scene.ngeom):
            g = scene.geoms[i]
            if not (g.objtype == _GEOM and 0 <= g.objid < self.model.ngeom and self.dynamic[g.objid]):
                keep.append(i)
        for dst, src in enumerate(keep):
            if dst != src:
                _copy_geom(scene.geoms[dst], scene.geoms[src])
        scene.ngeom = len(keep)
        static = r.render().copy()
        return {"full": full, "static": static, "seg": seg, "depth": depth}

    def _env_background(self, view: object, K: np.ndarray, w: int, h: int) -> dict[str, np.ndarray]:
        """The panorama as a background with no depth (the sim always wins where it has a surface)."""
        xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        cam = np.stack([(xs - K[0, 2]) / K[0, 0], -(ys - K[1, 2]) / K[1, 1], -np.ones_like(xs)], -1)  # MuJoCo camera axes
        d = cam @ view.rotation.T
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        rgb = _equirect_lookup(self._env, d)
        return {"rgb": rgb, "alpha": np.zeros((h, w), np.float32), "depth": np.zeros((h, w), np.float32)}

    def render(self, data: mujoco.MjData, camera: int | str, width: int, height: int) -> np.ndarray:
        """The composite RGB ``uint8`` image of ``camera`` at ``width`` x ``height``.

        Returns:
            The image.
        """
        if isinstance(camera, str):
            camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        s = self.settings
        ss = max(1, min(s.supersample, self.model.vis.global_.offwidth // width, self.model.vis.global_.offheight // height))
        w2, h2 = width * ss, height * ss
        p = self._passes(data, camera, w2, h2, static_pass=not s.near_field)
        view = camera_view(self.model, data, camera, w2, h2)
        world_to_cam, K = capture_camera_matrices(view, self.transform)
        if self._env is not None and s.near_field:
            sp = self._env_background(view, K, w2, h2)
        else:
            sp = self.splat.render(world_to_cam, K, w2, h2)
        splat_rgb = sp["rgb"]
        splat_depth = sp.get("median_depth", sp["depth"]) * self.transform.scale
        splat_alpha = sp["alpha"]

        ids, kinds = p["seg"][..., 0], p["seg"][..., 1]
        valid = (ids >= 0) & (kinds == _GEOM)
        dyn = np.zeros(ids.shape, dtype=bool)
        dyn[valid] = self.dynamic[ids[valid]]
        nearer = (splat_alpha < 0.5) | (splat_depth <= 0) | (p["depth"] <= splat_depth + s.depth_margin_m)
        show_sim = (valid if s.near_field else dyn) & nearer

        # Shadow ratio on static sim surfaces, softened, applied to the splat.
        full = p["full"].astype(np.float32).mean(axis=2)
        static = p["static"].astype(np.float32).mean(axis=2)
        ratio = np.where(valid & ~dyn & ~s.near_field, np.clip((full + 1.0) / (static + 1.0), 0.0, 1.0), 1.0)
        if s.shadow_blur_px > 0:
            ratio = cv2.GaussianBlur(ratio, (0, 0), s.shadow_blur_px * ss)
        shade = 1.0 - s.shadow_strength * (1.0 - ratio)
        background = splat_rgb * shade[..., None]

        sim = self.grade(p["full"])
        out = np.where(show_sim[..., None], sim, background)
        if ss > 1:
            out = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
        if s.blur_px > 0:
            out = cv2.GaussianBlur(out, (0, 0), s.blur_px)
        if s.vignette > 0:
            ys, xs = np.mgrid[0:height, 0:width]
            r2 = ((xs - (width - 1) / 2) ** 2 + (ys - (height - 1) / 2) ** 2) / ((width / 2) ** 2 + (height / 2) ** 2)
            out = out * (1.0 - s.vignette * r2)[..., None]
        out = np.clip(out * 255.0, 0, 255)
        if s.noise_sigma > 0:
            out = out + np.random.default_rng().normal(0, s.noise_sigma, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)


def _equirect_lookup(env: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Sample an equirectangular panorama (Blender's convention: u = 0.5 looks along -x) along unit directions."""
    h, w = env.shape[:2]
    lat = np.arcsin(np.clip(dirs[..., 2], -1, 1))
    lon = np.arctan2(dirs[..., 1], -dirs[..., 0])
    u = ((lon + np.pi) / (2 * np.pi)) * w - 0.5
    v = ((np.pi / 2 - lat) / np.pi) * h - 0.5
    return cv2.remap(env, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def cull_workspace_air(fields: dict[str, np.ndarray], transform: Similarity, box: tuple, keep: list[tuple]) -> dict[str, np.ndarray]:
    """Drop the Gaussians floating in the air over the table (scene frame ``box`` = (x0, x1, y0, y1, z0, z1)),
    except inside ``keep`` boxes (fixtures: (cx, cy, yaw_deg, half_x, half_y, z_max)).

    Returns:
        The PLY fields without those Gaussians.
    """
    p = transform.apply(fields["means"])
    x0, x1, y0, y1, z0, z1 = box
    air = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & (p[:, 2] > z0) & (p[:, 2] < z1)
    for cx, cy, yaw, hx, hy, zmax in keep:
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        d = p[:, :2] - [cx, cy]
        lx, ly = d @ [c, s_], d @ [-s_, c]
        air &= ~((np.abs(lx) < hx) & (np.abs(ly) < hy) & (p[:, 2] < zmax))
    return {k: v[~air] for k, v in fields.items()}


def _copy_geom(dst: object, src: object) -> None:
    """Copy one ``mjvGeom`` over another (field by field; the bindings have no assignment)."""
    for field in (
        "type", "dataid", "objtype", "objid", "category", "matid", "texcoord", "segid",
        "size", "pos", "mat", "rgba", "emission", "specular", "shininess", "reflectance",
        "label", "camdist", "modelrbound", "transparent",
    ):
        try:
            value = getattr(src, field)
        except AttributeError:
            continue
        try:
            if isinstance(value, np.ndarray):
                getattr(dst, field)[...] = value
            else:
                setattr(dst, field, value)
        except (AttributeError, TypeError):
            continue


LIGHT_LINKS = ("link_4", "link_5", "link_6", "carriage_left", "carriage_right", "gripper_left", "gripper_right")
"""The real WidowX AI's lighter grey parts (wrist motor housings, gripper); the rest is black anodised."""


def _link_texture_mask(model: mujoco.MjModel, texture: int, bodies: Iterable[int]) -> np.ndarray:
    """Texels of ``texture`` that the meshes of ``bodies`` map to (their UV triangles, rasterised)."""
    w, h = int(model.tex_width[texture]), int(model.tex_height[texture])
    mask = np.zeros((h, w), np.uint8)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bodies or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        adr, num = model.mesh_texcoordadr[mesh], model.mesh_texcoordnum[mesh]
        if adr < 0 or num == 0:
            continue
        uv = model.mesh_texcoord[adr:adr + num]
        faces = model.mesh_facetexcoord[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
        px = np.stack([uv[:, 0] * w, uv[:, 1] * h], -1)  # texel rows follow v (checked: the wrist and gripper light up)
        tris = np.round(px[faces] * 4).astype(np.int32)  # 2 bits of sub-pixel precision
        cv2.fillPoly(mask, list(tris), 1, lineType=cv2.LINE_8, shift=2)
    return mask.astype(bool)


_LOOKED: dict[int, mujoco.MjModel] = {}
"""Models the look was applied to (by identity): the viewer applies it before the cameras' compositors load."""


def apply_sim_look(model: mujoco.MjModel, look: dict[str, object]) -> None:
    """Give the MuJoCo-rendered parts (arms, fingers, props) the look measured for the photoreal composite.

    MuJoCo shades sRGB values directly, so a linear-light tint ``k`` becomes ``k ** (1 / 2.2)`` on the texture.
    ``look``: ``robot_tint`` (linear, per channel), ``robot_light_gain`` (linear, on :data:`LIGHT_LINKS`'
    texels), ``robot_cap`` (linear albedo ceiling: the lettering's white must not glow), ``robot_specular``,
    ``robot_shininess``, ``finger_rgba`` (sRGB; the fingers have no material), ``body_rgba`` ({body: sRGB rgba}).
    Must run before any renderer for ``model`` exists (textures upload when a context is made). Applying it to the
    same model again does nothing.
    """
    if _LOOKED.get(id(model)) is model:
        return
    _LOOKED[id(model)] = model
    tint = np.asarray(look.get("robot_tint", (1.0, 1.0, 1.0)), dtype=np.float32) ** (1 / 2.2)
    gain = float(look.get("robot_light_gain", 1.0)) ** (1 / 2.2)
    cap = float(look.get("robot_cap", 1.0)) ** (1 / 2.2) * 255.0
    done = set()
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        side = "left_" if body.startswith("left_") else "right_" if body.startswith("right_") else None
        if side is None:
            continue
        mat = int(model.geom_matid[g])
        if mat < 0:
            if body.split("_", 1)[1] in ("gripper_left", "gripper_right") and "finger_rgba" in look and model.geom_group[g] != 3:
                model.geom_rgba[g] = look["finger_rgba"]
            continue
        model.mat_specular[mat] = float(look.get("robot_specular", model.mat_specular[mat]))
        model.mat_shininess[mat] = float(look.get("robot_shininess", model.mat_shininess[mat]))
        tex = int(model.mat_texid[mat][mujoco.mjtTextureRole.mjTEXROLE_RGB])
        if tex < 0 or tex in done:
            continue
        done.add(tex)
        light_bodies = {b for b in range(model.nbody) if model.body(b).name in {side + n for n in LIGHT_LINKS}}
        w, h, c = int(model.tex_width[tex]), int(model.tex_height[tex]), int(model.tex_nchannel[tex])
        adr = int(model.tex_adr[tex])
        img = model.tex_data[adr:adr + w * h * c].reshape(h, w, c).astype(np.float32)
        light = _link_texture_mask(model, tex, light_bodies)
        k = np.where(light[..., None], tint * gain, tint)
        img[..., :3] = np.minimum(img[..., :3] * k, np.maximum(cap, img[..., :3]))
        model.tex_data[adr:adr + w * h * c] = np.clip(img, 0, 255).astype(np.uint8).ravel()
    for body, rgba in dict(look.get("body_rgba", {})).items():  # type: ignore[arg-type]
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        for g in range(model.ngeom):
            if b >= 0 and model.geom_bodyid[g] == b:
                if model.geom_matid[g] >= 0:
                    model.mat_rgba[model.geom_matid[g]] = rgba
                else:
                    model.geom_rgba[g] = rgba


class CameraCompositor:
    """``CameraService`` compositor adapter: ``(renderer, data, camera, frame) -> frame`` via a shared :class:`BoothCompositor`."""

    def __init__(self, booth: BoothCompositor) -> None:
        self._booth = booth

    def __call__(self, renderer: mujoco.Renderer, data: mujoco.MjData, camera: str, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        return self._booth.render(data, camera, width, height)


def load_photoreal_compositors(meta: dict[str, object], model: mujoco.MjModel) -> dict[str, CameraCompositor]:
    """One photoreal compositor per named camera, from ``real2sim.json``'s ``photoreal`` block.

    ``photoreal``: ``{"splat_ply": path, "transform": Similarity dict (capture -> scene frame),
    "cameras": [names] (default: every named camera), "dynamic_bodies": [...], "dynamic_prefixes": [...],
    "grade": {"gain": 3x3, "bias": 3, "gamma": g}, "settings": {CompositeSettings fields},
    "sim_look": {apply_sim_look fields}}``.

    Returns:
        Compositors by camera name.
    """
    from pathlib import Path  # noqa: PLC0415

    from physicalai_mujoco_plugin.real2sim.geometry import Similarity  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer  # noqa: PLC0415

    cfg = dict(meta["photoreal"])  # type: ignore[arg-type]
    if cfg.get("sim_look"):
        apply_sim_look(model, cfg["sim_look"])
    grade = cfg.get("grade") or {}
    booth = BoothCompositor(
        model,
        MetalSplatRenderer(Path(str(cfg["splat_ply"]))),
        Similarity.from_dict(cfg["transform"]),
        dynamic_bodies=cfg.get("dynamic_bodies", ("box", "cover")),
        dynamic_prefixes=cfg.get("dynamic_prefixes", ("left_", "right_")),
        grade=Grade(np.asarray(grade.get("gain", np.eye(3))), np.asarray(grade.get("bias", np.zeros(3))), float(grade.get("gamma", 1.0))),
        settings=CompositeSettings(**cfg.get("settings", {})),
    )
    names = cfg.get("cameras") or [model.camera(i).name for i in range(model.ncam) if model.camera(i).name]
    near = set(cfg.get("near_field_cameras", ()))
    comps = {name: CameraCompositor(booth) for name in names if name not in near}
    if near:
        from physicalai_mujoco_plugin.real2sim.splats import read_ply_fields  # noqa: PLC0415

        nf = cfg.get("near_field", {})
        fields = read_ply_fields(Path(str(cfg["splat_ply"])))
        culled = cull_workspace_air(fields, booth.transform, tuple(nf["air"]), [tuple(k) for k in nf.get("keep", [])])
        near_booth = BoothCompositor(
            model, MetalSplatRenderer(culled), booth.transform,
            dynamic_bodies=cfg.get("dynamic_bodies", ("box", "cover")),
            dynamic_prefixes=cfg.get("dynamic_prefixes", ("left_", "right_")),
            grade=booth.grade, settings=dataclasses.replace(booth.settings, near_field=True, env_map=str(nf.get("env_map", ""))),
        )
        comps |= {name: CameraCompositor(near_booth) for name in names if name in near}
    return comps
