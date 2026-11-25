# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The photoreal kit's splat as the browser viewer's background, composited like the offline renders.

Drawing the Gaussians themselves in the browser mixes them with the sim's meshes in 3D: the splat's
surfaces are fuzzy layers a few centimetres thick (the table top spreads about +-3 cm, the real trays
and the bracket are soft blobs), so they overlap the small task objects and wash them out, and
viser's WebGL splats keep only the view-independent colour. The offline composites never ask the
splat for geometry: they render it as an image and lay the sim over it, and above the table the
splat may never cover the sim.

Here each browser gets the same: whenever its camera moves, the full splat is rendered on Metal
from that camera (view-dependent colour, exact per-tile sorting) and sent as viser's background
image with a depth map, and viser draws only the sim meshes over it. The depth map is the
composite's occlusion rule:

- inside the workspace box over the table, the splat never occludes (its depth is pushed back);
- the static fixtures the viewer leaves to the splat (table, bin, bracket, poles, bar) still hide
  what is behind them: their sim depth is rendered by MuJoCo and wins there (the offline renders'
  holdouts);
- elsewhere the splat's median depth, where it is opaque.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.real2sim.appearance import View, capture_camera_matrices

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from physicalai_mujoco_plugin.real2sim.geometry import Similarity

FAR_M = 160.0
"""Depth that never occludes (viser's fixed-point depth tops out at 167.77 m)."""
WORKSPACE = ((-0.47, 0.65), (-0.52, 0.58), (-0.05, 0.75))
"""Scene-frame box over the table top where the splat never covers the sim (the offline composites' rule)."""
_MUJOCO_FROM_OPENCV = np.diag([1.0, -1.0, -1.0])


class SplatViewerBackground:
    """Per-client splat backgrounds for a viser server."""

    def __init__(
        self,
        server: Any,  # noqa: ANN401 - viser.ViserServer
        model: mujoco.MjModel,
        ply: Path,
        transform: Similarity,
        holdout_bodies: frozenset[str],
        *,
        drawn_bodies: tuple[str, ...] = ("tv_screen",),
        workspace: tuple[tuple[float, float], ...] | None = WORKSPACE,
        depth_margin: float = 0.0,
        max_width: int = 1280,
        moving_scale: float = 0.5,
        settle_s: float = 0.25,
    ) -> None:
        """Load the splat and register the client hooks.

        Args:
            server: The viser server.
            model: The scene (the holdouts' geometry); copied, not modified.
            ply: The trained splat, in the capture frame.
            transform: Capture frame to scene frame.
            holdout_bodies: Bodies the viewer hides because the splat shows them; they still occlude.
            drawn_bodies: Static bodies drawn by MuJoCo into the background instead of the splat (the TV
                screen: it was masked in training, so the splat has junk there; the offline renders draw it too).
            workspace: Scene-frame box where the splat never occludes the sim; ``None`` for none.
            depth_margin: Pushes the splat's occluding depth back by this much (metres): its surfaces stand a
                little proud, and would bite into sim bodies resting on them.
            max_width: Widest background rendered (pixels).
            moving_scale: Resolution factor while the camera is moving.
            settle_s: How long the camera must be still before the full-resolution render.
        """
        from physicalai_mujoco_plugin.real2sim.metal_splat import MetalSplatRenderer  # noqa: PLC0415

        self._splat = MetalSplatRenderer(ply)
        self._transform = transform
        self._workspace = workspace
        self._depth_margin = depth_margin
        self._max_width = max_width
        self._moving_scale = moving_scale
        self._settle_s = settle_s
        self._model = copy.deepcopy(model)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        def geoms_of(names: Iterable[str]) -> np.ndarray:
            bodies = {b for b in range(self._model.nbody) if self._model.body(b).name in set(names)}
            return np.array([self._model.geom_bodyid[g] in bodies for g in range(self._model.ngeom)])

        self._holdout = geoms_of(set(holdout_bodies) | set(drawn_bodies))
        self._drawn = geoms_of(drawn_bodies)
        self._model.vis.global_.offwidth = max(self._model.vis.global_.offwidth, max_width)
        self._model.vis.global_.offheight = max(self._model.vis.global_.offheight, max_width)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        # one thread renders for every browser: the Metal splat and MuJoCo's OpenGL context stay on it
        self._clients: dict[int, dict[str, Any]] = {}
        self._wake = threading.Event()
        server.on_client_connect(self._connect)
        server.on_client_disconnect(self._disconnect)
        threading.Thread(target=self._serve, daemon=True, name="splat-viewer-bg").start()

    def _connect(self, client: Any) -> None:  # noqa: ANN401 - viser.ClientHandle
        state = {"client": client, "stamp": 0.0, "sent": 0.0, "sharp": 0.0}

        @client.camera.on_update
        def _moved(_camera: Any) -> None:  # noqa: ANN401
            state["stamp"] = time.monotonic()
            self._wake.set()

        self._clients[client.client_id] = state

    def _disconnect(self, client: Any) -> None:  # noqa: ANN401 - viser.ClientHandle
        self._clients.pop(client.client_id, None)

    def _serve(self) -> None:
        """Render each moving camera at reduced size, then once at full size when it has settled."""
        while True:
            self._wake.wait(timeout=self._settle_s / 2)
            self._wake.clear()
            now = time.monotonic()
            for state in list(self._clients.values()):
                stamp = state["stamp"]
                if stamp > state["sent"]:
                    state["sent"] = stamp
                    self._send(state["client"], self._moving_scale)
                elif stamp > state["sharp"] and now - stamp >= self._settle_s:
                    state["sharp"] = stamp
                    self._send(state["client"], 1.0)

    def _send(self, client: Any, scale: float) -> None:  # noqa: ANN401
        try:
            camera = client.camera
            width, height = int(camera.image_width), int(camera.image_height)
            if width <= 0 or height <= 0:
                return
            factor = min(1.0, self._max_width / max(width, height)) * scale  # tall windows too: the offscreen buffer is max_width square
            w, h = max(16, round(width * factor)), max(16, round(height * factor))
            rotation = np.asarray(_quat_to_matrix(np.asarray(camera.wxyz, dtype=float)))  # OpenCV camera to world
            rgb, depth = self.render(np.asarray(camera.position, dtype=float), rotation, float(camera.fov), w, h)
            client.scene.set_background_image((rgb * 255).astype(np.uint8), format="jpeg", jpeg_quality=92, depth=depth)
        except Exception as exc:  # noqa: BLE001 - a dropped frame must not end the client's thread
            logger.opt(exception=exc).warning("viewer background frame failed: {}", exc)

    def render(
        self, position: np.ndarray, rotation: np.ndarray, fov_y: float, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The background and its occlusion depth for an OpenCV camera (``rotation``: camera to scene frame).

        Returns:
            ``rgb`` ``(H, W, 3)`` in ``[0, 1]`` and ``depth`` ``(H, W)`` metres along the optical axis.
        """
        fy = height / 2 / np.tan(fov_y / 2)
        view = View(position, rotation @ _MUJOCO_FROM_OPENCV, fy, fy, width / 2, height / 2, width, height)
        world_to_cam, K = capture_camera_matrices(view, self._transform)  # noqa: N806
        sp = self._splat.render(world_to_cam, K, width, height, near=0.01 / self._transform.scale, **self._splat_overrides())
        self._place_camera(position, rotation, fy, height)
        holdout = self._sim_pass(self._holdout, width, height)[1]
        drawn_rgb, drawn = self._sim_pass(self._drawn, width, height, rgb=True)
        depth = sp["median_depth"] * self._transform.scale
        ys, xs = np.mgrid[0:height, 0:width] + 0.5
        cam = np.stack([(xs - width / 2) / fy * depth, (ys - height / 2) / fy * depth, depth], -1)
        points = cam @ rotation.T + position
        inside = np.zeros(depth.shape, dtype=bool)
        if self._workspace is not None:
            inside[:] = True
            for axis, (lo, hi) in enumerate(self._workspace):
                inside &= (points[..., axis] > lo) & (points[..., axis] < hi)
        splat_depth = np.where((sp["alpha"] > 0.5) & (depth > 0) & ~inside, depth + self._depth_margin, FAR_M)
        # the drawn bodies replace the splat wherever no holdout fixture (pole, bar) stands in front of them
        rgb = np.where(((drawn < FAR_M) & (drawn <= holdout + 1e-3))[..., None], drawn_rgb, np.clip(sp["rgb"], 0.0, 1.0))
        return rgb, np.minimum(splat_depth, holdout).astype(np.float32)

    def _splat_overrides(self) -> dict[str, Any]:
        """Per-render ``means``/``quats`` overrides for the splat (subclasses whose splat has moving parts)."""
        return {}

    def refresh(self) -> None:
        """Render every browser's background again (the scene behind the sim changed)."""
        now = time.monotonic()
        for state in list(self._clients.values()):
            state["stamp"] = now
        self._wake.set()

    def _place_camera(self, position: np.ndarray, rotation: np.ndarray, fy: float, height: int) -> None:
        """Move camera 0 of the private model copy to this OpenCV pose and vertical focal length."""
        m = self._model
        m.cam_mode[0], m.cam_bodyid[0] = 0, 0
        m.cam_pos[0] = position
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rotation @ _MUJOCO_FROM_OPENCV).ravel())
        m.cam_quat[0] = quat
        m.cam_sensorsize[0] = 0.0
        m.cam_fovy[0] = np.degrees(2 * np.arctan(height / 2 / fy))
        mujoco.mj_forward(m, self._data)

    def _sim_pass(self, geoms: np.ndarray, width: int, height: int, *, rgb: bool = False) -> tuple[np.ndarray | None, np.ndarray]:
        """MuJoCo's render of ``geoms`` alone from camera 0: RGB in ``[0, 1]`` (if asked) and depth (``FAR_M`` where empty)."""
        from physicalai_mujoco_plugin.real2sim.photoreal import _copy_geom  # noqa: PLC0415

        m = self._model
        if (width, height) not in self._renderers:
            if len(self._renderers) > 4:  # the canvas was resized: drop the old sizes' contexts
                for old in self._renderers.values():
                    old.close()
                self._renderers.clear()
            self._renderers[(width, height)] = mujoco.Renderer(m, height, width)
        r = self._renderers[(width, height)]
        r.update_scene(self._data, camera=0)
        scene = r.scene
        keep = [i for i in range(scene.ngeom) if scene.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM
                and 0 <= scene.geoms[i].objid < m.ngeom and geoms[scene.geoms[i].objid]]
        for dst, src in enumerate(keep):
            if dst != src:
                _copy_geom(scene.geoms[dst], scene.geoms[src])
        scene.ngeom = len(keep)
        colour = r.render().astype(np.float32) / 255.0 if rgb else None
        r.enable_depth_rendering()
        depth = r.render().copy()
        r.disable_depth_rendering()
        return colour, np.where((depth > 0) & (depth < m.stat.extent * m.vis.map.zfar * 0.99), depth, FAR_M)


def _quat_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, wxyz / np.linalg.norm(wxyz))
    return mat.reshape(3, 3)
