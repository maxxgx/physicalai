# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A 3D Gaussian Splatting renderer for Apple GPUs (Metal through ``torch.mps``), no CUDA needed.

:class:`MetalSplatRenderer` is a tile-based rasteriser in the style of gsplat: one preprocessing
kernel projects every Gaussian (EWA, 0.3 px² low-pass, SH colour, opacity-aware bounding box), a
second writes one ``(tile, depth)`` key per tile each Gaussian touches, ``torch.sort`` orders the
keys, and one threadgroup per 16x16 tile composites its pixels front to back through threadgroup
memory. Besides colour it returns the accumulated alpha and two depths for occlusion against
MuJoCo geometry. :class:`MetalSplatBackground` puts it behind the same interface as
:class:`~physicalai_mujoco_plugin.real2sim.appearance.SplatBackground`.

The maths is gsplat's classic (``antialiased=False``) path, which is the INRIA reference. Deliberate
choices, where those two differ or leave something open:

* Alpha is clamped at 0.99 (INRIA; gsplat uses 0.999). A pixel stops at the first Gaussian that
  would bring its transmittance below 1e-4; that Gaussian is not composited (INRIA's ``<``; gsplat
  uses ``<=``).
* Pixel centres are at ``(j + 0.5, i + 0.5)`` (gsplat, and COLMAP's intrinsics convention; INRIA
  uses integer centres).
* Each Gaussian covers the tiles of gsplat's opacity-aware box (``min(3.33,
  sqrt(2 ln(255 opacity)))`` standard deviations per axis), outside which its alpha is below
  1/255 anyway, so tiling drops nothing. INRIA's 3-sigma circle can drop faint fringes.
* Keys are ``tile << 32 | float32 depth bits`` sorted by ``torch.sort``: Gaussians of exactly equal
  float32 depth in one tile composite in an unspecified order. Depth is the camera-space ``z`` of
  the centre, as in gsplat.
* The returned colour is clamped to ``[0, 1]`` after adding the background (gsplat leaves that to
  the caller; :class:`SplatBackground` clamps too).
* ``depth`` is the expected depth ``sum(w z) / sum(w)`` and ``median_depth`` the ``z`` of the
  Gaussian that brings the transmittance from above 0.5 to 0.5 or below; both are 0 where
  undefined (no Gaussian composited, or the alpha never reaches 0.5).
"""

from __future__ import annotations

import functools
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai_mujoco_plugin.real2sim._metal_splat_kernels import SOURCE, TILE
from physicalai_mujoco_plugin.real2sim.appearance import (
    SplatBackground,
    capture_camera_matrices,
    object_members,
)
from physicalai_mujoco_plugin.real2sim.splats import read_ply_fields

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.real2sim.appearance import View
    from physicalai_mujoco_plugin.real2sim.geometry import Similarity

_Tensor = Any
"""A torch tensor (torch is imported lazily, like in ``appearance``)."""


@functools.cache
def _library() -> Any:  # noqa: ANN401 - torch._C._mps_ShaderLibrary
    """The compiled kernels (compiled once per process).

    Returns:
        The shader library.
    """
    import torch  # noqa: PLC0415

    return torch.mps.compile_shader(SOURCE)


def mps_available() -> bool:
    """Whether torch is installed and an MPS (Metal) device is usable.

    Returns:
        ``True`` when the renderer can run.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


@dataclass(frozen=True)
class _Frame:
    """Image and tile-grid size of one render."""

    width: int
    height: int
    tiles_x: int
    tiles_y: int


@dataclass(frozen=True)
class _Projected:
    """Per-Gaussian output of the preprocessing kernel (tensors on the GPU)."""

    xy: _Tensor
    """``(N, 4)``: pixel x, pixel y, camera depth, opacity."""
    conic: _Tensor
    """``(N, 4)``: inverse 2D covariance ``a, b, c`` and padding."""
    rgb: _Tensor
    """``(N, 4)``: SH colour and padding."""
    rect: _Tensor
    """``(N, 4)`` int32: tile box ``x0, y0, x1, y1`` (end exclusive); empty when culled."""
    offsets: _Tensor
    """``(N,)`` int32: inclusive prefix sum of the tile counts."""
    total: int
    """Number of tile-Gaussian pairs."""


class _Laps:
    """Per-stage wall times when profiling (synchronising the GPU at each lap); a no-op otherwise."""

    def __init__(self, torch: Any | None) -> None:  # noqa: ANN401 - the torch module
        self._torch = torch
        self._stamp = time.perf_counter()
        self.timings: dict[str, float] = {}

    def lap(self, name: str) -> None:
        if self._torch is None:
            return
        self._torch.mps.synchronize()
        now = time.perf_counter()
        self.timings[name] = (now - self._stamp) * 1e3
        self._stamp = now


def _camera_params(world_to_cam: np.ndarray, K: np.ndarray, near: float) -> list[float]:  # noqa: N803
    """The preprocessing kernel's ``cam`` argument.

    Returns:
        Row-major rotation, translation, ``fx fy cx cy``, the near plane and the camera centre.
    """
    world_to_cam = np.asarray(world_to_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)  # noqa: N806
    rotation, translation = world_to_cam[:3, :3], world_to_cam[:3, 3]
    center = -rotation.T @ translation
    intrinsics = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
    return [*rotation.reshape(-1).tolist(), *translation.tolist(), *intrinsics, float(near), *center.tolist()]


class MetalSplatRenderer:
    """A trained 3DGS splat (INRIA ``.ply``) kept on the GPU and rendered with Metal kernels."""

    def __init__(self, ply_path: Path | str | Mapping[str, np.ndarray], device: str = "mps") -> None:
        """Load the splat onto ``device`` once.

        Args:
            ply_path: The ``.ply``, or fields already read by :func:`read_ply_fields`.
            device: Torch device; only ``"mps"`` is supported.

        Raises:
            RuntimeError: If MPS is not available.
        """
        import torch  # noqa: PLC0415

        if not str(device).startswith("mps") or not torch.backends.mps.is_available():
            msg = f"MetalSplatRenderer needs an available MPS device, got {device!r}"
            raise RuntimeError(msg)
        fields = ply_path if isinstance(ply_path, Mapping) else read_ply_fields(Path(ply_path))
        self._torch = torch
        self.device = torch.device(device)
        tensor = functools.partial(torch.as_tensor, dtype=torch.float32, device=self.device)
        self.means = tensor(np.ascontiguousarray(fields["means"])).contiguous()
        self.quats = tensor(np.ascontiguousarray(fields["quats"])).contiguous()
        self.scales = tensor(np.exp(fields["log_scales"])).contiguous()
        self.opacities = tensor(1.0 / (1.0 + np.exp(-fields["opacity_logits"]))).contiguous()
        sh = np.asarray(fields["sh"])
        self.sh = tensor(np.ascontiguousarray(sh)).contiguous()
        self.sh_coeffs = sh.shape[1]
        self.sh_degree = min(3, round(np.sqrt(self.sh_coeffs)) - 1)
        self.profile = False
        """When set, :meth:`render` synchronises between stages and fills :attr:`timings`."""
        self.timings: dict[str, float] = {}
        """Milliseconds per stage of the last profiled render (``preprocess``, ``sort``, ``raster``, ``readback``)."""
        self.last_intersections = 0
        """Tile-Gaussian pairs of the last render."""
        self._lib = _library()

    def __len__(self) -> int:
        """Number of Gaussians.

        Returns:
            The count.
        """
        return int(self.means.shape[0])

    def _device_array(self, value: np.ndarray | _Tensor, columns: int) -> _Tensor:
        torch = self._torch
        out = torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(-1, columns).contiguous()
        if out.shape[0] != len(self):
            msg = f"override has {out.shape[0]} rows, the splat {len(self)} Gaussians"
            raise ValueError(msg)
        return out

    def render(
        self,
        world_to_cam: np.ndarray,
        K: np.ndarray,  # noqa: N803 - the usual name of the intrinsics
        width: int,
        height: int,
        *,
        bg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        means: np.ndarray | _Tensor | None = None,
        quats: np.ndarray | _Tensor | None = None,
        near: float = 0.01,
        median_depth: bool = True,
        _defer: bool = False,
    ) -> dict[str, np.ndarray]:
        """Render one pinhole view.

        Args:
            world_to_cam: 4x4 OpenCV world-to-camera matrix (+x right, +y down, +z forward), in the PLY's frame.
            K: 3x3 pixel intrinsics.
            width: Image width in pixels.
            height: Image height in pixels.
            bg: Background colour composited behind the splat.
            means: Optional ``(N, 3)`` centres replacing the loaded ones (numpy or a torch tensor).
            quats: Optional ``(N, 4)`` w-first quaternions replacing the loaded ones.
            near: Near plane in scene units; Gaussians whose centre is closer are dropped.
            median_depth: Also return ``median_depth`` (it costs nothing on the GPU, only a copy).

        Returns:
            ``rgb`` ``(H, W, 3)`` in ``[0, 1]``, ``alpha`` ``(H, W)``, ``depth`` ``(H, W)`` (expected
            camera ``z``; 0 where nothing was composited) and ``median_depth`` ``(H, W)``; all float32.
        """
        width, height = int(width), int(height)
        frame = _Frame(width, height, (width + TILE - 1) // TILE, (height + TILE - 1) // TILE)
        laps = _Laps(self._torch if self.profile else None)

        means_t = self.means if means is None else self._device_array(means, 3)
        quats_t = self.quats if quats is None else self._device_array(quats, 4)
        projected = self._project(means_t, quats_t, _camera_params(world_to_cam, K, near), frame)
        laps.lap("preprocess")
        ranges, ids = self._sort(projected, frame)
        laps.lap("sort")
        out = self._torch.empty(width * height * 6, dtype=self._torch.float32, device=self.device)
        self._lib.rasterize(
            ranges,
            ids,
            projected.xy,
            projected.conic,
            projected.rgb,
            out,
            [float(c) for c in bg],
            [width, height, frame.tiles_x],
            threads=(frame.tiles_x * TILE, frame.tiles_y * TILE),
            group_size=(TILE, TILE),
        )
        laps.lap("raster")
        handle = (out, width, height, median_depth, laps)
        return handle if _defer else self.collect(handle)

    def submit(self, *args, **kwargs) -> tuple:
        """:meth:`render` up to the GPU work queued, without waiting for it (r12): the caller does other work, then
        :meth:`collect`. The result is the same as :meth:`render`'s (the same computation, collected later)."""
        kwargs["_defer"] = True
        return self.render(*args, **kwargs)

    def collect(self, handle: tuple) -> dict[str, np.ndarray]:
        """The rendered view of a :meth:`submit` (waits for the GPU)."""
        out, width, height, median_depth, laps = handle
        hw = width * height
        host = (out if median_depth else out[: 5 * hw]).cpu().numpy()
        laps.lap("readback")
        if self.profile:
            self.timings = laps.timings

        result = {
            "rgb": host[: 3 * hw].reshape(height, width, 3),
            "alpha": host[3 * hw : 4 * hw].reshape(height, width),
            "depth": host[4 * hw : 5 * hw].reshape(height, width),
        }
        if median_depth:
            result["median_depth"] = host[5 * hw :].reshape(height, width)
        return result

    def _project(self, means: _Tensor, quats: _Tensor, cam: list[float], frame: _Frame) -> _Projected:
        """Project every Gaussian and count the tiles its box touches (one kernel, then a prefix sum).

        Returns:
            The per-Gaussian screen-space data.
        """
        torch = self._torch
        n = len(self)
        xy, conic, rgb = (torch.empty((n, 4), dtype=torch.float32, device=self.device) for _ in range(3))
        rect = torch.empty((n, 4), dtype=torch.int32, device=self.device)
        count = torch.empty(n, dtype=torch.int32, device=self.device)
        if n == 0:
            return _Projected(xy, conic, rgb, rect, count, 0)
        self._lib.preprocess(
            means,
            quats,
            self.scales,
            self.opacities,
            self.sh,
            cam,
            [n, self.sh_coeffs, self.sh_degree, frame.width, frame.height, frame.tiles_x, frame.tiles_y],
            xy,
            conic,
            rgb,
            rect,
            count,
            threads=n,
        )
        offsets = torch.cumsum(count, 0, dtype=torch.int32)
        total = int(offsets[-1])  # the one mid-frame sync: the key buffers need their size
        self.last_intersections = total
        return _Projected(xy, conic, rgb, rect, offsets, total)

    def _sort(self, projected: _Projected, frame: _Frame) -> tuple[_Tensor, _Tensor]:
        """Sort the (tile, depth) keys and find each tile's run.

        Returns:
            ``(tiles, 2)`` int32 ``[start, end)`` ranges into the sorted Gaussian ids, and the ids.
        """
        torch = self._torch
        total = projected.total
        ranges = torch.zeros((frame.tiles_x * frame.tiles_y, 2), dtype=torch.int32, device=self.device)
        if not total:
            return ranges, torch.zeros(1, dtype=torch.int32, device=self.device)
        keys = torch.empty(total, dtype=torch.int64, device=self.device)
        values = torch.empty(total, dtype=torch.int32, device=self.device)
        self._lib.emit_keys(
            projected.rect, projected.offsets, projected.xy, keys, values, [len(self), frame.tiles_x], threads=len(self)
        )
        keys, order = torch.sort(keys)
        self._lib.tile_ranges(keys, ranges, [total], threads=total)
        return ranges, values[order]


class MetalSplatBackground:
    """:class:`~physicalai_mujoco_plugin.real2sim.appearance.SplatBackground` on Metal instead of gsplat/CUDA.

    Same constructor and :meth:`frame`; moved objects carry their Gaussians along exactly as there
    (the posing code is :class:`SplatBackground`'s own).
    """

    moves_objects = True

    def __init__(
        self,
        ply: Path,
        transform: Similarity,
        *,
        objects: list[dict[str, object]] | None = None,
        device: str = "mps",
    ) -> None:
        """Load ``ply`` onto ``device``.

        Args:
            ply: The trained splat, in the capture frame.
            transform: Capture frame to scene frame.
            objects: ``real2sim.json``'s objects (name, centre, size): the Gaussians inside each
                one's box move with its body.
            device: Torch device (MPS).

        Raises:
            RuntimeError: If no MPS device is available.
        """
        import torch  # noqa: PLC0415

        if not torch.backends.mps.is_available():
            msg = "no MPS device"
            raise RuntimeError(msg)
        fields = read_ply_fields(Path(ply))
        self._renderer = MetalSplatRenderer(fields, device=device)
        # The attributes SplatBackground._posed reads (see frame).
        self._torch = torch
        self._device = self._renderer.device
        self._transform = transform
        self._means = self._renderer.means
        self._quats = self._renderer.quats
        scene_means = transform.apply(fields["means"])
        self._members: dict[str, tuple[np.ndarray, _Tensor]] = {}
        for obj in objects or []:
            inside = object_members(scene_means, np.asarray(obj["center"]), np.asarray(obj["size"]))
            index = torch.as_tensor(np.flatnonzero(inside), device=self._device)
            self._members[str(obj["name"])] = (np.asarray(obj["center"], dtype=float), index)

    @property
    def renderer(self) -> MetalSplatRenderer:
        """The underlying renderer (for alpha and depth, or profiling).

        Returns:
            The renderer.
        """
        return self._renderer

    def frame(self, view: View) -> np.ndarray:
        """Render the splat from ``view``.

        Returns:
            An RGB ``uint8`` image of the view's size.
        """
        world_to_cam, intrinsics = capture_camera_matrices(view, self._transform)
        # SplatBackground's own posing code, run on this object: it reads only the attributes set in __init__.
        means, quats = SplatBackground._posed(self, view)  # type: ignore[arg-type]  # noqa: SLF001
        result = self._renderer.render(
            world_to_cam,
            intrinsics,
            view.width,
            view.height,
            means=None if means is self._means else means,
            quats=None if quats is self._quats else quats,
            near=0.01 / self._transform.scale,
            median_depth=False,
        )
        # Truncate like SplatBackground's ``clamp(0, 1).mul(255).byte()``.
        return (result["rgb"] * 255).astype(np.uint8)
