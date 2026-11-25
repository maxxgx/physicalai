# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The Metal splat renderer against a slow, exact reference written here from the INRIA/gsplat maths.

The reference projects every Gaussian in float64 (no tiles, no bounding boxes) and composites each
pixel over the depth-sorted list. Synthetic scenes always run where MPS is available; the trained
splat of the conference-room capture is used when it is on disk (``REAL_PLY``).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from physicalai_mujoco_plugin.real2sim.splats import quats_to_matrices

torch = pytest.importorskip("torch")
metal_splat = pytest.importorskip("physicalai_mujoco_plugin.real2sim.metal_splat")

pytestmark = pytest.mark.skipif(not metal_splat.mps_available(), reason="needs an MPS (Metal) device")

REAL_PLY = Path.home() / "real2sim-data/conf/splat/point_cloud_29999.ply"
REAL_SPARSE = Path.home() / "real2sim-data/conf/cap/sfm3/undistorted/sparse"

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
ALPHA_MIN, ALPHA_MAX, T_MIN, EPS2D = 1.0 / 255.0, 0.99, 1e-4, 0.3


# --------------------------------------------------------------------------------------------------
# Reference renderer
# --------------------------------------------------------------------------------------------------


def reference_sh_colors(sh: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """INRIA ``computeColorFromSH``: SH ``(N, K, 3)`` in unit directions ``(N, 3)``, +0.5, clamped at 0.

    Returns:
        ``(N, 3)`` colours.
    """
    degree = min(3, round(math.sqrt(sh.shape[1])) - 1)
    x, y, z = (dirs[:, k : k + 1] for k in range(3))
    out = SH_C0 * sh[:, 0]
    if degree >= 1:
        out = out - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
    if degree >= 2:  # noqa: PLR2004
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out = (
            out
            + SH_C2[0] * xy * sh[:, 4]
            + SH_C2[1] * yz * sh[:, 5]
            + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6]
            + SH_C2[3] * xz * sh[:, 7]
            + SH_C2[4] * (xx - yy) * sh[:, 8]
        )
    if degree >= 3:  # noqa: PLR2004
        out = (
            out
            + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9]
            + SH_C3[1] * xy * z * sh[:, 10]
            + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
            + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
            + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
            + SH_C3[5] * z * (xx - yy) * sh[:, 14]
            + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15]
        )
    return np.maximum(out + 0.5, 0.0)


def reference_project(
    means: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    sh: np.ndarray,
    world_to_cam: np.ndarray,
    K: np.ndarray,  # noqa: N803
    width: int,
    height: int,
    near: float,
) -> dict[str, np.ndarray]:
    """gsplat's classic EWA projection in float64, for every Gaussian in front of the near plane.

    Returns:
        ``index``, ``mean2d``, ``conic`` (a, b, c of the inverse 2D covariance), ``depth``,
        ``opacity`` and ``rgb`` of the kept Gaussians.
    """
    rotation, translation = world_to_cam[:3, :3], world_to_cam[:3, 3]
    pc = means @ rotation.T + translation
    z = pc[:, 2]
    keep = (z >= near) & (z <= 1e10) & (opacities >= ALPHA_MIN)
    m = quats_to_matrices(quats) * scales[:, None, :]
    cov = m @ np.swapaxes(m, 1, 2)
    covc = rotation @ cov @ rotation.T
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tan_x, tan_y = 0.5 * width / fx, 0.5 * height / fy
    zs = np.where(keep, z, 1.0)
    tx = zs * np.clip(pc[:, 0] / zs, -(cx / fx + 0.3 * tan_x), (width - cx) / fx + 0.3 * tan_x)
    ty = zs * np.clip(pc[:, 1] / zs, -(cy / fy + 0.3 * tan_y), (height - cy) / fy + 0.3 * tan_y)
    jac = np.zeros((len(means), 2, 3))
    jac[:, 0, 0] = fx / zs
    jac[:, 0, 2] = -fx * tx / zs**2
    jac[:, 1, 1] = fy / zs
    jac[:, 1, 2] = -fy * ty / zs**2
    cov2 = jac @ covc @ np.swapaxes(jac, 1, 2)
    a, b, c = cov2[:, 0, 0] + EPS2D, cov2[:, 0, 1], cov2[:, 1, 1] + EPS2D
    det = a * c - b * b
    keep &= det > 0
    index = np.flatnonzero(keep)
    det = det[index]
    center = -rotation.T @ translation
    dirs = means[index] - center
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return {
        "index": index,
        "mean2d": np.column_stack([fx * pc[index, 0] / z[index] + cx, fy * pc[index, 1] / z[index] + cy]),
        "conic": np.column_stack([c[index] / det, -b[index] / det, a[index] / det]),
        "depth": z[index],
        "opacity": opacities[index],
        "rgb": reference_sh_colors(sh[index], dirs),
    }


def reference_render(
    means: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    sh: np.ndarray,
    world_to_cam: np.ndarray,
    K: np.ndarray,  # noqa: N803
    width: int,
    height: int,
    *,
    bg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near: float = 0.01,
    block: int = 8,
) -> dict[str, np.ndarray]:
    """Composite every pixel front to back over all Gaussians, in float64.

    Per pixel, along the depth-sorted list: ``alpha = min(0.99, opacity exp(-power))``; skip when
    ``power < 0`` or ``alpha < 1/255``; stop (without that Gaussian) when the transmittance would
    fall below 1e-4. Written as a cumulative product over the sorted list, which is the same
    sequence of operations. Per ``block`` x ``block`` pixels it drops only Gaussians whose alpha is
    provably below 1/255 at every pixel centre of the block (smallest conic eigenvalue times the
    distance to the block), so the result is exact.

    Returns:
        ``rgb``, ``alpha``, ``depth`` (expected) and ``median_depth``, as the fast renderer.
    """
    g = reference_project(means, quats, scales, opacities, sh, world_to_cam, K, width, height, near)
    order = np.argsort(g["depth"], kind="stable")
    mean2d, conic, depth = g["mean2d"][order], g["conic"][order], g["depth"][order]
    opacity, colors = g["opacity"][order], g["rgb"][order]
    ca, cb, cc = conic.T
    eig_min = 0.5 * (ca + cc) - np.sqrt((0.5 * (ca - cc)) ** 2 + cb**2)
    bg_arr = np.asarray(bg, dtype=np.float64)
    rgb = np.zeros((height, width, 3))
    alpha = np.zeros((height, width))
    exp_depth = np.zeros((height, width))
    med_depth = np.zeros((height, width))
    for y0 in range(0, height, block):
        for x0 in range(0, width, block):
            y1, x1 = min(y0 + block, height), min(x0 + block, width)
            # Distance from each centre to the block's rectangle of pixel centres.
            ddx = np.maximum(0.0, np.maximum(x0 + 0.5 - mean2d[:, 0], mean2d[:, 0] - (x1 - 0.5)))
            ddy = np.maximum(0.0, np.maximum(y0 + 0.5 - mean2d[:, 1], mean2d[:, 1] - (y1 - 0.5)))
            bound = opacity * np.exp(-0.5 * np.maximum(eig_min, 0.0) * (ddx**2 + ddy**2))
            sel = np.flatnonzero(bound >= ALPHA_MIN * (1 - 1e-6))
            ys, xs = np.mgrid[y0:y1, x0:x1]
            px, py = xs.ravel() + 0.5, ys.ravel() + 0.5
            if len(sel) == 0:
                rgb[y0:y1, x0:x1] = bg_arr
                continue
            dx = mean2d[sel, 0][None, :] - px[:, None]
            dy = mean2d[sel, 1][None, :] - py[:, None]
            power = 0.5 * (ca[sel] * dx * dx + cc[sel] * dy * dy) + cb[sel] * dx * dy
            a = np.minimum(ALPHA_MAX, opacity[sel] * np.exp(-power))
            a = np.where((power >= 0) & (a >= ALPHA_MIN), a, 0.0)
            t_after = np.cumprod(1.0 - a, axis=1)
            included = t_after >= T_MIN  # T never rises, so this is a prefix: the stop rule
            t_before = np.concatenate([np.ones((len(px), 1)), t_after[:, :-1]], axis=1)
            w = np.where(included, a * t_before, 0.0)
            k = included.sum(axis=1)
            t_final = np.where(k > 0, t_after[np.arange(len(px)), np.maximum(k - 1, 0)], 1.0)
            wsum = w.sum(axis=1)
            color = w @ colors[sel] + t_final[:, None] * bg_arr
            cross = included & (t_before > 0.5) & (t_after <= 0.5)  # noqa: PLR2004
            med = np.where(cross.any(axis=1), depth[sel][np.argmax(cross, axis=1)], 0.0)
            shape = (y1 - y0, x1 - x0)
            rgb[y0:y1, x0:x1] = np.clip(color, 0.0, 1.0).reshape(*shape, 3)
            alpha[y0:y1, x0:x1] = (1.0 - t_final).reshape(shape)
            exp_depth[y0:y1, x0:x1] = np.where(wsum > 0, (w @ depth[sel]) / np.maximum(wsum, 1e-300), 0.0).reshape(
                shape
            )
            med_depth[y0:y1, x0:x1] = med.reshape(shape)
    return {"rgb": rgb, "alpha": alpha, "depth": exp_depth, "median_depth": med_depth}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio of two images in ``[0, 1]``.

    Returns:
        dB (infinite when equal).
    """
    mse = float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(1.0 / mse)


def fields_to_arrays(fields: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Activated float64 parameters of PLY fields, rounded through float32 like the GPU copy.

    Returns:
        ``means, quats, scales, opacities, sh``.
    """
    f32 = lambda x: np.asarray(x, dtype=np.float32).astype(np.float64)  # noqa: E731
    return (
        f32(fields["means"]),
        f32(fields["quats"]),
        f32(np.exp(fields["log_scales"])),
        f32(1.0 / (1.0 + np.exp(-fields["opacity_logits"]))),
        f32(fields["sh"]),
    )


def synthetic_fields(count: int, seed: int, degree: int = 3) -> dict[str, np.ndarray]:
    """Random Gaussians in front of a camera at the origin looking along +z (PLY field layout).

    Returns:
        Fields as :func:`read_ply_fields` returns them.
    """
    rng = np.random.default_rng(seed)
    means = np.column_stack([rng.uniform(-1.2, 1.2, count), rng.uniform(-0.9, 0.9, count), rng.uniform(0.5, 4, count)])
    means[: count // 20, 2] = rng.uniform(-1, 0.005, count // 20)  # some behind the camera or the near plane
    quats = rng.normal(size=(count, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    log_scales = rng.uniform(np.log(0.004), np.log(0.15), (count, 3))
    log_scales[: count // 10, 0] = np.log(0.0005)  # needles and discs
    log_scales[count // 10 : count // 8] = np.log(0.5)  # a few large ones covering many tiles
    sh = rng.normal(scale=0.4, size=(count, (degree + 1) ** 2, 3))
    sh[:, 0] = rng.uniform(-1.5, 1.5, (count, 3))
    return {
        "means": means,
        "quats": quats,
        "log_scales": log_scales,
        "opacity_logits": rng.uniform(-6, 5, count),
        "sh": sh,
    }


def _look(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV world-to-camera matrix at ``position`` looking at ``target`` (world +y down-ish).

    Returns:
        4x4 matrix.
    """
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, -1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = -rotation @ position
    return out


def _compare(fast: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict[str, float]:
    alpha_err = float(np.abs(fast["alpha"] - ref["alpha"]).max())
    covered = (ref["alpha"] > 0.05) & (fast["alpha"] > 0.05)  # noqa: PLR2004
    depth_rel = np.abs(fast["depth"] - ref["depth"])[covered] / np.maximum(ref["depth"][covered], 1e-9)
    median_same = np.isclose(fast["median_depth"], ref["median_depth"], rtol=1e-4, atol=1e-6)
    return {
        "psnr": psnr(fast["rgb"], ref["rgb"]),
        "alpha_max_err": alpha_err,
        "depth_rel_p99": float(np.quantile(depth_rel, 0.99)) if depth_rel.size else 0.0,
        "median_agree": float(median_same.mean()),
    }


def _check(stats: dict[str, float]) -> None:
    assert stats["psnr"] >= 45.0, stats
    assert stats["alpha_max_err"] < 0.02, stats  # noqa: PLR2004
    assert stats["depth_rel_p99"] < 1e-3, stats  # noqa: PLR2004
    assert stats["median_agree"] > 0.995, stats  # noqa: PLR2004


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "degree", "size"), [(0, 3, (96, 72)), (1, 1, (37, 53)), (2, 0, (64, 64))])
def test_matches_reference_on_synthetic_scenes(seed: int, degree: int, size: tuple[int, int]) -> None:
    fields = synthetic_fields(400, seed, degree)
    width, height = size
    world_to_cam = _look(np.array([0.1, -0.05, -0.2]), np.array([0.0, 0.0, 2.0]))
    K = np.array([[0.9 * width, 0, width / 2 + 1.3], [0, 0.9 * width, height / 2 - 0.7], [0, 0, 1]])  # noqa: N806
    bg = (0.2, 0.5, 0.9)

    fast = metal_splat.MetalSplatRenderer(fields).render(world_to_cam, K, width, height, bg=bg)
    ref = reference_render(*fields_to_arrays(fields), world_to_cam, K, width, height, bg=bg)

    assert fast["rgb"].shape == (height, width, 3)
    assert fast["rgb"].dtype == np.float32
    assert fast["alpha"].shape == fast["depth"].shape == (height, width)
    assert 0.2 < fast["alpha"].mean() < 0.999  # noqa: PLR2004 - a scene with both coverage and gaps
    _check(_compare(fast, ref))


def test_means_and_quats_override_moves_gaussians() -> None:
    fields = synthetic_fields(300, 3)
    renderer = metal_splat.MetalSplatRenderer(fields)
    width, height = 80, 60
    world_to_cam = _look(np.array([0.0, 0.0, -0.5]), np.array([0.0, 0.0, 2.0]))
    K = np.array([[70.0, 0, 40], [0, 70.0, 30], [0, 0, 1]])  # noqa: N806
    moved = {k: np.array(v, copy=True) for k, v in fields.items()}
    subset = np.arange(0, 300, 3)
    moved["means"][subset] += [0.3, -0.1, 0.2]
    turn = np.array([math.cos(0.4), 0, 0, math.sin(0.4)])  # w-first, about z
    w, x, y, z = moved["quats"][subset].T
    aw, _, _, az = turn
    moved["quats"][subset] = np.column_stack([aw * w - az * z, aw * x - az * y, aw * y + az * x, aw * z + az * w])

    overridden = renderer.render(
        world_to_cam,
        K,
        width,
        height,
        means=torch.as_tensor(moved["means"], dtype=torch.float32, device="mps"),
        quats=moved["quats"],
    )
    reloaded = metal_splat.MetalSplatRenderer(moved).render(world_to_cam, K, width, height)
    original = renderer.render(world_to_cam, K, width, height)

    np.testing.assert_allclose(overridden["rgb"], reloaded["rgb"], atol=1e-6)
    np.testing.assert_allclose(overridden["depth"], reloaded["depth"], atol=1e-6)
    assert psnr(original["rgb"], overridden["rgb"]) < 40  # noqa: PLR2004 - the move is visible
    with pytest.raises(ValueError, match="rows"):
        renderer.render(world_to_cam, K, width, height, means=np.zeros((5, 3)))


def test_empty_view_is_background() -> None:
    fields = synthetic_fields(50, 4)
    renderer = metal_splat.MetalSplatRenderer(fields)
    behind = _look(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 20.0]))
    out = renderer.render(behind, np.array([[50.0, 0, 20], [0, 50.0, 15], [0, 0, 1]]), 40, 30, bg=(1.0, 0.0, 0.5))
    np.testing.assert_array_equal(out["rgb"], np.broadcast_to(np.float32([1.0, 0.0, 0.5]), (30, 40, 3)))
    assert not out["alpha"].any()
    assert not out["depth"].any()


@pytest.fixture(scope="module")
def real_scene() -> tuple[dict[str, np.ndarray], object]:
    if not REAL_PLY.exists() or not REAL_SPARSE.exists():
        pytest.skip("the conference-room splat is not on this machine")
    from physicalai_mujoco_plugin.real2sim.colmap import read_model  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.splats import read_ply_fields  # noqa: PLC0415

    return read_ply_fields(REAL_PLY), read_model(REAL_SPARSE)


def _real_view(model: object, index: int, scale: float, crop: tuple[int, int, int, int] | None = None) -> tuple:
    images = sorted(model.images.values(), key=lambda image: image.name)  # type: ignore[attr-defined]
    image = images[index % len(images)]
    camera = model.cameras[image.camera_id]  # type: ignore[attr-defined]
    width, height = round(camera.width * scale), round(camera.height * scale)
    camera = camera.scaled(width, height)
    K = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]])  # noqa: N806
    if crop is not None:  # a window of the image: shift the principal point
        x0, y0, width, height = crop
        K[0, 2] -= x0
        K[1, 2] -= y0
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = image.rotation
    world_to_cam[:3, 3] = image.translation
    return world_to_cam, K, width, height


@pytest.mark.parametrize(
    ("index", "scale", "crop"),
    [(0, 1 / 16, None), (7, 1.0, (700, 380, 72, 56)), (23, 0.5, (300, 200, 64, 48))],
)
def test_matches_reference_on_the_trained_splat(
    real_scene: tuple, index: int, scale: float, crop: tuple | None
) -> None:
    fields, model = real_scene
    world_to_cam, K, width, height = _real_view(model, index, scale, crop)  # noqa: N806

    fast = metal_splat.MetalSplatRenderer(fields).render(world_to_cam, K, width, height, bg=(0.0, 0.0, 0.0))
    ref = reference_render(*fields_to_arrays(fields), world_to_cam, K, width, height)

    _check(_compare(fast, ref))


def test_crop_equals_the_window_of_the_full_render(real_scene: tuple) -> None:
    fields, model = real_scene
    renderer = metal_splat.MetalSplatRenderer(fields)
    world_to_cam, K, width, height = _real_view(model, 11, 0.25)  # noqa: N806
    full = renderer.render(world_to_cam, K, width, height)
    x0, y0, w, h = 128, 64, 64, 48
    cropped = renderer.render(world_to_cam, K - np.array([[0, 0, x0], [0, 0, y0], [0, 0, 0]]), w, h)
    # Not bit-equal: gsplat's EWA Jacobian clamps off-screen directions at limits that depend on the
    # image bounds, so Gaussians centred just outside the window project slightly differently.
    assert psnr(cropped["rgb"], full["rgb"][y0 : y0 + h, x0 : x0 + w]) > 45  # noqa: PLR2004
    np.testing.assert_allclose(cropped["alpha"], full["alpha"][y0 : y0 + h, x0 : x0 + w], atol=0.02)


def _write_ply(path: Path, fields: dict[str, np.ndarray]) -> None:
    """Write fields in the INRIA layout (``f_rest`` channel-major), as a trainer exports them."""
    count = len(fields["means"])
    rest = np.swapaxes(fields["sh"][:, 1:], 1, 2).reshape(count, -1)  # (N, 3 * (K - 1)), channel-major
    columns = [fields["means"], fields["sh"][:, 0], rest, fields["opacity_logits"][:, None]]
    columns += [fields["log_scales"], fields["quats"]]
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", *(f"f_rest_{i}" for i in range(rest.shape[1]))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    path.write_bytes(header.encode("ascii") + np.hstack(columns).astype("<f4").tobytes())


def test_background_moves_object_gaussians_like_a_moved_capture(tmp_path: Path) -> None:
    from physicalai_mujoco_plugin.real2sim import appearance  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.geometry import Similarity  # noqa: PLC0415
    from physicalai_mujoco_plugin.real2sim.splats import matrix_to_quat  # noqa: PLC0415

    yaw = math.radians(35.0)
    turn = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    capture_rotation = quats_to_matrices(np.array([[0.9, 0.3, -0.2, 0.25]]))[0]
    transform = Similarity(capture_rotation.T, np.array([1.5, -0.7, 2.0]), 0.5)  # scene = 0.5 R (p - o)
    to_capture = lambda p: transform.origin + (p @ transform.rotation) / transform.scale  # noqa: E731

    rng = np.random.default_rng(5)
    box = np.column_stack([rng.uniform(-0.04, 0.04, (150, 2)), rng.uniform(0.01, 0.09, 150)])
    table = np.column_stack([rng.uniform(-0.4, 0.4, 400), rng.uniform(0.08, 0.5, 400), np.zeros(400)])
    scene = np.vstack([box, table])
    fields = synthetic_fields(len(scene), 6)
    fields["means"] = to_capture(scene)
    fields["log_scales"] = rng.uniform(np.log(0.01), np.log(0.04), (len(scene), 3))
    fields["opacity_logits"] = rng.uniform(0, 5, len(scene))
    _write_ply(tmp_path / "splat.ply", fields)

    # The same capture with the box already moved: rotated by ``turn`` about its body origin, then shifted.
    position = np.array([0.12, -0.05, 0.0])
    moved = {k: np.array(v, copy=True) for k, v in fields.items()}
    moved["means"][:150] = to_capture(box @ turn.T + position)
    motion = transform.rotation.T @ turn @ transform.rotation  # the turn seen in the capture frame
    moved["quats"][:150] = [matrix_to_quat(motion @ m) for m in quats_to_matrices(fields["quats"][:150])]
    _write_ply(tmp_path / "moved.ply", moved)

    objects = [{"name": "box", "center": [0.0, 0.0, 0.0], "size": [0.1, 0.1, 0.1]}]
    background = metal_splat.MetalSplatBackground(tmp_path / "splat.ply", transform, objects=objects)
    reference = metal_splat.MetalSplatBackground(tmp_path / "moved.ply", transform)
    eye, target = np.array([0.1, -0.6, 0.35]), np.array([0.0, 0.1, 0.03])
    back = (eye - target) / np.linalg.norm(eye - target)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    view = appearance.View(eye, np.column_stack([right, np.cross(back, right), back]), 150, 150, 80, 60, 160, 120)

    still = background.frame(dataclasses.replace(view, objects={"box": (np.zeros(3), np.eye(3))}))
    carried = background.frame(dataclasses.replace(view, objects={"box": (position, turn)}))
    expected = reference.frame(view)

    assert still.shape == (120, 160, 3)
    assert still.dtype == np.uint8
    assert background.renderer.last_intersections > 0
    diff = np.abs(carried.astype(int) - expected.astype(int))
    assert diff.max() <= 2, diff.max()  # float32 rounding of the two routes to the same means
    assert np.abs(still.astype(int) - expected.astype(int)).mean() > 5  # noqa: PLR2004 - the move shows
