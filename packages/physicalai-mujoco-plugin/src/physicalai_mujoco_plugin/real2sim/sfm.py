# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""From a phone video of the bench to posed, undistorted photos (COLMAP via pycolmap).

``extract_frames`` keeps the sharpest frame of each stretch of the video (a handheld orbit has
motion blur that structure from motion cannot use); ``run_sfm`` extracts SIFT features, matches
them, reconstructs the cameras and the sparse points, and undistorts the photos to a pinhole
model, which is what the sim cameras are. The result is what ``real2sim build`` reads.

pycolmap is optional (``pip install pycolmap``); everything else in real2sim works without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


@dataclass(frozen=True)
class SfmResult:
    """Where the undistorted capture is, and how much of it registered."""

    images: Path
    sparse: Path
    registered: int
    total: int
    points: int


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low for blurred frames.

    Returns:
        The sharpness score.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(video: Path, out_dir: Path, *, count: int = 150, max_side: int = 1600) -> list[Path]:
    """Write the sharpest frame of each of ``count`` equal stretches of ``video`` as JPEGs.

    Args:
        video: Any file OpenCV can read (``.mov``, ``.mp4``).
        out_dir: Created if missing; frames are named ``frame_0000.jpg``...
        count: Frames to keep (a slow 1-2 minute orbit wants 100-200).
        max_side: Frames are downscaled to this longest side (SfM and the gate need no more).

    Returns:
        The written frames, in video order.

    Raises:
        ValueError: If the video cannot be read.
    """
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or total <= 0:
        msg = f"cannot read video {video}"
        raise ValueError(msg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = np.linspace(0, total, count + 1).astype(int)
    written: list[Path] = []
    best: tuple[float, np.ndarray] | None = None
    stretch, index = 0, 0
    while stretch < count:
        ok, frame = capture.read()
        if not ok:
            break
        scale = max_side / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        score = sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if best is None or score > best[0]:
            best = (score, frame)
        index += 1
        if index >= bounds[stretch + 1]:
            path = out_dir / f"frame_{stretch:04d}.jpg"
            cv2.imwrite(str(path), best[1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            written.append(path)
            best, stretch = None, stretch + 1
    capture.release()
    logger.info("Kept {} of {} video frames in {}", len(written), total, out_dir)
    return written


def run_sfm(images: Path, work_dir: Path, *, matching: str = "exhaustive") -> SfmResult:
    """Reconstruct ``images`` with COLMAP and undistort them to one pinhole camera.

    Args:
        images: The photos (one phone, fixed zoom).
        work_dir: Gets ``database.db``, ``sparse/`` (raw) and ``undistorted/{images,sparse}``.
        matching: ``exhaustive`` (best for up to ~200 photos) or ``sequential`` (video order).

    Returns:
        The undistorted capture.

    Raises:
        ValueError: If nothing could be reconstructed.
    """
    import pycolmap  # noqa: PLC0415 - optional dependency

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    database.unlink(missing_ok=True)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    if matching == "sequential":
        pycolmap.match_sequential(database)
    else:
        pycolmap.match_exhaustive(database)
    raw = work_dir / "sparse"
    raw.mkdir(exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(database, images, raw)
    if not reconstructions:
        msg = f"COLMAP registered no images in {images}"
        raise ValueError(msg)
    index, best = max(reconstructions.items(), key=lambda item: item[1].num_reg_images())
    undistorted = work_dir / "undistorted"
    pycolmap.undistort_images(undistorted, raw / str(index), images)
    total = len([p for p in Path(images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return SfmResult(
        images=undistorted / "images",
        sparse=undistorted / "sparse",
        registered=best.num_reg_images(),
        total=total,
        points=best.num_points3D(),
    )
