# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Record booth episodes through the plugin into a LeRobot v3.0 dataset, and replay them.

Recording runs the registered ``booth_task`` scene in :class:`~physicalai_mujoco_plugin.robot.MuJoCoRobot`
(bimanual WidowX AI, headless) with the autopilot driving, one fresh simulation per episode, at the
camera rate (30 fps by default, 20 physics steps per frame). Each frame stores:

- ``observation.state``: the 14 joints as the plugin reports them, named like physicalai's composite
  robot (``left_shoulder_pan.pos`` ... ``right_gripper.pos``): rotating joints in degrees, grippers in
  metres, the real WidowX AI driver's signs and zeros;
- ``action``: the 14 position targets the autopilot commanded at that frame, in the same units;
- ``observation.images.{overview,left_wrist,right_wrist}``: MuJoCo renders (640x480 by default),
  encoded as AV1 (``libsvtav1``, as LeRobot writes them) or H.264 when ffmpeg lacks it.

The layout is LeRobot's v3.0 (``meta/info.json``, ``meta/stats.json``, ``meta/tasks.parquet``,
``meta/episodes/...``, ``data/...``, ``videos/...``), which Studio imports and LeRobot loads.
``meta/booth_task.json`` adds what replay needs: each episode's reset seed and recorded end state.

Replay builds the same start state, sends the recorded actions back through ``send_action`` one per
frame, and compares the end state with the recorded one.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import json
import math
import shutil
import subprocess  # noqa: S404 - ffmpeg encodes the camera streams
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.booth_task.demonstrator import EPISODE_END
from physicalai_mujoco_plugin.booth_task.frames import body_pose, yaw_of
from physicalai_mujoco_plugin.booth_task.harness import check_end
from physicalai_mujoco_plugin.booth_task.layout import Layout

CAMERAS = ("overview", "left_wrist", "right_wrist")
TASK = (
    "Take the INTEL box from the bin, hand it over, lay it in the red bracket and cover it; then put everything back."
)
ROBOT_TYPE = "bimanual_widowxai"
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
FRAME_STEP_TOL = 1e-3
"""How far one frame may be from a whole number of physics steps (as a fraction of the frame)."""


@dataclass
class Episode:
    """One recorded episode's frames (images go straight to the encoders)."""

    states: list[np.ndarray]
    actions: list[np.ndarray]
    image_samples: dict[str, list[np.ndarray]]
    seed: int
    end: dict[str, object]


def _robot(seed: int, fps: int):  # noqa: ANN202 - MuJoCoRobot, imported lazily
    from physicalai_mujoco_plugin.robot import MuJoCoRobot  # noqa: PLC0415

    robot = MuJoCoRobot(
        "trossen_wxai", scene="booth_task", bimanual=True, cameras=[], rate_hz=fps, seed=seed, http_port=0
    )
    robot.connect()
    if round(1.0 / (fps * robot._sim.model.opt.timestep)) * robot._sim.model.opt.timestep * fps - 1.0 > FRAME_STEP_TOL:  # noqa: SLF001
        msg = f"{fps} fps is not a whole number of physics steps"
        raise ValueError(msg)
    robot.reset()  # the booth reset with the fixed seed: arms asleep, objects drawn within D7's ranges
    return robot


def _end_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    box, cover = body_pose(data, model.body("box").id), body_pose(data, model.body("cover").id)
    return {
        "box_pos": box.pos.tolist(),
        "box_rot": box.rot.tolist(),
        "cover_pos": cover.pos.tolist(),
        "cover_yaw_deg": math.degrees(yaw_of(cover.rot, 1)),
    }


class _Encoder:
    """One ffmpeg process per camera, fed raw RGB frames of every episode in turn."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.codec, args = _codec()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            msg = "ffmpeg is needed to encode the camera streams (brew install ffmpeg)"
            raise RuntimeError(msg)
        cmd = [
            ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", *args, "-pix_fmt", "yuv420p", str(path),
        ]  # fmt: skip
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # noqa: S603
        self.frames = 0

    def write(self, rgb: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())  # type: ignore[union-attr]
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()  # type: ignore[union-attr]
        if self.proc.wait() != 0:
            msg = "ffmpeg failed"
            raise RuntimeError(msg)


def _codec() -> tuple[str, list[str]]:
    """AV1 like LeRobot's own writer when ffmpeg has SVT-AV1, else H.264.

    Returns:
        The codec's name and its ffmpeg arguments.
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False).stdout  # noqa: S603
    if "libsvtav1" in encoders:
        return "av1", ["-c:v", "libsvtav1", "-g", "2", "-crf", "30", "-preset", "12"]
    return "h264", ["-c:v", "libx264", "-g", "2", "-crf", "23", "-preset", "veryfast"]


def record(  # noqa: PLR0914 - the recording loop keeps its per-episode buffers together
    out: Path,
    *,
    episodes: int = 2,
    seed: int = 0,
    perturb: bool = False,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    appearance: Path | None = None,
) -> None:
    """Record *episodes* full cycles into a LeRobot v3.0 dataset at *out* (replaced if it exists).

    Args:
        out: Dataset root.
        episodes: Number of full cycles.
        seed: Episode ``k`` resets the scene with seed ``seed + k``.
        perturb: Unused; the scene reset always draws the objects within RUBRIC D7's start ranges.
        width: Camera width.
        height: Camera height.
        fps: Frame rate, also the control rate.
        appearance: A ``real2sim.json`` with a ``photoreal`` block: the camera streams are composited over the
            trained splat (``real2sim.photoreal``) instead of MuJoCo's plain render.
    """
    del perturb
    out = Path(out).expanduser()
    if out.exists():
        shutil.rmtree(out)
    encoders = {
        cam: _Encoder(out / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4", width, height, fps)
        for cam in CAMERAS
    }
    recorded: list[Episode] = []
    started = time.monotonic()
    n_frames = round(EPISODE_END * fps)
    for k in range(episodes):
        robot = _robot(seed + k, fps)
        sim = robot._sim  # noqa: SLF001 - the plugin's own simulation, read for rendering
        renderer = mujoco.Renderer(sim.model, height, width)
        compositors = {}
        if appearance is not None:
            import json  # noqa: PLC0415

            from physicalai_mujoco_plugin.real2sim.photoreal import load_photoreal_compositors  # noqa: PLC0415

            compositors = load_photoreal_compositors(json.loads(Path(appearance).read_text()), sim.model)
        box_start = body_pose(sim.data, sim.model.body("box").id)
        cover_start = body_pose(sim.data, sim.model.body("cover").id)
        robot._automation.set_autopilot("drive")  # noqa: SLF001
        episode = Episode([], [], {cam: [] for cam in CAMERAS}, seed + k, {})

        def capture(
            index: int, ep: Episode = episode, rend: mujoco.Renderer = renderer, data: mujoco.MjData = sim.data,
            comps: dict = compositors,
        ) -> None:
            for cam in CAMERAS:
                rend.update_scene(data, camera=cam)
                rgb = rend.render()
                if cam in comps:
                    rgb = comps[cam](rend, data, cam, rgb)
                encoders[cam].write(rgb)
                if index % 10 == 0:
                    ep.image_samples[cam].append(rgb[::4, ::4].copy())

        state = np.concatenate([ch.read_positions() for ch in sim.channels])
        capture(0)
        t0 = time.monotonic()
        for i in range(n_frames):
            obs = robot.get_observation()
            leader = robot._automation.leader_snapshot()  # noqa: SLF001 - the autopilot's targets, in the robot's units
            episode.states.append(state.astype(np.float32))
            episode.actions.append(np.asarray(leader["joint_positions"], dtype=np.float32))
            state = np.asarray(obs.joint_positions, dtype=np.float64)
            if i + 1 < n_frames:
                capture(i + 1)
        episode.end = _end_state(sim.model, sim.data)
        check = check_end(
            sim.model, Layout.from_model(sim.model), body_pose(sim.data, sim.model.body("box").id),
            body_pose(sim.data, sim.model.body("cover").id), box_start, cover_start,
        )  # fmt: skip
        episode.end["check"] = {k2: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k2, v in check.items()}
        logger.info(
            "Episode {}: {} frames in {:.0f} s ({:.2f}x real time)",
            k,
            n_frames,
            time.monotonic() - t0,
            EPISODE_END / (time.monotonic() - t0),
        )
        renderer.close()
        robot.disconnect()
        recorded.append(episode)
    codec = next(iter(encoders.values())).codec
    for enc in encoders.values():
        enc.close()
    _write_dataset(out, recorded, fps=fps, width=width, height=height, codec=codec)
    logger.info("Recorded {} episodes in {:.0f} s into {}", episodes, time.monotonic() - started, out)


# --- LeRobot v3.0 files -------------------------------------------------------------------------


def _names() -> list[str]:
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")
    return [f"{side}_{joint}.pos" for side in ("left", "right") for joint in joints]


def _stats(values: np.ndarray) -> dict[str, list]:
    """LeRobot's per-feature statistics over the first axis.

    Returns:
        Each statistic as nested lists.
    """
    out = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    for q in QUANTILES:
        out[f"q{round(q * 100):02d}"] = np.quantile(values, q, axis=0).tolist()
    return out


def _image_stats(samples: list[np.ndarray]) -> dict[str, list]:
    """Per-channel statistics of sampled frames, in [0, 1], shaped (3, 1, 1) like LeRobot's.

    Returns:
        Each statistic as (3, 1, 1) nested lists.
    """
    pixels = np.concatenate([s.reshape(-1, 3) for s in samples]).astype(np.float64) / 255.0
    base = _stats(pixels)
    out = {k: [[[v]] for v in vals] for k, vals in base.items() if k != "count"}
    out["count"] = [len(samples)]
    return out


def _write_dataset(out: Path, episodes: list[Episode], *, fps: int, width: int, height: int, codec: str) -> None:  # noqa: PLR0914, PLR0915
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        msg = "Writing the dataset needs pyarrow: pip install pyarrow"
        raise RuntimeError(msg) from exc

    names = _names()
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    columns: dict[str, list] = {
        k: []
        for k in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")
    }
    episode_rows = []
    all_numeric: dict[str, list[np.ndarray]] = {
        k: []
        for k in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")
    }
    index = 0
    video_from = 0.0
    for e, ep in enumerate(episodes):
        n = len(ep.states)
        states, actions = np.array(ep.states, dtype=np.float32), np.array(ep.actions, dtype=np.float32)
        ts = (np.arange(n) / fps).astype(np.float32)
        per = {
            "observation.state": states,
            "action": actions,
            "timestamp": ts[:, None].astype(np.float64),
            "frame_index": np.arange(n)[:, None].astype(np.float64),
            "episode_index": np.full((n, 1), e, dtype=np.float64),
            "index": (index + np.arange(n))[:, None].astype(np.float64),
            "task_index": np.zeros((n, 1)),
        }
        columns["observation.state"] += [s.tolist() for s in states]
        columns["action"] += [a.tolist() for a in actions]
        columns["timestamp"] += ts.tolist()
        columns["frame_index"] += list(range(n))
        columns["episode_index"] += [e] * n
        columns["index"] += list(range(index, index + n))
        columns["task_index"] += [0] * n
        for k, v in per.items():
            all_numeric[k].append(v)
        row: dict[str, object] = {
            "episode_index": e,
            "tasks": [TASK],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": index,
            "dataset_to_index": index + n,
        }
        for cam in CAMERAS:
            row[f"videos/observation.images.{cam}/chunk_index"] = 0
            row[f"videos/observation.images.{cam}/file_index"] = 0
            row[f"videos/observation.images.{cam}/from_timestamp"] = video_from
            row[f"videos/observation.images.{cam}/to_timestamp"] = video_from + n / fps
        for k, v in per.items():
            for stat, value in _stats(v).items():
                row[f"stats/{k}/{stat}"] = value
        for cam in CAMERAS:
            for stat, value in _image_stats(ep.image_samples[cam]).items():
                row[f"stats/observation.images.{cam}/{stat}"] = value
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        episode_rows.append(row)
        index += n
        video_from += n / fps

    vector = pa.list_(pa.float32(), 14)
    data_table = pa.table({
        "observation.state": pa.array(columns["observation.state"], type=vector),
        "action": pa.array(columns["action"], type=vector),
        "timestamp": pa.array(columns["timestamp"], type=pa.float32()),
        "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
        "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
        "index": pa.array(columns["index"], type=pa.int64()),
        "task_index": pa.array(columns["task_index"], type=pa.int64()),
    })
    hf_features = {
        "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 14, "_type": "List"},
        "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 14, "_type": "List"},
        "timestamp": {"dtype": "float32", "_type": "Value"},
        "frame_index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "index": {"dtype": "int64", "_type": "Value"},
        "task_index": {"dtype": "int64", "_type": "Value"},
    }
    data_table = data_table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": hf_features}})})
    pq.write_table(data_table, out / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(episode_rows), out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    tasks = pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array([TASK], type=pa.string())})
    pandas_meta = {
        "index_columns": ["task"],
        "column_indexes": [
            {
                "name": None,
                "field_name": None,
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": {"encoding": "UTF-8"},
            }
        ],
        "columns": [
            {
                "name": "task_index",
                "field_name": "task_index",
                "pandas_type": "int64",
                "numpy_type": "int64",
                "metadata": None,
            },
            {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": None},
        ],
        "attributes": {},
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.3.3",
    }
    pq.write_table(tasks.replace_schema_metadata({"pandas": json.dumps(pandas_meta)}), out / "meta" / "tasks.parquet")

    stats: dict[str, dict] = {k: _stats(np.concatenate(v)) for k, v in all_numeric.items()}
    for cam in CAMERAS:
        stats[f"observation.images.{cam}"] = _image_stats([s for ep in episodes for s in ep.image_samples[cam]])
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": names},
        "action": {"dtype": "float32", "shape": [14], "names": names},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": codec,
                "video.pix_fmt": "yuv420p",
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
                "is_depth_map": False,
            },
        }
    for name, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[name] = {"dtype": dtype, "shape": [1], "names": None}
    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "features": features,
        "total_episodes": len(episodes),
        "total_frames": index,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": ROBOT_TYPE,
        "splits": {"train": f"0:{len(episodes)}"},
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    sidecar = {
        "scene": "booth_task",
        "robot": "trossen_wxai x2 (left_, right_)",
        "units": (
            "rotating joints in degrees, grippers in metres (finger travel), as physicalai_trossen_plugin.WidowXAI"
        ),
        "control": f"actions are the position targets commanded at each frame, held for 1/{fps} s",
        "episodes": [{"episode_index": e, "seed": ep.seed, "end": ep.end} for e, ep in enumerate(episodes)],
    }
    (out / "meta" / "booth_task.json").write_text(json.dumps(sidecar, indent=1, default=float))


# --- replay ---------------------------------------------------------------------------------------


def replay(dataset: Path, episode: int = 0) -> bool:  # noqa: PLR0914 - one routine: rebuild, replay, compare
    """Replay a recorded episode's actions from the same start state; compare the end state.

    Returns:
        Whether the replayed end state matches the recorded one (box in the same slot, cover within
        1 cm) and passes the task's end-state check.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    dataset = Path(dataset).expanduser()
    info = json.loads((dataset / "meta" / "info.json").read_text())
    meta = json.loads((dataset / "meta" / "booth_task.json").read_text())
    entry = meta["episodes"][episode]
    table = pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet", columns=["action", "episode_index"])
    episodes = np.array(table.column("episode_index").to_pylist())
    actions = np.array(table.column("action").to_pylist(), dtype=np.float64)[episodes == episode]
    robot = _robot(int(entry["seed"]), int(info["fps"]))
    sim = robot._sim  # noqa: SLF001
    layout = Layout.from_model(sim.model)
    box_start = body_pose(sim.data, sim.model.body("box").id)
    cover_start = body_pose(sim.data, sim.model.body("cover").id)
    for action in actions:
        robot.send_action(action, goal_time=0.0)
        robot.get_observation()
    end = _end_state(sim.model, sim.data)
    check = check_end(
        sim.model,
        layout,
        body_pose(sim.data, sim.model.body("box").id),
        body_pose(sim.data, sim.model.body("cover").id),
        box_start,
        cover_start,
    )
    robot.disconnect()
    recorded = entry["end"]
    cover_err = float(np.linalg.norm(np.array(end["cover_pos"])[:2] - np.array(recorded["cover_pos"])[:2]))
    box_err = float(np.linalg.norm(np.array(end["box_pos"]) - np.array(recorded["box_pos"])))
    same_slot = check.get("box_slot") == recorded.get("check", {}).get("box_slot", check.get("box_slot"))
    ok = bool(check["ok"] and same_slot and cover_err <= 0.01)  # noqa: PLR2004
    print(  # noqa: T201
        json.dumps(
            {
                "episode": episode,
                "frames": len(actions),
                "box_end_diff_m": box_err,
                "cover_end_diff_m": cover_err,
                "end_check": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in check.items()},
                "reproduced": ok,
            },
            indent=1,
        )
    )
    return ok


__all__ = ["CAMERAS", "record", "replay"]
