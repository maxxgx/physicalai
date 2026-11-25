# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Command line for the booth task: run one episode, the robustness sweep, record, replay.

python -m physicalai_mujoco_plugin.booth_task run --seed 0
python -m physicalai_mujoco_plugin.booth_task robustness --seeds 0-49 --perturb --jobs 8 --out r.json
python -m physicalai_mujoco_plugin.booth_task record --episodes 2 --out ~/booth_dataset
python -m physicalai_mujoco_plugin.booth_task replay --dataset ~/booth_dataset --episode 0
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from physicalai_mujoco_plugin.booth_task.harness import run_quiet


def _seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _summary(report: dict) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    detail = "" if report["passed"] else " | " + "; ".join(report["failures"])[:400]
    return (
        f"seed {report['seed']:3d} {status} pen {report['worst_penetration_mm']:.2f} mm, "
        f"arm contact {report['max_arm_contact_force']:.1f} N, speed {report['max_joint_speed_ratio']:.2f}x, "
        f"wall {report['wall_s']:.0f} s{detail}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        The exit code: 0 on success, 1 when the run or the replay fails, 2 for an unknown command.
    """
    parser = argparse.ArgumentParser(prog="python -m physicalai_mujoco_plugin.booth_task")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="one episode through the reviewer-style checks")
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--perturb", action="store_true")
    rob = sub.add_parser("robustness", help="many seeds through the reviewer-style checks")
    rob.add_argument("--seeds", default="0-49")
    rob.add_argument("--perturb", action="store_true")
    rob.add_argument("--jobs", type=int, default=4)
    rob.add_argument("--out", type=Path, default=None)
    rec = sub.add_parser("record", help="record full-cycle episodes as a LeRobot v3.0 dataset")
    rec.add_argument("--episodes", type=int, default=2)
    rec.add_argument("--out", type=Path, required=True)
    rec.add_argument("--seed", type=int, default=0, help="first seed; episode k uses seed + k")
    rec.add_argument("--perturb", action="store_true")
    rec.add_argument("--width", type=int, default=640)
    rec.add_argument("--height", type=int, default=480)
    rec.add_argument("--fps", type=int, default=30)
    rec.add_argument("--appearance", type=Path, default=None, help="real2sim.json with a photoreal block (splat compositing)")
    rep = sub.add_parser("replay", help="replay a recorded episode's actions and compare the end state")
    rep.add_argument("--dataset", type=Path, required=True)
    rep.add_argument("--episode", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "run":
        report = run_quiet((args.seed, args.perturb))
        print(_summary(report))  # noqa: T201
        print(json.dumps({k: report[k] for k in ("assembled", "end")}, indent=1, default=str))  # noqa: T201
        return 0 if report["passed"] else 1
    if args.command == "robustness":
        seeds = _seeds(args.seeds)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            reports = []
            for report in pool.map(run_quiet, [(s, args.perturb) for s in seeds]):
                print(_summary(report), flush=True)  # noqa: T201
                reports.append(report)
        passed = sum(r["passed"] for r in reports)
        print(f"{passed}/{len(reports)} passed")  # noqa: T201
        if args.out is not None:
            args.out.write_text(
                json.dumps({"passed": passed, "total": len(reports), "reports": reports}, indent=1, default=str)
            )
        return 0
    if args.command == "record":
        from physicalai_mujoco_plugin.booth_task.record import record  # noqa: PLC0415

        record(
            args.out,
            episodes=args.episodes,
            seed=args.seed,
            perturb=args.perturb,
            width=args.width,
            height=args.height,
            fps=args.fps,
            appearance=args.appearance,
        )
        return 0
    if args.command == "replay":
        from physicalai_mujoco_plugin.booth_task.record import replay  # noqa: PLC0415

        ok = replay(args.dataset, args.episode)
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
