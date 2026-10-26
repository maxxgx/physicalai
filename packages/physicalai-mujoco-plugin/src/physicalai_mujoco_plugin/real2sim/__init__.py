# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bring a captured real environment into the MuJoCo sim (real2sim proof of concept).

A capture (posed photos from COLMAP, optionally a trained Gaussian splat) becomes a robot-free
scene XML for ``MuJoCoRobot(model_path=...)``: the table and the objects on it as collision
geometry in a metric, z-up table frame, a ``robot_mount`` on the table, and the capture's cameras
with their real intrinsics. The splat, moved into the same frame, supplies the appearance.
"""
