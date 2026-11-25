# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test-session defaults."""

import os

# The autopilot plans attempts in a worker process by default (bucket_sort_demo._PlanProcess). Tests plan in-process,
# so a test can patch the planner it holds; tests/test_bucket_sort_plan_process.py opts in explicitly.
os.environ.setdefault("PHYSICALAI_PLAN_PROCESS", "0")
