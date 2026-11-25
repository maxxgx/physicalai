"""Analysis only (r10-r12, retired in r13): the drawing-only finger transform that r10-r12's kits applied to the drawn
finger meshes (``finger_visual``). r13 removed it from the plugin: drawing and collision now use one shape (the CAD),
and the left/right asymmetry it absorbed was the wrist camera's mount (real/r13/wrist_joint_fit.json). Kept here so the
r10-r13 fit scripts (fit_finger_joint.py, finger_tips.py, fit_wrist_joint.py variant C) still run."""
from __future__ import annotations

import mujoco
import numpy as np


_FINGER_ORIGINAL: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def apply_finger_visual(m: mujoco.MjModel, spec: dict | None) -> None:  # noqa: C901
    """Thin the drawn finger jaws across the closing direction, about their inner (gripping) face (r10, D2).

    The real fingers' jaws are thinner across than the model's: on allowed wrist frames the inner edges agree to 0-2 px
    and the outer ones lie ~40 px further out in the sim (scripts fit_finger_visual.py). For each finger body in
    ``spec["tip_axis"]`` (body: axis in its frame), its drawn mesh vertices (geom group < 3) within ``jaw_m`` of the tip
    along that axis (a linear ramp over ``ramp_m`` behind it) move toward the jaw's inner face along the slide axis by
    the factor ``scale``; normals
    follow (inverse transpose). Collision geoms, masses and the slide are unchanged. Call before creating a renderer
    (the meshes upload then); idempotent per model (it always starts from the original vertices)."""
    if not spec:
        return
    jaw, ramp, k = float(spec["jaw_m"]), max(float(spec.get("ramp_m", 0.0)), 1e-9), float(spec["scale"])
    rot = np.zeros(9)
    for name, axis in spec["tip_axis"].items():
        tip = np.asarray(axis, float); tip /= np.linalg.norm(tip)
        body = m.body(name).id; slide = np.asarray(m.jnt_axis[m.body_jntadr[body]], float)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] < 3]
        frames, inner, end = [], np.inf, -np.inf
        for g in geoms:   # the tip: the farthest drawn vertex along the axis
            mujoco.mju_quat2Mat(rot, m.geom_quat[g]); mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            src = _FINGER_ORIGINAL.get((id(m), mid), (m.mesh_vert[a:a + n],))[0]
            end = max(end, float(((src @ rot.reshape(3, 3).T + m.geom_pos[g]) @ tip).max()))
        start = end - jaw
        for g in geoms:
            mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            orig = _FINGER_ORIGINAL.setdefault((id(m), mid), (m.mesh_vert[a:a + n].copy(), m.mesh_normal[m.mesh_normaladr[mid]:m.mesh_normaladr[mid] + m.mesh_normalnum[mid]].copy()))
            mujoco.mju_quat2Mat(rot, m.geom_quat[g]); R = rot.reshape(3, 3).copy()
            P = orig[0] @ R.T + m.geom_pos[g]                                      # body frame
            w = np.clip((P @ tip - start) / ramp, 0.0, 1.0)
            if (w > 0).any(): inner = min(inner, float((P @ slide)[w > 0].min()))
            frames.append((g, mid, a, n, R, P, w, orig))
        if not np.isfinite(inner):
            continue
        tip_scale = float((spec.get("tip_scale") or {}).get(name, 1.0)); shear = float((spec.get("tip_shear") or {}).get(name, 0.0))
        for g, mid, a, n, R, P, w, orig in frames:
            along = P @ slide; f = 1.0 + w * (k - 1.0)                             # 1 at the carriage, k on the jaw
            Q = P + np.outer((inner + (along - inner) * f) - along, slide)
            # r11: the jaw's length along the tip axis (x tip_scale beyond the jaw's start) and a lean of the tip across
            # (tip_shear metres outward per metre beyond the start): both 0 at the start, so the carriage is untouched
            beyond = np.maximum(P @ tip - start, 0.0)
            if tip_scale != 1.0 or shear != 0.0:
                Q = Q + np.outer(beyond * (tip_scale - 1.0), tip) + np.outer(beyond * shear, slide)
            m.mesh_vert[a:a + n] = ((Q - m.geom_pos[g]) @ R).astype(m.mesh_vert.dtype)
            na, nn = int(m.mesh_normaladr[mid]), int(m.mesh_normalnum[mid])
            if nn == n:   # one normal per vertex: scale the slide component by 1/f, then renormalise
                N = orig[1] @ R.T; N = N + np.outer((N @ slide) * (1.0 / f - 1.0), slide)
                N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
                m.mesh_normal[na:na + nn] = (N @ R).astype(m.mesh_normal.dtype)
