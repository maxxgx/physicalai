"""r13: collision that coincides with the drawing. Shared by build_collision_r13.py (makes the pieces), the audit
(audit_visual_collision.py) and the tests.

- ``winding(P, V, F)``: the generalised winding number of points P about the triangle surface (V, F): ~1 inside a
  closed solid, ~0 outside; robust to small holes (the CAD's finger shell is not watertight). Inside = w > 0.5.
- ``decompose(V, F, tol, ...)``: convex pieces whose union contains the drawn solid and protrudes beyond it by at most
  ``tol``: the drawn surface clipped to an axis-aligned box, the convex hull of what lies in it; if that hull sticks out
  of the drawn solid by more than ``tol`` anywhere (its surface sampled), the box is halved across its longest side,
  and again, down to ``min_box``. Every drawn surface point lies in some box, so inside some piece: nothing drawn is
  without collision; and no piece reaches more than ``tol`` beyond the drawing (up to the sampling, ~0.4 mm).
- ``protrusion(...)`` / ``uncovered(...)``: the two one-sided distances the audit reports, signed by inside/outside."""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def winding(P: np.ndarray, V: np.ndarray, F: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Generalised winding number (van Oosterom-Strackee solid angles summed over the triangles / 4 pi)."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        a, b, c = A[None] - p, B[None] - p, C[None] - p
        la, lb, lc = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2), np.linalg.norm(c, axis=2)
        num = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la + np.einsum("ijk,ijk->ij", c, a) * lb
        out[s:s + chunk] = 2 * np.arctan2(num, den).sum(1) / (4 * np.pi)
    return out


def sample_surface(V: np.ndarray, F: np.ndarray, spacing: float, rng: np.random.Generator, cap: int = 300_000) -> np.ndarray:
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    if A.sum() <= 0: return V.copy()
    n = int(min(max(A.sum() / spacing ** 2, 100), cap))
    idx = rng.choice(len(F), n, p=A / A.sum()); u, v = rng.random(n), rng.random(n); f = u + v > 1; u[f], v[f] = 1 - u[f], 1 - v[f]
    T = V[F[idx]]
    return np.vstack([T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0]), V])


def hull_surface(H: ConvexHull, spacing: float, rng: np.random.Generator) -> np.ndarray:
    return sample_surface(H.points, H.simplices, spacing, rng, cap=60_000)


def inside_hulls(P: np.ndarray, hulls: list[np.ndarray], eps: float = 1e-6) -> np.ndarray:
    """True where P lies inside (or on) any of the convex pieces (given by their points)."""
    res = np.zeros(len(P), bool)
    for pts in hulls:
        E = ConvexHull(pts).equations
        res |= ((P @ E[:, :3].T + E[:, 3]) <= eps).all(1)
    return res


def outside_distance_hulls(P: np.ndarray, hulls: list[np.ndarray]) -> np.ndarray:
    """Distance of P outside the union of the convex pieces (0 inside), via dense samples of the pieces' surfaces for
    the outside points (exact enough at the sampling spacing)."""
    rng = np.random.default_rng(1)
    ins = inside_hulls(P, hulls); out = np.zeros(len(P))
    if (~ins).any():
        S = np.vstack([hull_surface(ConvexHull(h), 0.0004, rng) for h in hulls])
        out[~ins] = cKDTree(S).query(P[~ins])[0]
    return out


def _clip_points(V: np.ndarray, F: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The drawn surface's points within the box [lo, hi]: its vertices inside, and where its edges and triangles cross
    the box (the triangles clipped by the six planes, Sutherland-Hodgman)."""
    pts = []
    tri = V[F]
    keep = ~((tri.max(1) < lo).any(1) | (tri.min(1) > hi).any(1))
    for t in tri[keep]:
        poly = list(t)
        for ax in range(3):
            for bound, sgn in ((lo[ax], 1.0), (hi[ax], -1.0)):
                if not poly: break
                out = []
                for i in range(len(poly)):
                    p, q = poly[i], poly[(i + 1) % len(poly)]
                    dp, dq = sgn * (p[ax] - bound), sgn * (q[ax] - bound)
                    if dp >= 0: out.append(p)
                    if (dp >= 0) != (dq >= 0):
                        out.append(p + (q - p) * (dp / (dp - dq)))
                poly = out
        pts += poly
    return np.array(pts) if pts else np.zeros((0, 3))


def oriented_samples(V: np.ndarray, F: np.ndarray, spacing: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Surface samples with their faces' unit normals (for a closed mesh with consistent outward winding)."""
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    n = int(min(max(A.sum() / spacing ** 2, 100), 2_000_000))
    idx = rng.choice(len(F), n, p=A / A.sum()); u, v = rng.random(n), rng.random(n); f = u + v > 1; u[f], v[f] = 1 - u[f], 1 - v[f]
    T = V[F[idx]]; N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0]), N


def decompose(V: np.ndarray, F: np.ndarray, tol: float = 0.0015, min_box: float = 0.002, max_pieces: int = 32,
              spacing: float = 0.0004, seed: int = 0, frame: str = "pca", inside: str = "winding") -> tuple[list[np.ndarray], dict]:
    """``inside``: "winding" (robust to holes; slow on big meshes) or "normal" (a watertight mesh with outward
    winding: a point is outside when it lies on the outer side of its nearest surface sample; fast)."""
    """``frame``: "pca" splits in the drawn surface's principal axes (a slanted finger is cut along itself), "body"
    in the body frame's axes. The pieces come back in the body frame."""
    if frame == "pca":
        c = V.mean(0); _, _, Rt = np.linalg.svd(V - c, full_matrices=False)
        if np.linalg.det(Rt) < 0: Rt[-1] *= -1   # a rotation, not a reflection (that would turn the normals inward)
        hulls, info = decompose((V - c) @ Rt.T, F, tol, min_box, max_pieces, spacing, seed, frame="body", inside=inside)
        return [h @ Rt + c for h in hulls], {**info, "split_frame": "pca"}
    rng = np.random.default_rng(seed)
    if inside == "normal":
        S, SN = oriented_samples(V, F, spacing, rng)
    else:
        S = sample_surface(V, F, spacing, rng)
    tree = cKDTree(S)
    def stick_out(pts: np.ndarray) -> float:
        H = ConvexHull(pts); Q = hull_surface(H, spacing * 1.5, rng)
        if inside == "normal":
            dist, k = tree.query(Q); out = ((Q - S[k]) * SN[k]).sum(1) > 0.5 * spacing
            return float(dist[out].max()) if out.any() else 0.0
        w = winding(Q, V, F); outside = Q[w < 0.5]
        return float(tree.query(outside)[0].max()) if len(outside) else 0.0
    import heapq
    def piece(lo, hi):
        pts = _clip_points(V, F, lo, hi)
        if len(pts) < 4: return None
        pts = np.unique(np.round(pts, 7), axis=0)
        try:
            ConvexHull(pts)
        except Exception:   # flat: thicken by 0.2 mm
            pts = np.vstack([pts, pts + 0.0002 * np.eye(3)[np.argmin(np.ptp(pts, 0))]])
        return pts, stick_out(pts)
    lo0, hi0 = V.min(0) - 1e-6, V.max(0) + 1e-6; first = piece(lo0, hi0)
    heap = [(-first[1], 0, lo0, hi0, first[0])]; k = 1   # best first: always split the piece that sticks out most
    final = []   # too small to split further
    while heap and -heap[0][0] > tol and len(heap) + len(final) < max_pieces:
        e, _, lo, hi, pts = heapq.heappop(heap); ext = hi - lo
        if ext.max() < min_box:
            final.append((e, k, lo, hi, pts)); k += 1; continue
        ax = int(np.argmax(ext)); mid = (lo[ax] + hi[ax]) / 2
        hi1 = hi.copy(); hi1[ax] = mid; lo2 = lo.copy(); lo2[ax] = mid
        for a_, b_ in ((lo, hi1), (lo2, hi)):
            r = piece(a_, b_)
            if r is not None: heapq.heappush(heap, (-r[1], k, a_, b_, r[0])); k += 1
    heap += final
    done = [h[4] for h in heap]; worst = max(-h[0] for h in heap); forced = sum(-h[0] > tol for h in heap)
    # thin each piece's points to its hull vertices (what MuJoCo keeps anyway)
    done = [thicken(p) for p in done]
    done = [p[ConvexHull(p).vertices] for p in done]
    return done, {"pieces": len(done), "worst_protrusion_m": worst, "pieces_over_tol": forced, "tol_m": tol}


def thicken(pts: np.ndarray, min_thick: float = 0.0003) -> np.ndarray:
    """A piece thinner than *min_thick* along its least principal axis (MuJoCo cannot hull coplanar points) gets a copy
    of its points moved by *min_thick* along that axis, outward of the drawn solid's side it lies on (either way <=
    min_thick of extra reach)."""
    c = pts.mean(0); _, sv, Vt = np.linalg.svd(pts - c, full_matrices=False); n = Vt[-1]
    thick = float(np.ptp((pts - c) @ n))
    if thick >= min_thick: return pts
    return np.vstack([pts, pts + (min_thick - thick) * n])


def report(V: np.ndarray, F: np.ndarray, hulls: list[np.ndarray], spacing: float = 0.0004, seed: int = 0) -> dict:
    """protrusion: the pieces' surface outside the drawn solid, distance to the drawn surface; uncovered: the drawn
    surface outside every piece, distance to the pieces. mm: median, p95, max."""
    rng = np.random.default_rng(seed)
    S = sample_surface(V, F, spacing, rng)
    Q = np.vstack([hull_surface(ConvexHull(h), spacing, rng) for h in hulls])
    w = winding(Q, V, F); prot = np.zeros(len(Q)); o = w < 0.5
    prot[o] = cKDTree(S).query(Q[o])[0]
    unc = outside_distance_hulls(S, hulls)
    st = lambda x: {"median_mm": round(float(np.median(x)) * 1000, 2), "p95_mm": round(float(np.percentile(x, 95)) * 1000, 2), "max_mm": round(float(x.max()) * 1000, 2)}
    return {"collision_beyond_drawn": st(prot), "drawn_outside_collision": st(unc)}
