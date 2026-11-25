"""Refine each bucket's planar pose (x, y, yaw on the desk) so its CAD silhouette matches the bucket's colour silhouette
in ALLOWED phone frames (photos are used for fitting only, never rendered). Per frame (occluders: only buckets nearer the camera): the real mask = the outer contour
of the bucket's colour near its projection (holes from its contents filled); the sim mask = the CAD mesh's faces filled,
at the SfM pose; pixels inside the other buckets' projections and outside the image are ignored. Loss = median over
frames of 1 - IoU (robust to frames where a hand covers or moves a bucket). Writes align/buckets_photo.json and a check
sheet notes/sheets/buckets_photo.jpg (allowed frames only).  usage: fit_buckets_photo.py [COLOURS]"""
import json, os, sys, numpy as np, cv2, mujoco
from pathlib import Path
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import held
ROOT = Path(os.environ["REAL2SIM_ROOT"]).expanduser()
PLUG = Path(__file__).resolve().parents[3]
cal = json.load(open(ROOT / "calibration.json")); G = cal["sfm_to_world"]; M, t, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
plate = -cal["desk_top_z"]
cams = json.load(open(ROOT / "colmap/pinhole_cameras.json"))
DS = 4
m = mujoco.MjModel.from_xml_path(str(PLUG / "urdf/scenes/bucket_sort_desk_v2/scene.xml"))
mid = m.mesh("bucket").id; va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
V0 = m.mesh_vert[va:va + vn].astype(float); F0 = m.mesh_face[fa:fa + fn]
gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("bucket_red").id and m.geom_dataid[g] == mid][0]
# the mesh in the bucket body's frame (geom pos/quat inside the body)
Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[gid]); Vb = V0 @ Rg.reshape(3, 3).T + m.geom_pos[gid]
KZ = float(os.environ.get("KZ", "1"))   # experiment: the bucket's height scaled (the CAD is 120 mm tall)
Vb = Vb * [1.0, 1.0, KZ]
OUT = os.environ.get("FIT_OUT")         # experiment: write the results elsewhere
print("bucket mesh", vn, "verts", fn, "faces; body-frame extent", Vb.min(0).round(3), Vb.max(0).round(3), flush=True)
def world(p):   # x, y, yaw -> mesh vertices in the SfM frame
    x, y, yaw = p; c, s = np.cos(yaw), np.sin(yaw); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    W = Vb @ R.T + [x, y, 0]; W[:, 2] -= plate
    return (W - t) @ M / sc
def project(Ps, cam):
    T = np.array(cam["cam_from_world"]); K = np.array(cam["K"]) / [[DS], [DS], [1]]
    Xc = Ps @ T[:, :3].T + T[:, 3]; z = Xc[:, 2]; uv = (Xc[:, :2] / z[:, None]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z
def sil(p, cam, w, h):
    uv, z = project(world(p), cam)
    if (z <= 0.05).any(): return None
    out = np.zeros((h, w), np.uint8); tri = np.round(uv[F0] * 8).astype(np.int32)
    cv2.fillPoly(out, list(tri), 1, shift=3); return out
HSV = {"green": lambda H, S, V: (H >= 45) & (H <= 85) & (S > 80), "red": lambda H, S, V: ((H <= 10) | (H >= 165)) & (S > 110) & (V > 60),
       "blue": lambda H, S, V: (H >= 98) & (H <= 125) & (S > 110),
       "black": lambda H, S, V: (V < 75) & (S < 110)}   # dark grey plastic on a white desk; contents filled as holes
colours = sys.argv[1].split(",") if len(sys.argv) > 1 else ["green", "red", "blue"]
init = {c: np.array([cal["buckets"][c]["x"], cal["buckets"][c]["y"], cal["buckets"][c]["yaw"]]) for c in cal["buckets"]}
if os.environ.get("INIT_POSES"):   # r14 (hand-event rule): start from (and compare against) the pose carried over
    for c, v in json.loads(os.environ["INIT_POSES"]).items(): init[c] = np.array(v, float)
names = sorted(n for n in cams if not held(n))
if os.environ.get("FRAME_RANGE"):   # experiment: only frames fNNNNN in [lo, hi]
    _lo, _hi = map(int, os.environ["FRAME_RANGE"].split(",")); names = [n for n in names if _lo <= int(n[1:6]) <= _hi]
STEP = int(os.environ.get("FRAME_STEP", "3"))
data = {c: [] for c in colours}
for n in names[::STEP]:
    cam = cams[n]; w, h = cam["w"] // DS, cam["h"] // DS
    sims = {c: sil(init[c], cam, w, h) for c in init}
    if any(s is None for s in sims.values()): continue
    img = None
    for c in colours:
        s = sims[c]; ys, xs = np.nonzero(s)
        if os.environ.get("PARTIAL"):   # experiment: also buckets partly out of view (>= 40 % of their projection in it)
            uv_, _ = project(world(init[c]), cam); inside = ((uv_[:, 0] >= 0) & (uv_[:, 0] < w) & (uv_[:, 1] >= 0) & (uv_[:, 1] < h)).mean()
            if len(xs) < 600 or inside < 0.4: continue
        elif len(xs) < 600 or xs.min() < 3 or ys.min() < 3 or xs.max() > w - 4 or ys.max() > h - 4: continue   # whole bucket in view
        if img is None:
            img = cv2.resize(cv2.imread(str(ROOT / "colmap/pinhole/images" / n)), (w, h), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
        near = cv2.dilate(s, np.ones((41, 41), np.uint8)) > 0
        col = (HSV[c](hsv[..., 0], hsv[..., 1], hsv[..., 2]) & near).astype(np.uint8)
        if c == "black":   # dark pixels inside a coloured bucket (the black chips in the blue one) are not this bucket
            for oc in ("green", "red", "blue"):
                om = HSV[oc](hsv[..., 0], hsv[..., 1], hsv[..., 2]).astype(np.uint8)
                om = cv2.morphologyEx(om, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
                oc_cnts, _ = cv2.findContours(om, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                filled = np.zeros_like(om)
                cv2.drawContours(filled, [cc for cc in oc_cnts if cv2.contourArea(cc) > 300], -1, 1, -1)
                col[filled > 0] = 0
        col = cv2.morphologyEx(col, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        k, lab, st, _ = cv2.connectedComponentsWithStats(col)
        if k < 2: continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); comp = (lab == i).astype(np.uint8)
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); real = np.zeros_like(comp)
        cv2.drawContours(real, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
        # ignore only the pixels of buckets NEARER the camera (they may hide this one); a bucket behind it hides nothing
        def depth(col):
            T = np.array(cam["cam_from_world"]); b0 = init[col]; ctr = world(np.array([b0[0], b0[1], b0[2]]))[:].mean(0)
            return float((T[:, :3] @ ctr + T[:, 3])[2])
        dc = depth(c); others = np.zeros_like(s)
        for o, so in sims.items():
            if o != c and depth(o) < dc: others |= so
        valid = ~(others > 0)
        iou0 = ((real & s)[valid].sum()) / max(((real | s)[valid]).sum(), 1)
        if iou0 < 0.35: continue
        data[c].append((n, cam, w, h, real, valid))
res = {}
for c in colours:
    D = data[c]
    if os.environ.get("CLOSE_Q"):   # only the frames where the bucket is largest (close views)
        _a = np.array([float(r[4].sum()) for r in D]); D = [r for r, a_ in zip(D, _a) if a_ >= np.quantile(_a, float(os.environ["CLOSE_Q"]))]
    print(c, len(D), "frames", flush=True)
    def loss(p, per=False):
        L = []
        for n, cam, w, h, real, valid in D:
            s = sil(p, cam, w, h)
            if s is None: L.append(1.0); continue
            L.append(1 - (real & s)[valid].sum() / max((real | s)[valid].sum(), 1))
        if per: return np.array(L)
        if os.environ.get("AREA_WEIGHT"):   # close views count by their size: pixel gaps there are what an overlay measures
            L = np.array(L); wts = np.array([float(real.sum()) for _, _, _, _, real, _ in D]); keep = L <= np.quantile(L, 0.85)
            return float((L[keep] * wts[keep]).sum() / wts[keep].sum())
        return float(np.median(L) * 0.7 + np.mean(np.sort(L)[: max(1, int(0.8 * len(L)))]) * 0.3)
    p0 = init[c]; l0 = loss(p0)
    best = (l0, p0)
    for dx in (-0.03, -0.015, 0, 0.015, 0.03):
        for dy in (-0.03, -0.015, 0, 0.015, 0.03):
            for dyaw in np.radians([-8, 0, 8]):
                p = p0 + [dx, dy, dyaw]; l = loss(p)
                if l < best[0]: best = (l, p)
    r = minimize(loss, best[1], method="Nelder-Mead", options={"xatol": 5e-4, "fatol": 1e-4, "initial_simplex": best[1] + np.array([[0, 0, 0], [0.008, 0, 0], [0, 0.008, 0], [0, 0, np.radians(3)]])})
    per0, per1 = loss(p0, True), loss(r.x, True)
    res[c] = {"x": float(r.x[0]), "y": float(r.x[1]), "yaw": float(r.x[2]), "frames": len(D), "frame_list": [n for n, *_ in D], "iou_median_before": float(1 - np.median(per0)),
              "iou_median_after": float(1 - np.median(per1)), "shift_m": float(np.hypot(*(r.x[:2] - p0[:2]))), "dyaw_deg": float(np.degrees(r.x[2] - p0[2]))}
    print(c, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[c].items()}, flush=True)
if OUT:
    json.dump({"kz": KZ, "buckets": res}, open(OUT, "w"), indent=1); sys.exit()
_prev = json.load(open(ROOT / "align/buckets_photo.json"))["buckets"] if (ROOT / "align/buckets_photo.json").exists() else {}
_prev.update(res)
json.dump({"method": __doc__.split("\n")[0], "downscale": DS, "buckets": _prev}, open(ROOT / "align/buckets_photo.json", "w"), indent=1)
# check sheet: real mask (green), sim silhouette at the fit (magenta outline), allowed frames only
tiles = []
for c in colours:
    D = data[c]; p = np.array([res[c]["x"], res[c]["y"], res[c]["yaw"]])
    for n, cam, w, h, real, valid in [D[i] for i in np.linspace(0, len(D) - 1, 4).astype(int)]:
        assert not held(n)
        img = cv2.resize(cv2.imread(str(ROOT / "colmap/pinhole/images" / n)), (w, h), interpolation=cv2.INTER_AREA)
        img[real > 0] = (0.5 * img[real > 0] + [0, 120, 0]).astype(np.uint8)
        s = sil(p, cam, w, h); cnts, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, (255, 0, 255), 1); cv2.putText(img, f"{c} {n}", (4, 14), 0, 0.45, (0, 255, 255), 1)
        tiles.append(img)
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite(str(ROOT / "notes/sheets/buckets_photo.jpg"), np.vstack(rows))
