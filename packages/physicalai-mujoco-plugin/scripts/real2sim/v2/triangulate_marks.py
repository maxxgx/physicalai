"""Triangulate hand marks (MARKS.json: {"marks": {frame: {point: [u, v]}}}) with the SfM model's poses and its distorted
camera (linear DLT on normalised rays, then a Gauss-Newton refinement); prints each point (SfM frame) and its
reprojection errors. usage: triangulate_marks.py MODEL_DIR MARKS.json OUT.json"""
import json, sys, numpy as np, pycolmap
r = pycolmap.Reconstruction(sys.argv[1]); M = json.load(open(sys.argv[2]))
by = {im.name: im for im in r.images.values()}
pts = sorted({p for v in M["marks"].values() for p in v})
out = {}
for p in pts:
    obs = [(by[f], np.array(v[p])) for f, v in M["marks"].items() if p in v and f in by]
    A = []; Ps = []
    for im, uv in obs:
        cam = r.cameras[im.camera_id]; x = cam.cam_from_img(uv[None])[0]
        T = im.cam_from_world().matrix(); Ps.append((cam, T))
        A.append(x[0] * T[2] - T[0]); A.append(x[1] * T[2] - T[1])
    X = np.linalg.svd(np.array(A))[2][-1]; X = X[:3] / X[3]
    errs = []
    for (cam, T), (_, uv) in zip(Ps, obs):
        pc = T[:, :3] @ X + T[:, 3]; proj = cam.img_from_cam(pc[None])[0]; errs.append(float(np.linalg.norm(proj - uv)))
    out[p] = {"X": X.tolist(), "reproj_px": [round(e, 2) for e in errs], "n_views": len(obs)}
    print(p, np.round(X, 4), "reproj px", [round(e, 2) for e in errs])
json.dump(out, open(sys.argv[3], "w"), indent=1)
