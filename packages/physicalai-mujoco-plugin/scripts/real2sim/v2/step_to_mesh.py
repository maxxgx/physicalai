"""r15 (D2, the user's mount): tessellate a STEP file and list its features.

Writes OUT_DIR/mesh.stl (the STEP's own units, mm for this file) and OUT_DIR/features.json: per solid the bounding box,
and every cylindrical face (radius, axis direction, axis point, angular extent, height) and planar face (normal, centroid,
area), so holes (bolt pattern) and the sensor seat can be read off. Needs OpenCascade (cadquery-ocp); runs in its own venv
(not the plugin's): `uv venv ocpenv && uv pip install -p ocpenv/bin/python cadquery-ocp numpy`.
usage: step_to_mesh.py STEP OUT_DIR [--deflection 0.05]"""
import json, sys
from pathlib import Path
import numpy as np
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

step, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
defl = float(sys.argv[sys.argv.index("--deflection") + 1]) if "--deflection" in sys.argv else 0.05
r = STEPControl_Reader(); assert r.ReadFile(step) == IFSelect_RetDone; r.TransferRoots(); shape = r.OneShape()
BRepMesh_IncrementalMesh(shape, defl, False, 0.2, True)

V, F, feats = [], [], {"solids": [], "cylinders": [], "planes": []}
ex = TopExp_Explorer(shape, TopAbs_SOLID)
while ex.More():
    b = Bnd_Box(); BRepBndLib.Add_s(ex.Current(), b); mn, mx = b.CornerMin(), b.CornerMax(); x0, y0, z0, x1, y1, z1 = mn.X(), mn.Y(), mn.Z(), mx.X(), mx.Y(), mx.Z()
    p = GProp_GProps(); BRepGProp.VolumeProperties_s(ex.Current(), p)
    feats["solids"].append({"bbox": [x0, y0, z0, x1, y1, z1], "volume": p.Mass()}); ex.Next()
ex = TopExp_Explorer(shape, TopAbs_FACE)
while ex.More():
    f = TopoDS.Face(ex.Current()); loc = TopLoc_Location(); tri = BRep_Tool.Triangulation_s(f, loc)
    if tri is not None:
        T = loc.Transformation(); base = len(V)
        for i in range(1, tri.NbNodes() + 1):
            q = tri.Node(i).Transformed(T); V.append((q.X(), q.Y(), q.Z()))
        rev = f.Orientation() == TopAbs_REVERSED
        for i in range(1, tri.NbTriangles() + 1):
            a, b_, c = tri.Triangle(i).Get(); F.append((base + a - 1, base + c - 1, base + b_ - 1) if rev else (base + a - 1, base + b_ - 1, base + c - 1))
    s = BRepAdaptor_Surface(f); p = GProp_GProps(); BRepGProp.SurfaceProperties_s(f, p); cg = p.CentreOfMass()
    if s.GetType() == GeomAbs_Cylinder:
        c = s.Cylinder(); ax = c.Axis(); d = ax.Direction(); o = ax.Location()
        feats["cylinders"].append({"radius": c.Radius(), "axis": [d.X(), d.Y(), d.Z()], "point": [o.X(), o.Y(), o.Z()], "centroid": [cg.X(), cg.Y(), cg.Z()],
                                   "u_span_deg": float(np.degrees(s.LastUParameter() - s.FirstUParameter())), "v_span": s.LastVParameter() - s.FirstVParameter(),
                                   "area": p.Mass(), "reversed": f.Orientation() == TopAbs_REVERSED})
    elif s.GetType() == GeomAbs_Plane:
        n = s.Plane().Axis().Direction(); sg = -1 if f.Orientation() == TopAbs_REVERSED else 1
        feats["planes"].append({"normal": [sg * n.X(), sg * n.Y(), sg * n.Z()], "centroid": [cg.X(), cg.Y(), cg.Z()], "area": p.Mass()})
    ex.Next()
V, F = np.asarray(V), np.asarray(F)
with open(out / "mesh.stl", "w") as fh:   # ascii STL, the STEP's units
    fh.write("solid mount\n")
    for t in F:
        a, b_, c = V[t]; n = np.cross(b_ - a, c - a); n = n / (np.linalg.norm(n) + 1e-12)
        fh.write(f"facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n outer loop\n" + "".join(f"  vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in (a, b_, c)) + " endloop\nendfacet\n")
    fh.write("endsolid mount\n")
feats["mesh"] = {"vertices": len(V), "triangles": len(F), "bbox": V.min(0).tolist() + V.max(0).tolist(), "deflection": defl}
json.dump(feats, open(out / "features.json", "w"), indent=1)
print(json.dumps({"solids": feats["solids"], "mesh": feats["mesh"], "cylinders": len(feats["cylinders"]), "planes": len(feats["planes"])}))
