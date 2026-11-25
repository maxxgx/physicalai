"""Write the plugin's bucket_sort_desk scene (urdf/scenes/bucket_sort_desk/scene.xml) from calibration.json:
the real white desk, the robot's base plate, four buckets (black, green, red, blue) in the measured row,
the three task objects and the overview camera on its pole as measured from the splat.

usage: build_desk_scene.py PLUGIN_DIR"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT

plugin = Path(sys.argv[1]); cal = json.load(open(ROOT / "calibration.json"))
src = (plugin / "urdf/scenes/bucket_sort/scene.xml").read_text()
G = cal["sfm_to_world"]; Ms, ts, sc = np.array(G["rotation"]), np.array(G["translation"]), G["scale"]
B = json.load(open(ROOT / "align/buckets3d.json")); sb = B["s_units_per_m"]; pl = B["plane"]
o, u, v, n = (np.array(pl[k]) for k in "ouvn")
def plane_to_world(q):  # bucket-scale plane metres -> world
    P = o + sb * (q[0] * u + q[1] * v + q[2] * n); return sc * Ms @ P + ts
h = -cal["desk_top_z"]  # the base plate: the scene keeps the desk top at z = 0 and mounts the robot on the plate
zt = 0.0
def plane_to_scene(q): return plane_to_world(q) + [0, 0, h]
# desk: a box between the front and back edges, laterally from the measured ends
d = cal["desk"]; fd = np.array(d["front_edge_dir"]); fp = np.array(d["front_edge_point"]); nrm = np.array([-fd[1], fd[0]])
if nrm @ (np.array(d["back_edge_point"]) - fp) < 0: nrm = -nrm
depth = d["depth_m"]
ends = [plane_to_world(q) for q in ((-0.70, 0.0, 0.0), (0.95, 0.0, 0.0))]  # left and right ends in plane x (bucket metres)
lat = sorted(float((e[:2] - fp) @ fd) for e in ends)
centre = fp + fd * (lat[0] + lat[1]) / 2 + nrm * depth / 2; half = ((lat[1] - lat[0]) / 2, depth / 2)
desk_yaw = float(np.arctan2(fd[1], fd[0]))
# the camera: the white body above the ball head (median of its splat points). Its orientation is not measurable
# from outside (the body's thinnest axis, a 277-point blob, gives 29 deg down, which would put the task at the image's
# bottom edge); it is aimed at bucket_sort's task point, as the real one was set up to the scene's schematic. Level image.
# the bar's centre: its back face (the RealSense label between two mounting holes) triangulated from hand marks in two
# allowed frames 50 s apart (f00236, f00991; 1-2.5 px reprojection), plus half the bar's depth toward the front. The
# back face's normal faces within 3 deg of the task point horizontally; the pitch is taken from that aim.
_mk = json.load(open(ROOT / "align/overview_marks2.json")); _f = -np.array(_mk["normal_back"])
eye = np.array(_mk["back"]) + 0.0125 * _f
aim = np.array([0.45, 0.0, 0.05])
z = aim - eye; z /= np.linalg.norm(z)
x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(x, z)  # MuJoCo cameras look along -z: xyaxes = (x, y)
xy = f"{x[0]:.4f} {x[1]:.4f} {x[2]:.4f} {y[0]:.4f} {y[1]:.4f} {y[2]:.4f}"
clamp = plane_to_scene((-0.481, -0.237, 0.0))
# the pole: vertical from its clamp up to about 0.43 m (a RANSAC line through the splat's dark points, align/pole.json),
# then bent toward the camera head
POLE = json.load(open(ROOT / "align/pole.json")); pole_bot = np.array(POLE["bottom"])
_u = np.array(POLE["top"]) - pole_bot; pole_knee = pole_bot + _u * (0.43 / _u[2])

out = src.split("<worldbody>")[0].replace('assetdir="assets"', 'assetdir="../bucket_sort/assets"').replace("bucket_sort: a Franka FR3 on a table with three buckets (black, blue, red)",
    "bucket_sort_desk: the real reBot B601 desk (real2sim) with four buckets (black, green, red, blue)")
out = out.replace('<material name="bucket_red"', '<material name="bucket_green" rgba="0.10 0.55 0.27 1" specular="0.3" shininess="0.4"/>\n    <material name="bucket_red"')
out = out.replace('<material name="table_top" texture="table_top" texrepeat="3 3" texuniform="true" specular="0.15"/>',
                  '<material name="table_top" rgba="0.90 0.90 0.88 1" specular="0.1"/>\n    <material name="plate" rgba="0.09 0.09 0.10 1" specular="0.3"/>\n    <material name="pole" rgba="0.08 0.08 0.08 1" specular="0.3"/>')
lines = ["  <worldbody>",
 '    <light pos="0.3 0.6 2.6" dir="-0.1 -0.2 -1" directional="false" cutoff="60" exponent="1" diffuse="0.55 0.55 0.55"/>',
 '    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.45 0.45 0.45" castshadow="false"/>',
 f'    <geom name="floor" type="plane" size="0 0 0.01" pos="0 0 {zt - 0.75:.4f}" material="floor" condim="3" friction="0.5 0.005 0.0001"/>',
 f'    <!-- The white desk as measured (calibration.json): {2 * half[0]:.3f} x {2 * half[1]:.3f} m, the robot on a {h * 1000:.1f} mm base plate. -->',
 f'    <geom name="table_top" type="box" pos="{centre[0]:.4f} {centre[1]:.4f} {zt - 0.0125:.4f}" euler="0 0 {desk_yaw:.4f}" size="{half[0]:.4f} {half[1]:.4f} 0.0125" material="table_top" condim="3" friction="0.6 0.005 0.0001"/>',
 f'    <geom name="base_plate" type="box" pos="0 0 {h / 2:.4f}" size="0.07 0.10 {h / 2:.4f}" material="plate"/>']  # 14 x 20 cm, from a straight-down splat render
CLUTTER = json.load(open(ROOT / "align/clutter.json"))
CONTENTS = {"black": "resistor_1", "green": "battery_2", "red": "cpu_0", "blue": "resistor_0"}
MASS = {"resistor": 0.040, "battery": 0.045, "cpu": 0.030}
PARTS = {"resistor": [("resistor_body", "resistor_body"), ("resistor_band_brown", "band_brown"), ("resistor_band_red", "band_red"), ("resistor_band_yellow", "band_yellow")],
         "battery": [("battery_metal", "battery_metal"), ("battery_label", "battery_label"), ("battery_band", "battery_band")],
         "cpu": [("cpu_body", "cpu_body"), ("cpu_marking", "cpu_marking"), ("cpu_legs", "cpu_legs")]}
for col in ("black", "green", "red", "blue"):
    b = cal["buckets"][col]; q = (np.cos(b["yaw"] / 2), np.sin(b["yaw"] / 2))
    body = src[src.index(f'    <body name="bucket_black"'):src.index("    </body>", src.index('<body name="bucket_black"')) + 12]
    body = body.replace("black", col)
    body = body.replace('pos="0.645 0.22 0" quat="0.707107 0 0 0.707107"', f'pos="{b["x"]:.4f} {b["y"]:.4f} {zt:.4f}" quat="{q[0]:.6f} 0 0 {q[1]:.6f}"')
    # the bucket's contents, as in the video: settled piles (settle_clutter.py), geoms of the bucket body so they
    # move with it; the task hides the top ones for a lower fill (TaskLayout.fill)
    items = []
    for k, it in enumerate(CLUTTER[CONTENTS[col]]):
        pos = " ".join(f"{x:.4f}" for x in it["pos"]); quat = " ".join(f"{x:.5f}" for x in it["quat"])
        kind = it["type"]; items.append(f'      <geom class="object_collision" name="bucket_{col}_item{k}" mesh="{kind}_hull" pos="{pos}" quat="{quat}" mass="{MASS[kind]}"/>')
        for part, mat in PARTS[kind]:
            items.append(f'      <geom class="visual" name="bucket_{col}_item{k}_{part}" mesh="{part}" material="{mat}" pos="{pos}" quat="{quat}"/>')
    body = body.replace("    </body>", "\n".join(items) + "\n    </body>")
    lines.append(body.rstrip("\n"))
objs = src[src.index("    <!-- Objects, real size"):src.index('    <frame name="robot_mount"/>')]
lines.append(objs.replace('0.0155"', f'{0.0155 + zt:.4f}"').replace('0.0091"', f'{0.0091 + zt:.4f}"').replace('0.0110"', f'{0.0110 + zt:.4f}"').rstrip())
lines += [f'    <frame name="robot_mount" pos="0 0 {h:.4f}"/>',
 f'    <!-- The overview camera on its leaning pole, left of the robot: position from the splat (the camera body above',
 f'         its ball head), aimed at bucket_sort\'s task point ({aim[0]:.2f}, {aim[1]:.2f}, {aim[2]:.2f}): the real camera was set up',
 f'         to that schematic, and its orientation cannot be measured from outside. Level image. -->',
 f'    <geom name="overview_pole" type="capsule" fromto="{pole_bot[0]:.4f} {pole_bot[1]:.4f} 0.0000 {pole_knee[0]:.4f} {pole_knee[1]:.4f} {pole_knee[2]:.4f}" size="0.011" material="pole"/>',
 f'    <geom name="overview_pole_upper" type="capsule" fromto="{pole_knee[0]:.4f} {pole_knee[1]:.4f} {pole_knee[2]:.4f} {eye[0]:.4f} {eye[1]:.4f} {eye[2] - 0.04:.4f}" size="0.011" material="pole"/>',
 f'    <body name="overview_camera_rig" pos="{eye[0]:.4f} {eye[1]:.4f} {eye[2]:.4f}" xyaxes="{xy}">',
 '      <body name="overview_camera_tilt">', '        <camera name="overview" mode="fixed" fovy="42"/>',
 '        <!-- the housing behind the front face (the camera looks along -z): an Intel RealSense D4xx bar (the label is legible in the video), D435-sized 90 x 25 x 25 mm, silver -->',
 '        <geom name="overview_housing" type="box" pos="0 0 0.0125" size="0.045 0.0125 0.0125" rgba="0.80 0.80 0.80 1" contype="0" conaffinity="0" group="2"/>',
 '      </body>', '    </body>',
 '    <!-- a free viewpoint for renders at other poses (the real2sim compositor moves this body; no geoms) -->',
 '    <body name="render_camera_rig" pos="0 0 2.0">', '      <camera name="render" mode="fixed" fovy="45"/>', '    </body>',
 "  </worldbody>", "</mujoco>", ""]
dst = plugin / "urdf/scenes/bucket_sort_desk/scene.xml"; dst.parent.mkdir(exist_ok=True)
dst.write_text(out + "\n".join(lines))
print("wrote", dst, "| desk", np.round(centre, 3), np.round(half, 3), round(np.degrees(desk_yaw), 1), "| eye", eye.round(3), "| top z", round(zt, 4))
