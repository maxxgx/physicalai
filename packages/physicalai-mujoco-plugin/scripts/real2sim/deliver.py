"""Assemble a delivery folder deliveries/rNN (review/RUBRIC.md section 3): versions, data use, assets, commands.
The MANIFEST's prose sections (declarations, self-assessment) come from MANIFEST_NOTES.md next to this script's output.

usage: deliver.py rNN [--splat splat/v1]"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, held

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--splat", default="splat/v1c"); a = ap.parse_args()
PLUGIN = Path(__file__).resolve().parents[2]; WT = PLUGIN.parents[1]; PY = Path(sys.executable)
out = ROOT / "deliveries" / a.name; out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

# 1. versions
commit = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
diff = subprocess.run(["git", "-C", str(WT), "diff", "HEAD"], capture_output=True, text=True).stdout
untracked = subprocess.run(["git", "-C", str(WT), "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout.split()
(out / "worktree.patch").write_text(diff)
ut = out / "untracked"; ut.mkdir(exist_ok=True)
for f in untracked:
    if "__pycache__" in f or f.endswith(".pyc"): continue
    dst = ut / f; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(WT / f, dst)
scripts = sorted(Path(__file__).parent.glob("*.py"))
sd = out / "scripts"; sd.mkdir(exist_ok=True)
for s in scripts: shutil.copy2(s, sd / s.name)
freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(PY)], capture_output=True, text=True).stdout
(out / "pip_freeze.txt").write_text(freeze)
brush = os.environ.get("BRUSH_APP") or shutil.which("brush_app") or ""
brush_v = subprocess.run([brush, "--version"], capture_output=True, text=True).stdout.strip() if brush and Path(brush).exists() else "unknown"

# 2. data use
train = json.load(open(ROOT / a.splat / "train_list.json"))["train"]
frames = {f"f{f['idx']:05d}.jpg": f["t"] for f in json.load(open(ROOT / "colmap/frames.json"))["frames"]}
lines = []
for n in train:
    t = frames[Path(n).name]; assert not held(Path(n).name); lines.append(f"{ROOT / 'colmap/images' / Path(n).name}\t{t:.3f}")
(out / "TRAIN_IMAGES.txt").write_text("\n".join(lines) + "\n")
(out / "TRAIN_PATHS.txt").write_text("\n".join(l.split("\t")[0] for l in lines) + "\n")
photo_dirs = [ROOT / "video", ROOT / "colmap/images", ROOT / "colmap/brush_all/images", ROOT / "splat/v0/images", ROOT / "splat/v1/images"]
(out / "PHOTO_DIRS.txt").write_text("\n".join(str(p) for p in photo_dirs) + "\n")

# 3. assets
assets = {"scene": PLUGIN / "urdf/scenes/bucket_sort_desk/scene.xml", "calibration": ROOT / "calibration.json",
          "robot_look": ROOT / "align/robot_look.json", "clutter": ROOT / "align/clutter.json",
          "robot_gs": ROOT / "align/robot_gs.ply", "robot_gs_links": ROOT / "align/robot_gs_links.json",
          "dev_split": ROOT / "notes/DEV_SPLIT.json", "overview_marks": ROOT / "align/overview_marks2.json",
          "wrist_marks": ROOT / "align/wrist_cam_marks.json", "pole": ROOT / "align/pole.json",
          "splat": ROOT / a.splat / "out/export_30000.ply"}
for k, p in assets.items():
    shutil.copy2(p, out / ("splat.ply" if k == "splat" else p.name))
snap = out / "plugin"
if snap.exists(): shutil.rmtree(snap)
for sub in ("src", "urdf"):
    shutil.copytree(PLUGIN / sub, snap / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
if (out / "sfm").exists(): shutil.rmtree(out / "sfm")
shutil.copytree(ROOT / "colmap/sfm/0", out / "sfm/distorted")
shutil.copytree(ROOT / "colmap/brush_all/sparse", out / "sfm/pinhole")
shutil.copy2(ROOT / "colmap/pinhole_cameras.json", out / "pinhole_cameras.json")
(out / "real2sim.json").write_text(json.dumps({"kind": "desk_splat", "splat": "splat.ply", "calibration": "calibration.json",
                                                "look": "robot_look.json", "robot_gs": "robot_gs.ply"}, indent=1) + "\n")
hashes = {str(p): sha(p) for p in [*assets.values(), *scripts]}
json.dump({"commit": commit, "untracked": untracked, "brush": brush_v, "sha256": hashes}, open(out / "versions.json", "w"), indent=1)
print("delivery", out, "| commit", commit[:10], "| patch lines", diff.count("\n"), "| train images", len(lines))

# 4. the MANIFEST
R = str(out); PYX = f"REAL2SIM_ROOT={ROOT} {PY} -I {R}/scripts/bucketsim.py"
cmd = lambda c: f"cd {R} && {PYX} {c}"
notes = (ROOT / "notes/MANIFEST_NOTES.md").read_text()
speed = (ROOT / "sim/speed.txt").read_text().strip() if (ROOT / "sim/speed.txt").exists() else "not measured"
man = f"""# Delivery {a.name}: bucket_sort_desk real2sim (reBot B601-RS, four buckets)

## 1. Versions
- Worktree `{WT}` at commit `{commit}`; uncommitted changes in `worktree.patch`, new files in `untracked/`. The plugin as
  run is frozen in `plugin/` (`src/` and `urdf/`); `bucketsim.py` imports it from there.
- Builder scripts: `scripts/` (sha256 in `versions.json`). Python: `{PY}` (`pip_freeze.txt`).
- Brush: `{brush_v or 'brush_app 0.3 (aarch64-apple-darwin release)'}`.

## 2. Data use
- `TRAIN_IMAGES.txt`: the 300 frames that trained the splat, one path per line, a tab, then the video time in seconds;
  `TRAIN_PATHS.txt` has the paths alone (for `phone_eval.py leakcheck`). The same frames gave the arm colours (20 of them,
  `robot_look.py`) and every fit in `calibration.json`.
- `PHOTO_DIRS.txt`: every directory with photos or frames.
- Photo-derived textures: none (section 6). Declared exposures: section 6.

## 3. Assets
- SfM: `sfm/distorted` (SIMPLE_RADIAL 1920x1080, `colmap/images` names) and `sfm/pinhole` (the undistorted PINHOLE model
  of the same SfM, 1880x1057), both COLMAP binary; `pinhole_cameras.json` is the PINHOLE model as {{name: cam_from_world, K}}.
  For `phone_eval.py refs`: `--model sfm/distorted --pinhole sfm/pinhole`.
- Splat: `splat.ply` (3DGS PLY, SH degree 3, SfM frame and units): Brush on the training frames with the robot masked
  (`robot_masks.py`), then the real arm's leftovers deleted in 3D (`clean_splat.py`, 4 cm around the arm, 12 cm around
  the wrist mount).
- Scene: `plugin/urdf/scenes/bucket_sort_desk/scene.xml` (written by `scripts/build_desk_scene.py`), registered as
  `bucket_sort_desk` with the `rebot_b601` profile (`scene_registry.py`, layout `REBOT_DESK_LAYOUT` in `bucket_sort_arms.py`).
- `robot_gs.ply` + `robot_gs_links.json`: the arm's Gaussians and the link each rides on (section 6).
- `real2sim.json`: the kit file the plugin's `--appearance` reads (paths relative to it).
- `calibration.json` (D7 X1), `robot_look.json` (arm colours), `clutter.json` (bucket contents), `DEV_SPLIT.json` (the
  builder's own dev frames, allowed frames only).

## 4. Commands
All run headless on the Mac from a fresh shell; none reads photos (checked under the section 7 sandbox). Times are on the
M4 Max with nothing else running.

- **HELDOUT_RENDER** (matched state; writes `fNNNNN.png`, `_seg.png`, `_outline.png`, `cameras.json`; ~0.5 s/frame)
  `{cmd("heldout_render --frames f00001,f00002 --out DIR")}`
- **VIDEO** (composite along the phone path, 1280x720, 15 fps, demonstrator seed 7; ~0.1 s/frame; `--sbs` puts the real
  video on the left, black in the held-out windows, and reads the MOV, so it is not for the sandbox)
  `{cmd("video --t0 T0 --t1 T1 --seed 7 --out FILE.mp4")}`
- **DEMO** (one episode per seed and pair; JSON per episode with `physics_flags`; ~10 s/episode)
  `{cmd("demo --seeds 1,2,3 --pairs cpu:green,resistor:blue,battery:black --out OUT.json")}`
- **ROBOTCAM** (one episode: `overview.mp4`, `wrist.mp4` at 640x480 30 fps, every 10th frame as PNG with `_seg.png`; ~2 min)
  `{cmd("robotcam --seed S --out DIR")}`
- **SPEED** `{cmd("speed --frames 300")}`  (builder's run: {speed})
- **RAND** (overview + wrist for each seed with every knob drawn; `knobs.json` lists ranges and defaults)
  `{cmd("rand --seeds 1,2,3,4,5,6 --out DIR")}`
- **CAL** `{cmd("cal")}`
- **PLUGIN_CLI** (the scene through the plugin's own command line, headless, from the frozen plugin, with the kit
  `real2sim.json`: the plugin's `overview` and `wrist` streams, the ones Studio records, are the composited ones; the
  autopilot runs the demonstrator, episodes reset themselves; stop with `POST /shutdown`)
  `cd {R} && PYTHONPATH=plugin/src {PY} -m physicalai_mujoco_plugin start --profile rebot_b601 --scene bucket_sort_desk --no-gui --seed 1 --http-port 8931 --appearance real2sim.json`
  then `curl -X POST localhost:8931/autopilot -H 'content-type: application/json' -d '{{"mode":"drive"}}'`,
  `curl localhost:8931/health` (field `episode`: instruction, phase, episode_count) and
  `curl -o overview.jpg localhost:8931/cameras/overview/frame.jpg` (also `wrist`; MJPEG at `/cameras/<name>/mjpeg`).
  The streams publish at the compositor's rate (`SPEED` measures the same compositor for both cameras together), not 30 fps.

## 5. Convenience outputs
`composite_full.mp4`, `sbs_full.mp4` (real half black in the held-out windows and guard), `robotcam_seed5/`, `rand/`.

{notes}"""
(out / "MANIFEST.md").write_text(man); print("MANIFEST", out / "MANIFEST.md")
