"""Freeze a delivery REAL2SIM_ROOT/deliveries/rNN for the v2 review (RUBRIC section 7): versions, data use, assets, the
kit, commands. The prose sections (change log, declarations) come from notes/MANIFEST_NOTES_V2.md.
usage: deliver_v2.py rNN --splat PLY --robot-gs PLY [--speed "x fps"]"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import held, ROOT  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--splat", required=True); ap.add_argument("--robot-gs", required=True)
ap.add_argument("--train-list", required=True, help="the splat's train_list.json"); ap.add_argument("--speed", default="not measured")
ap.add_argument("--no-video", action="store_true", help="skip the R4 phone-path video (a dry run)")
a = ap.parse_args()
HERE = Path(__file__).resolve().parent; PLUGIN = HERE.parents[2]; WT = PLUGIN.parents[1]; PY = Path(sys.executable)
out = ROOT / "deliveries" / a.name
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

# 1. versions
commit = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
diff = subprocess.run(["git", "-C", str(WT), "diff", "HEAD"], capture_output=True, text=True).stdout
untracked = [f for f in subprocess.run(["git", "-C", str(WT), "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout.split()
             if "__pycache__" not in f and not f.endswith(".pyc") and not f.endswith("MUJOCO_LOG.TXT")]
(out / "worktree.patch").write_text(diff)
for f in untracked:
    dst = out / "untracked" / f; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(WT / f, dst)
sd = out / "scripts"; sd.mkdir()
scripts = sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sh")) + [HERE.parent / n for n in ("cad.py", "common.py", "fit_robot.py", "robot_masks.py", "robot_gaussians.py", "clean_splat.py")]
for s in scripts: shutil.copy2(s, sd / s.name)
shutil.copy2(ROOT / "review/HELDOUT_PHONE.json", out / "HELDOUT_PHONE.json")   # the rule's own copy: commands never read review/
(out / "pip_freeze.txt").write_text(subprocess.run(["uv", "pip", "freeze", "--python", str(PY)], capture_output=True, text=True).stdout)
brush = Path.home() / "real2sim-data/tools/brush/brush-app-aarch64-apple-darwin/brush_app"
brush_v = subprocess.run([str(brush), "--version"], capture_output=True, text=True).stdout.strip()
import mujoco
# 2. data use
frames = {f"f{f['idx']:05d}.jpg": f["idx"] / 30 for f in json.load(open(ROOT / "colmap/frames.json"))["allowed"]}
train = json.load(open(a.train_list))["train"]
lines = []
for n in train:
    assert not held(n), n; lines.append(f"{ROOT / 'colmap/images' / n}\t{frames[n]:.3f}")
extra = (ROOT / "notes/FIT_IMAGES.txt").read_text().splitlines() if (ROOT / "notes/FIT_IMAGES.txt").exists() else []
for l in extra:
    n = Path(l.split()[0]).name; assert not held(n), n
    if l.split()[0] not in {x.split("\t")[0] for x in lines}: lines.append(f"{ROOT / 'colmap/images' / n}\t{int(n[1:6]) / 30:.3f}")
(out / "TRAIN_PATHS.txt").write_text("\n".join(lines) + "\n")
photo_dirs = [Path.home() / "Downloads", ROOT / "video", ROOT / "colmap/images", ROOT / "colmap/pinhole/images", ROOT / "colmap/old_v0/images",
              ROOT / "splat", ROOT / "align/masks_robot", ROOT / "notes/sheets", ROOT / "real", ROOT / "review/export/robotcam_allowed", Path.home() / "real2sim-data/office-threejs/app/data/scans"]
(out / "PHOTO_DIRS.txt").write_text("\n".join(str(p) for p in photo_dirs) + "\n")
# 3. assets
shutil.copy2(a.splat, out / "splat.ply"); shutil.copy2(a.robot_gs, out / "robot_gs.ply")
shutil.copy2(Path(a.robot_gs).with_name(Path(a.robot_gs).stem + "_links.json"), out / "robot_gs_links.json")
for f in ("calibration.json", "align/clutter_v2.json", "align/robot_look.json", "colmap/pinhole_cameras.json", "align/overview_marks.json",
          "align/overview_tri.json", "align/overview_cam.json", "align/buckets3d.json", "align/robot_fit.json", "align/lidar_reg.json",
          "align/buckets_photo.json", "align/bucket_height.json", "align/bucket_moves.json", "align/bucket_moves_frames.json", "align/desk.json",
          "align/arm_photo.json", "notes/DISCLOSURE.md"):
    if (ROOT / f).exists(): shutil.copy2(ROOT / f, out / Path(f).name)
# r08+: the real robot-episode records (allowed episodes only; each fit carries its frame list) and the declared
# exposures; the preview stills of the round (journal/current) go to preview/
real = out / "real"; real.mkdir()
for f in ("real/fit_overview.json", "real/fit_gripper.json", "real/fit_gripper2.json", "real/colour_overview.json",
          "real/colour_gripper.json", "real/noise_real.json", "real/dscheck_real.json", "real/grip_widths.json",
          "real/states_x5.json", "real/frames_x5/decode_record.json", "notes/EXPOSURES.md",
          "real/latency_overview.json", "real/latency_gripper.json", "real/finger_gap.json", "align/object_size.json",
          "real/upright_count.json", "real/finger_visual.json", "real/finger_mount.json", "real/flicker_real.json",
          "align/bucket_layouts_episodes.json", "sim/r10/kinds_5600.json", "sim/r10/kinds_record_7400.json",
          "sim/r10/demo_all_pairs/demo_all_pairs.json", 
          "sim/r10/studio_dup/dup_stamp2.json", "sim/r14/devreplay/dev_replay_metrics.json", "sim/r10/stacking_rate.json",
          "real/finger_joint.json", "real/finger_tips.json", "real/t3_measure.json", "real/still_codec_real.json",
          "real/noise_lowfreq_fit.json", "real/still_motion_real.json", "real/layout_spread.json", "sim/r11/kinds_default_5700f.json",
          "sim/r11/kinds_rand_7500f.json", "sim/r11/kinds_record_7400c.json", "sim/r11/kinds_record_307720.json",
          "sim/r11/kinds_default_5600d.json", "sim/r11/studio_dup/dup.json", "sim/r11/studio_dup/dup2.json", "sim/r11/speed_1_30.txt",
          "real/hold_openings.json", "real/headlight_fit.json", "real/latency_overview_close.json", "real/bucket_gain_fit.json",
          "sim/r12/kinds_default_5700_cut15.json", "sim/r12/speed_seed1.txt",
          "real/r13/wrist_joint_fit.json", "real/r13/wrist_bucket_edges.json", "real/r13/wrist_bucket_edges_joff.json",
          "real/r13/overview_arm_fit.json", "real/r13/arm_both_views.json", "sim/r13/audit_visual_collision.json",
          "sim/r13/kinds_default_5700.json", "sim/r13/kinds_record_7400.json", "align/collision_r13.json",
          "sim/pilot/first_frame_check.json", "sim/r13/check/first_frame_check_r13.json", "sim/r13/check/dscheck_lite_r13.txt",
          "sim/r13/check/clips.json", "sim/r13/check/clip_check.md", "sim/r13/speed_seed1.txt",
          "sim/r13/bucket_regions_r13.json", "sim/r13/bucket_regions_r12boxes.json",
          "real/r14/resistor_hold_width.json", "real/r14/resistor_hold_sheet.jpg", "real/r14/arm_both_views_6j.json",
          "align/hand_events.json", "align/hand_events_applied.json", "sim/r14/kinds_default_5700.json", "sim/r14/kinds_record_probe30b.json",
          "sim/r14/speed_seed1.txt", "sim/r14/check/first_frame_check_r14.json", "sim/r14/check/dscheck_lite_r14.txt",
          "sim/r14/audit_visual_collision.json", "sim/r14/bucket_regions_r14.json", "sim/r14/audit_link6.json", "sim/r14/kinds_gap_probe30.json", "real/r14/tone_fit.json", "sim/r14/kinds_default_5700_t6.json",
          "sim/r14/check/first_frame_check_free.json", "sim/r14/check/dscheck_lite_free.txt",
          # r15
          "mount/mount_link6.json", "mount/features.json", "real/r15/wrist_step_eval.json", "real/r15/wrist_step_eval_datasheet.json",
          "real/r15/mount_colour.json", "sim/r15/real_drift_stats.json", "sim/r15/real_still_stats.json", "sim/r15/drift_stats.json",
          "sim/r15/sim_still_by_phase.txt", "sim/r15/drift_stats_final.json", "sim/r15/d4_648379_repro.log", "sim/r15/d4_648379_ccd.log", "sim/r15/d4_648379_after_slivers15.log",
          "sim/r15/slip_ablate.txt", "sim/r15/thicken_slivers.json", "sim/r15/thicken_slivers15.log", "sim/r15/physics_step_time.txt",
          "sim/r15/r6_planner_timing.txt", "sim/r15/demo_all_pairs.txt", "sim/r15/diag_9107.txt", "sim/r15/diag_7403.txt", "sim/r15/diag_housing.txt", "sim/r15/abl_5783.txt", "sim/r15/abl_default_fails.txt", "sim/r15/tests_full.txt", "sim/r15/speed_paths.txt", "sim/r15/speed_w2_fixed.txt", "sim/r15/speed_ov2.txt", "sim/r15/speed_final.txt",
          "sim/r15/kinds_default_5700_final.json", "sim/r15/kinds_record_7400_final.json", "sim/r15/kinds_record_7400_r15_multiccdoff.json",
          "sim/r15/kinds_compare_default.json", "sim/r15/kinds_compare_record.json",
          "sim/r15/check/first_frame_check_r15.json", "sim/r15/check/dscheck_lite_r15.txt", "notes/R15_RESULTS.md"):
    if (ROOT / f).exists(): shutil.copy2(ROOT / f, real / Path(f).name)
for f in sorted((ROOT / "sim").glob("noise_dryrun_r08*.json")): shutil.copy2(f, real / f.name)
# r16: the round's records under r16/, their paths relative to REAL2SIM_ROOT kept (several sets share file names);
# JSON and text only (no logs, no images)
for pat in ("real/r16/*.json", "sim/r16/*.json", "sim/r16/*.txt", "sim/r16/ticks/*.json", "sim/r16/d3a/*.json", "sim/r16/d4/*.json",
            "sim/r16/hit5/*.json", "sim/r16/attrib/*.json", "sim/r16/final2/summary.txt", "sim/r16/final_accel/README.txt",
            "sim/r16/fitlayout/*.json", "sim/r16/devreplay_*/metrics.json", "sim/r16/check4/*.json", "sim/r16/check4/*.txt",
            "notes/R16_PLAN.md",
            # r17
            "sim/r17/*.json", "sim/r17/*.txt", "sim/r17/finger_fade.jpg", "sim/r17/keyframes_handover.jpg", "sim/r17/d3b_r17.jpg",
            "sim/r17/d3b_f17.jpg", "sim/r17/d3b_tint.jpg", "sim/r17/d4/*.json", "sim/r17/margin/*.json", "sim/r17/reuse/*.json",
            "sim/r17/ahead_*.json", "sim/r17/fitlayout_jz/*.json", "sim/r17/ovcand/*.json", "sim/r17/check5/*.json",
            "sim/r17/check5/*.txt", "sim/r17/glitch/*.json", "sim/r17/glitch/*.txt", "sim/r17/glitch/slip_*.log",
            "sim/r17/devreplay_*/metrics.json", "notes/R17_STILL_ANALYSIS.md",
            "sim/r17/glitch/*.xml", "sim/r17/check6/*.json", "sim/r17/check6/*.txt",   # r18 item 7: the ablation scenes
            # r18
            "sim/r18/*.json", "sim/r18/*.txt", "sim/r18/*.xml", "sim/r18/glitch/*.json", "sim/r18/glitch/*.txt",
            "sim/r18/glitch/*.xml", "sim/r18/timing/*.json", "sim/r18/timing_val/*.json", "sim/r18/timing_abl/*.json",
            "sim/r18/parallel/*.json", "sim/r18/identity/*.json", "sim/r18/speed/*.json", "sim/r18/speed/*.txt",
            "sim/r18/rebaseline/*.json", "sim/r18/studio_gate/*.json", "sim/r18/studio_gate/*.out",
            "sim/r18/studio_gate/status.txt", "sim/r18/studio_gate/run_gate.sh", "notes/R18_*.md"):
    for f in sorted(ROOT.glob(pat)):
        rnd = next((r for r in ("r18", "r17") if f"/{r}/" in str(f) or f"{r.upper()}_" in f.name), "r16")
        dst = out / rnd / f.relative_to(ROOT); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
if (ROOT / "real/hand_sheets").is_dir():   # r14: the hand-event review sheets (allowed phone frames only)
    shutil.copytree(ROOT / "real/hand_sheets", real / "hand_sheets")
if (ROOT / "journal/current").is_dir() and any((ROOT / "journal/current").iterdir()):
    shutil.copytree(ROOT / "journal/current", out / "preview")
snap = out / "plugin"
for sub in ("src", "urdf"):
    shutil.copytree(PLUGIN / sub, snap / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copytree(ROOT / "colmap/sfm/best2", out / "sfm/distorted"); shutil.copytree(ROOT / "colmap/pinhole/sparse", out / "sfm/pinhole")
kit = json.load(open(ROOT / "kit/real2sim.json"))
kit.update({"splat": "splat.ply", "calibration": "calibration.json", "look": "robot_look.json" if (out / "robot_look.json").exists() else None,
            "robot_gs": "robot_gs.ply", "pinhole_cameras": "pinhole_cameras.json"})
(out / "real2sim.json").write_text(json.dumps(kit, indent=1) + "\n")
labels = {"1": "black", "2": "green", "3": "red", "4": "blue", "5": "desk top", "10": "robot incl. gripper", "11": "overview camera housing + mount",
          "12": "wrist camera housing", "20-99": "task objects (resistor 20, battery 21, cpu 22, pool objects 30+ in scene order)",
          "bucket_order": ["black", "green", "red", "blue"]}
(out / "labels.json").write_text(json.dumps(labels, indent=1) + "\n")
hashes = {str(p): sha(p) for p in [*scripts, out / "splat.ply", out / "robot_gs.ply", out / "calibration.json"]}
json.dump({"commit": commit, "untracked": untracked, "brush": brush_v, "mujoco": mujoco.__version__, "python": str(PY), "sha256": hashes},
          open(out / "versions.json", "w"), indent=1)
print("delivery", out, "| commit", commit[:10], "| patch lines", diff.count("\n"), "| train images", len(lines))

# 4. MANIFEST
R = str(out); ENV = f"REAL2SIM_ROOT={ROOT} KIT={R}/real2sim.json PHYSICALAI_REBOT_B601_JOINT_ZERO={R}/real2sim.json"
RUN = "/tmp/r2s_v2_run"   # every command runs from here: no __pycache__ or MUJOCO_LOG.TXT lands in the delivery
cmd = lambda c: f"mkdir -p {RUN} && cd {RUN} && {ENV} {PY} -I -B {R}/scripts/bucketsim_v2.py {c}"   # -I ignores PYTHON* env vars: -B keeps bytecode out
notes = (ROOT / "notes/MANIFEST_NOTES_V2.md").read_text()
man = f"""# Delivery {a.name}: bucket_sort_desk_v2 real2sim (reBot B601, front overview camera, Orbbec Gemini 305 x2)

## 1. Versions
- Worktree `{WT}` (branch max/bucket-real2sim-v2) at commit `{commit}`; uncommitted changes in `worktree.patch` (applies to
  that commit with `git apply`), new files in `untracked/`. The plugin as run is frozen in `plugin/` (`src/`, `urdf/`);
  every command imports it from there.
- Builder scripts: `scripts/` (sha256 in `versions.json`; v2 pipeline in `scripts/real2sim/v2/` of the worktree).
- Python `{PY}` (3.13, uv venv; `pip_freeze.txt`), mujoco {mujoco.__version__}, torch with MPS. Splat trainer: `{brush_v}` (Brush, Apple silicon, Metal).

{notes}

## 5. Commands
All run headless on this Mac from a fresh shell. Each line changes to `{RUN}` first and runs Python with -B, so
nothing is written into this folder (give output paths outside it too). `[sb]` = runs under the rubric's sandbox (it reads no photo:
the splat, the kit, the plugin snapshot, the SfM cameras and `HELDOUT_PHONE.json` from this folder, and `REAL2SIM_ROOT`
only for nothing else). Times on the M4 Max with nothing else running.

| Command | Line | Outputs, time |
|---|---|---|
| HELDOUT_RENDER [sb] | `{cmd("heldout_render --frames f00004,f00008 --out DIR")}` | `fNNNNN.png`, `_seg.png`, `_outline.png`, `cameras.json` (PINHOLE 1884x1059, poses unchanged from `sfm/pinhole`); ~1 s/frame |
| VIDEO [sb] | `{cmd("video --t0 T0 --t1 T1 --seed S --out FILE.mp4")}` | the phone path (registered poses, slerp between them), 1280x720, 15 fps, the demonstrator running from seed S; ~0.15 s/frame. `--sbs` adds the real video (black inside windows + guard; reads the MOV: not for the sandbox) |
| DEMO | `{cmd("demo --seeds 1,2 --pairs resistor:green,cpu:red --variant default --out OUT.json --dump DIR")}` | per episode: seed, variant (knob values), pair, placed objects, instruction text, success, failure, glitch, sim_s, physics_flags, max_penetration_m, settle_s; DIR/<seed>.npz (t, obj_pos [T, n, 3], obj_names) every control step (0.02 s). Pairs are `type:bucket` with a declared type (section 4: resistor, cpu, battery; the generic shapes are unnamed distractors, not types) and the bucket BODY (black, green, red, blue = labels 1-4); in `rand` the bucket may show another colour, the instruction names the shown one (`bucket_shown`). ~3-10 s per episode |
| ROBOTCAM [sb] | `{cmd("robotcam --seed S --variant default --out DIR")}` | `overview.mp4`, `wrist.mp4` 640x480 30 fps; `frames.json` (sim time, joints), `events.json` (grasp = phase `close`, release = phase `release`), `poses_sfm.json` (both cameras, OpenCV cam_from_world, SfM frame, `_metres_per_sfm_unit`), PNG + `_seg` + `_outline` every 10th frame and at e-1, e, e+1 of every event, `episode.json`; ~2-3 min |
| RAND [sb] | `{cmd("rand --seeds 1,2,3,4,5,6,7,8,9,10,11,12 --out DIR")}` | per seed: overview + wrist at t = 0 and at the first grasp, `<seed>_variant.json`; `knobs.json`, `rand_grid.jpg`; ~20 s per seed |
| SPEED | `{cmd("speed --frames 300 --seeds 1,2,3,4,5")}` | r15 (R-50): NEW images/s per camera (overview, wrist; 640x480) on the recording path (`--path service`, the default: the plugin as it streams to Studio, its camera thread, the kit's free-running camera processes, the planner in its own process; physics paced to real time); `--path loop` is r14's in-process loop, counting only new images (it starves on piles: one call per camera per loop iteration). Builder's run: {a.speed} |
| CAL | `{cmd("cal")}` | prints `calibration.json` |
| PLUGIN_CLI | `mkdir -p {RUN} && cd {RUN} && PYTHONDONTWRITEBYTECODE=1 PHYSICALAI_REBOT_B601_JOINT_ZERO={R}/real2sim.json PYTHONPATH={R}/plugin/src {PY} -B -m physicalai_mujoco_plugin start --profile rebot_b601 --scene bucket_sort_desk_v2 --no-gui --seed 1 --http-port 8931 --appearance {R}/real2sim.json` | then `curl -X POST localhost:8931/autopilot -H 'content-type: application/json' -d '{{"mode":"drive"}}'`, `curl localhost:8931/health` (`episode`: instruction, success), `curl -o o.jpg localhost:8931/cameras/overview/frame.jpg` (and `wrist`); stop with `curl -X POST localhost:8931/shutdown`. S2: the same compositor class (`desk_compositor.Compositor`) renders the plugin's `overview`/`wrist` streams (`--appearance`) and ROBOTCAM |
| GRAB (S2) | `{cmd("grab --seed 3 --port 8942 --after 6 --out GRABDIR")}` then `{cmd("robotcam --state GRABDIR --out REPRODIR")}` | GRAB starts the PLUGIN_CLI line as a subprocess with `R2S_GRAB_DIR`, drives the autopilot for 6 s, then saves the next frame of both streams exactly as streamed (`overview.png`, `wrist.png`, lossless) with the exact state each was rendered from (`<camera>_state.npz`: qpos, the synced model arrays, sim time), and always shuts the plugin down. ROBOTCAM `--state` renders those states offline with the same compositor class: `<camera>_robotcam.png`, `s2_psnr.json` (builder: overview 103 dB, wrist identical). ~1 min |
| STUDIO_RECORD | see section 6, "Studio recording": a separate Studio worktree + venv with this plugin installed, then `{PY} -I -B {R}/scripts/studio_record_v2.py --episodes 3 --project bucket_v2_review --dataset rNN_check` (its IP cameras capture at 60 fps by default, `--camera-fps`) | a Studio dataset (LeRobot v3) of 3 episodes recorded by Studio's own pipeline (teleop from the generic `MuJoCo_Virtual_Leader`, the autopilot in leader mode); its folder is printed; ~1 min |
| STABILITY | `{cmd("stability --episodes 10 --port 8941")}` | starts the plugin CLI as above as a subprocess, autopilot `drive`, waits for 10 episodes; prints time, RSS of the plugin and its compositor processes before/after, and how many transient HTTP errors were retried (a snapshot may answer 503 while the cameras start or restart at a reset; each request retries for up to 30 s); always shuts the plugin down, also on failure; ~3 min |
| ROBOTCAM_REPLAY [sb] | `{cmd("replay --states STATES_E.parquet --times E:t,E:t,... --out DIR")}` | Both cameras at the recorded joint states and times (RUBRIC 6.3): `--states` is a parquet with `episode_index`, `timestamp` (episode-relative s) and `observation.state` (the driver's public units, the real dataset's order) - the reviewer's `states_E.parquet` copy; `--times` takes `E:t` (episode, seconds) or plain `t` for a one-episode file. Per time: `<E>_<t>_overview.png` and `<E>_<t>_gripper.png` (the sim `wrist` camera is the real `gripper`), each with `_seg.png` and `_outline.png`, and `replay.json` (the public state used per camera). The state of t is linearly interpolated between the rows around t, at t minus the camera latency (kit `camera_latency_s`, 0.067 s, measured on allowed episodes); joints through the profile's channel conversion; the buckets of episode E per bucket (r12, the user's U-4: "red moved on ep2 onwards and again on ep7", counting from 1; each bucket's segment the median (x, y, yaw) of the allowed episodes' unchanged fits in it: red {0} from 0, {1-5} from 1, 2, 3, {6-10} from 6, 7, 9, 10; green and black {0-3} from 0-3, {4-10} from 6, 7, 9, 10; blue one pool over all allowed episodes; kit `bucket_layouts_episodes.bucket_segments`; `replay.json` records per bucket the segment and the episodes pooled in `bucket_layout_from_episode`; `--layout matched` gives r09's phone-matched buckets; `--layout FILE` (r12, U-5) takes the poses of a FIT_LAYOUT output for every time, and `replay.json` records which layout was used (`layout`, and per time the file and its sha256)); the fingers drawn where the recorded reading puts them (r13: the r12 replay-only hold rule is removed: the replay draws what a recording would show), the real look, every task object parked (the sim does not know the real objects); sensor noise on, drawn per t. Reads only `--states` and this delivery. ~3 s per time |
| FIT_LAYOUT [sb] | `{cmd("fit_layout --video EP_overview.mp4 --t0 S --t1 S --states STATES_E.parquet [--episode E] [--start S] --out DIR/layout.json")}` | r12 (user decision U-5): one real episode's four bucket poses from its overview video, by the builder's own per-episode fitter (`scripts/layout_fitter.py`: the method of `fit_bucket_layouts.py`, which made the kit's per-episode fits, unchanged). Frames, a fixed rule: the video's decoded frames k (file time k/30 s, the video's own clock from 0) with t0 <= k/30 <= t1, every 5th from the first of them, of which 25 spread evenly by index (np.linspace); their per-pixel temporal median (the moving arm drops out). No frame is skipped; the arm is excluded instead as a region: the sim arm's silhouette at the median joint angles of the 25 frames, dilated 15 px, removed from both masks. Those joints come from `--states` (as ROBOTCAM_REPLAY's: the episode's rows; `--episode` picks one from a multi-episode file; row of frame k = the timestamp nearest `--start` + k/30, within half a frame; `--start` = the episode time of the video's first frame, default 0), so the sandbox must allow reading the video and the states file. Fit: per bucket, the real mask by colour near the bucket's phone-matched place, the pose (x, y, yaw) maximising the silhouette IoU with the sim bucket in the kit's overview camera (grid +-30 mm / +-3 deg, then +-6 mm / +-1 deg; one bucket at a time, two rounds). Writes `layout.json` (`poses` per bucket, the input for `replay --layout FILE`; `iou_fitted`, `iou_phone_layout`, `moved_mm_deg` from the phone layout, `frame_list`, the inputs' sha256) and `layout_overlay.jpg` (the median frame | fitted sim outlines per bucket colour, real colour masks magenta, excluded arm hatched grey). Check: on allowed episode 7 (`--t0 0 --start 0`) it reproduces the stored fit exactly (`tests/test_bucketsim_fit_layout.py`); on episodes 6 and 9 exactly with `--t0 0 --start 0.5`, the others with `--t0 0 --start 0` (r16: the stored fits ARE this command's output, at the 120 mm bucket height, for all eight allowed episodes; `--t0` sets which decoded frames are taken, `--start` only maps video frames to the states' episode time; r17 correction: r16's table still gave r13's `--t0 0.0333` for episode 6, which gives black y +3 mm). The builder never runs it on held-out material. ~15 s |
"""
(out / "MANIFEST.md").write_text(man); print("MANIFEST", out / "MANIFEST.md")

# 5. R4: the composite video over the whole phone path, rendered here in the foreground by this delivery's own VIDEO
#    line (from /tmp/r2s_v2_run, this folder's scripts/ and plugin/), then the per-file hashes: one command freezes it all
if not a.no_video:
    vid = out / "outputs" / "video_phone_path_seed2.mp4"; vid.parent.mkdir(exist_ok=True)
    tmp = Path(RUN) / f"{a.name}_video.mp4"; Path(RUN).mkdir(parents=True, exist_ok=True)
    line = cmd(f"video --t0 0 --t1 157 --seed 2 --out {tmp}")
    print("rendering the phone-path video (~6 min) ...", flush=True)
    subprocess.run(line, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.move(str(tmp), vid)
    with open(out / "MANIFEST.md", "a") as f:
        f.write(f"\n## 7. Delivered outputs (R4)\n- `outputs/video_phone_path_seed2.mp4` (sha256 {sha(vid)}): this delivery's VIDEO line, "
                f"`video --t0 0 --t1 157 --seed 2`, run by deliver_v2.py right after the delivery was built (from `{RUN}`, this folder's "
                "`scripts/` and `plugin/`); 1280x720, 15 fps, sim only (no photo pixels).\n")
sums = [f"{sha(p)}  ./{p.relative_to(out)}" for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS_DELIVERY.txt"]
(out / "SHA256SUMS_DELIVERY.txt").write_text("\n".join(sums) + "\n")
print("frozen:", out, "| MANIFEST sha256", sha(out / "MANIFEST.md"))
