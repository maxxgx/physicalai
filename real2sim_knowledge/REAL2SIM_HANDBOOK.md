# Real2sim handbook: the reBot bucket-sort twin, Oct-Nov 2026

This handbook collects what we learned building a simulated copy of a real robot cell, training a policy mostly in that
simulation, and running it on the real arm at a trade fair. It is written for an engineer who joins later and wants to
repeat or extend the work. Every number has its source in brackets. "Inferred" marks a statement no source states
directly.

Path shorthands used in the citations:

| Short | Full path |
|---|---|
| BV2 | `~/real2sim-data/bucket-v2` |
| V1 | `~/real2sim-data/bucket` |
| MEM | `~/.claude/projects/-Users-xiangxi2-physicalai/memory` |
| REP | `~/.claude/reports/bucket-v2` |
| SKILL | `~/.claude/skills/build-real2sim-scene` |
| WT1 | `~/physicalai-wt-bucket/packages/physicalai-mujoco-plugin` (v1 code, uncommitted) |
| WT2 | `~/physicalai-wt-bucket-v2/packages/physicalai-mujoco-plugin` (v2 code, uncommitted) |
| NUM | `BV2/retro_1123/timeline/numbers.json` |
| WW / MS | `BV2/retro_1123/timeline/what_worked.md` / `mistakes.md` |

Terms used throughout:

- Real2sim: building a simulation that matches a real cell closely enough that a policy trained in it works on the real robot (sim2real).
- Digital twin: the simulated cell (robot model, objects, cameras, look) that stands in for the real one.
- pi05: Physical Intelligence's vision-language-action model. It takes two camera images, the joint state and a text prompt, and returns a chunk of 50 future joint actions.
- Episode: one recorded task attempt (images, joint states, actions, prompt).
- Splat (Gaussian splat): a 3D scene made of many small coloured blobs, trained from photos, that renders photoreal images from new viewpoints.
- SfM (structure from motion): recovering camera poses and sparse 3D points from overlapping photos (COLMAP).
- FK (forward kinematics): computing where the arm's links are from its joint angles.
- DAgger: run the model, let an expert take over when it goes wrong, record the recovery, and add it to the training data.
- RTC (real-time chunking): an execution mode that keeps the arm moving while the next chunk is computed and guides the new chunk to continue the actions already being sent.
- kbox: the Kontron edge PC (Intel Core Ultra X7 358H, Panther Lake, integrated GPU) that ran the booth demo (NUM).

---

## 1. What we built and the result

We built a MuJoCo twin of one real reBot B601 arm on a desk with four coloured boxes, an overview camera and a wrist
camera. The task is "put the <battery | CPU | bracket> in the <red | blue | green | black> box", given as text. A
scripted planner in the sim generated the demonstrations. pi05 was trained on them, improved with DAgger, exported to
OpenVINO and run with RTC on the kbox iGPU.

Timeline (NUM): brainstorm 8 Oct; "script it in sim first" set 20 Oct; reBot desk enters 30 Oct; v2 build starts 6 Nov;
first training on reBot sim 9 Nov; first real-robot test 10 Nov; final model s26c frozen 20 Nov; SPS booth in Nuremberg
23 Nov. That is 46 days from brainstorm to booth and 13 days from the first real test.

The final model, s26c:

- Lineage: pi05_base, then nine stages of about 92k steps: sim_r21 12k, r21c 12k, S21-D1 8k, S22-D1 8k, S23 6k, S23-D1 6k, S24 8k, S25 16k, s26c 16k (NUM; `BV2/retro_1123/review/rounds/r09_review.md`). Every stage before s26c used sim data only.
- Last-stage mix: 3,228 unique sim episodes (the S25 mix) plus 50 real episodes copied 16 times, 4,028 rows and 1.63M frames. By unique episodes that is 98.5% sim. By frames it is 75.6% sim and 24.4% real, because the real copies count 16 times (NUM; `BV2/retro_1123/review/rounds/r08_review.md`).
- The 50 real episodes are 20 pile demos (16,320 frames) and 30 "hold still under shadows / return from box" episodes (8,467 frames) (NUM).
- Real A/B on the kbox, 20 Nov: s26c 7/8 (piles 5/6, hold still 2/2) against its parent S25 4/8 (`BV2/HANDOVER.md`, entry 2026-11-20).

The booth, 23 Nov (`BV2/retro_1123/booth_analysis/analysis.md`):

- 86 runs on booth day plus 2 lab runs on 22 Nov (88 in total), 57.4 min under model control, all s26c RTC.
- Strict success 74 of 86 (86%). Counting only the 79 runs where the policy had a fair chance (5 paused within seconds, 2 with no target object on the table removed), 74 of 79 (94%).
- No object went into a wrong box. About 111 objects were placed. Median time to the first object in the box was 9.7 s (n = 71).
- By object (fair-chance runs): battery 41/44, CPU 22/24, bracket 11/11. It stayed 19/21 through two box-layout swaps, so it followed colour, not position.
- The recorder logged no success field. Every verdict is a visual judgement from both robot cameras; 70 of the 74 successes are high confidence.
- The losses were operational: a USB/CAN outage, Studio restarts, iGPU sharing with Chrome, one 446 s inference stall. The model drove the robot for only 10% of the 9.5 h between first and last run.

Cost, where tracked (NUM): v1 desk about $150; v2 builder, coordinator and reviewers $926.73 up to 9 Nov; earlier booth and
dataset-room experiments about $530. Cost after 9 Nov was not tracked (MS section H).

---

## 2. The pipeline, end to end

There were two builds. v1 (30-31 Oct) used one 75 s phone video and plateaued at 3.0/5 with its reviewer (`V1/handoff/LEARNINGS.md`).
v2 (from 6 Nov) used a new video, a LiDAR scan and real robot-camera episodes, and then grew through sim versions r17 to
r25 as the real-robot results came in (`BV2/BRIEF.md`; `BV2/sim/r25/README_r2*.md`). The method skill `SKILL/SKILL.md`
describes the v1 workflow step by step; v2 extended it.

### 2.1 Capture

- What: walk around the cell with an iPhone (v1: 75 s; v2: 157 s, 4K, 30 fps), both in HDR HLG (`V1/handoff/LEARNINGS.md`; `BV2/notes/JOURNAL.md` log). v2 also had an iPad LiDAR scan of the office (3 Nov) and about 11 real episodes from the robot's own cameras (`BV2/BRIEF.md`).
- Tool: `avconvert -s in.MOV -p Preset3840x2160 -o video/sdr_4k.mov`, then `extract_frames.py` (`WT1/scripts/real2sim/`).
- Check: `ffprobe` shows `color_transfer=bt709` after conversion. HLG left as is makes every colour wrong (`V1/handoff/LEARNINGS.md` section 3).
- Held-out split first: choose the held-out phone windows from the video duration alone, before looking at any frame. v1 used five 2 s windows (about 15% of the video) with 0.5 s guards in `review/HELDOUT_PHONE.json` (`V1/handoff/LEARNINGS.md` section 2). For the real episodes v2 kept 0, 1, 2, 3, 6, 7, 9, 10 and held out 4, 5, 8 (`BV2/notes/JOURNAL.md`, r08).
- LiDAR: the ICP registration of the scan to the splat was weak (28% inliers, scale 3.79 units/m and drifting), so it was reported but not used; the desk came from the splat (`BV2/notes/MANIFEST_NOTES_V2.md`; `BV2/notes/JOURNAL.md` 01:00). The office scan lessons (RoomPlan invents walls behind glass, its boxes are bounds and not objects) are in `~/real2sim-data/office-threejs/handoff/LEARNINGS.md` section 3.

### 2.2 SfM and camera calibration

- What: COLMAP SfM of the phone frames, scale and placement from known CAD, then fitting the robot cameras to real robot-camera frames.
- Tools: `sfm_phone.py` (keep torch out of this process), `fit_buckets3d.py`, `fit_robot.py`, `make_calibration.py`, `calibration_extras.py` (`WT1/scripts/real2sim/`); v2 adds `calib_robotcams.py`, `fit_robotcam.py`, `fit_arm_both_views.py` and many `fit_*` scripts (`BV2/sim/r25/scripts/`).
- Typical numbers: v1 registered all 376 frames (`V1/handoff/LEARNINGS.md` section 3). v2 registered 821 of 937 (`BV2/notes/JOURNAL.md` 00:40).
- Scale from two sources: v1 buckets gave 3.255 units/m and the robot CAD 3.30, 1.4% apart; v2 gave 3.558 and 3.546, 0.34% apart (`V1/handoff/LEARNINGS.md`; `BV2/notes/JOURNAL.md` 01:00). Declare the disagreement.
- Robot cameras (r08): overview and wrist pose and field of view fitted on allowed real episodes to RMS 1.09 and 1.44 px, plus a tone curve and sensor noise per camera (`BV2/notes/JOURNAL.md` r08).
- Check: CAD outlines sit on real edges in allowed frames. Fit cameras from SfM, FK and PnP anchors, never from arm silhouettes (section 3.1).
- The overview camera physically moves between days (+3/+1/-7/-2 px), so r23 added per-episode camera jitter (MS section B). After the jig, the overview refit improved from about 7 to about 3 px (NUM calibration).

### 2.3 Gaussian splat

- What: train a splat of the room and desk on the Mac, mask the real robot out of it, clean it, and cut the arm's own Gaussians out of an unmasked splat so they can ride on the sim links.
- Tools: `make_train_set.py` (drops held-out frames), Brush 0.3, `robot_masks.py`, `clean_splat.py`, `robot_gaussians.py`, `topdown.py`, `dev_eval.py` (`WT1/scripts/real2sim/`). Brush command: `brush . --total-steps 30000 --sh-degree 3 --growth-stop-iter 20000 --max-resolution 1920` (`V1/handoff/LEARNINGS.md` section 4).
- Typical numbers: about 25 min for 30k steps at 1880x1057 on the M4 Max (`V1/handoff/LEARNINGS.md`); about 5 min in the earlier dataset-room run (`MEM/dsroom-calibration-lessons.md`).
- Check: a top-down render shows table, objects and robot sharply. Brush needs a clean `sparse/0` with only the three `.txt` files and has no pose optimisation, so pose noise blurs (`SKILL/references/lessons.md`).

### 2.4 Robot and CAD calibration

The arm in the sim must stand where the real one stands and read the same angles. This took most of the calendar.

- Per-unit joint zeros: the plugin applies zeros for this one arm when `PHYSICALAI_REBOT_B601_JOINT_ZERO` names a kit file; unset, the stock profile is unchanged. Recordings store driver readings, as on the real arm (`BV2/notes/DISCLOSURE.md`, r13).
- The phantom plate: r17 to r21 placed the arm on an extra 9.8 mm plate, but the CAD `base_link` already includes its 10 mm plate. The sim shoulder-lift axis sat at 154.8 mm against 145.0 mm real. The elbow and wrist zeros silently compensated (MS section B; `BV2/calib_jig/JIG.md`). r22 removed the plate (`BV2/sim/r25/README_r22.md`).
- The calibration jig (`BV2/calib_jig/`, design in `JIG.md`, field notes in `AUTO_NOTES.md`):
  - A printed piece placed against the base plate, with 7 cone sockets. A cap on the fingers holds an 8 mm steel ball.
  - Guide the ball into each socket by hand (float mode in `capture_readings.py`) or let the auto mode visit them.
  - `fit_geometry.py` returns `corrections.json`: base height, joint zeros of pan, lift, elbow, wrist_flex and wrist_yaw, and pan against the base plate. `fit_cameras.py` gives camera poses from a printed marker sheet.
  - It does not settle the wrist_roll zero, the gripper opening or link lengths (taken from CAD).
- Jig numbers: the printed towers F and G sat about 230 mm apart instead of the designed 220 mm. The first fit trusted the design and was biased (2.08 mm rms, arm 10 mm low). With tower positions fitted, the residual dropped to about 1.2 mm rms and the pan zero came out at -3.03 to -3.12 deg depending on the fit, against -1.388 in the scene (`BV2/calib_jig/AUTO_NOTES.md`). r23 adopted pan -3.011 and wrist_yaw 0.234 (`BV2/sim/r25/README_r23.md`). NUM gives the A-E fit as 1.19 mm rms.
- Use the ruler: fit with `--base-z 0` from a ruler reading; a free base height is only known to 0.65-1.3 mm (`AUTO_NOTES.md`, A-E section).
- Objects and gripper at real size: battery x1.42 and CPU x1.28 in r21, then measured dimensions in r23 (`BV2/sim/r25/README_r21.md`, `README_r23.md`). The jaw opening went from 70.1 mm (measured map) to about 85 mm (Max's own measurement, adopted in r16) (`BV2/notes/JOURNAL.md` r16). Finger flex while holding became opening-dependent in r24: 24 deg at the 38 mm CPU, 10 deg at the 25 mm battery, 4.5 deg on the bracket (`README_r24.md`).
- Dynamics: r20 put the real arm's control law (low-gain PD with gravity feedforward) into the sim; command against reached position fell from 5.2 mm to 0.65 mm median (WW section 2; `BV2/sim2real/WEEKEND_LOG.md`). In the gap viewer (real arm, sim ghost, leader ghost; `BV2/sim2real/gapview/`) the fingertip gap went from 5.3/19.7 mm to 3.0/9.0 mm median/p95 between r20 and r22 (NUM). MuJoCo's `noslip_iterations = 3` stopped held objects creeping out: centred carries 8/13 to 15/15 (`README_r21.md`).

### 2.5 Compositor for the robot cameras

- What: render the splat as a 2D background image from each camera and draw the sim (arm, objects, boxes) over it by depth. "Composite, don't mix": splat surfaces are fuzzy layers a few cm thick and swallow small objects when mixed into the 3D scene (`SKILL/references/lessons.md`).
- Tools: `real2sim/desk_compositor.py`, `metal_splat.py`, `_metal_splat_kernels.py`, `viewer_background.py` in the plugin; the kit file `real2sim.json` names splat, calibration, looks, arm Gaussians and cameras. Run with `physicalai-mujoco start --profile rebot_b601 --scene <scene> --appearance <kit>`; Studio then records the composited `overview` and `wrist` streams (`SKILL/SKILL.md`).
- Typical numbers: the Metal splat kernel gives 30-40 fps per robot camera on the Mac; a CUDA port on mclx-25 took a chunk from 100 s to 2.5 s (WW section 2). Two staggered render processes per camera reach about 39-40 new images/s (`BV2/notes/R15_RESULTS.md`).
- Check: sim boxes, not splat boxes, in the wrist view (the splat smears at close range); shadows only on desk pixels; frustum-cull Gaussians outside the view, or a downward-looking camera darkens; re-apply geom visibility on every render (`V1/handoff/LEARNINGS.md` sections 5-6).

### 2.6 The scene in the MuJoCo plugin

- What: scene XML from the calibration, registered in `scene_registry.py`; camera mounts in `profiles/rebot_b601.py`; the task in `bucket_sort.py`, `bucket_sort_arms.py`, `bucket_sort_objects.py`.
- Tools: `build_desk_scene.py` (v1), `build_scene_v2.py`, `collision_decomp.py`, `audit_visual_collision.py`, `bucket_regions.py` (v2) (`MEM/drawn-equals-collision.md`; `BV2/sim/r25/scripts/`).
- Check: drawn equals collision within 2 mm for every body that touches objects (section 3.2).
- Each sim version is a frozen copy with its own plugin, kit and README. Select one with `source BV2/sim/rNN/env.sh`. The lineage (`BV2/sim/r25/README_r21.md` to `README.md`; `MEM/bucket-v2-bracket-tray-r19.md`):

| Version | What it added |
|---|---|
| r17 | Frozen reviewed delivery; the first 300 training episodes |
| r19 | Metal L bracket and tray, piles |
| r20 | Real control law, gripper twin with pads, per-episode zero and table jitter |
| r21 | Gap-viewer dynamics, noslip grip, real-size battery and CPU, open-table layouts |
| r22 | Arm at its measured height (phantom plate removed), mechanical zeros |
| r23 | Jig pan and wrist-yaw zeros, measured object sizes, finger flex, camera jitter |
| r24 | Spawns over the whole reachable table, tray anywhere, more piles, arc waypoints round the base |
| r25 | Lighting and shadow randomisation |

### 2.7 Scripted demonstrator

- What: an IK planner (`bucket_sort_demo.py`, class `Planner`) that picks the named objects and drops them in the named box. Save only clean successes.
- Typical numbers: the stock v1 demonstrator missed about 1 grasp in 10, all with the object 26-40 deg off the forward axis at 0.30-0.36 m (`V1/handoff/LEARNINGS.md` section 1). In v2, planning proxies cut clearance planning from 30 s to 2.2 s per episode and lifted default success to 96/100 (`BV2/notes/JOURNAL.md` r14). r24's arc waypoints round the pan axis raised near-base demos from 10/18 to 15/16 (`BV2/notes/r24/STATUS.md`).
- Check: measure success over at least 50 seeds before recording (`BV2/BRIEF.md`). Reseed the demonstrator from the episode seed at every reset, or the same seed gives different motion alone and in a batch (`BV2/notes/R6_LOCKSTEP_DESIGN.md` section 1).
- Known gap: the sim is stiller than a human operator, 18.77% near-still frames against 6.36-8.14% real, from slow min-jerk move ends and planning stalls (`BV2/notes/R17_STILL_ANALYSIS.md`).

### 2.8 Recording through Studio

- What: record sim episodes with Studio's normal teleop path, using the generic `MuJoCo_Virtual_Leader` with the plugin's autopilot in leader mode (physicalai #404). The dataset's actions are the leader's commands, which squeeze past the object as a human operator does (`MEM/sim-recording-via-virtual-leader.md`).
- Tools: `studio_record_v2.py` / `record_300.sh`, `first_frame_check.py`, `dup_per_episode.py`, `spotcheck_v2.py` (`BV2/notes/RECORDING_PLAN.md`). Run Studio isolated (own `SETTINGS_FILE`, port 7871), AV1 (`cpu-av1`), 640x480 at 30 fps, `--keep perfect`, `--start-settle 0.5`.
- Typical numbers: the first set from r17 is 300 episodes, 156,443 frames (NUM). Pilot 2 saved 50 episodes in 59 tries (`BV2/notes/JOURNAL.md`).
- Later rounds used an offline, batched recorder with fresh worker pools, which contained a renderer memory leak and produced about 600 episodes a night: r20 663, r21 632, r22 606, r23 531, r24 766, r25 582, plus about 50 absent-object episodes each (WW section 2; NUM). r25 ran on mclx-25 with GPU render lanes and memory guards (`BV2/sim2real/r25_tools/`; `BV2/notes/r25/STATUS.md`). Check every set with `verify_ds.py`.
- In total about 6,450 sim episodes were recorded (NUM).

### 2.9 Domain randomisation

- Per-episode look (from r08): real look with p 0.3, otherwise randomised table colour and pattern, box colours and places, light, colour temperature, exposure, camera jitter, image noise and blur (`BV2/notes/RECORDING_PLAN.md`). That plan predates the scope cut to battery, CPU and bracket; its object mix is out of date.
- Geometry jitter: per-episode joint-zero and table jitter from r20, overview camera jitter from r23 (`MEM/bucket-v2-bracket-tray-r19.md`; MS section B).
- Lights (r25): 85% of episodes get 1-3 shadow lights at 0-60 deg tilt and 0.8-3.0 m, shadow strength 0.2-0.9, a wrist-bracket shadow in 40%, brightness x0.68-1.4, colour temperature +-15%, drifting distractor shadows in 30% (`BV2/sim/r25/README.md`).
- Check: hold out one light sector for evaluation. A hard back-left light cut pooled success of four older models from 76 to 63 of 120 (p 0.015) (`BV2/notes/r25/STATUS.md`).

### 2.10 Shadow and pile data from the real robot

- What: Max's phrase "shadow augmentations" means 30 real episodes recorded on purpose, not image augmentation (`BV2/retro_1123/review/TASTE.md`; `BV2/retro_1123/timeline/timeline.md`). Five scenes times six prompts: empty table, empty table with moving shadows, task done with shadows, only the other object type with shadows, return from the box.
- Why: S23-D1 and S24 "chased their own shadow" on the real arm. The sim had one light straight down and no wrist shadow (`BV2/retro_1123/timeline/timeline.md`, 18 Nov).
- Tool: a shadow runner page, `BV2/trials/shadow_1119/shadow_runner.py`. The 30 episodes took about 10 min to record (WW section 4).
- Together with 20 pile demos they are the only real data in s26c (section 1).

### 2.11 Training stages and mixes

- Settings, never changed (`BV2/notes/pi05-training-recipe.v1.html`): pi05 from pi05_base, full fine-tune, bf16 with Kahan-compensated AdamW (Studio PR #1355), lr 2.5e-5, 1,000 warmup steps, cosine to 2.5e-6, batch 16 on one H100, chunk 50, 10 denoising steps, quantile normalisation. 0.55-0.75 steps/s, so 12k steps is about 5 h.
- Stages: the first model 12k steps from pi05_base, then 4k-8k step fine-tunes from the last checkpoint (12k-16k when a large new recording comes in), warmup and cosine restarted each stage (same source).
- Mixes are built by copying episodes: real demos x5 to x16, DAgger corrections x3, absent-object episodes x3. Sim hold frames get the operator's gripper squeeze (action -5) and pauses over 0.5 s are trimmed (`BV2/notes/DISCLOSURE.md`; WW section 4).
- Two model lines: a co-trained line with about a third to a half real frames (F2 at 53% real 14/15; r12, 76k steps in 9 stages, 14/15), and a sim-only line (S23-D1 13/15, 52k steps) that led to S25 and s26c (recipe; NUM).
- Typical numbers: checkpoint 9.4 GB, OpenVINO export 6.3 GB (NUM). 16k beat 8k on the r25 sim (r15 32 to 47 of 60, p 0.001; S25 44 to 51, p 0.04) (`REP/sim_eval_r25_8k_vs_16k.html`).
- Tools: our trainer on h100s in `~/xiangxi2_vla` with `launch_1gpu.sh` (local copy in `BV2/train_night/n1117/`); pass `--max-epochs 1000` (`BV2/HANDOVER.md`). Hugging Face was the transfer bus: model repo `iMaxxgx/pi05-bucket-sort-runs` (public), datasets private.

### 2.12 DAgger

- Sim DAgger with the scripted planner as expert: the policy drives, the harness takes over on failure triggers (random cut, missed lift, budget, hover, drop) and records the recovery.
- Tools: `BV2/sim/dagger/` (round 1), `BV2/sim2real/r21_dagger/`, `BV2/sim2real/r23_dagger/` (`go.sh`, `triggers.py`, `takeover.py`, README).
- Typical numbers: round 1 (9-10 Nov) kept 1,125 and used 965 corrections and took the 12k model from 14/30 to 30/30 in sim in three rounds (`MEM/bucket-v2-training-night.md`; NUM). On r21, 425 corrections took S21 from 22 to 33 of 45 on the real-trial replay, while extra steps alone gave 25 (WW section 4). The r23 round kept 402 of 1,334 tries; the policy alone succeeded 74.4% (`MEM/bucket-v2-coordination-1112.md`, 08:50).
- Real DAgger: `kontron_dagger_1112`, 41 corrections, with a local Studio patch that glides teleop takeovers over 10 s (`BV2/HANDOVER.md`).

### 2.13 Export to OpenVINO

- Chain per run, unattended on h100s (`after_1gpu.sh`): keep the checkpoint, upload it, export OpenVINO, apply the noise fix, run NOISECHECK, build the RTC export, run RTCCHECK and PARITY, upload, mark the run "candidate" in the model hub (WW section 4).
- Noise fix: pi05's start noise must be a model input that the runtime draws fresh each call (physicalai #442 and #455; Studio #1414 and the converter #1393). Before the fix the iGPU scored 7/20 against 18/20 for torch; after it 16/20 (`BV2/retro_1123/code_delta/inference_bugs.md` item 9; NUM). A noisefix export needs physicalai 0.4.0 or newer.
- RTC export: `BV2/sim2real/rtc_real/export/export_ov_rtc.py` sets `policy.rtc_enabled = True` before export and checks the four RTC graph inputs; then the noise fix (inference_bugs.md item 8).
- Check: "the same input twice must give different actions" on the target device. Set the OpenVINO device to GPU explicitly; "auto" picked the CPU at 8.8 s per chunk against about 250 ms on the iGPU (`BV2/HANDOVER.md`).

### 2.14 RTC execution on the edge box

- Modes: Sync stops and waits for each chunk; Async keeps playing the old chunk and crossfades; RTC keeps playing and guides the new chunk to continue the old one (`REP/inference-modes-sim-ab.html`).
- Sim A/B with the runtime bugs fixed, 290 ms latency, 60 paired seeds and two models: Sync 101, Async 0.5 95, RTC 98 successes of 120. RTC joined chunks within 1.1 deg against 4.0 deg for Sync and finished about 1 s sooner (same source).
- First real RTC trial (17 Nov): 0% time waiting, 0.97 deg seam, 29.8 Hz (NUM). Max: "MUCH smoother" (WW section 6).
- Studio had no RTC mode; we added it on the local branch `max/studio-rtc` (also in `max/demo-conference`). kbox settings: horizon 25, guidance weight 5, initial latency 0.307 s (WW section 6).
- Our four RTC timing fixes cut chunk-switch jumps from 30.4 deg to 6.5 deg maximum (median 3.0 to 1.1). They ran at the booth from a local overlay and are still draft PR #482 (inference_bugs.md items 3-6).
- Typical numbers on the kbox: s26c RTC p50 310 ms, p95 316 ms; booth morning 313 ms; about 280 ms with Studio alone; a live Rerun viewer added 40-50 ms (NUM; `BV2/sps2026_booth/README.md`).

### 2.15 Evaluation

- Real protocol: 3 objects x 5 taped spots (S1-S5) = 15 trials per model, 60 s limit, regrasps allowed, rotated model order, an eval-runner page that drives Studio, a Stream Deck for outcomes (WW section 5).
- Always-on trial recorder on the robot PC (`BV2/trials/trial_recorder.py`): joints, commands, both cameras, Studio events.
- Sim evals on mclx-25 with paired seeds and sign or McNemar tests; the real-trial replay rebuilds each real trial in sim three times (WW section 5; `REP/sim_evals_1117.html`). One rerun of the same config flipped 7 of 30 seeds, so only multi-seed gaps mean anything (`REP/sim_eval_r25_8k_vs_16k.html`).
- Model hub: `BV2/model_hub/` (registry, lineage, sim and real results, rollouts); 42 models registered (NUM).
- Real results over time (NUM): 13 Nov F2 14/15; 16 Nov S21-D1, the first sim-only model, 10/15; 17 Nov r12 14/15 and sim-only S23-D1 13/15; piles r14 5/9, r15 3/9; 20 Nov s26c 7/8.

---

## 3. Lessons

### 3.1 Calibration

- Trust SfM, FK and PnP over silhouettes. Arm silhouettes are fattened by cables and dark parts trap optimisers; one silhouette fit put a camera at 0.82 m where PnP gave 0.58 m (`MEM/dsroom-calibration-lessons.md`).
- Camera datasheets are not measurements. The D405's vertical field of view was about 62 deg, not 58; the v1 wrist lens sat 21 mm from its CAD position; the Orbbec wrist mount was off by 2 mm and 3 deg (MS section B).
- Check one absolute dimension with a ruler before fitting zeros. The 145 mm shoulder-axis reading exposed the phantom 9.8 mm plate, which had cost about a week of datasets (r17-r21) with about 1 cm height error (MS section B; WW section 3).
- Use hard stops, not pressure-dependent touches. The paper-sheet calibration of 13 Nov moved heights by about 2 mm with touch pressure and disagreed with a hand mark on pan by 5.5 deg (MS section B; `MEM/real2sim-calibration-jig-todo.md`).
- Do not trust the printed part's design dimensions. The jig towers flexed and sat off-design, which biased the first fit and cost about 3 h at the robot; fit the tower positions or use only the stiff low sockets (`BV2/calib_jig/AUTO_NOTES.md`; MS section B).
- Test auto procedures with gear play and print tolerance in the sim. The auto mode's first real run rejected all 6 visits because its seat test was tuned without backlash (MS section B).
- An overview camera pitch that the capture never looks along cannot be recovered; ask for it at intake. It capped v1's overview grade at 1 for all 10 rounds (`V1/handoff/LEARNINGS.md` section 1).

### 3.2 Sim fidelity

- Drawn equals collision, within 2 mm. In r12 the drawn fingers were shortened to match camera edges, held objects floated 8.8-10.6 mm from the fingers, and the 50-episode pilot was useless. The real cause was a wrist camera mount error. When edges disagree, refit the camera or joint calibration with the CAD unchanged (`MEM/drawn-equals-collision.md`; MS section B).
- Get object sizes and the gripper from measurements. The battery was 1.42x too small for weeks; finger flex was fitted at 24 deg against about 9 deg real, so r23 closed about 13 deg too far on the battery (MS section B).
- One light straight down taught the policy to chase shadows. Randomised lights (r25) helped in sim, but the fix that worked on the robot was 30 real hold-still episodes (MS section C). Inferred: r25's lighting alone was never isolated on the real arm.
- Coverage gaps show up as real failures: r23 objects sat only 23-39 cm from the base, and S24 reached past the desk edge on an empty table, a safety issue (MS section C).
- Do not mix data from an old calibration into a new mix. The first sim_r23 mix was half r21 corrections with the old geometry and had to be restarted (MS section C).

### 3.3 Held-out frame discipline

- Filter held-out frames in every preview, contact sheet and debug image, not only in the training set. A mask-check sheet once showed two held-out thumbnails and had to be declared (`MEM/heldout-filter-every-preview.md`).
- Declare every exposure, even metadata, in `notes/EXPOSURES.md` and the delivery MANIFEST (`BV2/notes/EXPOSURES.md`).
- One breach costs a round: on 7 Nov the coordinator told the builder to read original episodes; r08 scored 0 and needed about 2 h of re-derivation (MS section F).
- The independent reviewer with a frozen rubric caught real bugs (contents drawn through walls, empty wrist boxes, overclaims) at about $15 per round (`SKILL/references/lessons.md`).

### 3.4 Gripper control and recording

- Record sim demonstrations through a virtual leader, not by observing the follower. Observe mode stores the object width as the gripper action, so a position-controlled gripper never squeezes (`MEM/sim-recording-via-virtual-leader.md`).
- Relabel sim hold frames with the operator's squeeze convention (gripper action -5) when mixing with real data; it is a declared deviation (`BV2/notes/DISCLOSURE.md`, 2026-11-09).
- The reBot gripper needs stall logic: at its open stop it pushes about 0.5 s and goes limp; on an object it holds at 0.5 N m (#454) (`BV2/notes/rebot_gripper_setup.md`).
- A killed park loses the gripper zero. Studio's 10 s worker timeout cut a 30 deg/s park, the zero shifted about 285 deg, and #1412 raised the timeout to 40 s. Trace command against measured before patching drivers (MS section D). Since plugin 0.9 a Star Arm leader must be set to follower type b601 (`BV2/notes/rebot_gripper_setup.md`).

### 3.5 Sim and real disagree on rankings

- Sim scores never predicted the real robot. ft_r3 scored 30/30 in sim and 1/3 real; on the 17 Nov models the sim-real Spearman correlation was -0.92; s26c scored 48 in sim against S25's 51 and then won the real A/B 7/8 to 4/8 (MS section C; NUM).
- On every one of the 13 rows where the robot failed on 17 Nov, the sim succeeded on at least 2 of 3 replays; 9 were real timeouts after repeated regrasps, a failure the sim does not produce (`REP/sim_evals_1117.html`).
- Validation loss also misranks: real copies leak into validation, and the best-validation model of 11 Nov lost on the robot (`BV2/HANDOVER.md`; recipe section 05).
- Use sim as a bug detector and coarse filter, and pick models with a fixed real protocol (WW section 5). The reviewer's realism score did not track usefulness: v2 never reached the 7/10 pass bar, yet a 5/10 sim plus DAgger gave real successes the next day (WW section 1; MS section C).

### 3.6 Training

- Data moved the score, settings did not. Real frames at a third to a half (or a small targeted real set at the end), DAgger corrections and keeping all old data mattered; more steps on the same data gave no clear gain (r13 dropped to 11/15) (recipe section 05).
- Real-only from scratch with 60 episodes failed (0/2): the real demos contain no recoveries (recipe section 05).
- Plain bf16 without fp32 master weights left about 98% of weights unupdated; Kahan AdamW fixed it, 83% against 53% on the earlier Franka task (MS section A4; NUM).
- `train.py` defaults to `--max-epochs 5`, which stopped one run at 11.8k of 16k steps without saving (`BV2/HANDOVER.md`).

### 3.7 OpenVINO and inference bugs

Details and code lines are in `BV2/retro_1123/code_delta/inference_bugs.md`.

- Never put a random op in an exported graph, and test "same input twice must differ" on the target device (section 2.13).
- Every execution mode needs an end-to-end test per export type. Async at threshold 0 froze the arm (#465), RTC had four timing bugs (#482), and an RTC export under Sync raises KeyError while a plain export under RTC runs unguided without a warning.
- OpenVINO Sync scored 17/30 against PyTorch 22-24/30 on the same seeds; the cause was never found (item 12).
- Camera order: pi05 stacks cameras in dict order; datasets overview-first against eval wrist-first cost about 1.5 days of wrong conclusions. Bind cameras by name (#411) (MS section A4).
- Defaults differed: the runtime's Sync threshold is 0.5, Studio's 0, and Studio resets it on page load (MS section A2). A sweep of 0, 0.25 and 0.5 gave 75, 77 and 80 of 120, no significant difference (`MEM/bucket-v2-coordination-1112.md`, 10:30).
- A live Rerun viewer on the same iGPU added 40-50 ms per inference (item 10).

### 3.8 Process and infrastructure

- Disks and memory filled repeatedly: our checkpoints crashed a teammate's run on h100s, and the Mac froze or panicked at least three times. Run `df` before every launch and keep sims off the Mac (MS section E; NUM).
- One coordinator, specialist agents with a BRIEF.md and STATUS.md each, and HANDOVER.md as the single entry point let up to 7 agents work overnight. Unverified relayed numbers went wrong several times (WW section 8; MS section F).

---

## 4. Where everything lives

None of `~/real2sim-data` or `~/.claude` is under git. A backup of the v1 worktree changes and the r25 plugin is in
`~/real2sim-data/knowledge_backup_20261125/` (base commit `9d591e71`).

| Artifact | Path | In git |
|---|---|---|
| v1 real2sim runtime (compositor, Metal splat, viewer background) | `WT1/src/physicalai_mujoco_plugin/real2sim/` | No (untracked in worktree, branch `max/bucket-real2sim`) |
| v1 pipeline scripts and order | `WT1/scripts/real2sim/` and its `README.md` | No |
| v1 scene | `WT1/urdf/scenes/bucket_sort_desk/scene.xml` | No |
| v1 edits to plugin files (robot, cameras, profile, registry, tests) | `WT1/src/...`, `knowledge_backup_20261125/tracked_changes.diff` | No (modified, uncommitted) |
| v2 worktree (task objects, collision, scripts/real2sim/v2, 158 files) | `WT2/` (branch `max/bucket-real2sim-v2`) | No (uncommitted) |
| Final sim snapshot r25 (plugin, scripts, kit, splat, calibration) | `BV2/sim/r25/` (`plugin/`, `scripts/`, `real2sim.json`, `splat.ply`) | No |
| Earlier sim versions and their READMEs | `BV2/sim/r17` to `BV2/sim/r24`, `BV2/deliveries/rNN/` | No |
| Calibration jig (CAD, STL, fit code, notes) | `BV2/calib_jig/` (`JIG.md`, `AUTO_NOTES.md`) | No |
| Gap viewer, touch-off, RTC real kit, DAgger tools | `BV2/sim2real/` (`gapview/`, `rtc_real/`, `r21_dagger/`, `r23_dagger/`, `r2x_tools/`) | No |
| Offline recorder | `BV2/sim/offline/` | No |
| Sim eval harness and dashboards | `BV2/sim/eval/`, `BV2/sim/eval_dash/`, `BV2/sim/exec_ab/` | No |
| Reviewer rubric, held-out lists, reports | `BV2/review/` (`RUBRIC.md`, `HELDOUT_*.json`, `reports/r01-r17.md`) | No |
| Journal, disclosures, exposures, recording plan | `BV2/notes/` | No |
| Handover and coordination state | `BV2/HANDOVER.md`, `BV2/PENDING_PRS.md` | No |
| Model hub | `BV2/model_hub/` (`registry.json`, `register_run.py`, `refresh.py`) | No |
| Real trials and eval results | `BV2/trials/` (`trials.jsonl`, `eval_1113` to `eval_1119_pile`, `shadow_1119/`) | No |
| Booth recordings and logs | `BV2/sps2026_booth/` | No |
| Retrospective, booth analysis, code inventory | `BV2/retro_1123/` | No |
| Training scripts (launch, export chain) | h100s `~/xiangxi2_vla/scripts`; partial copies in `BV2/train_night/n11*/` | No |
| Reports (training recipe, sim evals, inference A/B, RTC plan) | `REP/*.html`, `BV2/notes/pi05-training-recipe.v1.html` | No |
| Method skill | `SKILL/` (also in the backup folder) | No |
| v1 handoff and the office LiDAR handoff | `V1/handoff/`, `~/real2sim-data/office-threejs/handoff/` | No |
| Models and exports | Hugging Face `iMaxxgx/pi05-bucket-sort-runs` (public) | n/a |
| Sim and real datasets | Hugging Face private repos (for example `iMaxxgx/bucket-sort-r23-sim`, `iMaxxgx/bucket-sort-r25-sim`, `iMaxxgx/bucket-v2-real-1112`) | n/a |
| Merged upstream pieces | physicalai #404 (virtual leader recording), #411 (camera order), #442 and #455 (noise), #454 (gripper stall), #465 (Async); Studio #1355 (Kahan), #1393, #1411, #1412, #1413, #1414 | Yes |
| RTC timing fixes | physicalai draft PR #482, branch `fix/rtc-execution-timing` | Draft |
| Studio RTC mode, booth demo features | Studio branches `max/studio-rtc`, `max/demo-conference` and others | Local branches, not merged |

---

## 5. Open problems and what we would do next

1. Commit the code. The real2sim runtime, the v2 scene and the jig live only in uncommitted worktrees and in `BV2/sim/r25/`, on one Mac. The v2 worktree is not in the backup folder (inferred from `knowledge_backup_20261125/status.txt`, which lists only v1 files). First step: land the compositor and scene from the r25 snapshot as a reviewed plugin PR.
2. Make main run what the booth ran. 22 of 24 booth features were off main on 23 Nov. The proposal maps 391 issues to 35 PRs; tier 0 is #482 plus a release with #465, cameras by serial number, one export for all modes, RTC in Studio, settings persistence and an iGPU device default (`MEM/sps2026-retrospective.md`; `BV2/retro_1123/`).
3. Settle the OpenVINO against PyTorch gap (17/30 against 22-24/30 in sim). Every OpenVINO deployment inherits it (inference_bugs.md item 12).
4. Find a sim metric that predicts the robot. Validate the realism rubric against policy success early, and model the real failure the sim lacks: repeated regrasps that time out (MS section C; `REP/sim_evals_1117.html`).
5. Turn the jig into a repeatable calibration process: stiffer towers, auto mode tested with backlash, and a wrist_roll and gripper-opening check it cannot do today (`MEM/real2sim-calibration-jig-todo.md`; `BV2/calib_jig/JIG.md`).
6. Fix the merged-dataset quantile statistics, which were far too narrow (joint 0 stored as -18 to 42 deg against a real -91 to 124 deg) and were never fixed (MS section A4).
7. Close the sim stillness gap and live planning stalls, or build the lockstep recording path so recordings are deterministic (`BV2/notes/R17_STILL_ANALYSIS.md`; `BV2/notes/R6_LOCKSTEP_DESIGN.md`).
8. Untested items: r26 with random shadows and the r16 run were planned and never trained; SnapFlow (136 against 307 ms on the kbox) tied in sim overall but was weaker on multi-object tasks (9 against 15 of 23) and never tried on the robot (MS sections C and G; `BV2/HANDOVER.md` 2026-11-22). When the named object was absent, 6 of 7 models picked a wrong present object in the 13 Nov probe (NUM; MS section C); the real absent-object case was not measured again later (`REP/sim_eval_r25_8k_vs_16k.html` notes the sim seeds never leave the CPU out).
9. Measure what was not measured: the booth success rate in the recorder itself, cost after 9 Nov, and how much the uncommitted RTC fixes contributed on the real arm (MS section H).

### Where sources disagree

- Booth run count: WW's headline says s26c "ran 88 booth runs on 23 Nov"; the booth README and analysis say 86 on 23 Nov plus 2 lab runs on 22 Nov. NUM gives 58 min under model control for all 88; the analysis gives 57.4 min for the 86 booth runs.
- Booth success: the user-facing summary often quotes 74/86; the fair-chance figure is 74/79. NUM says the success rate was "not logged by the recorder", which is true of the recorder, while the analysis supplies the visual verdicts.
- Chrome slowdown at the booth: NUM gives 420-456 ms; the booth analysis gives 414 ms p50.
- v2 reviewer rounds: NUM counts 17 reviewed rounds (r01-r17); WW and MS say 18 rounds (r18 was built but paused before review, per `BV2/notes/JOURNAL.md`).
- Shoulder-axis height: a 13 Nov note said 147 mm; `JIG.md` explains that was read from the collision mesh and gives 145.0 mm from CAD, matching the ruler.
- Pan zero: NUM gives -3.01 deg (the value r23 used); `AUTO_NOTES.md` gives -3.03 to -3.12 deg for different fit variants.
- Sim scores of model G: 27/30 in the "nocalib" sim (`BV2/HANDOVER.md` 13 Nov) and 28/30 on the iGPU r17 queue (same file, 12 Nov); the recipe page quotes 28/30.
- Kahan effect: NUM and MS give success rates (83% against 53%) on the Franka task; the training recipe reports only validation losses for that study and notes nobody tested those models on a robot.
- `BV2/notes/RECORDING_PLAN.md` still describes resistors and a 70 mm gripper; both were later dropped or changed (resistors cut 12 Nov, jaws 85 mm from r16).
