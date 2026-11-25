# Grading rubric: bucket_sort real2sim (reBot B601-RS, four buckets)

Written 2026-10-30 by the independent reviewer, before any build. FROZEN. Neither the reviewer
nor the builder may change a criterion below. Later readings go to `review/NOTES.md`; they may
clarify a criterion, never relax one. Frozen files (check them against `review/SHA256SUMS` at the
start of every review): `RUBRIC.md`, `HELDOUT_PHONE.json`, `tools/phone_eval.py`, `heldout/phone/*.jpg`.

Paths: `REV = $REAL2SIM_ROOT/review`, `PY = <reviewer venv>/bin/python -I`
(this venv has numpy, OpenCV, scikit-image, torch with MPS, lpips with cached AlexNet weights,
pycolmap 4). Run multi-word shell variables through `bash -c`; zsh does not split them.

---

## 0. Goal being graded

- **Milestone 1 (phone recreation).** The user's phone video, re-rendered along the phone's own
  camera path, with the reconstructed room and the sim reBot performing the bucket-sort task
  inside it.
- **Milestone 2 (robot cameras).** A clean real2sim scene in which the robot's overview camera
  (on the black pole behind-left of the robot) and its wrist camera render frames as close to
  the real cameras' frames as possible, and the scripted IK demonstrator completes the task end
  to end, so the scene can later produce training data and DAgger data and run the real policy.

Real reference: the phone video `~/Downloads/rebot_bucket_sort.MOV` (75.3 s, static scene, the
real robot does not move). The builder's SDR conversion `$REAL2SIM_ROOT/video/sdr_4k.mov`
and its 377 frames `$REAL2SIM_ROOT/colmap/images/fNNNNN.jpg` (t = NNNNN/15 s) are the
references. Later the user may also supply real overview and wrist recordings (section 6).

## 1. How grades work

- Seven dimensions D1-D7, each scored 0-5. A level is awarded only if every condition listed for
  it holds. The score is the highest such level.
- Every number is measured by the reviewer on the Mac, from the delivered files and the
  MANIFEST's commands. A claim the reviewer cannot reproduce counts as missing. A command that
  fails, needs CUDA, needs the network (other than a documented `uv`/`pip` install), needs
  undocumented manual steps, or runs longer than twice its stated time (or 20 min if no time is
  stated) counts as missing for this review.
- Metrics use all frames, episodes and pairs the procedure names. Nothing may be dropped,
  re-rolled or replaced. Seeds and segment times are drawn by the reviewer at review time
  (`python -c "import secrets; print(secrets.randbelow(10**6))"`) and written in the report.
- If a criterion is ambiguous, apply the reading the text most plausibly means. Write the
  reading in `NOTES.md`, and use it the same way in every later review.

### 1.1 Overall grade

Weights: D1 2, D2 2, D3 1.5, D4 1.5, D5 1, D6 1, D7 1 (sum 10).

```
raw     = (2*D1 + 2*D2 + 1.5*D3 + 1.5*D4 + D5 + D6 + D7) / 10
overall = floor(2*raw) / 2                      # rounded down to the nearest 0.5
cap     : overall <= m + 2, where m = min(D1, D2, D3, D4, D7)
          (once real robot-camera recordings exist, m also includes D5 and D6)
```

Until real robot-camera recordings exist, D5 and D6 cannot exceed 3 (section 6). That limit is
expected and not a defect. The highest overall grade before such recordings is therefore 4.5;
5.0 requires real robot-camera evidence.

Also report two milestone sub-grades (same rounding):
`M1 = floor(2*(2*D1 + 2*D2 + 1.5*D3 + 1.5*D4)/7)/2` and
`M2 = floor(2*(1.5*D3 + 1.5*D4 + 2*D5 + 2*D6 + D7)/8)/2`. Mark M2 "provisional" while D5/D6 are
limited to 3.

**Hard fail.** A violation of H1, H2, H3 or H5 (section 2), or tampering with frozen files (H6),
makes the review result `FAIL (rule violation)` with overall = 0 for that review. The report
still lists every dimension's level, with the affected dimensions at 0, so progress stays
visible. Nothing else is a hard fail.

Every report shows each dimension's level next to the previous review's level, and for each
dimension it names the one unmet condition that blocks the next level.

---

## 2. Hard rules

**H1 Held-out phone frames.** `HELDOUT_PHONE.json` lists time windows (seconds of the video) and
a guard. Every video time in a window, or within `guard_s` of one, is held out.

- The builder never opens, views or prints held-out frames or crops of them, from any source:
  the MOV, the SDR copy, `colmap/images`, undistorted copies, `review/heldout/`.
- Held-out frames are used for nothing except: (a) being registered by the joint SfM run (poses
  and intrinsics come from the global reconstruction, and its sparse points may initialise a
  splat), and (b) rendering at their registered poses. They must not be used to train or
  fine-tune any splat, texture, colour/exposure model or network. They must not be used to
  optimise any pose, the robot's estimated joint configuration, the layout, scale or lighting,
  or any render setting. No per-image refinement of their poses beyond the global SfM is
  allowed.
- Any real|sim side-by-side video the builder makes shows black in place of the real half
  inside the windows and guard.
- `phone_eval.py frames` lists which of the 377 frames are held out (evaluated) and which are
  guard (excluded, not evaluated). Frames the builder extracts from the video at other times
  follow the same time rule.

**H2 No photo pixels at render time.** Photo content may enter a judged image only (a) through
3D Gaussian splats (or other trained 3D radiance models) trained on allowed frames, or (b) as
albedo textures on meshes that stand for a real surface at its true 3D place, UV-mapped and lit
by the renderer. Not allowed: photo billboards or backdrops, screen-space photo backgrounds,
projective texturing from photos at render time, nearest-photo lookup, image-based rendering,
inpainting from photos at render time, and photo-derived environment maps that are visible as
background (they may be used for lighting and reflections only). Every photo-derived texture is
listed in the MANIFEST with its source frames.

**H3 Held-out robot-camera data.** Section 6.2 defines the rule. It is the same as H1, applied
to the real recordings.

**H4 Regeneration.** Every graded image, video, number and episode is produced by the reviewer
from the MANIFEST's commands. Delivered renders and videos are only for convenience. If a
delivered key-frame render differs from the reviewer's regeneration (PSNR < 40 dB on any key
frame), the reviewer grades its own regeneration and records the mismatch. A mismatch that
looks deliberate is treated as an H5 violation.

**H5 Disclosure.** The MANIFEST declares every exposure to held-out content (including
accidental), every photo-derived texture, every physics deviation (welds, attachments, kinematic
carrying, edited `qpos`), and every simplification. An undeclared violation found by the
reviewer is a hard fail.

**H6 Frozen review material.** The builder does not write anything under `review/`. Checksums
are verified at every review.

Allowed without restriction: the video outside the windows and guard (at any frame rate), the
upstream CAD and code, datasheets, and measurements or facts the user supplies (declared in the
MANIFEST). Reading privileged sim state inside the scripted demonstrator is also allowed.

---

## 3. Delivery (one folder per review)

`$REAL2SIM_ROOT/deliveries/rNN/` (NN = 01, 02, ...), read-only to the reviewer,
containing `MANIFEST.md` with these sections:

1. **Versions.** The git commit of the plugin checkout and a patch file for uncommitted
   changes, the builder's scripts (path and sha256), the Python env (path, lock file or
   `pip freeze`), and the Brush version, if Brush is used.
2. **Data use.** `TRAIN_IMAGES.txt`: one path per line for every image used to train or fit any
   appearance model (splats, textures, colour fits), each with its video time. `PHOTO_DIRS`:
   every directory that holds photos or frames (for the sandbox). Photo-derived textures with
   their source frames. Declared exposures to held-out content.
3. **Assets.** The distorted SfM model (`SIMPLE_RADIAL`, 1920x1080, `colmap/images` names) and
   the undistorted PINHOLE model of the same SfM, the splat file(s) and format, the scene MJCF and
   assets, and `calibration.json` (section D7, X1).
4. **Commands.** For each one: the exact shell line, working directory, environment, outputs
   and expected run time. All run headless on the Mac.
   - `HELDOUT_RENDER --frames f00001,f00002,... --out DIR`: renders any registered frame name in
     the **matched state** (see D1) at the PINHOLE camera of that name, using the registered pose
     unchanged. It writes `fNNNNN.png` (composite), `fNNNNN_seg.png`, `fNNNNN_outline.png` and
     `cameras.json`, following the conventions in the header of `tools/phone_eval.py`. It must
     run under the sandbox of section 7, step 2.
   - `VIDEO --t0 T0 --t1 T1 --out FILE`: the composite along the phone path (all registered
     poses, interpolated, missing ones included) from T0 to T1 (default: the full 0-75.3 s), at
     the PINHOLE size or at least 1280x720, 15 fps or more, with the sim running the
     demonstrator. Deterministic for a given seed. Runs under the sandbox.
   - `DEMO --seeds S1,... --pairs obj:bucket,... --out JSON`: headless episodes in the
     reconstructed scene. It writes per episode: the seed, the pair, success, the failure reason,
     the sim duration, and `physics_flags` (any equality constraint, mocap or `qpos` write
     involving an object or a finger during the episode).
   - `ROBOTCAM --seed S --out DIR`: one full episode rendered from the overview and the wrist
     cameras at the stated resolution and fps (`overview.mp4`, `wrist.mp4`, plus PNG frames with
     `_seg.png` for every 10th frame).
   - `SPEED`, `RAND --seeds ... --out DIR`, `CAL` (prints `calibration.json`): see D7.
   - After real robot-camera recordings exist: `ROBOTCAM_REPLAY --episode E --out DIR`, which
     renders both cameras at the recorded joint states and timestamps (section 6).
5. **Convenience outputs** (not graded as delivered): `composite_full.mp4`,
   `sbs_full.mp4` (real half blacked out as in H1), held-out renders, `ROBOTCAM` output.
6. **Declarations.** Physics deviations, simplifications, deviations from this rubric's
   conventions. The builder may add a self-assessment per dimension; the reviewer ignores it for
   scoring.

---

## 4. Dimensions

Px values on phone frames are pixels of the PINHOLE render (about 1900 px wide). "Key frames" are
the 10 frames printed on the last line of `phone_eval.py frames`. The tool picks the sharpest
frame of each half of each window, so the same frames are used in every review.

### D1 Phone novel-view fidelity (held-out frames)

**What.** The composite in the matched state, rendered by `HELDOUT_RENDER` at every held-out
frame's pose. **Matched state:** the scene as it was when the video was shot. The sim robot is
at the joint configuration the builder estimated for the real, static arm (from allowed frames).
The buckets and their contents are as in the video. Nothing appears that is not in the video,
and no task objects are added to the desk. The reconstruction's own copy of the real robot is
removed or hidden wherever the sim robot is drawn.

**How.** `refs` undistorts the frozen references to the PINHOLE camera. `metrics` (default scale
0.5, INTER_AREA) computes, over the valid area eroded by 5 px: `psnr_cc` (PSNR after a
per-channel gain and bias fit, which compensates only the phone's exposure and white balance),
`ssim_cc`, `lpips_cc` (AlexNet, spatial) and `psnr_raw`. It reports the median and the worst
value over all scored frames. A held-out frame that is not registered in the delivered SfM is
not scored, but the registered fraction is a condition. `cameras.json` must match the SfM poses
and intrinsics to 1e-4, or D1 = 0 (the poses were changed). Then view real | render for all 10
key frames at full resolution. A **workspace defect** is a floater, hole, smear, ghost or wrong
geometry larger than 16 px, inside the desk area within about 0.6 m of the robot, on the
buckets, or on the robot.

| Level | psnr_cc median / worst                       | ssim_cc median | lpips_cc median / worst | registered | key-frame workspace defects |
| ----- | -------------------------------------------- | -------------- | ----------------------- | ---------- | --------------------------- |
| 5     | >= 25.0 / >= 20.0                            | >= 0.82        | <= 0.18 / <= 0.32       | >= 95 %    | none                        |
| 4     | >= 23.0 / >= 18.0                            | >= 0.77        | <= 0.24 / <= 0.40       | >= 95 %    | in at most 2 key frames     |
| 3     | >= 21.0 / >= 16.0                            | >= 0.70        | <= 0.32 / <= 0.50       | >= 85 %    | any                         |
| 2     | >= 18.0 / any                                | any            | <= 0.42 / any           | >= 70 %    | any                         |
| 1     | renders exist for >= 50 % of held-out frames |                |                         |            |                             |
| 0     | missing, poses changed, or H1/H2 violated    |                |                         |            |                             |

Report the per-window medians too. They are informational, not graded.

### D2 Phone video with the sim running (integration, temporal stability)

**Evidence.**

- (a) `VIDEO` output, regenerated by the reviewer for one 5 s segment drawn at review time and
  compared with `composite_full.mp4` at the same times (H4).
- (b) 1 fps contact sheets of `composite_full.mp4` (6x7 tiles, as `ffmpeg -vf
fps=1,scale=480:-2,tile=6x7`).
- (c) Every frame of three 3 s segments chosen by the reviewer: one where the arm sweeps through
  the space the real arm occupies, one where a foreground object (pole, cable, bucket wall)
  passes in front of the arm, and one with a grasp or a release.
- (d) A blind test: `pairs` on the 10 key frames (bare names, so each 640 px crop is centred on
  the robot and the buckets), real vs matched composite. Answer all 10 in `answers.txt`, write
  its sha256 in the report, then open the key.
- (e) `colour`: CIEDE2000 of each bucket colour, real vs matched composite, over the held-out
  frames.

**Checklist** (pass/fail, judged on (b), (c) and the key frames):

- **C1 No ghosts.** No remnant of the real robot (lime green, silver or black fragments, a
  blurred arm, floaters) where the sim robot is not, in any inspected frame. Space behind the
  real arm's static pose is filled with a plausible desk or background (no hole or smear larger
  than 30 px). Same for any real object that the sim moves.
- **C2 Occlusion.** Splat foreground (pole, overview camera, cables, bucket walls, desk edge)
  hides the sim robot and objects correctly, and vice versa. No halo or fringe wider than 2 px
  around sim parts.
- **C3 Shadows.** The sim robot and the objects cast soft shadows on the desk that move with
  them, with direction and softness consistent with the real shadows in the video (for example
  of the buckets and the pole). Resting objects have contact shadows. No shadow from something
  that is not there.
- **C4 Lighting and materials.** The robot's lime green, silver and black parts, the buckets and
  the objects match the exposure, white balance and specular look of the splat around them.
  Nothing looks self-lit or flat-lit.
- **C5 Temporal stability.** In (c), at most 2 popping or flicker events in total (a region of
  20 px or more changing abruptly in a way camera motion does not explain). Resting objects do
  not jitter.
- **C6 No drift.** The sim robot base and the buckets stay fixed against the splat desk: within
  2 px frame to frame and 4 px across each segment of (c).
- **C7 Task visible.** At least two complete pick-and-place episodes play out in the video,
  mostly in view, with the objects visibly ending inside the named buckets.
- **C8 Plausible contact.** No object visibly passes through the desk, a bucket wall or the
  fingers. A carried object stays between the fingers.

| Level | Conditions                                                                                                             |
| ----- | ---------------------------------------------------------------------------------------------------------------------- |
| 5     | C1-C8 pass; blind <= 7/10 correct; every bucket colour mean dE2000 <= 6 (colours with >= 2 scored frames); (a) matches |
| 4     | C1, C2, C6, C7 pass, at most one other fails; blind <= 9/10; dE2000 <= 10                                              |
| 3     | C1 and C7 pass, at most three others fail; dE2000 <= 15                                                                |
| 2     | a composite video along the phone path with the sim doing the task exists, but C1 or C7 fails, or more than three fail |
| 1     | composites only at still poses, or the sim not composited into the reconstruction                                      |
| 0     | missing, or H1/H2 violated                                                                                             |

### D3 Scene layout and geometric fidelity

**Overlay test.** Run `overlay` on the `HELDOUT_RENDER` outlines (matched state) for the 10 key
frames. View the 2x quadrant crops. For each visible element, estimate the largest perpendicular
gap between the outline and the matching real edge, in reference px, using the 10 px grid; round
up to the next 2 px. Elements: E1 robot silhouette at the matched pose (cables excluded); E2
robot base plate and clamp; E3 each bucket's rim and outer edges; E4 desk edges within about
0.6 m of the robot base. Per key frame, take the largest gap over E1-E4. Report the median and
the worst over the 10 key frames. A missing outline for a visible element counts as 100 px.

**Checklist.**

- **L1 Buckets.** Four buckets (black, green, red, blue) built from the bucket CAD, in one row,
  in the real order, with the real orientation (low-lip open end toward the robot).
- **L2 Bucket contents.** Each bucket's contents match the video (object types per bucket, about
  the same fill level), as sim bodies or splat. If a bucket can move in the sim, its contents
  move with it; otherwise the buckets are declared fixed.
- **L3 Desk.** The white top has the real outline near the robot (edges, corner), with the
  adjacent desk surfaces present. The sim's collision desk coincides with the visual desk:
  resting sim objects show no visible gap or sinking larger than 2 px in key frames or in the
  segments of D2(c).
- **L4 Robot mounting.** The base plate is at the real place at the desk corner, with the real
  yaw (the matched sim arm overlays the real arm). The clamp is present.
- **L5 Overview pole and camera head.** Present at their real place. If the arm can reach the
  pole, the pole is also a sim collision body.
- **L6 Clutter and background.** The near clutter is present (cables on the desk, the tape
  measure, the small items toward the desk's far end, the other robots and devices on the
  neighbouring desks), and so is the office background (chairs, desks, monitors, glass wall).
  Nothing large is missing in any key frame.
- **L7 Metric scale.** Stated with a 1-sigma uncertainty, from at least two independent
  references (for example the robot CAD and the bucket CAD), which agree within 1 %.
- **L8 No duplicates.** Nothing appears twice in a key frame (for example a splat bucket next to
  an offset sim bucket, or a splat arm next to the sim arm).

| Level | Overlay median / worst                                                         | Checklist                                   |
| ----- | ------------------------------------------------------------------------------ | ------------------------------------------- |
| 5     | <= 4 / <= 8 px                                                                 | L1-L8 pass                                  |
| 4     | <= 8 / <= 15 px                                                                | at most one fails, and it is not L1 or L4   |
| 3     | <= 15 / <= 30 px                                                               | L1 and L4 pass                              |
| 2     | <= 40 px median                                                                | or L1 fails (a bucket missing, wrong order) |
| 1     | upstream layout roughly placed in the reconstruction, or no outlines delivered |                                             |
| 0     | missing                                                                        |                                             |

### D4 Task in the reconstructed scene

Run `DEMO` with 20 seeds drawn by the reviewer and 20 (object, bucket) pairs: the 12 pairs
(resistor, 9 V battery, CPU) x (black, green, red, blue) once each, plus 8 more drawn at random,
so that every bucket is named at least 4 times. Physics deviations and plugin registration are
checked with `git diff` against upstream and with `grep -nE "equality|weld|mocap"` on the scene
files. Read `physics_flags`. Make contact sheets of `ROBOTCAM` overview frames for two episodes
and confirm that a success is real (the object visibly lies inside the named bucket).

Success means the named object rests inside the named bucket, out of the gripper, within the
episode's time limit (the upstream `BucketSortTask.evaluate` semantics; objects already in a
bucket at the start do not count against it). The layout is the reconstructed one: the buckets
at their reconstructed poses, and the objects spawned on the desk within the arm's reach.

| Level | Conditions                                                                                                                                                                                                                                                                                                      |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5     | >= 19/20 successes; all 4 buckets and all 3 object types succeed at least once; reconstructed layout (D3 L1 passes); registered as a scene or scene variant in `physicalai-mujoco-plugin` with the `rebot_b601` profile and runnable from its CLI; contact-only (no `physics_flags`, no attachment or teleport) |
| 4     | >= 17/20; otherwise as level 5                                                                                                                                                                                                                                                                                  |
| 3     | >= 14/20; four buckets; contact-only, or one declared deviation; may run as a standalone script                                                                                                                                                                                                                 |
| 2     | >= 1/20 in the reconstructed layout; or only three buckets; or attachments/welds used (declared)                                                                                                                                                                                                                |
| 1     | runs only in the upstream layout                                                                                                                                                                                                                                                                                |
| 0     | nothing runs; or an undeclared physics shortcut (hard fail under H5)                                                                                                                                                                                                                                            |

A crash, hang (over 3x the median episode time) or exception counts as a failure.

### D5 Overview camera fidelity

Before real overview recordings exist, the maximum is 3. Checks:

- **O1 Device and intrinsics.** The device model is identified from the phone imagery (housing
  shape, any visible label), with resolution, aspect, field of view and distortion taken from its
  datasheet or measured, and justified in the MANIFEST.
- **O2 Position.** The optical centre is estimated in the scene frame from the phone SfM (for
  example by triangulating the housing in at least 3 allowed frames), with a stated 1-sigma
  uncertainty. In the key frames where the camera head is visible, the sim overview-camera
  housing outline (orange) lies within 10 px of the real housing.
- **O3 Orientation.** Estimated from the housing's front-face direction (and the aim the real
  mount allows), with a stated uncertainty of 10 deg or less, and consistent with the housing
  seen in the key frames.
- **O4 View content** (`ROBOTCAM`, every 10th frame plus every grasp and release). All four
  buckets, the object area and the gripper during grasps are in view. There are no holes,
  floaters, smears or inside-out splat in the workspace. Stated resolution and fps are met.

| Level | Before real recordings                                                                                             |
| ----- | ------------------------------------------------------------------------------------------------------------------ |
| 3     | O1-O4 pass on every inspected frame                                                                                |
| 2     | O2 within 25 px, or O1/O3 only partly justified, or workspace defects in <= 20 % of inspected frames               |
| 1     | the camera exists, but at the upstream nominal pose or with unjustified parameters, or defects in > 20 % of frames |
| 0     | missing                                                                                                            |

After real recordings exist (section 6), level 3 must hold, plus, on the held-out real overview
frames (20 frames sampled uniformly over the held-out episodes, `prep` then `ROBOTCAM_REPLAY`
renders, `metrics --scale 1.0` with a mask excluding seg labels 20-99 dilated 25 px and the
reviewer's rectangles around real objects whose positions the sim cannot know, `overlay` for
bucket-rim and desk-edge outlines, and `pairs` on 10 of the 20 frames):

| Level | Edge gap median (native px) | psnr_cc median | lpips_cc median | blind   | bucket dE2000 |
| ----- | --------------------------- | -------------- | --------------- | ------- | ------------- |
| 5     | <= 2                        | >= 25.0        | <= 0.15         | <= 7/10 | <= 6          |
| 4     | <= 4                        | >= 21.0        | <= 0.25         | <= 9/10 | <= 10         |

If, once real frames exist, the edge gap exceeds 12 px, D5 is at most 2 (the calibration is
shown wrong).

### D6 Wrist camera fidelity

Before real wrist recordings exist, the maximum is 3. Checks:

- **W1 Mount.** Pose from the mount CAD (`REBOT_B601_WRIST_CAMERA` or a justified correction), on
  the correct side of the gripper. In the key frames where the real wrist camera is visible, the
  sim housing outline (green, matched state) lies within 10 px of it.
- **W2 Intrinsics.** The FOV for the stated stream resolution is derived from the D405 colour
  datasheet, with the derivation written down. The distortion model is stated.
- **W3 View content** (`ROBOTCAM` wrist stream, every 10th frame plus 3 frames around every grasp
  and release). The fingers are visible at the place the CAD implies. The grasped object sits
  between the fingers at grasp. There are no holes, floaters or smears at close range (desk at
  5-30 cm, bucket interiors and contents). No splat blob larger than 16 px at 640x480 is visible
  on the desk or the buckets.
- **W4** The stream's resolution and fps are as stated, and frames are time-aligned with the
  joint states.

| Level | Before real recordings                                                                          |
| ----- | ----------------------------------------------------------------------------------------------- |
| 3     | W1-W4 pass on every inspected frame                                                             |
| 2     | W1 within 25 px, or close-range defects in <= 20 % of inspected frames                          |
| 1     | the camera exists, but its mount or FOV is wrong or unjustified, or defects in > 20 % of frames |
| 0     | missing                                                                                         |

After real recordings exist (same procedure as D5, on wrist frames; the fingers are included in
the metrics):

| Level | Edge gap median (native px) | psnr_cc median | lpips_cc median | blind   | bucket dE2000 |
| ----- | --------------------------- | -------------- | --------------- | ------- | ------------- |
| 5     | <= 4                        | >= 23.0        | <= 0.20         | <= 7/10 | <= 6          |
| 4     | <= 8                        | >= 19.0        | <= 0.30         | <= 9/10 | <= 10         |

If, once real frames exist, the edge gap exceeds 20 px, D6 is at most 2.

### D7 Real2sim2real readiness

- **X1 Calibration.** `calibration.json` gives each of the following with a 1-sigma uncertainty
  and its method: the similarity between the sim world and the SfM (scale, rotation,
  translation), the robot base pose on the desk, the estimated static joint configuration of the
  real arm in the video, the overview intrinsics and extrinsics, and the wrist mount and
  intrinsics. The stated uncertainties must be consistent with what the reviewer observes: the
  D3 worst gaps for E1-E3 lie within 3 sigma converted to px at the key frames' depth (about
  1 mm per px at 1.4 m).
- **X2 Randomization.** At least 5 documented knobs with ranges and defaults, each shown in the
  `RAND` grid (overview and wrist, 6 seeds). They include lighting (intensity, colour
  temperature, direction), camera-extrinsics jitter for both cameras, object poses and count,
  bucket poses (with their contents), and image-level exposure, white balance and noise. Changes
  stay physically consistent: nothing floats, and contents move with their buckets.
- **X3 Render speed.** `SPEED`: simulated frames per wall-clock second, where one frame is one
  overview image plus one wrist image at the stated stream resolution (640x480 unless the real
  dataset says otherwise), with physics and the demonstrator running. Measured by the reviewer
  over at least 300 frames on the Mac, with no other heavy job running.
- **X4 Reproducibility.** Every MANIFEST command runs from a fresh shell using only what the
  MANIFEST lists. The same seed twice gives identical `DEMO` JSON, and 3 frames rendered twice
  differ by at most 2/255 on 99.9 % of pixels.
- **X5 Plugin integration.** Registered in `physicalai-mujoco-plugin` with the `rebot_b601`
  profile (its channel names and units). The cameras are named `overview` and `wrist`. It runs
  through the plugin's autopilot or virtual-leader path, so that Studio recording can later use
  it unchanged.
- **X6 Policy-facing config.** Camera names, resolutions and fps match the real policy's
  dataset config if the user has supplied it. Otherwise the assumption is written down and
  used consistently.

| Level | Conditions                                                    |
| ----- | ------------------------------------------------------------- |
| 5     | X1-X6 pass; X3 >= 10 fps                                      |
| 4     | X1, X3, X4 pass, at most one of X2/X5/X6 fails; X3 >= 4 fps   |
| 3     | X1 stated (uncertainty may be coarse), X4 passes; X3 >= 1 fps |
| 2     | runs only as standalone scripts, or X4 fails, or X3 < 1 fps   |
| 1     | renders exist but cannot be reproduced from the MANIFEST      |
| 0     | nothing                                                       |

---

## 5. What this rubric cannot establish

- Robot-camera realism without real robot-camera frames (D5 and D6 are limited to 3 until then).
- The real arm's dynamics, grasp forces and object physics. The video is static, so D4 grades
  task completion and honesty, not fidelity of motion.
- Transfer of the real policy. This is not graded until a policy and real recordings exist; D7
  grades only readiness.
- The blind test has one grader, who has seen the real video. That grader is a stricter observer
  than a casual viewer.

---

## 6. Real robot-camera recordings (when the user supplies them)

### 6.1 What counts

Real overview and/or wrist frames of this scene, preferably as a Studio/LeRobot dataset with
synchronised joint states. Without joint states, wrist frames can be graded only on static-region
geometry (edge gaps), not on psnr/lpips. In that case D6 levels 4-5 require the joint states.

### 6.2 Split rule (deterministic, applied before anyone looks at the recordings)

- **Episodes.** Rank the episode indices i by the hex string
  `sha256(f"bucket-heldout:{i}".encode()).hexdigest()`, ascending. The first `max(1, ceil(0.25 *
n_episodes))` are held out.
- **One continuous recording per camera.** Cut it into consecutive 10 s blocks (block b = [10b,
  10b+10) s). Rank the blocks by the same hash with b in place of i. The first `max(1, ceil(0.25 *
n_blocks))` are held out, with a 1 s guard on each side.
- The builder applies the rule, moves the held-out parts unopened to
  `$REAL2SIM_ROOT/robotcam_heldout/`, writes `HELDOUT_ROBOTCAM.json` (episodes or blocks,
  with the hash values) into the delivery, and never opens them. The reviewer recomputes the rule.
- All other real robot-camera data may be used for anything: intrinsics, hand-eye calibration,
  colour and exposure, training splats or textures.
- If the user records several sessions, the rule applies per session, and every session
  contributes held-out data.

---

## 7. Review procedure (fresh reviewer, about one hour)

Read only this file, the MANIFEST and the evidence. Do not read the builder's notes or chat. You
may read `review/NOTES.md` and the previous report in `review/reports/` for readings and deltas.
Work in `REV/work/rNN/`. Write the report to `REV/reports/rNN.md`.

1. **Integrity (2 min).** `cd REV && shasum -a 256 -c SHA256SUMS`. A mismatch stops the review
   (H6). Record the MANIFEST's sha256.
2. **Rules (10 min).**
   - `PY tools/phone_eval.py leakcheck --train <TRAIN_IMAGES.txt> --out work/rNN/leak.json`. Any
     flag is H1, unless its declared time (checked by decoding that time from `sdr_4k.mov`) lies
     outside the windows and guard.
   - Grep the render path for photo reads (`imread`, `VideoCapture`, `ffmpeg`, `Image.open`,
     `cv2.imdecode`) and check each one against H2.
   - Write `work/rNN/deny.sb`:
     ```
     (version 1)(allow default)
     (deny file-read* (subpath "<HOME>/Downloads") (subpath "<REAL2SIM_ROOT>/video")
       (subpath "<REAL2SIM_ROOT>/colmap/images") (subpath "<REAL2SIM_ROOT>/review/heldout")
       (subpath "<REAL2SIM_ROOT>/robotcam_heldout")
       (regex #"^<HOME>/.*\.(jpe?g|JPE?G|mov|MOV|mp4|MP4|heic|HEIC)$"))
     ```
     Add a `(subpath ...)` for every `PHOTO_DIRS` entry. Run `HELDOUT_RENDER` and the `VIDEO`
     segment with `sandbox-exec -f work/rNN/deny.sb <command>`. A command that fails only under
     the sandbox violates H2 unless the failing read is clearly not a photo (record it).
3. **D1 (10 min, start the render first).** `HELDOUT_RENDER` on all held-out frame names (from
   `frames`) into `work/rNN/ho/`. Run `refs --model <distorted> --pinhole <pinhole> --out
work/rNN/refs`, then `metrics --renders work/rNN/ho --refs work/rNN/refs --out
work/rNN/d1.json`. View the 10 key frames real | render.
4. **D4 in the background (start right after 3).** `DEMO` with the reviewer's seeds and pairs.
5. **D3 and D2 (25 min).** `overlay` on the key frames, then measure. `colour`, then `pairs`
   followed by answering. Make the contact sheets and the three segments, then go through the
   checklists. Regenerate the 5 s `VIDEO` segment and compare it with the delivered video.
6. **D5/D6 (5 min).** `ROBOTCAM` with a reviewer seed; check O1-O4 and W1-W4. After real
   recordings exist, follow section 6 and the post-real tables.
7. **D7 (5 min).** `SPEED`, `RAND`, a determinism rerun of 3 key-frame renders and 2 `DEMO`
   seeds, and a review of `calibration.json`.
8. **Report.** For each dimension: the level, the previous level, the key numbers, evidence paths
   and the condition blocking the next level. Then the overall grade with the formula filled in,
   M1, M2, the rule checks, and the readings added to `NOTES.md`. Do not tell the builder
   anything about the held-out frames' content.
