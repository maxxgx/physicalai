---
name: build-real2sim-scene
description: Rebuilds a real robot work cell as a MuJoCo plugin scene from a phone video (SfM, a Gaussian splat on the Mac, CAD calibration, a compositor for the robot cameras) and grades it with an independent reviewer on held-out frames. Use when the user hands over a video of a real robot setup and wants it reconstructed, matched, made photoreal or turned into a digital twin in the plugin's simulation, even if they only say "make the sim look like this video". DO NOT USE FOR recording datasets or training policies, or for adding a robot profile without a real capture.
license: Apache-2.0
---

# Build a real2sim scene from a phone video

The result is a plugin scene whose geometry, cameras and task match the real cell, plus a kit (`real2sim.json`) that
composites the splat behind the sim: `physicalai-mujoco start ... --appearance real2sim.json` then streams photoreal
`overview`/`wrist` cameras that Studio records, and the browser viewer shows the splat behind the meshes.
Reference run: `bucket_sort_desk` (reBot B601, one 75 s iPhone video), graded 3.0/5 over ten reviews.
This is a local skill; the code it drives is not on `main` yet. It needs a physicalai checkout whose plugin has the
real2sim runtime (`packages/physicalai-mujoco-plugin/src/physicalai_mujoco_plugin/real2sim/desk_compositor.py`; today
the uncommitted branch `max/bucket-real2sim`) and the pipeline scripts in `packages/physicalai-mujoco-plugin/scripts/real2sim/`
(read its `README.md` for the order). Find that checkout first; if none has them, say so and stop rather than rewrite them.

## 1. Intake: ask once, before any heavy work

Probe the video's metadata first (`ffprobe`: duration, resolution, frame rate, HDR transfer) and check whether it, or
the footage it was cut from, already belongs to a real2sim project (a folder with `review/HELDOUT_PHONE.json`). If it
does, that project's held-out windows apply, and the first question is whether to continue that project or start a new one.
Choose the held-out windows (`references/review-loop.md`) from the duration alone before you look at any frame, then
skim a contact sheet of about 12 allowed frames. Then ask, in one `AskUserQuestion` call, only what the user's message
and the video do not already answer; when the user has stated something, including "there are none", take it as given.
These are the inputs that blocked or capped the reference run when they were missing:

1. **Robot and starting scene**: the plugin profile and the closest existing scene/task; which objects are the exact
   CAD the plugin already has (they give the metric scale) and which are new.
2. **Robot cameras**: model and mount of each (overview pose, especially its pitch; wrist bracket), and whether real
   recordings from them exist or can be made (a Studio dataset, a teleop session). Without them the camera grades cap at 3.
3. **Compute and budget**: Mac only (Metal, Brush) or a GPU host and when it is free; a spend or time cap.
4. **Process**: independent reviewer with held-out frames (recommended), a journal page, and what not to do
   (no dataset recording, no commits, no real-robot motion) unless asked.

Offer a recommended default for each so the user can accept in one click. Later unknowns do not stop the work: take a
measured or stated default, write it down as an open question in the journal and the delivery, and keep going.

## 2. Workflow

Do not rely on paths from earlier runs: check the tools and install what is missing (`references/setup.md`, ask
first). Set `REAL2SIM_ROOT` to a new project folder outside the repo and run scripts with `python -I`.

1. **Held-out split and reviewer setup**: write the windows chosen at intake and the frozen rubric (`references/review-loop.md`).
   - Done when: `review/HELDOUT_PHONE.json`, `review/RUBRIC.md` and `review/tools/` exist and you have not opened a held-out frame.
2. **Frames and SfM**: convert HDR to SDR (`avconvert` on macOS), `extract_frames.py`, `sfm_phone.py`.
   - Done when: at least 90% of frames register; record intrinsics in `colmap/pinhole_cameras.json`.
3. **Unmasked splat v0** (Brush, `make_train_set.py` drops held-out frames).
   - Done when: a top-down render (`topdown.py`) shows the table, objects and robot sharply.
4. **Calibration** from the known CAD: scale from the objects, robot base and joints from the arm (`fit_*`, `overlay_*`,
   `make_calibration.py`). Two independent scale sources should agree within about 1.5%.
   - Done when: CAD outlines sit on the real edges in allowed frames; uncertainties are in `calibration.json`.
5. **Masked, cleaned splat** (`robot_masks.py`, train v1, `clean_splat.py`) and the arm's own Gaussians (`robot_gaussians.py`).
6. **Scene and task in the plugin**: scene XML from the calibration, registered in `scene_registry.py` with a layout
   around the measured one; camera mounts in the profile; the kit `real2sim.json`.
   - Done when: the scripted demonstrator succeeds in the reconstructed layout (`desk_episodes.py`) and the plugin tests pass.
7. **Looks**: material colours fitted under the compositor's light (`robot_look.py`, object variants).
8. **Review loop**: self-check on the dev split (`dev_eval.py`), then a delivery and a fresh reviewer
   (`references/review-loop.md`). Fix the top blockers it names and repeat.
9. **Report**: the grade per dimension, what blocks the next level, and what the user must supply. Keep the journal page current.

## Required checks

- Never open, print or preview held-out frames; filter every contact sheet with `held()` and declare any exposure.
- No photo pixels at render time: real appearance only through the splat or fitted colours.
- Declare every physics deviation, simplification and assumed value in the delivery.
- Ask before moving a real robot, recording a dataset, training, committing or using a shared GPU.

## Keep it cheap

The reference run cost about $150, mostly reviewer rounds and image reads. Grep tool output down to summary lines; look
at small JPEG sheets (about 1200 px wide) and full-resolution crops only where a defect is; run renders and training
in the background without polling; build inline and use subagents only for reviews; score changes on the dev split
before paying for a review (about 250k tokens each); stop when two reviews in a row change no level and report.

## References

- `references/setup.md`: packages, system tools, versions and the Brush command.
- `references/lessons.md`: what worked, what failed, and pitfalls with their fixes. Read before step 2.
- `references/review-loop.md`: held-out split, rubric, reviewer brief, delivery and stop rule.
- `assets/rubric-example.md`, `assets/phone_eval.py`: the reference run's rubric and frozen scoring tool, to adapt.
