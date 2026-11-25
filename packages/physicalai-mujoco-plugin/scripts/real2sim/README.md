# real2sim pipeline scripts

The scripts that rebuilt the reBot B601 bucket-sort desk (`bucket_sort_desk`) from one phone video: SfM, a Gaussian splat
on the Mac (Brush, Metal), calibration against the CAD, the scene, and the deliveries an independent reviewer graded.
The runtime part (the compositor, `--appearance`, the viewer background) lives in `src/physicalai_mujoco_plugin/real2sim/`.

Every script reads the project folder from `REAL2SIM_ROOT` and runs with the plugin's venv, isolated:

```bash
export REAL2SIM_ROOT=/path/to/project   # outside the repo
python -I scripts/real2sim/<script>.py ...
```

Project folder layout: `video/`, `colmap/` (frames, SfM, `pinhole_cameras.json`), `splat/vN/`, `align/` (fits, masks,
looks), `calibration.json`, `kit/real2sim.json`, `notes/`, `review/` (the reviewer's; the builder never writes there),
`deliveries/rNN/`.

## Order

| Step                                                      | Scripts                                                                                                | Generic?                            |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| Frames, named by video index (the held-out rule needs it) | `extract_frames.py`                                                                                    | yes                                 |
| SfM, undistorted pinhole copy                             | `sfm_phone.py`                                                                                         | yes                                 |
| Training set without held-out frames                      | `make_train_set.py`                                                                                    | yes                                 |
| Scale and placement from known CAD                        | `fit_buckets3d.py`, `fit_black.py`, `scale_profile.py`, `overlay_buckets.py`                           | per scene: fit the objects you have |
| Robot fit at the calibrated pose                          | `robot_points.py`, `fit_robot.py`, `refit_robot_grip.py`, `robot_scale_profile.py`, `overlay_robot.py` | reBot-specific joint names          |
| Calibration file                                          | `make_calibration.py`, `calibration_extras.py`                                                         | per scene                           |
| Robot masks, cleaned splat, arm Gaussians                 | `robot_masks.py`, `clean_splat.py`, `robot_gaussians.py`                                               | yes (radii tuned for a desk arm)    |
| Scene XML and contents                                    | `build_desk_scene.py`, `settle_clutter.py`                                                             | per scene                           |
| Colours under the compositor's light                      | `robot_look.py`, `bucket_look.py`                                                                      | robot: yes; buckets: per scene      |
| Checks and renders                                        | `topdown.py`, `dev_eval.py`, `desk_episodes.py`, `render_phone.py`, `record_demo.py`                   | mostly                              |
| Reviewer commands and deliveries                          | `bucketsim.py`, `deliver.py`, `compositor.py`, `common.py`                                             | per scene                           |

`cad.py` loads the upstream scene the fits sample from. Start a new scene by copying the per-scene scripts and replacing
the bucket-specific parts; the skill `build-real2sim-scene` walks through it.
