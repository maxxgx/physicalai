# Lessons from two real2sim builds

From the Trossen booth reconstruction and the reBot `bucket_sort_desk` run (ten independent reviews, plateau at 3.0/5).

## Approach

- **Composite, do not mix.** Render the splat as an image from each camera and lay the sim over it by depth. Splat
  surfaces are fuzzy layers a few cm thick; mixed into the 3D scene they wash out small objects. The same rule holds
  in the browser viewer (`viewer_background.py`): splat as a per-client background image with a depth map.
- **Recreating the phone video with the sim running is the strongest first target.** It needs only the capture and
  forces calibration, scale, cleanup and timing to be right. Then turn to the robot cameras.
- **The robot cameras are what a policy sees.** The arm itself need not be very photoreal; spend effort on the camera
  poses, intrinsics, near field and colour. Real recordings from those cameras are the most valuable extra input.
- **The independent reviewer helped.** It caught real bugs the builder missed (contents drawn through walls, empty
  buckets in the wrist view, a deadlocked determinism test, an overclaim in the delivery notes). It also costs: about
  $15 per round. A fresh reviewer each round and a frozen rubric kept the grade honest.

## Calibration

- Trust SfM and forward kinematics over silhouettes. Silhouette fits of arms and cameras were degenerate (cables fatten
  the arm, dark parts trap optimisers); a 3D fit to dense splat points and PnP to SfM points worked.
- Take the scale from objects with known CAD and check it with a second source (the robot). In the reference run
  buckets and robot disagreed by 1.4%; declare that rather than hide it.
- Camera specs are not measurements. The D405's real vertical field of view at 640x480 was about 62 deg, not the
  datasheet's 58; the wrist camera's CAD position was 21 mm off. Triangulate marks in several frames.
- An overview camera's pitch is hard to recover when the video never looks along it. Ask the user early (intake).

## Splat (Brush 0.3 on Apple silicon)

- About 25 min for 30k steps at 1880x1057 on an M-series Mac; no pose optimisation, so pose noise blurs.
- Brush cannot read a COLMAP folder with extra model or rig files: give it a clean `sparse/0` with the three `.txt` files.
- pycolmap and torch in one process abort (duplicate libomp); keep them apart.
- HDR iPhone video (HLG) must be converted to SDR first, or colours are wrong everywhere.
- Mask the real robot before training (its CAD at the calibrated pose, dilated for cables and blur), then delete what
  is left near the arm. Desk-coloured raised Gaussians and baked shadows bite into objects resting there.

## Rendering pitfalls and their fixes

- Whole-frame darkening when a camera looked straight down: big Gaussians beside the camera were splatted across the
  image. Frustum-cull Gaussians whose centres lie outside the view, as gsplat does (`_metal_splat_kernels.py`).
- Sim geoms that a reset hides keep their groups only until the next reset: re-apply visibility on every render.
- Pick visual contents by their collision geom's contacts; visual geoms always have contacts off.
- White CAD rims show where the CAD silhouette is a few px wider than the real arm's Gaussians; occlude the avatar only
  by scene objects, not by the CAD it stands for.
- Shadows only on desk pixels (sim depth close to the splat's desk), and none from parts touching a camera.

## Process

- Filter held-out frames in every preview, not only in the training set; a mask-check sheet once showed two.
- Determinism: wait for the planner thread (`planner_running`) without stepping the sim, or runs differ.
- `pkill -f` patterns can match their own shell; a `head` pipe can kill a long job before it saves.
- The stock demonstrator missed about 1 grasp in 10 on the reBot (object 26 to 40 deg off-axis at 0.30 to 0.36 m).
  Report such limits; do not narrow the layout to hide them.
