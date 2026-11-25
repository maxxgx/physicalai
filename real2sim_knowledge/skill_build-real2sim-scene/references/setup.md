# Setup

Assume nothing is installed and no earlier project exists. Check each tool, then install what is missing after the user
approves (installs change their machine). If the user cannot be reached, install only into venvs inside the project folder (nothing
system-wide) and declare it; a Homebrew or system install waits for them. Versions below are the ones the reference run used on an M-series Mac.

## Environments

pycolmap and torch abort when loaded in one process (two copies of libomp), so keep them apart: run the SfM steps and
the torch steps as separate commands. A single venv works as long as no one script imports both.

| Venv     | Install                                                                                                 | Used by                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Builder  | `uv pip install -e packages/physicalai-mujoco-plugin "torch==2.14.*" scipy pycolmap==4.2.1`             | all scripts, the plugin, the Metal splat renderer (torch MPS)                 |
| Reviewer | a separate venv: `uv pip install numpy opencv-python-headless scikit-image lpips torch pycolmap==4.2.1` | `review/tools/phone_eval.py` only, so the reviewer never imports builder code |

The LPIPS AlexNet weights download on first use; fetch them once at setup so reviews run offline.

## System tools

- `ffmpeg`/`ffprobe` (Homebrew: `brew install ffmpeg`).
- `avconvert` (ships with macOS) for HDR (HLG) iPhone video to SDR: `avconvert -s in.MOV -p Preset3840x2160 -o sdr_4k.mov`;
  check `ffprobe` reports `color_transfer=bt709` afterwards.
- Brush 0.3.0 (Apache-2.0), the `brush-app-aarch64-apple-darwin` release from github.com/ArthurBrussee/brush. Check
  its `.sha256`, unpack it anywhere and set `BRUSH_APP` to the binary.

## Splat training

In a folder from `make_train_set.py` (`images/`, `sparse/0/` with the three `.txt` files, optional `masks/`):

```bash
"$BRUSH_APP" . --total-steps 30000 --sh-degree 3 --growth-stop-iter 20000 --max-resolution 1920 \
  --export-every 30000 --export-path out > brush.log 2>&1
```

About 25 min on an M4 Max at 1880x1057; run it in the background. The result is `out/export_30000.ply` (SfM frame).

## Without a Mac GPU

The Metal renderer needs Apple silicon. On a CUDA host the booth kit's gsplat path renders instead; the desk compositor
has not been ported to it.
