# AIGC Asset Configs

This directory will store reproducible configuration files for:

- `object_B_text_to_3d`
- `object_C_image_to_3d`

Keep prompts, paths, and experiment parameters here once stabilized.

## Step 3 environment constraints

- `threestudio_constraints.txt` pins the local AIGC environment to the README-tested PyTorch CUDA 11.8 stack and prevents unpinned dependencies, especially `xformers`, from upgrading torch unexpectedly.
- `threestudio_requirements_step3_no_git.txt` is the non-Git subset of upstream `third_party/threestudio/requirements.txt`; Git/CUDA extension packages are installed separately from pinned source clones or direct source installs because repeated GitHub clones were unstable.
- Step 3 currently pins `accelerate==0.23.0` and `huggingface_hub==0.19.4` for compatibility with `diffusers==0.19.3`.
- Keep `opencv-python` and `opencv-python-headless` below `4.10` so the existing Object C background-removal stack remains stable.
- `scripts/aigc/env_step3.sh` exports local model snapshot defaults for reproducible smoke tests:
  - `AIGC_SD15_MODEL`: project-cache `runwayml/stable-diffusion-v1-5` snapshot.
  - `AIGC_ZERO123_MODEL`: project-cache `bennyguo/zero123-diffusers` snapshot with required fp16 weights.
- Object B/C smoke scripts pass these local model paths explicitly to avoid repeated Hugging Face endpoint failures.
- `AIGC_TRAIN_LOG_INTERVAL` defaults to `50`; when positive, patched threestudio prints `[TRAIN_METRICS]` lines with available loss, gradient, and learning-rate scalars.
- Formal run scripts write realtime stdout/stderr to `<trial_dir>/train.log` using `tee -a`.
- Object B keeps failed attempts as historical evidence. The failed wooden-duck run and the stopped rubber-duck drift run must not be overwritten.
- `configs/aigc_assets/object_B_rubber_duck.yaml` records the next conservative rubber-duck retry settings: simplified prompt, `3000` max steps, `1000`-step preview cadence, seed sweep candidates, guidance `70`, and timestep range `0.05`-`0.80`.
- `configs/aigc_assets/object_B_traffic_cone.yaml` records the resumed Object B mainline after rejecting duck variants: toy traffic cone, `500`-step preview/checkpoint cadence, four validation views, seed sweep candidates, guidance `70`, and timestep range `0.05`-`0.80`.
- `configs/aigc_assets/object_C_cat_box.yaml` and `configs/aigc_assets/object_C_green_canister.yaml` record the two resumed Object C candidates. Both use single-image Magic123 coarse runs with true RGBA inputs, `500`-step preview/checkpoint cadence, four validation views, reduced SD guidance scale, and light sparsity/opaque regularization.
- AIGC run scripts refuse to reuse a non-empty trial directory so future failure cases remain available for the report.
- Object B and Object C formal scripts write explicit `training.done`, `training.failed`, or `training.stopped` markers.
- `scripts/aigc/render_object_B_checkpoint_videos.sh` watches Object B formal checkpoints and renders intermediate `it<step>-test.mp4` previews every `AIGC_OBJECT_B_VIDEO_INTERVAL` steps while the main training process continues.
