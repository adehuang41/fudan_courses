# Object C Notes

## 2026-05-31 green container formal input

- Current formal Object C: a simple dark green cylindrical container with a flat lid and smooth glossy surface.
- Prompt: `a simple dark green cylindrical container with a flat lid, smooth glossy surface`
- Raw input path: `data/raw/object_C_green_container/container_raw.jpg`
- Resolved storage path: `/data/dechao/cv_final_pj/data/raw/object_C_green_container/container_raw.jpg`
- File type: JPEG, 1024x2260, RGB
- Visual inspection: single dark green cylindrical container with a flat lid, mostly centered, on a light background; the object is not transparent and has a glossy highlight.
- Decision: use this image for both Object C smoke testing and formal Object C training. Do not run the earlier apple smoke-test path unless the user explicitly reverts this decision.
- Source note: user-confirmed self-captured single-image input, suitable as final Object C source.
- Prompt decision: locked to `a simple dark green cylindrical container with a flat lid, smooth glossy surface`; use this wording for Magic123 smoke tests, formal runs, and acceptance notes unless the run clearly fails.

## 2026-05-31 RGBA and smoke-test execution

- RGBA input path: `data/processed/object_C_green_container/container_rgba.png`.
- Mask path: `data/interim/object_C_green_container/container_mask.png`.
- Preview path: `outputs/previews/aigc_assets/object_C_image_to_3d/green_container/object_C_green_container_rgba_preview.png`.
- RGBA, mask, and preview are all `1024x1024`; RGBA is PNG with alpha, mask is grayscale, preview is RGB.
- Background removal method: `rembg` Python API with `u2netp`.
- `rembg` model cache: `data/interim/cache/rembg/u2netp.onnx`, total cache size about `4.4M`.
- RGBA summary: `data/interim/object_C_green_container/container_rgba_summary.json`; alpha nonzero fraction `0.3842201232910156`, alpha opaque fraction `0.35272789001464844`.
- Reproducible config: `configs/aigc_assets/object_C_green_container.yaml`.
- Smoke-test entrypoint: `scripts/aigc/run_object_C_magic123_coarse_smoke.sh`.
- Target output root: `outputs/aigc_assets/object_C_image_to_3d/green_container/`.
- Threestudio Magic123 dependency recovery is complete enough for non-gradio `launch.py` use: `pip check` passes, core CUDA extension imports pass, and `threestudio` / `BaseSystem` imports pass.
- Smoke script now uses local `AIGC_SD15_MODEL` (`runwayml/stable-diffusion-v1-5`) for prompt processing and SD guidance.
- First Object C launch loaded local SD successfully but failed while downloading Zero123 weights from `cas-bridge.xethub.hf.co` with a TLS EOF.
- Required Zero123 fp16 files were downloaded into the local project cache. `AIGC_ZERO123_MODEL` now points to `data/interim/cache/huggingface/hub/models--bennyguo--zero123-diffusers/snapshots/b5289c24d8549e3a4737d0c34ab1347e5f074fbe/`.
- Successful run: `outputs/aigc_assets/object_C_image_to_3d/green_container/smoke_test/coarse/magic123-coarse-sd/container_rgba.png-a_simple_dark_green_cylindrical_container_with_a_flat_lid,_smooth_glossy_surface@20260531-105653/`.
- Run settings: CUDA device 6, `trainer.max_steps=25`, `trainer.val_check_interval=25`, `checkpoint.every_n_train_steps=25`.
- Output check: `2` checkpoints, `4` validation PNGs, `120` test PNGs, `it25-test.mp4`, config snapshots, TensorBoard logs, and CSV metrics were written.
- Visual note: `25` steps produce only an initial blob-like shape, so this is a runtime smoke pass, not final Object C geometry.

## 2026-05-31 formal logging entrypoint

- Formal coarse script: `scripts/aigc/run_object_C_magic123_coarse_formal.sh`.
- Default output root: `outputs/aigc_assets/object_C_image_to_3d/green_container/final/coarse/`.
- Default settings after the formal-preview update: `10000` coarse steps, validation every `1000` steps, checkpoint every `1000` steps.
- The script precomputes the trial directory and writes realtime stdout/stderr to `train.log` with `tee -a`.
- `TrainMetricsPrinterCallback` prints `[TRAIN_METRICS]` lines every `AIGC_TRAIN_LOG_INTERVAL` steps; default interval is `50`.
- Expected metric names include `train/loss_rgb`, `train/loss_mask`, `train/loss_sd`, `train/loss_sd_3d`, `train/grad_norm`, `train/grad_norm_3d`, and `lr-Adam/*` when available.
- Refine-stage logging should use the same callback and `tee` pattern after the coarse checkpoint is accepted and a refine script is prepared.

## 2026-06-01 Object C formal-preview policy

- After the rubber-duck videos were visually rejected, Object C followed the same preservation/preview discipline.
- Object C formal coarse training remains `10000` steps, matching the upstream `magic123-coarse-sd.yaml` and the existing formal script default.
- `scripts/aigc/run_object_C_magic123_coarse_formal.sh` now defaults to validation every `1000` steps and checkpoint every `1000` steps, so validation PNGs and checkpoints align for best-checkpoint selection.
- New helper script: `scripts/aigc/render_object_C_checkpoint_videos.sh`.
- The formal script defaults `AIGC_OBJECT_C_COARSE_RENDER_AFTER_TRAIN=1`: after training finishes, it renders or verifies orbit videos for each `1000`-step checkpoint on the same CUDA device used for training. This avoids using a second GPU while still preserving videos for best-checkpoint selection.
- Expected per-1000-step artifacts in a successful full coarse run: `ckpts/epoch=0-step<N>.ckpt`, `save/it<N>-*.png` validation outputs, and `save/it<N>-test.mp4` orbit videos for `N=1000,2000,...,10000`.
- The script still refuses to reuse a non-empty trial directory and writes `training.done`, `training.failed`, or `training.stopped` markers.

## 2026-06-01 Object C formal coarse run started

- A first detached launcher attempt created `outputs/aigc_assets/object_C_image_to_3d/green_container/final/coarse/magic123-coarse-sd/green_container_coarse_10000steps@20260601-045913/` and wrote the run header only; it did not reach `launch.py` or use the GPU. Preserve this directory as a start-attempt record and do not overwrite it.
- Active formal coarse run directory: `outputs/aigc_assets/object_C_image_to_3d/green_container/final/coarse/magic123-coarse-sd/green_container_coarse_10000steps@20260601-050016/`.
- Launch method: detached `setsid` run so training continues after the terminal session ends.
- CUDA policy: `CUDA_DEVICE=6`; post-training checkpoint-video rendering will use the same device through `AIGC_OBJECT_C_VIDEO_CUDA_DEVICE=6`.
- Run settings: `10000` steps, validation every `1000`, checkpoint every `1000`, post-training checkpoint-video render every `1000`, metric print interval `50`.
- Realtime log: `outputs/aigc_assets/object_C_image_to_3d/green_container/final/coarse/magic123-coarse-sd/green_container_coarse_10000steps@20260601-050016/train.log`.

## 2026-06-01 Object C formal coarse visual issue

- Run inspected: `outputs/aigc_assets/object_C_image_to_3d/green_container/final/coarse/magic123-coarse-sd/green_container_coarse_10000steps@20260601-050016/`.
- Validation finding: front-view PNGs such as `save/it4000-0.png` look close to the input cylinder, but side/back validation PNGs such as `save/it1000-1.png` and `save/it4000-1.png` contain a large side bulge.
- The bulge is visible in RGB, normal, and opacity panels, so it is real learned geometry rather than only texture, lighting, or a display artifact.
- The side bulge is already present at `1000` steps and persists through `4000` steps, so it is an early single-view hallucination rather than a late overtraining drift.
- Likely cause for report: the single front-view input constrains the visible front silhouette, but Magic123/Zero123 and SD guidance must infer unobserved side/back geometry. The word `container` plus weak explicit negatives can be interpreted as a handled cup/container, and the current config has `lambda_sparsity: 0.0` and `lambda_opaque: 0.0`, so extra side mass is not strongly penalized.
- The run was paused/stopped after the side-bulge diagnosis. The process group was terminated without deleting outputs.
- Stopped run artifacts: `training.stopped` marker exists; validation PNGs and checkpoints are preserved through `6000` steps. The train log reached about `6900` steps before termination, but no `7000` checkpoint/validation set was written.
- Prompt audit: this run did pass `system.prompt_processor.prompt=a simple dark green cylindrical container with a flat lid, smooth glossy surface` as part of the threestudio Magic123 command. This prompt was already in the Object C plan/config, but it should be described carefully in the report as a Magic123 caption/text prior used with a single input image, not as a text-to-3D Object B-style source.
- Conservative next-run note: if strict single-image compliance is prioritized, avoid detailed negative prompts and use only a minimal factual caption required by the method, or reduce text-conditioned SD influence after inspecting the Magic123 config knobs.

## 2026-06-01 Object C canister retry launched

- User confirmed that a moderate text prompt is acceptable for the single-image-to-3D method, as long as the source remains the single Object C image and the report records the configuration clearly.
- New run goal: avoid the previous side-handle/side-bulge hallucination and cap training at `3000` steps instead of `10000`.
- Training script update: `scripts/aigc/run_object_C_magic123_coarse_formal.sh` now accepts and logs `AIGC_OBJECT_C_NEGATIVE_PROMPT`, passes it to `system.prompt_processor.negative_prompt`, and forwards it to checkpoint-video rendering.
- Video render script update: `scripts/aigc/render_object_C_checkpoint_videos.sh` now uses the same Object C negative prompt for `--test` renders.
- Config file updated: `configs/aigc_assets/object_C_green_container.yaml`.
- Prompt: `a plain dark green cylindrical canister with a flat lid, smooth glossy surface, symmetric shape`.
- Negative prompt: `handle, mug, cup, side bulge, side protrusion, spout, extra part, asymmetric shape, deformed cylinder`.
- Run settings: CUDA `6` only, `3000` max steps, validation every `1000`, checkpoint every `1000`, post-training checkpoint-video render every `1000`.
- Planned run directory: `outputs/aigc_assets/object_C_image_to_3d/green_container/final/coarse/magic123-coarse-sd/green_container_canister_nobulge_3000steps@20260601-061923/`.
- Preserve all previous Object C directories, including the stopped `green_container_coarse_10000steps@20260601-050016` run, for report evidence.

## 2026-06-02 resumed Object C candidates

- Object C is now selected from two single-image candidates rather than continuing only the old `green_container` line.
- Candidate C1 is `cat_box`: a light green cat-shaped container with a rounded body, two ears, and a simple face.
  - Raw input: `data/raw/object_C_cat_box/cat_box_raw.jpg`.
  - RGBA target: `data/processed/object_C_cat_box/cat_box_rgba.png`.
  - Reproducible config: `configs/aigc_assets/object_C_cat_box.yaml`.
- Candidate C2 is `green_canister`: a light green cylindrical container/canister with a flat circular lid.
  - Raw input: `data/raw/object_C_green_canister/canister_raw.jpg`.
  - RGBA target: `data/processed/object_C_green_canister/canister_rgba.png`.
  - Reproducible config: `configs/aigc_assets/object_C_green_canister.yaml`.
- Terminology note: do not call C2 a kettle/water bottle in prompts or report text. It has no handle or spout; the canonical project term is green canister.
- Validation policy: keep Magic123's four validation views and make the cadence explicit every `500` steps, because the previous side-bulge failure was only clear in side/back validation views.
- Run order: try C1 first. If C1 clearly fails at `500` or `1000`, move to C2. If C1 is maybe/pass by `1500` or `3000`, still run C2 at least to `1500` as a backup and report comparison.
- Conservative Magic123 policy for this resumed pass: reduce SD guidance scale from the previous `100` to `50`, keep Zero123 guidance scale at `5`, and add light sparsity/opaque regularization (`lambda_sparsity=0.1`, `lambda_opaque=0.001`) to discourage detached extra geometry without turning this into a large ablation.

## 2026-06-02 Object C candidate results

- The user-provided root images were moved into canonical raw-data directories:
  - `猫盒子.jpg` -> `data/raw/object_C_cat_box/cat_box_raw.jpg`.
  - `绿色水壶.jpg` -> `data/raw/object_C_green_canister/canister_raw.jpg`.
- C2 terminology was corrected before prompting: despite the source filename, the object has no handle or spout, so the canonical term is `green_canister`, not kettle or water bottle.
- RGBA preprocessing used `scripts/aigc/prepare_object_c_rgba.py --method rembg`.
  - C1 RGBA: `data/processed/object_C_cat_box/cat_box_rgba.png`.
  - C1 mask: `data/interim/object_C_cat_box/cat_box_mask.png`.
  - C1 preview: `outputs/previews/aigc_assets/object_C_image_to_3d/cat_box/object_C_cat_box_rgba_preview.png`.
  - C1 summary: `data/interim/object_C_cat_box/cat_box_rgba_summary.json`.
  - C2 RGBA: `data/processed/object_C_green_canister/canister_rgba.png`.
  - C2 mask: `data/interim/object_C_green_canister/canister_mask.png`.
  - C2 preview: `outputs/previews/aigc_assets/object_C_image_to_3d/green_canister/object_C_green_canister_rgba_preview.png`.
  - C2 summary: `data/interim/object_C_green_canister/canister_rgba_summary.json`.

Shared C candidate settings:

```text
Magic123 coarse
seed: 0
sd_guidance_scale: 50
zero123_guidance_scale: 5.0
lambda_sparsity: 0.1
lambda_opaque: 0.001
validation/checkpoint interval: 500
validation views per checkpoint: 4
GPU policy: CUDA 6 only
```

- C1 cat-box run: `outputs/aigc_assets/object_C_image_to_3d/cat_box/final/coarse/magic123-coarse-sd/cat_box_coarse_gsd50_z5_sparse01_3000steps@20260602-181407/`.
  - Stopped after the `1000` validation review; marker: `training.stopped`.
  - Preserved checkpoints: `500`, `1000`, plus `last.ckpt`.
  - Evidence sheets: `outputs/previews/aigc_assets/object_C_image_to_3d/cat_box/object_C_cat_box_it500_views.jpg`, `outputs/previews/aigc_assets/object_C_image_to_3d/cat_box/object_C_cat_box_it1000_views.jpg`.
  - Visual result: shape-only backup. The run produced a single rounded container-like object with faint ears in some views, but the face was missing and side/back views remained generic ellipsoids.
- C2 green-canister run: `outputs/aigc_assets/object_C_image_to_3d/green_canister/final/coarse/magic123-coarse-sd/green_canister_coarse_gsd50_z5_sparse01_3000steps@20260602-182425/`.
  - Stopped after the `1500` validation review; marker: `training.stopped`.
  - Preserved checkpoints: `500`, `1000`, `1500`, plus `last.ckpt`.
  - Final validation PNGs for selected checkpoint: `save/it1500-0.png` through `save/it1500-3.png`.
  - Selected video: `save/it1500-test.mp4`.
  - Evidence sheets: `outputs/previews/aigc_assets/object_C_image_to_3d/green_canister/object_C_green_canister_it500_views.jpg`, `outputs/previews/aigc_assets/object_C_image_to_3d/green_canister/object_C_green_canister_it1000_views.jpg`, `outputs/previews/aigc_assets/object_C_image_to_3d/green_canister/object_C_green_canister_it1500_views.jpg`.
  - Visual result: current best C candidate but imperfect. It stays a single object with no handle, spout, or detached second object, but the side views remain wedge-like rather than a clean cylinder. The issue did not improve from `500` to `1500`, so the run was not extended blindly to `3000`.
- Current Object C selection: C2 green-canister, step `1500`.
  - Acceptance status: usable as a simple canister fallback, with the side-view wedge artifact documented as a single-image inference failure.

## 2026-06-03 cat-box seed sweep

- Additional `cat_box` runs used seeds `7` and `42`, because the earlier C1 screening only covered seed `0` and was stopped at `1000` steps.
- Shared settings:

```text
Magic123 coarse
sd_guidance_scale: 50
zero123_guidance_scale: 5.0
lambda_sparsity: 0.1
lambda_opaque: 0.001
max steps: 1500
validation/checkpoint interval: 500
validation views per checkpoint: 4
GPU policy: CUDA 6 only
prompt: a light green cat-shaped container with a rounded body, two small ears, a simple cat face, smooth matte surface, single solid object
negative prompt: deformed face, extra ears, extra parts, asymmetry, broken lid, detached pieces, floaters, noisy geometry
```

- Seed `7` run: `outputs/aigc_assets/object_C_image_to_3d/cat_box/final/coarse/magic123-coarse-sd/cat_box_coarse_seed7_gsd50_z5_sparse01_1500steps@20260603-0635/`.
  - Marker: `training.done`.
  - Preserved checkpoints: `500`, `1000`, `1500`, plus `last.ckpt`.
  - Final video: `save/it1500-test.mp4`.
  - Evidence sheets: `outputs/previews/aigc_assets/object_C_image_to_3d/cat_box/object_C_cat_box_seed7_it500_views.jpg`, `outputs/previews/aigc_assets/object_C_image_to_3d/cat_box/object_C_cat_box_seed7_it1000_views.jpg`, `outputs/previews/aigc_assets/object_C_image_to_3d/cat_box/object_C_cat_box_seed7_it1500_views.jpg`.
  - Visual result: better than seed `0` because the single-object shape is stable and ear-like protrusions remain visible in multiple views, but the face is still missing and side/back views remain generic rounded container silhouettes.
- Seed `42` run: `outputs/aigc_assets/object_C_image_to_3d/cat_box/final/coarse/magic123-coarse-sd/cat_box_coarse_seed42_gsd50_z5_sparse01_1500steps@20260603-0648/`.
  - Marker: `training.done`.
  - Preserved checkpoints: `500`, `1000`, `1500`, plus `last.ckpt`.
  - Final video: `save/it1500-test.mp4`.
  - Evidence sheets: `outputs/previews/aigc_assets/object_C_image_to_3d/cat_box/object_C_cat_box_seed42_it500_views.jpg`, `outputs/previews/aigc_assets/object_C_image_to_3d/cat_box/object_C_cat_box_seed42_it1000_views.jpg`, `outputs/previews/aigc_assets/object_C_image_to_3d/cat_box/object_C_cat_box_seed42_it1500_views.jpg`.
  - Visual result: very close to seed `7`; ear silhouette is slightly clearer in some views, but the cat face still does not appear and side/back geometry is still a rounded canister-like volume.
- Cat-box decision after the seed sweep: seed `42` at `1500` is the best `cat_box` backup if a cat-shaped attempt is desired, but it does not clearly supersede the selected `green_canister` step `1500` as the safer Object C candidate. The core failure remains semantic detail loss from the single image: Magic123 preserves a rough body/ear silhouette but does not recover the face.

## 2026-05-31 apple smoke-test input

- Raw input path: `data/raw/object_C_apple_smoketest/C_apple.jpeg`
- Resolved storage path: `/data/dechao/cv_final_pj/data/raw/object_C_apple_smoketest/C_apple.jpeg`
- File type: JPEG, 5160x3440, RGB
- Visual inspection: single red apple on a mostly white background, object centered horizontally in the lower-middle area, with visible stem and reflection/shadow region near the bottom.
- Source note: provided by the user as a provisional Object C apple smoke-test input; the user stated it was found online and photographed by someone else.
- Superseded: this apple path was replaced by `object_C_green_container`; do not run apple smoke tests or promote the apple image to final Object C unless explicitly requested.
