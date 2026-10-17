# AIGC Asset Pipeline Notes

## 2026-05-31 storage preparation

Large mutable project paths were moved to `/home/dechao/data/cv_final_pj/`, which resolves through `/home/dechao/data -> /data/dechao`.

Migrated paths now kept as project-local symlinks:

```text
data/raw
data/processed
data/interim
outputs/reconstruction_2dgs
outputs/checkpoints
outputs/renders
outputs/previews
outputs/eval
outputs/aigc_assets
third_party/2d-gaussian-splatting
third_party/threestudio
```

`outputs/aigc_assets` and `third_party/threestudio` did not exist before this migration; their target directories were created under `/home/dechao/data/cv_final_pj/`.

Before running Step 3 AIGC commands, source:

```bash
source scripts/aigc/env_step3.sh
```

This keeps Hugging Face, Torch, pip, conda package, XDG, and Torch extension caches under `data/interim/cache/`. External pretrained weights should go under `data/raw/pretrained_models/`; do not create top-level `cache/` or `weights/`.

Step 3 skeleton paths now exist:

```text
data/raw/pretrained_models/
data/raw/object_C_apple_smoketest/
data/interim/cache/
data/interim/object_C_apple_smoketest/
data/processed/object_C_apple_smoketest/
data/raw/object_C_green_container/
data/interim/object_C_green_container/
data/processed/object_C_green_container/
```

Accepted Step 2 paths were verified after the symlink migration:

```text
outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/
outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/
data/processed/object_A_book/
data/raw/background_mipnerf360/counter/
```

The original source directories were renamed with `.bak_20260531T063645Z` and left in place for user confirmation before deletion.

## 2026-05-31 Step 3 planning decisions

- AIGC dependencies will use a new isolated conda environment, `cv-final-aigc`, preferably created at `/data/dechao/cv_final_pj/conda_envs/cv-final-aigc`.
- Existing `cv-final-recon` and `cv-final-2dgs` environments may be inspected for CUDA/PyTorch reference only; do not install threestudio or Magic123 dependencies into `cv-final-2dgs`.
- GPU selection policy: prefer the highest-numbered available CUDA device, currently CUDA 6; if occupied, try CUDA 5 and continue downward. Check `nvidia-smi` before every GPU run.
- Background-removal model downloads, including `rembg`, Segment Anything, or similar tools, must use `data/interim/cache/` through `scripts/aigc/env_step3.sh`; record the resulting cache/model size.
- User-provided Object C apple smoke-test input was moved from the project root to `data/raw/object_C_apple_smoketest/C_apple.jpeg`.
- Object C apple input remains provisional for smoke testing; the user stated it was found online and photographed by someone else, so it should not be treated as final Object C unless source/license information is documented and the user explicitly accepts that choice.
- The earlier idea of replacing the apple with a user-captured simple opaque object was resolved by the new green container input.
- Object B is locked as the wooden duck prompt from `plan/pj_step3_plan.md`.
- Object C was changed to a formal green container input. The root file `罐子.jpg` was visually inspected and moved to `data/raw/object_C_green_container/container_raw.jpg`; the user confirmed it is self-captured. Use this image and the locked prompt `a simple dark green cylindrical container with a flat lid, smooth glossy surface` for both smoke testing and formal Object C training, and do not run the apple path unless explicitly requested.

## 2026-05-31 environment recovery and validation

Resume checks:

- No residual `pip install`, `conda install`, `git clone`, CUDA extension build, or `launch.py` processes were running.
- Step 3 symlinks resolved to `/data/dechao/cv_final_pj/...`.
- `/data` had about 2.4T free.
- `third_party/threestudio` commit: `28d9d80d9d00f308244adfcf3be8b17ca0cb6465`.

AIGC environment:

```text
env path: /data/dechao/cv_final_pj/conda_envs/cv-final-aigc
python: 3.10.20
torch: 2.0.0+cu118
torch cuda: 11.8
torch CUDA available on CUDA_VISIBLE_DEVICES=6: true
torchvision: 0.15.1+cu118
numpy: 1.26.4
opencv-python/headless: 4.9.0.80
```

Dependency recovery details:

- Full upstream `third_party/threestudio/requirements.txt` initially failed on intermittent GitHub TLS clone errors, missing `pkg_resources`, CUDA 11.8 rejecting system GCC 13.3, missing CUDA Thrust include paths, missing `libcuda` link paths, and missing `pybind11`.
- `setuptools` was pinned to `69.5.1` so PyTorch CUDA extension builds can import `pkg_resources`.
- CUDA extension builds now use existing `cv-final-2dgs` tooling only as a compiler/toolkit source: CUDA 11.8 nvcc, conda-forge GCC/G++ `11.4`, Thrust include path under `targets/x86_64-linux/include`, and `libcuda` stub/driver link paths.
- Persistent Git source clones were kept under `data/interim/cache/git_sources/` for unstable GitHub sources:
  - `tiny-cuda-nn`: `749dd70c5afc5a9dadb85e5652ed65d55e0ba187`
  - `nvdiffrast`: `253ac4fcea7de5f396371124af597e6cc957bfae`
  - `envlight`: `05b5851e854429d72ecaf5b206ed64ce55fae677`
  - `CLIP`: `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`

Installed key packages:

```text
nerfacc: 0.5.2
tinycudann: 2.0
nvdiffrast: 0.4.0
pysdf: 0.1.9
envlight: 0.1.0
clip: 1.0
xformers: 0.0.19
diffusers: 0.19.3
transformers: 4.28.1
accelerate: 0.23.0
huggingface_hub: 0.19.4
pytorch-lightning: 2.3.3
libigl: 2.6.2
```

Repository changes made for reproducibility:

- `scripts/aigc/env_step3.sh` now exports the verified AIGC env path, project cache paths, CUDA_HOME, GCC/G++ 11.4 compilers, CUDA include path, `libcuda` link paths, `MAX_JOBS=4`, and Git HTTP/1.1 settings.
- `configs/aigc_assets/threestudio_requirements_step3_no_git.txt` records the non-Git subset of upstream requirements; Git/CUDA packages are installed separately from pinned source clones or direct source installs.
- `configs/aigc_assets/threestudio_constraints.txt` now protects torch, torchvision, xformers, numpy, OpenCV, accelerate, and huggingface_hub versions.
- `third_party/threestudio/threestudio/utils/ops.py` has a small `libigl 2.6.2` compatibility shim for `fast_winding_number_for_meshes` and `read_obj`.

Validation passed:

- `bash -n` passed for Step 3 shell scripts.
- `py_compile` passed for Step 3 Python scripts.
- `pip check` reported no broken requirements.
- Core imports passed: `torch`, `torchvision`, `numpy`, `cv2`, `nerfacc`, `tinycudann`, `nvdiffrast.torch`, `pysdf`, `envlight`, `clip`.
- ML-stack imports passed: `diffusers`, `transformers`, `accelerate`, `pytorch_lightning`.
- `threestudio` and `threestudio.systems.base.BaseSystem` imports passed.
- `scripts/aigc/run_threestudio_help.sh` passed.

Known limitation:

- Do not use `launch.py --gradio`. The top-level `lightning` / `gradio` import path has a FastAPI/Pydantic compatibility issue. The planned non-gradio smoke-test path uses `pytorch_lightning` and imports successfully.

## 2026-05-31 smoke-test execution

User approved running Object B first and Object C second. Both smoke tests used CUDA device 6.

Model cache decisions:

- The first Object B launch inherited `HF_ENDPOINT=https://hf-mirror.com` and failed before training because that endpoint could not resolve `stabilityai/stable-diffusion-2-1-base`.
- To avoid endpoint-dependent smoke tests, copied `runwayml/stable-diffusion-v1-5` into the project Hugging Face cache at `data/interim/cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/` and exported `AIGC_SD15_MODEL` from `scripts/aigc/env_step3.sh`.
- Object C initially failed while downloading Zero123 weights from `cas-bridge.xethub.hf.co` with a TLS EOF. Required Zero123 fp16 files were then downloaded into the project cache and `AIGC_ZERO123_MODEL` now points to the local snapshot `data/interim/cache/huggingface/hub/models--bennyguo--zero123-diffusers/snapshots/b5289c24d8549e3a4737d0c34ab1347e5f074fbe/`.
- `scripts/aigc/run_object_B_smoke.sh` and `scripts/aigc/run_object_C_magic123_coarse_smoke.sh` now pass local model paths explicitly, so repeat smoke runs should not need remote SD/Zero123 downloads.

Smoke results:

- Object B completed `25` steps at `outputs/aigc_assets/object_B_text_to_3d/smoke_test/dreamfusion-sd/a_small_matte_wooden_duck_toy,_hand-painted_yellow_body_and_orange_beak,_simple_smooth_geometry,_realistic_object,_clean_shape,_non-transparent,_no_fur,_no_thin_structures@20260531-104828/`.
- Object C completed `25` steps at `outputs/aigc_assets/object_C_image_to_3d/green_container/smoke_test/coarse/magic123-coarse-sd/container_rgba.png-a_simple_dark_green_cylindrical_container_with_a_flat_lid,_smooth_glossy_surface@20260531-105653/`.
- Both runs produced checkpoints, config snapshots, metrics, validation PNGs, `120` test PNGs, and `it25-test.mp4`.
- The `25`-step visual outputs are expectedly blob-like. They validate the Step 3 runtime path and output writing, not final asset quality.

Next step:

- Do not start formal long training without explicit user approval.
- For formal Object B/Object C runs, reuse the existing local SD/Zero123 cache paths and decide formal iteration counts/output names before launching.

## 2026-05-31 realtime train logging

Training log behavior was added for future AIGC runs:

- `third_party/threestudio/threestudio/utils/callbacks.py` now has `TrainMetricsPrinterCallback`.
- `third_party/threestudio/launch.py` enables that callback when `AIGC_TRAIN_LOG_INTERVAL` or `THREESTUDIO_TRAIN_LOG_INTERVAL` is positive.
- `scripts/aigc/env_step3.sh` defaults `AIGC_TRAIN_LOG_INTERVAL=50` and sets `PYTHONUNBUFFERED=1`.
- A metric line is printed every `AIGC_TRAIN_LOG_INTERVAL` steps with prefix `[TRAIN_METRICS]`; it includes scalar `train/loss*`, `train/grad_norm*`, and `lr-*` values available from PyTorch Lightning.
- AIGC is not a classification task, so there is no accuracy metric. Use losses, previews, rendered videos, and exported mesh inspection for acceptance.
- Smoke scripts and formal scripts now pipe stdout/stderr through `tee -a <trial_dir>/train.log`, so `tail -f <trial_dir>/train.log` shows progress in realtime.

Formal training entrypoints with realtime logs:

```text
scripts/aigc/run_object_B_formal.sh
scripts/aigc/run_object_C_magic123_coarse_formal.sh
```

Default formal settings:

```text
Object B max steps: 3000
Object C coarse max steps: 10000
Object B validation interval: 1000
Object C validation interval: 1000
checkpoint interval: 1000
metric print interval: 50
```

Override examples:

```bash
AIGC_TRAIN_LOG_INTERVAL=25 AIGC_OBJECT_B_MAX_STEPS=3000 AIGC_OBJECT_B_SEED=7 bash scripts/aigc/run_object_B_formal.sh
AIGC_TRAIN_LOG_INTERVAL=25 AIGC_OBJECT_C_COARSE_MAX_STEPS=10000 bash scripts/aigc/run_object_C_magic123_coarse_formal.sh
```

## 2026-05-31 Object B rubber-duck retry preparation

- The first Object B formal wooden-duck attempt completed normally at `10000` steps and is preserved under `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/wooden_duck_formal_10000steps@20260531-144954/`; do not overwrite it.
- That attempt is visually rejected and should be kept as report evidence of an AIGC failure mode.
- Added `configs/aigc_assets/object_B_rubber_duck.yaml` for the next prompt attempt.
- Updated `scripts/aigc/run_object_B_formal.sh` to default to the new yellow rubber-duck prompt, pass a negative prompt, write new runs under a `rubber_duck_formal_<steps>steps@...` trial tag, and validate every `1000` steps by default.
- Added `scripts/aigc/render_object_B_checkpoint_videos.sh`; Object B formal runs now launch it by default so intermediate checkpoint videos are rendered every `1000` steps while training continues.

## 2026-05-31 Object B rubber-duck drift stop and next retry policy

- The rubber-duck `10000`-step retry was stopped after previews drifted badly; preserve it at `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_formal_10000steps@20260531-171926/`.
- The stopped run has checkpoints, validation images, and videos through `7000` steps. Treat it as a rejected failure case for the report, not an accepted completion.
- Observed failure mode: early `1000`/`2000` previews were closer to a simple duck, but later steps increasingly converted the beak into a red/orange horizontal band.
- Object B future-run policy is now conservative: default `3000` max steps, validate/checkpoint/render every `1000`, preserve every trial directory, and inspect intermediate videos instead of trusting the final checkpoint.
- The default prompt was simplified to `a simple yellow rubber duck bath toy, smooth rounded body, small orange beak, black dot eyes, glossy plastic`.
- The default negative prompt now explicitly blocks the observed banding failure: `red stripe, orange band, ring around body, scarf, oversized beak, wide mouth, two beaks, extra parts, extra limbs, feathers, realistic bird, noisy texture, dirty surface, floaters, artifacts`.
- `scripts/aigc/run_object_B_formal.sh` exposes `AIGC_OBJECT_B_SEED`, `AIGC_OBJECT_B_GUIDANCE_SCALE`, `AIGC_OBJECT_B_MIN_STEP_PERCENT`, and `AIGC_OBJECT_B_MAX_STEP_PERCENT`; defaults are seed `0`, guidance `70`, min step percent `0.05`, and max step percent `0.80`.
- Object B and Object C AIGC run scripts now refuse to reuse a non-empty trial directory, so future failed cases are not overwritten accidentally.
- Object B formal and Object C formal scripts now write `training.done`, `training.failed`, or `training.stopped` markers so interrupted training is easier to distinguish from accepted completion.
- For the Object B short seed sweep, use only CUDA `6`. Object B formal training now defaults to no live checkpoint-video monitor; render checkpoint videos afterward on CUDA `6` before starting the next seed.

## 2026-05-31 Object B rubber-duck short seed sweep

- Ran a strict single-GPU Object B seed sweep on CUDA `6` only, using the simplified prompt, negative prompt, `3000` max steps, guidance `70`, timestep range `0.05`-`0.80`, and `1000`-step checkpoint/validation cadence.
- Seed `0`: completed `3000` steps at `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_simple_seed0_g70_3000steps@20260531-181724/`; generated `1000`, `2000`, and `3000` checkpoint videos after training on CUDA `6`; rejected because the result is pink/purple and not a clear yellow rubber duck.
- Seed `7`: completed `3000` steps at `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_simple_seed7_g70_3000steps@20260531-184143/`; generated `1000`, `2000`, and `3000` checkpoint videos after training on CUDA `6`; best checkpoint is `2000`, but it is only a weak candidate.
- Seed `42`: stopped after the `1000` checkpoint at `outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_simple_seed42_g70_3000steps@20260531-190238/` because it already showed a split/two-object duck-like shape; generated the `1000` checkpoint video on CUDA `6`; rejected.
- Combined preview contact sheet: `outputs/aigc_assets/object_B_text_to_3d/final/object_B_rubber_duck_seed_sweep_preview_contact_sheet.jpg`.
- Overall status: no robust final Object B rubber-duck candidate from this sweep. Seed `7` step `2000` is the best preserved weak fallback, but further blind training with the same setup is not recommended.

## 2026-06-01 Object C formal preview setup

- Object B rubber-duck runs are skipped for now after visual rejection; proceed with Object C green-container Magic123 coarse training.
- Object C formal coarse remains a `10000`-step run.
- Updated `scripts/aigc/run_object_C_magic123_coarse_formal.sh` to align validation and checkpoint cadence at every `1000` steps.
- Added `scripts/aigc/render_object_C_checkpoint_videos.sh`. The Object C formal script now defaults to rendering checkpoint orbit videos after training, on the same CUDA device, for each `1000`-step checkpoint.
- Expected Object C formal artifacts: realtime `train.log` with `[TRAIN_METRICS]`, validation PNGs every `1000`, checkpoints every `1000`, and `it<N>-test.mp4` videos for `N=1000,2000,...,10000`.

## 2026-06-01 Object C canister retry policy

- The first Object C formal coarse run `green_container_coarse_10000steps@20260601-050016` was stopped and preserved after a side-handle/side-bulge hallucination appeared in validation views.
- User confirmed a moderate prompt is acceptable for Object C single-image-to-3D as long as the configuration is recorded for the report.
- Object C retry prompt: `a plain dark green cylindrical canister with a flat lid, smooth glossy surface, symmetric shape`.
- Object C retry negative prompt: `handle, mug, cup, side bulge, side protrusion, spout, extra part, asymmetric shape, deformed cylinder`.
- New Object C retry policy: `3000` max steps, validation/checkpoint every `1000`, post-training checkpoint videos every `1000`, strict CUDA `6` only, and a fresh timestamped output directory for every run.

## 2026-06-11 Object C green-yoyo diagnostic branch

- This branch was exploratory and is not recommended as the next mainline Object C asset.
- Source image was archived as `data/raw/object_C_green_yoyo/yoyo_source.jpg`.
- v1 preprocessing used `scripts/aigc/prepare_object_c_green_yoyo_rgba.py` to create `data/processed/object_C_green_yoyo/yoyo_body_stringless_rgba.png`. The rope/string was removed by selecting the largest saturated-green component and writing a single RGBA cutout.
- v1 runs:
  - `outputs/aigc_assets/object_C_image_to_3d/green_yoyo/final/coarse/magic123-coarse-sd/green_yoyo_body_seed42_gsd40_z5_sparse01_1500review@20260611-063540/`
  - `outputs/aigc_assets/object_C_image_to_3d/green_yoyo/final/coarse/magic123-coarse-sd/green_yoyo_body_seed42_gsd40_z5_sparse01_resume1500_to3000@20260611-065100/`
  - `outputs/aigc_assets/object_C_image_to_3d/green_yoyo/final/coarse/magic123-coarse-sd/green_yoyo_body_seed42_gsd40_z5_sparse01_resume3000_to4000@20260611-070430/`
- v1 settings were seed `42`, SD guidance `40`, Zero123 guidance `5.0`, `lambda_sparsity=0.1`, and `lambda_opaque=0.001`.
- v1 result: the yoyo was recognizable from some views, but the central groove was structurally wrong. It appeared as side/top-bottom bands or misplaced grooves, not as one real central waist groove. The 3000-to-4000 extension was stopped and preserved because continued coarse training was reinforcing the wrong geometry.
- v2 preprocessing used `scripts/aigc/prepare_object_c_green_yoyo_geometry_guide.py` to create `data/processed/object_C_green_yoyo/yoyo_body_central_waist_guide_rgba.png`, `data/interim/object_C_green_yoyo/yoyo_body_central_waist_guide_mask.png`, and `outputs/previews/aigc_assets/object_C_image_to_3d/green_yoyo/object_C_green_yoyo_body_central_waist_guide_preview.png`.
- v2 guide was source-green but geometry-biased: it used the original image for palette/object concept, not as a pure photo crop. The goal was to force one central waist groove and remove the rope at the RGBA stage.
- v2 run path: `outputs/aigc_assets/object_C_image_to_3d/green_yoyo/final/coarse/magic123-coarse-sd/green_yoyo_body_v2_waistguide_seed42_gsd30_z75_1500review@20260611-071020/`.
- v2 command summary: seed `42`, prompt `a bright green plastic yoyo body, side-view spool-shaped toy, two rounded disks joined by one narrow central waist groove, glossy smooth surface, single centered object`, negative prompt `extra grooves, multiple grooves, grooves on outer faces, top stripe, bottom stripe, side stripe, handle, side protrusion, detached part, extra part, floaters, white rim, text, asymmetric bulge, deformed geometry`, SD guidance `30`, Zero123 guidance `7.5`, `lambda_sparsity=0.1`, `lambda_opaque=0.001`, `1500` planned max steps, `500` validation/checkpoint interval, and `8` validation views.

## 2026-06-12 Object C Rubik's cube try10-try12

- Keqs try9 was demoted as a final-candidate source because its preprocessing incorrectly treated white stickers as missing cubies. Corrected white-sticker RGBA variants were generated, but the mainline moved back to the older `rubiks_cube_rgba.png` source after user review favored try4/try8.
- Ran three parallel Magic123 coarse trials on CUDA `0/1/2`, all using the older source image `data/processed/object_C_rubiks_cube/rubiks_cube_rgba.png`, `1000` max steps, validation/checkpoint every `200`, and `8` validation views.
- `try10`: `outputs/aigc_assets/object_C_image_to_3d/rubiks_cube/final/coarse/magic123-coarse-sd/rubiks_cube_try10_try8_seed0_sameparams_1000@20260612-201907/`
  - Seed `0`, same main recipe as try8: SD guidance `30`, Zero123 guidance `10`, `lambda_sparsity=0.015`, `lambda_opaque=0.002`, `lambda_normal_smoothness_2d=1200`, `lambda_z_variance=50`, elevation `15`, azimuth `45`, fovy `30`.
  - It reached step `1000`, wrote `ckpts/epoch=0-step=1000.ckpt`, `save/it1000-0.png` through `save/it1000-7.png`, and `save/it1000-test.mp4`, but the shell marker is `training.stopped` with code `143` because the process received TERM during/after the final test-save phase.
  - Visual result: recognizable Rubik's-like texture and a mostly closed mask, but not a clear improvement over try8; color balance still drifts strongly green/blue in some views.
- `try11`: `outputs/aigc_assets/object_C_image_to_3d/rubiks_cube/final/coarse/magic123-coarse-sd/rubiks_cube_try11_try8_seed7_sameparams_1000@20260612-201907/`
  - Seed `7`, otherwise same recipe as try8.
  - Completed with `training.done`; wrote `ckpts/epoch=0-step=1000.ckpt`, `save/it1000-0.png` through `save/it1000-7.png`, and `save/it1000-test.mp4`.
  - Visual result: similar to try10/try8, with no major seed breakthrough. Geometry is cube-like but not cleaner enough to replace try8 by default.
- `try12`: `outputs/aigc_assets/object_C_image_to_3d/rubiks_cube/final/coarse/magic123-coarse-sd/rubiks_cube_try12_try8_seed42_closedgeom_fovy25_1000@20260612-201908/`
  - Seed `42`, combined geometry-closure changes: fovy `25`, `lambda_sparsity=0.008`, `lambda_opaque=0.004`, same SD/Zero123 `30/10`, same smoothness/z variance `1200/50`, and prompt/negative prompt strengthened around closed solid cube geometry.
  - Completed with `training.done`; wrote `ckpts/epoch=0-step=1000.ckpt`, `save/it1000-0.png` through `save/it1000-7.png`, and `save/it1000-test.mp4`.
  - Visual result: slightly better closed-mask/side solidity than pure seed changes, but not a dramatic quality jump. It is the best of try10-try12 on geometry, while try4/try8 remain important visual baselines.
- Current interpretation: changing seed alone did not solve the Object C Rubik's defects. The combined closure recipe in try12 has modest positive signal and is more promising than a broad seed sweep. Do not resume try10. Compare try12 directly against try4/try8 before deciding whether to run one more targeted branch.
- v2 status: stopped by user request. `training.stopped` contains `143`. The run generated `ckpts/epoch=0-step=500.ckpt`, `ckpts/last.ckpt`, and `save/it500-0.png` through `save/it500-7.png`; training log indicates it reached about step `917` before termination.
- v2 diagnosis: `it500` RGB views showed a central dark-line illusion, but opacity/mask stayed close to a rounded capsule. This means the model learned shading/normal texture for the groove rather than real waist geometry. The guide's RGB dark line was too strong and its alpha/mask waist pinch was too weak.
- Prompt note: do not add `no string`; the rope must be removed by RGBA preprocessing. Do not add `ring` to the negative prompt because the real yoyo has circular/ring-like edges.
- Recommendation for new server: do not continue or refine yoyo v1/v2. If Object C is retried, prefer a simpler opaque object with a clean silhouette. If yoyo must be retried, start a v3 coarse from scratch, make the central waist visible in alpha/mask rather than only RGB, remove `side-view` from the base prompt to avoid view-prompt contradiction, inspect opacity at 500 steps, and use CUDA `6`.
