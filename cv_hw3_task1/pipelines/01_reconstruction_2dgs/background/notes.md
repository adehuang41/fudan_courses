# Background Notes

## Current Background Target

- Step: `2.8`
- Scene id: `background_counter`
- Dataset: Mip-NeRF 360
- Scene: `counter`
- Role: real host scene for later asset fusion

The `counter` scene is the preferred background target because it is an indoor tabletop/counter scene and should be easier to align with small inserted objects than large outdoor scenes.

## Paths

Planned local paths:

```text
data/raw/background_mipnerf360/counter/
data/processed/background_counter/
outputs/reconstruction_2dgs/background_counter/
```

Created staging directories:

```text
data/raw/background_mipnerf360/counter/
data/processed/background_counter/metadata/
outputs/reconstruction_2dgs/background_counter/
pipelines/01_reconstruction_2dgs/background/logs/
```

At Step 2.8 entry, no Mip-NeRF 360 data was present locally. The `counter` scene was then selectively extracted as recorded below.

## Dataset Source Check

Official project page:

```text
https://jonbarron.info/mipnerf360/
```

The page links two Google Storage dataset packages:

```text
http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
http://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
```

HTTP HEAD checks on 2026-05-30:

- `360_v2.zip`: `12535427936` bytes, about 11.67 GiB
- `360_extra_scenes.zip`: `4488140217` bytes, about 4.17 GiB

Initial local disk check:

- root filesystem available space: about 9.4 GiB
- status: insufficient for safely downloading and extracting the full main dataset package

HTTP Range central-directory checks confirmed:

- `counter` is inside `360_v2.zip`
- `counter` has 964 files
- `counter/images/` has 240 images
- `counter` has 5 sparse-related archive entries, including sparse directories and the `sparse/0` bin files
- compressed bytes: `1212076580`, about 1.13 GiB
- uncompressed bytes: `1241388712`, about 1.16 GiB

Because the selective `counter/` extract fits current disk, the project extracted only the `counter` scene and did not store the full 11.67 GiB zip.

Extraction command:

```bash
python3 scripts/reconstruction/extract_mipnerf360_scene_http.py \
  --url http://storage.googleapis.com/gresearch/refraw360/360_v2.zip \
  --scene counter \
  --output_dir data/raw/background_mipnerf360 \
  --summary_path data/processed/background_counter/metadata/dataset_extraction_summary.json \
  --progress_every 25
```

Extraction result:

- output scene path: `data/raw/background_mipnerf360/counter/`
- scene size on disk: about 1.2 GiB
- `images/`: 240 files
- `sparse/0/`: `cameras.bin`, `images.bin`, `points3D.bin`
- extraction summary: `data/processed/background_counter/metadata/dataset_extraction_summary.json`
- remaining root filesystem space after extract and smoke: about 8.5 GiB

Do not download the full Mip-NeRF 360 package on this filesystem unless space is freed or a larger storage path is chosen.

## 2DGS Command Source

The 2DGS README says Mip-NeRF 360 scenes can be trained directly with:

```bash
python train.py -s <path to m360>/<scene> -m <output path>
```

For this project the intended paths are:

```bash
python train.py \
  -s data/raw/background_mipnerf360/counter \
  -m outputs/reconstruction_2dgs/background_counter/<run_name>
```

Before any actual smoke test or formal run, re-read:

```bash
third_party/2d-gaussian-splatting/README.md
python train.py --help
python render.py --help
```

Do not guess changed arguments.

## Planned Validation Sequence

1. Verify enough disk space or choose a larger data location. Done for selective `counter/` extract.
2. Confirm which official package contains `counter`. Done: `360_v2.zip`.
3. Download or extract only the `counter` scene if possible. Done via HTTP Range extraction.
4. Verify the scene contains the 2DGS-compatible COLMAP-style structure, especially:

```text
images/
sparse/0/
```

Done.

5. Run a low-cost smoke test first, not formal training. Done.
6. Use `nvidia-smi` before any CUDA run and select only one idle GPU with `CUDA_VISIBLE_DEVICES`.
7. Keep smoke test short, at most 200 iterations, only verifying data loading, CUDA extension execution, and output generation.
8. Render train-view or sample views only after smoke/formal output exists.

## Step 2.8 Background 2DGS Smoke Test

Before the smoke test, the 2DGS README Mip-NeRF example and `python train.py --help` were read again.

GPU selection:

- `nvidia-smi` was run on 2026-05-30.
- GPU 4 was idle at 0 MiB before the smoke test.
- The smoke test used `CUDA_VISIBLE_DEVICES=4`.
- GPU 4 returned to 0 MiB after the smoke test.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=4 CV_FINAL_2DGS_LOG_INTERVAL=50 conda run -n cv-final-2dgs python train.py \
  -s /home/dechao/cv_final_pj/data/raw/background_mipnerf360/counter \
  -m /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/background_counter/2dgs_smoke_test \
  --iterations 200 \
  -r 4 \
  --save_iterations 200
```

Smoke output:

```text
outputs/reconstruction_2dgs/background_counter/2dgs_smoke_test/
```

Generated files:

- `cameras.json`
- `cfg_args`
- `input.ply`
- `point_cloud/iteration_200/point_cloud.ply`
- `train_smoke.log`

Smoke result:

- cameras read: 240/240
- initial points: 155767
- iterations: 200/200
- final smoke loss: `0.10645`
- final smoke points: 155767
- result: background 2DGS can read the Mip-NeRF 360 `counter` COLMAP data, run the CUDA extension, and write output files

Logging note:

- The first shell command used `tee` before the model directory existed, so the shell returned code 1 even though the 2DGS training process completed.
- A concise `train_smoke.log` was written afterward from the captured command output and records the key smoke lines.
- For future runs, create the output directory before piping to `tee`.

## Current Decision

Step 2.8 has been entered, the `counter` scene has been selectively extracted, the background 2DGS smoke test passed, and the first formal background 2DGS run completed. No AIGC, Blender, or report writing has been started.

## Step 2.8 Background 2DGS First Formal Run

Input:

```text
data/raw/background_mipnerf360/counter/
```

Output:

```text
outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/
```

Run policy:

- Do not re-download Mip-NeRF 360.
- Do not re-organize `counter` data.
- Do not overwrite smoke test output.
- Use GPU 6 only through `CUDA_VISIBLE_DEVICES=6`.
- Run `nvidia-smi` before training.
- Create the output directory before `tee` writes `train.log`.

GPU check:

- `nvidia-smi` was run before training on 2026-05-30.
- GPU 6 was idle at 0 MiB and 0% utilization.
- GPU 6 was released after training; post-run `nvidia-smi` showed 0 MiB on GPU 6 and no remaining `train.py` process.

Logging fix:

- The output directory was created before the formal run:

```bash
mkdir -p outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k
```

- The first start used plain `conda run`, which buffered output and left `train.log` at 0 bytes while training ran.
- That early start was stopped before the formal run proceeded.
- The final formal run used `conda run --no-capture-output`, so `tee` wrote realtime log lines to `train.log`.

Final formal training command:

```bash
CUDA_VISIBLE_DEVICES=6 CV_FINAL_2DGS_LOG_INTERVAL=50 conda run --no-capture-output -n cv-final-2dgs python train.py \
  -s /home/dechao/cv_final_pj/data/raw/background_mipnerf360/counter \
  -m /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k \
  --iterations 30000 \
  -r 4 \
  --save_iterations 7000 30000 \
  2>&1 | tee /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/train.log
```

Timing:

- command start: `2026-05-30T12:17:04+00:00`
- log output folder timestamp: `2026-05-30 12:17:17`
- training complete timestamp: `2026-05-30 12:29:40`
- wall time from command start to completion: about 12 minutes 36 seconds
- training loop time reported by tqdm: about 11 minutes 23 seconds

Result:

- completed: `30000/30000`
- final training progress line:

```text
[ITER 30000] Loss=0.02551 distort=0.00000 normal=0.00476 Points=525815
```

- train eval:

```text
[ITER 30000] Evaluating train: L1 0.016841717809438706 PSNR 29.945239257812503
```

Generated key files:

- `cameras.json`
- `cfg_args`
- `input.ply`
- `train.log`
- `point_cloud/iteration_7000/point_cloud.ply`
- `point_cloud/iteration_30000/point_cloud.ply`

Current recommendation:

- The first formal background `r4_30k` run is complete and technically valid.
- Next step should be background render acceptance: read `render.py --help`, render the `iteration_30000` model, prepare a contact sheet, and perform visual inspection.
- Do not enter AIGC, Blender, or report writing before render acceptance.

## Step 2.8 Background Render Acceptance

Before rendering, the 2DGS README render examples and `python render.py --help` were read again.

Render input:

```text
data/raw/background_mipnerf360/counter/
```

Model:

```text
outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/
```

Render command:

```bash
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n cv-final-2dgs python render.py \
  -s /home/dechao/cv_final_pj/data/raw/background_mipnerf360/counter \
  -m /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k \
  --iteration 30000 \
  -r 4 \
  --skip_test \
  --skip_mesh \
  2>&1 | tee /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/render_train/render.log
```

Official repo output:

```text
outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/train/ours_30000/
```

Organized render output:

```text
outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/render_train/
```

Organized files:

- `renders/`: 240 PNG files
- `gt/`: 240 PNG files
- `vis/`: 240 visualization files
- `render.log`

Contact sheet:

```text
outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/render_contact_sheet.jpg
```

Contact sheet summary:

```text
outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/render_contact_sheet_summary.json
```

Recorded formal result values:

- iterations: `30000`
- resolution: `-r 4`
- final loss: `0.02551`
- final points: `525815`
- final normal: `0.00476`
- final distort: `0.00000`
- train eval L1: `0.016841717809438706`
- train eval PSNR: `29.945239257812503`

Visual judgment from the render contact sheet:

- train-view renders form a coherent orbit through the counter scene
- no obvious black frames, camera failure, or catastrophic holes were visible in the contact sheet
- no obvious large-scale floaters were visible at contact-sheet scale

Recommendation:

- Accept `outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/` as the current formal background `counter` 2DGS result.
- Keep human close-up inspection of `render_train/renders/` versus `render_train/gt/` as the final visual check before Blender fusion.
- Do not enter AIGC, Blender, or report writing in this step.

Step 2 completion status: background `counter` formal reconstruction is accepted; do not rerun COLMAP or 2DGS training unless explicitly requested.
