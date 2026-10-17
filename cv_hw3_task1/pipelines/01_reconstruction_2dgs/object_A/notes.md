# Object A Notes

## Current Object A Instance

- Instance id: `object_A_book`
- Input asset: `data/raw/object_A_book/videos/book_raw.mp4`
- Object: a real book/booklet with rich text and texture
- Capture type: self-captured phone video around the object
- Initial video check: 1080x1920, about 29.997 fps, 6652 frames, about 221.755 seconds

## Step 2.1 Data Archive

The raw video is archived as the canonical Object A input:

```text
data/raw/object_A_book/videos/book_raw.mp4
```

The raw video should not be committed to Git.

## Frame Directory Roles

- `data/processed/object_A_book/images_raw/`: candidate frame pool extracted from the video
- `data/processed/object_A_book/images_smoke/`: 15-20 frames for small-scale COLMAP smoke test, default 18
- `data/processed/object_A_book/images_selected/`: future formal-scale COLMAP input, expected 150-180 selected frames after filtering

Do not treat `images_raw/` as final selected input. Do not treat `images_smoke/` as formal reconstruction input.

## Step 2.2 Recommended Extraction Command

```bash
conda run -n cv-final-recon python scripts/reconstruction/extract_video_frames.py \
  --video data/raw/object_A_book/videos/book_raw.mp4 \
  --output_dir data/processed/object_A_book/images_raw \
  --start_time 0 \
  --end_time 180 \
  --interval_sec 1 \
  --max_frames 180 \
  --overwrite
```

Expected output:

```text
data/processed/object_A_book/images_raw/frame_000001.jpg
...
data/processed/object_A_book/metadata/frame_extraction_summary.json
```

## Step 2.3 Contact Sheet Command

```bash
conda run -n cv-final-recon python scripts/reconstruction/make_frame_contact_sheet.py \
  --image_dir data/processed/object_A_book/images_raw \
  --output outputs/reconstruction_2dgs/object_A_book/frame_contact_sheet.jpg \
  --overwrite
```

Manual checks on the contact sheet:

- blurred frames
- duplicate viewpoints
- occluded frames
- extreme viewpoints
- exposure failures or strong reflections
- object too small in frame

## Step 2.4 Smoke Frame Selection Command

```bash
conda run -n cv-final-recon python scripts/reconstruction/select_smoke_frames.py \
  --source_dir data/processed/object_A_book/images_raw \
  --output_dir data/processed/object_A_book/images_smoke \
  --count 18 \
  --overwrite
```

## Step 2.4 COLMAP Small-Scale Smoke Test Command

```bash
conda run -n cv-final-recon python scripts/reconstruction/run_colmap_object_smoke.py \
  --image_dir data/processed/object_A_book/images_smoke \
  --output_dir outputs/reconstruction_2dgs/object_A_book/colmap_smoke \
  --single_camera \
  --matcher exhaustive \
  --overwrite
```

`run_colmap_object_smoke.py` refuses to clear a non-empty output directory unless `--overwrite` or `--clean_output` is explicitly provided.

## Step 2.4 Pass Criteria

This smoke test only decides whether to enter formal-scale COLMAP, not whether to enter 2DGS.

Numeric guide:

- registered images >= 70%: can enter formal-scale COLMAP if manual checks pass
- registered images >= 80%: ideal
- registered images < 50%: return to frame selection

Required manual checks:

- camera trajectory roughly surrounds the object
- sparse points mainly cover the book/tabletop near the book
- no obviously flying cameras
- no severe wrong matches
- not only reconstructing the background

## Current Run Result

Step 2.2-2.4 were run in `cv-final-recon`.

Frame extraction:

- `images_raw/`: 180 candidate frames
- summary: `data/processed/object_A_book/metadata/frame_extraction_summary.json`

Contact sheet:

- output: `outputs/reconstruction_2dgs/object_A_book/frame_contact_sheet.jpg`
- dimensions: 1080 x 8040
- summary: `data/processed/object_A_book/metadata/frame_contact_sheet_summary.json`

Smoke subset:

- `images_smoke/`: 18 frames selected by uniform sampling from `images_raw/`
- selected frames: `frame_000001.jpg`, `frame_000012.jpg`, `frame_000022.jpg`, `frame_000033.jpg`, `frame_000043.jpg`, `frame_000054.jpg`, `frame_000064.jpg`, `frame_000075.jpg`, `frame_000085.jpg`, `frame_000096.jpg`, `frame_000106.jpg`, `frame_000117.jpg`, `frame_000127.jpg`, `frame_000138.jpg`, `frame_000148.jpg`, `frame_000159.jpg`, `frame_000169.jpg`, `frame_000180.jpg`
- summary: `data/processed/object_A_book/metadata/smoke_frame_selection_summary.json`

COLMAP smoke test:

- output: `outputs/reconstruction_2dgs/object_A_book/colmap_smoke/`
- input images: 18
- registered images: 13
- registration ratio: 72.22%
- sparse points: 3473
- numeric result: can enter formal-scale COLMAP if manual checks pass
- unregistered frames: `frame_000117.jpg`, `frame_000127.jpg`, `frame_000138.jpg`, `frame_000159.jpg`, `frame_000169.jpg`
- summary: `outputs/reconstruction_2dgs/object_A_book/colmap_smoke/summary.txt`
- converted inspection files: `outputs/reconstruction_2dgs/object_A_book/colmap_smoke/sparse_txt/`, `outputs/reconstruction_2dgs/object_A_book/colmap_smoke/sparse_points.ply`

Current decision:

- Numerically, the smoke test is viable for moving to formal-scale COLMAP.
- Manual checks are still required before Step 2.5/formal-scale COLMAP: inspect the contact sheet, sparse point cloud, and camera trajectory.
- This does not mean the project is ready for 2DGS. `images_selected/` is intentionally still empty.

## Step 2.5 Formal-Scale COLMAP Preparation

The user manually checked `outputs/reconstruction_2dgs/object_A_book/colmap_smoke/sparse_points.ply` and confirmed:

- the smoke-test point cloud is generally reasonable
- there is continuous structure
- there are no obvious large-scale flying points
- the workflow may enter formal-scale COLMAP
- the workflow still must not enter 2DGS yet

Formal input selection:

- source: `data/processed/object_A_book/images_raw/`
- output: `data/processed/object_A_book/images_selected/`
- source frames: 180
- manually rejected frames: 30
- selected frames: 150
- metadata: `data/processed/object_A_book/metadata/selected_frames.json`
- contact sheet: `outputs/reconstruction_2dgs/object_A_book/selected_frame_contact_sheet.jpg`

Formal COLMAP output:

- output: `outputs/reconstruction_2dgs/object_A_book/colmap_full/`
- input images: 150
- registered images: 150
- registration ratio: 100%
- sparse points: 41278
- unregistered images: 0
- summary: `outputs/reconstruction_2dgs/object_A_book/colmap_full/summary.txt`
- inspection files: `outputs/reconstruction_2dgs/object_A_book/colmap_full/sparse_txt/`, `outputs/reconstruction_2dgs/object_A_book/colmap_full/sparse_points.ply`

Current Step 2.5 decision:

- Numeric COLMAP result passes strongly.
- Still do not enter 2DGS until the formal-scale sparse point cloud and camera trajectory are manually inspected and accepted.

## Step 2.5 Camera Trajectory Check Without GUI

`colmap gui` was not used because the shared server has no display available and reports `cannot connect to display`.
COLMAP was not reinstalled.

The existing formal COLMAP TXT model was used:

```text
outputs/reconstruction_2dgs/object_A_book/colmap_full/sparse_txt/
```

Trajectory visualization command:

```bash
conda run -n cv-final-recon python scripts/reconstruction/visualize_colmap_trajectory.py \
  --model_txt_dir outputs/reconstruction_2dgs/object_A_book/colmap_full/sparse_txt \
  --output_dir outputs/reconstruction_2dgs/object_A_book/colmap_full
```

Generated inspection files:

- `outputs/reconstruction_2dgs/object_A_book/colmap_full/camera_trajectory_top.png`
- `outputs/reconstruction_2dgs/object_A_book/colmap_full/camera_trajectory_3d.png`
- `outputs/reconstruction_2dgs/object_A_book/colmap_full/camera_trajectory_summary.json`

Script-based geometry check:

- registered images: 150
- sparse points: 41278
- possible outlier cameras: 0
- geometry check: `pass_no_obvious_camera_outliers`
- the trajectory plots show a continuous camera path around the reconstructed book point cloud

Current Step 2.5 geometry decision:

- Formal COLMAP passes the no-GUI camera trajectory check.
- Do not enter 2DGS in this step. Enter the next stage only after explicit user approval.

## Step 2.6 2DGS Code Preparation and Smoke Test

Entered Step 2.6 after the user accepted the formal COLMAP sparse point cloud and camera trajectory.

2DGS repository:

- repo URL: `https://github.com/hbb1/2d-gaussian-splatting.git`
- local path: `third_party/2d-gaussian-splatting/`
- commit: `335ad612f2e783a4e57b9cbc4d1e167bd599fc98`
- submodules:
  - `e0ed0207b3e0669960cfad70852200a4a5847f61 submodules/diff-surfel-rasterization`
  - `5c46b9c07008ae65cb81ab79cd677ecc1934b903 submodules/diff-surfel-rasterization/third_party/glm`
  - `f155ec04131cb579f53443a06879d37115f4612f submodules/simple-knn`

Environment:

- conda env: `cv-final-2dgs`
- PyTorch: `2.0.0`
- PyTorch CUDA: `11.8`
- `torch.cuda.is_available()`: `True`
- explicit CUDA toolkit used for extensions:

```bash
CUDA_HOME=/home/dechao/.conda/envs/cv-final-2dgs
PATH=/home/dechao/.conda/envs/cv-final-2dgs/bin:$PATH
```

- `nvcc --version`: CUDA compilation tools `11.8`, `V11.8.89`

Environment repair:

- Initial audit found `PIL` and `Open3D` failed with `ImportError: libtiff.so.5: cannot open shared object file`.
- `conda install -y -n cv-final-2dgs -c conda-forge pillow` changed `pillow` to `10.4.0`, but did not fix the ABI mismatch.
- `conda install -y -n cv-final-2dgs 'libtiff=4.4.0'` installed a compatible `libtiff.so.5` inside the conda environment and fixed both imports.
- Final import checks passed:
  - `from PIL import Image`
  - `import open3d as o3d`
  - `import diff_surfel_rasterization`
  - `import simple_knn._C`

Third-party source patch:

- `third_party/2d-gaussian-splatting/submodules/simple-knn/simple_knn.cu` was patched with `#include <float.h>` to fix the earlier `FLT_MAX` compile error.
- `simple-knn` also contains local build artifacts from installation: `build/` and `simple_knn.egg-info/`.

2DGS input preparation:

- target path: `data/processed/object_A_book/2dgs_input/`
- original selected images were not modified: `data/processed/object_A_book/images_selected/`
- original formal COLMAP output was not modified: `outputs/reconstruction_2dgs/object_A_book/colmap_full/`
- direct use of `colmap_full/sparse/0` was not valid for 2DGS because its camera model was `SIMPLE_RADIAL`.
- 2DGS accepts only undistorted `PINHOLE` or `SIMPLE_PINHOLE` COLMAP models, so COLMAP `image_undistorter` was run only inside the new `2dgs_input/` target.

Input conversion command:

```bash
conda run -n cv-final-recon colmap image_undistorter \
  --image_path data/processed/object_A_book/images_selected \
  --input_path outputs/reconstruction_2dgs/object_A_book/colmap_full/sparse/0 \
  --output_path data/processed/object_A_book/2dgs_input \
  --output_type COLMAP

mkdir -p data/processed/object_A_book/2dgs_input/sparse/0
mv data/processed/object_A_book/2dgs_input/sparse/*.bin \
  data/processed/object_A_book/2dgs_input/sparse/0/
```

Prepared input:

- `data/processed/object_A_book/2dgs_input/images/`: 150 undistorted images
- `data/processed/object_A_book/2dgs_input/sparse/0/`: undistorted COLMAP model
- verified camera model: `PINHOLE`, width `1051`, height `1870`

2DGS parameter check:

- README says custom COLMAP datasets use `python train.py -s <path to COLMAP or NeRF Synthetic dataset>`.
- `train.py --help` confirmed:
  - source path: `-s` / `--source_path`
  - output path: `-m` / `--model_path`
  - image folder: `-i` / `--images`, default `images`
  - resolution: `-r` / `--resolution`
  - iterations: `--iterations`
  - eval split: `--eval`

Smoke test command:

```bash
env CUDA_VISIBLE_DEVICES=4 \
  CUDA_HOME=/home/dechao/.conda/envs/cv-final-2dgs \
  PATH=/home/dechao/.conda/envs/cv-final-2dgs/bin:$PATH \
  /home/dechao/.conda/envs/cv-final-2dgs/bin/python train.py \
  -s /home/dechao/cv_final_pj/data/processed/object_A_book/2dgs_input \
  -m /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/object_A_book/2dgs_smoke_test \
  --iterations 100 \
  -r 4 \
  --quiet \
  2>&1 | tee /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/object_A_book/2dgs_smoke_test/train_smoke.log
```

Smoke test result:

- GPU used: A800 GPU 4, selected after `nvidia-smi` showed 0 MiB and 0% utilization.
- iterations: `100`, below the 200-iteration smoke-test cap.
- resolution: `-r 4`.
- training started and reached `100/100` iterations.
- initial point count: `41278`.
- this run only validates data loading, CUDA extension execution, and output generation; it does not evaluate reconstruction quality.

Smoke test output:

```text
outputs/reconstruction_2dgs/object_A_book/2dgs_smoke_test/
  cameras.json
  cfg_args
  input.ply
  point_cloud/iteration_100/point_cloud.ply
  train_smoke.log
```

Current Step 2.6 decision:

- 2DGS smoke test passed as a low-cost technical validation.
- Stop here. Do not enter Step 2.7 formal Object A 2DGS training without explicit user approval.

## Step 2.7 Object A 2DGS First Formal Run

Entered Step 2.7 after the user explicitly approved the first formal Object A 2DGS run.

Run target:

- output path: `outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/`
- input path: `data/processed/object_A_book/2dgs_input/`
- GPU: CUDA device `6`, selected after `nvidia-smi` showed 0 MiB and 0% utilization.
- resolution: `-r 4`
- iterations: `30000`
- eval mode: not enabled; all 150 images are training cameras.

Realtime logging change:

- `third_party/2d-gaussian-splatting/train.py` was patched to support `CV_FINAL_2DGS_LOG_INTERVAL`.
- With `CV_FINAL_2DGS_LOG_INTERVAL=50`, training prints one flushed `[ITER ...]` line every 50 iterations.
- `train.log` was written through `tee -a`, so it updated during training rather than only at process exit.
- The final log contains 600 interval loss lines for `30000 / 50`.

Formal training command:

```bash
bash -lc 'set -o pipefail; env \
  CUDA_VISIBLE_DEVICES=6 \
  CV_FINAL_2DGS_LOG_INTERVAL=50 \
  CUDA_HOME=/home/dechao/.conda/envs/cv-final-2dgs \
  PATH=/home/dechao/.conda/envs/cv-final-2dgs/bin:$PATH \
  /home/dechao/.conda/envs/cv-final-2dgs/bin/python -u train.py \
  -s /home/dechao/cv_final_pj/data/processed/object_A_book/2dgs_input \
  -m /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k \
  --iterations 30000 \
  -r 4 \
  2>&1 | tee -a /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/train.log'
```

Run result:

- status: completed successfully
- training progress reached `30000/30000`
- training loop time shown by tqdm: about `06:59`
- no residual training process after completion
- GPU 6 was released after completion
- Tensorboard was not available, so no Tensorboard event files were produced.

Final metrics printed by the training script:

- iteration 7000 train eval: L1 `0.03496456369757652`, PSNR `26.3801326751709`
- iteration 30000 train eval: L1 `0.024293892830610276`, PSNR `28.56709861755371`
- final logged EMA values at iteration 30000:
  - Loss `0.03195`
  - distort `0.00000`
  - normal `0.00522`
  - Points `226409`

Output files:

```text
outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/
  cameras.json
  cfg_args
  input.ply
  point_cloud/iteration_7000/point_cloud.ply
  point_cloud/iteration_30000/point_cloud.ply
  train.log
```

Current Step 2.7 decision:

- First formal Object A 2DGS training run is complete.
- Next recommended action is a render/visual inspection pass from the `iteration_30000` model before deciding whether to tune parameters or accept this Object A reconstruction.

## Step 2.7 Result Acceptance Check

Scope:

- This check only validates the Object A `r4_30k` first formal run.
- No background, AIGC, Blender, report writing, or retraining was started.

Output completeness check:

```text
outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/
  cameras.json
  cfg_args
  input.ply
  point_cloud/iteration_7000/point_cloud.ply
  point_cloud/iteration_30000/point_cloud.ply
  train.log
```

All required files were present before rendering.

Training recap:

- training completed: `30000/30000`
- final points: `226409`
- final logged loss: `0.03195`
- final logged normal: `0.00522`
- final train eval: L1 `0.024293892830610276`, PSNR `28.56709861755371`

Render command confirmation:

- README documents `python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>`.
- `render.py --help` confirms:
  - source path: `-s` / `--source_path`
  - model path: `-m` / `--model_path`
  - checkpoint selection: `--iteration`
  - train/test filtering: `--skip_train`, `--skip_test`
  - mesh skipping: `--skip_mesh`
  - resolution: `-r` / `--resolution`

Train-view render command:

```bash
bash -lc 'set -o pipefail; env \
  CUDA_VISIBLE_DEVICES=6 \
  CUDA_HOME=/home/dechao/.conda/envs/cv-final-2dgs \
  PATH=/home/dechao/.conda/envs/cv-final-2dgs/bin:$PATH \
  /home/dechao/.conda/envs/cv-final-2dgs/bin/python -u render.py \
  -s /home/dechao/cv_final_pj/data/processed/object_A_book/2dgs_input \
  -m /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k \
  --iteration 30000 \
  -r 4 \
  --skip_test \
  --skip_mesh \
  2>&1 | tee /home/dechao/cv_final_pj/outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_train/render.log'
```

Official `render.py` output was first written to:

```text
outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/train/ours_30000/
```

It was then organized into the requested path:

```text
outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_train/
  renders/  # 150 rendered train-view PNGs
  gt/       # 150 matching GT PNGs
  vis/      # 150 depth TIFFs
  render.log
```

Render contact sheet:

```bash
conda run -n cv-final-recon python scripts/reconstruction/make_frame_contact_sheet.py \
  --image_dir outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_train/renders \
  --output outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_contact_sheet.jpg \
  --cols 10 \
  --thumb_width 120 \
  --thumb_height 220 \
  --label_height 24 \
  --summary_path outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_contact_sheet_summary.json \
  --overwrite
```

Generated files:

- `outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_contact_sheet.jpg`
- `outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_contact_sheet_summary.json`

Visual acceptance observation:

- Train-view renders show a coherent book across the full orbit.
- The front, back, spine, and page-side views are all present and stable.
- No obvious catastrophic holes, camera failures, or completely broken views were visible in the render contact sheet.
- Fine text is soft at `-r 4`, which is expected from this first formal run and resolution choice.

Metrics note:

- `metrics.py` exists in the 2DGS repo, but it reads `model/test/...`.
- This run did not use an eval/test split, so no metrics were run; train-view metrics would not replace manual visual inspection.

Acceptance recommendation:

- Accept `r4_30k` as the current Object A first formal reconstruction result.
- Do not start `r2_30k` immediately.
- Consider `r2_30k` only if later close-up render requirements need sharper text or higher texture fidelity.

Final acceptance:

- `outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/` is accepted as the Object A first formal 2DGS reconstruction.
- input images: `150`
- COLMAP registered images: `150/150`
- COLMAP sparse points: `41278`
- 2DGS iterations: `30000`
- resolution: `-r 4`
- final points: `226409`
- render output: `outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_train/`
- visual judgment: train-view renders are very close to GT; no obvious holes, floaters, or camera failure.

Step 2 completion status: Object A formal reconstruction is accepted; do not rerun COLMAP or 2DGS training unless explicitly requested.
