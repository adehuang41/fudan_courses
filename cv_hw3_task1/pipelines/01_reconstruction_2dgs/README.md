# Reconstruction Pipeline

This main pipeline covers real-object and real-scene reconstruction.

## Subdirectories

- `object_A/`: self-captured multi-view object reconstruction
- `background/`: host-scene reconstruction

## Boundaries

- Put reproducible parameters under `configs/reconstruction_2dgs/`
- Put formal scripts under `scripts/reconstruction/`
- Put generated results under `outputs/`, not inside this pipeline folder

## Phase 2 Object A Book Paths

Current Object A instance:

```text
object_A_book
```

Input and intermediate paths:

```text
data/raw/object_A_book/videos/book_raw.mp4
data/processed/object_A_book/images_raw/
data/processed/object_A_book/images_smoke/
data/processed/object_A_book/images_selected/
data/processed/object_A_book/metadata/
```

Output paths:

```text
outputs/reconstruction_2dgs/object_A_book/frame_contact_sheet.jpg
outputs/reconstruction_2dgs/object_A_book/colmap_smoke/
```

Directory roles:

- `images_raw/`: candidate frame pool extracted from the video
- `images_smoke/`: 15-20 frame small-scale COLMAP sanity-check input
- `images_selected/`: future formal-scale COLMAP input after filtering

Do not run 2DGS from the smoke-test result. Formal-scale COLMAP must pass first.

## Step 2 Completion Summary

Task 1 Step 2 is complete.

### Object A Final Accepted Result

- Object: `object_A_book`
- `images_selected`: `150`
- COLMAP registered images: `150/150`
- COLMAP sparse points: `41278`
- 2DGS final output:

```text
outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/
```

- final iterations: `30000`
- resolution: `-r 4`
- final points: `226409`
- visual judgment: train-view renders and GT are almost identical; accepted as Object A formal reconstruction

### Background Final Accepted Result

- dataset: Mip-NeRF 360 `counter`
- data path:

```text
data/raw/background_mipnerf360/counter/
```

- 2DGS final output:

```text
outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/
```

- final iterations: `30000`
- resolution: `-r 4`
- final points: `525815`
- train eval: L1 `0.01684`, PSNR `29.95`
- visual judgment: train-view renders and GT are almost identical; accepted as formal background reconstruction

Do not rerun COLMAP or 2DGS training for Step 2 unless explicitly requested.
