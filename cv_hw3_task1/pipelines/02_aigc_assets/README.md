# AIGC Asset Pipeline

This main pipeline covers synthetic asset generation from text and image inputs.

## Subdirectories

- `object_B_text_to_3d/`: text-conditioned asset generation
- `object_C_image_to_3d/`: image-conditioned asset generation

## Boundaries

- Put reproducible parameters under `configs/aigc_assets/`
- Put formal scripts under `scripts/aigc/`
- Put generated results under `outputs/`, not inside this pipeline folder
