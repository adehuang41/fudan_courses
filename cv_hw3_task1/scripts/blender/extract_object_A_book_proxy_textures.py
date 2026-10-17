from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")
SOURCE_DIR = (
    PROJECT_ROOT
    / "outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_train/renders"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/renders/blender_fusion/task1_render_level_compositing/textures"
)

TEXTURES = {
    "object_A_book_front_render_00085_crop.png": {
        "source": "00085.png",
        "crop": (28, 34, 238, 415),
    },
    "object_A_book_back_render_00020_crop.png": {
        "source": "00020.png",
        "crop": (31, 38, 229, 401),
    },
    "object_A_book_pages_render_00120_crop.png": {
        "source": "00120.png",
        "crop": (68, 15, 168, 438),
    },
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for output_name, spec in TEXTURES.items():
        source = Image.open(SOURCE_DIR / spec["source"]).convert("RGB")
        crop = source.crop(spec["crop"])
        output = OUTPUT_DIR / output_name
        crop.save(output, quality=95)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
