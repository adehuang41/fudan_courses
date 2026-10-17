import os
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


FRAME_MODE = os.environ.get("TASK1_FRAME_MODE", "keyframes")
if FRAME_MODE not in {"keyframes", "all"}:
    raise SystemExit("TASK1_FRAME_MODE must be 'keyframes' or 'all'")

DEFAULT_COMPOSITING_DIR = (
    PROJECT_ROOT
    / "outputs/renders/blender_fusion/task1_render_level_compositing"
)
COMPOSITING_DIR = project_path(os.environ.get("TASK1_COMPOSITING_DIR", DEFAULT_COMPOSITING_DIR))
BACKGROUND_DIR = project_path(
    os.environ.get(
        "TASK1_BACKGROUND_DIR",
        "outputs/reconstruction_2dgs/background_counter/2dgs_final_r4_30k/traj/ours_30000/renders",
    )
)
FOREGROUND_DIR = COMPOSITING_DIR / "foreground_alpha" / (
    "frames" if FRAME_MODE == "all" else "keyframes"
)
OUTPUT_DIR = COMPOSITING_DIR / "composite" / (
    "frames" if FRAME_MODE == "all" else "keyframes"
)
CONTACT_SHEET = COMPOSITING_DIR / (
    "composite_preview_contact_sheet.png" if FRAME_MODE == "all" else "composite_keyframes_contact_sheet.png"
)
FRAMES = list(range(240)) if FRAME_MODE == "all" else [0, 40, 80, 120, 160, 200, 239]
SHEET_FRAMES = [0, 40, 80, 120, 160, 200, 239]
LABEL = os.environ.get("TASK1_COMPOSITE_LABEL", "composite preview")
ALLOW_OVERWRITE = os.environ.get("TASK1_ALLOW_OVERWRITE", "0") == "1"


def fail_if_outputs_exist() -> None:
    if ALLOW_OVERWRITE:
        return

    existing = []
    for frame in FRAMES:
        output = OUTPUT_DIR / f"composite_{frame:05d}.png"
        if output.exists():
            existing.append(output)
    if CONTACT_SHEET.exists():
        existing.append(CONTACT_SHEET)
    if existing:
        paths = "\n".join(f"  - {path}" for path in existing[:20])
        raise SystemExit(
            "Refusing to overwrite existing composite outputs. "
            "Use a fresh TASK1_COMPOSITING_DIR for a new run.\n"
            f"{paths}"
        )


def main() -> None:
    fail_if_outputs_exist()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_composites = []

    for frame in FRAMES:
        background = Image.open(BACKGROUND_DIR / f"{frame:05d}.png").convert("RGBA")
        foreground = Image.open(FOREGROUND_DIR / f"foreground_{frame:05d}.png").convert("RGBA")
        if foreground.size != background.size:
            foreground = foreground.resize(background.size, Image.Resampling.LANCZOS)
        composite = Image.alpha_composite(background, foreground).convert("RGB")
        composite.save(OUTPUT_DIR / f"composite_{frame:05d}.png")
        if frame in SHEET_FRAMES:
            sheet_composites.append((frame, composite.resize((389, 259), Image.Resampling.LANCZOS)))

    pad = 8
    label_h = 28
    width, height = sheet_composites[0][1].size
    sheet = Image.new(
        "RGB",
        (width * len(sheet_composites) + pad * (len(sheet_composites) + 1), height + label_h + pad * 2),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (frame, image) in enumerate(sheet_composites):
        x = pad + index * (width + pad)
        y = pad + label_h
        sheet.paste(image, (x, y))
        draw.text((x, pad), f"{LABEL} {frame:05d}", fill=(20, 20, 20))
    sheet.save(CONTACT_SHEET)
    print(f"Wrote {CONTACT_SHEET}")


if __name__ == "__main__":
    main()
