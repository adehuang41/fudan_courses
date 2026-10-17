import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


BACKGROUND_DIR = project_path(
    os.environ.get(
        "TASK1_BACKGROUND_DIR",
        "outputs/reconstruction_2dgs/background_garden/"
        "2dgs_final_r4_30k_attempt002_20260610-0232/traj/ours_30000/renders",
    )
)
A_CUTOUT_DIR = project_path(
    os.environ.get(
        "TASK1_OBJECT_A_CUTOUT_DIR",
        "outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/"
        "evidence/8view_no_background",
    )
)
OUTPUT_ROOT = project_path(
    os.environ.get(
        "TASK1_OUTPUT_ROOT",
        "outputs/renders/blender_fusion/task1_garden_compositing/"
        "v017_object_A_2dgs_render_sprite_visible_keyframes_20260614",
    )
)
ALLOW_OVERWRITE = os.environ.get("TASK1_ALLOW_OVERWRITE", "0") == "1"
PLACEMENT_PRESET = os.environ.get("TASK1_PLACEMENT_PRESET", "visible")
RUN_LABEL = os.environ.get("TASK1_RUN_LABEL", "v017 A 2DGS")

FRAMES = [0, 40, 80, 120, 160, 200, 239]

# Screen-space placements copied from the previous 3D layout, then tuned for
# the 2DGS book cutouts. Anchor is bottom-center of the book in background px.
PROJECTED_PLACEMENTS = {
    0: {"view": 3, "height": 198, "anchor": (772, 372), "label": "spine"},
    40: {"view": 7, "height": 224, "anchor": (906, 440), "label": "back-pages"},
    80: {"view": 6, "height": 302, "anchor": (772, 582), "label": "open-pages"},
    120: {"view": 5, "height": 276, "anchor": (466, 548), "label": "front-side"},
    160: {"view": 0, "height": 218, "anchor": (400, 418), "label": "back-side"},
    200: {"view": 3, "height": 202, "anchor": (574, 360), "label": "spine"},
    239: {"view": 3, "height": 198, "anchor": (768, 371), "label": "spine"},
}

VISIBLE_PLACEMENTS = {
    0: {"view": 5, "height": 220, "anchor": (795, 405), "label": "front-side-visible"},
    40: {"view": 7, "height": 224, "anchor": (906, 440), "label": "back-pages"},
    80: {"view": 6, "height": 302, "anchor": (772, 582), "label": "open-pages"},
    120: {"view": 5, "height": 276, "anchor": (466, 548), "label": "front-side"},
    160: {"view": 0, "height": 218, "anchor": (400, 418), "label": "back-side"},
    200: {"view": 0, "height": 205, "anchor": (454, 423), "label": "back-side-visible"},
    239: {"view": 5, "height": 220, "anchor": (795, 405), "label": "front-side-visible"},
}

# Same render-level Object A cutouts as v017, but tuned specifically for a
# standing book whose lower edge reads as supported by the table surface.
UPRIGHT_CONTACT_PLACEMENTS = {
    0: {"view": 5, "height": 205, "anchor": (795, 424), "label": "front-upright-contact"},
    40: {"view": 7, "height": 208, "anchor": (906, 456), "label": "back-upright-contact"},
    80: {"view": 6, "height": 276, "anchor": (772, 600), "label": "open-upright-contact"},
    120: {"view": 5, "height": 252, "anchor": (466, 566), "label": "front-upright-contact"},
    160: {"view": 0, "height": 204, "anchor": (400, 438), "label": "back-upright-contact"},
    200: {"view": 0, "height": 192, "anchor": (454, 440), "label": "back-upright-contact"},
    239: {"view": 5, "height": 205, "anchor": (795, 424), "label": "front-upright-contact"},
}

PLACEMENT_PRESETS = {
    "projected": PROJECTED_PLACEMENTS,
    "visible": VISIBLE_PLACEMENTS,
    "upright_contact": UPRIGHT_CONTACT_PLACEMENTS,
}


def cutout_path(view_index: int) -> Path:
    matches = sorted(A_CUTOUT_DIR.glob(f"object_A_2dgs_no_bg_view{view_index:03d}_src*.png"))
    if not matches:
        raise FileNotFoundError(f"Missing Object A cutout for view {view_index:03d} in {A_CUTOUT_DIR}")
    return matches[0]


def remove_white_matte(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = bytearray(rgba.tobytes())
    for i in range(0, len(pixels), 4):
        alpha = pixels[i + 3] / 255.0
        if alpha <= 0.0:
            pixels[i] = pixels[i + 1] = pixels[i + 2] = 0
            continue
        if alpha < 0.98:
            for channel in range(3):
                value = (pixels[i + channel] - 255.0 * (1.0 - alpha)) / alpha
                pixels[i + channel] = max(0, min(255, int(round(value))))
    return Image.frombytes("RGBA", rgba.size, bytes(pixels))


def load_book_sprite(view_index: int, target_height: int) -> Image.Image:
    sprite = Image.open(cutout_path(view_index)).convert("RGBA")
    bbox = sprite.getbbox()
    if bbox is None:
        raise ValueError(f"Empty sprite for view {view_index}")
    sprite = sprite.crop(bbox)
    sprite = remove_white_matte(sprite)

    alpha = sprite.getchannel("A")
    alpha = alpha.point(lambda value: 0 if value < 8 else value)
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.35))
    sprite.putalpha(alpha)

    scale = target_height / sprite.height
    target_width = max(1, int(round(sprite.width * scale)))
    sprite = sprite.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Garden render is softer/darker than the indoor book capture.
    rgb = sprite.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(0.88)
    rgb = ImageEnhance.Contrast(rgb).enhance(0.94)
    rgb = ImageEnhance.Color(rgb).enhance(0.92)
    sprite = Image.merge("RGBA", (*rgb.split(), sprite.getchannel("A")))
    return sprite


def add_contact_shadow(layer: Image.Image, anchor: tuple[int, int], sprite_size: tuple[int, int]) -> None:
    width, height = sprite_size
    cx, cy = anchor
    shadow_w = max(42, int(width * 1.02))
    shadow_h = max(13, int(height * 0.070))
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    bbox = (
        int(cx - shadow_w / 2),
        int(cy - shadow_h * 0.55),
        int(cx + shadow_w / 2),
        int(cy + shadow_h * 0.65),
    )
    draw.ellipse(bbox, fill=(0, 0, 0, 86))
    foot_w = max(20, int(width * 0.48))
    foot_h = max(3, int(height * 0.018))
    draw.rounded_rectangle(
        (
            int(cx - foot_w / 2),
            int(cy - foot_h * 0.70),
            int(cx + foot_w / 2),
            int(cy + foot_h * 0.55),
        ),
        radius=max(1, foot_h // 2),
        fill=(0, 0, 0, 70),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, int(shadow_h * 0.45))))
    layer.alpha_composite(shadow)


def make_contact_sheet(images: list[tuple[int, Image.Image]], output_path: Path, label: str) -> None:
    pad = 8
    label_h = 28
    thumb_w, thumb_h = 389, 259
    sheet = Image.new(
        "RGB",
        (thumb_w * len(images) + pad * (len(images) + 1), thumb_h + label_h + pad * 2),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (frame, image) in enumerate(images):
        x = pad + index * (thumb_w + pad)
        y = pad + label_h
        thumb = image.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.text((x, pad), f"{label} {frame:05d}", fill=(20, 20, 20))
    sheet.save(output_path)


def fail_if_outputs_exist(paths: list[Path]) -> None:
    if ALLOW_OVERWRITE:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        sample = "\n".join(f"  - {path}" for path in existing[:20])
        raise SystemExit(
            "Refusing to overwrite existing outputs. Set TASK1_ALLOW_OVERWRITE=1 or use fresh TASK1_OUTPUT_ROOT.\n"
            f"{sample}"
        )


def main() -> None:
    if PLACEMENT_PRESET not in PLACEMENT_PRESETS:
        valid = ", ".join(sorted(PLACEMENT_PRESETS))
        raise SystemExit(f"Unknown TASK1_PLACEMENT_PRESET={PLACEMENT_PRESET!r}. Valid: {valid}")
    placements = PLACEMENT_PRESETS[PLACEMENT_PRESET]

    foreground_dir = OUTPUT_ROOT / "foreground_alpha/keyframes"
    composite_dir = OUTPUT_ROOT / "composite/keyframes"
    metadata_dir = OUTPUT_ROOT / "metadata"
    foreground_sheet = OUTPUT_ROOT / "object_A_2dgs_sprite_foreground_contact_sheet.png"
    composite_sheet = OUTPUT_ROOT / "composite_keyframes_contact_sheet.png"
    metadata_path = metadata_dir / "object_A_2dgs_render_sprite_v017.json"
    report_note_path = OUTPUT_ROOT / "report_method_note_object_A_2dgs_render_compositing.md"

    expected_outputs = [foreground_sheet, composite_sheet, metadata_path, report_note_path]
    expected_outputs.extend(foreground_dir / f"foreground_{frame:05d}.png" for frame in FRAMES)
    expected_outputs.extend(composite_dir / f"composite_{frame:05d}.png" for frame in FRAMES)
    fail_if_outputs_exist(expected_outputs)

    foreground_dir.mkdir(parents=True, exist_ok=True)
    composite_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    foreground_images = []
    composite_images = []
    resolved = {}

    for frame in FRAMES:
        background = Image.open(BACKGROUND_DIR / f"{frame:05d}.png").convert("RGBA")
        placement = placements[frame]
        sprite = load_book_sprite(placement["view"], placement["height"])
        anchor_x, anchor_y = placement["anchor"]
        paste_x = int(round(anchor_x - sprite.width / 2))
        paste_y = int(round(anchor_y - sprite.height))

        foreground = Image.new("RGBA", background.size, (0, 0, 0, 0))
        add_contact_shadow(foreground, (anchor_x, anchor_y), sprite.size)
        foreground.alpha_composite(sprite, (paste_x, paste_y))

        composite = Image.alpha_composite(background, foreground).convert("RGB")

        foreground_path = foreground_dir / f"foreground_{frame:05d}.png"
        composite_path = composite_dir / f"composite_{frame:05d}.png"
        foreground.save(foreground_path)
        composite.save(composite_path)

        foreground_images.append((frame, foreground))
        composite_images.append((frame, composite))
        resolved[frame] = {
            **placement,
            "sprite_size": sprite.size,
            "paste_xy": (paste_x, paste_y),
            "cutout": str(cutout_path(placement["view"]).relative_to(PROJECT_ROOT)),
            "foreground": str(foreground_path.relative_to(PROJECT_ROOT)),
            "composite": str(composite_path.relative_to(PROJECT_ROOT)),
        }

    make_contact_sheet(foreground_images, foreground_sheet, "A 2DGS fg")
    make_contact_sheet(composite_images, composite_sheet, RUN_LABEL)

    metadata = {
        "method": "render-level compositing",
        "placement_preset": PLACEMENT_PRESET,
        "object_A_source": "2DGS render cutouts with alpha mask; no mesh conversion",
        "background_source": str(BACKGROUND_DIR.relative_to(PROJECT_ROOT)),
        "frames": FRAMES,
        "placements": resolved,
        "notes": [
            "Object A point cloud still contains original capture background.",
            "Foreground PNGs use A's real 2DGS rendered appearance after alpha masking.",
            "Composition is image-space RGBA, not a single unified 3D scene.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report_note_path.write_text(
        "\n".join(
            [
                "# Object A Render-Level Compositing",
                "",
                "Object A is reconstructed and rendered with 2D Gaussian Splatting. "
                "The original Object A Gaussian point cloud is not a clean object-only asset: "
                "it still contains parts of the capture background. "
                "Therefore, I did not use the failed mesh-conversion result for final fusion.",
                "",
                "For final composition, I rendered Object A from its 2DGS model, removed the capture "
                "background with an alpha mask, and obtained RGBA foreground sprites. "
                "The garden background D was also rendered to RGB frames. "
                "I unified the two representations at render level by compositing the Object A RGBA "
                "foreground over the D RGB render frames in image space, with manually selected "
                "view, scale, screen position, and a soft contact shadow for each keyframe.",
                "",
                "This path preserves the real visual appearance of the Object A 2DGS reconstruction. "
                "It is not a single unified 3D scene and does not provide exact 3D occlusion, lighting "
                "interaction, or parallax between A and D.",
                "",
                f"Keyframes: {FRAMES}",
                f"Placement preset: {PLACEMENT_PRESET}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {composite_sheet}")
    print(f"Wrote {foreground_sheet}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {report_note_path}")


if __name__ == "__main__":
    main()
