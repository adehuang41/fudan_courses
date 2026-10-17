from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

if os.environ.get("TASK1_WRITE_VIDEO", "0") == "1":
    import mediapy as media
else:
    media = None


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")
V018_ROOT = PROJECT_ROOT / (
    "outputs/renders/blender_fusion/task1_garden_compositing/"
    "v018_object_A_2dgs_render_sprite_upright_contact_keyframes_20260614"
)
METADATA_PATH = V018_ROOT / "metadata/object_A_2dgs_render_sprite_v017.json"
OUTPUT_ROOT = PROJECT_ROOT / (
    "outputs/renders/blender_fusion/task1_garden_compositing/"
    "v018_diagnostic_sprite_transition_samples_20260616"
)
FULL_FRAMES = os.environ.get("TASK1_FULL_FRAMES", "0") == "1"


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


@lru_cache(maxsize=16)
def load_base_sprite(cutout_str: str) -> Image.Image:
    cutout = Path(cutout_str)
    sprite = Image.open(cutout).convert("RGBA")
    bbox = sprite.getbbox()
    if bbox is None:
        raise ValueError(f"empty cutout: {cutout}")
    sprite = remove_white_matte(sprite.crop(bbox))
    alpha = sprite.getchannel("A")
    alpha = alpha.point(lambda value: 0 if value < 8 else value)
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.35))
    sprite.putalpha(alpha)
    return sprite


def load_sprite(cutout: Path, target_height: int) -> Image.Image:
    sprite = load_base_sprite(str(cutout)).copy()
    scale = target_height / sprite.height
    target_width = max(1, int(round(sprite.width * scale)))
    sprite = sprite.resize((target_width, target_height), Image.Resampling.LANCZOS)

    rgb = sprite.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(0.88)
    rgb = ImageEnhance.Contrast(rgb).enhance(0.94)
    rgb = ImageEnhance.Color(rgb).enhance(0.92)
    return Image.merge("RGBA", (*rgb.split(), sprite.getchannel("A")))


def add_contact_shadow(layer: Image.Image, anchor: tuple[int, int], sprite_size: tuple[int, int]) -> None:
    width, height = sprite_size
    cx, cy = anchor
    shadow_w = max(42, int(width * 1.02))
    shadow_h = max(13, int(height * 0.070))
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.ellipse(
        (
            int(cx - shadow_w / 2),
            int(cy - shadow_h * 0.55),
            int(cx + shadow_w / 2),
            int(cy + shadow_h * 0.65),
        ),
        fill=(0, 0, 0, 86),
    )
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


def interp_scalar(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t


def interpolate_placement(frame: int, keyframes: list[int], placements: dict[str, dict]) -> dict:
    if frame <= keyframes[0]:
        return {**placements[str(keyframes[0])], "nearest_keyframe": keyframes[0]}
    if frame >= keyframes[-1]:
        return {**placements[str(keyframes[-1])], "nearest_keyframe": keyframes[-1]}

    left = max(k for k in keyframes if k <= frame)
    right = min(k for k in keyframes if k >= frame)
    if left == right:
        return {**placements[str(left)], "nearest_keyframe": left}

    lp = placements[str(left)]
    rp = placements[str(right)]
    t = (frame - left) / float(right - left)
    nearest = left if t < 0.5 else right
    np_placement = placements[str(nearest)]
    ax = interp_scalar(float(lp["anchor"][0]), float(rp["anchor"][0]), t)
    ay = interp_scalar(float(lp["anchor"][1]), float(rp["anchor"][1]), t)
    height = interp_scalar(float(lp["height"]), float(rp["height"]), t)
    return {
        **np_placement,
        "anchor": [int(round(ax)), int(round(ay))],
        "height": int(round(height)),
        "nearest_keyframe": nearest,
        "left_keyframe": left,
        "right_keyframe": right,
        "t": t,
    }


def make_contact_sheet(frames: list[tuple[int, Image.Image]], path: Path) -> None:
    pad = 8
    label_h = 28
    thumb_w, thumb_h = 320, 207
    cols = 4
    rows = int(np.ceil(len(frames) / cols))
    sheet = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * pad, rows * (thumb_h + label_h) + (rows + 1) * pad),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    for i, (frame, image) in enumerate(frames):
        row, col = divmod(i, cols)
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h)
        draw.text((x, y), f"interp sprite {frame:05d}", fill=(20, 20, 20))
        sheet.paste(image.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y + label_h))
    sheet.save(path)


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUTPUT_ROOT}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    background_dir = PROJECT_ROOT / metadata["background_source"]
    placements = metadata["placements"]
    keyframes = sorted(int(k) for k in placements)

    frame_dir = OUTPUT_ROOT / "composite/frames"
    frame_dir.mkdir(parents=True)

    video_frames = []
    sheet_frames = []
    resolved = {}
    sample_frames = [
        0,
        19,
        20,
        21,
        40,
        59,
        60,
        61,
        80,
        99,
        100,
        101,
        120,
        139,
        140,
        141,
        160,
        179,
        180,
        181,
        200,
        219,
        220,
        221,
        239,
    ]
    frames_to_render = range(240) if FULL_FRAMES or media is not None else sample_frames

    for frame in frames_to_render:
        background = Image.open(background_dir / f"{frame:05d}.png").convert("RGBA")
        placement = interpolate_placement(frame, keyframes, placements)
        cutout = PROJECT_ROOT / placement["cutout"]
        sprite = load_sprite(cutout, int(placement["height"]))
        anchor_x, anchor_y = int(placement["anchor"][0]), int(placement["anchor"][1])
        paste_x = int(round(anchor_x - sprite.width / 2))
        paste_y = int(round(anchor_y - sprite.height))

        foreground = Image.new("RGBA", background.size, (0, 0, 0, 0))
        add_contact_shadow(foreground, (anchor_x, anchor_y), sprite.size)
        foreground.alpha_composite(sprite, (paste_x, paste_y))
        composite = Image.alpha_composite(background, foreground).convert("RGB")
        composite.save(frame_dir / f"composite_{frame:05d}.png")
        if media is not None:
            video_frames.append(np.asarray(composite))
        sheet_frames.append((frame, composite))
        resolved[str(frame)] = {
            "view": placement["view"],
            "height": placement["height"],
            "anchor": placement["anchor"],
            "nearest_keyframe": placement["nearest_keyframe"],
            "paste_xy": [paste_x, paste_y],
            "cutout": str(cutout.relative_to(PROJECT_ROOT)),
        }

    make_contact_sheet(sheet_frames, OUTPUT_ROOT / "diagnostic_contact_sheet.png")
    (OUTPUT_ROOT / "diagnostic_metadata.json").write_text(
        json.dumps(
            {
                "source": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
                "method": "diagnostic full-frame interpolation of v018 keyframe sprite placements",
                "warning": "This is still image-space RGBA compositing with nearest-keyframe view selection.",
                "frames": resolved,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if media is not None:
        media.write_video(OUTPUT_ROOT / "diagnostic_v018_nearest_sprite_30fps.mp4", video_frames, fps=30)
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
