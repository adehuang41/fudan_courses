from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "report" / "cn_draft" / "generated"


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def trim_near_white(im: Image.Image, threshold: int = 245, pad: int = 6) -> Image.Image:
    rgb = im.convert("RGB")
    px = rgb.load()
    w, h = rgb.size
    xs, ys = [], []
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if min(r, g, b) < threshold:
                xs.append(x)
                ys.append(y)
    if not xs:
        return im
    left = max(0, min(xs) - pad)
    top = max(0, min(ys) - pad)
    right = min(w, max(xs) + pad + 1)
    bottom = min(h, max(ys) + pad + 1)
    return im.crop((left, top, right, bottom))


def make_yoyo_failure() -> None:
    src = Image.open(OUT / "object_C_yoyo_guide_clean.png").convert("RGB")
    # The lower row is the side view where the yoyo waist becomes an unstable notch.
    crop = src.crop((0, 580, src.width, src.height))
    crop = trim_near_white(crop, threshold=252, pad=0)
    crop.save(OUT / "object_C_yoyo_guide_side_failure.png")


def make_rubik_try16_eight_views() -> None:
    src = Image.open(ROOT / "report" / "assets" / "object_C_try16_final_views.png").convert("RGB")
    x0s = [16, 372, 728, 1084]
    y0s = [68, 378]
    cell_w, cell_h = 341, 260
    panels = []
    for y0 in y0s:
        for x0 in x0s:
            cell = src.crop((x0, y0, x0 + cell_w, y0 + cell_h))
            rgb = cell.convert("RGB")
            px = rgb.load()
            dark_xs, dark_ys = [], []
            for y in range(rgb.height):
                for x in range(rgb.width):
                    r, g, b = px[x, y]
                    if max(r, g, b) < 45:
                        dark_xs.append(x)
                        dark_ys.append(y)
            if dark_xs:
                left = max(0, min(dark_xs) - 115)
                right = min(rgb.width, max(dark_xs) + 8)
                top = max(0, min(dark_ys) - 10)
                bottom = min(rgb.height, max(dark_ys) + 10)
                panels.append(rgb.crop((left, top, right, bottom)))
            else:
                panels.append(trim_near_white(cell, threshold=248, pad=8))

    panel_w, panel_h = 560, 220
    gap_x, gap_y = 18, 38
    label_h = 28
    canvas = Image.new(
        "RGB",
        (4 * panel_w + 3 * gap_x, 2 * (panel_h + label_h) + gap_y),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    f = font(22)
    for i, panel in enumerate(panels):
        scale = min(panel_w / panel.width, panel_h / panel.height)
        panel = panel.resize((int(panel.width * scale), int(panel.height * scale)), Image.Resampling.LANCZOS)
        col = i % 4
        row = i // 4
        x = col * (panel_w + gap_x) + (panel_w - panel.width) // 2
        y = row * (panel_h + label_h + gap_y)
        canvas.paste(panel, (x, y))
        draw.text((col * (panel_w + gap_x), y + panel_h + 4), f"view {i}", fill=(45, 45, 45), font=f)
    canvas.save(OUT / "object_C_rubik_try16_eight_views_grid.png")

    def save_2x2(indices: list[int], output_name: str) -> None:
        panel_w, panel_h = 900, 320
        gap_x, gap_y = 26, 42
        label_h = 30
        canvas = Image.new(
            "RGB",
            (2 * panel_w + gap_x, 2 * (panel_h + label_h) + gap_y),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        f = font(24)
        for j, idx in enumerate(indices):
            panel = panels[idx]
            scale = min(panel_w / panel.width, panel_h / panel.height)
            panel = panel.resize(
                (int(panel.width * scale), int(panel.height * scale)),
                Image.Resampling.LANCZOS,
            )
            col = j % 2
            row = j // 2
            x = col * (panel_w + gap_x) + (panel_w - panel.width) // 2
            y = row * (panel_h + label_h + gap_y)
            canvas.paste(panel, (x, y))
            draw.text((col * (panel_w + gap_x), y + panel_h + 4), f"view {idx}", fill=(45, 45, 45), font=f)
        canvas.save(OUT / output_name)

    save_2x2([0, 1, 2, 3], "object_C_rubik_try16_views_0_3_grid2x2.png")
    save_2x2([4, 5, 6, 7], "object_C_rubik_try16_views_4_7_grid2x2.png")


def make_fusion_outputs() -> None:
    run = ROOT / "outputs" / "gaussian_fusion" / "task1_abc_mesh_gaussians_v005_a_front_out_c_yaw_m90" / "keyframes_v001" / "renders"
    keys = ["00000", "00040", "00080", "00120", "00160", "00200", "00239"]
    frames = [Image.open(run / f"{k}.png").convert("RGB") for k in keys]

    tile_w, tile_h = 520, 337
    gap = 10
    label_h = 24
    canvas = Image.new("RGB", (4 * tile_w + 3 * gap, 2 * (tile_h + label_h) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    f = font(18)
    positions = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)]
    for frame, key, (col, row) in zip(frames, keys, positions):
        tile = frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = col * (tile_w + gap)
        y = row * (tile_h + label_h + gap)
        canvas.paste(tile, (x, y))
        draw.text((x + 6, y + tile_h + 2), key, fill=(45, 45, 45), font=f)
    canvas.save(OUT / "fusion_v005_keyframes_grid_4x3.png")

    representative = Image.open(run / "00120.png").convert("RGB")
    representative.save(OUT / "fusion_v005_representative_frame_00120.png")
    closeup_source = Image.open(run / "00080.png").convert("RGB")
    closeup_source.crop((320, 250, 1120, 620)).save(OUT / "fusion_v005_final_closeup.png")

    def save_three_frame_grid(run_dir: Path, output_name: str, keys_for_grid: list[str]) -> None:
        frames = [Image.open(run_dir / f"{key}.png").convert("RGB") for key in keys_for_grid]
        tile_w, tile_h = 690, 447
        gap = 12
        label_h = 26
        canvas = Image.new("RGB", (3 * tile_w + 2 * gap, tile_h + label_h), "white")
        draw = ImageDraw.Draw(canvas)
        f = font(19)
        for idx, (frame, key) in enumerate(zip(frames, keys_for_grid)):
            tile = frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            x = idx * (tile_w + gap)
            canvas.paste(tile, (x, 0))
            draw.text((x + 6, tile_h + 3), key, fill=(45, 45, 45), font=f)
        canvas.save(OUT / output_name)

    def save_single_frame(run_dir: Path, output_name: str, key: str, crop_square: bool = False) -> None:
        frame = Image.open(run_dir / f"{key}.png").convert("RGB")
        if crop_square:
            w, h = frame.size
            side = min(w, h)
            left = max(0, (w - side) // 2)
            frame = frame.crop((left, 0, left + side, side))
        frame.save(OUT / output_name)

    def save_two_frame_grid(
        run_dir: Path,
        output_name: str,
        keys_for_grid: list[str],
        crop_square: bool = False,
    ) -> None:
        frames = [Image.open(run_dir / f"{key}.png").convert("RGB") for key in keys_for_grid]
        tile_w, tile_h = (840, 840) if crop_square else (960, 622)
        gap = 14
        label_h = 30
        canvas = Image.new("RGB", (2 * tile_w + gap, tile_h + label_h), "white")
        draw = ImageDraw.Draw(canvas)
        f = font(22)
        for idx, (frame, key) in enumerate(zip(frames, keys_for_grid)):
            if crop_square:
                w, h = frame.size
                side = min(w, h)
                left = max(0, (w - side) // 2)
                frame = frame.crop((left, 0, left + side, side))
            tile = frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            x = idx * (tile_w + gap)
            canvas.paste(tile, (x, 0))
            draw.text((x + 8, tile_h + 3), key, fill=(45, 45, 45), font=f)
        canvas.save(OUT / output_name)

    def save_two_named_frames(image_paths: list[Path], labels: list[str], output_name: str) -> None:
        frames = [Image.open(path).convert("RGB") for path in image_paths]
        tile_w, tile_h = 960, 622
        gap = 14
        label_h = 30
        canvas = Image.new("RGB", (2 * tile_w + gap, tile_h + label_h), "white")
        draw = ImageDraw.Draw(canvas)
        f = font(22)
        for idx, (frame, label) in enumerate(zip(frames, labels)):
            tile = frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            x = idx * (tile_w + gap)
            canvas.paste(tile, (x, 0))
            draw.text((x + 8, tile_h + 3), label, fill=(45, 45, 45), font=f)
        canvas.save(OUT / output_name)

    def save_four_frame_grid(run_dir: Path, output_name: str, keys_for_grid: list[str]) -> None:
        frames = [Image.open(run_dir / f"{key}.png").convert("RGB") for key in keys_for_grid]
        tile_w, tile_h = 900, 583
        gap_x, gap_y = 14, 20
        label_h = 30
        canvas = Image.new("RGB", (2 * tile_w + gap_x, 2 * (tile_h + label_h) + gap_y), "white")
        draw = ImageDraw.Draw(canvas)
        f = font(22)
        positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for frame, key, (col, row) in zip(frames, keys_for_grid, positions):
            tile = frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            x = col * (tile_w + gap_x)
            y = row * (tile_h + label_h + gap_y)
            canvas.paste(tile, (x, y))
            draw.text((x + 8, y + tile_h + 3), key, fill=(45, 45, 45), font=f)
        canvas.save(OUT / output_name)

    def save_three_frame_large(run_dir: Path, output_name: str, keys_for_grid: list[str]) -> None:
        frames = [Image.open(run_dir / f"{key}.png").convert("RGB") for key in keys_for_grid]
        tile_w, tile_h = 900, 583
        gap_x, gap_y = 14, 20
        label_h = 30
        canvas = Image.new("RGB", (2 * tile_w + gap_x, 2 * (tile_h + label_h) + gap_y), "white")
        draw = ImageDraw.Draw(canvas)
        f = font(22)
        positions = [(0, 0), (1, 0), (0.5, 1)]
        for frame, key, (col, row) in zip(frames, keys_for_grid, positions):
            tile = frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            x = int(col * (tile_w + gap_x))
            y = row * (tile_h + label_h + gap_y)
            canvas.paste(tile, (x, y))
            draw.text((x + 8, y + tile_h + 3), key, fill=(45, 45, 45), font=f)
        canvas.save(OUT / output_name)

    fusion_root = ROOT / "outputs" / "gaussian_fusion"
    blender_keyframes = ROOT / "final_blender" / "composite" / "keyframes"
    representative_keys = ["00040", "00120", "00200"]
    synthetic_dir = fusion_root / "task1_synthetic_magenta_cube_v001" / "keyframes_v001" / "renders"
    v001_dir = fusion_root / "task1_abc_mesh_gaussians_v001" / "keyframes_v001" / "renders"
    v004_dir = fusion_root / "task1_abc_mesh_gaussians_v004_c_yaw_m90" / "keyframes_v001" / "renders"

    save_three_frame_grid(
        synthetic_dir,
        "fusion_synthetic_cube_keyframes_3.png",
        representative_keys,
    )
    save_three_frame_grid(
        v001_dir,
        "fusion_v001_keyframes_3.png",
        representative_keys,
    )
    save_three_frame_grid(
        v004_dir,
        "fusion_v004_keyframes_3.png",
        representative_keys,
    )
    save_single_frame(synthetic_dir, "fusion_synthetic_cube_frame_00040.png", "00040")
    save_single_frame(synthetic_dir, "fusion_synthetic_cube_frame_00120.png", "00120")
    save_single_frame(synthetic_dir, "fusion_synthetic_cube_frame_00040_square.png", "00040", crop_square=True)
    save_single_frame(synthetic_dir, "fusion_synthetic_cube_frame_00120_square.png", "00120", crop_square=True)
    save_two_named_frames(
        [blender_keyframes / "composite_00040.png", blender_keyframes / "composite_00120.png"],
        ["00040", "00120"],
        "fusion_blender_alpha_keyframes_large_pair.png",
    )
    save_two_frame_grid(run, "fusion_v005_success_pair.png", ["00040", "00120"], crop_square=True)
    save_two_frame_grid(v001_dir, "fusion_v001_keyframes_large_pair.png", ["00040", "00120"])
    save_two_frame_grid(v004_dir, "fusion_v004_keyframes_large_pair.png", ["00120", "00200"])
    save_four_frame_grid(run, "fusion_v005_keyframes_0_3_large.png", ["00000", "00040", "00080", "00120"])
    save_three_frame_large(run, "fusion_v005_keyframes_4_6_large.png", ["00160", "00200", "00239"])


def main() -> None:
    make_yoyo_failure()
    make_rubik_try16_eight_views()
    make_fusion_outputs()


if __name__ == "__main__":
    main()
