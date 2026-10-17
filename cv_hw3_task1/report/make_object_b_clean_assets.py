from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "report" / "assets"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)

BG = (255, 255, 255)
LINE = (218, 224, 232)
TEXT = (30, 35, 42)
MUTED = (92, 99, 112)


def img(rel: str) -> Image.Image:
    return Image.open(ROOT / rel).convert("RGB")


def corner_background(im: Image.Image) -> tuple[int, int, int]:
    rgb = im.convert("RGB")
    samples = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((rgb.width - 1, 0)),
        rgb.getpixel((0, rgb.height - 1)),
        rgb.getpixel((rgb.width - 1, rgb.height - 1)),
    ]
    return tuple(sum(p[i] for p in samples) // len(samples) for i in range(3))


def fit(im: Image.Image, width: int, height: int, bg: tuple[int, int, int] = BG) -> Image.Image:
    scale = min(width / im.width, height / im.height)
    new_size = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
    resized = im.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), bg)
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font=FONT_BOLD, fill=TEXT) -> None:
    draw.text(xy, text, font=font, fill=fill)


def make_row_grid(
    out_name: str,
    columns: list[str],
    rows: list[tuple[str, list[str]]],
    cell_w: int = 560,
    cell_h: int = 190,
    label_w: int = 260,
) -> None:
    pad = 18
    header_h = 58
    row_gap = 18
    width = label_w + len(columns) * cell_w + (len(columns) + 2) * pad
    height = header_h + len(rows) * (cell_h + row_gap) + pad
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    x0 = label_w + 2 * pad
    for i, col in enumerate(columns):
        draw.text((x0 + i * cell_w + 10, 16), col, font=FONT_BOLD, fill=TEXT)
    draw.line((pad, header_h - 8, width - pad, header_h - 8), fill=LINE, width=2)

    y = header_h
    for label, sources in rows:
        draw.text((pad, y + 12), label, font=FONT_BOLD, fill=TEXT)
        for i, src in enumerate(sources):
            panel = fit(img(src), cell_w - 10, cell_h)
            x = x0 + i * cell_w
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell_w - 10, y + cell_h), outline=LINE, width=1)
        y += cell_h + row_gap

    canvas.save(ASSET_DIR / out_name, quality=95)


def make_column_grid(
    out_name: str,
    columns: list[tuple[str, str]],
    cell_w: int = 390,
    cell_h: int = 135,
) -> None:
    pad = 16
    label_h = 42
    width = len(columns) * cell_w + (len(columns) + 1) * pad
    height = label_h + cell_h + 2 * pad
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    for i, (label, src) in enumerate(columns):
        x = pad + i * (cell_w + pad)
        draw.text((x + 6, 12), label, font=FONT_SMALL, fill=MUTED)
        panel = fit(img(src), cell_w, cell_h)
        canvas.paste(panel, (x, label_h + pad))
        draw.rectangle((x, label_h + pad, x + cell_w, label_h + pad + cell_h), outline=LINE, width=1)

    canvas.save(ASSET_DIR / out_name, quality=95)


def make_wrapped_checkpoint_grid(
    out_name: str,
    columns: list[tuple[str, str]],
    cols_per_row: int = 3,
    cell_w: int = 610,
    cell_h: int = 205,
) -> None:
    pad = 18
    label_h = 44
    row_gap = 18
    rows = (len(columns) + cols_per_row - 1) // cols_per_row
    width = cols_per_row * cell_w + (cols_per_row + 1) * pad
    height = rows * (label_h + cell_h) + (rows + 1) * pad + (rows - 1) * row_gap
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    for idx, (label, src) in enumerate(columns):
        row = idx // cols_per_row
        col = idx % cols_per_row
        x = pad + col * (cell_w + pad)
        y = pad + row * (label_h + cell_h + row_gap)
        draw.text((x + 6, y + 8), label, font=FONT_BOLD, fill=MUTED)
        panel = fit(img(src), cell_w, cell_h)
        panel_y = y + label_h
        canvas.paste(panel, (x, panel_y))
        draw.rectangle((x, panel_y, x + cell_w, panel_y + cell_h), outline=LINE, width=1)

    canvas.save(ASSET_DIR / out_name, quality=95)


def make_early_targets() -> None:
    rows = [
        (
            "duck seed7",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_simple_seed7_g70_3000steps@20260531-184143/save/it3000-0.png",
            ],
        ),
        (
            "duck seed42",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_simple_seed42_g70_3000steps@20260531-190238/save/it1000-0.png",
            ],
        ),
        (
            "duck long",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubber_duck_formal_10000steps@20260531-171926/save/it7000-0.png",
            ],
        ),
        (
            "cone seed7",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/traffic_cone_seed7_g70_resume1500_to_3000steps@20260602-180305/save/it3000-0.png",
            ],
        ),
        (
            "white-ring",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/traffic_cone_seed7_white_painted_ring_targeted_1000steps@20260603-0920/save/it500-0.png",
            ],
        ),
    ]
    make_row_grid("object_B_early_targets_clean.png", ["RGB / normal / opacity"], rows, cell_w=950, cell_h=320, label_w=220)


def make_dreamfusion_rubik() -> None:
    rows = [
        (
            "Try1 direct",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try1_resume1000_to3000_solved_g70_val500@20260612-202637/save/it3000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try1_resume1000_to3000_solved_g70_val500@20260612-202637/save/it3000-5.png",
            ],
        ),
        (
            "Try2 geom g70",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try2_geometry_first_clean_cube_g70_3000steps@20260612-214444/save/it3000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try2_geometry_first_clean_cube_g70_3000steps@20260612-214444/save/it3000-5.png",
            ],
        ),
        (
            "Try3 geom g50",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try3_geometry_first_clean_cube_g50_darkness_control_3000steps@20260612-214444/save/it3000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try3_geometry_first_clean_cube_g50_darkness_control_3000steps@20260612-214444/save/it3000-5.png",
            ],
        ),
        (
            "Try4A harden",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try4A_combined_hardened_g60_3000steps@20260612-224539/save/it2000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try4A_combined_hardened_g60_3000steps@20260612-224539/save/it2000-5.png",
            ],
        ),
        (
            "Try4B g70",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try4B_combined_hardened_g70_3000steps@20260612-224539/save/it1500-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd/rubiks_cube_try4B_combined_hardened_g70_3000steps@20260612-224539/save/it1500-5.png",
            ],
        ),
    ]
    make_row_grid("object_B_rubik_dreamfusion_clean.png", ["view 0", "view 5"], rows)


def make_magic3d_blackening() -> None:
    run = "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-sd/B0_s42@20260613-B0e/save"
    columns = [(f"{step}", f"{run}/it{step}-0.png") for step in ("500", "1000", "2000", "2500", "3000")]
    make_wrapped_checkpoint_grid("object_B_rubik_blackening_clean.png", columns)


def make_geometry_prior() -> None:
    rows = [
        (
            "B0IF prompt",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0if_s7@20260613-B0if/save/it3000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0if_s7@20260613-B0if/save/it3000-5.png",
            ],
        ),
        (
            "B0k prior",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0k_s7@20260613-B0k-cubeprior-s7/save/it2000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0k_s7@20260613-B0k-cubeprior-s7/save/it2000-5.png",
            ],
        ),
        (
            "B1 refine",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-sd/B1_s7@20260613-B1t_t010/save/it100-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-sd/B1_s7@20260613-B1t_t010/save/it100-5.png",
            ],
        ),
        (
            "B2 fixed",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-scaffold-if/B2if_s7@20260613-B2if/save/it1000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-scaffold-if/B2if_s7@20260613-B2if/save/it1000-5.png",
            ],
        ),
        (
            "B2tIF 1000",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3/save/it1000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3/save/it1000-5.png",
            ],
        ),
        (
            "B2tIF 1501",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3r2000/save/it1501-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3r2000/save/it1501-5.png",
            ],
        ),
    ]
    make_row_grid("object_B_rubik_geometry_clean.png", ["view 0", "view 5"], rows)


def make_apple() -> None:
    rows = [
        (
            "A0 3000",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if-apple/A0apple_s7@20260613-A0apple-if-s7/save/it3000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if-apple/A0apple_s7@20260613-A0apple-if-s7/save/it3000-5.png",
            ],
        ),
        (
            "A1stem 500",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if-apple/A1appleStem_s7@20260614-A1appleStem-s7/save/it500-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if-apple/A1appleStem_s7@20260614-A1appleStem-s7/save/it500-5.png",
            ],
        ),
        (
            "A1b 1000",
            [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if-apple/A1bappleStem_s7@20260614-A1bappleStem-farcam-s7/save/it1000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if-apple/A1bappleStem_s7@20260614-A1bappleStem-farcam-s7/save/it1000-5.png",
            ],
        ),
    ]
    mesh_sources = [
        "outputs/aigc_assets/final_candidates/object_B_apple_A1b_step1000_thr10/evidence/mesh_probe_textured/object_B_apple_A1b_step1000_thr10_textured_view_00.png",
        "outputs/aigc_assets/final_candidates/object_B_apple_A1b_step1000_thr10/evidence/mesh_probe_textured/object_B_apple_A1b_step1000_thr10_textured_view_05.png",
    ]

    pad = 18
    header_h = 58
    row_gap = 18
    label_w = 260
    cell_w = 560
    normal_h = 190
    mesh_h = 320
    columns = ["view 0", "view 5"]
    width = label_w + len(columns) * cell_w + (len(columns) + 2) * pad
    height = header_h + len(rows) * (normal_h + row_gap) + mesh_h + row_gap + pad
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    x0 = label_w + 2 * pad
    for i, col in enumerate(columns):
        draw.text((x0 + i * cell_w + 10, 16), col, font=FONT_BOLD, fill=TEXT)
    draw.line((pad, header_h - 8, width - pad, header_h - 8), fill=LINE, width=2)

    y = header_h
    for label, sources in rows:
        draw.text((pad, y + 12), label, font=FONT_BOLD, fill=TEXT)
        for i, src in enumerate(sources):
            panel = fit(img(src), cell_w - 10, normal_h)
            x = x0 + i * cell_w
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell_w - 10, y + normal_h), outline=LINE, width=1)
        y += normal_h + row_gap

    draw.text((pad, y + 12), "textured mesh", font=FONT_BOLD, fill=TEXT)
    for i, src in enumerate(mesh_sources):
        source = img(src)
        panel = fit(source, cell_w - 10, mesh_h, bg=corner_background(source))
        x = x0 + i * cell_w
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + cell_w - 10, y + mesh_h), outline=LINE, width=1)

    canvas.save(ASSET_DIR / "object_B_apple_clean.png", quality=95)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    make_early_targets()
    make_dreamfusion_rubik()
    make_magic3d_blackening()
    make_geometry_prior()
    make_apple()


if __name__ == "__main__":
    main()
