from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_SMALL = load_font(18)
FONT = load_font(22)
FONT_BOLD = load_bold_font(22)
FONT_TITLE = load_font(32)
FONT_TITLE_BOLD = load_bold_font(32)
BLACK = (28, 30, 34)
GRAY = (96, 101, 110)
LIGHT = (226, 230, 236)
BLUE = (37, 99, 235)
GREEN = (16, 141, 90)
RED = (220, 38, 38)
AMBER = (217, 119, 6)
PURPLE = (124, 58, 237)
TEAL = (13, 148, 136)


def rel(path: str) -> Path:
    return ROOT / path


def text_size(draw: ImageDraw.ImageDraw, text: str, font=FONT) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(draw: ImageDraw.ImageDraw, rect, text: str, fill=BLACK, font=FONT) -> None:
    x0, y0, x1, y1 = rect
    w, h = text_size(draw, text, font)
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2), text, fill=fill, font=font)


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        if len(candidate) <= width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def wrap_text_px(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    fill=GRAY,
    font=FONT_SMALL,
    line_gap: int = 5,
) -> int:
    x, y = xy
    for line in wrap_text_px(draw, text, font, max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += text_size(draw, line, font)[1] + line_gap
    return y


def draw_card(draw: ImageDraw.ImageDraw, rect, title: str, fill=(255, 255, 255), outline=LIGHT) -> None:
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=outline, width=1)
    x0, y0, _, _ = rect
    draw.text((x0 + 18, y0 + 16), title, fill=BLACK, font=FONT_BOLD)


def copy_asset(src: str, dst_name: str) -> None:
    s = rel(src)
    d = OUT / dst_name
    if not s.exists():
        print(f"missing: {src}")
        return
    shutil.copy2(s, d)
    print(f"copied: {dst_name}")


def contain(im: Image.Image, size: tuple[int, int], bg=(255, 255, 255)) -> Image.Image:
    im = im.convert("RGB")
    tw, th = size
    scale = min(tw / im.width, th / im.height)
    nw = max(1, int(im.width * scale))
    nh = max(1, int(im.height * scale))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    resized = im.resize((nw, nh), resample)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def make_contact_sheet(
    sources: list[str],
    dst_name: str,
    labels: list[str] | None = None,
    cols: int = 4,
    thumb: tuple[int, int] = (360, 240),
    title: str | None = None,
) -> None:
    paths = [rel(s) for s in sources]
    images = []
    valid_labels = []
    for i, p in enumerate(paths):
        if not p.exists():
            print(f"missing sheet input: {p}")
            continue
        images.append(contain(Image.open(p), thumb))
        valid_labels.append(labels[i] if labels and i < len(labels) else p.stem)
    if not images:
        return
    rows = math.ceil(len(images) / cols)
    pad = 16
    label_h = 34
    title_h = 52 if title else 0
    w = cols * thumb[0] + (cols + 1) * pad
    h = rows * (thumb[1] + label_h) + (rows + 1) * pad + title_h
    canvas = Image.new("RGB", (w, h), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((pad, 12), title, fill=BLACK, font=FONT_TITLE)
    y0 = pad + title_h
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (thumb[0] + pad)
        y = y0 + r * (thumb[1] + label_h + pad)
        canvas.paste(im, (x, y))
        draw.rectangle((x, y, x + thumb[0], y + thumb[1]), outline=(210, 215, 222), width=1)
        draw.text((x + 4, y + thumb[1] + 8), valid_labels[idx][:55], fill=GRAY, font=FONT_SMALL)
    canvas.save(OUT / dst_name, quality=92)
    print(f"sheet: {dst_name}")


def source_path(path: str) -> Path:
    p = rel(path)
    return p if p.exists() else Path(path)


def load_panel(path: str, size: tuple[int, int], bg=(255, 255, 255)) -> Image.Image:
    p = source_path(path)
    if not p.exists():
        canvas = Image.new("RGB", size, (255, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=RED, width=2)
        draw_wrapped(draw, (12, 18), f"Missing: {path}", size[0] - 24, fill=RED, font=FONT_SMALL)
        return canvas
    return contain(Image.open(p), size, bg=bg)


def make_labeled_grid(
    dst_name: str,
    title: str,
    subtitle: str,
    col_labels: list[str],
    rows: list[dict],
    thumb: tuple[int, int] = (380, 145),
    label_w: int = 430,
    footer: str | None = None,
) -> None:
    pad = 18
    top_h = 128
    col_h = 42
    row_h = max(thumb[1] + 36, 150)
    footer_h = 144 if footer else 24
    w = label_w + len(col_labels) * thumb[0] + (len(col_labels) + 2) * pad
    h = top_h + col_h + len(rows) * row_h + footer_h
    canvas = Image.new("RGB", (w, h), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 22), title, fill=BLACK, font=FONT_TITLE_BOLD)
    draw_wrapped(draw, (pad, 66), subtitle, w - 2 * pad, fill=GRAY, font=FONT, line_gap=5)

    x0 = pad
    y0 = top_h
    draw.rectangle((x0, y0, x0 + label_w, y0 + col_h), fill=(226, 232, 240), outline=LIGHT)
    draw.text((x0 + 12, y0 + 12), "Experiment", fill=BLACK, font=FONT_BOLD)
    x = x0 + label_w + pad
    for label in col_labels:
        draw.rectangle((x, y0, x + thumb[0], y0 + col_h), fill=(226, 232, 240), outline=LIGHT)
        draw_centered(draw, (x, y0, x + thumb[0], y0 + col_h), label, fill=BLACK, font=FONT_BOLD)
        x += thumb[0] + pad

    y = y0 + col_h
    for idx, row in enumerate(rows):
        fill = (255, 255, 255) if idx % 2 == 0 else (248, 250, 252)
        draw.rectangle((x0, y, w - pad, y + row_h), fill=fill, outline=LIGHT)
        accent = row.get("accent", BLUE)
        draw.rectangle((x0, y, x0 + 8, y + row_h), fill=accent)
        draw.text((x0 + 16, y + 14), row["label"], fill=BLACK, font=FONT_BOLD)
        draw_wrapped(draw, (x0 + 16, y + 46), row.get("note", ""), label_w - 34, fill=GRAY, font=FONT_SMALL, line_gap=4)
        x = x0 + label_w + pad
        for src in row["sources"]:
            panel = load_panel(src, thumb, bg=(250, 250, 250))
            canvas.paste(panel, (x, y + 14))
            draw.rectangle((x, y + 14, x + thumb[0], y + 14 + thumb[1]), outline=(204, 211, 221), width=1)
            x += thumb[0] + pad
        y += row_h

    if footer:
        draw.rounded_rectangle((pad, y + 18, w - pad, h - 24), radius=10, fill=(255, 251, 235), outline=AMBER)
        draw_wrapped(draw, (pad + 18, y + 36), footer, w - 2 * pad - 36, fill=BLACK, font=FONT, line_gap=7)
    canvas.save(OUT / dst_name, quality=92)
    print(f"sheet: {dst_name}")


def read_log_text(path: str) -> str:
    return rel(path).read_text(errors="ignore").replace("\r", "\n")


def parse_2dgs(path: str) -> dict:
    text = read_log_text(path)
    train = []
    for m in re.finditer(
        r"\[ITER\s+(\d+)\]\s+Loss=([0-9.]+)\s+distort=([0-9.]+)\s+normal=([0-9.]+)\s+Points=(\d+)",
        text,
    ):
        train.append(
            {
                "iter": int(m.group(1)),
                "loss": float(m.group(2)),
                "distort": float(m.group(3)),
                "normal": float(m.group(4)),
                "points": int(m.group(5)),
            }
        )
    if not train:
        for m in re.finditer(
            r"Training progress:.*?\|\s*(\d+)/30000\s+\[.*?Loss=([0-9.]+),\s+distort=([0-9.]+),\s+normal=([0-9.]+),\s+Points=(\d+)",
            text,
        ):
            train.append(
                {
                    "iter": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "distort": float(m.group(3)),
                    "normal": float(m.group(4)),
                    "points": int(m.group(5)),
                }
            )
    evals = []
    for m in re.finditer(r"\[ITER\s+(\d+)\]\s+Evaluating\s+\w+:\s+L1\s+([0-9.eE+-]+)\s+PSNR\s+([0-9.eE+-]+)", text):
        evals.append({"iter": int(m.group(1)), "l1": float(m.group(2)), "psnr": float(m.group(3))})
    return {"train": train, "eval": evals}


def parse_csv_series(path: str, columns: list[str]) -> dict[str, tuple[list[float], list[float]]]:
    out = {c: ([], []) for c in columns}
    with rel(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_raw = row.get("step", "")
            if not step_raw:
                continue
            try:
                step = float(step_raw)
            except ValueError:
                continue
            for c in columns:
                val = row.get(c, "")
                if val == "":
                    continue
                try:
                    y = float(val)
                except ValueError:
                    continue
                out[c][0].append(step)
                out[c][1].append(y)
    return out


def moving_average(values: list[float], radius: int = 3) -> list[float]:
    if len(values) < 3:
        return values
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def downsample(xs: list[float], ys: list[float], max_n: int = 900) -> tuple[list[float], list[float]]:
    if len(xs) <= max_n:
        return xs, ys
    stride = math.ceil(len(xs) / max_n)
    return xs[::stride], ys[::stride]


def normalize_series(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
    if not pairs:
        return [], []
    base = next((abs(y) for _, y in pairs if abs(y) > 1e-12), 1.0)
    return [x for x, _ in pairs], [y / base for _, y in pairs]


def draw_plot(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    series: list[tuple[str, list[float], list[float], tuple[int, int, int]]],
    title: str,
    xlabel: str,
    ylabel: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, fill=(255, 255, 255), outline=LIGHT)
    plot = (x0 + 74, y0 + 54, x1 - 28, y1 - 58)
    px0, py0, px1, py1 = plot
    draw.text((x0 + 16, y0 + 14), title, fill=BLACK, font=FONT)
    all_x = [x for _, xs, _, _ in series for x in xs]
    all_y = [y for _, _, ys, _ in series for y in ys if math.isfinite(y)]
    if not all_x or not all_y:
        draw.text((px0, py0), "No data", fill=RED, font=FONT)
        return
    xmin, xmax = min(all_x), max(all_x)
    ymin = min(all_y) if y_min is None else y_min
    ymax = max(all_y) if y_max is None else y_max
    if abs(ymax - ymin) < 1e-9:
        ymax += 1
        ymin -= 1
    ypad = (ymax - ymin) * 0.08
    ymin -= ypad
    ymax += ypad
    draw.line((px0, py1, px1, py1), fill=GRAY, width=1)
    draw.line((px0, py0, px0, py1), fill=GRAY, width=1)
    for i in range(5):
        t = i / 4
        x = px0 + (px1 - px0) * t
        y = py1 - (py1 - py0) * t
        draw.line((x, py1, x, py1 + 4), fill=GRAY)
        draw.line((px0 - 4, y, px0, y), fill=GRAY)
        xv = xmin + (xmax - xmin) * t
        yv = ymin + (ymax - ymin) * t
        draw.text((x - 26, py1 + 12), f"{xv:.0f}", fill=GRAY, font=FONT_SMALL)
        draw.text((x0 + 10, y - 9), f"{yv:.2g}", fill=GRAY, font=FONT_SMALL)
    draw.text(((px0 + px1) / 2 - 35, y1 - 32), xlabel, fill=GRAY, font=FONT_SMALL)

    def project(x: float, y: float) -> tuple[int, int]:
        xx = px0 + (x - xmin) / (xmax - xmin + 1e-12) * (px1 - px0)
        yy = py1 - (y - ymin) / (ymax - ymin + 1e-12) * (py1 - py0)
        return int(xx), int(yy)

    leg_x, leg_y = px0 + 6, py0 + 8
    for label, xs, ys, color in series:
        points = [project(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        elif len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.line((leg_x, leg_y + 10, leg_x + 28, leg_y + 10), fill=color, width=4)
        draw.text((leg_x + 38, leg_y), label, fill=BLACK, font=FONT_SMALL)
        leg_y += 25


def make_training_curves() -> None:
    a = parse_2dgs("outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/train.log")
    d = parse_2dgs("outputs/reconstruction_2dgs/background_garden/2dgs_final_r4_30k_attempt002_20260610-0232/logs/train.log")
    b = parse_csv_series(
        "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if-apple/A1bappleStem_s7@20260614-A1bappleStem-farcam-s7/csv_logs/version_0/metrics.csv",
        ["train/loss_sds", "train/loss_sparsity", "train/loss_opaque"],
    )
    c = parse_csv_series(
        "object_C_rubiks_cube/rubiks_cube_try16_try13_resume1000to2000_s008_o004_fovy25@20260612-221639/csv_logs/version_0/metrics.csv",
        ["train/loss_rgb", "train/loss_mask", "train/loss_sd_3d", "train/loss_sparsity"],
    )

    canvas = Image.new("RGB", (1800, 1200), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 18), "Training telemetry exported from local logs/CSV scalars", fill=BLACK, font=FONT_TITLE)

    a_x = [r["iter"] for r in a["train"]]
    a_y = moving_average([r["loss"] for r in a["train"]], 4)
    d_x = [r["iter"] for r in d["train"]]
    d_y = moving_average([r["loss"] for r in d["train"]], 4)
    a_x, a_y = downsample(a_x, a_y)
    d_x, d_y = downsample(d_x, d_y)

    draw_plot(
        draw,
        (28, 54, 878, 565),
        [("Object A 2DGS loss", a_x, a_y, BLUE), ("Garden D 2DGS loss", d_x, d_y, GREEN)],
        "2DGS reconstruction loss",
        "iteration",
        "loss",
        y_min=0,
    )

    psnr_series = []
    if a["eval"]:
        psnr_series.append(("Object A PSNR", [e["iter"] for e in a["eval"]], [e["psnr"] for e in a["eval"]], BLUE))
    if d["eval"]:
        psnr_series.append(("Garden D PSNR", [e["iter"] for e in d["eval"]], [e["psnr"] for e in d["eval"]], GREEN))
    draw_plot(draw, (922, 54, 1772, 565), psnr_series, "2DGS evaluation PSNR", "iteration", "PSNR")

    b_series = []
    for name, color, short in [
        ("train/loss_sds", RED, "SDS"),
        ("train/loss_sparsity", TEAL, "sparsity"),
        ("train/loss_opaque", AMBER, "opacity"),
    ]:
        xs, ys = normalize_series(*b[name])
        xs, ys = downsample(xs, moving_average(ys, 8))
        b_series.append((short, xs, ys, color))
    draw_plot(draw, (28, 610, 878, 1120), b_series, "Object B Magic3D IF losses", "step", "normalized")

    c_series = []
    for name, color, short in [
        ("train/loss_rgb", BLUE, "RGB"),
        ("train/loss_mask", GREEN, "mask"),
        ("train/loss_sd_3d", RED, "Zero123"),
        ("train/loss_sparsity", PURPLE, "sparsity"),
    ]:
        xs, ys = normalize_series(*c[name])
        xs, ys = downsample(xs, moving_average(ys, 8))
        c_series.append((short, xs, ys, color))
    draw_plot(draw, (922, 610, 1772, 1120), c_series, "Object C Magic123 losses", "step", "normalized")

    canvas.save(OUT / "training_curves.png")
    summary = {
        "object_a_2dgs": {
            "final_loss": a["train"][-1]["loss"],
            "final_normal": a["train"][-1]["normal"],
            "final_points": a["train"][-1]["points"],
            "eval": a["eval"],
        },
        "garden_d_2dgs": {
            "final_loss": d["train"][-1]["loss"],
            "final_normal": d["train"][-1]["normal"],
            "final_points": d["train"][-1]["points"],
            "eval": d["eval"],
        },
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print("plot: training_curves.png")


def eval_start_end(run: dict) -> tuple[dict, dict] | None:
    evals = run.get("eval", [])
    if len(evals) < 2:
        return None
    return evals[0], evals[-1]


def format_count(n: float | int) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def draw_value_table(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    columns: list[tuple[str, int]],
    rows: list[list[str]],
    header_fill=(241, 245, 249),
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, fill=(255, 255, 255), outline=LIGHT)
    h_header = 48
    draw.rectangle((x0, y0, x1, y0 + h_header), fill=header_fill, outline=LIGHT)
    x = x0
    for name, width in columns:
        draw.text((x + 10, y0 + 14), name, fill=BLACK, font=FONT_BOLD)
        draw.line((x, y0, x, y1), fill=LIGHT)
        x += width
    draw.line((x1, y0, x1, y1), fill=LIGHT)
    row_h = (y1 - y0 - h_header) / max(1, len(rows))
    for r, row in enumerate(rows):
        yy = int(y0 + h_header + r * row_h)
        draw.line((x0, yy, x1, yy), fill=LIGHT)
        x = x0
        for c, cell in enumerate(row):
            width = columns[c][1]
            draw_wrapped(draw, (x + 10, yy + 12), cell, width - 20, fill=GRAY if c else BLACK, font=FONT_SMALL)
            x += width
    draw.line((x0, y1, x1, y1), fill=LIGHT)


def make_2dgs_analytics() -> None:
    a = parse_2dgs("outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/train.log")
    d = parse_2dgs("outputs/reconstruction_2dgs/background_garden/2dgs_final_r4_30k_attempt002_20260610-0232/logs/train.log")

    canvas = Image.new("RGB", (1900, 1060), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), "2DGS reconstruction convergence and validation gains", fill=BLACK, font=FONT_TITLE_BOLD)
    draw_wrapped(
        draw,
        (36, 66),
        "The important evidence is not just that training ran: the 7k-to-30k validation gains show why the final 30k checkpoints were used for both the foreground reconstruction and the garden background.",
        1540,
        fill=GRAY,
        font=FONT,
    )

    a_x = [r["iter"] for r in a["train"]]
    a_y = moving_average([r["loss"] for r in a["train"]], 4)
    d_x = [r["iter"] for r in d["train"]]
    d_y = moving_average([r["loss"] for r in d["train"]], 4)
    a_x, a_y = downsample(a_x, a_y)
    d_x, d_y = downsample(d_x, d_y)
    draw_plot(
        draw,
        (36, 118, 1228, 605),
        [("Object A book", a_x, a_y, BLUE), ("Garden D background", d_x, d_y, GREEN)],
        "Training loss: compact object vs. full garden scene",
        "iteration",
        "loss",
        y_min=0,
    )

    draw_card(draw, (1260, 118, 1860, 605), "Validation improvement from 7k to 30k")
    rows = [("Object A", a, BLUE), ("Garden D", d, GREEN)]
    y = 190
    for label, run, color in rows:
        pair = eval_start_end(run)
        if pair is None:
            continue
        start, end = pair
        psnr_delta = end["psnr"] - start["psnr"]
        l1_delta = (end["l1"] - start["l1"]) / start["l1"] * 100
        draw.text((1290, y), label, fill=BLACK, font=FONT_BOLD)
        draw.text((1290, y + 36), "PSNR", fill=GRAY, font=FONT_SMALL)
        draw.line((1390, y + 46, 1640, y + 18), fill=color, width=4)
        draw.ellipse((1384, y + 40, 1396, y + 52), fill=color)
        draw.ellipse((1634, y + 12, 1646, y + 24), fill=color)
        draw.text((1660, y + 20), f"{start['psnr']:.2f} -> {end['psnr']:.2f} dB", fill=BLACK, font=FONT_SMALL)
        draw.text((1660, y + 48), f"+{psnr_delta:.2f} dB", fill=color, font=FONT_BOLD)
        draw.text((1290, y + 94), "L1", fill=GRAY, font=FONT_SMALL)
        draw.line((1390, y + 92, 1640, y + 118), fill=color, width=4)
        draw.ellipse((1384, y + 86, 1396, y + 98), fill=color)
        draw.ellipse((1634, y + 112, 1646, y + 124), fill=color)
        draw.text((1660, y + 92), f"{start['l1']:.4f} -> {end['l1']:.4f}", fill=BLACK, font=FONT_SMALL)
        draw.text((1660, y + 120), f"{l1_delta:.1f}%", fill=color, font=FONT_BOLD)
        y += 178

    table_rows = []
    for label, run, frames in [
        ("A book", a, "150 captured frames"),
        ("D garden", d, "185 dataset images"),
    ]:
        final = run["train"][-1]
        eval_end = run["eval"][-1]
        table_rows.append(
            [
                label,
                frames,
                format_count(final["points"]),
                f"{final['loss']:.5f}",
                f"{eval_end['psnr']:.2f}",
                f"{eval_end['l1']:.4f}",
            ]
        )
    draw_value_table(
        draw,
        (36, 650, 976, 860),
        [("Scene", 150), ("Input", 245), ("Final splats", 170), ("Loss", 135), ("PSNR", 120), ("L1", 120)],
        table_rows,
    )

    draw_card(draw, (1020, 650, 1860, 940), "Interpretation")
    takeaways = [
        "Both reconstructions still gain about 2.2 dB PSNR after the 7k checkpoint, so the 30k checkpoints are justified rather than arbitrary.",
        "Garden D uses roughly 9.8x more final Gaussians than Object A; raw loss values should therefore be read with scene complexity, not as a universal quality score.",
        "A and D differ in their final role: D remains the native Gaussian background, while A is converted through mesh extraction and re-sampled for scene fusion.",
    ]
    yy = 704
    for item in takeaways:
        draw.ellipse((1042, yy + 8, 1051, yy + 17), fill=BLUE)
        yy = draw_wrapped(draw, (1064, yy), item, 740, fill=GRAY, font=FONT_SMALL, line_gap=5) + 12

    canvas.save(OUT / "curves_2dgs_reconstruction.png", quality=92)
    print("plot: curves_2dgs_reconstruction.png")


def csv_last_values(path: str, columns: list[str]) -> tuple[float | None, dict[str, float]]:
    step: float | None = None
    values: dict[str, float] = {}
    with rel(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_step = float(row.get("step", ""))
            except ValueError:
                row_step = step
            for col in columns:
                raw = row.get(col, "")
                if raw == "":
                    continue
                try:
                    values[col] = float(raw)
                    step = row_step
                except ValueError:
                    continue
    return step, values


def make_aigc_analytics() -> None:
    b_csv = "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if-apple/A1bappleStem_s7@20260614-A1bappleStem-farcam-s7/csv_logs/version_0/metrics.csv"
    c_csv = "object_C_rubiks_cube/rubiks_cube_try16_try13_resume1000to2000_s008_o004_fovy25@20260612-221639/csv_logs/version_0/metrics.csv"
    b = parse_csv_series(b_csv, ["train/loss_sds", "train/loss_sparsity", "train/loss_opaque"])
    c = parse_csv_series(c_csv, ["train/loss_rgb", "train/loss_mask", "train/loss_sd_3d", "train/loss_sparsity"])
    b_step, b_final = csv_last_values(b_csv, ["train/loss_sds", "train/loss_sparsity", "train/loss_opaque", "train/grad_norm"])
    c_step, c_final = csv_last_values(
        c_csv,
        ["train/loss_rgb", "train/loss_mask", "train/loss_sd_3d", "train/loss_sparsity", "train/grad_norm", "train/grad_norm_3d"],
    )

    canvas = Image.new("RGB", (1900, 1060), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), "AIGC optimization traces: useful, but not sufficient alone", fill=BLACK, font=FONT_TITLE_BOLD)
    draw_wrapped(
        draw,
        (36, 66),
        "The diffusion-guided losses are diagnostic signals rather than direct geometry metrics. The report should pair these curves with multi-view validation renders and mesh cleanup evidence.",
        1540,
        fill=GRAY,
        font=FONT,
    )

    b_series = []
    for name, color, short in [
        ("train/loss_sds", RED, "SDS"),
        ("train/loss_sparsity", TEAL, "sparsity"),
        ("train/loss_opaque", AMBER, "opacity"),
    ]:
        xs, ys = normalize_series(*b[name])
        xs, ys = downsample(xs, moving_average(ys, 10))
        b_series.append((short, xs, ys, color))
    draw_plot(
        draw,
        (36, 118, 930, 650),
        b_series,
        "Object B text-to-3D apple, normalized losses",
        "step",
        "relative to first nonzero value",
    )

    c_series = []
    for name, color, short in [
        ("train/loss_rgb", BLUE, "RGB"),
        ("train/loss_mask", GREEN, "mask"),
        ("train/loss_sd_3d", RED, "Zero123"),
        ("train/loss_sparsity", PURPLE, "sparsity"),
    ]:
        xs, ys = normalize_series(*c[name])
        xs, ys = downsample(xs, moving_average(ys, 10))
        c_series.append((short, xs, ys, color))
    draw_plot(
        draw,
        (970, 118, 1860, 650),
        c_series,
        "Object C single-image Magic123 resume, normalized losses",
        "step",
        "relative to first nonzero value",
    )

    draw_card(draw, (36, 700, 930, 920), "Text-to-3D conclusion for Object B", fill=(255, 251, 242), outline=AMBER)
    b_lines = [
        f"Final scalar checkpoint: step {b_step:.0f}, SDS {b_final.get('train/loss_sds', 0):.2f}, sparsity {b_final.get('train/loss_sparsity', 0):.3f}, opacity {b_final.get('train/loss_opaque', 0):.4f}.",
        "The selected apple works because the prompt target has a compact organic prior; earlier duck/cone trials exposed semantic drift and local-detail failures.",
        "The loss curve is therefore used as optimization evidence, while final acceptance comes from textured orbit probes and Gaussian fusion renders.",
    ]
    yy = 755
    for line in b_lines:
        yy = draw_wrapped(draw, (60, yy), line, 820, fill=GRAY, font=FONT_SMALL, line_gap=5) + 12

    draw_card(draw, (970, 700, 1860, 920), "Single-image conclusion for Object C", fill=(245, 243, 255), outline=PURPLE)
    c_lines = [
        f"Final scalar checkpoint: step {c_step:.0f}, RGB {c_final.get('train/loss_rgb', 0):.4f}, mask {c_final.get('train/loss_mask', 0):.4f}, Zero123 {c_final.get('train/loss_sd_3d', 0):.2f}.",
        "RGB/mask terms can improve while hard-surface topology remains rounded or warped; this explains why the Rubik asset still needs a limitations discussion.",
        "The valuable result is the trade-off: single-view guidance preserves front-view appearance but cannot fully determine unseen cube sides.",
    ]
    yy = 755
    for line in c_lines:
        yy = draw_wrapped(draw, (994, yy), line, 820, fill=GRAY, font=FONT_SMALL, line_gap=5) + 12

    canvas.save(OUT / "curves_aigc_optimization.png", quality=92)
    print("plot: curves_aigc_optimization.png")


def draw_quality_chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, color) -> None:
    x, y = xy
    w = text_size(draw, label, FONT_SMALL)[0] + 26
    draw.rounded_rectangle((x, y, x + w, y + 30), radius=15, fill=color, outline=color)
    draw_centered(draw, (x, y, x + w, y + 30), label, fill=(255, 255, 255), font=FONT_SMALL)


def make_method_comparison_matrix() -> None:
    canvas = Image.new("RGB", (1900, 1030), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), "Quality comparison across the three required generation routes", fill=BLACK, font=FONT_TITLE_BOLD)
    draw_wrapped(
        draw,
        (36, 66),
        "This figure separates what each route is good at from what it costs. The report can use it to satisfy the geometry, texture, and computation comparison requirement without relying on subjective screenshots alone.",
        1600,
        fill=GRAY,
        font=FONT,
    )

    x0, y0 = 48, 126
    columns = [
        ("Route", 250),
        ("Geometry", 275),
        ("Texture detail", 275),
        ("Cost / time", 250),
        ("Representation gap", 325),
        ("Final interpretation", 425),
    ]
    rows = [
        [
            "Multi-view 2DGS\nA book / D garden",
            ("High", GREEN, "Multi-view camera constraints recover stable spatial layout; A/D reached 28.57/28.88 dB PSNR at 30k."),
            ("High", GREEN, "Photometric supervision preserves real object/background appearance better than generative priors."),
            ("High", RED, "Requires capture or dataset images, COLMAP, and 30k 2DGS optimization."),
            ("Medium", AMBER, "D is already Gaussian; A needed mesh extraction, cleanup, and re-sampling."),
            "Best for real captured content. Provides the physically grounded background and the real-object branch.",
        ],
        [
            "Text-to-3D\nMagic3D / IF",
            ("Medium", AMBER, "Good semantic shape for compact objects; weak local bindings caused duck/cone failures."),
            ("Medium", AMBER, "Prompt-guided texture is plausible but not a measured reconstruction of a real object."),
            ("Medium", AMBER, "Apple run used 3000 steps; additional time was spent on prompt and seed selection."),
            ("Medium", AMBER, "Exports textured mesh, then must be sampled into Gaussian records."),
            "Best for adding an object that need not match a real capture. Apple selected because it matches the prior.",
        ],
        [
            "Single-image 3D\nMagic123 / Zero123",
            ("Medium-low", RED, "Front view is constrained, but unseen sides remain ambiguous; Rubik topology stayed rounded/warped."),
            ("Medium", AMBER, "Input colors and stickers transfer partly, but texture can fake geometry."),
            ("Medium", AMBER, "Try16 resumed 1000->2000 steps after many failed candidates and input preprocessing trials."),
            ("Medium", AMBER, "Cleaned OBJ/MTL must be converted into table-aligned Gaussians."),
            "Best as a stress test. The final Rubik is useful precisely because its residual artifacts reveal the route's limits.",
        ],
    ]

    header_h = 62
    row_h = 250
    total_w = sum(w for _, w in columns)
    draw.rectangle((x0, y0, x0 + total_w, y0 + header_h), fill=(226, 232, 240), outline=LIGHT)
    x = x0
    for name, width in columns:
        draw.text((x + 12, y0 + 20), name, fill=BLACK, font=FONT_BOLD)
        draw.line((x, y0, x, y0 + header_h + row_h * len(rows)), fill=LIGHT)
        x += width
    draw.line((x0 + total_w, y0, x0 + total_w, y0 + header_h + row_h * len(rows)), fill=LIGHT)

    y = y0 + header_h
    for ridx, row in enumerate(rows):
        fill = (255, 255, 255) if ridx % 2 == 0 else (248, 250, 252)
        draw.rectangle((x0, y, x0 + total_w, y + row_h), fill=fill, outline=LIGHT)
        x = x0
        for cidx, cell in enumerate(row):
            width = columns[cidx][1]
            if cidx == 0:
                yy = y + 24
                for line in cell.split("\n"):
                    draw.text((x + 12, yy), line, fill=BLACK if yy == y + 24 else GRAY, font=FONT_BOLD if yy == y + 24 else FONT_SMALL)
                    yy += 31
            elif isinstance(cell, tuple):
                chip, color, desc = cell
                draw_quality_chip(draw, (x + 12, y + 24), chip, color)
                draw_wrapped(draw, (x + 12, y + 68), desc, width - 28, fill=GRAY, font=FONT_SMALL, line_gap=5)
            else:
                draw_wrapped(draw, (x + 12, y + 24), cell, width - 28, fill=GRAY, font=FONT_SMALL, line_gap=5)
            x += width
        y += row_h

    canvas.save(OUT / "method_comparison_matrix.png", quality=92)
    print("plot: method_comparison_matrix.png")


def make_gaussian_fusion_accounting() -> None:
    meta = json.loads(
        rel("outputs/gaussian_fusion/task1_abc_mesh_gaussians_v005_a_front_out_c_yaw_m90/metadata/task1_gaussian_fusion.json").read_text()
    )
    base = meta["base_vertex_count"]
    objects = meta["objects"]
    total = meta["merged_vertex_count"]
    canvas = Image.new("RGB", (1900, 960), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), "Gaussian-level fusion accounting", fill=BLACK, font=FONT_TITLE_BOLD)
    draw_wrapped(
        draw,
        (36, 66),
        "The final scene is a single 2DGS-readable Gaussian PLY. A/B/C are not overlaid in image space; they are sampled from meshes, encoded as Gaussian records, appended to D, and rendered by the same 2DGS renderer.",
        1650,
        fill=GRAY,
        font=FONT,
    )

    draw_card(draw, (50, 130, 1260, 410), "Merged point-cloud composition")
    bar = (90, 220, 1215, 282)
    x0, y0, x1, y1 = bar
    segments = [("D garden", base, GREEN)] + [(o["name"].replace("object_", ""), o["count"], c) for o, c in zip(objects, [BLUE, AMBER, PURPLE])]
    x = x0
    for name, count, color in segments:
        w = max(2, int((x1 - x0) * count / total))
        draw.rectangle((x, y0, min(x1, x + w), y1), fill=color)
        x += w
    draw.rectangle(bar, outline=BLACK, width=1)
    draw.text((90, 178), f"{format_count(base)} background Gaussians", fill=GREEN, font=FONT_BOLD)
    draw.text((760, 178), f"+ {format_count(meta['added_vertex_count'])} object Gaussians", fill=BLACK, font=FONT_BOLD)
    draw.text((90, 305), f"Total merged records: {format_count(total)}", fill=BLACK, font=FONT_BOLD)

    lx, ly = 90, 342
    for name, count, color in segments:
        pct = count / total * 100
        draw.rectangle((lx, ly, lx + 22, ly + 22), fill=color)
        draw.text((lx + 32, ly - 1), f"{name}: {format_count(count)} ({pct:.2f}%)", fill=GRAY, font=FONT_SMALL)
        lx += 260

    table_rows = []
    for obj in objects:
        if obj.get("target_dims"):
            size = "dims " + "x".join(f"{v:.2f}" for v in obj["target_dims"])
        elif obj.get("target_height"):
            size = f"height {obj['target_height']:.2f} m"
        else:
            size = f"side {obj.get('target_side', 0):.2f} m"
        orientation = []
        if obj.get("front_flip"):
            orientation.append("front flip")
        if abs(obj.get("yaw_delta_degrees", 0.0)) > 1e-6:
            orientation.append(f"yaw {obj['yaw_delta_degrees']:.0f} deg")
        table_rows.append(
            [
                obj["name"].replace("object_", "Object "),
                format_count(obj["count"]),
                size,
                f"alpha {obj['alpha']:.2f}",
                f"gain {obj['color_gain']:.2f}, gamma {obj['color_gamma']:.2f}",
                ", ".join(orientation) if orientation else "identity",
            ]
        )
    draw_value_table(
        draw,
        (50, 455, 1260, 725),
        [("Object", 150), ("Splats", 115), ("Scale target", 210), ("Opacity", 150), ("Color", 230), ("Orientation", 250)],
        table_rows,
    )

    draw_card(draw, (1320, 130, 1848, 725), "Renderer-compatible record")
    encoding_lines = [
        "position: sampled world-space surface point",
        "rotation: WXYZ quaternion from tangent/normal frame",
        "scale: log(world radius), estimated from mesh area/count",
        "opacity: logit(alpha), consumed through sigmoid",
        "color: SH DC coefficient f_dc = (rgb - 0.5) / 0.28209479",
        "f_rest: zero, leaving inserted assets with stable DC color",
    ]
    yy = 190
    for line in encoding_lines:
        draw.ellipse((1344, yy + 8, 1353, yy + 17), fill=TEAL)
        yy = draw_wrapped(draw, (1366, yy), line, 420, fill=GRAY, font=FONT_SMALL, line_gap=5) + 18
    draw.line((1340, yy + 4, 1818, yy + 4), fill=LIGHT, width=1)
    yy += 32
    draw_wrapped(
        draw,
        (1340, yy),
        "This is the technical core of the final report: implicit or mesh assets from threestudio are unified with the explicit 2DGS background by converting all foreground objects into the same Gaussian schema.",
        470,
        fill=BLACK,
        font=FONT,
        line_gap=7,
    )

    canvas.save(OUT / "gaussian_fusion_accounting.png", quality=92)
    print("plot: gaussian_fusion_accounting.png")


def make_failure_insight_map() -> None:
    canvas = Image.new("RGB", (1900, 1260), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), "Failure cases as evidence, not clutter", fill=BLACK, font=FONT_TITLE_BOLD)
    draw_wrapped(
        draw,
        (36, 66),
        "The failed branches are valuable because each one identifies a constraint that shaped the final design. This map turns scattered notes into a reviewer-readable chain of observations, diagnoses, and decisions.",
        1650,
        fill=GRAY,
        font=FONT,
    )

    columns = [
        ("Branch", 235),
        ("Observed failure", 415),
        ("Technical diagnosis", 500),
        ("Decision / report value", 600),
    ]
    rows = [
        [
            "B / duck seed sweep",
            "Many seeds produced incomplete bodies or view-specific artifacts.",
            "Text prior was semantically aligned but weakly constrained in 3D; seed sweeps alone do not guarantee geometry.",
            "Use orbit contact sheets as selection evidence; reject despite recognizable front-view semantics.",
        ],
        [
            "B / traffic cone",
            "Global silhouette was stable, but stripe/ring details and bottom geometry were unreliable.",
            "Text-to-3D preserves coarse object category better than local structured markings.",
            "Switch Object B to a compact apple target that better matches the generative prior.",
        ],
        [
            "B / apple final",
            "Textured mesh remained stable under orbit probes and later Gaussian insertion.",
            "Round organic objects are compatible with coarse Magic3D IF geometry.",
            "Selected as the text-generated object and converted from textured OBJ into Gaussians.",
        ],
        [
            "C / cat-box and yoyo",
            "Some runs had plausible silhouettes but lost target semantics or turned image edges into grooves.",
            "Single-image generation can overfit the visible view while hallucinating hidden sides.",
            "Make validation orbit and mesh cleanup mandatory before accepting any single-image asset.",
        ],
        [
            "C / Rubik baseline",
            "Sticker grid and cube edges became rounded or warped under default settings.",
            "Hard-surface regularity is a stress test for image-to-3D guidance.",
            "Tune FOV, guidance balance, sparsity, and resume length instead of accepting the first mesh.",
        ],
        [
            "C / try16 final",
            "One connected cleaned mesh, but top bulge and sticker distortion remain visible.",
            "Scalar losses can improve while hard-surface topology is still imperfect.",
            "Use as final Object C, with residual artifacts discussed as a route-specific limitation.",
        ],
        [
            "Fusion / Blender alpha",
            "Image overlay looked plausible but had no unified depth, occlusion, or renderer state.",
            "A visual composite is not a common 3D representation.",
            "Keep it as diagnostic evidence; make Gaussian-level fusion the report's technical contribution.",
        ],
        [
            "Fusion / v001-v005",
            "v001 hid objects due placement/orientation issues; later versions fixed visibility, yaw, and book front direction.",
            "Fixed-keyframe diagnostics expose coordinate-frame and surface-normal errors.",
            "Final v005 appends 125k object Gaussians to D and renders all assets with one 2DGS renderer.",
        ],
    ]

    x0, y0 = 52, 130
    header_h = 60
    row_h = 130
    total_w = sum(w for _, w in columns)
    draw.rectangle((x0, y0, x0 + total_w, y0 + header_h), fill=(226, 232, 240), outline=LIGHT)
    x = x0
    for label, width in columns:
        draw.text((x + 12, y0 + 20), label, fill=BLACK, font=FONT_BOLD)
        draw.line((x, y0, x, y0 + header_h + row_h * len(rows)), fill=LIGHT)
        x += width
    draw.line((x0 + total_w, y0, x0 + total_w, y0 + header_h + row_h * len(rows)), fill=LIGHT)

    y = y0 + header_h
    for ridx, row in enumerate(rows):
        fill = (255, 255, 255) if ridx % 2 == 0 else (248, 250, 252)
        draw.rectangle((x0, y, x0 + total_w, y + row_h), fill=fill, outline=LIGHT)
        accent = BLUE if row[0].startswith("B") else PURPLE if row[0].startswith("C") else TEAL
        draw.rectangle((x0, y, x0 + 8, y + row_h), fill=accent)
        x = x0
        for cidx, cell in enumerate(row):
            width = columns[cidx][1]
            draw_wrapped(
                draw,
                (x + 14, y + 18),
                cell,
                width - 28,
                fill=BLACK if cidx == 0 else GRAY,
                font=FONT_BOLD if cidx == 0 else FONT_SMALL,
                line_gap=5,
            )
            x += width
        y += row_h

    canvas.save(OUT / "failure_insight_map.png", quality=92)
    print("plot: failure_insight_map.png")


def make_object_b_rubik_dreamfusion_ablation() -> None:
    base = "outputs/aigc_assets/object_B_text_to_3d/final/dreamfusion-sd"

    def triplet(run: str, step: str) -> list[str]:
        return [f"{base}/{run}/save/it{step}-{v}.png" for v in (0, 5, 7)]

    rows = [
        {
            "label": "Try1 direct Rubik, g70",
            "note": "step 3000: enters a Rubik-like semantic basin, but grid and silhouette remain fuzzy.",
            "sources": triplet("rubiks_cube_try1_resume1000_to3000_solved_g70_val500@20260612-202637", "3000"),
            "accent": BLUE,
        },
        {
            "label": "Try2 geometry-first, g70",
            "note": "step 3000: stronger cube wording keeps colors but does not lock planar faces.",
            "sources": triplet("rubiks_cube_try2_geometry_first_clean_cube_g70_3000steps@20260612-214444", "3000"),
            "accent": PURPLE,
        },
        {
            "label": "Try3 geometry-first, g50",
            "note": "step 3000: lower guidance reduces dark clutter but weakens Rubik/grid semantics.",
            "sources": triplet("rubiks_cube_try3_geometry_first_clean_cube_g50_darkness_control_3000steps@20260612-214444", "3000"),
            "accent": AMBER,
        },
        {
            "label": "Try4A hardening, g60",
            "note": "step 2000: opacity/timestep/grad changes stabilize noise only partially.",
            "sources": triplet("rubiks_cube_try4A_combined_hardened_g60_3000steps@20260612-224539", "2000"),
            "accent": TEAL,
        },
        {
            "label": "Try4B hardening, g70",
            "note": "step 1500: higher guidance pushes toward foggy or semitransparent volume.",
            "sources": triplet("rubiks_cube_try4B_combined_hardened_g70_3000steps@20260612-224539", "1500"),
            "accent": RED,
        },
    ]
    make_labeled_grid(
        "object_B_rubik_dreamfusion_guidance.png",
        "Object B Rubik: DreamFusion prompt and guidance ablations",
        "Fixed views show that prompt and guidance changes move between semantic color, dark noise, and fuzzy volume, but none produces true hard cube geometry.",
        ["view 0", "view 5", "view 7"],
        rows,
        thumb=(390, 132),
        label_w=450,
        footer="Conclusion: text-only SDS can recognize the Rubik category, but it tends to satisfy the target through noisy color/grid appearance while the density field stays rounded, fuzzy, or semitransparent.",
    )


def make_object_b_rubik_magic3d_blackening() -> None:
    run = "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-sd/B0_s42@20260613-B0e/save"
    steps = ["500", "1000", "1500", "2000", "2500", "3000"]
    rows = []
    for view, label in [(0, "view 0"), (5, "view 5")]:
        rows.append(
            {
                "label": f"B0e white-cube {label}",
                "note": "same checkpoint series; after 2001 the validation becomes visibly darker.",
                "sources": [f"{run}/it{step}-{view}.png" for step in steps],
                "accent": GREEN if view == 0 else TEAL,
            }
        )
    make_labeled_grid(
        "object_B_rubik_magic3d_blackening.png",
        "Object B Rubik: Magic3D step timeline and blackening diagnosis",
        "The B0e white-cube run reveals why step comparisons must consider renderer/material state, not only model quality.",
        [f"{s} steps" for s in steps],
        rows,
        thumb=(245, 90),
        label_w=330,
        footer="Key finding: the late darkening appears after the Magic3D material regime changes around ambient_only_steps=2001. This is a logging/validation comparability issue as well as a model-quality symptom, so later shape-only runs used a high ambient-only override.",
    )


def make_object_b_rubik_geometry_prior_ablation() -> None:
    rows = [
        {
            "label": "B0IF prompt-only",
            "note": "IF improves compactness, but the object remains chamfered/rounded rather than six planar faces.",
            "sources": [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0if_s7@20260613-B0if/save/it3000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0if_s7@20260613-B0if/save/it3000-5.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0if_s7@20260613-B0if/save/it3000-7.png",
            ],
            "accent": BLUE,
        },
        {
            "label": "B0k cube prior",
            "note": "weak cube density prior reduces fog but still yields wedge/rounded block geometry.",
            "sources": [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0k_s7@20260613-B0k-cubeprior-s7/save/it2000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0k_s7@20260613-B0k-cubeprior-s7/save/it2000-5.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0k_s7@20260613-B0k-cubeprior-s7/save/it2000-7.png",
            ],
            "accent": TEAL,
        },
        {
            "label": "B1 refine threshold",
            "note": "DMTet/threshold hardens the wrong coarse density; debris changes but cylinder/tower remains.",
            "sources": [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-sd/B1_s7@20260613-B1t_t010/save/it100-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-sd/B1_s7@20260613-B1t_t010/save/it100-5.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-sd/B1_s7@20260613-B1t_t010/save/it100-7.png",
            ],
            "accent": RED,
        },
        {
            "label": "B2 fixed scaffold",
            "note": "fixed cube geometry can learn Rubik-like texture, proving the bottleneck is geometry.",
            "sources": [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-scaffold-if/B2if_s7@20260613-B2if/save/it1000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-scaffold-if/B2if_s7@20260613-B2if/save/it1000-5.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-scaffold-if/B2if_s7@20260613-B2if/save/it1000-7.png",
            ],
            "accent": GREEN,
        },
        {
            "label": "B2tIF trainable init",
            "note": "cube initialization helps the shape, but trainable geometry remains rough and non-planar.",
            "sources": [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3/save/it1000-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3/save/it1000-5.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3/save/it1000-7.png",
            ],
            "accent": PURPLE,
        },
        {
            "label": "B2tIF resumed",
            "note": "additional optimization smooths surfaces but starts rounding the hard cube again.",
            "sources": [
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3r2000/save/it1501-0.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3r2000/save/it1501-5.png",
                "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-refine-cube-init-trainable-if/B2tif_s7@20260613-B2tif3r2000/save/it1501-7.png",
            ],
            "accent": AMBER,
        },
    ]
    make_labeled_grid(
        "object_B_rubik_geometry_prior_ablation.png",
        "Object B Rubik: geometry-factorization ablations",
        "The B0/B1/B2 sequence isolates the failure source: texture can appear when cube geometry is provided, but prompt-only implicit volume does not reliably discover hard planar cube geometry.",
        ["view 0", "view 5", "view 7"],
        rows,
        thumb=(380, 128),
        label_w=450,
        footer="Conclusion: the Rubik failure was not merely prompt wording. Refine and threshold sweeps cannot recover cube geometry from a contaminated coarse field, while explicit cube priors improve shape but must be disclosed as geometry priors.",
    )


def make_object_c_rubik_camera_guidance_ablation() -> None:
    base = "outputs/aigc_assets/object_C_image_to_3d/rubiks_cube/final/coarse/magic123-coarse-sd"
    rows = [
        {
            "label": "Baseline try1",
            "note": "seed42, high sparsity; front recognizability hides wedge/twist/holes.",
            "sources": [
                f"{base}/rubiks_cube_seed42_gsd40_z75_1500review@20260611-175402/save/try_1/it1500-{v}.png"
                for v in (0, 5, 7)
            ],
            "accent": BLUE,
        },
        {
            "label": "Pose try2",
            "note": "elev 15, az 45, fovy 30; better depth, but side/back still inconsistent.",
            "sources": [
                f"{base}/rubiks_cube_try2_pose15az45_fovy30_gsd30_z10_sparse003_1000@20260611-183237/save/try_2/it1000-{v}.png"
                for v in (0, 5, 7)
            ],
            "accent": GREEN,
        },
        {
            "label": "Try7 Zero123-heavy",
            "note": "SD/Zero123 22/15; stronger view guidance does not dominate the bad geometry basin.",
            "sources": [f"{base}/rubiks_cube_try7_resume400_to1000_z15_sd22_colorpose@20260611-194300/save/it801-{v}.png" for v in (0, 5, 7)],
            "accent": AMBER,
        },
        {
            "label": "Try8 balanced",
            "note": "SD/Zero123 30/10 with visible-color prompt; better front/grid, still hidden-view defects.",
            "sources": [f"{base}/rubiks_cube_try8_try4weights_visiblecolors_1000@20260611-194301/save/it1000-{v}.png" for v in (0, 5, 7)],
            "accent": PURPLE,
        },
    ]
    make_labeled_grid(
        "object_C_rubik_camera_guidance_ablation.png",
        "Object C Rubik: camera pose and guidance balance",
        "For single-image Magic123, camera assumptions and SD/Zero123 balance change the optimization basin, but fixed side/rear views still reveal hidden geometry errors.",
        ["view 0 front", "view 5 side", "view 7 rear"],
        rows,
        thumb=(390, 132),
        label_w=450,
        footer="Conclusion: pose calibration is not cosmetic. The elevation/azimuth/FOV assumption changes depth extrapolation, while the best guidance balance was 30/10 rather than a more Zero123-heavy setting.",
    )


def make_object_c_rubik_closure_alpha_ablation() -> None:
    base = "outputs/aigc_assets/object_C_image_to_3d/rubiks_cube/final/coarse/magic123-coarse-sd"
    rows = [
        {
            "label": "Try11 seed 7",
            "note": "seed improves front/grid basin, but seed alone does not close side/rear geometry.",
            "sources": [f"{base}/rubiks_cube_try11_try8_seed7_sameparams_1000@20260612-201907/save/it1000-{v}.png" for v in (0, 5, 7)],
            "accent": BLUE,
        },
        {
            "label": "Try12 closure",
            "note": "fovy 25, sparsity .008, opacity .004; holes reduce but stickers soften.",
            "sources": [f"{base}/rubiks_cube_try12_try8_seed42_closedgeom_fovy25_1000@20260612-201908/save/it1000-{v}.png" for v in (0, 5, 7)],
            "accent": GREEN,
        },
        {
            "label": "Try13 seed + closure",
            "note": "combines seed 7 with closure recipe; good base, but body remains rounded/noisy.",
            "sources": [f"{base}/rubiks_cube_try13_seed7_try12closure_fovy25_s008_o004_1000@20260612-214923/save/it1000-{v}.png" for v in (0, 5, 7)],
            "accent": TEAL,
        },
        {
            "label": "Try14 over-closure",
            "note": "fovy 22, stronger smoothness/z variance; dents and softness become more visible.",
            "sources": [f"{base}/rubiks_cube_try14_seed7_strongclosure_fovy22_s005_o006_n1600_z75_1000@20260612-214923/save/it1000-{v}.png" for v in (0, 5, 7)],
            "accent": RED,
        },
        {
            "label": "Try15 solid alpha",
            "note": "solid-alpha input improves front readability and hole-like artifacts, not rear dents.",
            "sources": [f"{base}/rubiks_cube_try15_seed7_solidalpha_fovy25_s006_o005_1000@20260612-214924/save/it1000-{v}.png" for v in (0, 5, 7)],
            "accent": AMBER,
        },
        {
            "label": "Try16 final resume",
            "note": "resume try13 to 2000; harder body and grid, still top bulge and noisy texture.",
            "sources": [f"{base}/rubiks_cube_try16_try13_resume1000to2000_s008_o004_fovy25@20260612-221639/save/it1801-{v}.png" for v in (0, 5, 7)],
            "accent": PURPLE,
        },
    ]
    make_labeled_grid(
        "object_C_rubik_closure_alpha_resume.png",
        "Object C Rubik: seed, closure, alpha, and final resume",
        "These later ablations show the main tradeoff: closing holes and hardening the cube can soften stickers, create dents, or preserve residual rounded geometry.",
        ["view 0 front", "view 5 side", "view 7 rear"],
        rows,
        thumb=(380, 128),
        label_w=470,
        footer="Conclusion: the final try16 is a tradeoff, not a perfect reconstruction. It is the best fusion candidate because it is closed and mesh-exportable, while its artifacts remain important evidence about single-view ambiguity.",
    )


def make_object_c_rubik_step_diagnostics() -> None:
    base = "outputs/aigc_assets/object_C_image_to_3d/rubiks_cube/final/coarse/magic123-coarse-sd/rubiks_cube_try16_try13_resume1000to2000_s008_o004_fovy25@20260612-221639/save"
    steps = ["1001", "1201", "1401", "1601", "1801"]
    rows = []
    for view, label, color in [(0, "front view 0", BLUE), (5, "side view 5", GREEN), (7, "rear view 7", PURPLE)]:
        rows.append(
            {
                "label": label,
                "note": "same final-resume run; compare texture/grid sharpening against persistent shape defects.",
                "sources": [f"{base}/it{step}-{view}.png" for step in steps],
                "accent": color,
            }
        )
    make_labeled_grid(
        "object_C_rubik_try16_step_diagnostics.png",
        "Object C Rubik: try16 resume step diagnostics",
        "Unlike bad-geometry resumes, try16 starts from the acceptable try13 checkpoint; the timeline shows why extra optimization helped but could not erase all geometry artifacts.",
        [f"{s} steps" for s in steps],
        rows,
        thumb=(285, 95),
        label_w=360,
        footer="Conclusion: more steps help only when the starting geometry is already plausible. Here the grid/body become harder, but top bulge, rounded edges, and noisy side/rear normals persist.",
    )


def make_curated_reasoning_map() -> None:
    canvas = Image.new("RGB", (2300, 1420), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), "Curated diagnostic iteration map", fill=BLACK, font=FONT_TITLE_BOLD)
    draw_wrapped(
        draw,
        (36, 66),
        "The main report should not list every run. It should show the runs that changed the next design decision: problem, hypothesis, intervention, observation, and decision.",
        1900,
        fill=GRAY,
        font=FONT,
    )

    columns = [
        ("Branch", 250),
        ("Problem / hypothesis", 440),
        ("Intervention", 420),
        ("Observation", 520),
        ("Decision carried forward", 560),
    ]
    rows = [
        [
            "B duck",
            "A recognizable text target may look correct from one view but fail as a 3D asset.",
            "Long training and seed sweep; inspect orbit/fixed multi-view evidence.",
            "Beak/face semantics drift into bands, extra parts, or two-object splits.",
            "Adopt multi-view validation as a hard acceptance rule; reject single-view successes.",
        ],
        [
            "B traffic cone",
            "A simpler global shape may stabilize geometry, but local attributes may remain weak.",
            "Seed sweep, seed-7 resume, and targeted white-ring prompt.",
            "Cone body is stable, but the reflective band is weak; targeted local prompt damages global shape.",
            "Separate global geometry quality from local semantic binding; do not overfit one missing detail.",
        ],
        [
            "B Rubik B0/B1/B2",
            "Rubik failure may come from geometry, not lack of color/grid texture.",
            "Factorize into white-cube generation, DMTet refine, and cube scaffold/texture ablations.",
            "B0 gives rounded blocks; B1 hardens wrong base/cylinder; fixed scaffold can learn grid texture.",
            "Conclude prompt-only SDS struggles with strict planar cube geometry; use Rubik as stress-test evidence.",
        ],
        [
            "B apple final",
            "A rounded compact object may match the implicit-volume prior better than a hard cube.",
            "Magic3D-IF stem prompt, farther camera, restored sparsity/density settings, threshold-10 export.",
            "Step-1000 mesh keeps red body and stem while reducing bottom fog/base artifact.",
            "Select apple as final text-to-3D asset because the choice follows from earlier failure analysis.",
        ],
        [
            "C early targets",
            "Single-image guidance may preserve appearance but fail hidden geometry or topology.",
            "Try green container, cat-box, yoyo, geometry guide, and prompt/crop changes.",
            "Canister collapses in side opacity; cat loses identity; yoyo groove becomes texture rather than topology.",
            "Use Rubik as a compact hard-surface diagnostic and evaluate RGB, normal, and opacity together.",
        ],
        [
            "C Rubik",
            "Front-view recognizability can hide side/rear holes and soft cube geometry.",
            "Ablate pose/FOV, SD vs Zero123 guidance, seed, closure, solid alpha, and resume strategy.",
            "Closure reduces holes but softens stickers; over-closure causes dents; try16 is best but imperfect.",
            "Select try16 cleaned main component as final C, while explicitly reporting residual artifacts.",
        ],
        [
            "Fusion",
            "A visual overlay is not a unified 3D representation with the 2DGS garden.",
            "Compare Blender alpha, schema audit, synthetic Gaussian cube, and v001-v005 keyframe loop.",
            "Naive overlay lacks depth; first Gaussian merge had orientation/normal issues; v005 resolves placement.",
            "Final scene uses mesh-to-Gaussian sampling and one merged 2DGS renderer path.",
        ],
    ]

    x0, y0 = 36, 132
    header_h = 60
    row_h = 166
    total_w = sum(w for _, w in columns)
    draw.rectangle((x0, y0, x0 + total_w, y0 + header_h), fill=(226, 232, 240), outline=LIGHT)
    x = x0
    for title, width in columns:
        draw.text((x + 12, y0 + 20), title, fill=BLACK, font=FONT_BOLD)
        draw.line((x, y0, x, y0 + header_h + row_h * len(rows)), fill=LIGHT)
        x += width
    draw.line((x0 + total_w, y0, x0 + total_w, y0 + header_h + row_h * len(rows)), fill=LIGHT)

    colors = [BLUE, TEAL, PURPLE, GREEN, AMBER, RED, BLACK]
    y = y0 + header_h
    for ridx, row in enumerate(rows):
        fill = (255, 255, 255) if ridx % 2 == 0 else (248, 250, 252)
        draw.rectangle((x0, y, x0 + total_w, y + row_h), fill=fill, outline=LIGHT)
        draw.rectangle((x0, y, x0 + 8, y + row_h), fill=colors[ridx])
        x = x0
        for cidx, cell in enumerate(row):
            width = columns[cidx][1]
            draw_wrapped(
                draw,
                (x + 14, y + 18),
                cell,
                width - 28,
                fill=BLACK if cidx == 0 else GRAY,
                font=FONT_BOLD if cidx == 0 else FONT_SMALL,
                line_gap=5,
            )
            x += width
        y += row_h
    canvas.save(OUT / "curated_diagnostic_iteration_map.png", quality=92)
    print("plot: curated_diagnostic_iteration_map.png")


def make_object_b_target_reasoning() -> None:
    rows = [
        {
            "label": "Duck seed sweep",
            "note": "Problem: plausible toy semantics did not survive orbit validation. Lesson: local parts such as beak/face are weakly bound.",
            "sources": ["outputs/aigc_assets/object_B_text_to_3d/final/object_B_rubber_duck_seed_sweep_preview_contact_sheet.jpg"],
            "accent": BLUE,
        },
        {
            "label": "Traffic cone",
            "note": "Problem: global cone shape is easier than local white-ring material. Targeted white-ring prompts damaged the whole object.",
            "sources": ["outputs/previews/aigc_assets/object_B_text_to_3d/traffic_cone/object_B_traffic_cone_seed7_resume_it3000_views.jpg"],
            "accent": AMBER,
        },
        {
            "label": "Rubik stress test",
            "note": "Problem: strict hard-surface cube requires geometry and grid locking. Result: semantics appears before true planar geometry.",
            "sources": ["outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if/B0k_s7@20260613-B0k-cubeprior-s7/save/it2000-0.png"],
            "accent": PURPLE,
        },
        {
            "label": "Apple final",
            "note": "Decision: select a rounded compact object that matches the SDS prior while still requiring stem, texture, and mesh export quality.",
            "sources": [
                "outputs/aigc_assets/final_candidates/object_B_apple_A1b_step1000_thr10/evidence/mesh_probe_textured/object_B_apple_A1b_step1000_thr10_textured_view_00.png"
            ],
            "accent": GREEN,
        },
    ]
    make_labeled_grid(
        "object_B_target_reasoning.png",
        "Object B target selection as diagnostic reasoning",
        "The final apple is not a random replacement; it is the engineering decision implied by earlier duck, cone, and Rubik failures.",
        ["evidence"],
        rows,
        thumb=(650, 260),
        label_w=650,
        footer="Report use: show this as the compact overview, then deep-dive into Rubik B0/B1/B2 only because that branch best exposes the geometry bottleneck.",
    )


def make_object_c_target_reasoning() -> None:
    rows = [
        {
            "label": "Cat-box fallback",
            "note": "Problem: global body completeness can improve while identity-defining details such as face/ears disappear.",
            "sources": ["outputs/aigc_assets/final_candidates/object_C_cat_box_seed42_step1500/evidence/8view/object_C_cat_box_seed42_step1500_8view_contact_sheet.jpg"],
            "accent": BLUE,
        },
        {
            "label": "Yoyo topology cue",
            "note": "Problem: central groove becomes a visible RGB/normal line, not necessarily a real pinched opacity/geometry topology.",
            "sources": ["outputs/previews/aigc_assets/object_C_image_to_3d/green_yoyo/object_C_green_yoyo_seed42_it500_8view_contact_sheet.jpg"],
            "accent": AMBER,
        },
        {
            "label": "Rubik baseline",
            "note": "Problem: front-view recognizability hides side/rear holes, twist, and non-cube geometry.",
            "sources": ["outputs/previews/aigc_assets/object_C_image_to_3d/rubiks_cube/rubiks_cube_try2_step1000_contact_sheet.jpg"],
            "accent": PURPLE,
        },
        {
            "label": "Rubik try16 final",
            "note": "Decision: best tradeoff after pose, seed, closure, alpha, and resume ablations; still reported with limitations.",
            "sources": [
                "object_C_rubiks_cube/rubiks_cube_try16_try13_resume1000to2000_s008_o004_fovy25@20260612-221639/save/it1801-0.png"
            ],
            "accent": GREEN,
        },
    ]
    make_labeled_grid(
        "object_C_target_reasoning.png",
        "Object C target selection under single-image ambiguity",
        "Object C is framed as a diagnostic image-to-3D process: each target exposes a different single-view ambiguity.",
        ["evidence"],
        rows,
        thumb=(650, 260),
        label_w=650,
        footer="Report use: use this overview to justify why Rubik receives the deepest C-side ablation treatment, while cat/yoyo are summarized as targeted failure evidence.",
    )


def arrow(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color=GRAY) -> None:
    draw.line((a[0], a[1], b[0], b[1]), fill=color, width=3)
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    for da in (2.6, -2.6):
        x = b[0] - 12 * math.cos(angle + da)
        y = b[1] - 12 * math.sin(angle + da)
        draw.line((b[0], b[1], x, y), fill=color, width=3)


def box(draw: ImageDraw.ImageDraw, rect, title: str, lines: list[str], fill, outline) -> None:
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=outline, width=2)
    x0, y0, x1, _ = rect
    draw.text((x0 + 14, y0 + 12), title, fill=BLACK, font=FONT)
    y = y0 + 44
    for line in lines:
        for part in wrap_text(line, 31):
            draw.text((x0 + 14, y), part, fill=GRAY, font=FONT_SMALL)
            y += 24


def make_pipeline() -> None:
    canvas = Image.new("RGB", (1800, 760), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((38, 24), "Unified Gaussian Fusion Pipeline", fill=BLACK, font=FONT_TITLE)

    sources = [
        ((42, 90, 420, 195), "A: real book", ["video frames -> COLMAP -> 2DGS", "mesh extraction and cleanup"], (235, 244, 255), BLUE),
        ((42, 225, 420, 330), "B: text object", ["Magic3D coarse IF", "textured apple OBJ/MTL"], (255, 241, 242), RED),
        ((42, 360, 420, 465), "C: single image", ["Magic123 SD + Zero123", "cleaned Rubik OBJ/MTL"], (245, 243, 255), PURPLE),
        ((42, 495, 420, 600), "D: garden background", ["Mip-NeRF360 garden images", "2DGS Gaussian scene"], (236, 253, 245), GREEN),
    ]
    for r, title, lines, fill, outline in sources:
        box(draw, r, title, lines, fill, outline)

    box(
        draw,
        (548, 170, 930, 415),
        "Mesh-to-Gaussian conversion",
        [
            "sample colored mesh surfaces",
            "transform to garden table coordinates",
            "estimate triangle tangent and normal",
            "encode f_dc, opacity, scale, rotation",
            "A 50k + B 35k + C 40k splats",
        ],
        (255, 251, 235),
        AMBER,
    )
    box(
        draw,
        (1045, 205, 1395, 380),
        "Representation-level merge",
        [
            "append to D point_cloud.ply",
            "copy D cameras and cfg_args",
            "2,217,492 + 125,000 = 2,342,492 records",
        ],
        (240, 253, 250),
        TEAL,
    )
    box(
        draw,
        (1510, 225, 1760, 360),
        "One renderer",
        [
            "same 2DGS renderer",
            "fixed keyframes, then 240-frame trajectory",
            "final v005",
        ],
        (239, 246, 255),
        BLUE,
    )
    for y in (142, 278, 413):
        arrow(draw, (420, y), (548, 278), GRAY)
    arrow(draw, (420, 548), (1045, 292), GRAY)
    arrow(draw, (930, 292), (1045, 292), GRAY)
    arrow(draw, (1395, 292), (1510, 292), GRAY)
    draw.text((42, 675), "Key point: A/B/C meshes are intermediate sampling sources; the final scene is one merged 2DGS Gaussian representation.", fill=BLACK, font=FONT)
    canvas.save(OUT / "pipeline_unified_gaussian.png")
    print("plot: pipeline_unified_gaussian.png")


def main() -> None:
    copies = {
        "object_A_2dgs_8view.jpg": "outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/evidence/8view/object_A_2dgs_reconstruction_8view_contact_sheet.jpg",
        "object_A_nobg_8view.jpg": "outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/evidence/8view_no_background/object_A_2dgs_no_background_8view_report_grid_white.jpg",
        "object_A_clean_mesh.jpg": "outputs/reconstruction_2dgs/object_A_book/mesh_candidates_from_2dgs_res384/book_region_rank00_margin012/top_artifact_cleaning/targeted_cleanup_candidates/probe_combo_top_z88_white_y18_contact_sheet.jpg",
        "object_B_duck_seed_sweep.jpg": "outputs/aigc_assets/object_B_text_to_3d/final/object_B_rubber_duck_seed_sweep_preview_contact_sheet.jpg",
        "object_B_traffic_cone.jpg": "outputs/aigc_assets/final_candidates/object_B_traffic_cone_seed7_step3000/evidence/8view/object_B_traffic_cone_seed7_step3000_8view_orbit_contact_sheet.jpg",
        "object_C_cat_box.jpg": "outputs/aigc_assets/final_candidates/object_C_cat_box_seed42_step1500/evidence/8view/object_C_cat_box_seed42_step1500_8view_contact_sheet.jpg",
        "object_C_yoyo.jpg": "outputs/previews/aigc_assets/object_C_image_to_3d/green_yoyo/object_C_green_yoyo_seed42_it500_8view_contact_sheet.jpg",
        "object_C_rubik_baseline.jpg": "outputs/previews/aigc_assets/object_C_image_to_3d/rubiks_cube/rubiks_cube_try2_step1000_contact_sheet.jpg",
        "fusion_blender_alpha.png": "final_blender/composite_keyframes_contact_sheet.png",
        "fusion_synthetic_cube.png": "outputs/gaussian_fusion/task1_synthetic_magenta_cube_v001/keyframes_v001/keyframes_contact_sheet.png",
        "fusion_v001.png": "outputs/gaussian_fusion/task1_abc_mesh_gaussians_v001/keyframes_v001/keyframes_contact_sheet.png",
        "fusion_v002.png": "outputs/gaussian_fusion/task1_abc_mesh_gaussians_v002/keyframes_v001/keyframes_contact_sheet.png",
        "fusion_v004.png": "outputs/gaussian_fusion/task1_abc_mesh_gaussians_v004_c_yaw_m90/keyframes_v001/keyframes_contact_sheet.png",
        "fusion_v005_keyframes.png": "outputs/gaussian_fusion/task1_abc_mesh_gaussians_v005_a_front_out_c_yaw_m90/keyframes_v001/keyframes_contact_sheet.png",
        "fusion_v005_final.png": "outputs/gaussian_fusion/task1_abc_mesh_gaussians_v005_a_front_out_c_yaw_m90/traj/ours_30000/render_traj_contact_sheet.png",
    }
    for dst, src in copies.items():
        copy_asset(src, dst)

    make_contact_sheet(
        [
            f"outputs/aigc_assets/final_candidates/object_B_apple_A1b_step1000_thr10/evidence/mesh_probe_textured/object_B_apple_A1b_step1000_thr10_textured_view_{i:02d}.png"
            for i in range(8)
        ],
        "object_B_apple_final_views.png",
        [f"view {i}" for i in range(8)],
        cols=4,
        thumb=(340, 260),
        title="Object B final apple textured mesh",
    )
    make_contact_sheet(
        [
            f"object_C_rubiks_cube/rubiks_cube_try16_try13_resume1000to2000_s008_o004_fovy25@20260612-221639/save/it1801-{i}.png"
            for i in range(8)
        ],
        "object_C_try16_final_views.png",
        [f"view {i}" for i in range(8)],
        cols=4,
        thumb=(340, 260),
        title="Object C final try16 Rubik validation views",
    )
    make_contact_sheet(
        [
            "outputs/previews/aigc_assets/object_C_image_to_3d/rubiks_cube/object_C_rubiks_cube_rgba_preview.png",
            "outputs/previews/aigc_assets/object_C_image_to_3d/rubiks_cube/object_C_rubiks_cube_solidalpha_preview.png",
            "outputs/previews/aigc_assets/object_C_image_to_3d/rubiks_cube/object_C_rubiks_cube_geometry_guide_preview.png",
        ],
        "object_C_input_preprocessing.png",
        ["RGBA input", "solid alpha", "geometry guide"],
        cols=3,
        thumb=(400, 300),
        title="Object C input preprocessing ablation",
    )
    make_contact_sheet(
        [
            "outputs/gaussian_fusion/task1_abc_mesh_gaussians_v001/keyframes_v001/keyframes_contact_sheet.png",
            "outputs/gaussian_fusion/task1_abc_mesh_gaussians_v002/keyframes_v001/keyframes_contact_sheet.png",
            "outputs/gaussian_fusion/task1_abc_mesh_gaussians_v004_c_yaw_m90/keyframes_v001/keyframes_contact_sheet.png",
            "outputs/gaussian_fusion/task1_abc_mesh_gaussians_v005_a_front_out_c_yaw_m90/keyframes_v001/keyframes_contact_sheet.png",
        ],
        "fusion_iteration_grid.png",
        ["v001 normal sign issue", "v002 visible objects", "v004 C yaw -90", "v005 A front outward"],
        cols=1,
        thumb=(1200, 260),
        title="Fixed-keyframe diagnostic loop",
    )
    make_training_curves()
    make_2dgs_analytics()
    make_aigc_analytics()
    make_method_comparison_matrix()
    make_gaussian_fusion_accounting()
    make_failure_insight_map()
    make_object_b_rubik_dreamfusion_ablation()
    make_object_b_rubik_magic3d_blackening()
    make_object_b_rubik_geometry_prior_ablation()
    make_object_c_rubik_camera_guidance_ablation()
    make_object_c_rubik_closure_alpha_ablation()
    make_object_c_rubik_step_diagnostics()
    make_curated_reasoning_map()
    make_object_b_target_reasoning()
    make_object_c_target_reasoning()
    make_pipeline()


if __name__ == "__main__":
    main()
