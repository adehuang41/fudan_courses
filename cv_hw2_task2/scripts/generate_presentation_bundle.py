from __future__ import annotations

import csv
import json
import math
import textwrap
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
PPT_PATH = REPORT_DIR / "hw2_task2_presentation.pptx"
HTML_PATH = REPORT_DIR / "presentation_script.html"

SLIDE_W = 13.333
SLIDE_H = 7.5
EMU = 914400

BG = "F6F3EE"
PANEL = "FFFFFF"
ACCENT = "1F5A4E"
ACCENT_2 = "C8682A"
TEXT = "1E2430"
MUTED = "5B6577"
LIGHT = "E9E3D8"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yolov8m_stats() -> dict:
    rows = list(csv.DictReader((REPO_ROOT / "checkpoints/yolov8m_visdrone_640/results.csv").open()))
    best = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    last = rows[-1]
    return {
        "epochs_recorded": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "best_map50": float(best["metrics/mAP50(B)"]),
        "best_map": float(best["metrics/mAP50-95(B)"]),
        "last_epoch": int(float(last["epoch"])),
        "last_map50": float(last["metrics/mAP50(B)"]),
        "last_map": float(last["metrics/mAP50-95(B)"]),
    }


def load_conf_ablation() -> list[dict]:
    with (REPO_ROOT / "outputs/conf_ablation.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "conf": float(row["conf"]),
            "mean_dets_per_frame": float(row["mean_dets_per_frame"]),
            "unique_ids": int(row["unique_ids"]),
            "line_crossings": int(row["line_crossings"]),
        }
        for row in rows
    ]


def maybe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return load_json(path)


def load_yolov8m_video_results() -> dict:
    test_summary = maybe_load_json(REPO_ROOT / "outputs/yolov8m_compare/test_clip_best/line_count_summary.json")
    fudan_summary = maybe_load_json(REPO_ROOT / "outputs/yolov8m_compare/fudan_handan_best/line_count_summary.json")
    return {
        "ready": bool(test_summary and fudan_summary),
        "test_clip": test_summary,
        "fudan_handan": fudan_summary,
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, max_width: int, fill: str, font_obj, line_gap: int = 8) -> int:
    x, y = xy
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            if draw.textlength(trial, font=font_obj) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font_obj)
        y += font_obj.size + line_gap
    return y


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str = PANEL, outline: str = LIGHT) -> None:
    fill_val = fill if fill.startswith("#") else "#" + fill
    outline_val = outline if outline.startswith("#") else "#" + outline
    draw.rounded_rectangle(box, radius=28, fill=fill_val, outline=outline_val, width=2)


def make_model_compare(y8m: dict) -> Path:
    out = FIG_DIR / "model_compare.png"
    models = ["YOLOv8s", "YOLOv8m"]
    map50 = [0.412, y8m["best_map50"]]
    map95 = [0.246, y8m["best_map"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=160)
    colors = ["#1F5A4E", "#C8682A"]

    axes[0].bar(models, map50, color=colors, width=0.55)
    axes[0].set_title("VisDrone val mAP50")
    axes[0].set_ylim(0, 0.5)
    axes[0].grid(axis="y", alpha=0.2)
    for i, v in enumerate(map50):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    axes[1].bar(models, map95, color=colors, width=0.55)
    axes[1].set_title("VisDrone val mAP50-95")
    axes[1].set_ylim(0, 0.3)
    axes[1].grid(axis="y", alpha=0.2)
    for i, v in enumerate(map95):
        axes[1].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle("Detector Comparison: Baseline vs Stronger Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, facecolor="#f6f3ee", bbox_inches="tight")
    plt.close(fig)
    return out


def make_fudan_compare() -> Path:
    out = FIG_DIR / "fudan_compare.png"
    labels = ["GT", "640/0.25", "1280/0.10", "1280/0.15"]
    totals = [14, 9, 10, 9]
    cars = [11, 8, 7, 6]
    vans = [2, 0, 1, 1]
    buses = [1, 1, 1, 1]

    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=160)
    axes[0].bar(labels, totals, color=["#1F5A4E", "#7C8DA6", "#C8682A", "#B7825F"])
    axes[0].set_title("Total Crossings")
    axes[0].set_ylim(0, 16)
    axes[0].grid(axis="y", alpha=0.2)
    for i, v in enumerate(totals):
        axes[0].text(i, v + 0.3, str(v), ha="center", fontsize=10)

    width = 0.22
    xs = list(x)
    axes[1].bar([i - width for i in xs], cars, width=width, label="car", color="#1F5A4E")
    axes[1].bar(xs, vans, width=width, label="van", color="#C8682A")
    axes[1].bar([i + width for i in xs], buses, width=width, label="bus", color="#8B6C8E")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Per-Class Counting")
    axes[1].set_ylim(0, 12)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)

    fig.suptitle("Fudan Handan: Domain Shift Hurts Counting", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, facecolor="#f6f3ee", bbox_inches="tight")
    plt.close(fig)
    return out


def make_triptych(out_name: str, title: str, images: list[tuple[Path, str]]) -> Path:
    out = FIG_DIR / out_name
    canvas = Image.new("RGB", (1800, 820), "#f6f3ee")
    draw = ImageDraw.Draw(canvas)
    title_font = font(42, bold=True)
    caption_font = font(22, bold=False)
    draw.text((50, 30), title, fill="#" + TEXT, font=title_font)

    margin_x = 50
    gap = 30
    panel_w = (1800 - 2 * margin_x - 2 * gap) // 3
    panel_h = 610
    y0 = 110

    for idx, (img_path, caption) in enumerate(images):
        x0 = margin_x + idx * (panel_w + gap)
        panel(draw, (x0, y0, x0 + panel_w, y0 + panel_h))
        with Image.open(img_path).convert("RGB") as im:
            im.thumbnail((panel_w - 30, panel_h - 90))
            px = x0 + (panel_w - im.width) // 2
            py = y0 + 18
            canvas.paste(im, (px, py))
        draw.text((x0 + 18, y0 + panel_h - 52), caption, fill="#" + MUTED, font=caption_font)

    canvas.save(out)
    return out


def make_occlusion_montage_small() -> Path:
    out = FIG_DIR / "occlusion_montage_presentation.png"
    canvas = Image.new("RGB", (1720, 1060), "#f6f3ee")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 24), "真实视频中的 ID 跳变：同一辆车从 ID 316 变成 ID 322", fill="#" + TEXT, font=font(38, bold=True))
    draw.text((44, 72), "这不是人工遮挡，而是主视频里自然发生的真实错误案例。", fill="#" + MUTED, font=font(22))

    frames = [
        (FIG_DIR / "frame_00234.png", "Frame 234：车辆刚从左边界入画"),
        (FIG_DIR / "frame_00236.png", "Frame 236：仍然被跟踪为 ID 316"),
        (FIG_DIR / "frame_00238.png", "Frame 238：几乎同一位置被重新编号为 ID 322"),
        (FIG_DIR / "frame_00240.png", "Frame 240：新 ID 322 继续存在"),
    ]

    positions = [
        (40, 130, 820, 500),
        (860, 130, 1640, 500),
        (40, 540, 820, 910),
        (860, 540, 1640, 910),
    ]

    for (img_path, caption), (x1, y1, x2, y2) in zip(frames, positions):
        panel(draw, (x1, y1, x2, y2))
        with Image.open(img_path).convert("RGB") as im:
            im.thumbnail((x2 - x1 - 26, y2 - y1 - 88))
            px = x1 + (x2 - x1 - im.width) // 2
            py = y1 + 16
            canvas.paste(im, (px, py))
        draw.text((x1 + 18, y2 - 48), caption, fill="#" + TEXT, font=font(20, bold=False))

    summary = (
        "读图要点：左边界附近的同一辆车先是 ID 316，中间短暂丢失后，在几乎同一位置又以 ID 322 重建，"
        "这就是 natural ID switch，也就是“真实视频里自然发生的 ID 跳变”。"
    )
    draw_wrapped(draw, (50, 942), summary, max_width=1620, fill="#" + MUTED, font_obj=font(22), line_gap=8)
    canvas.save(out)
    return out


def make_artificial_occlusion_panel() -> Path:
    out = FIG_DIR / "artificial_occlusion_summary.png"
    canvas = Image.new("RGB", (1800, 980), "#f6f3ee")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 30), "Controlled Occlusion: ByteTrack Can Recover the Same ID", fill="#" + TEXT, font=font(40, bold=True))

    items = [
        (REPO_ROOT / "outputs/artificial_occlusion/id28_case/occluded_tracked_00184.png", "Short occlusion: before masking"),
        (REPO_ROOT / "outputs/artificial_occlusion/id28_case/occluded_tracked_00192.png", "Short occlusion: same ID after recovery"),
        (REPO_ROOT / "outputs/artificial_occlusion/id28_case_long/occluded_long_tracked_00204.png", "Longer occlusion: still returns to the same ID"),
    ]

    margin_x = 50
    gap = 28
    panel_w = (1800 - 2 * margin_x - 2 * gap) // 3
    y0 = 120
    h = 610
    for idx, (img_path, caption) in enumerate(items):
        x0 = margin_x + idx * (panel_w + gap)
        panel(draw, (x0, y0, x0 + panel_w, y0 + h))
        with Image.open(img_path).convert("RGB") as im:
            im.thumbnail((panel_w - 28, h - 86))
            canvas.paste(im, (x0 + (panel_w - im.width) // 2, y0 + 16))
        draw.text((x0 + 18, y0 + h - 48), caption, fill="#" + MUTED, font=font(20))

    summary = (
        "Takeaway: moderate artificial occlusion does not automatically trigger an ID switch. "
        "The real failure case is more about boundary truncation and one-frame detection drop."
    )
    draw_wrapped(draw, (60, 785), summary, max_width=1680, fill="#" + TEXT, font_obj=font(24), line_gap=10)
    canvas.save(out)
    return out


def make_fudan_panel(samples_path: Path, compare_path: Path) -> Path:
    out = FIG_DIR / "fudan_case_panel.png"
    canvas = Image.new("RGB", (1800, 1100), "#f6f3ee")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 30), "Fudan Handan: Better Stress Test, Worse Generalization", fill="#" + TEXT, font=font(40, bold=True))
    panel(draw, (50, 100, 1750, 1080))
    with Image.open(samples_path).convert("RGB") as samples:
        samples.thumbnail((1660, 430))
        canvas.paste(samples, (70, 130))
    with Image.open(compare_path).convert("RGB") as comp:
        comp.thumbnail((1660, 420))
        canvas.paste(comp, (70, 610))
    canvas.save(out)
    return out


def generate_figures() -> dict[str, Path]:
    y8m = load_yolov8m_stats()
    model_compare = make_model_compare(y8m)
    fudan_compare = make_fudan_compare()
    triptych = make_triptych(
        "fudan_samples_triptych.png",
        "Fudan Handan Sample Frames",
        [
            (REPO_ROOT / "outputs/fudan_handan/sample_00000.png", "Frame 0: camera and stop-line setup"),
            (REPO_ROOT / "outputs/fudan_handan/sample_00430.png", "Frame 430: dense vehicles near the stop line"),
            (REPO_ROOT / "outputs/fudan_handan/sample_00859.png", "Frame 859: late-stage traffic and perspective change"),
        ],
    )
    occlusion = make_occlusion_montage_small()
    artificial = make_artificial_occlusion_panel()
    fudan_panel = make_fudan_panel(triptych, fudan_compare)
    return {
        "model_compare": model_compare,
        "occlusion": occlusion,
        "artificial": artificial,
        "fudan_panel": fudan_panel,
        "conf_ablation": FIG_DIR / "conf_ablation.png",
        "per_class_ap_val": FIG_DIR / "per_class_ap_val.png",
    }


@dataclass
class TextBox:
    x: float
    y: float
    w: float
    h: float
    paragraphs: list[str]
    font_size: int = 20
    color: str = TEXT
    bold_first: bool = False
    fill: str | None = None


@dataclass
class PictureBox:
    path: Path
    x: float
    y: float
    w: float
    h: float


@dataclass
class SlideSpec:
    title: str
    subtitle: str | None = None
    textboxes: list[TextBox] = field(default_factory=list)
    pictures: list[PictureBox] = field(default_factory=list)


def inches(val: float) -> int:
    return int(val * EMU)


def fit_image(img_path: Path, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    with Image.open(img_path) as im:
        aspect = im.width / im.height
    box_aspect = w / h
    if aspect > box_aspect:
        new_w = w
        new_h = w / aspect
        return x, y + (h - new_h) / 2, new_w, new_h
    new_h = h
    new_w = h * aspect
    return x + (w - new_w) / 2, y, new_w, new_h


def xml_text_paragraph(text: str, font_size: int, color: str, bold: bool = False) -> str:
    bullet = text.startswith("• ")
    clean = text[2:] if bullet else text
    ppr = '<a:pPr marL="228600" indent="-228600"><a:buChar char="•"/></a:pPr>' if bullet else ""
    bold_attr = ' b="1"' if bold else ""
    return (
        f"<a:p>{ppr}<a:r><a:rPr lang=\"zh-CN\" sz=\"{font_size * 100}\"{bold_attr} dirty=\"0\" smtClean=\"0\">"
        f"<a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill></a:rPr>"
        f"<a:t>{escape(clean)}</a:t></a:r><a:endParaRPr lang=\"zh-CN\" sz=\"{font_size * 100}\"/></a:p>"
    )


def shape_xml(obj_id: int, name: str, box: TextBox) -> str:
    fill_xml = ""
    if box.fill:
        fill_xml = f"<a:solidFill><a:srgbClr val=\"{box.fill}\"/></a:solidFill>"
    paragraphs = []
    for idx, text in enumerate(box.paragraphs):
        paragraphs.append(xml_text_paragraph(text, box.font_size, box.color, bold=(idx == 0 and box.bold_first)))
    tx_body = "".join(paragraphs)
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{obj_id}" name="{escape(name)}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{inches(box.x)}" y="{inches(box.y)}"/>
          <a:ext cx="{inches(box.w)}" cy="{inches(box.h)}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        {fill_xml}
        <a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" rtlCol="0" anchor="t"/>
        <a:lstStyle/>
        {tx_body}
      </p:txBody>
    </p:sp>
    """


def picture_xml(obj_id: int, name: str, pic: PictureBox, rel_id: str) -> str:
    x, y, w, h = fit_image(pic.path, pic.x, pic.y, pic.w, pic.h)
    return f"""
    <p:pic>
      <p:nvPicPr>
        <p:cNvPr id="{obj_id}" name="{escape(name)}"/>
        <p:cNvPicPr/>
        <p:nvPr/>
      </p:nvPicPr>
      <p:blipFill>
        <a:blip r:embed="{rel_id}"/>
        <a:stretch><a:fillRect/></a:stretch>
      </p:blipFill>
      <p:spPr>
        <a:xfrm>
          <a:off x="{inches(x)}" y="{inches(y)}"/>
          <a:ext cx="{inches(w)}" cy="{inches(h)}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
      </p:spPr>
    </p:pic>
    """


def build_slides(figs: dict[str, Path]) -> list[SlideSpec]:
    y8m = load_yolov8m_stats()
    y8m_video = load_yolov8m_video_results()
    detector_status_lines = [
        "YOLOv8s baseline",
        "• val mAP50 = 0.412",
        "• val mAP50-95 = 0.246",
        "• test-dev mAP50 = 0.326",
        "• test-dev mAP50-95 = 0.191",
        "",
        "YOLOv8m current status",
        f"• 已训练到 {y8m['epochs_recorded']} epochs，并产出 best.pt",
        f"• 当前 best val mAP50 = {y8m['best_map50']:.3f}",
        f"• 当前 best val mAP50-95 = {y8m['best_map']:.3f}",
    ]
    if y8m_video["ready"]:
        detector_status_lines.append("• 两段视频对比结果已补齐，可直接用于最终展示")
    else:
        detector_status_lines.append("• 视频级最终对比仍待补齐")

    y8m_slide_lines = [
        "Current status",
        f"• YOLOv8m has already reached val mAP50 {y8m['best_map50']:.3f} and val mAP50-95 {y8m['best_map']:.3f}",
        "• 这已经说明更大模型在 VisDrone 上有小幅提升",
        "",
    ]
    if y8m_video["ready"]:
        test_total = y8m_video["test_clip"]["total"]
        fudan_total = y8m_video["fudan_handan"]["total"]
        y8m_slide_lines += [
            "Video-level comparison is ready",
            f"• new_drone_video / test_clip: YOLOv8m total crossings = {test_total}",
            f"• fudan_handan: YOLOv8m total crossings = {fudan_total}",
            "• 现在可以直接把这一页讲成最终结果，而不是实验计划",
            "",
            "Interpretation slot",
            "• 如果主视频更稳，说明更大检测器能改善下游计数稳定性",
            "• 如果 Fudan Handan 仍明显落后 GT，说明域偏移仍是主瓶颈",
        ]
    else:
        y8m_slide_lines += [
            "Only two final video experiments are still needed",
            "• 在 new_drone_video（脚本副本名 test_clip）上用最佳配置跑 1 组，与 YOLOv8s 对比",
            "• 在 fudan_handan 上用最佳配置跑 1 组，与 YOLOv8s 对比",
            "",
            "Reserved conclusion slot",
            "• 如果视频级结果也更稳，就可以把结论升级为：更大检测器能同时改善检测和下游计数",
            "• 如果提升有限，也能得出：域偏移比模型规模更关键",
        ]
    return [
        SlideSpec(
            title="HW2 Task 2",
            subtitle="VisDrone detection, ByteTrack tracking, occlusion analysis, and line counting",
            textboxes=[
                TextBox(0.85, 1.9, 11.6, 1.0, ["YOLOv8s baseline + YOLOv8m enhancement path"], font_size=30, color=ACCENT, bold_first=True),
                TextBox(
                    0.95, 3.0, 11.2, 2.2,
                    [
                        "• 主视频 new_drone_video（脚本副本名 test_clip）：系统能工作，计数稳定",
                        "• 真实失败案例：边界附近 car 316 -> 322",
                        "• 人工遮挡实验：中等遮挡下 ID 仍可恢复",
                        "• fudan_handan：域外场景中计数和 ID 稳定性明显下降",
                    ],
                    font_size=24,
                ),
            ],
        ),
        SlideSpec(
            title="1. Detector Results",
            textboxes=[
                TextBox(
                    0.6, 1.1, 5.2, 4.8,
                    detector_status_lines,
                    font_size=20,
                ),
            ],
            pictures=[
                PictureBox(figs["model_compare"], 6.0, 1.2, 6.8, 2.5),
                PictureBox(figs["per_class_ap_val"], 6.0, 3.9, 6.8, 2.5),
            ],
        ),
        SlideSpec(
            title="2. Tracking on new_drone_video.mp4",
            textboxes=[
                TextBox(
                    0.65, 1.15, 4.4, 5.4,
                    [
                        "Main pipeline output",
                        "• 511 frames, about 17.03 seconds",
                        "• 12,791 detections across the clip",
                        "• 78 rough unique track IDs",
                        "• 输出视频：tracked_bytetrack.mp4",
                        "",
                        "Key interpretation",
                        "• 大部分车辆轨迹已连续",
                        "• 代表性失败不是拥挤路口整体崩溃",
                        "• 而是边界入画时的单帧漏检导致轨迹重启",
                    ],
                    font_size=20,
                ),
            ],
            pictures=[
                PictureBox(figs["occlusion"], 5.3, 1.15, 7.3, 5.95),
            ],
        ),
        SlideSpec(
            title="3. Occlusion Analysis",
            textboxes=[
                TextBox(
                    0.65, 1.12, 4.5, 5.8,
                    [
                        "Natural failure case",
                        "• frame 236: car tracked as ID 316",
                        "• frame 237: one-frame interruption",
                        "• frame 238: same target rebuilt as ID 322",
                        "",
                        "Supplementary controlled experiment",
                        "• 人工遮挡 186-190：无 ID switch",
                        "• 人工遮挡 186-200：仍无 ID switch",
                        "",
                        "Conclusion",
                        "• 遮挡并不必然导致换 ID",
                        "• 真正关键是检测连续性、边界截断和恢复窗口",
                    ],
                    font_size=20,
                ),
            ],
            pictures=[
                PictureBox(figs["artificial"], 5.35, 1.25, 7.25, 5.85),
            ],
        ),
        SlideSpec(
            title="4. Line Counting on new_drone_video.mp4",
            textboxes=[
                TextBox(
                    0.65, 1.15, 4.6, 5.7,
                    [
                        "Virtual line",
                        "• x = 1280 vertical line",
                        "",
                        "Automatic result",
                        "• total crossings = 18",
                        "• all counted as car",
                        "",
                        "Ablation result",
                        "• conf 从 0.15 到 0.50，越线总数始终是 18",
                        "• 主车流计数对阈值并不敏感",
                        "• 这段视频适合展示“系统可用性”",
                    ],
                    font_size=20,
                ),
            ],
            pictures=[
                PictureBox(figs["conf_ablation"], 5.45, 1.5, 7.0, 5.2),
            ],
        ),
        SlideSpec(
            title="5. Fudan Handan Failure Case",
            textboxes=[
                TextBox(
                    0.65, 1.08, 4.3, 5.95,
                    [
                        "Why add a second video",
                        "• test_clip 太顺，不能说明系统边界",
                        "• fudan_handan 更像域外压力测试",
                        "",
                        "Manual GT",
                        "• car = 11, van = 2, bus = 1",
                        "• total = 14",
                        "",
                        "Best current auto setting",
                        "• 1280 / conf=0.10 -> total = 10",
                        "• 仍少 4 car、少 1 van，并多 1 pedestrian",
                    ],
                    font_size=20,
                ),
            ],
            pictures=[
                PictureBox(figs["fudan_panel"], 5.1, 1.15, 7.65, 5.95),
            ],
        ),
        SlideSpec(
            title="6. YOLOv8m Final Comparison Plan",
            textboxes=[
                TextBox(
                    0.7, 1.3, 11.9, 4.7,
                    y8m_slide_lines,
                    font_size=22,
                ),
            ],
        ),
        SlideSpec(
            title="7. Final Takeaways",
            textboxes=[
                TextBox(
                    0.75, 1.25, 11.6, 4.9,
                    [
                        "• 主成功案例：YOLOv8s + ByteTrack 在 new_drone_video 上已能稳定输出检测、跟踪和计数结果",
                        "• 主失败案例：真实 ID switch 更像边界漏检，而不是“遮挡就必然失败”",
                        "• 补充证据：人工遮挡实验说明 ByteTrack 在中等遮挡下仍可恢复原 ID",
                        "• 域外案例：fudan_handan 显示系统的主要短板在域泛化、边界召回和轨迹稳定性",
                        "• 增强方向：YOLOv8m 是最小成本、最适合课堂展示的下一步增强",
                    ],
                    font_size=24,
                ),
                TextBox(
                    0.95, 6.35, 10.8, 0.5,
                    ["Files: report/task2_report.md  |  report/presentation_script.html  |  report/hw2_task2_presentation.pptx"],
                    font_size=18,
                    color=MUTED,
                ),
            ],
        ),
    ]


def build_slide_xml(spec: SlideSpec, slide_index: int, image_map: list[tuple[str, PictureBox]]) -> str:
    elems = []
    obj_id = 1
    elems.append(
        """
        <p:nvGrpSpPr>
          <p:cNvPr id="1" name=""/>
          <p:cNvGrpSpPr/>
          <p:nvPr/>
        </p:nvGrpSpPr>
        <p:grpSpPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="0" cy="0"/>
            <a:chOff x="0" y="0"/>
            <a:chExt cx="0" cy="0"/>
          </a:xfrm>
        </p:grpSpPr>
        """
    )
    obj_id += 1
    banner = TextBox(0.0, 0.0, SLIDE_W, 0.78, [spec.title], font_size=28, color="FFFFFF", bold_first=True, fill=ACCENT)
    elems.append(shape_xml(obj_id, f"Title {slide_index}", banner))
    obj_id += 1
    if spec.subtitle:
        sub = TextBox(0.85, 0.95, 11.6, 0.7, [spec.subtitle], font_size=18, color=MUTED)
        elems.append(shape_xml(obj_id, f"Subtitle {slide_index}", sub))
        obj_id += 1
    for idx, box in enumerate(spec.textboxes, start=1):
        elems.append(shape_xml(obj_id, f"TextBox {slide_index}-{idx}", box))
        obj_id += 1
    for idx, (rel_id, pic) in enumerate(image_map, start=1):
        elems.append(picture_xml(obj_id, f"Picture {slide_index}-{idx}", pic, rel_id))
        obj_id += 1

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      {''.join(elems)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def build_slide_rels(image_rels: list[tuple[str, str]]) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    for rid, target in image_rels:
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>
"""


def build_pptx(figs: dict[str, Path]) -> None:
    slides = build_slides(figs)
    media_files: dict[Path, str] = {}
    for slide in slides:
        for pic in slide.pictures:
            if pic.path not in media_files:
                ext = pic.path.suffix.lower()
                media_files[pic.path] = f"image{len(media_files) + 1}{ext}"

    with zipfile.ZipFile(PPT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        def write(name: str, data: str | bytes) -> None:
            zf.writestr(name, data)

        write("[Content_Types].xml", content_types_xml(len(slides), media_files))
        write("_rels/.rels", root_rels_xml())
        write("docProps/core.xml", core_xml())
        write("docProps/app.xml", app_xml(len(slides)))
        write("ppt/presentation.xml", presentation_xml(len(slides)))
        write("ppt/_rels/presentation.xml.rels", presentation_rels_xml(len(slides)))
        write("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        write("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels_xml())
        write("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        write("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels_xml())
        write("ppt/theme/theme1.xml", theme_xml())

        for idx, slide in enumerate(slides, start=1):
            rel_pairs = []
            rid_num = 2
            for pic in slide.pictures:
                rel_pairs.append((f"rId{rid_num}", pic))
                rid_num += 1
            write(f"ppt/slides/slide{idx}.xml", build_slide_xml(slide, idx, rel_pairs))
            rels = [(rid, media_files[pic.path]) for rid, pic in rel_pairs]
            write(f"ppt/slides/_rels/slide{idx}.xml.rels", build_slide_rels(rels))

        for path, media_name in media_files.items():
            zf.write(path, f"ppt/media/{media_name}")


def content_types_xml(slide_count: int, media_files: dict[Path, str]) -> str:
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Default Extension="jpg" ContentType="image/jpeg"/>',
        '<Default Extension="jpeg" ContentType="image/jpeg"/>',
    ]
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx in range(1, slide_count + 1):
        overrides.append(
            f'<Override PartName="/ppt/slides/slide{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  {''.join(defaults)}
  {''.join(overrides)}
</Types>
"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def core_xml() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>HW2 Task 2 Presentation</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{ts}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{ts}</dcterms:modified>
</cp:coreProperties>
"""


def app_xml(slide_count: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Slides</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>{slide_count}</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="{slide_count}" baseType="lpstr">
      {''.join('<vt:lpstr>Slide</vt:lpstr>' for _ in range(slide_count))}
    </vt:vector>
  </TitlesOfParts>
</Properties>
"""


def presentation_xml(slide_count: int) -> str:
    slide_ids = "".join(
        f'<p:sldId id="{256 + idx}" r:id="rId{idx + 1}"/>' for idx in range(slide_count)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="{inches(SLIDE_W)}" cy="{inches(SLIDE_H)}" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""


def presentation_rels_xml(slide_count: int) -> str:
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for idx in range(1, slide_count + 1):
        rels.append(f'<Relationship Id="rId{idx + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{idx}.xml"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>
"""


def slide_master_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="Master">
    <p:bg>
      <p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill></p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2"
            accent1="accent1" accent2="accent2" accent3="accent3"
            accent4="accent4" accent5="accent5" accent6="accent6"
            hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="1" r:id="rId1"/>
  </p:sldLayoutIdLst>
  <p:txStyles>
    <p:titleStyle/>
    <p:bodyStyle/>
    <p:otherStyle/>
  </p:txStyles>
</p:sldMaster>
"""


def slide_master_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""


def slide_layout_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""


def slide_layout_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Minimal Theme">
  <a:themeElements>
    <a:clrScheme name="Custom">
      <a:dk1><a:srgbClr val="000000"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1E2430"/></a:dk2>
      <a:lt2><a:srgbClr val="F6F3EE"/></a:lt2>
      <a:accent1><a:srgbClr val="1F5A4E"/></a:accent1>
      <a:accent2><a:srgbClr val="C8682A"/></a:accent2>
      <a:accent3><a:srgbClr val="7C8DA6"/></a:accent3>
      <a:accent4><a:srgbClr val="8B6C8E"/></a:accent4>
      <a:accent5><a:srgbClr val="5B6577"/></a:accent5>
      <a:accent6><a:srgbClr val="C7B9A3"/></a:accent6>
      <a:hlink><a:srgbClr val="0563C1"/></a:hlink>
      <a:folHlink><a:srgbClr val="954F72"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Custom Fonts">
      <a:majorFont>
        <a:latin typeface="DejaVu Sans"/>
        <a:ea typeface="Microsoft YaHei"/>
        <a:cs typeface="DejaVu Sans"/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="DejaVu Sans"/>
        <a:ea typeface="Microsoft YaHei"/>
        <a:cs typeface="DejaVu Sans"/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Custom Format">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="lt1"/></a:solidFill>
        <a:solidFill><a:schemeClr val="accent1"/></a:solidFill>
        <a:solidFill><a:schemeClr val="accent2"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525"><a:solidFill><a:schemeClr val="accent1"/></a:solidFill></a:ln>
        <a:ln w="25400"><a:solidFill><a:schemeClr val="accent2"/></a:solidFill></a:ln>
        <a:ln w="38100"><a:solidFill><a:schemeClr val="accent3"/></a:solidFill></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
      </a:effectStyleLst>
      <a:bgFillStyleLst>
        <a:solidFill><a:schemeClr val="lt2"/></a:solidFill>
        <a:solidFill><a:schemeClr val="lt1"/></a:solidFill>
        <a:solidFill><a:schemeClr val="accent1"/></a:solidFill>
      </a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>
"""


def build_html(figs: dict[str, Path]) -> None:
    y8m = load_yolov8m_stats()
    y8m_video = load_yolov8m_video_results()
    if y8m_video["ready"]:
        y8m_status_html = (
            f"<li>YOLOv8m 当前已经训练到 {y8m['epochs_recorded']} 个 epoch，best val mAP50 = {y8m['best_map50']:.3f}，并且两段视频级对比也已经补齐。</li>"
        )
        y8m_end_html = f"""
      <p>最后一部分我准备把 `YOLOv8m` 作为增强版对比。现在这部分已经不是预留位，而是有结果可讲：在 `new_drone_video / test_clip` 上，YOLOv8m 自动越线总数是 {y8m_video["test_clip"]["total"]}；在 `fudan_handan` 上，YOLOv8m 自动越线总数是 {y8m_video["fudan_handan"]["total"]}。</p>
      <ul>
        <li>如果主视频更稳，就说明更大检测器可以改善下游跟踪和计数。</li>
        <li>如果 `fudan_handan` 仍明显偏离人工 GT，就说明域偏移依然是当前主要瓶颈。</li>
      </ul>
      <div class="notes">讲稿：最后总结一下。这个项目已经完成了从检测到跟踪再到下游计数的完整链路，也补上了 YOLOv8m 的最终对比。现在展示时可以把它当成最终结论页，而不是待完成事项。</div>
"""
    else:
        y8m_status_html = (
            f"<li>YOLOv8m 当前已经训练到 {y8m['epochs_recorded']} 个 epoch，best val mAP50 = {y8m['best_map50']:.3f}，已经略高于 YOLOv8s。</li>"
        )
        y8m_end_html = """
      <p>最后一部分我准备把 `YOLOv8m` 作为增强版对比。因为它在验证集上已经比 `YOLOv8s` 略强，所以后续我只会用最佳配置在两个视频上各跑一组，分别和 `YOLOv8s` 对比，而不会重复整套大规模实验。</p>
      <ul>
        <li>如果视频级结果也更稳，就能说明更大检测器可以改善下游跟踪和计数。</li>
        <li>如果提升有限，也能反过来说明当前真正的瓶颈是域偏移和轨迹稳定性，不只是模型大小。</li>
      </ul>
      <div class="notes">讲稿：最后总结一下。这个项目已经完成了从检测到跟踪再到下游计数的完整链路，也给出了成功案例、失败案例和可控实验。最后再补两组 YOLOv8m 的视频级对比，这个作业的展示就会比较完整。</div>
"""
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>HW2 Task 2 Presentation Script</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --panel: #ffffff;
      --text: #1e2430;
      --muted: #5b6577;
      --accent: #1f5a4e;
      --accent2: #c8682a;
      --line: #e9e3d8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", sans-serif;
      background: radial-gradient(circle at top right, #efe8dc, var(--bg) 42%);
      color: var(--text);
      line-height: 1.65;
    }}
    .wrap {{
      width: min(1100px, calc(100vw - 40px));
      margin: 32px auto 80px;
    }}
    .hero {{
      background: linear-gradient(135deg, #163f36, #2c6a5d 72%);
      color: #fff;
      padding: 34px 38px;
      border-radius: 26px;
      box-shadow: 0 18px 40px rgba(22,63,54,.18);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 38px; }}
    .hero p {{ margin: 0; font-size: 18px; color: rgba(255,255,255,.86); }}
    .meta {{
      margin-top: 14px;
      font-size: 15px;
      color: rgba(255,255,255,.78);
    }}
    .slide {{
      margin-top: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px 30px;
      box-shadow: 0 10px 28px rgba(31,41,55,.06);
    }}
    .slide h2 {{
      margin: 0 0 10px;
      font-size: 28px;
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.15fr .95fr;
      gap: 22px;
      align-items: start;
    }}
    .notes {{
      background: #faf7f2;
      border-left: 4px solid var(--accent2);
      padding: 14px 16px;
      border-radius: 14px;
      margin-top: 14px;
    }}
    ul {{ padding-left: 20px; margin: 10px 0; }}
    li {{ margin: 6px 0; }}
    img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
      margin-right: 14px;
    }}
    .links a:hover {{ text-decoration: underline; }}
    .small {{ color: var(--muted); font-size: 14px; }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 30px; }}
      .slide h2 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>HW2 Task 2 演示讲稿</h1>
      <p>对着这份 HTML 念稿，同时展示生成好的 PPT 即可。总时长按 3 分钟左右控制。</p>
      <div class="meta">生成文件：<code>report/hw2_task2_presentation.pptx</code> · <code>report/task2_report.md</code></div>
    </section>

    <section class="slide">
      <h2>Slide 1 · 开场</h2>
      <p>这次作业我完成的是一个从检测训练到视频理解展示的完整流水线。前半部分是在 VisDrone 上训练 YOLOv8 检测器，后半部分是在新交通视频上做 ByteTrack 跟踪、遮挡分析和虚拟线越线计数。</p>
      <div class="notes">讲稿：我今天主要展示四点。第一，系统在主视频上是能工作的。第二，我分析了一个真实的 ID 跳变案例。第三，我做了一个人工遮挡实验来解释为什么有时会丢 ID、有时不会。第四，我又加入了一个更难的视频 `fudan_handan`，把系统的局限性也展示出来。</div>
    </section>

    <section class="slide">
      <h2>Slide 2 · 检测器结果</h2>
      <div class="grid">
        <div>
          <ul>
            <li>YOLOv8s 基线：val mAP50 = 0.412，val mAP50-95 = 0.246。</li>
            <li>test-dev 上也能保持可用结果，说明检测器本身是合格的。</li>
            <li>为了增强项目深度，我又训练了 YOLOv8m。</li>
            {y8m_status_html}
          </ul>
          <div class="notes">讲稿：这一页想表达的是，后面所有跟踪和计数实验不是建立在一个很弱的检测器上。YOLOv8s 已经能打底，而 YOLOv8m 也已经显示出小幅提升，所以后面只需要补两组视频级对比，不需要重新做整套实验。</div>
        </div>
        <div>
          <img src="figures/model_compare.png" alt="model compare">
          <img src="figures/per_class_ap_val.png" alt="per class ap" style="margin-top:14px;">
        </div>
      </div>
    </section>

    <section class="slide">
      <h2>Slide 3 · 主视频跟踪效果</h2>
      <div class="grid">
        <div>
          <ul>
            <li>主视频是 <code>new_drone_video.mp4</code>，脚本里处理的副本名是 <code>test_clip.mp4</code>，共 511 帧，约 17 秒。</li>
            <li>基线配置下得到约 12,791 个检测记录，78 个粗略唯一轨迹 ID。</li>
            <li>整体上车辆轨迹已经比较连续，适合展示系统的可用性。</li>
            <li>但在边界和短时漏检处，仍会发生 ID 跳变。</li>
          </ul>
          <div class="links small">
            <a href="../outputs/tracked_bytetrack.mp4">tracked_bytetrack.mp4</a>
            <a href="../outputs/line_count_demo.mp4">line_count_demo.mp4</a>
          </div>
          <div class="notes">讲稿：这一页先给老师和同学一个整体印象，就是系统不是只会输出一些静态图，它已经能在完整视频上稳定跟踪大部分车辆。后面的分析是在这个基础上进一步解释它为什么还会失败。</div>
        </div>
        <div>
          <img src="figures/occlusion_montage_presentation.png" alt="occlusion montage">
        </div>
      </div>
    </section>

    <section class="slide">
      <h2>Slide 4 · 遮挡与 ID 跳变</h2>
      <div class="grid">
        <div>
          <ul>
            <li>真实失败案例发生在 frame 236 到 238 左右，一辆车从 ID 316 变成了 ID 322。</li>
            <li>中间只丢了一帧，所以它更像边界截断导致的检测中断，而不是长时间遮挡。</li>
            <li>为了验证这一点，我又做了人工遮挡实验。</li>
            <li>结果是：遮挡 5 帧和约 15 帧后，目标都还能恢复成原来的同一个 ID。</li>
          </ul>
          <div class="notes">讲稿：这页是分析的重点。我想说明，ByteTrack 并不是遇到遮挡就一定换 ID。真正影响更大的是检测连续性。如果边界入画时检测不稳定，就算只漏一帧，也可能触发轨迹重建。</div>
        </div>
        <div>
          <img src="figures/artificial_occlusion_summary.png" alt="artificial occlusion">
        </div>
      </div>
    </section>

    <section class="slide">
      <h2>Slide 5 · 越线计数</h2>
      <div class="grid">
        <div>
          <ul>
            <li>在主视频上，我加了一条竖直虚拟线 <code>x=1280</code> 做越线计数。</li>
            <li>基线自动结果是总共 18 次越线，而且全部被统计为 car。</li>
            <li>我还做了置信度阈值消融，conf 从 0.15 调到 0.50，越线总数始终保持 18。</li>
            <li>这说明主车流计数对阈值并不敏感，所以这部分适合作为系统成功案例展示。</li>
          </ul>
          <div class="notes">讲稿：这里想表达的是，下游应用不是随便凑的，而是真的有一定稳定性。虽然检测框数量和唯一 ID 数会随阈值变化，但最后的主车流计数基本不变，这说明这个场景上的系统行为比较稳。</div>
        </div>
        <div>
          <img src="figures/conf_ablation.png" alt="conf ablation">
        </div>
      </div>
    </section>

    <section class="slide">
      <h2>Slide 6 · Fudan Handan 失败案例</h2>
      <div class="grid">
        <div>
          <ul>
            <li>为了让项目不只展示成功案例，我又加入了 <code>fudan_handan.mp4</code>。</li>
            <li>这个视频更像域外压力测试，车辆启停多、交互密集、停止线附近更复杂。</li>
            <li>人工 GT 是 car 11、van 2、bus 1，总共 14。</li>
            <li>自动计数最好的当前设置也只有 10，而且还多检了 1 个 pedestrian。</li>
            <li>说明问题不只是分辨率，而是域偏移、规则近似和轨迹稳定性共同造成的。</li>
          </ul>
          <div class="links small">
            <a href="../outputs/fudan_handan/compare_stopline/hires_1280_conf010/hires_1280_conf010_line_count_demo.mp4">best current line_count_demo</a>
          </div>
          <div class="notes">讲稿：这一页很重要，因为它告诉老师这个项目不是只会挑一段好看的视频。换一个更难、分布不同的场景以后，系统马上就暴露出漏计和 ID 不稳定问题，这反而让整个分析更完整。</div>
        </div>
        <div>
          <img src="figures/fudan_case_panel.png" alt="fudan case panel">
        </div>
      </div>
    </section>

    <section class="slide">
      <h2>Slide 7 · YOLOv8m 与结尾</h2>
      {y8m_end_html}
    </section>
  </div>
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    figs = generate_figures()
    build_pptx(figs)
    build_html(figs)
    print(f"Generated {PPT_PATH}")
    print(f"Generated {HTML_PATH}")


if __name__ == "__main__":
    main()
