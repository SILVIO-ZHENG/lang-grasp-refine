"""Render checkpoint traces as RGB, center heatmap, and prediction overlays.

将检查点重放结果绘制为 RGB、中心热图和抓取框对照图。
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw


SIZE = 400
LABEL = "#e6fa73"
BEFORE = "#ff4667"
AFTER = "#18ecff"


def corners(box):
    """Get the ordered corners in the same normalized square-image frame.

    在同一个归一化方形图像坐标系中，计算有固定顺序的四个角点。
    """
    x, y, w, h, theta = box
    opening = np.array([math.cos(theta), math.sin(theta)]) * w / 2
    height = np.array([-math.sin(theta), math.cos(theta)]) * h / 2
    return np.array([x, y]) + np.stack((
        -opening - height, opening - height, opening + height, -opening + height,
    ))


def view(row):
    # Share a crop across the three panels so that apparent position and size
    # changes come from the predicted boxes rather than from a separate zoom.
    # 三个面板使用同一裁剪范围，图中位置与大小的变化来自抓取框本身。
    points = np.concatenate([corners(row[key]) for key in ("groundTruth", "before", "after")])
    low, high = points.min(0), points.max(0)
    side = min(1., max(.47, (high - low).max() * 1.19))
    x, y = np.clip((low + high - side) / 2, 0, 1 - side)
    return x, y, side


def crop(image, field):
    x, y, side = field
    return image.transform((SIZE, SIZE), Image.Transform.EXTENT,
                           (x * image.width, y * image.height,
                            (x + side) * image.width, (y + side) * image.height),
                           Image.Resampling.BICUBIC)


def point(coordinates, field):
    x, y, side = field
    return (np.asarray(coordinates) - [x, y]) * (SIZE / side)


def stroke(draw, vertices, color, dashed=False):
    # Draw a dark outline under each color so rectangles remain legible
    # over both bright and dark regions. Dashed lime is the annotation.
    # 每种颜色下方加深色描边，保证亮底和暗底都清楚。黄绿色虚线表示标注框。
    vertices = np.vstack((vertices, vertices[0]))
    for start, end in zip(vertices[:-1], vertices[1:], strict=True):
        distance = np.linalg.norm(end - start)
        if distance == 0:
            continue
        spans = [(j, min(j + 9, distance)) for j in range(0, math.ceil(distance), 17)] if dashed else [(0, distance)]
        for a, b in spans:
            segment = [tuple(start + (end - start) * t / distance) for t in (a, b)]
            draw.line(segment, fill="#111822", width=9)
            draw.line(segment, fill=color, width=5)


def pose_image(base, row, field, stage):
    """Overlay the prediction and one displayed label on the same RGB crop.

    在同一张 RGB 裁剪图上画出预测框和本例展示的单个标注框。
    """
    result = base.copy()
    draw = ImageDraw.Draw(result)
    stroke(draw, point(corners(row["groundTruth"]), field), LABEL, dashed=True)
    chosen = row[stage]
    color = BEFORE if stage == "before" else AFTER
    stroke(draw, point(corners(chosen), field), color)
    u, v = point(chosen[:2], field)
    draw.ellipse((u - 7, v - 7, u + 7, v + 7), fill=color, outline="#111822", width=3)
    return result


def heatmap(base, record, field):
    # Keep the display range fixed across all examples to make weak center
    # responses remain weak. These probabilities precede the final ranking.
    # 所有样例使用相同的显示范围，避免把弱响应夸大。这些是最终排序前的中心概率。
    scores = np.asarray(record["map"], dtype=np.float32)
    data = np.asarray(crop(Image.fromarray(scores), field)).clip(0, .72) / .72
    overlay = matplotlib.colormaps["turbo"](data)[..., :3]
    strength = np.clip((data - .04) * 2.40, 0, .86)[..., None]
    pixels = np.asarray(base) / 255.
    result = pixels * (1 - strength) + overlay * strength
    return Image.fromarray((result.clip(0, 1) * 255).astype(np.uint8))


def generate(trace, images, destination):
    """Write four tiles per traced instruction and a manifest for captions.

    每条重放指令输出四张小图，并生成供图注使用的清单。
    """
    measurements = json.loads(trace.read_text(encoding="utf-8-sig"))
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for split, label in (("seen", "Seen"), ("unseen", "Unseen")):
        for i, case in enumerate(measurements.get(f"test_{split}", {}).get("geometry_refinement", ())):
            tag = f"{split}{i + 1}" if i else split
            field = view(case)
            with Image.open(images / case["image"]) as source:
                image = source.convert("RGB")
            w, h = image.size
            if w != h:
                side = max(w, h)
                square = Image.new("RGB", (side, side))
                square.paste(image, ((side - w) // 2, (side - h) // 2))
                image = square
            base = crop(image, field)
            outputs = {"raw": base, "map": heatmap(base, case, field)}
            for phase in ("before", "after"):
                outputs[phase] = pose_image(base, case, field, phase)
            for panel, picture in outputs.items():
                picture.save(destination / f"geometry_{tag}_{panel}.png", optimize=True)
            # Display metrics refer to one visible annotation. The success
            # flags still come from the evaluation against all annotations.
            # 展示的指标对应图中这一条标注。成功标志仍来自对全部标注的正式评测。
            row = {k: case[k] for k in ("prompt", "beforeSuccess", "afterSuccess", "sameCandidate", "winnerIndex")}
            row.update(tag=tag, label=label)
            for phase in ("before", "after"):
                row[f"{phase}Metric"] = f'IoU {case[f"{phase}IoU"]:.2f}, {case[f"{phase}Angle"]:.0f}°'
            records.append(row)
    (destination / "samples.json").write_text(
        json.dumps({"geometry": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "images", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    generate(args.trace, args.images, args.output)


if __name__ == "__main__":
    main()
