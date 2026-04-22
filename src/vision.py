from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.coco_utils import _coco_bbox_to_xyxy
from src.utils import _valid_bbox


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if not label:
        return
    try:
        tb = draw.textbbox((0, 0), label)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        tw, th = max(1, len(label) * 6), 11
    pad = 4
    top = max(0, y1 - (th + pad * 2))
    draw.rectangle([x1, top, x1 + tw + pad * 2, top + th + pad * 2], fill=color)
    draw.text((x1 + pad, top + pad), label, fill=(255, 255, 255))


def _save_visualization(
    img_path: Path,
    output_dir: Path,
    gt_boxes: list[dict],
    pred_boxes: list[list[float]],
    class_names: list[str],
) -> None:
    with Image.open(img_path) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for ann in gt_boxes:
        bbox = _coco_bbox_to_xyxy(ann["bbox"])
        cls = int(ann.get("category_id", 1)) - 1
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        _draw_labeled_box(draw, bbox, f"GT {name}", (46, 160, 67))

    for pred in pred_boxes:
        x1, y1, x2, y2, score, cls_id = pred
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if not _valid_bbox(w, h):
            continue
        name = class_names[int(cls_id)] if 0 <= int(cls_id) < len(class_names) else str(int(cls_id))
        _draw_labeled_box(draw, (x1, y1, x2, y2), f"PR {name} {score:.2f}", (214, 48, 49))

    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / f"{img_path.stem}.png")
