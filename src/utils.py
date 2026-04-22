from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y", "si"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false")


MIN_BOX_DIM = 0.1


def _valid_bbox(w: float, h: float) -> bool:
    return w > MIN_BOX_DIM and h > MIN_BOX_DIM


def _result_to_boxes(result) -> list[list[float]]:
    """Extract boxes from a YOLO result as [x1, y1, x2, y2, score, cls]."""
    if result is None or result.boxes is None:
        return []
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        score = float(box.conf[0].item()) if box.conf is not None else 0.0
        cls = int(box.cls[0].item()) if box.cls is not None else 0
        boxes.append([x1, y1, x2, y2, score, cls])
    return boxes


def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1 = min(max(0.0, x1), float(width))
    y1 = min(max(0.0, y1), float(height))
    x2 = min(max(0.0, x2), float(width))
    y2 = min(max(0.0, y2), float(height))
    return x1, y1, x2, y2


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _nms_per_class(boxes: list[list[float]], iou_thr: float) -> list[list[float]]:
    if not boxes:
        return []

    out: list[list[float]] = []
    classes = sorted({int(b[5]) for b in boxes})
    for cls in classes:
        cls_boxes = [b for b in boxes if int(b[5]) == cls]
        cls_boxes.sort(key=lambda b: float(b[4]), reverse=True)

        kept: list[list[float]] = []
        while cls_boxes:
            best = cls_boxes.pop(0)
            kept.append(best)
            best_xyxy = [float(best[0]), float(best[1]), float(best[2]), float(best[3])]

            remaining = []
            for cand in cls_boxes:
                cand_xyxy = [float(cand[0]), float(cand[1]), float(cand[2]), float(cand[3])]
                if _iou_xyxy(best_xyxy, cand_xyxy) <= iou_thr:
                    remaining.append(cand)
            cls_boxes = remaining

        out.extend(kept)

    return out


def _progress_iter(iterable, total: int, desc: str):
    """Wrapper de progreso con fallback si tqdm no esta disponible."""
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)
