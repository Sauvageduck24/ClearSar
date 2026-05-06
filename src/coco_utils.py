from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .preprocessing import (
    _load_image_gray,
    _load_image_raw,
    _save_image_raw,
    _to_gray,
)
from .utils import _progress_iter


def _yolo_to_coco_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    w = max(0.0, width * float(image_width))
    h = max(0.0, height * float(image_height))
    x = (x_center * float(image_width)) - w / 2.0
    y = (y_center * float(image_height)) - h / 2.0
    return [x, y, w, h]


def _coco_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def _load_coco(annotation_path: Path) -> tuple[dict, Dict[int, dict]]:
    import json

    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images_meta = {int(img["id"]): img for img in coco.get("images", [])}
    return coco, images_meta


def _boxes_should_merge(box_a: List[float], box_b: List[float]) -> bool:
    """Heuristic merge for contiguous/intersecting boxes in the same image."""
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    area_a = aw * ah
    area_b = bw * bh
    small_area = min(area_a, area_b)
    big_area = max(area_a, area_b)

    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter_area = inter_w * inter_h

    if inter_area > 0 and small_area <= (0.7 * big_area):
        return True

    y_overlap = max(0.0, min(ay2, by2) - max(ay, by))
    y_overlap_ratio = y_overlap / max(1e-6, min(ah, bh))
    if ax2 < bx:
        x_gap = bx - ax2
    elif bx2 < ax:
        x_gap = ax - bx2
    else:
        x_gap = 0.0

    max_x_gap = max(3.0, 0.25 * min(aw, bw))
    if y_overlap_ratio >= 0.4 and x_gap <= max_x_gap:
        return True

    x_overlap = max(0.0, min(ax2, bx2) - max(ax, bx))
    x_overlap_ratio = x_overlap / max(1e-6, min(aw, bw))
    if ay2 < by:
        y_gap = by - ay2
    elif by2 < ay:
        y_gap = ay - by2
    else:
        y_gap = 0.0

    max_y_gap = max(2.0, 0.25 * min(ah, bh))
    if x_overlap_ratio >= 0.5 and y_gap <= max_y_gap:
        return True

    return False


def _merge_contiguous_boxes(boxes: List[List[float]]) -> List[List[float]]:
    """Merge connected components of boxes that satisfy _boxes_should_merge."""
    if len(boxes) <= 1:
        return [list(map(float, b)) for b in boxes]

    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_should_merge(boxes[i], boxes[j]):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged: List[List[float]] = []
    for idxs in groups.values():
        xs, ys, x2s, y2s = [], [], [], []
        for idx in idxs:
            x, y, w, h = [float(v) for v in boxes[idx]]
            xs.append(x)
            ys.append(y)
            x2s.append(x + w)
            y2s.append(y + h)
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(x2s), max(y2s)
        merged.append([x_min, y_min, x_max - x_min, y_max - y_min])

    return merged


def _bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _random_non_overlapping_position(
    crop_w: int,
    crop_h: int,
    img_w: int,
    img_h: int,
    existing_boxes: List[List[float]],
    max_overlap: float,
    max_tries: int = 50,
) -> Optional[List[float]]:
    if crop_w <= 0 or crop_h <= 0 or crop_w > img_w or crop_h > img_h:
        return None

    max_x = img_w - crop_w
    max_y = img_h - crop_h

    for _ in range(max_tries):
        x0 = random.randint(0, max_x)
        y0 = random.randint(0, max_y)
        candidate = [float(x0), float(y0), float(crop_w), float(crop_h)]
        overlaps = [_bbox_iou(candidate, b) for b in existing_boxes]
        if not overlaps or max(overlaps) <= max_overlap:
            return candidate
    return None


def _apply_small_box_copy_paste(
    img: np.ndarray,
    boxes: List[List[float]],
    copy_paste_p: float,
    copy_paste_max_h: float,
    copy_paste_n: int,
    max_overlap: float = 0.3,
) -> tuple[np.ndarray, List[List[float]], int]:
    if copy_paste_n <= 0 or copy_paste_p <= 0:
        return img, boxes, 0

    img_h, img_w = int(img.shape[0]), int(img.shape[1])
    if img_h <= 1 or img_w <= 1:
        return img, boxes, 0

    pool: List[List[float]] = []
    for b in boxes:
        x, y, w, h = [float(v) for v in b]
        if w <= 0 or h <= 0:
            continue
        if h < copy_paste_max_h:
            pool.append([x, y, w, h])

    if not pool:
        return img, boxes, 0

    out_img = img.copy()
    out_boxes = [list(map(float, b)) for b in boxes]
    added = 0

    for _ in range(copy_paste_n):
        if random.random() > copy_paste_p:
            continue

        src = random.choice(pool)
        sx, sy, sw, sh = [int(round(v)) for v in src]
        if sw <= 0 or sh <= 0:
            continue

        sx = max(0, min(sx, img_w - 1))
        sy = max(0, min(sy, img_h - 1))
        ex = max(sx + 1, min(sx + sw, img_w))
        ey = max(sy + 1, min(sy + sh, img_h))
        crop = out_img[sy:ey, sx:ex]
        if crop.size == 0:
            continue

        ch, cw = int(crop.shape[0]), int(crop.shape[1])
        new_box = _random_non_overlapping_position(
            crop_w=cw,
            crop_h=ch,
            img_w=img_w,
            img_h=img_h,
            existing_boxes=out_boxes,
            max_overlap=max_overlap,
        )
        if new_box is None:
            continue

        nx, ny, nw, nh = [int(round(v)) for v in new_box]
        if out_img.ndim == 2:
            out_img[ny : ny + nh, nx : nx + nw] = crop
        else:
            out_img[ny : ny + nh, nx : nx + nw, ...] = crop

        out_boxes.append([float(nx), float(ny), float(nw), float(nh)])
        added += 1

    return out_img, out_boxes, added


def evaluate_box_snr_local(img_gray, box, buffer=20):
    """Calcula el SNR comparando la caja con su vecindad inmediata."""
    x, y, w, h = map(float, box)
    h_img, w_img = img_gray.shape

    xmin, ymin = int(max(0, x)), int(max(0, y))
    xmax, ymax = int(min(w_img, x + w)), int(min(h_img, y + h))

    box_pixels = img_gray[ymin:ymax, xmin:xmax]

    ctx_xmin, ctx_ymin = int(max(0, xmin - buffer)), int(max(0, ymin - buffer))
    ctx_xmax, ctx_ymax = int(min(w_img, xmax + buffer)), int(min(h_img, ymax + buffer))

    context_mask = np.zeros((ctx_ymax - ctx_ymin, ctx_xmax - ctx_xmin), dtype=bool)
    context_mask.fill(True)
    local_xmin, local_ymin = xmin - ctx_xmin, ymin - ctx_ymin
    local_xmax, local_ymax = xmax - ctx_xmin, ymax - ctx_ymin
    context_mask[local_ymin:local_ymax, local_xmin:local_xmax] = False

    context_pixels = img_gray[ctx_ymin:ctx_ymax, ctx_xmin:ctx_xmax][context_mask]

    if box_pixels.size == 0 or context_pixels.size == 0:
        return 0.0

    mu_box = np.mean(box_pixels)
    mu_bg = np.mean(context_pixels)
    std_bg = np.std(context_pixels)

    if std_bg < 1e-6:
        return 10.0
    return np.abs(mu_box - mu_bg) / std_bg


def evaluate_box_snr(img_gray, box):
    """Calcula el SNR/CNR de una sola caja COCO [x, y, w, h] en una imagen."""
    x, y, w, h = map(float, box)
    xmin, ymin = int(x), int(y)
    xmax, ymax = int(x + w), int(y + h)

    ymax = min(ymax, img_gray.shape[0])
    xmax = min(xmax, img_gray.shape[1])

    box_pixels = img_gray[ymin:ymax, xmin:xmax]

    mask = np.ones(img_gray.shape, dtype=bool)
    mask[ymin:ymax, xmin:xmax] = False
    bg_pixels = img_gray[mask]

    if box_pixels.size == 0 or bg_pixels.size == 0:
        return 0.0

    mu_box = np.mean(box_pixels)
    mu_bg = np.mean(bg_pixels)
    std_bg = np.std(bg_pixels)
    std_box = np.std(box_pixels)

    denominador = np.sqrt(std_box**2 + std_bg**2)
    if denominador == 0:
        return 0.0

    return np.abs(mu_box - mu_bg) / denominador


def _convert_coco_to_yolo(
    coco: dict,
    images_meta: Dict[int, dict],
    output_labels_dir: Path,
    image_ids: Optional[List[int]] = None,
    images_dir: Optional[Path] = None,
    min_snr: Optional[float] = None,
    max_box_height: Optional[float] = None,
    min_box_height: Optional[float] = None,
    skip_vertical_boxes: bool = False,
    small_box_copy_paste: bool = False,
    copy_paste_p: float = 0.5,
    copy_paste_max_h: float = 12.0,
    copy_paste_n: int = 3,
    merge_contiguous_boxes: bool = False,
    resized_image_size: Optional[int] = None,
    y_component_multiplier: float = 1.0,
    apply_letterboxing: bool = False,
    slicing: bool = False,
    slice_height: int = 256,
    slice_height_overlap: float = 0.2,
    slice_width: Optional[int] = None,
    slice_width_overlap: float = 0.0,
    progress_desc: str = "labels",
) -> None:
    """Convert COCO annotations to YOLO txt format (class cx cy w h, normalized)."""
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    anns_by_image: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    ids_to_process = image_ids if image_ids is not None else list(images_meta.keys())

    total_boxes = 0
    filtered_snr_boxes = 0
    filtered_height_boxes = 0
    filtered_small_boxes = 0
    filtered_vertical_boxes = 0
    unscored_boxes = 0
    copy_paste_added_boxes = 0
    premerge_boxes = 0
    postmerge_boxes = 0

    for img_id in _progress_iter(ids_to_process, total=len(ids_to_process), desc=progress_desc):
        if img_id not in images_meta:
            continue
        meta = images_meta[img_id]
        img_w, img_h = float(meta["width"]), float(meta["height"])
        stem = Path(meta["file_name"]).stem
        suffix = Path(meta["file_name"]).suffix

        if slicing:
            from .dataset import _compute_horizontal_slices, _compute_vertical_slices

            h_windows = _compute_horizontal_slices(
                int(round(img_h)),
                slice_height,
                slice_height_overlap,
            )
            v_windows = _compute_vertical_slices(
                int(round(img_w)),
                slice_width,
                slice_width_overlap,
            )
        else:
            h_windows = [(0, int(round(img_h)))]
            v_windows = [(0, int(round(img_w)))]

        has_vertical = slicing and len(v_windows) > 1
        for row_idx, (y0, y1) in enumerate(h_windows):
            for col_idx, (x0, x1) in enumerate(v_windows):
                if slicing:
                    if has_vertical:
                        out_name = f"{stem}_sl{row_idx:03d}_sw{col_idx:03d}{suffix}"
                    else:
                        out_name = f"{stem}_sl{row_idx:03d}{suffix}"
                else:
                    out_name = meta["file_name"]

                out_stem = Path(out_name).stem
                label_path = output_labels_dir / f"{out_stem}.txt"

                if resized_image_size is not None and resized_image_size <= 0:
                    raise ValueError("resized_image_size must be > 0")

                # apply_letterboxing=True: images are saved at original aspect ratio;
                # YOLO letterboxes internally, so labels stay normalized by original dims.
                # apply_letterboxing=False: images are stretched to a square of
                # resized_image_size x resized_image_size, so labels scale accordingly.
                if apply_letterboxing or resized_image_size is None:
                    out_img_w, out_img_h = img_w, img_h
                else:
                    out_img_w = out_img_h = float(resized_image_size)

                img_gray = None
                img_raw = None
                img_path = None
                if images_dir is not None:
                    img_path = images_dir / out_name

                if (min_snr is not None or small_box_copy_paste) and img_path is not None:
                    if img_path.exists():
                        img_raw = _load_image_raw(img_path)

                if min_snr is not None and img_raw is not None:
                    img_gray = _to_gray(img_raw)
                elif min_snr is not None and images_dir is not None:
                    if img_path is not None and img_path.exists():
                        img_gray = _load_image_gray(img_path)

                raw_boxes: List[List[float]] = []
                for ann in anns_by_image.get(img_id, []):
                    x, y, w, h = [float(v) for v in ann["bbox"]]
                    if w <= 0 or h <= 0:
                        continue

                    if slicing:
                        inter_x0 = max(x, float(x0))
                        inter_x1 = min(x + w, float(x1))
                        inter_y0 = max(y, float(y0))
                        inter_y1 = min(y + h, float(y1))
                        inter_w = inter_x1 - inter_x0
                        inter_h = inter_y1 - inter_y0
                        if inter_w <= 0 or inter_h <= 0:
                            continue

                        src_slice_w = max(1e-6, float(x1 - x0))
                        src_slice_h = max(1e-6, float(y1 - y0))
                        stretch_x = img_w / src_slice_w
                        stretch_y = img_h / src_slice_h
                        x_t = (inter_x0 - float(x0)) * stretch_x
                        y_t = (inter_y0 - float(y0)) * stretch_y
                        w_t = inter_w * stretch_x
                        h_t = inter_h * stretch_y
                    else:
                        x_t, y_t, w_t, h_t = x, y, w, h

                    if resized_image_size is not None and not apply_letterboxing:
                        sx = float(resized_image_size) / max(1e-6, img_w)
                        sy = float(resized_image_size) / max(1e-6, img_h)
                        x_t *= sx
                        y_t *= sy
                        w_t *= sx
                        h_t *= sy

                    raw_boxes.append([x_t, y_t, w_t, h_t])

                if merge_contiguous_boxes and raw_boxes:
                    premerge_boxes += len(raw_boxes)
                    boxes_to_use = _merge_contiguous_boxes(raw_boxes)
                    postmerge_boxes += len(boxes_to_use)
                else:
                    boxes_to_use = raw_boxes

                if (
                    small_box_copy_paste
                    and img_raw is not None
                    and img_path is not None
                    and copy_paste_n > 0
                    and copy_paste_p > 0
                    and boxes_to_use
                ):
                    # copy_paste_max_h is specified in original COCO pixel space, but
                    # boxes_to_use are already scaled to the resized image space.
                    # Scale the threshold to match so the pool filter works correctly.
                    if resized_image_size is not None and not apply_letterboxing:
                        _cp_sy = float(resized_image_size) / max(1e-6, img_h)
                        _effective_max_h = copy_paste_max_h * _cp_sy
                    else:
                        _effective_max_h = copy_paste_max_h
                    img_aug, boxes_aug, added = _apply_small_box_copy_paste(
                        img_raw,
                        boxes_to_use,
                        copy_paste_p=copy_paste_p,
                        copy_paste_max_h=_effective_max_h,
                        copy_paste_n=copy_paste_n,
                        max_overlap=0.3,
                    )
                    if added > 0:
                        saved = _save_image_raw(img_path, img_aug)
                        if saved:
                            boxes_to_use = boxes_aug
                            copy_paste_added_boxes += added

                lines: List[str] = []
                for box in boxes_to_use:
                    x, y, w, h = box
                    if w <= 0 or h <= 0:
                        continue
                    total_boxes += 1

                    if min_box_height is not None and h < min_box_height:
                        filtered_small_boxes += 1
                        continue

                    if skip_vertical_boxes and h >= w:
                        filtered_vertical_boxes += 1
                        continue

                    if max_box_height is not None and h > max_box_height:
                        filtered_height_boxes += 1
                        continue

                    if min_snr is not None:
                        if img_gray is None:
                            unscored_boxes += 1
                        else:
                            score = evaluate_box_snr_local(img_gray, box)
                            if score < min_snr:
                                filtered_snr_boxes += 1
                                continue

                    cx = max(0.0, min(1.0, (x + w / 2.0) / out_img_w))
                    cy = max(0.0, min(1.0, (y + h / 2.0) / out_img_h))
                    nw = max(0.0, min(1.0, w / out_img_w))
                    nh = max(0.0, min(1.0, h / out_img_h))
                    if nw > 0 and nh > 0:
                        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                label_path.write_text("\n".join(dict.fromkeys(lines)), encoding="utf-8")

    if (
        min_snr is not None
        or max_box_height is not None
        or min_box_height is not None
        or skip_vertical_boxes
        or small_box_copy_paste
    ):
        kept_boxes = (
            total_boxes
            - filtered_snr_boxes
            - filtered_height_boxes
            - filtered_small_boxes
            - filtered_vertical_boxes
        )
        parts: List[str] = []
        if merge_contiguous_boxes:
            parts.append(f"merge_contiguous=true boxes_before={premerge_boxes} boxes_after={postmerge_boxes}")
        if min_box_height is not None:
            parts.append(f"min_box_height={min_box_height:.3f} filtered_small={filtered_small_boxes}")
        if skip_vertical_boxes:
            parts.append(f"skip_vertical_boxes=true filtered_vertical={filtered_vertical_boxes}")
        if small_box_copy_paste:
            parts.append(
                f"small_box_copy_paste=true p={copy_paste_p:.3f} max_h={copy_paste_max_h:.3f} n={copy_paste_n} added={copy_paste_added_boxes}"
            )
        if max_box_height is not None:
            parts.append(f"max_box_height={max_box_height:.3f} filtered_height={filtered_height_boxes}")
        if min_snr is not None:
            parts.append(f"SNR min={min_snr:.3f} filtered_snr={filtered_snr_boxes} unscored={unscored_boxes}")
        parts.append(f"boxes kept={kept_boxes} total={total_boxes}")
        print(f"[filter] {' | '.join(parts)}")
    elif merge_contiguous_boxes or min_box_height is not None or skip_vertical_boxes or small_box_copy_paste:
        parts = []
        if merge_contiguous_boxes:
            parts.append(f"merge_contiguous=true boxes_before={premerge_boxes} boxes_after={postmerge_boxes}")
        if min_box_height is not None:
            parts.append(f"min_box_height={min_box_height:.3f} filtered_small={filtered_small_boxes}")
        if skip_vertical_boxes:
            parts.append(f"skip_vertical_boxes=true filtered_vertical={filtered_vertical_boxes}")
        if small_box_copy_paste:
            parts.append(
                f"small_box_copy_paste=true p={copy_paste_p:.3f} max_h={copy_paste_max_h:.3f} n={copy_paste_n} added={copy_paste_added_boxes}"
            )
        print(f"[filter] {' | '.join(parts)}")
