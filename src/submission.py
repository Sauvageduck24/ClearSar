from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import zipfile

import numpy as np


def xyxy_to_coco_xywh(box: Sequence[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] -> [x, y, w, h] with non-negative size."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def coco_xywh_clip(box: Sequence[float], width: Optional[int], height: Optional[int]) -> List[float]:
    """Clip COCO [x, y, w, h] box to image bounds when size is available."""
    x, y, w, h = [float(v) for v in box]
    if width is None or height is None:
        return [x, y, max(0.0, w), max(0.0, h)]

    x1 = min(max(0.0, x), float(width))
    y1 = min(max(0.0, y), float(height))
    x2 = min(max(0.0, x + w), float(width))
    y2 = min(max(0.0, y + h), float(height))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _to_numpy(data: Any) -> np.ndarray:
    """Convert torch tensor / list / ndarray into numpy ndarray."""
    if hasattr(data, "detach") and hasattr(data, "cpu"):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


def predictions_to_submission_rows(
    outputs: Sequence[Mapping[str, Any]],
    metas: Sequence[Mapping[str, Any]],
    *,
    score_threshold: float = 0.001,
    category_id: int = 1,
    max_detections_per_image: Optional[int] = 300,
    box_format: str = "xyxy",
    skip_invalid_boxes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert model outputs to competition submission rows.

    Critical safety guarantees:
    - `image_id` is always read from `metas[i]['image_id']`, never from loop index.
    - output rows follow exact required schema:
      {"image_id": int, "category_id": 1, "bbox": [x, y, w, h], "score": float}

    Expected output dict keys:
    - boxes: shape [N, 4]
    - scores: shape [N]

    Expected meta keys:
    - image_id (real COCO ID for this test image)
    - width (optional, for clipping)
    - height (optional, for clipping)
    """
    if len(outputs) != len(metas):
        raise ValueError(
            f"outputs and metas must have same length; got {len(outputs)} vs {len(metas)}"
        )

    rows: List[Dict[str, Any]] = []

    for output, meta in zip(outputs, metas):
        if "image_id" not in meta:
            raise KeyError("Each meta dict must include the REAL 'image_id'.")

        image_id = int(meta["image_id"])
        width = meta.get("width")
        height = meta.get("height")

        boxes = _to_numpy(output.get("boxes", []))
        scores = _to_numpy(output.get("scores", []))

        if boxes.ndim == 1 and boxes.size == 0:
            continue

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"boxes must be [N,4], got shape={boxes.shape}")

        if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
            raise ValueError(
                "scores must be [N] and same length as boxes. "
                f"Got scores={scores.shape}, boxes={boxes.shape}"
            )

        order = np.argsort(-scores)
        if max_detections_per_image is not None:
            order = order[:max_detections_per_image]

        for idx in order:
            score = float(scores[idx])
            if score < score_threshold:
                continue

            box = boxes[idx].tolist()
            if box_format == "xyxy":
                coco_box = xyxy_to_coco_xywh(box)
            elif box_format == "xywh":
                coco_box = [float(v) for v in box]
            else:
                raise ValueError(f"Unsupported box_format={box_format}")

            coco_box = coco_xywh_clip(coco_box, width=width, height=height)
            if skip_invalid_boxes and (coco_box[2] <= 0.0 or coco_box[3] <= 0.0):
                continue

            rows.append(
                {
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": [float(v) for v in coco_box],
                    "score": score,
                }
            )

    return rows


def build_submission_from_filename_predictions(
    predictions_by_filename: Mapping[str, Mapping[str, Any]],
    filename_to_image_id: Mapping[str, int],
    *,
    score_threshold: float = 0.001,
    category_id: int = 1,
    max_detections_per_image: Optional[int] = 300,
    box_format: str = "xyxy",
) -> List[Dict[str, Any]]:
    """
    Build submission when your inference pipeline keys predictions by filename.

    This helper explicitly maps filename -> real image_id before writing rows.
    """
    rows: List[Dict[str, Any]] = []

    for file_name, pred in predictions_by_filename.items():
        if file_name not in filename_to_image_id:
            raise KeyError(
                f"Filename '{file_name}' missing in filename_to_image_id mapping. "
                "Cannot build valid submission with unknown image_id."
            )

        image_id = int(filename_to_image_id[file_name])
        output = {
            "boxes": pred.get("boxes", []),
            "scores": pred.get("scores", []),
        }
        meta = {
            "image_id": image_id,
            "width": pred.get("width"),
            "height": pred.get("height"),
        }

        per_file_rows = predictions_to_submission_rows(
            [output],
            [meta],
            score_threshold=score_threshold,
            category_id=category_id,
            max_detections_per_image=max_detections_per_image,
            box_format=box_format,
        )
        rows.extend(per_file_rows)

    return rows


def save_submission_json(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Save submission rows to JSON with deterministic formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "image_id": int(r["image_id"]),
            "category_id": int(r["category_id"]),
            "bbox": [float(v) for v in r["bbox"]],
            "score": float(r["score"]),
        }
        for r in rows
    ]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=True)

    return output_path


def save_submission_csv(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """
    Save submission as CSV.

    The bbox field is stored as a compact JSON string to keep exact COCO format.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "category_id", "bbox", "score"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "image_id": int(r["image_id"]),
                    "category_id": int(r["category_id"]),
                    "bbox": json.dumps([float(v) for v in r["bbox"]], ensure_ascii=True),
                    "score": float(r["score"]),
                }
            )

    return output_path


def save_submission_zip_with_json(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    inner_json_name: str = "submission.json",
) -> Path:
    """Save submission as ZIP containing a JSON file with competition schema."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "image_id": int(r["image_id"]),
            "category_id": int(r["category_id"]),
            "bbox": [float(v) for v in r["bbox"]],
            "score": float(r["score"]),
        }
        for r in rows
    ]
    payload = json.dumps(serialized, ensure_ascii=True)

    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_json_name, payload)

    return output_path


def save_submission_auto(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """
    Save submission based on output extension.

    Supported:
    - .zip: creates zip with submission.json inside
    - .csv: creates CSV file
    - .json: creates raw JSON file
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix == ".zip":
        return save_submission_zip_with_json(rows, output_path)
    if suffix == ".csv":
        return save_submission_csv(rows, output_path)
    if suffix == ".json":
        return save_submission_json(rows, output_path)

    raise ValueError(
        f"Unsupported output extension '{suffix}'. Use .zip, .csv or .json"
    )


def validate_submission_schema(rows: Sequence[Mapping[str, Any]]) -> None:
    """Quick structural validation before upload."""
    required = {"image_id", "category_id", "bbox", "score"}
    for i, row in enumerate(rows):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Row {i} missing keys: {missing}")

        if int(row["category_id"]) != 1:
            raise ValueError(f"Row {i} has category_id={row['category_id']}, expected 1")

        bbox = row["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Row {i} has invalid bbox={bbox}")

        x, y, w, h = [float(v) for v in bbox]
        if w < 0.0 or h < 0.0:
            raise ValueError(f"Row {i} has negative bbox size: {bbox}")
