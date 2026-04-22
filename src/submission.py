from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import _valid_bbox

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


def _get_train_avg_boxes(json_path: Path) -> float:
    if not json_path.exists():
        return 0.0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    n_imgs = len(data.get("images", []))
    return len(data.get("annotations", [])) / n_imgs if n_imgs else 0.0


def _load_test_id_mapping(test_images_dir: Path, mapping_path: Path) -> dict[str, int]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    df = pd.read_parquet(mapping_path)
    if "id" not in df.columns:
        raise ValueError(f"Parquet missing 'id' column: {mapping_path}")
    mapping = {
        Path(norm).name: int(Path(norm).stem)
        for item_id in df["id"].astype(str)
        if "/images/test/" in (norm := item_id.replace("\\", "/"))
        and Path(norm).stem.isdigit()
    }
    missing = [
        p.name
        for p in sorted(test_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name not in mapping
    ]
    if missing:
        raise ValueError(
            f"Missing IDs for test images in mapping. Examples: {', '.join(missing[:5])}"
        )
    return mapping


def _boxes_to_submission_rows(
    boxes: list[list[float]],
    image_id: int,
) -> list[dict[str, Any]]:
    rows = []
    for x1, y1, x2, y2, score, cls in boxes:
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        if not _valid_bbox(w, h):
            continue
        rows.append(
            {
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [float(x1), float(y1), w, h],
                "score": float(score),
            }
        )
    return rows


def _validate_submission(rows: list[dict]) -> None:
    for i, row in enumerate(rows):
        for key in ("image_id", "category_id", "bbox", "score"):
            if key not in row:
                raise ValueError(f"Row {i} missing required key '{key}'")
        if not isinstance(row["bbox"], list) or len(row["bbox"]) != 4:
            raise ValueError(f"Row {i} has invalid bbox: {row['bbox']}")


def _save_submission(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(payload, encoding="utf-8")
        return
    zip_path = output_path if output_path.suffix.lower() == ".zip" else output_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", payload)
