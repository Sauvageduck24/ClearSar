from __future__ import annotations

"""
YOLO inference module for ClearSAR.

Runs a single-pass inference over test images and evaluates the model on
the holdout split using pycocotools metrics.

Usage:
    # Holdout evaluation only
    python yolo_inference.py --checkpoint models/yolo_best.pt --mode holdout

    # Test submission only
    python yolo_inference.py --checkpoint models/yolo_best.pt --mode test

    # Both (default)
    python yolo_inference.py --checkpoint models/yolo_best.pt
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS   = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
DEFAULT_CONF = 0.0
DEFAULT_IOU  = 0.5
DEFAULT_MAX_DET = 500
MIN_BOX_DIM  = 0.1


# ---------------------------------------------------------------------------
# Image size utilities
# ---------------------------------------------------------------------------

def _parse_image_size(value: str) -> int | list[int]:
    raw = str(value).strip()
    if not raw:
        raise argparse.ArgumentTypeError("--image-size cannot be empty")
    if raw.isdigit():
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("--image-size must be > 0")
        return parsed
    parts = [p.strip() for p in raw.strip("()[]").split(",") if p.strip()]
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        h, w = int(parts[0]), int(parts[1])
        if h <= 0 or w <= 0:
            raise argparse.ArgumentTypeError("--image-size values must be > 0")
        return [h, w]
    raise argparse.ArgumentTypeError(
        "Invalid --image-size format. Use 640 or [512,1024] / (512,1024)."
    )


def _imgsz_for_ultralytics(imgsz: int | list[int]) -> int | list[int]:
    if isinstance(imgsz, list):
        if len(imgsz) != 2:
            raise ValueError("--image-size as list must have exactly 2 values")
        h, w = int(imgsz[0]), int(imgsz[1])
        return h if h == w else [h, w]
    return int(imgsz)


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def _valid_bbox(w: float, h: float) -> bool:
    return w > MIN_BOX_DIM and h > MIN_BOX_DIM


def _result_to_boxes(result: Any) -> list[list[float]]:
    """Extract boxes from a YOLO result as list of [x1, y1, x2, y2, score, cls]."""
    if result is None or result.boxes is None:
        return []
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        score = float(box.conf[0].item()) if box.conf is not None else 0.0
        cls   = int(box.cls[0].item())   if box.cls  is not None else 0
        boxes.append([x1, y1, x2, y2, score, cls])
    return boxes


def _yolo_to_coco_bbox(
    x_center: float, y_center: float,
    width: float, height: float,
    image_width: int, image_height: int,
) -> list[float]:
    w = max(0.0, width  * float(image_width))
    h = max(0.0, height * float(image_height))
    x = (x_center * float(image_width))  - w / 2.0
    y = (y_center * float(image_height)) - h / 2.0
    return [x, y, w, h]


def _coco_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict(
    model: YOLO,
    img_path: Path,
    imgsz: int | list[int],
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[list[float]]:
    results = model.predict(
        source  = str(img_path),
        conf    = conf,
        iou     = iou,
        imgsz   = _imgsz_for_ultralytics(imgsz),
        max_det = max_det,
        augment = False,
        device  = device,
        verbose = False,
    )
    return _result_to_boxes(results[0] if results else None)


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def _parse_holdout_yaml(yaml_path: Path) -> tuple[Path, Path, list[str]]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Holdout yaml not found: {yaml_path}")
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    root = Path(payload.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    val_value  = str(payload.get("val", "images/val"))
    images_dir = (root / val_value).resolve()
    labels_dir = (root / val_value.replace("images", "labels", 1)).resolve()

    names = payload.get("names", [])
    if isinstance(names, dict):
        class_names = [str(names[k]) for k in sorted(names, key=int)]
    else:
        class_names = [str(n) for n in names]
    if not class_names:
        class_names = [f"class_{i}" for i in range(int(payload.get("nc", 0)))]

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Holdout images dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Holdout labels dir not found: {labels_dir}")

    return images_dir, labels_dir, class_names


def _build_holdout_coco_gt(
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
) -> tuple[dict, dict[str, int], list[Path]]:
    image_files = [
        p for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not image_files:
        raise ValueError(f"No images found in holdout dir: {images_dir}")

    images, annotations = [], []
    filename_to_id: dict[str, int] = {}
    ann_id = 1

    for idx, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        image_id = int(img_path.stem) if img_path.stem.isdigit() else idx
        filename_to_id[img_path.name] = image_id
        images.append({"id": image_id, "file_name": img_path.name, "width": img_w, "height": img_h})

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x, y, w, h = _yolo_to_coco_bbox(*map(float, parts[1:5]), img_w, img_h)
            if not _valid_bbox(w, h):
                continue
            annotations.append({
                "id": ann_id, "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
    coco_gt = {"images": images, "annotations": annotations, "categories": categories}
    return coco_gt, filename_to_id, image_files


def _compute_coco_metrics(
    coco_gt: dict,
    pred_rows: list[dict],
) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("pycocotools required. Install with: pip install pycocotools") from exc

    with tempfile.TemporaryDirectory(prefix="clearsar_holdout_") as tmp:
        tmp_path = Path(tmp)
        gt_json   = tmp_path / "gt.json"
        pred_json = tmp_path / "pred.json"
        gt_json.write_text(json.dumps(coco_gt), encoding="utf-8")
        pred_json.write_text(json.dumps(pred_rows), encoding="utf-8")

        coco_gt_api = COCO(str(gt_json))
        coco_dt_api = coco_gt_api.loadRes(str(pred_json))
        evaluator   = COCOeval(coco_gt_api, coco_dt_api, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {
        "map50_95": float(evaluator.stats[0]),
        "map50":    float(evaluator.stats[1]),
        "map75":    float(evaluator.stats[2]),
        "recall":   float(evaluator.stats[8]),
    }


def _evaluate_holdout(
    model: YOLO,
    holdout_yaml_path: Path,
    imgsz: int | list[int],
    device: str,
    save_viz: bool = False,
    viz_output_dir: Path | None = None,
    images_dir_override: Path | None = None,
) -> dict[str, float]:
    yaml_images_dir, labels_dir, class_names = _parse_holdout_yaml(holdout_yaml_path)
    images_dir = images_dir_override if images_dir_override is not None else yaml_images_dir
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Holdout images dir not found: {images_dir}")
    coco_gt, filename_to_id, image_files = _build_holdout_coco_gt(
        images_dir, labels_dir, class_names
    )

    annotations_by_image: dict[int, list] = {}
    if save_viz:
        if viz_output_dir is None:
            raise ValueError("viz_output_dir is required when save_viz=True")
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        for ann in coco_gt["annotations"]:
            annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    coco_preds: list[dict] = []
    for img_path in tqdm(image_files, desc="Holdout inference"):
        image_id = filename_to_id[img_path.name]
        boxes = _predict(model, img_path, imgsz, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MAX_DET, device)

        for x1, y1, x2, y2, score, cls in boxes:
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if not _valid_bbox(w, h):
                continue
            coco_preds.append({
                "image_id":   image_id,
                "category_id": int(cls) + 1,
                "bbox":  [float(x1), float(y1), w, h],
                "score": float(score),
            })

        if save_viz:
            _save_visualization(
                img_path    = img_path,
                output_dir  = viz_output_dir,
                gt_boxes    = annotations_by_image.get(image_id, []),
                pred_boxes  = boxes,
                class_names = class_names,
            )

    vertical = sum(1 for r in coco_preds if r["bbox"][3] > r["bbox"][2])
    print(f"[holdout] Vertical boxes: {vertical} of {len(coco_preds)}")

    return _compute_coco_metrics(coco_gt, coco_preds)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

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
        cls  = int(ann.get("category_id", 1)) - 1
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        _draw_labeled_box(draw, bbox, f"GT {name}", (46, 160, 67))

    for pred in pred_boxes:
        x1, y1, x2, y2, score, cls_id = pred
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if not _valid_bbox(w, h):
            continue
        name = class_names[int(cls_id)] if 0 <= int(cls_id) < len(class_names) else str(int(cls_id))
        _draw_labeled_box(draw, (x1, y1, x2, y2), f"PR {name} {score:.2f}", (214, 48, 49))

    canvas.save(output_dir / f"{img_path.stem}.png")


# ---------------------------------------------------------------------------
# Submission utilities
# ---------------------------------------------------------------------------

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
        p.name for p in sorted(test_images_dir.iterdir())
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
        rows.append({
            "image_id":    image_id,
            "category_id": int(cls) + 1,
            "bbox":  [float(x1), float(y1), w, h],
            "score": float(score),
        })
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO inference for ClearSAR")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument(
        "--mode", default="both", choices=["test", "holdout", "both"],
        help="holdout: evaluate metrics; test: generate submission; both: run both.",
    )
    parser.add_argument("--mapping-path",       default="catalog.v1.parquet")
    parser.add_argument("--test-images-dir",   default="data/images/test")
    parser.add_argument("--holdout-yaml",       default="data/yolo/holdout.yaml")
    parser.add_argument(
        "--holdout-images-dir", default=None,
        help="Pre-stacked holdout images dir. Overrides the images path from --holdout-yaml. "
             "Labels are still read from the YAML. Example: data/images/holdout_run1",
    )
    parser.add_argument("--run-name", default=None,
        help="Nombre del run (ej: run1). Deriva rutas de output y viz automáticamente.")
    parser.add_argument("--output", default=None,
        help="Ruta de salida del zip. Por defecto: outputs/submission_<run-name>.zip")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-size", type=_parse_image_size, default=640,
        help="Detector input size. Examples: 640 or [512,1024].",
    )
    parser.add_argument(
        "--save-holdout-viz", action="store_true",
        help="Save per-image visualization of GT vs predictions to outputs/holdout_inference/.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    def _resolve(path_str: str, root: Path) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else root / p

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[1]
    )
    run_name        = args.run_name
    mapping_path    = _resolve(args.mapping_path,    project_root)
    test_images_dir = _resolve(args.test_images_dir, project_root)
    holdout_yaml    = _resolve(args.holdout_yaml,    project_root)

    _output_str = args.output or (
        f"outputs/submission_{run_name}.zip" if run_name else "outputs/submission_yolo.zip"
    )
    output_path = _resolve(_output_str, project_root)

    holdout_images_dir = _resolve(args.holdout_images_dir, project_root) if args.holdout_images_dir else None

    model = YOLO(str(Path(args.checkpoint).resolve()))

    print(
        f"[inference] imgsz={args.image_size} | conf={DEFAULT_CONF} | "
        f"iou={DEFAULT_IOU} | max_det={DEFAULT_MAX_DET} | device={args.device}"
    )

    # -----------------------------------------------------------------------
    # Holdout evaluation
    # -----------------------------------------------------------------------
    if args.mode in ("holdout", "both"):
        print("\n" + "=" * 55)
        print("HOLDOUT EVALUATION")
        print("=" * 55)

        _viz_suffix = f"holdout_inference_{run_name}" if run_name else "holdout_inference"
        viz_dir = project_root / "outputs" / _viz_suffix if args.save_holdout_viz else None
        if holdout_images_dir:
            print(f"[holdout] Using pre-stacked images from: {holdout_images_dir}")
        metrics = _evaluate_holdout(
            model                = model,
            holdout_yaml_path    = holdout_yaml,
            imgsz                = args.image_size,
            device               = args.device,
            save_viz             = args.save_holdout_viz,
            viz_output_dir       = viz_dir,
            images_dir_override  = holdout_images_dir,
        )
        print(
            f"[holdout] map50-95={metrics['map50_95']:.4f} | "
            f"map50={metrics['map50']:.4f} | "
            f"map75={metrics['map75']:.4f} | "
            f"recall={metrics['recall']:.4f}"
        )
        if args.save_holdout_viz:
            print(f"[holdout] Visualizations saved to: {viz_dir}")

        if args.mode == "holdout":
            return

    # -----------------------------------------------------------------------
    # Test inference
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("TEST INFERENCE")
    print("=" * 55)

    if not test_images_dir.is_dir():
        raise FileNotFoundError(f"Test images dir not found: {test_images_dir}")

    avg_train = _get_train_avg_boxes(project_root / "data" / "annotations" / "instances_train.json")
    filename_to_id = _load_test_id_mapping(test_images_dir, mapping_path)
    image_files = [
        p for p in sorted(test_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    submission_rows = []
    for img_path in tqdm(image_files, desc="Test inference"):
        if img_path.name not in filename_to_id:
            continue
        image_id = int(filename_to_id[img_path.name])
        boxes    = _predict(model, img_path, args.image_size, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MAX_DET, args.device)
        submission_rows.extend(_boxes_to_submission_rows(boxes, image_id))

    avg_test = len(submission_rows) / len(image_files) if image_files else 0.0
    print(f"\n{'=' * 55}")
    print(f"Train avg boxes/img : {avg_train:.2f}")
    print(f"Test  avg boxes/img : {avg_test:.2f}")
    print(f"Total boxes         : {len(submission_rows)}")
    print(f"{'=' * 55}\n")

    _validate_submission(submission_rows)
    _save_submission(submission_rows, output_path)
    print(f"Submission saved to: {output_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Ejemplos de ejecucion (desde la raiz del proyecto)
# ---------------------------------------------------------------------------
# Basico: corre holdout + test (modo both por defecto).
# python src/inference.py --checkpoint models/yolo_best.pt
#
# Solo evaluacion holdout (sin generar submission).
# python src/inference.py --checkpoint models/yolo_best.pt --mode holdout
#
# Solo inferencia de test para generar submission.
# python src/inference.py --checkpoint models/yolo_best.pt --mode test
#
# Guardar visualizaciones de holdout (GT vs predicciones).
# python src/inference.py --checkpoint models/yolo_best.pt --mode holdout --save-holdout-viz
#
# Cambiar tamano de entrada (cuadrado o rectangular).
# python src/inference.py --checkpoint models/yolo_best.pt --image-size 960
# python src/inference.py --checkpoint models/yolo_best.pt --image-size [512,1024]
#
# Forzar ejecucion en CPU.
# python src/inference.py --checkpoint models/yolo_best.pt --device cpu
#
# Definir salida personalizada (.zip o .json).
# python src/inference.py --checkpoint models/yolo_best.pt --mode test --output outputs/submission_custom.zip