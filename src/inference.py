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
import os
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    src_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from src.coco_utils import _yolo_to_coco_bbox
from src.dataset import _compute_horizontal_slices, _compute_vertical_slices
from src.submission import (
    _boxes_to_submission_rows,
    _get_train_avg_boxes,
    _load_test_id_mapping,
    _save_submission,
    _validate_submission,
)
from src.utils import _clip_xyxy, _nms_per_class, _result_to_boxes, _str2bool, _valid_bbox
from src.vision import _save_visualization


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS   = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
DEFAULT_CONF = 0.0
DEFAULT_IOU  = 0.5
DEFAULT_MAX_DET = 300
DEFAULT_POSTPROCESS_WORKERS = max(1, min(8, (os.cpu_count() or 1)))


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


def _prepare_slices_for_image(
    img_path: Path,
    slice_height: int,
    slice_overlap: float,
    slice_width: int,
    slice_width_overlap: float,
    slice_height_overlap: float,
) -> tuple[int, int, list[tuple[int, int]], list[np.ndarray]]:
    """Load one image and build stretched slices for model input."""
    with Image.open(img_path) as pil_img:
        img = pil_img.convert("RGB")
        orig_w, orig_h = img.size
        windows = _compute_horizontal_slices(orig_h, slice_height, slice_overlap)
        vertical_windows = _compute_vertical_slices(orig_w, slice_width, slice_width_overlap)

        stretched_slices: list[np.ndarray] = []
        valid_windows: list[tuple[int, int]] = []
        for y0, y1 in windows:
            if y1 <= y0:
                continue
            crop = img.crop((0, y0, orig_w, y1))
            stretched = crop.resize((orig_w, orig_h), resample=Image.BILINEAR)
            stretched_slices.append(np.asarray(stretched))
            valid_windows.append((y0, y1))

    return orig_w, orig_h, valid_windows, stretched_slices


def _predict_slices_prefetch(
    model: YOLO,
    image_files: list[Path],
    imgsz: int | list[int],
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    batch_size: int,
    slice_height: int,
    slice_overlap: float,
    slice_width: int,
    slice_width_overlap: float,
    slice_height_overlap: float,
    prefetch_workers: int = 1,
):
    """Yield per-image raw slice predictions with threaded CPU prefetch."""

    workers = max(1, int(prefetch_workers))

    def _submit_prepare(executor: ThreadPoolExecutor, path: Path):
        return executor.submit(_prepare_slices_for_image, path, slice_height, slice_overlap, slice_width, slice_width_overlap, slice_height_overlap)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        files_iter = iter(image_files)
        try:
            current_path = next(files_iter)
        except StopIteration:
            return

        pending = _submit_prepare(ex, current_path)
        next_path: Path | None = None

        while pending is not None:
            orig_w, orig_h, valid_windows, stretched_slices = pending.result()

            try:
                next_path = next(files_iter)
                next_pending = _submit_prepare(ex, next_path)
            except StopIteration:
                next_path = None
                next_pending = None

            raw_slice_preds: list[tuple[int, int, list[list[float]]]] = []
            if stretched_slices:
                results = model.predict(
                    source=stretched_slices,
                    conf=conf,
                    iou=iou,
                    imgsz=_imgsz_for_ultralytics(imgsz),
                    max_det=max_det,
                    batch=min(batch_size, len(stretched_slices)),
                    augment=False,
                    device=device,
                    verbose=False,
                )
                for (y0, y1), result in zip(valid_windows, results):
                    raw_slice_preds.append((y0, y1, _result_to_boxes(result)))

            yield current_path, orig_w, orig_h, raw_slice_preds

            current_path = next_path
            pending = next_pending




def _predict_slices_raw(
    model: YOLO,
    img_path: Path,
    imgsz: int | list[int],
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    batch_size: int,
    slice_height: int,
    slice_overlap: float,
    slice_width: int,
    slice_width_overlap: float,
    slice_height_overlap: float,
) -> tuple[int, int, list[tuple[int, int, list[list[float]]]]]:
    """Run model over stretched horizontal slices and keep raw per-slice detections."""
    orig_w, orig_h, valid_windows, stretched_slices = _prepare_slices_for_image(
        img_path=img_path,
        slice_height=slice_height,
        slice_overlap=slice_overlap,
        slice_width=slice_width,
        slice_width_overlap=slice_width_overlap,
        slice_height_overlap=slice_height_overlap,
    )

    if not stretched_slices:
        return orig_w, orig_h, []

    results = model.predict(
        source  = stretched_slices,
        conf    = conf,
        iou     = iou,
        imgsz   = _imgsz_for_ultralytics(imgsz),
        max_det = max_det,
        batch   = min(batch_size, len(stretched_slices)),
        augment = False,
        device  = device,
        verbose = False,
    )

    raw: list[tuple[int, int, list[list[float]]]] = []
    for (y0, y1), result in zip(valid_windows, results):
        raw.append((y0, y1, _result_to_boxes(result)))
    return orig_w, orig_h, raw


def _postprocess_slices_to_boxes(
    orig_w: int,
    orig_h: int,
    raw_slice_preds: list[tuple[int, int, list[list[float]]]],
    iou: float,
) -> list[list[float]]:
    all_boxes: list[list[float]] = []
    for y0, y1, slice_boxes in raw_slice_preds:
        if y1 <= y0 or not slice_boxes:
            continue

        scale_y = (y1 - y0) / max(1.0, float(orig_h))
        for x1, y1p, x2, y2p, score, cls in slice_boxes:
            y1_orig = float(y0) + float(y1p) * scale_y
            y2_orig = float(y0) + float(y2p) * scale_y
            x1c, y1c, x2c, y2c = _clip_xyxy(
                float(x1), y1_orig, float(x2), y2_orig, orig_w, orig_h
            )
            if x2c <= x1c or y2c <= y1c:
                continue
            all_boxes.append([x1c, y1c, x2c, y2c, float(score), int(cls)])

    return _nms_per_class(all_boxes, iou_thr=iou)


def _postprocess_slices_worker(
    task: tuple[int, int, list[tuple[int, int, list[list[float]]]], float],
) -> list[list[float]]:
    orig_w, orig_h, raw_slice_preds, iou = task
    return _postprocess_slices_to_boxes(orig_w, orig_h, raw_slice_preds, iou)


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
    batch_size: int = 1,
    slicing: bool = False,
    slice_height: int = 256,
    slice_overlap: float = 0.2,
    slice_width: int = 256,
    slice_width_overlap: float = 0.2,
    slice_height_overlap: float = 0.2,
) -> list[list[float]]:
    batch_size = max(1, int(batch_size))

    if not slicing:
        results = model.predict(
            source  = str(img_path),
            conf    = conf,
            iou     = iou,
            imgsz   = _imgsz_for_ultralytics(imgsz),
            max_det = max_det,
            batch   = batch_size,
            augment = False,
            device  = device,
            verbose = False,
        )
        return _result_to_boxes(results[0] if results else None)

    orig_w, orig_h, raw_slice_preds = _predict_slices_raw(
        model=model,
        img_path=img_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        batch_size=batch_size,
        slice_height=slice_height,
        slice_overlap=slice_overlap,
        slice_width=slice_width,
        slice_width_overlap=slice_width_overlap,
        slice_height_overlap=slice_height_overlap,
    )
    return _postprocess_slices_to_boxes(orig_w, orig_h, raw_slice_preds, iou=iou)


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
        evaluator.params.maxDets = [1, 10, 100, DEFAULT_MAX_DET]
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
    batch_size: int = 1,
    postprocess_workers: int = DEFAULT_POSTPROCESS_WORKERS,
    slicing: bool = False,
    slice_height: int = 256,
    slice_overlap: float = 0.2,
    slice_width: int = 256,
    slice_width_overlap: float = 0.2,
    slice_height_overlap: float = 0.2,
    prefetch_workers: int = 1,
    save_viz: bool = False,
    viz_output_dir: Path | None = None,
) -> dict[str, float]:
    images_dir, labels_dir, class_names = _parse_holdout_yaml(holdout_yaml_path)
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
    if slicing:
        pred_jobs: list[tuple[int, Path, int, int, list[tuple[int, int, list[list[float]]]]]] = []
        for img_path, orig_w, orig_h, raw_slice_preds in tqdm(
            _predict_slices_prefetch(
                model=model,
                image_files=image_files,
                imgsz=imgsz,
                conf=DEFAULT_CONF,
                iou=DEFAULT_IOU,
                max_det=DEFAULT_MAX_DET,
                device=device,
                batch_size=batch_size,
                slice_height=slice_height,
                slice_overlap=slice_overlap,
                slice_width=slice_width,
                slice_width_overlap=slice_width_overlap,
                slice_height_overlap=slice_height_overlap,
                prefetch_workers=prefetch_workers,
            ),
            total=len(image_files),
            desc="Holdout slice prediction",
        ):
            image_id = filename_to_id[img_path.name]
            pred_jobs.append((image_id, img_path, orig_w, orig_h, raw_slice_preds))

        boxes_by_image_id: dict[int, list[list[float]]] = {}
        tasks = [
            (orig_w, orig_h, raw_slice_preds, DEFAULT_IOU)
            for _image_id, _img_path, orig_w, orig_h, raw_slice_preds in pred_jobs
        ]
        with ProcessPoolExecutor(max_workers=max(1, int(postprocess_workers))) as ex:
            fut_to_image_id = {
                ex.submit(_postprocess_slices_worker, task): pred_jobs[idx][0]
                for idx, task in enumerate(tasks)
            }
            for fut in tqdm(as_completed(fut_to_image_id), total=len(fut_to_image_id), desc="Holdout postprocess"):
                image_id = fut_to_image_id[fut]
                boxes_by_image_id[image_id] = fut.result()

        for image_id, img_path, _orig_w, _orig_h, _raw in pred_jobs:
            boxes = boxes_by_image_id.get(image_id, [])
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
    else:
        for img_path in tqdm(image_files, desc="Holdout inference"):
            image_id = filename_to_id[img_path.name]
            boxes = _predict(
                model,
                img_path,
                imgsz,
                DEFAULT_CONF,
                DEFAULT_IOU,
                DEFAULT_MAX_DET,
                device,
                batch_size=batch_size,
                slicing=False,
                slice_height=slice_height,
                slice_overlap=slice_overlap,
            )

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
    parser.add_argument("--mapping-path",    default="catalog.v1.parquet")
    parser.add_argument("--test-images-dir", default="data/images/test")
    parser.add_argument("--holdout-yaml",    default="data/yolo/holdout.yaml")
    parser.add_argument("--output",          default="outputs/submission_yolo.zip")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-size", type=_parse_image_size, default=640,
        help="Detector input size. Examples: 640 or [512,1024].",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for prediction (especially useful with slicing).",
    )
    parser.add_argument(
        "--postprocess-workers", type=int, default=DEFAULT_POSTPROCESS_WORKERS,
        help="Number of worker processes for CPU postprocessing in slicing mode.",
    )
    parser.add_argument(
        "--prefetch-workers", type=int, default=1,
        help="Number of threads to prefetch image loading/resize while GPU infers (slicing mode).",
    )
    parser.add_argument(
        "--slicing", type=_str2bool, default=False,
        help="If true, use horizontal slicing + stretch-back pipeline before prediction.",
    )
    parser.add_argument(
        "--slice-height", type=int, default=256,
        help="Slice height in pixels for horizontal slicing.",
    )
    parser.add_argument(
        "--slice-overlap", type=float, default=0.2,
        help="Fractional overlap between consecutive slices (0.0 to <1.0).",
    )
    parser.add_argument(
        "--slice-width", type=int, default=256,
        help="Slice width in pixels for vertical slicing.",
    )
    parser.add_argument(
        "--slice-width-overlap", type=float, default=0.2,
        help="Fractional overlap between consecutive vertical slices (0.0 to <1.0).",
    )
    parser.add_argument(
        "--slice-height-overlap", type=float, default=0.2,
        help="Fractional overlap between consecutive horizontal slices (0.0 to <1.0).",
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
    mapping_path     = _resolve(args.mapping_path,    project_root)
    test_images_dir  = _resolve(args.test_images_dir, project_root)
    holdout_yaml     = _resolve(args.holdout_yaml,    project_root)
    output_path      = _resolve(args.output,          project_root)

    model = YOLO(str(Path(args.checkpoint).resolve()))

    print(
        f"[inference] imgsz={args.image_size} | conf={DEFAULT_CONF} | "
        f"iou={DEFAULT_IOU} | max_det={DEFAULT_MAX_DET} | device={args.device} | "
        f"batch={args.batch_size} | pp_workers={args.postprocess_workers} | slicing={args.slicing} | "
        f"slice_h={args.slice_height} | slice_ov={args.slice_overlap} | slice_w={args.slice_width} | slice_w_ov={args.slice_width_overlap} | slice_h_ov={args.slice_height_overlap}"
    )

    # -----------------------------------------------------------------------
    # Holdout evaluation
    # -----------------------------------------------------------------------
    if args.mode in ("holdout", "both"):
        print("\n" + "=" * 55)
        print("HOLDOUT EVALUATION")
        print("=" * 55)

        viz_dir = project_root / "outputs" / "holdout_inference" if args.save_holdout_viz else None
        metrics = _evaluate_holdout(
            model             = model,
            holdout_yaml_path = holdout_yaml,
            imgsz             = args.image_size,
            device            = args.device,
            batch_size        = args.batch_size,
            postprocess_workers = args.postprocess_workers,
            slicing           = args.slicing,
            slice_height      = args.slice_height,
            slice_overlap     = args.slice_overlap,
            slice_width       = args.slice_width,
            slice_width_overlap = args.slice_width_overlap,
            slice_height_overlap = args.slice_height_overlap,
            prefetch_workers  = args.prefetch_workers,
            save_viz          = args.save_holdout_viz,
            viz_output_dir    = viz_dir,
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
    if args.slicing:
        pred_jobs: list[tuple[int, int, int, list[tuple[int, int, list[list[float]]]]]] = []
        for img_path, orig_w, orig_h, raw_slice_preds in tqdm(
            _predict_slices_prefetch(
                model=model,
                image_files=image_files,
                imgsz=args.image_size,
                conf=DEFAULT_CONF,
                iou=DEFAULT_IOU,
                max_det=DEFAULT_MAX_DET,
                device=args.device,
                batch_size=args.batch_size,
                slice_height=args.slice_height,
                slice_overlap=args.slice_overlap,
                slice_width=args.slice_width,
                slice_width_overlap=args.slice_width_overlap,
                slice_height_overlap=args.slice_height_overlap,
                prefetch_workers=args.prefetch_workers,
            ),
            total=len(image_files),
            desc="Test slice prediction",
        ):
            if img_path.name not in filename_to_id:
                continue
            image_id = int(filename_to_id[img_path.name])
            pred_jobs.append((image_id, orig_w, orig_h, raw_slice_preds))

        tasks = [
            (orig_w, orig_h, raw_slice_preds, DEFAULT_IOU)
            for _image_id, orig_w, orig_h, raw_slice_preds in pred_jobs
        ]
        with ProcessPoolExecutor(max_workers=max(1, int(args.postprocess_workers))) as ex:
            fut_to_image_id = {
                ex.submit(_postprocess_slices_worker, task): pred_jobs[idx][0]
                for idx, task in enumerate(tasks)
            }
            for fut in tqdm(as_completed(fut_to_image_id), total=len(fut_to_image_id), desc="Test postprocess"):
                image_id = fut_to_image_id[fut]
                boxes = fut.result()
                submission_rows.extend(_boxes_to_submission_rows(boxes, image_id))
    else:
        for img_path in tqdm(image_files, desc="Test inference"):
            if img_path.name not in filename_to_id:
                continue
            image_id = int(filename_to_id[img_path.name])
            boxes    = _predict(
                model,
                img_path,
                args.image_size,
                DEFAULT_CONF,
                DEFAULT_IOU,
                DEFAULT_MAX_DET,
                args.device,
                batch_size=args.batch_size,
                slicing=False,
                slice_height=args.slice_height,
                slice_overlap=args.slice_overlap,
            )
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