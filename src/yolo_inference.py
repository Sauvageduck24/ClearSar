from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
DEFAULT_CONF = 0.005
DEFAULT_IOU = 0.7
DEFAULT_MAX_DET = 500
MIN_BOX_DIM = 0.001

def _parse_image_size(value: str) -> int | list[int]:
    raw = str(value).strip()
    if not raw:
        raise argparse.ArgumentTypeError("--image-size no puede estar vacio")

    if raw.isdigit():
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("--image-size debe ser > 0")
        return parsed

    parts = [p.strip() for p in raw.strip("()[]").split(",") if p.strip()]
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        h, w = int(parts[0]), int(parts[1])
        if h <= 0 or w <= 0:
            raise argparse.ArgumentTypeError("--image-size requiere valores > 0")
        return [h, w]

    raise argparse.ArgumentTypeError(
        "Formato invalido para --image-size. Usa 640 o [512,1024] / (512,1024)."
    )


def _normalize_imgsz(imgsz: int | list[int]) -> tuple[int, int]:
    if isinstance(imgsz, list):
        if len(imgsz) != 2:
            raise ValueError("--image-size como lista debe tener exactamente 2 valores")
        return int(imgsz[0]), int(imgsz[1])
    return int(imgsz), int(imgsz)


def _imgsz_for_ultralytics(imgsz: int | list[int]) -> int | list[int]:
    h, w = _normalize_imgsz(imgsz)
    return h if h == w else [h, w]


def _result_boxes_as_list(result: Any) -> list[list[float]]:
    boxes_as_list: list[list[float]] = []
    if result is None or result.boxes is None:
        return boxes_as_list

    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        score = float(box.conf[0].item()) if box.conf is not None else 0.0
        class_id = int(box.cls[0].item()) if box.cls is not None else 0
        boxes_as_list.append([x1, y1, x2, y2, score, class_id])

    return boxes_as_list


def _merged_submission_rows(
    pred_result: Any,
    image_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    merged_boxes = _result_boxes_as_list(pred_result)
    for x1, y1, x2, y2, score, class_id in merged_boxes:
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        if not _valid_bbox(w, h):
            continue
        rows.append(
            {
                "image_id": image_id,
                "category_id": int(class_id) + 1,
                "bbox": [float(x1), float(y1), w, h],
                "score": float(score),
            }
        )
    return rows


def _valid_bbox(w: float, h: float) -> bool:
    return w > MIN_BOX_DIM and h > MIN_BOX_DIM


def _yolo_box_to_xywh_and_score(box: Any) -> tuple[list[float], float]:
    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
    score = float(box.conf[0].item()) if box.conf is not None else 0.0
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], score


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
    x = (x_center * float(image_width)) - (w / 2.0)
    y = (y_center * float(image_height)) - (h / 2.0)
    return [x, y, w, h]


def _parse_holdout_yaml(holdout_yaml_path: Path) -> tuple[Path, Path, list[str]]:
    if not holdout_yaml_path.exists():
        raise FileNotFoundError(f"No existe holdout-yaml: {holdout_yaml_path}")

    payload = yaml.safe_load(holdout_yaml_path.read_text(encoding="utf-8")) or {}
    root = Path(payload.get("path", holdout_yaml_path.parent))
    if not root.is_absolute():
        root = (holdout_yaml_path.parent / root).resolve()

    val_value = str(payload.get("val", "images/val"))
    images_dir = (root / val_value).resolve()
    labels_dir = (root / val_value.replace("images", "labels", 1)).resolve()

    names = payload.get("names", [])
    if isinstance(names, dict):
        ordered_keys = sorted(names.keys(), key=lambda k: int(k))
        class_names = [str(names[k]) for k in ordered_keys]
    else:
        class_names = [str(name) for name in names]

    if not class_names:
        num_classes = int(payload.get("nc", 0))
        class_names = [f"class_{i}" for i in range(num_classes)]

    if not images_dir.is_dir():
        raise FileNotFoundError(f"No existe carpeta de imagenes holdout: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"No existe carpeta de labels holdout: {labels_dir}")

    return images_dir, labels_dir, class_names


def _build_holdout_coco_gt(
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
) -> tuple[dict[str, Any], dict[str, int], list[Path]]:
    image_files = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not image_files:
        raise ValueError(f"No hay imagenes en holdout: {images_dir}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    filename_to_image_id: dict[str, int] = {}
    ann_id = 1

    for idx, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as image_obj:
            img_w, img_h = image_obj.size

        image_id = int(img_path.stem) if img_path.stem.isdigit() else idx
        filename_to_image_id[img_path.name] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            }
        )

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:5])
            x, y, w, h = _yolo_to_coco_bbox(x_center, y_center, width, height, img_w, img_h)
            if not _valid_bbox(w, h):
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    coco_gt = {"images": images, "annotations": annotations, "categories": categories}
    return coco_gt, filename_to_image_id, image_files


def _evaluate_holdout_with_pycocotools(
    model: YOLO,
    holdout_yaml_path: Path,
    imgsz: int | list[int],
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> dict[str, dict[str, float]]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError(
            "pycocotools no esta instalado. Instala con: pip install pycocotools"
        ) from exc

    images_dir, labels_dir, class_names = _parse_holdout_yaml(holdout_yaml_path)
    coco_gt, filename_to_image_id, image_files = _build_holdout_coco_gt(images_dir, labels_dir, class_names)

    def _compute_coco_metrics(pred_rows: list[dict[str, Any]]) -> dict[str, float]:
        with tempfile.TemporaryDirectory(prefix="clearsar_holdout_eval_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            gt_json = tmp_path / "gt.json"
            pred_json = tmp_path / "pred.json"
            gt_json.write_text(json.dumps(coco_gt), encoding="utf-8")
            pred_json.write_text(json.dumps(pred_rows), encoding="utf-8")

            coco_gt_api = COCO(str(gt_json))
            coco_dt_api = coco_gt_api.loadRes(str(pred_json))
            coco_eval = COCOeval(coco_gt_api, coco_dt_api, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        return {
            "map50_95": float(coco_eval.stats[0]),
            "map50": float(coco_eval.stats[1]),
            "map75": float(coco_eval.stats[2]),
            "recall": float(coco_eval.stats[8]),
        }

    coco_preds: list[dict[str, Any]] = []
    for img_path in tqdm(image_files, desc="Holdout (pycocotools)"):
        image_id = filename_to_image_id[img_path.name]
        pred_results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=_imgsz_for_ultralytics(imgsz),
            max_det=max_det,
            augment=False,
            device=device,
            verbose=False,
        )
        if not pred_results or pred_results[0].boxes is None:
            continue

        for box in pred_results[0].boxes:
            bbox_xywh, score = _yolo_box_to_xywh_and_score(box)
            _, _, w, h = bbox_xywh
            if not _valid_bbox(w, h):
                continue
            coco_preds.append(
                {
                    "image_id": image_id,
                    "category_id": int(box.cls[0].item()) + 1,
                    "bbox": [float(v) for v in bbox_xywh],
                    "score": float(score),
                }
            )

    vertical_preds = [r for r in coco_preds if r["bbox"][3] > r["bbox"][2]]
    print(f"[holdout/pycoco] Boxes verticales: {len(vertical_preds)} de {len(coco_preds)}")

    print("\n[holdout/pycoco] mAP SIN split_vertical_predictions")
    metrics_without_split = _compute_coco_metrics(coco_preds)

    return {
        "without_split": metrics_without_split,
    }


def validate_submission_schema(rows: list[dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        for key in ("image_id", "category_id", "bbox", "score"):
            if key not in row:
                raise ValueError(f"Fila {i} sin key requerida '{key}'")
        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Fila {i} tiene bbox invalido: {bbox}")


def save_submission_auto(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows)

    if output_path.suffix.lower() == ".json":
        output_path.write_text(payload, encoding="utf-8")
        return

    zip_path = output_path if output_path.suffix.lower() == ".zip" else output_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", payload)


def get_train_stats(json_path: Path) -> float:
    if not json_path.exists():
        return 0.0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    num_images = len(data.get("images", []))
    num_anns = len(data.get("annotations", []))
    return num_anns / num_images if num_images else 0.0


def _load_test_id_mapping(test_images_dir: Path, mapping_path: Path) -> dict[str, int]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"No existe mapping-path: {mapping_path}")

    df = pd.read_parquet(mapping_path)
    if "id" not in df.columns:
        raise ValueError(f"El parquet no contiene columna 'id': {mapping_path}")

    mapping = {
        Path(normalized).name: int(Path(normalized).stem)
        for item_id in df["id"].astype(str)
        if "/images/test/" in (normalized := item_id.replace("\\", "/"))
        and Path(normalized).stem.isdigit()
    }

    test_files = [p.name for p in sorted(test_images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    missing = [name for name in test_files if name not in mapping]
    if missing:
        raise ValueError(
            "Faltan IDs para imagenes de test en el mapping parquet. "
            f"Ejemplos: {', '.join(missing[:5])}"
        )

    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO inference for ClearSAR")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--checkpoint", required=True, help="Ruta al archivo best.pt")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["test", "holdout", "both"],
        help="Modo de ejecucion: holdout evalua metricas; test genera submission; both ejecuta holdout y luego test.",
    )
    parser.add_argument("--mapping-path", default="catalog.v1.parquet")
    parser.add_argument("--test-images-dir", default="data/images/test")
    parser.add_argument("--holdout-yaml", default="data/yolo/holdout.yaml")
    parser.add_argument("--output", default="outputs/submission_yolo.zip")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--image-size",
        type=_parse_image_size,
        default=640,
        help="Tamano de entrada del detector. Ejemplos: 640 o [512,1024].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def _resolve(path_str: str, root: Path) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else root / p

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    mapping_path = _resolve(args.mapping_path, project_root)
    test_images_dir = _resolve(args.test_images_dir, project_root)
    holdout_yaml_path = _resolve(args.holdout_yaml, project_root)
    output_path = _resolve(args.output, project_root)
    checkpoint_path = str(Path(args.checkpoint).resolve())
    best_iou, best_max_det = DEFAULT_IOU, DEFAULT_MAX_DET

    detection_model = YOLO(checkpoint_path)
    print(
        f"[inference] imgsz={args.image_size} | conf={DEFAULT_CONF} | iou={best_iou} "
        f"| max_det={best_max_det}"
    )

    if args.mode in ("holdout", "both"):
        print("\n" + "=" * 40)
        print("EVALUACION HOLDOUT (modo test-like)")
        print("=" * 40)
        holdout_metrics = _evaluate_holdout_with_pycocotools(
            model=detection_model,
            holdout_yaml_path=holdout_yaml_path,
            imgsz=args.image_size,
            conf=DEFAULT_CONF,
            iou=best_iou,
            max_det=best_max_det,
            device=args.device,
        )
        base = holdout_metrics["without_split"]
        print(
            f"[holdout/pycoco][sin split] map50-95={base['map50_95']:.4f} "
            f"| map50={base['map50']:.4f} | map75={base['map75']:.4f} | recall={base['recall']:.4f}"
        )
        
        if args.mode == "holdout":
            return

        print("\n" + "=" * 40)
        print("INFERENCIA EN TEST")
        print("=" * 40)

    if not test_images_dir.is_dir():
        raise FileNotFoundError(f"No existe test-images-dir: {test_images_dir}")

    avg_train = get_train_stats(project_root / "data" / "annotations" / "instances_train.json")
    filename_to_image_id = _load_test_id_mapping(test_images_dir, mapping_path)
    image_files = [p for p in sorted(test_images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    submission_rows: list[dict[str, Any]] = []

    for img_path in tqdm(image_files, desc="Inference"):
        if img_path.name not in filename_to_image_id:
            continue

        image_id = int(filename_to_image_id[img_path.name])
        pred_results = detection_model.predict(
            source=str(img_path),
            conf=DEFAULT_CONF,
            iou=best_iou,
            imgsz=_imgsz_for_ultralytics(args.image_size),
            max_det=best_max_det,
            augment=False,
            device=args.device,
            verbose=False,
        )

        if not pred_results:
            continue

        submission_rows.extend(
            _merged_submission_rows(
                pred_result=pred_results[0],
                image_id=image_id,
            )
        )

    total_boxes = len(submission_rows)
    avg_test = total_boxes / len(image_files) if image_files else 0.0

    print(f"\n{'=' * 40}")
    print("RESUMEN DE DENSIDAD")
    print(f"Promedio en TRAIN: {avg_train:.2f} cajas/img")
    print(f"Promedio en TEST:  {avg_test:.2f} cajas/img")
    print(f"Total cajas:       {total_boxes}")
    print(f"{'=' * 40}\n")

    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)
    print(f"Proceso completado. Archivo guardado en: {output_path}")


if __name__ == "__main__":
    main()
