from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _parse_image_size(value: str) -> int | List[int]:
    raw = str(value).strip()
    if not raw:
        raise argparse.ArgumentTypeError("--image-size no puede estar vacio")

    if raw.isdigit():
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("--image-size debe ser > 0")
        return parsed

    cleaned = raw.strip("()[]")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        h, w = int(parts[0]), int(parts[1])
        if h <= 0 or w <= 0:
            raise argparse.ArgumentTypeError("--image-size requiere valores > 0")
        return [h, w]

    raise argparse.ArgumentTypeError(
        "Formato invalido para --image-size. Usa 640 o [512,1024] / (512,1024)."
    )


def _normalize_imgsz(imgsz: int | List[int]) -> tuple[int, int]:
    if isinstance(imgsz, list):
        if len(imgsz) != 2:
            raise ValueError("--image-size como lista debe tener exactamente 2 valores")
        return int(imgsz[0]), int(imgsz[1])
    return int(imgsz), int(imgsz)


def get_train_stats(json_path: Path) -> float:
    if not json_path.exists():
        return 0.0
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    num_images = len(data.get("images", []))
    num_anns = len(data.get("annotations", []))
    return num_anns / num_images if num_images > 0 else 0.0


def _load_test_id_mapping(test_images_dir: Path, mapping_path: Path) -> dict[str, int]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"No existe mapping-path: {mapping_path}")

    df = pd.read_parquet(mapping_path)
    if "id" not in df.columns:
        raise ValueError(f"El parquet no contiene columna 'id': {mapping_path}")

    mapping: dict[str, int] = {}
    for item_id in df["id"].astype(str).tolist():
        normalized = item_id.replace("\\", "/")
        if "/images/test/" not in normalized:
            continue

        file_name = Path(normalized).name
        stem = Path(file_name).stem
        if not stem.isdigit():
            continue

        mapping[file_name] = int(stem)

    test_files = [
        p.name
        for p in sorted(test_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    missing = [name for name in test_files if name not in mapping]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Faltan IDs para imagenes de test en el mapping parquet. "
            f"Ejemplos: {preview}"
        )

    return mapping


def _prediction_to_xywh_and_score(prediction: Any) -> tuple[list[float], float]:
    bbox_obj = prediction.bbox
    if hasattr(bbox_obj, "to_xywh"):
        x, y, w, h = bbox_obj.to_xywh()
    else:
        x = float(bbox_obj.minx)
        y = float(bbox_obj.miny)
        w = float(bbox_obj.maxx - bbox_obj.minx)
        h = float(bbox_obj.maxy - bbox_obj.miny)

    score_obj = prediction.score
    if hasattr(score_obj, "value"):
        score = float(score_obj.value)
    else:
        score = float(score_obj)

    return [float(x), float(y), float(w), float(h)], score


def _yolo_box_to_xywh_and_score(box: Any) -> tuple[list[float], float]:
    xyxy = box.xyxy[0].tolist()
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    score = float(box.conf[0].item()) if box.conf is not None else 0.0
    return [x1, y1, w, h], score


def validate_submission_schema(rows: list[dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        for key in ["image_id", "category_id", "bbox", "score"]:
            if key not in row:
                raise ValueError(f"Fila {i} sin key requerida '{key}'")

        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Fila {i} tiene bbox invalido: {bbox}")


def save_submission_auto(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(rows), encoding="utf-8")
        return

    if output_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("submission.json", json.dumps(rows))
        return

    # Por defecto, guardar como .zip con submission.json
    zip_path = output_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO inference for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al archivo best.pt")
    parser.add_argument("--mapping-path", type=str, default="catalog.v1.parquet")
    parser.add_argument("--test-images-dir", type=str, default="data/images/test")
    parser.add_argument("--output", type=str, default="outputs/submission_yolo.zip")
    parser.add_argument("--conf", type=float, default=0.1, help="Umbral de confianza")
    parser.add_argument("--iou", type=float, default=0.5, help="Umbral IOU para fusion entre slices")
    parser.add_argument("--max-det", type=int, default=500, help="Maximo detecciones por imagen")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="sahi",
        choices=["sahi", "original"],
        help="Modo de inferencia: 'sahi' usa slicing, 'original' usa imagen completa.",
    )
    parser.add_argument("--slice-size", type=int, default=256, help="Tamano de cada slice para inferencia SAHI")
    parser.add_argument("--overlap-ratio", type=float, default=0.25, help="Overlap relativo entre slices")
    parser.add_argument(
        "--postprocess-type",
        type=str,
        default="GREEDYNMM",
        choices=["NMM", "NMS", "GREEDYNMM", "LSNMS"],
        help="Metodo de fusion de predicciones entre slices.",
    )
    parser.add_argument(
        "--image-size",
        type=_parse_image_size,
        default=640,
        help="Tamano de entrada del detector por slice. Ejemplos: 640 o [512,1024] / (512,1024).",
    )
    return parser.parse_args()


def main() -> None:
    AutoDetectionModel = None
    get_sliced_prediction = None
    try:
        from sahi import AutoDetectionModel as _AutoDetectionModel
        from sahi.predict import get_sliced_prediction as _get_sliced_prediction
        AutoDetectionModel = _AutoDetectionModel
        get_sliced_prediction = _get_sliced_prediction
    except ImportError:
        # Solo requerido si se usa --inference-mode sahi.
        pass

    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]

    mapping_path = Path(args.mapping_path)
    if not mapping_path.is_absolute():
        mapping_path = project_root / mapping_path

    test_images_dir = Path(args.test_images_dir)
    if not test_images_dir.is_absolute():
        test_images_dir = project_root / test_images_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    if not test_images_dir.exists() or not test_images_dir.is_dir():
        raise FileNotFoundError(f"No existe test-images-dir: {test_images_dir}")

    train_json = project_root / "data" / "annotations" / "instances_train.json"
    avg_train = get_train_stats(train_json)

    filename_to_image_id = _load_test_id_mapping(
        test_images_dir=test_images_dir,
        mapping_path=mapping_path,
    )

    image_files = [
        p
        for p in sorted(test_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    checkpoint_path = str(Path(args.checkpoint).resolve())

    if args.inference_mode == "sahi":
        if AutoDetectionModel is None or get_sliced_prediction is None:
            raise ImportError(
                "SAHI no esta instalado. Instala dependencias con: pip install sahi ultralytics"
            )
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=checkpoint_path,
            confidence_threshold=args.conf,
            device=args.device,
            image_size=_normalize_imgsz(args.image_size),
        )
    else:
        detection_model = YOLO(checkpoint_path)

    submission_rows: List[Dict[str, Any]] = []

    if args.inference_mode == "sahi":
        print(
            f"[sahi] slice={args.slice_size} | overlap={args.overlap_ratio} | "
            f"imgsz={args.image_size} | conf={args.conf} | iou={args.iou} | post={args.postprocess_type}"
        )
    else:
        print(
            f"[original] imgsz={args.image_size} | conf={args.conf} | iou={args.iou} | max_det={args.max_det}"
        )

    for img_path in tqdm(image_files, desc="Inference"):
        if img_path.name not in filename_to_image_id:
            continue

        if args.inference_mode == "sahi":
            result = get_sliced_prediction(
                image=str(img_path),
                detection_model=detection_model,
                slice_height=args.slice_size,
                slice_width=args.slice_size,
                overlap_height_ratio=args.overlap_ratio,
                overlap_width_ratio=args.overlap_ratio,
                perform_standard_pred=False,
                postprocess_type=args.postprocess_type,
                postprocess_match_metric="IOU",
                postprocess_match_threshold=args.iou,
                postprocess_class_agnostic=True,
                verbose=0,
            )

            object_predictions = result.object_prediction_list[: args.max_det]

            for pred in object_predictions:
                bbox, score = _prediction_to_xywh_and_score(pred)
                x, y, w, h = bbox
                if w <= 0.001 or h <= 0.001:
                    continue

                submission_rows.append(
                    {
                        "image_id": int(filename_to_image_id[img_path.name]),
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "score": score,
                    }
                )
        else:
            image_size = _normalize_imgsz(args.image_size)
            pred_results = detection_model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=image_size,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )

            if not pred_results:
                continue

            result = pred_results[0]
            for box in result.boxes:
                bbox, score = _yolo_box_to_xywh_and_score(box)
                x, y, w, h = bbox
                if w <= 0.001 or h <= 0.001:
                    continue

                submission_rows.append(
                    {
                        "image_id": int(filename_to_image_id[img_path.name]),
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "score": score,
                    }
                )

    num_test_images = len(image_files)
    total_boxes = len(submission_rows)
    avg_test = total_boxes / num_test_images if num_test_images > 0 else 0.0

    print("\n" + "=" * 40)
    print("RESUMEN DE DENSIDAD")
    print(f"Promedio en TRAIN: {avg_train:.2f} cajas/img")
    print(f"Promedio en TEST:  {avg_test:.2f} cajas/img")
    print(f"Total cajas:       {total_boxes}")
    print("=" * 40 + "\n")

    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)
    print(f"Proceso completado. Archivo guardado en: {output_path}")


if __name__ == "__main__":
    main()
