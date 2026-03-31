import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imports del proyecto
from src.config import default_config, ensure_dirs
from src.dataset import (
    ClearSARTestDataset,
    collate_detection_batch,
    load_test_id_mapping,
    xyxy_to_coco_xywh,
)
from src.submission import save_submission_auto, validate_submission_schema

def get_train_stats(json_path: Path) -> float:
    """Calcula el promedio de cajas por imagen en el dataset de entrenamiento."""
    if not json_path.exists():
        return 0.0
    with open(json_path, 'r') as f:
        data = json.load(f)
    num_images = len(data.get('images', []))
    num_anns = len(data.get('annotations', []))
    return num_anns / num_images if num_images > 0 else 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + SAHI inference for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al archivo best.pt")
    parser.add_argument("--mapping-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0)
    parser.add_argument("--iou", type=float, default=0.7, help="Umbral IOU para NMS")
    parser.add_argument("--max-det", type=int, default=100, dest="max_det", help="Máximo detecciones por imagen")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Parámetros SAHI
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.3)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = default_config(project_root=args.project_root)
    ensure_dirs(cfg)

    # 1. Cargar estadísticas de entrenamiento para referencia
    train_json = cfg.paths.project_root / "data" / "annotations" / "instances_train.json"
    avg_train = get_train_stats(train_json)

    # 2. Mapeo de IDs de imagen
    mapping_path = Path(args.mapping_path) if args.mapping_path else cfg.paths.test_id_mapping_path
    filename_to_image_id = load_test_id_mapping(
        test_images_dir=cfg.paths.test_images_dir,
        mapping_path=mapping_path,
        strict=True,
    )

    # 3. Preparar Dataset
    test_ds = ClearSARTestDataset(
        images_dir=cfg.paths.test_images_dir,
        filename_to_image_id=filename_to_image_id,
        transforms=None,
        preprocess_mode=cfg.inference.preprocess_mode,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.inference.num_workers,
        collate_fn=collate_detection_batch,
    )

    # 4. Inicializar SAHI Detection Model
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        raise ImportError("Por favor instala sahi: pip install sahi")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.checkpoint,
        confidence_threshold=args.conf,
        device=args.device,
    )

    # Aplicar valores directamente al modelo subyacente si está disponible.
    # Esto intenta ajustar `conf`, `iou` y `max_det` en backends como ultralytics.
    try:
        if hasattr(detection_model, "model"):
            try:
                setattr(detection_model.model, "conf", args.conf)
            except Exception:
                pass
            try:
                setattr(detection_model.model, "iou", args.iou)
            except Exception:
                pass
            try:
                setattr(detection_model.model, "max_det", args.max_det)
            except Exception:
                pass
    except Exception:
        pass

    submission_rows: List[Dict[str, Any]] = []

    # 5. Bucle de Inferencia con SAHI
    for _, metas in tqdm(test_loader, desc="SAHI Inference"):
        meta = metas[0] if isinstance(metas, (list, tuple)) else metas
        img_path = str(cfg.paths.test_images_dir / meta["file_name"])

        # Predicción por trozos (Slicing)
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            perform_standard_pred=True,   # Predicción global + por trozos
            postprocess_type="NMM",       # Non-Maximum Merging (mejor para SAR/Objetos densos)
            postprocess_match_threshold=0.5,
        )

        for obj in result.object_prediction_list:
            b = obj.bbox
            # Convertir a formato COCO: [x_min, y_min, width, height]
            coco_box = [b.minx, b.miny, b.maxx - b.minx, b.maxy - b.miny]

            # Filtro de cajas vacías o inválidas
            if coco_box[2] <= 0.001 or coco_box[3] <= 0.001:
                continue

            submission_rows.append({
                "image_id": int(meta["image_id"]),
                "category_id": 1,
                "bbox": [float(v) for v in coco_box],
                "score": float(obj.score.value),
            })

    # 6. Estadísticas Finales
    num_test_images = len(test_loader)
    total_boxes = len(submission_rows)
    avg_test = total_boxes / num_test_images if num_test_images > 0 else 0

    print("\n" + "="*40)
    print(f"📊 RESUMEN DE DENSIDAD (SAHI)")
    print(f"Promedio en TRAIN: {avg_train:.2f} cajas/img")
    print(f"Promedio en TEST:  {avg_test:.2f} cajas/img")
    print(f"Total cajas:       {total_boxes}")
    print("="*40 + "\n")

    # 7. Guardar resultados
    output_path = Path(args.output) if args.output else cfg.paths.outputs_dir / "submission_sahi.zip"
    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)

    print(f"[sahi] Proceso completado. Archivo: {output_path}")

if __name__ == "__main__":
    main()