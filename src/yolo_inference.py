import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import default_config, ensure_dirs
from src.dataset import (
    ClearSARTestDataset,
    collate_detection_batch,
    load_test_id_mapping,
    xyxy_to_coco_xywh,
)

from src.submission import save_submission_auto, validate_submission_schema
import json

def get_train_stats(json_path: Path) -> float:
    """Calcula el promedio de cajas por imagen en el dataset de entrenamiento."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    num_images = len(data['images'])
    num_anns = len(data['annotations'])
    avg = num_anns / num_images if num_images > 0 else 0
    return avg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO inference for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--mapping-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.0001) # 0.0005
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--tta", dest="tta", action="store_true")
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.set_defaults(tta=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-det", type=int, default=300, help="Limit detections per image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config(project_root=args.project_root)
    ensure_dirs(cfg)

    train_json = cfg.paths.project_root / "data" / "annotations" / "instances_train.json"
    avg_train = get_train_stats(train_json) if train_json.exists() else 0.0

    # 1. Cargar mapeo oficial (Evita el error de Image ID: 14 vs 3155)
    mapping_path = Path(args.mapping_path) if args.mapping_path else cfg.paths.test_id_mapping_path
    filename_to_image_id = load_test_id_mapping(
        test_images_dir=cfg.paths.test_images_dir,
        mapping_path=mapping_path,
        strict=True,
    )

    # 2. Configurar Dataset y DataLoader igual que en el script que funciona
    test_ds = ClearSARTestDataset(
        images_dir=cfg.paths.test_images_dir,
        filename_to_image_id=filename_to_image_id,
        transforms=None,  # YOLO gestiona sus propios transforms internamente
        preprocess_mode=cfg.inference.preprocess_mode,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.inference.num_workers,
        collate_fn=collate_detection_batch,
    )

    # 3. Cargar Modelo
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Instala ultralytics: pip install ultralytics")

    model = YOLO(args.checkpoint)
    submission_rows: List[Dict[str, Any]] = []

    # 4. Bucle de Inferencia
    for _, metas in tqdm(test_loader, desc="YOLO Inference"):
        # `collate_detection_batch` returns tuples of items per sample.
        # For batch_size=1 we expect a single meta dict inside `metas`.
        meta = metas[0] if isinstance(metas, (list, tuple)) else metas

        # Usamos la ruta original para que YOLO detecte el tamaño real de la imagen
        img_path = str(cfg.paths.test_images_dir / meta["file_name"])

        results = model.predict(
            source=img_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.image_size,
            augment=args.tta,
            device=args.device,
            max_det=args.max_det,
            verbose=False
        )

        result = results[0]
        # YOLO devuelve coordenadas en la escala original si se pasa un path como source
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        image_id = int(meta["image_id"])

        for b, s in zip(boxes, scores):
            # Convertir a COCO [x, y, w, h]
            coco_box = xyxy_to_coco_xywh([b[0], b[1], b[2], b[3]])

            # FILTRO CRÍTICO: Eliminar cajas con ancho o alto <= 0
            if coco_box[2] <= 0.001 or coco_box[3] <= 0.001:
                continue

            submission_rows.append({
                "image_id": image_id,
                "category_id": 1,
                # Sin redondeos, usando float puro como en inference.py
                "bbox": [float(v) for v in coco_box],
                "score": float(s),
            })

    # ESTADÍSTICAS FINALES
    num_test_images = len(test_loader)
    total_boxes = len(submission_rows)
    avg_test = total_boxes / num_test_images if num_test_images > 0 else 0

    print("\n" + "="*40)
    print(f"📊 RESUMEN DE DENSIDAD DE CAJAS")
    print(f"Promedio en TRAIN: {avg_train:.2f} cajas/img")
    print(f"Promedio en TEST (actual): {avg_test:.2f} cajas/img")
    print(f"Total de cajas generadas: {total_boxes}")
    
    if avg_test > avg_train * 10:
        print("⚠️ AVISO: Estás generando >10x más cajas que en train. Podría haber mucho ruido.")
    elif avg_test < avg_train * 0.5:
        print("⚠️ AVISO: Estás generando <50% cajas que en train. Podrías estar perdiendo Recall.")
    print("="*40 + "\n")

    # 5. Validar y Guardar
    output_path = Path(args.output) if args.output else cfg.paths.outputs_dir / "submission.zip"
    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)

    # Mostrar número de filas del JSON de submission generado
    rows_count = len(submission_rows)
    print(f"[yolo] Filas en submission: {rows_count}")

    print(f"[yolo] Inferencia completada. ID de imagen de ejemplo: {submission_rows[0]['image_id']}")
    print(f"[yolo] Archivo generado: {output_path}")


if __name__ == "__main__":
    main()