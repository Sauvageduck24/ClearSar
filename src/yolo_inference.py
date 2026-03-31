import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from torchvision.ops import nms

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from PIL import Image

# Imports del proyecto
from src.config import default_config, ensure_dirs
from src.dataset import (
    ClearSARTestDataset,
    collate_detection_batch,
    load_test_id_mapping,
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

def tta_predict(model, img_path, conf, iou, max_det, imgsz, device):
    """
    TTA manual con flips y NMS global para consolidar detecciones.
    """
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]

    # Definir variantes y funciones de inversión de coordenadas
    variants = [
        (img,                                    lambda b: b),                          
        (np.fliplr(img),                         lambda b: [W-b[2], b[1], W-b[0], b[3]]),  
        (np.flipud(img),                         lambda b: [b[0], H-b[3], b[2], H-b[1]]),  
        (np.fliplr(np.flipud(img)),              lambda b: [W-b[2], H-b[3], W-b[0], H-b[1]]),  
    ]

    all_boxes, all_scores = [], []

    for aug_img, inv_fn in variants:
        res = model.predict(
            source=aug_img,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        if res.boxes is not None:
            for box, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                inv_box = inv_fn(box.tolist())
                all_boxes.append(inv_box)
                all_scores.append(float(score))

    if not all_boxes:
        return [], []

    # --- NUEVO: NMS Global ---
    # Convertir a tensores para procesar con torchvision
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    
    # Aplicar NMS para eliminar solapamientos entre las diferentes versiones del TTA
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=iou)
    
    # Filtrar y limitar al máximo de detecciones permitidas
    final_boxes = boxes_tensor[keep_indices].numpy()[:max_det]
    final_scores = scores_tensor[keep_indices].numpy()[:max_det]

    return final_boxes, final_scores

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO inference for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al archivo best.pt")
    parser.add_argument("--mapping-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    parser.add_argument("--iou", type=float, default=0.7, help="Umbral IOU para NMS")
    parser.add_argument("--max-det", type=int, default=500, help="Máximo detecciones por imagen")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=640, help="Tamaño de entrada del modelo")
    
    # Flags de TTA (Test Time Augmentation)
    parser.add_argument("--tta", action="store_true", help="Activar Test Time Augmentation")
    parser.set_defaults(tta=False)

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

    # 4. Inicializar modelo Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Por favor instala ultralytics: pip install ultralytics")

    model = YOLO(args.checkpoint)
    submission_rows: List[Dict[str, Any]] = []

    # 5. Bucle de Inferencia
    for _, metas in tqdm(test_loader, desc="Inference"):
        meta = metas[0] if isinstance(metas, (list, tuple)) else metas
        img_path = cfg.paths.test_images_dir / meta["file_name"]

        if args.tta:
            # USAR FUNCIÓN TTA MANUAL (devuelve listas de cajas y scores)
            boxes_xyxy, scores = tta_predict(
                model=model,
                img_path=img_path,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                imgsz=args.image_size,
                device=args.device,
            )
        else:
            # Inferencia normal (sin TTA nativo de YOLO)
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                imgsz=args.image_size,
                augment=False,
                device=args.device,
                verbose=False,
            )

            res = results[0]
            boxes_xyxy = res.boxes.xyxy.cpu().numpy() if (res.boxes is not None) else []
            scores = res.boxes.conf.cpu().numpy() if (res.boxes is not None) else []

        # Procesar resultados (común para ambos métodos)
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            if w <= 0.001 or h <= 0.001:
                continue

            submission_rows.append({
                "image_id": int(meta["image_id"]),
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(scores[i]),
            })

    # 6. Estadísticas Finales
    num_test_images = len(test_loader)
    total_boxes = len(submission_rows)
    avg_test = total_boxes / num_test_images if num_test_images > 0 else 0

    print("\n" + "="*40)
    print(f"📊 RESUMEN DE DENSIDAD")
    print(f"Promedio en TRAIN: {avg_train:.2f} cajas/img")
    print(f"Promedio en TEST:  {avg_test:.2f} cajas/img")
    print(f"Total cajas:       {total_boxes}")
    print("="*40 + "\n")

    # 7. Guardar resultados
    output_path = Path(args.output) if args.output else cfg.paths.outputs_dir / "submission.zip"
    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)

    print(f"Proceso completado. Archivo guardado en: {output_path}")

if __name__ == "__main__":
    main()