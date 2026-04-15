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
from ensemble_boxes import weighted_boxes_fusion

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
MIN_BOX_DIM = 0.001

# --- FUNCIONES DE AYUDA (Copiadas de tu yolo_inference para que sea standalone) ---

def _yolo_to_coco_bbox(x_center, y_center, width, height, image_width, image_height):
    w = max(0.0, width * float(image_width))
    h = max(0.0, height * float(image_height))
    x = (x_center * float(image_width)) - (w / 2.0)
    y = (y_center * float(image_height)) - (h / 2.0)
    return [x, y, w, h]

def _parse_holdout_yaml(holdout_yaml_path: Path):
    payload = yaml.safe_load(holdout_yaml_path.read_text(encoding="utf-8")) or {}
    root = Path(payload.get("path", holdout_yaml_path.parent))
    if not root.is_absolute():
        root = (holdout_yaml_path.parent / root).resolve()
    val_value = str(payload.get("val", "images/val"))
    images_dir = (root / val_value).resolve()
    labels_dir = (root / val_value.replace("images", "labels", 1)).resolve()
    class_names = payload.get("names", ["RFI"]) # Asumimos 1 clase
    return images_dir, labels_dir, class_names

def _build_holdout_coco_gt(images_dir: Path, labels_dir: Path, class_names: list[str]):
    image_files = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images, annotations, filename_to_image_id = [], [], {}
    ann_id = 1
    for idx, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as image_obj:
            img_w, img_h = image_obj.size
        image_id = int(img_path.stem) if img_path.stem.isdigit() else idx
        filename_to_image_id[img_path.name] = image_id
        images.append({"id": image_id, "file_name": img_path.name, "width": img_w, "height": img_h})
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists(): continue
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5: continue
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:5])
            x, y, w, h = _yolo_to_coco_bbox(x_center, y_center, width, height, img_w, img_h)
            if w > MIN_BOX_DIM and h > MIN_BOX_DIM:
                annotations.append({
                    "id": ann_id, "image_id": image_id, "category_id": class_id + 1,
                    "bbox": [x, y, w, h], "area": float(w * h), "iscrowd": 0,
                })
                ann_id += 1
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    return {"images": images, "annotations": annotations, "categories": categories}, filename_to_image_id, image_files

# --- CORE DEL ENSEMBLE ---

def get_predictions_for_images(models_info, image_files, device):
    """Hace inferencia con todos los modelos y guarda las cajas NORMALIZADAS."""
    preds_cache = {}
    for img_path in tqdm(image_files, desc="Generando predicciones base"):
        with Image.open(img_path) as img_obj:
            img_w, img_h = img_obj.size
        
        img_data = {
            "width": img_w,
            "height": img_h,
            "models_boxes": [],
            "models_scores": [],
            "models_labels": []
        }
        
        for model, imgsz in models_info:
            # TTA y conf baja para no perder nada. El NMS interno de YOLO casi ni actuará
            results = model.predict(
                source=str(img_path), conf=0.005, iou=0.8, imgsz=imgsz, 
                max_det=1000, augment=False, device=device, verbose=False
            )
            
            m_boxes, m_scores, m_labels = [], [], []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxyn[0].tolist() # Coordenadas normalizadas [0, 1]
                    m_boxes.append([x1, y1, x2, y2])
                    m_scores.append(float(box.conf[0]))
                    m_labels.append(0) # Forzamos a 0 (1 sola clase)
            
            img_data["models_boxes"].append(m_boxes)
            img_data["models_scores"].append(m_scores)
            img_data["models_labels"].append(m_labels)
            
        preds_cache[img_path.name] = img_data
    return preds_cache

def evaluate_wbf_on_holdout(preds_cache, coco_gt, filename_to_image_id, weights, iou_thr, skip_thr):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_preds = []
    for filename, data in preds_cache.items():
        image_id = filename_to_image_id[filename]
        
        # Aplicar WBF
        boxes, scores, labels = weighted_boxes_fusion(
            data["models_boxes"], data["models_scores"], data["models_labels"],
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_thr
        )
        
        # Convertir cajas finales a formato COCO (px absolutos)
        img_w, img_h = data["width"], data["height"]
        for i in range(len(boxes)):
            nx1, ny1, nx2, ny2 = boxes[i]
            x1, y1 = nx1 * img_w, ny1 * img_h
            w, h = (nx2 - nx1) * img_w, (ny2 - ny1) * img_h
            
            if w > MIN_BOX_DIM and h > MIN_BOX_DIM:
                coco_preds.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(scores[i]),
                })
                
    if not coco_preds: return 0.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_json = Path(tmp_dir) / "gt.json"
        pred_json = Path(tmp_dir) / "pred.json"
        gt_json.write_text(json.dumps(coco_gt))
        pred_json.write_text(json.dumps(coco_preds))
        
        coco_gt_api = COCO(str(gt_json))
        coco_dt_api = coco_gt_api.loadRes(str(pred_json))
        coco_eval = COCOeval(coco_gt_api, coco_dt_api, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
    return float(coco_eval.stats[0]) # Devuelve mAP 50-95

def parse_args():
    parser = argparse.ArgumentParser()
    # Ahora le pasamos pares de "ruta,tamaño"
    parser.add_argument("--models", nargs='+', required=True, help="Lista de modelo,imgsz. Ej: run0.pt,512 run2.pt,640")
    parser.add_argument("--holdout-yaml", default="data/yolo/holdout.yaml")
    parser.add_argument("--test-images-dir", default="data/images/test")
    parser.add_argument("--mapping-path", default="catalog.v1.parquet")
    parser.add_argument("--output", default="outputs/submission_ensemble.zip")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _resolve_input_path(raw_path: str, module_dir: Path, repo_root: Path) -> Path:
    """Resuelve rutas admitiendo absolutas, relativas a la raiz y relativas al modulo."""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    root_first = (repo_root / candidate).resolve()
    if root_first.exists():
        return root_first

    module_first = (module_dir / candidate).resolve()
    if module_first.exists():
        return module_first

    return root_first

def main():
    args = parse_args()
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent
    
    # 1. Cargar Modelos y sus resoluciones
    print(f"\n[Ensemble] Cargando {len(args.models)} modelos...")
    models_info = []
    for m in args.models:
        path, size = m.split(',') # Separamos por la coma
        models_info.append((YOLO(path), int(size)))
    
    # 2. Cargar Holdout
    holdout_yaml = _resolve_input_path(args.holdout_yaml, module_dir, project_root)
    h_images_dir, h_labels_dir, h_names = _parse_holdout_yaml(holdout_yaml)
    coco_gt, h_file_to_id, h_image_files = _build_holdout_coco_gt(h_images_dir, h_labels_dir, h_names)
    
    # 3. Cachear Predicciones Holdout
    print("\n[Ensemble] Fase 1: Extrayendo cajas del Holdout...")
    holdout_cache = get_predictions_for_images(models_info, h_image_files, args.device)
    
    # 4. Grid Search WBF
    print("\n[Ensemble] Fase 2: Grid Search de WBF en Holdout...")
    best_map = 0.0
    best_params = {}
    weights = [1] * len(models_info) # Pesos iguales por defecto
    
    iou_thrs = [0.4, 0.5, 0.6]
    skip_thrs = [0.001, 0.005, 0.01]
    
    for iou in iou_thrs:
        for skip in skip_thrs:
            mAP = evaluate_wbf_on_holdout(holdout_cache, coco_gt, h_file_to_id, weights, iou, skip)
            print(f" -> iou_thr={iou}, skip_thr={skip} | mAP 50-95: {mAP:.4f}")
            if mAP > best_map:
                best_map = mAP
                best_params = {"iou_thr": iou, "skip_box_thr": skip}
                
    print(f"\n[Ensemble] ¡Mejor combinación! IOU: {best_params['iou_thr']}, Skip: {best_params['skip_box_thr']} (mAP: {best_map:.4f})")
    
    # 5. Inferencia en TEST
    print("\n[Ensemble] Fase 3: Procesando el Test Set...")
    test_dir = _resolve_input_path(args.test_images_dir, module_dir, project_root)
    test_files = [p for p in sorted(test_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    
    mapping_file = _resolve_input_path(args.mapping_path, module_dir, project_root)
    df = pd.read_parquet(mapping_file)
    test_file_to_id = {Path(p).name: int(Path(p).stem) for p in df["id"].astype(str) if "/test/" in p.replace("\\", "/")}
    
    test_cache = get_predictions_for_images(models_info, test_files, args.device)
    
    submission_rows = []
    for filename, data in test_cache.items():
        if filename not in test_file_to_id: continue
        image_id = test_file_to_id[filename]
        
        boxes, scores, labels = weighted_boxes_fusion(
            data["models_boxes"], data["models_scores"], data["models_labels"],
            weights=weights, iou_thr=best_params["iou_thr"], skip_box_thr=best_params["skip_box_thr"]
        )
        
        img_w, img_h = data["width"], data["height"]
        for i in range(len(boxes)):
            nx1, ny1, nx2, ny2 = boxes[i]
            x1, y1 = nx1 * img_w, ny1 * img_h
            w, h = max(0.0, (nx2 - nx1) * img_w), max(0.0, (ny2 - ny1) * img_h)
            
            if w > MIN_BOX_DIM and h > MIN_BOX_DIM:
                submission_rows.append({
                    "image_id": image_id, "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)], "score": float(scores[i]),
                })
                
    # 6. Guardar ZIP
    out_path = _resolve_input_path(args.output, module_dir, project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(submission_rows))
        
    print(f"\n[Ensemble] Proceso finalizado. Submission guardado en: {out_path}")

if __name__ == "__main__":
    main()