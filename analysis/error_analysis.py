"""
Análisis de Errores del Modelo
Compara predicciones con ground truth para identificar debilidades
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
from collections import defaultdict
import cv2

# Configuración
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_FILE = PROJECT_ROOT / "data" / "annotations" / "instances_train.json"
IMAGES_DIR = PROJECT_ROOT / "data" / "images" / "train"
HOLDOUT_IMAGES = PROJECT_ROOT / "data" / "yolo" / "holdout" / "images" / "val"
HOLDOUT_LABELS = PROJECT_ROOT / "data" / "yolo" / "holdout" / "labels" / "val"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth():
    """Carga ground truth desde JSON"""
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    annotations = defaultdict(list)
    for ann in coco['annotations']:
        annotations[ann['image_id']].append(ann)

    return images, annotations


def load_holdout_predictions(model_path, holdout_images_dir, conf=0.25):
    """Carga predicciones del modelo en holdout"""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    predictions = {}

    for img_path in holdout_images_dir.glob("*.*"):
        img = Image.open(img_path).convert('RGB')
        result = model.predict(source=img, imgsz=512, conf=conf, iou=0.6, verbose=False)

        preds = []
        if result and result[0].boxes is not None:
            for box in result[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                preds.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'conf': conf,
                    'area': (x2-x1) * (y2-y1)
                })

        predictions[img_path.stem] = preds

    return predictions


def compute_iou(box1, box2):
    """Calcula IoU entre dos boxes [x, y, w, h]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    return inter_area / (box1_area + box2_area - inter_area)


def match_predictions_to_gt(predictions, gt_boxes, iou_threshold=0.5):
    """Empareja predicciones con ground truth"""
    matched_gt = set()
    tp_preds = []  # True Positives
    fp_preds = []  # False Positives
    fn_boxes = []  # False Negatives

    # True Positives: predicciones que matchean con GT
    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp_preds.append({
                'pred': pred,
                'gt': gt_boxes[best_gt_idx],
                'iou': best_iou
            })
            matched_gt.add(best_gt_idx)
        else:
            fp_preds.append(pred)

    # False Negatives: GT boxes sin match
    for idx, gt in enumerate(gt_boxes):
        if idx not in matched_gt:
            fn_boxes.append(gt)

    return tp_preds, fp_preds, fn_boxes


def analyze_errors_by_size(tp, fp, fn):
    """Analiza errores por tamaño de box"""
    def classify(box):
        area = box['area'] if isinstance(box, dict) else (box['bbox'][2] * box['bbox'][3])
        if area < 32**2:
            return 'small'
        elif area < 96**2:
            return 'medium'
        else:
            return 'large'

    tp_by_size = defaultdict(list)
    fp_by_size = defaultdict(list)
    fn_by_size = defaultdict(list)

    for item in tp:
        size = classify(item['pred'] if isinstance(item, dict) else item)
        tp_by_size[size].append(item)

    for item in fp:
        size = classify(item)
        fp_by_size[size].append(item)

    for item in fn:
        size = classify(item['area'] if isinstance(item, dict) else (item['bbox'][2] * item['bbox'][3]))
        fn_by_size[size].append(item)

    return tp_by_size, fp_by_size, fn_by_size


def plot_error_analysis(tp_by_size, fp_by_size, fn_by_size):
    """Genera gráficas de análisis de errores"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Análisis de Errores por Tamaño', fontsize=16, fontweight='bold')

    sizes = ['small', 'medium', 'large']
    colors = {'small': '#55A868', 'medium': '#4C72B0', 'large': '#C44E52'}

    # TP por tamaño
    tp_counts = [len(tp_by_size[s]) for s in sizes]
    axes[0, 0].bar(sizes, tp_counts, color=[colors[s] for s in sizes])
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('True Positives por Tamaño')
    axes[0, 0].grid(alpha=0.3)

    # FP por tamaño
    fp_counts = [len(fp_by_size[s]) for s in sizes]
    axes[0, 1].bar(sizes, fp_counts, color=[colors[s] for s in sizes])
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('False Positives por Tamaño')
    axes[0, 1].grid(alpha=0.3)

    # FN por tamaño
    fn_counts = [len(fn_by_size[s]) for s in sizes]
    axes[0, 2].bar(sizes, fn_counts, color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('False Negatives por Tamaño')
    axes[0, 2].grid(alpha=0.3)

    # Precision por tamaño
    precision = [len(tp_by_size[s]) / (len(tp_by_size[s]) + len(fp_by_size[s]) + 1e-6) for s in sizes]
    axes[1, 0].bar(sizes, precision, color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Precision por Tamaño')
    axes[1, 0].grid(alpha=0.3)

    # Recall por tamaño
    recall = [len(tp_by_size[s]) / (len(tp_by_size[s]) + len(fn_by_size[s]) + 1e-6) for s in sizes]
    axes[1, 1].bar(sizes, recall, color=[colors[s] for s in sizes])
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Recall por Tamaño')
    axes[1, 1].grid(alpha=0.3)

    # F1-score por tamaño
    f1 = [2 * (p * r) / (p + r + 1e-6) for p, r in zip(precision, recall)]
    axes[1, 2].bar(sizes, f1, color=[colors[s] for s in sizes])
    axes[1, 2].set_ylabel('F1-Score')
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('F1-Score por Tamaño')
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'errors_by_size.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Guardado: errors_by_size.png")


def visualize_fp_fn_examples(images, predictions, gt_annotations, n_examples=6):
    """Visualiza ejemplos de falsos positivos y negativos"""
    from ultralytics import YOLO

    # Cargar modelo (usar el mejor disponible)
    model_path = PROJECT_ROOT / "models" / "yolo_best_yolov9s.pt"
    if not model_path.exists():
        # Buscar alternativas
        for pt_file in (PROJECT_ROOT / "submissions").glob("*.pt"):
            if "best" in pt_file.name or "run" in pt_file.name:
                model_path = pt_file
                break

    if not model_path.exists():
        print("[!] No se encontró modelo para visualización")
        return

    model = YOLO(str(model_path))

    # Recopilar ejemplos
    fp_examples = []
    fn_examples = []

    for img_info in images.values():
        img_path = IMAGES_DIR / img_info['file_name']
        if not img_path.exists():
            continue

        img_id = img_info['id']
        gt_boxes = gt_annotations.get(img_id, [])

        result = model.predict(source=str(img_path), imgsz=512, conf=0.25, iou=0.6, verbose=False)

        preds = []
        if result and result[0].boxes is not None:
            for box in result[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds.append([x1, y1, x2-x1, y2-y1, float(box.conf[0])])

        # Matchear predicciones con GT
        matched_gt = set()
        for pred in preds:
            best_iou = 0
            for gt in gt_boxes:
                iou = compute_iou(pred[:4], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= 0.5:
                for idx, gt in enumerate(gt_boxes):
                    if compute_iou(pred[:4], gt['bbox']) == best_iou:
                        matched_gt.add(idx)
                        break
            else:
                if len(fp_examples) < n_examples:
                    fp_examples.append((img_path, pred, gt_boxes))

        for idx, gt in enumerate(gt_boxes):
            if idx not in matched_gt and len(fn_examples) < n_examples:
                fn_examples.append((img_path, gt, preds))

    # Visualizar
    fig, axes = plt.subplots(2, n_examples, figsize=(20, 8))
    fig.suptitle('Errores del Modelo', fontsize=16, fontweight='bold')

    # Falsos Positivos
    for idx, (img_path, pred, gt_boxes) in enumerate(fp_examples):
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Dibujar predicción (azul)
        x, y, w, h = pred[:4]
        draw.rectangle([x, y, x+w, y+h], outline='blue', width=3)
        draw.text((x+5, y+5), f"FP {pred[4]:.2f}", fill='blue', font=None)

        # Dibujar GT (rojo, traslúcido)
        for gt in gt_boxes:
            gx, gy, gw, gh = gt['bbox']
            draw.rectangle([gx, gy, gx+gw, gy+gh], outline='red', width=2)

        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f'FP Example {idx+1}', fontsize=10)
        axes[0, idx].axis('off')

    # Falsos Negativos
    for idx, (img_path, gt, preds) in enumerate(fn_examples):
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Dibujar GT (verde)
        gx, gy, gw, gh = gt['bbox']
        draw.rectangle([gx, gy, gx+gw, gy+gh], outline='green', width=3)

        # Dibujar predicciones cercanas (rojo)
        for pred in preds:
            px, py, pw, ph = pred[:4]
            draw.rectangle([px, py, px+pw, py+ph], outline='red', width=1)

        axes[1, idx].imshow(img)
        axes[1, idx].set_title(f'FN Example {idx+1}', fontsize=10)
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Guardado: error_examples.png")


def main():
    print("Cargando ground truth...")
    images, gt_annotations = load_ground_truth()

    print("Cargando predicciones...")
    # Buscar modelo disponible
    model_path = PROJECT_ROOT / "models" / "yolo_best_yolov9s.pt"
    if not model_path.exists():
        for pt_file in (PROJECT_ROOT / "submissions").glob("*.pt"):
            model_path = pt_file
            break

    if not model_path.exists():
        print("[!] No se encontró modelo")
        return

    predictions = load_holdout_predictions(model_path, HOLDOUT_IMAGES)

    # Analizar por imagen
    print("Analizando errores...")
    all_tp, all_fp, all_fn = [], [], []

    for img_stem, preds in predictions.items():
        # Encontrar img_id correspondiente
        img_id = None
        for id_val, img_info in images.items():
            if img_info['file_name'].startswith(img_stem):
                img_id = id_val
                break

        if img_id is None:
            continue

        gt_boxes = gt_annotations.get(img_id, [])

        tp, fp, fn = match_predictions_to_gt(preds, gt_boxes)
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_fn.extend(fn)

    # Estadísticas globales
    print("\n" + "="*60)
    print("RESUMEN GLOBAL DE ERRORES")
    print("="*60)
    print(f"Total True Positives:  {len(all_tp)}")
    print(f"Total False Positives: {len(all_fp)}")
    print(f"Total False Negatives: {len(all_fn)}")

    # Análisis por tamaño
    tp_by_size, fp_by_size, fn_by_size = analyze_errors_by_size(all_tp, all_fp, all_fn)
    plot_error_analysis(tp_by_size, fp_by_size, fn_by_size)

    # Visualizar ejemplos
    print("\nGenerando ejemplos visuales...")
    visualize_fp_fn_examples(images, predictions, gt_annotations)

    print(f"\n[✓] Análisis completado. Resultados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
