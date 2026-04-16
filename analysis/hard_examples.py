"""
Identificación de Hard Examples
Encuentra casos difíciles del dataset que el modelo no detecta bien
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hard_examples"
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


def compute_iou(box1, box2):
    """Calcula IoU entre dos boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    return inter_area / (box1[2] * box1[3] + box2[2] * box2[3] - inter_area)


def identify_hard_examples(images, annotations, model_path, conf=0.25):
    """
    Identifica diferentes tipos de hard examples:

    1. Small boxes no detectados
    2. Large boxes no detectados
    3. Crowded scenes (muchos boxes)
    4. Low contrast scenes
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    hard_examples = {
        'small_missed': [],      # Pequeños que no se detectan
        'large_missed': [],      # Grandes que no se detectan
        'crowded': [],          # Muchos boxes en una imagen
        'low_confidence': [],    # Boxes detectados con baja confianza
    }

    for img_id, anns in annotations.items():
        if not anns:
            continue

        img_info = images[img_id]
        img_path = IMAGES_DIR / img_info['file_name']

        if not img_path.exists():
            continue

        # Inferencia
        result = model.predict(source=str(img_path), imgsz=512, conf=0.05, iou=0.6, verbose=False)

        if not result or result[0].boxes is None:
            # No se detectó nada
            hard_examples['large_missed'].append((img_path, anns))
            continue

        preds = []
        for box in result[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            preds.append({
                'bbox': [x1, y1, x2-x1, y2-y1],
                'conf': conf
            })

        # Matchear predicciones con GT
        matched_gt = set()
        for ann in anns:
            gt_bbox = ann['bbox']
            gt_area = gt_bbox[2] * gt_bbox[3]

            # Verificar si hay match
            best_iou = 0
            for pred in preds:
                iou = compute_iou(gt_bbox, pred['bbox'])
                if iou > best_iou:
                    best_iou = iou

            if best_iou >= 0.5:
                for idx, pred in enumerate(preds):
                    if compute_iou(gt_bbox, pred['bbox']) == best_iou:
                        matched_gt.add(idx)
                        if pred['conf'] < conf + 0.1:
                            hard_examples['low_confidence'].append((img_path, ann, pred))
                        break
            else:
                # No detectado
                if gt_area < 32**2:
                    hard_examples['small_missed'].append((img_path, ann))
                elif gt_area > 96**2:
                    hard_examples['large_missed'].append((img_path, ann))

        # Crowded scenes
        if len(anns) > 5:
            hard_examples['crowded'].append((img_path, anns, preds))

    return hard_examples


def visualize_hard_examples(hard_examples, n_examples=8):
    """Visualiza los diferentes tipos de hard examples"""
    categories = [
        ('small_missed', 'Small Boxes No Detectados'),
        ('large_missed', 'Large Boxes No Detectados'),
        ('low_confidence', 'Low Confidence Detections'),
        ('crowded', 'Crowded Scenes')
    ]

    fig, axes = plt.subplots(len(categories), n_examples, figsize=(20, 12))
    fig.suptitle('Hard Examples del Dataset', fontsize=16, fontweight='bold')

    for row_idx, (key, title) in enumerate(categories):
        examples = hard_examples[key][:n_examples]

        for col_idx, example in enumerate(examples):
            if key == 'crowded':
                img_path, anns, preds = example
            elif key == 'low_confidence':
                img_path, ann, pred = example
            else:
                img_path, ann = example
                preds = []

            img = Image.open(img_path).convert('RGB')
            draw = ImageDraw.Draw(img)

            # Dibujar GT (verde)
            if key != 'crowded':
                x, y, w, h = ann['bbox']
                draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
                draw.text((x+5, y+5), f"GT", fill='green', font=None)
            else:
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    draw.rectangle([x, y, x+w, y+h], outline='green', width=2)

            # Dibujar predicciones (rojo)
            if key == 'low_confidence':
                x, y, w, h = pred['bbox']
                conf = pred['conf']
                draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                draw.text((x+5, y+5), f"P {conf:.2f}", fill='red', font=None)
            elif key == 'crowded':
                for pred in preds:
                    x, y, w, h = pred['bbox']
                    conf = pred['conf']
                    draw.rectangle([x, y, x+w, y+h], outline='red', width=1)
                    draw.text((x+3, y+3), f"{conf:.1f}", fill='red', font=None)

            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_title(f'{title} {col_idx+1}', fontsize=10)
            axes[row_idx, col_idx].axis('off')

        # Si no hay suficientes ejemplos
        for col_idx in range(len(examples), n_examples):
            axes[row_idx, col_idx].text(0.5, 0.5, 'No examples',
                                          ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
            axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hard_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Guardado: hard_examples.png")


def print_hard_examples_summary(hard_examples):
    """Imprime resumen de hard examples"""
    print("\n" + "="*60)
    print("RESUMEN DE HARD EXAMPLES")
    print("="*60)
    for key, name in [
        ('small_missed', 'Small Boxes No Detectados'),
        ('large_missed', 'Large Boxes No Detectados'),
        ('low_confidence', 'Low Confidence Detections'),
        ('crowded', 'Crowded Scenes')
    ]:
        count = len(hard_examples[key])
        print(f"{name:30s}: {count:4d} ejemplos")

    print("="*60)


def main():
    print("Cargando ground truth...")
    images, annotations = load_ground_truth()

    # Buscar modelo
    model_path = PROJECT_ROOT / "models" / "yolo_best_yolov9s.pt"
    if not model_path.exists():
        for pt_file in (PROJECT_ROOT / "submissions").glob("*.pt"):
            if "best" in pt_file.name or "run" in pt_file.name:
                model_path = pt_file
                break

    if not model_path.exists():
        print("[!] No se encontró modelo")
        return

    print(f"Usando modelo: {model_path.name}")

    # Identificar hard examples
    print("Identificando hard examples...")
    hard_examples = identify_hard_examples(images, annotations, model_path)

    # Resumen
    print_hard_examples_summary(hard_examples)

    # Visualizar
    print("Generando visualizaciones...")
    visualize_hard_examples(hard_examples)

    print(f"\n[✓] Análisis completado. Resultados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
