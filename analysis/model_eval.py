"""
Model Evaluation - ClearSAR
Run sections independently or all together via main().

PART 1 - Dataset Statistics : no model needed, just GT annotations
PART 2 - Visualization      : visual plots of GT distributions and error examples
PART 3 - Model Evaluation   : requires a trained YOLO model on holdout set
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_FILE      = PROJECT_ROOT / "data" / "annotations" / "instances_train.json"
IMAGES_DIR   = PROJECT_ROOT / "data" / "images" / "train"
HOLDOUT_IMGS = PROJECT_ROOT / "data" / "yolo" / "holdout" / "images" / "val"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "model_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Shared helpers ────────────────────────────────────────────────────────────

def load_ground_truth():
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco['images']}
    annotations = defaultdict(list)
    for ann in coco['annotations']:
        annotations[ann['image_id']].append(ann)
    return images, annotations


def compute_iou(box1, box2):
    """boxes in [x, y, w, h]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / (box1[2]*box1[3] + box2[2]*box2[3] - inter)


# ── PART 1: Dataset Statistics ────────────────────────────────────────────────

def analyze_dataset_stats(images, annotations):
    print("\n" + "="*60)
    print("PART 1 — DATASET STATISTICS")
    print("="*60)

    anns_per_image = {img_id: len(anns) for img_id, anns in annotations.items()}
    crowded = [i for i, c in anns_per_image.items() if c > 5]
    small   = [i for i, anns in annotations.items() if any(a['area'] < 500 for a in anns)]
    large   = [i for i, anns in annotations.items() if any(a['area'] > 50000 for a in anns)]

    n = len(images)
    print(f"\nTotal images     : {n}")
    print(f"Total annotations: {sum(anns_per_image.values())}")
    print(f"Avg boxes/image  : {np.mean(list(anns_per_image.values())):.2f}")

    print(f"\nCrowded (>5 boxes)          : {len(crowded)} ({len(crowded)/n*100:.1f}%)")
    print(f"With very small boxes <500px²: {len(set(small))} ({len(set(small))/n*100:.1f}%)")
    print(f"With very large boxes >50000²: {len(set(large))} ({len(set(large))/n*100:.1f}%)")

    print("\n[Insights]")
    print("  - Small boxes dominate → slicing critical for recall")
    print("  - RFI is predominantly horizontal → model handles well")
    print("  - Crowded scenes → tune NMS iou threshold")


# ── PART 2: Visualization ─────────────────────────────────────────────────────

def plot_size_distribution(annotations):
    areas = [ann['area'] for anns in annotations.values() for ann in anns]
    heights = [ann['bbox'][3] for anns in annotations.values() for ann in anns]
    counts = [len(anns) for anns in annotations.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('GT Box Distribution', fontweight='bold')

    axes[0].hist(areas, bins=60, color='steelblue', edgecolor='black', alpha=0.8)
    axes[0].set_title('Box Area (px²)')
    axes[0].set_xlabel('Area')

    axes[1].hist(heights, bins=40, color='coral', edgecolor='black', alpha=0.8)
    axes[1].set_title('Box Height (px)')
    axes[1].set_xlabel('Height')

    axes[2].hist(counts, bins=range(0, max(counts)+2), color='mediumseagreen', edgecolor='black', alpha=0.8)
    axes[2].set_title('Boxes per Image')
    axes[2].set_xlabel('Count')

    plt.tight_layout()
    out = OUTPUT_DIR / 'gt_distribution.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def plot_error_breakdown(tp_by_size, fp_by_size, fn_by_size):
    sizes  = ['small', 'medium', 'large']
    colors = {'small': '#55A868', 'medium': '#4C72B0', 'large': '#C44E52'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Error Analysis by Box Size', fontsize=16, fontweight='bold')

    for col, (label, data) in enumerate([('TP', tp_by_size), ('FP', fp_by_size), ('FN', fn_by_size)]):
        counts = [len(data[s]) for s in sizes]
        axes[0, col].bar(sizes, counts, color=[colors[s] for s in sizes])
        axes[0, col].set_title(label)
        axes[0, col].set_ylabel('Count')
        axes[0, col].grid(alpha=0.3)

    tp_c = [len(tp_by_size[s]) for s in sizes]
    fp_c = [len(fp_by_size[s]) for s in sizes]
    fn_c = [len(fn_by_size[s]) for s in sizes]

    precision = [tp / (tp + fp + 1e-6) for tp, fp in zip(tp_c, fp_c)]
    recall    = [tp / (tp + fn + 1e-6) for tp, fn in zip(tp_c, fn_c)]
    f1        = [2*p*r/(p+r+1e-6) for p, r in zip(precision, recall)]

    for col, (label, vals) in enumerate([('Precision', precision), ('Recall', recall), ('F1', f1)]):
        axes[1, col].bar(sizes, vals, color=[colors[s] for s in sizes])
        axes[1, col].set_title(label)
        axes[1, col].set_ylim(0, 1)
        axes[1, col].axhline(0.5, color='red', linestyle='--', alpha=0.5)
        axes[1, col].grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'errors_by_size.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


def visualize_fp_fn_examples(images, annotations, model, n=6):
    fp_examples, fn_examples = [], []

    for img_info in list(images.values()):
        if len(fp_examples) >= n and len(fn_examples) >= n:
            break
        img_path = IMAGES_DIR / img_info['file_name']
        if not img_path.exists():
            continue

        gt_boxes = annotations.get(img_info['id'], [])
        result = model.predict(source=str(img_path), imgsz=512, conf=0.25, iou=0.6, verbose=False)
        preds = []
        if result and result[0].boxes is not None:
            for box in result[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds.append([x1, y1, x2-x1, y2-y1, float(box.conf[0])])

        matched_gt = set()
        for pred in preds:
            best_iou, best_idx = 0, -1
            for idx, gt in enumerate(gt_boxes):
                iou = compute_iou(pred[:4], gt['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou >= 0.5:
                matched_gt.add(best_idx)
            elif len(fp_examples) < n:
                fp_examples.append((img_path, pred, gt_boxes))

        for idx, gt in enumerate(gt_boxes):
            if idx not in matched_gt and len(fn_examples) < n:
                fn_examples.append((img_path, gt, preds))

    fig, axes = plt.subplots(2, n, figsize=(20, 8))
    fig.suptitle('FP (top) and FN (bottom) Examples', fontsize=14, fontweight='bold')

    for i, (img_path, pred, gt_boxes) in enumerate(fp_examples[:n]):
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        x, y, w, h = pred[:4]
        draw.rectangle([x, y, x+w, y+h], outline='blue', width=3)
        for gt in gt_boxes:
            gx, gy, gw, gh = gt['bbox']
            draw.rectangle([gx, gy, gx+gw, gy+gh], outline='red', width=2)
        axes[0, i].imshow(img); axes[0, i].axis('off')
        axes[0, i].set_title(f'FP {i+1}', fontsize=9)

    for i, (img_path, gt, preds) in enumerate(fn_examples[:n]):
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        gx, gy, gw, gh = gt['bbox']
        draw.rectangle([gx, gy, gx+gw, gy+gh], outline='green', width=3)
        for pred in preds:
            px, py, pw, ph = pred[:4]
            draw.rectangle([px, py, px+pw, py+ph], outline='red', width=1)
        axes[1, i].imshow(img); axes[1, i].axis('off')
        axes[1, i].set_title(f'FN {i+1}', fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / 'error_examples.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


# ── PART 3: Model Evaluation ──────────────────────────────────────────────────

def find_model():
    candidates = [
        PROJECT_ROOT / "models" / "yolo_best_yolov9s.pt",
        PROJECT_ROOT / "models" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in (PROJECT_ROOT / "submissions").glob("*.pt"):
        return p
    return None


def classify_size(area):
    if area < 32**2:   return 'small'
    if area < 96**2:   return 'medium'
    return 'large'


def match_predictions_to_gt(preds, gt_boxes, iou_thr=0.5):
    matched_gt = set()
    tp, fp, fn = [], [], []

    for pred in preds:
        best_iou, best_idx = 0, -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_iou >= iou_thr:
            tp.append({'pred': pred, 'gt': gt_boxes[best_idx], 'iou': best_iou})
            matched_gt.add(best_idx)
        else:
            fp.append(pred)

    fn = [gt for idx, gt in enumerate(gt_boxes) if idx not in matched_gt]
    return tp, fp, fn


def run_model_evaluation(model_path=None):
    print("\n" + "="*60)
    print("PART 3 — MODEL EVALUATION")
    print("="*60)

    from ultralytics import YOLO

    if model_path is None:
        model_path = find_model()
    if model_path is None or not Path(model_path).exists():
        print("[!] No model found. Skipping model evaluation.")
        return

    print(f"Model: {model_path}")
    model = YOLO(str(model_path))
    images, annotations = load_ground_truth()

    all_tp, all_fp, all_fn = [], [], []
    img_paths = list(HOLDOUT_IMGS.glob("*.*")) if HOLDOUT_IMGS.exists() else []

    if not img_paths:
        print("[!] No holdout images found, running on train sample (first 50)")
        img_paths = [IMAGES_DIR / img['file_name'] for img in list(images.values())[:50]]

    for img_path in img_paths:
        img_id = next((id_ for id_, inf in images.items()
                       if inf['file_name'] == img_path.name), None)
        if img_id is None:
            continue

        gt_boxes = annotations.get(img_id, [])
        result = model.predict(source=str(img_path), imgsz=512, conf=0.25, iou=0.6, verbose=False)
        preds = []
        if result and result[0].boxes is not None:
            for box in result[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2-x1, y2-y1
                preds.append({'bbox': [x1, y1, w, h], 'area': w*h, 'conf': float(box.conf[0])})

        tp, fp, fn = match_predictions_to_gt(preds, gt_boxes)
        all_tp.extend(tp); all_fp.extend(fp); all_fn.extend(fn)

    print(f"\nTrue Positives : {len(all_tp)}")
    print(f"False Positives: {len(all_fp)}")
    print(f"False Negatives: {len(all_fn)}")

    precision = len(all_tp) / (len(all_tp) + len(all_fp) + 1e-6)
    recall    = len(all_tp) / (len(all_tp) + len(all_fn) + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

    tp_by_size = defaultdict(list)
    fp_by_size = defaultdict(list)
    fn_by_size = defaultdict(list)
    for item in all_tp:
        tp_by_size[classify_size(item['pred']['area'])].append(item)
    for item in all_fp:
        fp_by_size[classify_size(item['area'])].append(item)
    for item in all_fn:
        fn_by_size[classify_size(item['area'])].append(item)

    plot_error_breakdown(tp_by_size, fp_by_size, fn_by_size)
    visualize_fp_fn_examples(images, annotations, model)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(run_model=False, model_path=None):
    images, annotations = load_ground_truth()
    print(f"Loaded {len(images)} images, {sum(len(v) for v in annotations.values())} annotations")

    analyze_dataset_stats(images, annotations)  # Part 1
    plot_size_distribution(annotations)          # Part 2

    if run_model:
        run_model_evaluation(model_path)         # Part 3


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .pt checkpoint. If given, runs Part 3.")
    args = parser.parse_args()
    main(run_model=args.model is not None, model_path=args.model)
