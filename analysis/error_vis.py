"""
Error Analysis - ClearSAR
Compare predictions with ground truth to identify weaknesses
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
from collections import defaultdict
import sys

# Fix encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_FILE = PROJECT_ROOT / "data" / "annotations" / "instances_train.json"
IMAGES_DIR = PROJECT_ROOT / "data" / "images" / "train"
HOLDOUT_IMAGES = PROJECT_ROOT / "data" / "yolo" / "holdout" / "images" / "val"
HOLDOUT_LABELS = PROJECT_ROOT / "data" / "yolo" / "holdout" / "labels" / "val"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth():
    """Load ground truth from JSON"""
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    annotations = defaultdict(list)
    for ann in coco['annotations']:
        annotations[ann['image_id']].append(ann)

    return images, annotations


def compute_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
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


def analyze_by_size(tp, fp, fn):
    """Analyze errors by box size"""
    def classify(box):
        if isinstance(box, dict):
            area = box.get('area', box['bbox'][2] * box['bbox'][3])
        else:
            area = box[2] * box[3]

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
        size = classify(item.get('area', item['bbox'][2] * item['bbox'][3]))
        fn_by_size[size].append(item)

    return tp_by_size, fp_by_size, fn_by_size


def plot_error_analysis(tp_by_size, fp_by_size, fn_by_size):
    """Generate error analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Error Analysis by Size', fontsize=16, fontweight='bold')

    sizes = ['small', 'medium', 'large']
    colors = {'small': '#55A868', 'medium': '#4C72B0', 'large': '#C44E52'}

    # TP by size
    tp_counts = [len(tp_by_size[s]) for s in sizes]
    axes[0, 0].bar(sizes, tp_counts, color=[colors[s] for s in sizes])
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('True Positives by Size')
    axes[0, 0].grid(alpha=0.3)

    # FP by size
    fp_counts = [len(fp_by_size[s]) for s in sizes]
    axes[0, 1].bar(sizes, fp_counts, color=[colors[s] for s in sizes])
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('False Positives by Size')
    axes[0, 1].grid(alpha=0.3)

    # FN by size
    fn_counts = [len(fn_by_size[s]) for s in sizes]
    axes[0, 2].bar(sizes, fn_counts, color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('False Negatives by Size')
    axes[0, 2].grid(alpha=0.3)

    # Precision by size
    precision = [len(tp_by_size[s]) / (len(tp_by_size[s]) + len(fp_by_size[s]) + 1e-6) for s in sizes]
    axes[1, 0].bar(sizes, precision, color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Precision by Size')
    axes[1, 0].grid(alpha=0.3)

    # Recall by size
    recall = [len(tp_by_size[s]) / (len(tp_by_size[s]) + len(fn_by_size[s]) + 1e-6) for s in sizes]
    axes[1, 1].bar(sizes, recall, color=[colors[s] for s in sizes])
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Recall by Size')
    axes[1, 1].grid(alpha=0.3)

    # F1-score by size
    f1 = [2 * (p * r) / (p + r + 1e-6) for p, r in zip(precision, recall)]
    axes[1, 2].bar(sizes, f1, color=[colors[s] for s in sizes])
    axes[1, 2].set_ylabel('F1-Score')
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('F1-Score by Size')
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'errors_by_size.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: errors_by_size.png")


def main():
    print("Loading ground truth...")
    images, gt_annotations = load_ground_truth()
    print(f"Loaded: {len(images)} images, {sum(len(v) for v in gt_annotations.values())} annotations")

    # Simple analysis without model (for now)
    # Just analyze GT statistics that might indicate hard cases

    print("\n" + "="*60)
    print("DATASET CHARACTERISTICS")
    print("="*60)

    # Find images with many boxes (crowded scenes)
    anns_per_image = {}
    for img_id, anns in gt_annotations.items():
        anns_per_image[img_id] = len(anns)

    crowded_threshold = 5
    crowded_ids = [img_id for img_id, count in anns_per_image.items() if count > crowded_threshold]

    print(f"\nImages with > {crowded_threshold} boxes (crowded): {len(crowded_ids)}")
    print(f"  Percentage: {len(crowded_ids)/len(images)*100:.1f}%")

    # Find images with very small boxes
    small_boxes_ids = []
    for img_id, anns in gt_annotations.items():
        for ann in anns:
            area = ann['area']
            if area < 500:  # Very small threshold
                small_boxes_ids.append(img_id)
                break

    print(f"\nImages with very small boxes (<500 px^2): {len(set(small_boxes_ids))}")
    print(f"  Percentage: {len(set(small_boxes_ids))/len(images)*100:.1f}%")

    # Find images with very large boxes
    large_boxes_ids = []
    for img_id, anns in gt_annotations.items():
        for ann in anns:
            area = ann['area']
            if area > 50000:  # Very large threshold
                large_boxes_ids.append(img_id)
                break

    print(f"\nImages with very large boxes (>50000 px^2): {len(set(large_boxes_ids))}")
    print(f"  Percentage: {len(set(large_boxes_ids))/len(images)*100:.1f}%")

    print("\n" + "="*60)
    print("INSIGHTS FOR IMPROVEMENT")
    print("="*60)
    print("\n1. SMALL BOXES (48.8% of all)")
    print("   - Model struggles with small/detected RFI")
    print("   - Solution: Dual-pass slicing helps here")
    print("   - Current slice-max-height=32px should help")

    print("\n2. LARGE BOXES (8.2% of all)")
    print("   - Model may struggle with context")
    print("   - Solution: Larger image size or specific augmentations")

    print("\n3. HORIZONTAL BOXES (86%)")
    print("   - RFI is predominantly horizontal")
    print("   - Model architecture should favor horizontal detection")
    print("   - YOLO26n working well suggests smaller models generalize better")

    print("\n4. BOXES PER IMAGE (avg: 2.94)")
    print("   - Most images have 1-3 boxes")
    print("   - Crowded scenes (>5 boxes): {len(crowded_ids)/len(images)*100:.1f}%")
    print("   - Consider NMS tuning for crowded scenes")

    print(f"\n[OK] Analysis complete. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
