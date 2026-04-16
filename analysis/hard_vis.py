"""
Hard Examples - ClearSAR (Simplified version)
Identify difficult cases that model doesn't detect well
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
from collections import defaultdict
import sys

# Fix encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_FILE = PROJECT_ROOT / "data" / "annotations" / "instances_train.json"
IMAGES_DIR = PROJECT_ROOT / "data" / "images" / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hard_examples"
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


def analyze_dataset_statistics(annotations, images):
    """Analyze dataset to identify potentially hard cases"""
    print("\n" + "="*60)
    print("HARD EXAMPLES IDENTIFICATION")
    print("="*60)

    # 1. Images with no boxes
    images_with_boxes = set(annotations.keys())
    images_no_boxes = [img_id for img_id in images.keys() if img_id not in images_with_boxes]

    print("1. IMAGES WITHOUT BOXES")
    print(f"   Count: {len(images_no_boxes)} ({len(images_no_boxes)/len(images)*100:.1f}%)")
    print("   Note: These are clean images, model should NOT predict here")

    # 2. Crowded images
    anns_per_image = {}
    for img_id, anns in annotations.items():
        anns_per_image[img_id] = len(anns)

    counts = list(anns_per_image.values())
    crowded_threshold = 5
    very_crowded_threshold = 10

    crowded = sum(1 for c in counts if c > crowded_threshold)
    very_crowded = sum(1 for c in counts if c > very_crowded_threshold)

    print(f"2. CROWDED IMAGES")
    print(f"   >5 boxes: {crowded} ({crowded/len(images)*100:.1f}%)")
    print(f"   >10 boxes: {very_crowded} ({very_crowded/len(images)*100:.1f}%)")
    print("   Note: May confuse NMS, consider tuning")

    # 3. Very small boxes
    tiny_boxes_ids = []
    for img_id, anns in annotations.items():
        for ann in anns:
            area = ann['area']
            if area < 200:  # 200px^2 threshold
                tiny_boxes_ids.append(img_id)
                break

    tiny_count = len(set(tiny_boxes_ids))
    print(f"3. IMAGES WITH TINY BOXES (<200px^2)")
    print(f"   Count: {tiny_count} ({tiny_count/len(images)*100:.1f}%)")
    print("   Note: Small RFI, hard to detect")
    print("   Solution: Dual-pass slicing helps here")
    print("   Current slice-max-height=32px should help")

    # 4. Very large boxes
    huge_boxes_ids = []
    for img_id, anns in annotations.items():
        for ann in anns:
            area = ann['area']
            if area > 100000:  # 100000px^2 threshold
                huge_boxes_ids.append(img_id)
                break

    huge_count = len(set(huge_boxes_ids))
    print(f"4. IMAGES WITH HUGE BOXES (>100000px^2)")
    print(f"   Count: {huge_count} ({huge_count/len(images)*100:.1f}%)")
    print("   Note: Model may lack context")
    print("   Solution: Larger image size or specific augmentations")

    # 5. Vertical boxes (h > 1.5w)
    vertical_boxes_ids = []
    for img_id, anns in annotations.items():
        for ann in anns:
            w, h = ann['bbox'][2], ann['bbox'][3]
            if h > w * 1.5:
                vertical_boxes_ids.append(img_id)
                break

    vertical_count = len(set(vertical_boxes_ids))
    print(f"5. IMAGES WITH VERTICAL BOXES (h > 1.5w)")
    print(f"   Count: {vertical_count} ({vertical_count/len(images)*100:.1f}%)")
    print("   Note: Stacked RFI, different pattern")
    print("   Solution: May require specialized handling")

    print("\n" + "="*60)
    print("STATISTICS SUMMARY")
    print("="*60)
    print(f"Total images: {len(images)}")
    print(f"Images with boxes: {len(images_with_boxes)}")
    print(f"Average boxes per image: {len(annotations)/len(images):.2f}")

    return {
        'no_boxes': images_no_boxes,
        'crowded': crowded,
        'tiny_boxes': set(tiny_boxes_ids),
        'huge_boxes': set(huge_boxes_ids),
        'vertical_boxes': set(vertical_boxes_ids)
    }


def visualize_hard_cases(images, hard_cases, n_examples=9):
    """Visualize different types of hard examples"""
    fig, axes = plt.subplots(2, n_examples, figsize=(16, 12))
    fig.suptitle('Hard Examples from Dataset', fontsize=16, fontweight='bold')

    categories = [
        ('no_boxes', 'Images without boxes', 'Clean samples'),
        ('crowded', 'Crowded scenes', '>5 boxes'),
        ('tiny_boxes', 'Tiny boxes (<200px^2)', 'Small RFI'),
        ('huge_boxes', 'Huge boxes (>100000px^2)', 'Context issues'),
        ('vertical_boxes', 'Vertical boxes (h > 1.5w)', 'Stacked RFI'),
    ]

    for row_idx, (key, title, subtitle) in enumerate(categories):
        examples = list(hard_cases[key])[:n_examples]

        for col_idx, example in enumerate(examples):
            if key == 'no_boxes' or key == 'crowded':
                img_path = IMAGES_DIR / images[example]['file_name']]
            elif key == 'tiny_boxes' or key == 'huge_boxes' or key == 'vertical_boxes':
                img_id = example
                img_path = IMAGES_DIR / images[img_id]['file_name']]

            if not img_path.exists():
                continue

            img = Image.open(img_path).convert('RGB')
            draw = ImageDraw.Draw(img)

            # Draw GT boxes (green)
            if key != 'no_boxes' and key != 'crowded':
                if key == 'crowded':
                    anns_per_img = annotations.get(img_id, [])
                else:
                    anns_per_img = [a for a in annotations.get(img_id, [])
                                          if a.get('area', 0) > 0]
                for ann in anns_per_img:
                    x, y, w, h = ann['bbox']
                    draw.rectangle([x, y, x+w, y+h], outline='green', width=2)
                    draw.text((x+5, y+5), 'GT', fill='green', font=None)

            # Draw annotations or info
            if key == 'no_boxes':
                draw.text((img.width//2, img.height//2), 'Clean', fill='blue', font=None)
            elif key == 'tiny_boxes':
                anns_per_img = [a for a in annotations.get(img_id, []) if a.get('area', 0) == 200]
                for ann in anns_per_img:
                    x, y, w, h = ann['bbox']
                    draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                draw.text((img.width//2, 20), 'Tiny!', fill='red', font=None)
            elif key == 'huge_boxes':
                anns_per_img = [a for a in annotations.get(img_id, []) if a.get('area', 0) == 100000]
                for ann in anns_per_img:
                    x, y, w, h = ann['bbox']
                    draw.rectangle([x, y, x+w, y+h], outline='orange', width=3)
                draw.text((img.width//2, img.height//2), 'Huge!', fill='orange', font=None)
            elif key == 'vertical_boxes':
                anns_per_img = [a for a in annotations.get(img_id, []) if a['bbox'][3] > a['bbox'][2] * 1.5]
                for ann in anns_per_img:
                    x, y, w, h = ann['bbox']
                    draw.rectangle([x, y, x+w, y+h], outline='purple', width=2)
                draw.text((img.width//2, img.height//2), 'Vert!', fill='purple', font=None)
            elif key == 'crowded':
                anns_per_img = annotations.get(img_id, [])
                for ann in anns_per_img:
                    x, y, w, h = ann['bbox']
                    draw.rectangle([x, y, x+w, y+h], outline='blue', width=2)
                n = min(5, len(anns_per_img))
                draw.text((5, 5), f'{n} boxes', fill='blue', font=None)

            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_title(f'{title} - {subtitle}', fontsize=10)
            axes[row_idx, col_idx].axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'hard_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: hard_examples.png")


def main():
    print("Loading dataset...")
    images, annotations = load_ground_truth()
    print(f"Loaded: {len(images)} images, {sum(len(v) for v in annotations.values())} annotations")

    # Analyze dataset
    hard_cases = analyze_dataset_statistics(annotations, images)

    # Visualize
    print("\nGenerating visualizations...")
    visualize_hard_cases(images, hard_cases)

    print(f"\n[OK] Analysis complete. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
