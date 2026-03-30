from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Allow running as a direct script: python src/visualize_preprocessing.py
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.config import default_config
from src.dataset import get_train_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualiza imagenes aleatorias con preprocess y augmentations "
            "del pipeline de entrenamiento"
        )
    )
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--num-augs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--with-boxes", action="store_true", default=True)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _load_coco(annotation_path: Path) -> Dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_anns_by_file(coco: Dict[str, Any]) -> Dict[str, List[List[float]]]:
    images = {int(img["id"]): str(img["file_name"]) for img in coco["images"]}
    by_file: Dict[str, List[List[float]]] = {}

    for ann in coco["annotations"]:
        img_id = int(ann["image_id"])
        file_name = images.get(img_id)
        if file_name is None:
            continue
        if file_name not in by_file:
            by_file[file_name] = []
        by_file[file_name].append([float(v) for v in ann["bbox"]])

    return by_file


def _draw_boxes(ax: Any, coco_boxes_xywh: Sequence[Sequence[float]]) -> None:
    for box in coco_boxes_xywh:
        x, y, w, h = [float(v) for v in box]
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)


def _coco_to_pascal_and_filter(
    coco_boxes_xywh: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> List[List[float]]:
    """Convert [x,y,w,h] -> [x1,y1,x2,y2] and drop invalid/outside boxes."""
    pascal: List[List[float]] = []
    for box in coco_boxes_xywh:
        x, y, w, h = [float(v) for v in box]
        x1 = max(0.0, min(float(width - 1), x))
        y1 = max(0.0, min(float(height - 1), y))
        x2 = max(0.0, min(float(width - 1), x + w))
        y2 = max(0.0, min(float(height - 1), y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        pascal.append([x1, y1, x2, y2])
    return pascal


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0.0, 255.0)
    return clipped.astype(np.uint8)


def main() -> None:
    args = parse_args()
    cfg = default_config(project_root=args.project_root)

    rng = random.Random(args.seed)

    train_dir = cfg.paths.train_images_dir
    ann_path = cfg.paths.train_annotations_path
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else cfg.paths.outputs_dir / "preprocess_preview"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    image_paths = sorted(
        p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise ValueError(f"No images found in {train_dir}")

    sample_n = min(args.num_images, len(image_paths))
    sampled = rng.sample(image_paths, k=sample_n)

    coco = _load_coco(ann_path)
    anns_by_file = _build_anns_by_file(coco)

    image_size = (args.image_size, args.image_size) if args.image_size is not None else None
    aug = get_train_transforms(image_size=image_size)

    for img_path in sampled:
        file_name = img_path.name
        original = np.array(Image.open(img_path).convert("RGB"))
        h, w = original.shape[:2]

        bboxes_coco = anns_by_file.get(file_name, [])
        bboxes_pascal = _coco_to_pascal_and_filter(bboxes_coco, width=w, height=h)
        labels = [1] * len(bboxes_pascal)

        augs: List[np.ndarray] = []
        all_transformed_bboxes: List[List[List[float]]] = []
        for _ in range(max(1, args.num_augs)):
            transformed = aug(image=original.copy(), bboxes=bboxes_pascal, labels=labels)
            aug_img = _to_uint8(np.asarray(transformed["image"]))
            augs.append(aug_img)
            transformed_bboxes_pascal = [list(map(float, b)) for b in transformed["bboxes"]]
            all_transformed_bboxes.append(transformed_bboxes_pascal)

        cols = 1 + len(augs)
        fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.2))
        if cols == 1:
            axes = [axes]

        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis("off")
        if args.with_boxes:
            _draw_boxes(axes[0], bboxes_coco)

        for i, (aug_img, transformed_bboxes) in enumerate(
            zip(augs, all_transformed_bboxes),
            start=1,
        ):
            axes[i].imshow(aug_img)
            axes[i].set_title(f"Augment {i}")
            axes[i].axis("off")
            if args.with_boxes:
                coco_boxes = [
                    [b[0], b[1], max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])]
                    for b in transformed_bboxes
                ]
                _draw_boxes(axes[i], coco_boxes)

        fig.suptitle(file_name)
        fig.tight_layout()

        out_path = output_dir / f"preview_{img_path.stem}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        if args.show:
            plt.show()
        plt.close(fig)

    print(f"Saved {sample_n} previews in: {output_dir}")


if __name__ == "__main__":
    main()
