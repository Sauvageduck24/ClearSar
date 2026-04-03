from __future__ import annotations

"""
DINO/DETR inference for ClearSAR — tiling alta resolución + NMS global.

Usage:
    python dino_inference.py \
        --checkpoint outputs/dino/best_hf \
        --output outputs/submission_dino.zip
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm

try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from src.config import default_config, ensure_dirs
from src.dataset import load_test_id_mapping
from src.submission import save_submission_auto, validate_submission_schema
from dino_train import compute_tiles, pad_tile, TILE_SIZE, TILE_OVERLAP

SCORE_THRESH = 0.3
NMS_IOU = 0.4
MAX_DET = 500


def predict_image_dino(
    model,
    processor,
    img_path: Path,
    device: torch.device,
    tile_size: int,
    overlap: int,
    score_thresh: float,
    nms_iou: float,
    max_det: int,
) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    tiles = compute_tiles(W, H, tile_size, overlap)
    all_boxes, all_scores = [], []

    model.eval()
    with torch.no_grad():
        for (tx1, ty1, tx2, ty2) in tiles:
            tile_img = pad_tile(img.crop((tx1, ty1, tx2, ty2)), tile_size)
            inputs = processor(images=tile_img, return_tensors="pt").to(device)

            outputs = model(**inputs)

            # Post-proceso: convertir logits a boxes
            target_sizes = torch.tensor([[tile_size, tile_size]], device=device)
            results = processor.post_process_object_detection(
                outputs,
                threshold=score_thresh,
                target_sizes=target_sizes,
            )[0]

            if len(results["boxes"]) == 0:
                continue

            boxes = results["boxes"].cpu().numpy()   # xyxy en coords tile
            scores = results["scores"].cpu().numpy()

            # Convertir a coordenadas globales
            boxes[:, 0] += tx1
            boxes[:, 1] += ty1
            boxes[:, 2] += tx1
            boxes[:, 3] += ty1
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H)

            all_boxes.append(boxes)
            all_scores.append(scores)

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    all_boxes_np = np.concatenate(all_boxes)
    all_scores_np = np.concatenate(all_scores)

    # NMS global
    keep = nms(
        torch.as_tensor(all_boxes_np, dtype=torch.float32),
        torch.as_tensor(all_scores_np, dtype=torch.float32),
        iou_threshold=nms_iou,
    ).numpy()

    return all_boxes_np[keep][:max_det], all_scores_np[keep][:max_det]


def parse_args():
    p = argparse.ArgumentParser(description="DINO inference for ClearSAR")
    p.add_argument("--project-root", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Carpeta HuggingFace (outputs/dino/best_hf) o archivo .pt")
    p.add_argument("--mapping-path", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--conf", type=float, default=SCORE_THRESH)
    p.add_argument("--iou", type=float, default=NMS_IOU)
    p.add_argument("--max-det", type=int, default=MAX_DET)
    p.add_argument("--tile-size", type=int, default=TILE_SIZE)
    p.add_argument("--tile-overlap", type=int, default=TILE_OVERLAP)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    if not HF_AVAILABLE:
        raise ImportError("pip install transformers accelerate")

    args = parse_args()
    cfg = default_config(project_root=args.project_root)
    ensure_dirs(cfg)
    device = torch.device(args.device)

    print(f"[dino_infer] Cargando desde: {args.checkpoint}")
    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    mapping_path = Path(args.mapping_path) if args.mapping_path else cfg.paths.test_id_mapping_path
    filename_to_image_id = load_test_id_mapping(
        test_images_dir=cfg.paths.test_images_dir,
        mapping_path=mapping_path,
        strict=True,
    )

    test_dir = cfg.paths.test_images_dir
    img_files = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))
    submission_rows: List[Dict[str, Any]] = []

    for img_path in tqdm(img_files, desc="DINO Inference"):
        image_id = filename_to_image_id.get(img_path.name)
        if image_id is None:
            continue

        boxes_xyxy, scores = predict_image_dino(
            model=model,
            processor=processor,
            img_path=img_path,
            device=device,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            score_thresh=args.conf,
            nms_iou=args.iou,
            max_det=args.max_det,
        )

        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w <= 0.001 or h <= 0.001:
                continue
            submission_rows.append({
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(scores[i]),
            })

    total = len(submission_rows)
    avg = total / max(len(img_files), 1)
    print(f"\n{'='*40}")
    print(f"Total detecciones: {total}")
    print(f"Media por imagen:  {avg:.2f}")
    print(f"{'='*40}\n")

    output_path = Path(args.output) if args.output else cfg.paths.outputs_dir / "submission_dino.zip"
    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)
    print(f"[dino_infer] Guardado en: {output_path}")


if __name__ == "__main__":
    main()
