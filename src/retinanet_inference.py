from __future__ import annotations

"""
RetinaNet inference for ClearSAR — tiling + NMS global.

Divide cada imagen en tiles solapados, corre el modelo en cada uno,
recompone las detecciones a coordenadas globales y aplica NMS final.

Usage:
    python retinanet_inference.py \
        --checkpoint outputs/retinanet/best.pt \
        --output outputs/submission_retinanet.zip
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.config import default_config, ensure_dirs
from src.dataset import load_test_id_mapping
from src.submission import save_submission_auto, validate_submission_schema

# Importar helpers de tiling y builder desde retinanet_train
from src.retinanet_train import build_model, compute_tiles, pad_to_square, TILE_SIZE, TILE_OVERLAP

SCORE_THRESH = 0.25
NMS_IOU_FINAL = 0.4
MAX_DET = 500


def predict_image(
    model: torch.nn.Module,
    img_path: Path,
    device: torch.device,
    tile_size: int,
    overlap: int,
    score_thresh: float,
    nms_iou: float,
    max_det: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inferencia con tiling sobre una imagen completa.
    Devuelve (boxes_xyxy, scores) en coordenadas absolutas de la imagen original.
    """
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    tiles = compute_tiles(W, H, tile_size, overlap)
    all_boxes, all_scores = [], []

    model.eval()
    with torch.no_grad():
        for (tx1, ty1, tx2, ty2) in tiles:
            tile_img = img.crop((tx1, ty1, tx2, ty2))
            tile_img = pad_to_square(tile_img, tile_size)
            tensor = TF.to_tensor(tile_img).unsqueeze(0).to(device)

            preds = model(tensor)[0]

            if len(preds["boxes"]) == 0:
                continue

            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()

            # Filtrar por score
            keep = scores >= score_thresh
            boxes, scores = boxes[keep], scores[keep]

            # Convertir de coordenadas de tile a coordenadas globales
            boxes[:, 0] += tx1
            boxes[:, 1] += ty1
            boxes[:, 2] += tx1
            boxes[:, 3] += ty1

            # Clip a límites de imagen
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H)

            all_boxes.append(boxes)
            all_scores.append(scores)

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    all_boxes_np = np.concatenate(all_boxes, axis=0)
    all_scores_np = np.concatenate(all_scores, axis=0)

    # NMS global para fusionar detecciones solapadas entre tiles
    boxes_t = torch.as_tensor(all_boxes_np, dtype=torch.float32)
    scores_t = torch.as_tensor(all_scores_np, dtype=torch.float32)
    keep_idx = nms(boxes_t, scores_t, iou_threshold=nms_iou)

    final_boxes = all_boxes_np[keep_idx.numpy()][:max_det]
    final_scores = all_scores_np[keep_idx.numpy()][:max_det]

    return final_boxes, final_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RetinaNet inference for ClearSAR")
    p.add_argument("--project-root", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--backbone", type=str, default="resnet50",
                   choices=["resnet50", "resnet101"])
    p.add_argument("--mapping-path", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--conf", type=float, default=SCORE_THRESH)
    p.add_argument("--iou", type=float, default=NMS_IOU_FINAL)
    p.add_argument("--max-det", type=int, default=MAX_DET)
    p.add_argument("--tile-size", type=int, default=TILE_SIZE)
    p.add_argument("--tile-overlap", type=int, default=TILE_OVERLAP)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config(project_root=args.project_root)
    ensure_dirs(cfg)

    device = torch.device(args.device)
    print(f"[retinanet_infer] Device: {device}")

    # Cargar modelo
    model = build_model(backbone_name=args.backbone, num_classes=2, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[retinanet_infer] Checkpoint cargado: {args.checkpoint}")

    # ID mapping
    mapping_path = Path(args.mapping_path) if args.mapping_path else cfg.paths.test_id_mapping_path
    filename_to_image_id = load_test_id_mapping(
        test_images_dir=cfg.paths.test_images_dir,
        mapping_path=mapping_path,
        strict=True,
    )

    submission_rows: List[Dict[str, Any]] = []
    test_dir = cfg.paths.test_images_dir
    img_files = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))

    for img_path in tqdm(img_files, desc="Inference"):
        image_id = filename_to_image_id.get(img_path.name)
        if image_id is None:
            continue

        boxes_xyxy, scores = predict_image(
            model=model,
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

    avg = len(submission_rows) / max(len(img_files), 1)
    print(f"\n{'='*40}")
    print(f"Total detecciones: {len(submission_rows)}")
    print(f"Media por imagen:  {avg:.2f}")
    print(f"{'='*40}\n")

    output_path = Path(args.output) if args.output else cfg.paths.outputs_dir / "submission_retinanet.zip"
    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)
    print(f"[retinanet_infer] Guardado en: {output_path}")


if __name__ == "__main__":
    main()
