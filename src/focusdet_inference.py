from __future__ import annotations

"""
FocusDet inference for ClearSAR.

Carga el modelo entrenado con focusdet_train.py y genera el submission.
Usa SIoU-SoftNMS gaussiano en el post-proceso (mismo que el paper).

Opcionalmente activa TTA (flips) para mejorar recall.

Usage:
    python focusdet_inference.py \
        --checkpoint outputs/focusdet/best.pt \
        --output outputs/submission_focusdet.zip

    # Con TTA (más lento, ligeramente mejor):
    python focusdet_inference.py \
        --checkpoint outputs/focusdet/best.pt \
        --tta \
        --output outputs/submission_focusdet_tta.zip
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.config import default_config, ensure_dirs
from src.dataset import load_test_id_mapping
from src.submission import save_submission_auto, validate_submission_schema

# Importar arquitectura y utilidades de focusdet_train
from focusdet_train import FocusDet, decode_predictions, soft_nms_gaussian, NUM_CLASSES

SCORE_THRESH = 0.25
NMS_SIGMA    = 0.5
MAX_DET      = 500


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image, imgsz: int) -> Tuple[torch.Tensor, float, int, int]:
    """
    Letterbox resize + normalize.
    Returns (tensor, scale, new_w, new_h)
    """
    orig_w, orig_h = img.size
    scale = imgsz / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_r = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(img_r, (0, 0))
    return TF.to_tensor(canvas), scale, new_w, new_h


def postprocess_boxes(boxes: np.ndarray, scale: float, orig_w: int, orig_h: int) -> np.ndarray:
    """Scale boxes back to original image coordinates and clip."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes / scale
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h)
    return boxes


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-IMAGE INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(
    model: FocusDet,
    img: Image.Image,
    device: torch.device,
    imgsz: int,
    score_thresh: float,
    nms_sigma: float,
    max_det: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inferencia sobre una imagen PIL. Devuelve (boxes_xyxy_original, scores)."""
    orig_w, orig_h = img.size
    tensor, scale, _, _ = preprocess(img, imgsz)
    tensor = tensor.unsqueeze(0).to(device)

    cls_p, box_p, ctr_p = model(tensor)
    # Quitar dimensión batch
    cls_p = [c[0] for c in cls_p]
    box_p = [b[0] for b in box_p]
    ctr_p = [c[0] for c in ctr_p]

    boxes, scores = decode_predictions(
        cls_p, box_p, ctr_p,
        score_thresh=score_thresh,
        nms_sigma=nms_sigma,
        max_det=max_det,
        img_h=imgsz, img_w=imgsz,
    )
    boxes = postprocess_boxes(boxes, scale, orig_w, orig_h)
    return boxes, scores


@torch.no_grad()
def predict_tta(
    model: FocusDet,
    img: Image.Image,
    device: torch.device,
    imgsz: int,
    score_thresh: float,
    nms_sigma: float,
    max_det: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TTA con flips horizontales/verticales.
    Fusiona detecciones con SoftNMS gaussiano global.
    """
    import torchvision.transforms.functional as TF_
    orig_w, orig_h = img.size

    variants = [
        (img,                             lambda b: b),
        (TF_.hflip(img),                  lambda b: np.array([orig_w - b[2], b[1], orig_w - b[0], b[3]])),
        (TF_.vflip(img),                  lambda b: np.array([b[0], orig_h - b[3], b[2], orig_h - b[1]])),
        (TF_.vflip(TF_.hflip(img)),       lambda b: np.array([orig_w - b[2], orig_h - b[3], orig_w - b[0], orig_h - b[1]])),
    ]

    all_boxes, all_scores = [], []
    for aug_img, inv_fn in variants:
        boxes, scores = predict_single(model, aug_img, device, imgsz, score_thresh * 0.5, nms_sigma, max_det)
        for box, s in zip(boxes, scores):
            all_boxes.append(inv_fn(box))
            all_scores.append(s)

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    boxes_t  = torch.as_tensor(np.stack(all_boxes), dtype=torch.float32)
    scores_t = torch.as_tensor(all_scores, dtype=torch.float32)
    boxes_t, scores_t = soft_nms_gaussian(boxes_t, scores_t,
                                           sigma=nms_sigma, score_thresh=score_thresh,
                                           max_det=max_det)
    return boxes_t.numpy(), scores_t.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# ARGS & MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FocusDet inference for ClearSAR")
    p.add_argument("--project-root",  type=str,   default=None)
    p.add_argument("--checkpoint",    type=str,   required=True,
                   help="Ruta a outputs/focusdet/best.pt")
    p.add_argument("--mapping-path",  type=str,   default=None)
    p.add_argument("--output",        type=str,   default=None)
    p.add_argument("--conf",          type=float, default=SCORE_THRESH)
    p.add_argument("--nms-sigma",     type=float, default=NMS_SIGMA,
                   help="Sigma para SoftNMS gaussiano (0.3=agresivo, 0.7=suave)")
    p.add_argument("--max-det",       type=int,   default=MAX_DET)
    p.add_argument("--imgsz",         type=int,   default=640,
                   help="Debe coincidir con el imgsz usado en entrenamiento")
    p.add_argument("--fpn-channels",  type=int,   default=256)
    p.add_argument("--tta",           action="store_true",
                   help="Test Time Augmentation (flips × 4)")
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = default_config(project_root=args.project_root)
    ensure_dirs(cfg)
    device = torch.device(args.device)

    # Cargar modelo
    model = FocusDet(num_classes=NUM_CLASSES, fpn_channels=args.fpn_channels, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[focusdet_infer] Checkpoint: {args.checkpoint}")
    print(f"[focusdet_infer] Device: {device} | imgsz: {args.imgsz} | TTA: {args.tta}")
    print(f"[focusdet_infer] SoftNMS σ={args.nms_sigma} | conf≥{args.conf}")

    # ID mapping
    mapping_path = Path(args.mapping_path) if args.mapping_path else cfg.paths.test_id_mapping_path
    filename_to_image_id = load_test_id_mapping(
        test_images_dir=cfg.paths.test_images_dir,
        mapping_path=mapping_path,
        strict=True,
    )

    test_dir = cfg.paths.test_images_dir
    img_files = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))
    submission_rows: List[Dict[str, Any]] = []

    predict_fn = predict_tta if args.tta else predict_single

    for img_path in tqdm(img_files, desc="FocusDet Inference"):
        image_id = filename_to_image_id.get(img_path.name)
        if image_id is None:
            continue

        img = Image.open(img_path).convert("RGB")

        if args.tta:
            boxes_xyxy, scores = predict_tta(model, img, device, args.imgsz,
                                              args.conf, args.nms_sigma, args.max_det)
        else:
            boxes_xyxy, scores = predict_single(model, img, device, args.imgsz,
                                                 args.conf, args.nms_sigma, args.max_det)

        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w <= 0.001 or h <= 0.001:
                continue
            submission_rows.append({
                "image_id":   int(image_id),
                "category_id": 1,
                "bbox":  [float(x1), float(y1), float(w), float(h)],
                "score": float(scores[i]),
            })

    total = len(submission_rows)
    avg   = total / max(len(img_files), 1)
    print(f"\n{'='*40}")
    print(f"Total detecciones: {total}")
    print(f"Media por imagen:  {avg:.2f}")
    print(f"{'='*40}\n")

    output_path = (Path(args.output) if args.output
                   else cfg.paths.outputs_dir / "submission_focusdet.zip")
    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)
    print(f"[focusdet_infer] Guardado en: {output_path}")


if __name__ == "__main__":
    main()
