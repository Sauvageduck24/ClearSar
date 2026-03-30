from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm

from src.config import default_config, ensure_dirs
from src.dataset import (
    ClearSARTestDataset,
    collate_detection_batch,
    load_test_id_mapping,
    xyxy_to_coco_xywh,  # Aseguramos la importación necesaria
)
from src.model import apply_checkpoint_model_hints, build_model, load_model_checkpoint
from src.submission import (
    predictions_to_submission_rows,
    save_submission_auto,
    validate_submission_schema,
)
from src.utils.repro import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference pipeline for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mapping-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--wbf", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--preprocess", type=str, choices=["none", "percentile"], default=None)
    return parser.parse_args()


def apply_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.tta:
        cfg.inference.use_tta = True
    if args.wbf:
        cfg.inference.use_wbf = True
    if args.batch_size is not None:
        cfg.inference.batch_size = args.batch_size
    if args.min_score is not None:
        cfg.inference.min_score = args.min_score
    if args.preprocess is not None:
        cfg.inference.preprocess_mode = args.preprocess


def _flip_boxes_h(boxes: torch.Tensor, width: int) -> torch.Tensor:
    flipped = boxes.clone()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped


def _flip_boxes_v(boxes: torch.Tensor, height: int) -> torch.Tensor:
    flipped = boxes.clone()
    flipped[:, 1] = height - boxes[:, 3]
    flipped[:, 3] = height - boxes[:, 1]
    return flipped


def _rot90_boxes_k1_to_original(boxes: torch.Tensor, height: int) -> torch.Tensor:
    """Map boxes from rot90(k=1) image coords back to original coords."""
    if boxes.numel() == 0:
        return boxes.clone()
    mapped = boxes.clone()
    mapped[:, 0] = boxes[:, 1]
    mapped[:, 1] = height - boxes[:, 2]
    mapped[:, 2] = boxes[:, 3]
    mapped[:, 3] = height - boxes[:, 0]
    return mapped


def _rot90_boxes_k3_to_original(boxes: torch.Tensor, width: int) -> torch.Tensor:
    """Map boxes from rot90(k=3) image coords back to original coords."""
    if boxes.numel() == 0:
        return boxes.clone()
    mapped = boxes.clone()
    mapped[:, 0] = width - boxes[:, 3]
    mapped[:, 1] = boxes[:, 0]
    mapped[:, 2] = width - boxes[:, 1]
    mapped[:, 3] = boxes[:, 2]
    return mapped


def _fuse_predictions_single_image(
    pred_list: Sequence[Dict[str, torch.Tensor]],
    width: int,
    height: int,
    use_wbf: bool,
    wbf_iou_thr: float,
    wbf_skip_box_thr: float,
    nms_iou_thr: float,
) -> Dict[str, torch.Tensor]:
    boxes_list: List[torch.Tensor] = []
    scores_list: List[torch.Tensor] = []

    for pred in pred_list:
        if pred["boxes"].numel() == 0:
            continue
        boxes_list.append(pred["boxes"])
        scores_list.append(pred["scores"])

    if not boxes_list:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
        }

    if use_wbf:
        try:
            from ensemble_boxes import weighted_boxes_fusion

            norm_boxes = []
            norm_scores = []
            norm_labels = []

            for b, s in zip(boxes_list, scores_list):
                b_np = b.detach().cpu().numpy()
                s_np = s.detach().cpu().numpy()
                b_norm = b_np.copy()
                b_norm[:, [0, 2]] = b_norm[:, [0, 2]] / max(1.0, float(width))
                b_norm[:, [1, 3]] = b_norm[:, [1, 3]] / max(1.0, float(height))
                b_norm = b_norm.clip(0.0, 1.0)

                norm_boxes.append(b_norm.tolist())
                norm_scores.append(s_np.tolist())
                norm_labels.append([1] * len(s_np))

            fused_boxes, fused_scores, _ = weighted_boxes_fusion(
                boxes_list=norm_boxes,
                scores_list=norm_scores,
                labels_list=norm_labels,
                iou_thr=wbf_iou_thr,
                skip_box_thr=wbf_skip_box_thr,
            )

            if len(fused_boxes) == 0:
                return {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }

            fused_boxes_t = torch.tensor(fused_boxes, dtype=torch.float32)
            fused_boxes_t[:, [0, 2]] = fused_boxes_t[:, [0, 2]] * float(width)
            fused_boxes_t[:, [1, 3]] = fused_boxes_t[:, [1, 3]] * float(height)
            fused_scores_t = torch.tensor(fused_scores, dtype=torch.float32)
            return {"boxes": fused_boxes_t, "scores": fused_scores_t}
        except ImportError:
            pass

    all_boxes = torch.cat(boxes_list, dim=0)
    all_scores = torch.cat(scores_list, dim=0)

    keep = nms(all_boxes, all_scores, iou_threshold=nms_iou_thr)
    return {
        "boxes": all_boxes[keep],
        "scores": all_scores[keep],
    }


def tta_predict_batch(model: torch.nn.Module, images: List[torch.Tensor], use_tta: bool) -> List[Dict[str, torch.Tensor]]:
    outputs_base = model(images)
    if not use_tta:
        return [{"boxes": _rescale_boxes(o["boxes"], img.shape[2], img.shape[1]), "scores": o["scores"].detach().cpu()} for o, img in zip(outputs_base, images)]

    images_h = [torch.flip(img, dims=[2]) for img in images]
    images_v = [torch.flip(img, dims=[1]) for img in images]
    images_r90 = [torch.rot90(img, k=1, dims=[1, 2]) for img in images]
    images_r270 = [torch.rot90(img, k=3, dims=[1, 2]) for img in images]

    outputs_h = model(images_h)
    outputs_v = model(images_v)
    outputs_r90 = model(images_r90)
    outputs_r270 = model(images_r270)

    merged: List[Dict[str, torch.Tensor]] = []
    for img, base, out_h, out_v, out_r90, out_r270 in zip(
        images,
        outputs_base,
        outputs_h,
        outputs_v,
        outputs_r90,
        outputs_r270,
    ):
        _, h, w = img.shape

        b0 = _rescale_boxes(base["boxes"], w, h)
        s0 = base["scores"].detach().cpu()

        bh = _flip_boxes_h(out_h["boxes"].detach().cpu(), width=w)
        sh = out_h["scores"].detach().cpu()

        bv = _flip_boxes_v(out_v["boxes"].detach().cpu(), height=h)
        sv = out_v["scores"].detach().cpu()

        br90 = _rot90_boxes_k1_to_original(out_r90["boxes"].detach().cpu(), height=h)
        sr90 = out_r90["scores"].detach().cpu()

        br270 = _rot90_boxes_k3_to_original(out_r270["boxes"].detach().cpu(), width=w)
        sr270 = out_r270["scores"].detach().cpu()

        merged.append(
            {
                "boxes": torch.cat([b0, bh, bv, br90, br270], dim=0)
                if b0.numel() or bh.numel() or bv.numel() or br90.numel() or br270.numel()
                else torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.cat([s0, sh, sv, sr90, sr270], dim=0)
                if s0.numel() or sh.numel() or sv.numel() or sr90.numel() or sr270.numel()
                else torch.zeros((0,), dtype=torch.float32),
            }
        )

    return merged

def _rescale_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Reescala las coordenadas de las cajas al tamaño original."""
    if boxes.numel() == 0:
        return boxes

    rescaled = boxes.clone()
    rescaled[:, [0, 2]] *= width
    rescaled[:, [1, 3]] *= height
    return rescaled


def main() -> None:
    args = parse_args()

    cfg = default_config(project_root=args.project_root)
    apply_overrides(cfg, args)
    ensure_dirs(cfg)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else cfg.paths.models_dir / "best_model.pt"
    mapping_path = Path(args.mapping_path) if args.mapping_path else cfg.paths.test_id_mapping_path
    output_path = Path(args.output) if args.output else cfg.paths.outputs_dir / "submission.zip"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if mapping_path is None:
        raise ValueError("No mapping path configured. Provide --mapping-path.")

    filename_to_image_id = load_test_id_mapping(
        test_images_dir=cfg.paths.test_images_dir,
        mapping_path=mapping_path,
        strict=True,
    )

    test_ds = ClearSARTestDataset(
        images_dir=cfg.paths.test_images_dir,
        filename_to_image_id=filename_to_image_id,
        transforms=None,
        preprocess_mode=cfg.inference.preprocess_mode,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        num_workers=cfg.inference.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
    )

    device = resolve_device()
    apply_checkpoint_model_hints(cfg.model, str(checkpoint_path))
    model = build_model(cfg.model, device=device)
    model = load_model_checkpoint(model=model, checkpoint_path=str(checkpoint_path), device=device)

    submission_rows: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for images, metas in tqdm(test_loader, total=len(test_loader), desc="Inference"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # 1. Obtener dimensiones originales y actuales
                orig_w = float(metas["width"][i])
                orig_h = float(metas["height"][i])
                curr_h, curr_w = images[i].shape[1], images[i].shape[2]

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for b, s, l in zip(boxes, scores, labels):
                    # 2. REESCALAR coordenadas al tamaño original
                    x1 = b[0] * (orig_w / curr_w)
                    y1 = b[1] * (orig_h / curr_h)
                    x2 = b[2] * (orig_w / curr_w)
                    y2 = b[3] * (orig_h / curr_h)

                    # 3. Convertir a formato COCO [x, y, w, h]
                    coco_box = xyxy_to_coco_xywh([x1, y1, x2, y2])

                    submission_rows.append({
                        "image_id": int(metas["image_id"][i]),
                        "category_id": int(l),
                        "bbox": [round(float(v), 2) for v in coco_box],
                        "score": round(float(s), 4),
                    })

    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)

    summary_path = cfg.paths.outputs_dir / "inference_summary.json"
    summary = {
        "checkpoint": str(checkpoint_path),
        "mapping_path": str(mapping_path),
        "num_rows": len(submission_rows),
        "output": str(output_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"Submission ready: {output_path}")
    print(f"Rows: {len(submission_rows)}")


if __name__ == "__main__":
    main()
