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
from src.dataset import ClearSARTestDataset, collate_detection_batch, load_test_id_mapping
from src.inference import tta_predict_batch
from src.model import apply_checkpoint_model_hints, build_model, load_model_checkpoint
from src.submission import predictions_to_submission_rows, save_submission_auto, validate_submission_schema
from src.utils.repro import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint ensemble inference for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--mapping-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, nargs="*", default=None)
    parser.add_argument("--top-checkpoints-json", type=str, default=None)
    parser.add_argument("--max-models", type=int, default=5)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--wbf", action="store_true")
    parser.add_argument("--no-wbf", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    return parser.parse_args()


def _get_checkpoint_list(args: argparse.Namespace, project_root: Path) -> List[Path]:
    if args.checkpoints:
        return [Path(p) for p in args.checkpoints]

    top_json = (
        Path(args.top_checkpoints_json)
        if args.top_checkpoints_json
        else (project_root / "outputs" / "top_checkpoints.json")
    )
    if not top_json.exists():
        raise FileNotFoundError(
            f"Checkpoint list not found: {top_json}. Provide --checkpoints or generate top_checkpoints.json"
        )

    with top_json.open("r", encoding="utf-8") as f:
        items = json.load(f)

    ckpts = [Path(x["path"]) for x in items[: max(1, args.max_models)]]
    missing = [str(p) for p in ckpts if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Some checkpoints are missing: {missing[:5]}")
    return ckpts


def _fuse_predictions_single_image(
    pred_list: Sequence[Dict[str, torch.Tensor]],
    width: int,
    height: int,
    use_wbf: bool,
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
            from ensemble_boxes import weighted_boxes_fusion  # type: ignore[import-not-found]

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
                iou_thr=0.55,
                skip_box_thr=0.001,
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
    return {"boxes": all_boxes[keep], "scores": all_scores[keep]}


def main() -> None:
    args = parse_args()

    cfg = default_config(project_root=args.project_root)
    ensure_dirs(cfg)

    project_root = cfg.paths.project_root
    ckpt_paths = _get_checkpoint_list(args, project_root)

    if args.batch_size is not None:
        cfg.inference.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.inference.num_workers = args.num_workers
    if args.tta:
        cfg.inference.use_tta = True
    if args.no_tta:
        cfg.inference.use_tta = False
    if args.wbf:
        cfg.inference.use_wbf = True
    if args.no_wbf:
        cfg.inference.use_wbf = False
    cfg.inference.min_score = args.min_score

    output_path = Path(args.output) if args.output else cfg.paths.outputs_dir / "submission_ensemble.zip"
    mapping_path = Path(args.mapping_path) if args.mapping_path else cfg.paths.test_id_mapping_path
    if mapping_path is None:
        raise ValueError("No mapping path available. Provide --mapping-path")

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
        persistent_workers=cfg.inference.num_workers > 0,
        collate_fn=collate_detection_batch,
    )

    device = resolve_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    if args.amp:
        use_amp = True
    if args.no_amp:
        use_amp = False

    rows: List[Dict[str, Any]] = []

    # Build one model per checkpoint once (instead of rebuilding each batch).
    models: List[torch.nn.Module] = []
    if ckpt_paths:
        apply_checkpoint_model_hints(cfg.model, str(ckpt_paths[0]))
    for ckpt in ckpt_paths:
        model = build_model(cfg.model, device=device)
        model = load_model_checkpoint(model, checkpoint_path=str(ckpt), device=device)
        model.eval()
        models.append(model)

    with torch.inference_mode():
        for images, metas in tqdm(test_loader, total=len(test_loader), desc="ensemble"):
            images_gpu = [img.to(device, non_blocking=True) for img in images]

            per_model_outputs: List[List[Dict[str, torch.Tensor]]] = []
            for model in models:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outs = tta_predict_batch(model, images_gpu, use_tta=cfg.inference.use_tta)
                per_model_outputs.append(outs)

            fused_outputs: List[Dict[str, torch.Tensor]] = []
            for i, meta in enumerate(metas):
                pred_list = [outs[i] for outs in per_model_outputs]
                fused = _fuse_predictions_single_image(
                    pred_list=pred_list,
                    width=int(meta["width"]),
                    height=int(meta["height"]),
                    use_wbf=cfg.inference.use_wbf,
                    nms_iou_thr=cfg.inference.nms_iou_thr,
                )
                fused_outputs.append(fused)

            batch_rows = predictions_to_submission_rows(
                outputs=fused_outputs,
                metas=metas,
                score_threshold=cfg.inference.min_score,
                category_id=1,
                max_detections_per_image=cfg.inference.max_detections_per_image,
                box_format="xyxy",
            )
            rows.extend(batch_rows)

    validate_submission_schema(rows)
    save_submission_auto(rows, output_path)

    for model in models:
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Ensemble submission saved: {output_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
