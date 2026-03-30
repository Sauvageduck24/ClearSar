from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import default_config, ensure_dirs
from src.dataset import ClearSARTestDataset, collate_detection_batch, load_test_id_mapping
from src.inference import _fuse_predictions_single_image, tta_predict_batch
from src.model import apply_checkpoint_model_hints, build_model, load_model_checkpoint
from src.submission import predictions_to_submission_rows
from src.utils.repro import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo labels in COCO format")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mapping-path", type=str, default=None)
    parser.add_argument("--output-annotations", type=str, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--max-dets", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = default_config(project_root=args.project_root)
    ensure_dirs(cfg)

    if args.batch_size is not None:
        cfg.inference.batch_size = args.batch_size

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    mapping_path = Path(args.mapping_path) if args.mapping_path else cfg.paths.test_id_mapping_path
    if mapping_path is None:
        raise ValueError("No mapping path available. Provide --mapping-path")

    out_ann_path = (
        Path(args.output_annotations)
        if args.output_annotations
        else cfg.paths.outputs_dir / "pseudo" / "instances_pseudo_test.json"
    )
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)

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
    model.eval()

    rows: List[Dict[str, Any]] = []
    images_out: Dict[int, Dict[str, Any]] = {}

    with torch.no_grad():
        for images, metas in tqdm(test_loader, total=len(test_loader), desc="pseudo"):
            images_gpu = [img.to(device) for img in images]
            tta_outputs = tta_predict_batch(model, images_gpu, use_tta=args.tta)

            fused_outputs: List[Dict[str, torch.Tensor]] = []
            for out, meta in zip(tta_outputs, metas):
                fused = _fuse_predictions_single_image(
                    pred_list=[out],
                    width=int(meta["width"]),
                    height=int(meta["height"]),
                    use_wbf=True,
                    wbf_iou_thr=cfg.inference.wbf_iou_thr,
                    wbf_skip_box_thr=cfg.inference.wbf_skip_box_thr,
                    nms_iou_thr=cfg.inference.nms_iou_thr,
                )
                fused_outputs.append(fused)

            batch_rows = predictions_to_submission_rows(
                outputs=fused_outputs,
                metas=metas,
                score_threshold=args.score_threshold,
                category_id=1,
                max_detections_per_image=args.max_dets,
                box_format="xyxy",
            )
            rows.extend(batch_rows)

            for m in metas:
                img_id = int(m["image_id"])
                images_out[img_id] = {
                    "id": img_id,
                    "file_name": str(m["file_name"]),
                    "width": int(m["width"]),
                    "height": int(m["height"]),
                }

    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    for r in rows:
        x, y, w, h = [float(v) for v in r["bbox"]]
        if w <= 0.0 or h <= 0.0:
            continue
        annotations.append(
            {
                "id": ann_id,
                "image_id": int(r["image_id"]),
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
                "score": float(r["score"]),
            }
        )
        ann_id += 1

    pseudo_coco = {
        "images": [images_out[k] for k in sorted(images_out.keys())],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "RFI"}],
    }

    with out_ann_path.open("w", encoding="utf-8") as f:
        json.dump(pseudo_coco, f, ensure_ascii=True)

    summary_path = out_ann_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "num_images": len(images_out),
                "num_annotations": len(annotations),
                "score_threshold": args.score_threshold,
                "output": str(out_ann_path),
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"Pseudo labels saved: {out_ann_path}")
    print(f"Images: {len(images_out)} | Annotations: {len(annotations)}")


if __name__ == "__main__":
    main()
