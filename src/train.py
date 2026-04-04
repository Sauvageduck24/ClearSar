from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from src.config import Config, default_config, ensure_dirs
from src.dataset import (
    ClearSARCocoDataset,
    collate_detection_batch,
    get_train_transforms,
    get_valid_transforms,
    split_train_val_image_ids,
    xyxy_to_coco_xywh,
)
from src.cascade_mmdet import is_cascade_architecture, train_cascade_rcnn
from src.model import build_model
from src.utils.repro import resolve_device, set_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ClearSAR detector")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--profile", type=str, choices=["fast", "balanced", "quality"], default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument(
        "--arch",
        type=str,
        choices=[
            "fasterrcnn_resnet50_fpn_v2",
            "fasterrcnn_mobilenet_v3_large_fpn",
            "cascade_rcnn_swin_l",
            "cascade_rcnn_convnext_xl",
        ],
        default=None,
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--disable-step-limits", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--preprocess", type=str, choices=["none", "percentile"], default=None)
    parser.add_argument("--save-top-k", type=int, default=None)
    parser.add_argument("--ssl-backbone-path", type=str, default=None)
    parser.add_argument("--extra-images-dir", type=str, default=None)
    parser.add_argument("--extra-annotations-path", type=str, default=None)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.profile == "fast":
        cfg.model.architecture = "fasterrcnn_mobilenet_v3_large_fpn"
        cfg.train.epochs = 5
        cfg.train.batch_size = min(cfg.train.batch_size, 4)
        cfg.train.early_stopping_patience = 3
        cfg.train.image_size = (320, 320)
        cfg.train.max_train_steps_per_epoch = 120
        cfg.train.max_val_steps = 40

    if args.profile == "balanced":
        cfg.model.architecture = "fasterrcnn_mobilenet_v3_large_fpn"
        cfg.train.epochs = max(cfg.train.epochs, 30)
        cfg.train.early_stopping_patience = max(cfg.train.early_stopping_patience, 8)
        cfg.train.image_size = None
        cfg.train.max_train_steps_per_epoch = None
        cfg.train.max_val_steps = None

    if args.profile == "quality":
        cfg.model.architecture = "fasterrcnn_resnet50_fpn_v2"
        cfg.train.epochs = cfg.train.epochs
        cfg.train.batch_size = min(cfg.train.batch_size, 3)
        cfg.train.early_stopping_patience = max(cfg.train.early_stopping_patience, 12)
        cfg.train.image_size = None
        cfg.train.max_train_steps_per_epoch = None
        cfg.train.max_val_steps = None

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.max_train_steps is not None:
        cfg.train.max_train_steps_per_epoch = args.max_train_steps
    if args.max_val_steps is not None:
        cfg.train.max_val_steps = args.max_val_steps
    if args.arch is not None:
        cfg.model.architecture = args.arch
    if args.image_size is not None:
        cfg.train.image_size = (args.image_size, args.image_size)
    if args.disable_step_limits:
        cfg.train.max_train_steps_per_epoch = None
        cfg.train.max_val_steps = None
    if args.grad_accum_steps is not None:
        cfg.train.grad_accum_steps = max(1, int(args.grad_accum_steps))
    if args.preprocess is not None:
        cfg.train.preprocess_mode = args.preprocess
    if args.save_top_k is not None:
        cfg.train.save_top_k = max(1, int(args.save_top_k))
    if args.ssl_backbone_path is not None:
        cfg.model.ssl_backbone_path = Path(args.ssl_backbone_path)

    if args.fast:
        cfg.model.architecture = "fasterrcnn_mobilenet_v3_large_fpn"
        cfg.train.epochs = min(cfg.train.epochs, 5)
        cfg.train.batch_size = min(cfg.train.batch_size, 4)
        cfg.train.early_stopping_patience = min(cfg.train.early_stopping_patience, 3)
        cfg.train.image_size = (320, 320)
        cfg.train.max_train_steps_per_epoch = 120
        cfg.train.max_val_steps = 40


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    max_steps: int | None,
    grad_accum_steps: int,
) -> float:
    model.train()
    loss_meter = 0.0
    n_steps = 0

    total_steps = min(len(loader), max_steps) if max_steps is not None else len(loader)
    iterator = itertools.islice(loader, total_steps) if max_steps is not None else loader

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(iterator, total=total_steps, desc="train", leave=False)

    for step_idx, (images, targets) in enumerate(pbar, start=1):
        images = [img.to(device) for img in images]
        targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets_gpu)
            loss = sum(v for v in loss_dict.values())

            pbar.set_postfix(loss=float(loss.item()))

            loss_for_backward = loss / float(grad_accum_steps)

        if use_amp and scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (step_idx % grad_accum_steps == 0) or (step_idx == total_steps)
        if not should_step:
            loss_meter += float(loss.item())
            n_steps += 1
            continue

        if grad_clip_norm is not None:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_meter += float(loss.item())
        n_steps += 1

    return loss_meter / max(1, n_steps)


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    coco_gt: COCO,
    cfg: Config,
    max_steps: Optional[int] = None,
) -> float:
    """
    Evalúa el modelo usando la métrica mAP de COCO, reescalando las predicciones
    al tamaño original de la imagen para una comparación válida.
    """
    model.eval()
    results = []
    image_ids = set()  # Para recopilar las IDs de las imágenes procesadas

    pbar = tqdm(val_loader, desc="Eval", leave=False)
    for i, (images, targets) in enumerate(pbar):
        if max_steps is not None and i >= max_steps:
            break

        images = [img.to(device) for img in images]
        outputs = model(images)

        for j, output in enumerate(outputs):
            image_id = int(targets[j]["image_id"].item())
            image_ids.add(image_id)  # Agregar la ID de la imagen procesada

            # 1. Obtener dimensiones originales del JSON de COCO
            img_info = coco_gt.loadImgs(image_id)[0]
            orig_w, orig_h = img_info['width'], img_info['height']

            # 2. Obtener dimensiones actuales de la imagen procesada
            # La imagen en el loader tiene forma [C, H, W]
            curr_h, curr_w = images[j].shape[1], images[j].shape[2]

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for b, s, l in zip(boxes, scores, labels):
                # 3. REESCALAR coordenadas al tamaño original
                # Multiplicamos la coordenada relativa por el tamaño original
                x1 = b[0] * (orig_w / curr_w)
                y1 = b[1] * (orig_h / curr_h)
                x2 = b[2] * (orig_w / curr_w)
                y2 = b[3] * (orig_h / curr_h)

                # 4. Convertir de [x1, y1, x2, y2] a COCO [x, y, w, h]
                coco_box = xyxy_to_coco_xywh([x1, y1, x2, y2])

                results.append({
                    "image_id": image_id,
                    "category_id": cfg.train.category_id,  # cfg is now passed as an argument
                    "bbox": coco_box,
                    "score": float(s),
                })

    # Si no hay detecciones, el mAP es 0
    if not results:
        return 0.0

    try:
        # Cargar resultados en formato COCO y evaluar
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

        # Decirle a COCOeval que solo evalúe las imágenes procesadas
        coco_eval.params.imgIds = list(image_ids)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # El primer valor de stats es mAP @ [IoU=0.50:0.95]
        return float(coco_eval.stats[0])
    except Exception as e:
        print(f"Error durante la evaluación de COCO: {e}")
        return 0.0


def main() -> None:
    args = parse_args()
    cfg = default_config(project_root=args.project_root)
    apply_overrides(cfg, args)
    ensure_dirs(cfg)

    if is_cascade_architecture(cfg.model.architecture):
        print(f"[train] Using MMDetection Cascade pipeline: {cfg.model.architecture}")
        train_cascade_rcnn(cfg)
        return

    set_seed(cfg.train.seed)
    device = resolve_device()

    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        torch.cuda.empty_cache()

    if device.type == "cpu" and cfg.train.num_workers > 0:
        cfg.train.num_workers = 0

    use_amp = cfg.train.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    train_ids, val_ids = split_train_val_image_ids(
        annotation_path=cfg.paths.train_annotations_path,
        val_fraction=cfg.train.val_fraction,
        seed=cfg.train.seed,
    )

    effective_batch_size = cfg.train.batch_size * max(1, cfg.train.grad_accum_steps)
    print(
        f"batch_size={cfg.train.batch_size}, grad_accum_steps={cfg.train.grad_accum_steps}, "
        f"effective_batch_size={effective_batch_size}"
    )

    train_ds = ClearSARCocoDataset(
        images_dir=cfg.paths.train_images_dir,
        annotation_path=cfg.paths.train_annotations_path,
        image_ids=train_ids,
        transforms=get_train_transforms(cfg.train.image_size),
        category_id=cfg.train.category_id,
        preprocess_mode=cfg.train.preprocess_mode,
    )

    if args.extra_images_dir and args.extra_annotations_path:
        extra_ds = ClearSARCocoDataset(
            images_dir=Path(args.extra_images_dir),
            annotation_path=Path(args.extra_annotations_path),
            image_ids=None,
            transforms=get_train_transforms(cfg.train.image_size),
            category_id=cfg.train.category_id,
            preprocess_mode=cfg.train.preprocess_mode,
        )
        train_ds = ConcatDataset([train_ds, extra_ds])
        print(f"Extra pseudo dataset enabled: {len(extra_ds)} images")
    val_ds = ClearSARCocoDataset(
        images_dir=cfg.paths.train_images_dir,
        annotation_path=cfg.paths.train_annotations_path,
        image_ids=val_ids,
        transforms=get_valid_transforms(cfg.train.image_size),
        category_id=cfg.train.category_id,
        preprocess_mode=cfg.train.preprocess_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
    )

    model = build_model(cfg.model, device=device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    warmup_epochs = min(5, max(0, cfg.train.epochs - 1))
    backbone_freeze_epochs = min(3, max(0, cfg.train.epochs - 1))
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, cfg.train.epochs - warmup_epochs))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.train.epochs))

    coco_gt = COCO(str(cfg.paths.train_annotations_path))

    best_map = -1.0
    epochs_without_improve = 0
    history: List[Dict[str, Any]] = []
    top_checkpoints: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        if getattr(model, "_ssl_backbone_frozen", False) and epoch == (backbone_freeze_epochs + 1):
            for param in model.backbone.body.parameters():
                param.requires_grad = True
            setattr(model, "_ssl_backbone_frozen", False)
            print(f"[train] Backbone unfrozen at epoch {epoch}")

        avg_train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=cfg.train.grad_clip_norm,
            use_amp=use_amp,
            scaler=scaler,
            max_steps=cfg.train.max_train_steps_per_epoch,
            grad_accum_steps=max(1, cfg.train.grad_accum_steps),
        )

        val_map = evaluate_map(
            model=model,
            val_loader=val_loader,
            device=device,
            coco_gt=coco_gt,
            cfg=cfg,  # Pass cfg here
            max_steps=cfg.train.max_val_steps,
        )

        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_map_50_95": val_map,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:03d} | loss={avg_train_loss:.4f} | "
            f"mAP50:95={val_map:.5f} | lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        improved = val_map > (best_map + cfg.train.early_stopping_min_delta)
        if improved:
            best_map = val_map
            epochs_without_improve = 0

            best_path = cfg.paths.models_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "best_map": best_map,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg.to_dict(),
                },
                best_path,
            )
            print(f"[checkpoint] best model saved -> {best_path}")
        else:
            epochs_without_improve += 1

        rank_path = cfg.paths.models_dir / f"ckpt_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "best_map": best_map,
                "val_map_50_95": val_map,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.to_dict(),
            },
            rank_path,
        )
        top_checkpoints.append(
            {
                "epoch": epoch,
                "val_map_50_95": val_map,
                "path": str(rank_path),
            }
        )
        top_checkpoints = sorted(top_checkpoints, key=lambda x: x["val_map_50_95"], reverse=True)
        if len(top_checkpoints) > cfg.train.save_top_k:
            to_remove = top_checkpoints[cfg.train.save_top_k :]
            top_checkpoints = top_checkpoints[: cfg.train.save_top_k]
            for item in to_remove:
                p = Path(item["path"])
                if p.exists():
                    p.unlink()

        if epochs_without_improve >= cfg.train.early_stopping_patience:
            print(
                "Early stopping activated: "
                f"no mAP improvement in {cfg.train.early_stopping_patience} epochs."
            )
            break

    last_path = cfg.paths.models_dir / "last_model.pt"
    torch.save(
        {
            "epoch": history[-1]["epoch"] if history else 0,
            "best_map": best_map,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.to_dict(),
        },
        last_path,
    )

    history_path = cfg.paths.outputs_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)

    cfg_path = cfg.paths.outputs_dir / "resolved_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=True, indent=2)

    print(f"Best mAP50:95 = {best_map:.5f}")
    print(f"History saved at {history_path}")

    top_path = cfg.paths.outputs_dir / "top_checkpoints.json"
    with top_path.open("w", encoding="utf-8") as f:
        json.dump(top_checkpoints, f, ensure_ascii=True, indent=2)
    print(f"Top checkpoints saved at {top_path}")


if __name__ == "__main__":
    main()
