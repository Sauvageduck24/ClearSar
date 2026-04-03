from __future__ import annotations

"""
RetinaNet training for ClearSAR — small-object optimized.

Strategy:
  - Custom anchors tuned for small/thin RFI stripes
  - Tiling: each image is split into overlapping tiles at train time
  - Torchvision RetinaNet backbone (ResNet-50-FPN or ResNet-101-FPN)

Usage:
    python retinanet_train.py --project-root . --epochs 80 --backbone resnet101
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import nms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.dataset import split_train_val_image_ids

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

TILE_SIZE = 512          # píxeles por tile (cuadrado)
TILE_OVERLAP = 128       # solapamiento entre tiles
MIN_BOX_AREA = 1.0       # descartar cajas menores a esta área en px²
NMS_IOU_THRESH = 0.4     # NMS en ensamble de tiles durante inferencia
SCORE_THRESH = 0.05      # umbral de score mínimo en detección
DETECTIONS_PER_IMG = 500

# Anchors custom para RFI (rayas muy finas y largas)
# sizes   → escalas absolutas en píxeles
# ratios  → aspect ratios (ancho/alto); RFI tiene ratios muy grandes
ANCHOR_SIZES   = ((8,), (16,), (32,), (64,), (128,))   # por nivel FPN
ANCHOR_RATIOS  = ((0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),) * 5  # mismo para cada nivel


# ──────────────────────────────────────────────────────────────────────────────
# TILING UTILS
# ──────────────────────────────────────────────────────────────────────────────

def compute_tiles(img_w: int, img_h: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Devuelve lista de (x1, y1, x2, y2) para cada tile."""
    stride = tile_size - overlap
    tiles = []
    y = 0
    while y < img_h:
        x = 0
        y2 = min(y + tile_size, img_h)
        while x < img_w:
            x2 = min(x + tile_size, img_w)
            # Ajustar para que el tile siempre tenga tile_size (pad implícito después)
            tiles.append((x, y, x2, y2))
            if x2 == img_w:
                break
            x += stride
        if y2 == img_h:
            break
        y += stride
    return tiles


def clip_boxes_to_tile(boxes: np.ndarray, tile: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recorta las cajas (xyxy absolutas) al tile y devuelve las cajas en
    coordenadas relativas al tile + máscara de cajas válidas.
    """
    tx1, ty1, tx2, ty2 = tile
    clipped = boxes.copy()
    clipped[:, 0] = np.clip(boxes[:, 0], tx1, tx2) - tx1
    clipped[:, 1] = np.clip(boxes[:, 1], ty1, ty2) - ty1
    clipped[:, 2] = np.clip(boxes[:, 2], tx1, tx2) - tx1
    clipped[:, 3] = np.clip(boxes[:, 3], ty1, ty2) - ty1

    w = clipped[:, 2] - clipped[:, 0]
    h = clipped[:, 3] - clipped[:, 1]
    area = w * h
    valid = (w > 1) & (h > 1) & (area >= MIN_BOX_AREA)
    return clipped[valid], valid


def pad_to_square(img: Image.Image, size: int) -> Image.Image:
    """Pad con ceros (negro) hasta size×size."""
    w, h = img.size
    if w == size and h == size:
        return img
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, (0, 0))
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class TiledCOCODataset(Dataset):
    """
    Dataset con tiling LAZY: los tiles se generan en __getitem__, no en __init__.
    El __init__ ahora es O(N_imagenes) en vez de O(N_imagenes × N_tiles).
    """

    def __init__(
        self,
        annotation_path: Path,
        images_dir: Path,
        image_ids: Optional[List[int]] = None,
        tile_size: int = TILE_SIZE,
        overlap: int = TILE_OVERLAP,
        augment: bool = True,
    ):
        with annotation_path.open() as f:
            coco = json.load(f)

        self.images_dir = images_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.augment = augment

        img_meta = {img["id"]: img for img in coco["images"]}
        anns_by_img: Dict[int, List] = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        ids = image_ids if image_ids else list(img_meta.keys())

        # Guardar solo metadatos por imagen (no expandir tiles aquí)
        self.image_records: List[Tuple[Dict, np.ndarray]] = []
        self._tile_offsets: List[int] = [0]   # offset acumulado para mapeo idx→(img, tile)

        for img_id in ids:
            meta = img_meta[img_id]
            W, H = meta["width"], meta["height"]
            anns = anns_by_img.get(img_id, [])
            boxes = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
            boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

            n_tiles = len(compute_tiles(W, H, tile_size, overlap))
            self.image_records.append((meta, boxes_np))
            self._tile_offsets.append(self._tile_offsets[-1] + n_tiles)

        self._total = self._tile_offsets[-1]
        print(f"[TiledDataset] {len(ids)} imágenes → {self._total} tiles (lazy)")

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        # Buscar a qué imagen y tile corresponde este idx global
        # bisect es O(log N_imagenes)
        import bisect
        img_idx = bisect.bisect_right(self._tile_offsets, idx) - 1
        tile_local_idx = idx - self._tile_offsets[img_idx]

        meta, boxes_np = self.image_records[img_idx]
        W, H = meta["width"], meta["height"]
        tiles = compute_tiles(W, H, self.tile_size, self.overlap)
        tx1, ty1, tx2, ty2 = tiles[tile_local_idx]

        img_path = self.images_dir / meta["file_name"]
        img = Image.open(img_path).convert("RGB")
        tile_img = img.crop((tx1, ty1, tx2, ty2))
        tile_img = pad_to_square(tile_img, self.tile_size)

        # Recortar boxes al tile (antes estaba en __init__)
        if len(boxes_np) > 0:
            boxes, _ = clip_boxes_to_tile(boxes_np, (tx1, ty1, tx2, ty2))
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        # Augmentación
        if self.augment:
            if random.random() > 0.5:
                tile_img = TF.hflip(tile_img)
                if len(boxes) > 0:
                    tw = tx2 - tx1
                    boxes_aug = boxes.copy()
                    boxes_aug[:, 0] = tw - boxes[:, 2]
                    boxes_aug[:, 2] = tw - boxes[:, 0]
                    boxes = boxes_aug
            if random.random() > 0.5:
                tile_img = TF.vflip(tile_img)
                if len(boxes) > 0:
                    th = ty2 - ty1
                    boxes_aug = boxes.copy()
                    boxes_aug[:, 1] = th - boxes[:, 3]
                    boxes_aug[:, 3] = th - boxes[:, 1]
                    boxes = boxes_aug
            tile_img = TF.adjust_brightness(tile_img, 1.0 + random.uniform(-0.2, 0.2))
            tile_img = TF.adjust_contrast(tile_img, 1.0 + random.uniform(-0.2, 0.2))

        img_tensor = TF.to_tensor(tile_img)

        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            keep = (boxes_t[:, 2] > boxes_t[:, 0]) & (boxes_t[:, 3] > boxes_t[:, 1])
            boxes_t = boxes_t[keep]
            labels = torch.zeros(len(boxes_t), dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        return img_tensor, {"boxes": boxes_t, "labels": labels}

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class FullImageValDataset(Dataset):
    """Dataset de validacion en imagen completa para evaluar mAP COCO real."""

    def __init__(self, annotation_path: Path, images_dir: Path, image_ids: List[int]):
        with annotation_path.open() as f:
            coco = json.load(f)
        self.images_dir = images_dir
        self.meta_by_id = {img["id"]: img for img in coco["images"]}
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        meta = self.meta_by_id[image_id]
        img = Image.open(self.images_dir / meta["file_name"]).convert("RGB")
        img_tensor = TF.to_tensor(img)
        return img_tensor, {"image_id": int(image_id)}


def collate_eval_fn(batch):
    images, metas = zip(*batch)
    return list(images), list(metas)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

def build_model(backbone_name: str = "resnet50", num_classes: int = 1, pretrained: bool = True) -> RetinaNet:
    """
    Construye un RetinaNet con:
      - Backbone ResNet-FPN (50 o 101)
      - Anchors custom para objetos pequeños y de aspecto extremo
    """
    trainable_layers = 3
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights="DEFAULT" if pretrained else None,
        trainable_layers=trainable_layers,
    )

    anchor_generator = AnchorGenerator(
        sizes=ANCHOR_SIZES,
        aspect_ratios=ANCHOR_RATIOS,
    )

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        score_thresh=SCORE_THRESH,
        nms_thresh=NMS_IOU_THRESH,
        detections_per_img=DETECTIONS_PER_IMG,
        fg_iou_thresh=0.45,
        bg_iou_thresh=0.35,
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def _validate_targets_for_retinanet(targets: List[Dict], num_classes: int) -> None:
    """Valida que labels esten en rango [0, num_classes-1] (API RetinaNet torchvision)."""
    for i, t in enumerate(targets):
        labels = t.get("labels")
        if labels is None or labels.numel() == 0:
            continue
        min_label = int(labels.min().item())
        max_label = int(labels.max().item())
        if min_label < 0 or max_label >= num_classes:
            raise ValueError(
                f"Target fuera de rango en batch item {i}: labels en [{min_label}, {max_label}] "
                f"pero num_classes={num_classes}."
            )


def train_one_epoch(model, optimizer, loader, device, scaler, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    num_classes = int(model.head.classification_head.num_classes)
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for imgs, targets in pbar:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        _validate_targets_for_retinanet(targets, num_classes=num_classes)

        amp_ctx = (
            torch.amp.autocast(device_type=device.type, enabled=True)
            if scaler is not None
            else nullcontext()
        )
        with amp_ctx:
            loss_dict = model(imgs, targets)
            losses = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix(loss=f"{losses.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Evaluación simplificada: promedio del loss en val (modo train para calcular loss)."""
    model.train()  # RetinaNet solo devuelve loss en modo train
    total = 0.0
    for imgs, targets in tqdm(loader, desc="Val", leave=False):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(imgs, targets)
            total += sum(loss_dict.values()).item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_map_tiled(
    model: torch.nn.Module,
    image_ids: List[int],
    images_dir: Path,
    meta_by_id: Dict[int, Dict[str, Any]],
    device: torch.device,
    coco_gt: COCO,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
    category_id: int = 1,
) -> float:
    model.eval()
    results: List[Dict[str, Any]] = []
    seen_image_ids: List[int] = []

    for image_id in tqdm(image_ids, desc="Val mAP (tiled)", leave=False):
        meta = meta_by_id[image_id]
        img_w, img_h = int(meta["width"]), int(meta["height"])
        img = Image.open(images_dir / meta["file_name"]).convert("RGB")

        tiles = compute_tiles(img_w, img_h, tile_size, overlap)
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []

        for tx1, ty1, tx2, ty2 in tiles:
            tile_img = pad_to_square(img.crop((tx1, ty1, tx2, ty2)), tile_size)
            inp = TF.to_tensor(tile_img).unsqueeze(0).to(device)
            output = model(inp)[0]

            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            if len(boxes) == 0:
                continue

            boxes[:, 0] += tx1
            boxes[:, 2] += tx1
            boxes[:, 1] += ty1
            boxes[:, 3] += ty1

            # Asegurar que queden dentro de la imagen original
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

            all_boxes.append(boxes)
            all_scores.append(scores)

        if not all_boxes:
            continue

        merged_boxes = np.concatenate(all_boxes, axis=0)
        merged_scores = np.concatenate(all_scores, axis=0)

        keep = nms(
            torch.as_tensor(merged_boxes, dtype=torch.float32),
            torch.as_tensor(merged_scores, dtype=torch.float32),
            iou_threshold=NMS_IOU_THRESH,
        ).cpu().numpy()

        seen_image_ids.append(image_id)
        for i in keep:
            x1, y1, x2, y2 = merged_boxes[i].tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0.0 or h <= 0.0:
                continue
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(merged_scores[i]),
                }
            )

    if not results:
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = sorted(set(seen_image_ids))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


# ──────────────────────────────────────────────────────────────────────────────
# ARGS & MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RetinaNet tiling training for ClearSAR")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--backbone", type=str, default="resnet50",
                   choices=["resnet50", "resnet101","resnet18"],
                   help="Backbone FPN: resnet50 (más rápido) o resnet101 (mejor calidad)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Tiles por batch. Con tile 512px y bs=4 cabe en ~8GB VRAM")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile-size", type=int, default=TILE_SIZE)
    p.add_argument("--tile-overlap", type=int, default=TILE_OVERLAP)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default=None, help="Ruta a checkpoint .pt para reanudar")
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def split_ids(annotation_path: Path, val_fraction: float, seed: int):
    return split_train_val_image_ids(
        annotation_path=annotation_path,
        val_fraction=val_fraction,
        seed=seed,
        stratify_by_presence=True,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_root = Path(args.project_root)
    ann_path = project_root / "data" / "annotations" / "instances_train.json"
    train_images_dir = project_root / "data" / "images" / "train"
    output_dir = project_root / "outputs" / "retinanet"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[retinanet] Device: {device}")

    # Split
    train_ids, val_ids = split_ids(ann_path, args.val_fraction, args.seed)
    print(f"[retinanet] Train: {len(train_ids)} imgs | Val: {len(val_ids)} imgs")

    # Datasets
    train_ds = TiledCOCODataset(ann_path, train_images_dir, train_ids,
                                 tile_size=args.tile_size, overlap=args.tile_overlap, augment=True)
    val_ds = TiledCOCODataset(ann_path, train_images_dir, val_ids,
                               tile_size=args.tile_size, overlap=args.tile_overlap, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    coco_gt = COCO(str(ann_path))
    meta_by_id = {int(img_id): meta for img_id, meta in coco_gt.imgs.items()}

    # Modelo
    model = build_model(
        backbone_name=args.backbone,
        num_classes=1,
        pretrained=not args.no_pretrained,
    )
    model.to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[retinanet] Resumed from epoch {start_epoch - 1}")

    # Optimizador con LR diferencial (backbone más bajo)
    backbone_params = list(model.backbone.parameters())
    head_params = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_map = -1.0
    patience_counter = 0
    PATIENCE = 20

    print(f"[retinanet] Iniciando entrenamiento por {args.epochs} épocas...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler, epoch)
        val_loss = evaluate(model, val_loader, device)
        val_map = evaluate_map_tiled(
            model,
            image_ids=val_ids,
            images_dir=train_images_dir,
            meta_by_id=meta_by_id,
            device=device,
            coco_gt=coco_gt,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            category_id=1,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | mAP50:95={val_map:.5f} | lr={lr_now:.2e}"
        )

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_map_50_95": val_map,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  ✓ Nuevo best checkpoint (mAP50:95={val_map:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[retinanet] Early stopping en época {epoch}")
                break

    print(f"[retinanet] Entrenamiento completo. Best mAP50:95={best_map:.5f}")
    print(f"[retinanet] Checkpoint guardado en: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
