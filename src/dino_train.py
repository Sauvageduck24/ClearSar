from __future__ import annotations

"""
DINO (Detection Transformer with Improved Denoising) training for ClearSAR.
Best-possible configuration for small objects:
  - Swin-L or ResNet-50 backbone (configurable)
  - High-resolution input via tiling (1024px tiles)
  - Multi-scale feature maps
  - No-anchor, end-to-end detection

Requires:
    pip install torch torchvision
    pip install git+https://github.com/IDEACVR/DINO.git
    # O alternativa con transformers:
    pip install transformers accelerate

Este script usa la implementación de DINO disponible en HuggingFace Transformers
(AutoModelForObjectDetection con "DINO" / "conditional-detr").

Si prefieres el repo oficial de IDEA-Research, adapta el loader del modelo.

Usage:
    python dino_train.py --project-root . --epochs 50 --backbone swin_l
"""

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.ops import nms

# Usamos transformers de HuggingFace para DINO
try:
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForObjectDetection,
        get_cosine_schedule_with_warmup,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

MIN_BOX_AREA = 1.0
TILE_SIZE = 1024         # Alta resolución para DINO
TILE_OVERLAP = 200
MAX_BOXES_PER_IMAGE = 300  # DINO tiene un límite de queries
NMS_IOU_THRESH = 0.4

# Alias de entrenamiento compatibles con AutoModelForObjectDetection.
# GroundingDINO no entra aqui porque requiere un pipeline de grounding con texto,
# no el flujo de deteccion supervisada usado por este script.
DINO_MODELS = {
    "dino_r50":    "facebook/detr-resnet-50",
    "dino_swin_t": "microsoft/conditional-detr-resnet-50",
    "dino_swin_b": "facebook/detr-resnet-101",
    # Para DINO/DETR puro:
    "dino_r50_detr": "facebook/detr-resnet-50",
    "dino_r101_detr": "facebook/detr-resnet-101",
    # Conditional DETR (más rápido que vanilla DETR, similar a DINO)
    "conditional_detr": "microsoft/conditional-detr-resnet-50",
}

# ──────────────────────────────────────────────────────────────────────────────
# TILING (igual que retinanet_train pero con tile_size mayor)
# ──────────────────────────────────────────────────────────────────────────────

def compute_tiles(img_w: int, img_h: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    stride = tile_size - overlap
    tiles = []
    y = 0
    while y < img_h:
        y2 = min(y + tile_size, img_h)
        x = 0
        while x < img_w:
            x2 = min(x + tile_size, img_w)
            tiles.append((x, y, x2, y2))
            if x2 == img_w:
                break
            x += stride
        if y2 == img_h:
            break
        y += stride
    return tiles


def pad_tile(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w == size and h == size:
        return img
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, (0, 0))
    return canvas


def clip_boxes_to_tile(boxes: np.ndarray, tile: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    tx1, ty1, tx2, ty2 = tile
    clipped = boxes.copy()
    clipped[:, 0] = np.clip(boxes[:, 0], tx1, tx2) - tx1
    clipped[:, 1] = np.clip(boxes[:, 1], ty1, ty2) - ty1
    clipped[:, 2] = np.clip(boxes[:, 2], tx1, tx2) - tx1
    clipped[:, 3] = np.clip(boxes[:, 3], ty1, ty2) - ty1
    w = clipped[:, 2] - clipped[:, 0]
    h = clipped[:, 3] - clipped[:, 1]
    valid = (w > 1) & (h > 1) & ((w * h) >= MIN_BOX_AREA)
    return clipped[valid], valid


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class TiledDINODataset(Dataset):
    """
    Dataset con tiling de alta resolución para DINO/DETR.
    Las cajas se normalizan a [0,1] en formato CXCYWH (requerido por DETR/DINO).
    """

    def __init__(
        self,
        annotation_path: Path,
        images_dir: Path,
        image_processor,
        image_ids: Optional[List[int]] = None,
        tile_size: int = TILE_SIZE,
        overlap: int = TILE_OVERLAP,
        augment: bool = True,
        max_boxes: int = MAX_BOXES_PER_IMAGE,
    ):
        with annotation_path.open() as f:
            coco = json.load(f)

        self.images_dir = images_dir
        self.processor = image_processor
        self.tile_size = tile_size
        self.overlap = overlap
        self.augment = augment
        self.max_boxes = max_boxes

        img_meta = {img["id"]: img for img in coco["images"]}
        anns_by_img: Dict[int, List] = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        ids = image_ids if image_ids else list(img_meta.keys())
        self.samples = []

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

            for tile in compute_tiles(W, H, tile_size, overlap):
                if len(boxes_np) > 0:
                    tile_boxes, _ = clip_boxes_to_tile(boxes_np, tile)
                else:
                    tile_boxes = np.zeros((0, 4), dtype=np.float32)
                self.samples.append((meta, tile, tile_boxes))

        print(f"[TiledDINODataset] {len(ids)} imgs → {len(self.samples)} tiles (tile={tile_size}px)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta, (tx1, ty1, tx2, ty2), boxes = self.samples[idx]
        img = Image.open(self.images_dir / meta["file_name"]).convert("RGB")
        tile_img = pad_tile(img.crop((tx1, ty1, tx2, ty2)), self.tile_size)
        tw = tx2 - tx1
        th = ty2 - ty1

        # Augmentación
        if self.augment:
            if random.random() > 0.5:
                tile_img = TF.hflip(tile_img)
                if len(boxes) > 0:
                    boxes_aug = boxes.copy()
                    boxes_aug[:, 0] = tw - boxes[:, 2]
                    boxes_aug[:, 2] = tw - boxes[:, 0]
                    boxes = boxes_aug
            if random.random() > 0.5:
                tile_img = TF.vflip(tile_img)
                if len(boxes) > 0:
                    boxes_aug = boxes.copy()
                    boxes_aug[:, 1] = th - boxes[:, 3]
                    boxes_aug[:, 3] = th - boxes[:, 1]
                    boxes = boxes_aug

        # Convertir a CXCYWH normalizado (formato DETR)
        if len(boxes) > 0:
            boxes_cxcywh = np.stack([
                (boxes[:, 0] + boxes[:, 2]) / 2 / self.tile_size,  # cx
                (boxes[:, 1] + boxes[:, 3]) / 2 / self.tile_size,  # cy
                (boxes[:, 2] - boxes[:, 0]) / self.tile_size,       # w
                (boxes[:, 3] - boxes[:, 1]) / self.tile_size,       # h
            ], axis=1)
            boxes_cxcywh = np.clip(boxes_cxcywh, 0, 1)
            # Limitar número de cajas
            if len(boxes_cxcywh) > self.max_boxes:
                boxes_cxcywh = boxes_cxcywh[:self.max_boxes]
        else:
            boxes_cxcywh = np.zeros((0, 4), dtype=np.float32)

        labels = [0] * len(boxes_cxcywh)  # clase única

        # Procesar con HuggingFace image_processor
        encoding = self.processor(
            images=tile_img,
            annotations={
                "image_id": idx,
                "annotations": [
                    {
                        "id": i,
                        "image_id": idx,
                        "category_id": 0,
                        "bbox": [
                            (b[0] - b[2] / 2) * self.tile_size,
                            (b[1] - b[3] / 2) * self.tile_size,
                            b[2] * self.tile_size,
                            b[3] * self.tile_size,
                        ],
                        "area": b[2] * b[3] * self.tile_size ** 2,
                        "iscrowd": 0,
                    }
                    for i, b in enumerate(boxes_cxcywh)
                ],
            },
            return_tensors="pt",
        )
        return encoding


def collate_dino(batch):
    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
    labels = [b["labels"][0] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}


class DinoValImageDataset(Dataset):
    """Dataset de validacion en imagen completa para mAP COCO real."""

    def __init__(self, annotation_path: Path, images_dir: Path, image_ids: List[int], processor):
        with annotation_path.open() as f:
            coco = json.load(f)
        self.images_dir = images_dir
        self.processor = processor
        self.meta_by_id = {img["id"]: img for img in coco["images"]}
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        meta = self.meta_by_id[image_id]
        img = Image.open(self.images_dir / meta["file_name"]).convert("RGB")
        encoding = self.processor(images=img, return_tensors="pt")
        pixel_values = encoding["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "image_id": int(image_id),
            "orig_h": int(meta["height"]),
            "orig_w": int(meta["width"]),
        }


def collate_dino_eval(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    image_ids = [b["image_id"] for b in batch]
    target_sizes = [[b["orig_h"], b["orig_w"]] for b in batch]
    return {"pixel_values": pixel_values, "image_ids": image_ids, "target_sizes": target_sizes}


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, scaler, scheduler, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar, start=1):
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        amp_ctx = (
            torch.amp.autocast(device_type=device.type, enabled=True)
            if scaler is not None
            else nullcontext()
        )
        with amp_ctx:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()
    pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / step:.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Val", leave=False)
    for step, batch in enumerate(pbar, start=1):
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss_value = outputs.loss.item()
        total_loss += loss_value
        pbar.set_postfix(loss=f"{loss_value:.4f}", avg=f"{total_loss / step:.4f}")
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_map_tiled(
    model,
    processor,
    image_ids: List[int],
    images_dir: Path,
    meta_by_id: Dict[int, Dict[str, Any]],
    device,
    coco_gt: COCO,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
    category_id: int = 1,
):
    model.eval()
    results: List[Dict[str, Any]] = []
    image_ids_eval: List[int] = []

    for image_id in tqdm(image_ids, desc="Val mAP (tiled)", leave=False):
        image_id = int(image_id)
        meta = meta_by_id[image_id]
        img_w, img_h = int(meta["width"]), int(meta["height"])
        img = Image.open(images_dir / meta["file_name"]).convert("RGB")

        tiles = compute_tiles(img_w, img_h, tile_size, overlap)
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []

        for tx1, ty1, tx2, ty2 in tiles:
            tile_img = pad_tile(img.crop((tx1, ty1, tx2, ty2)), tile_size)
            encoding = processor(images=tile_img, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)

            outputs = model(pixel_values=pixel_values)
            target_sizes = torch.tensor([[tile_size, tile_size]], device=device)
            detections = processor.post_process_object_detection(
                outputs,
                threshold=0.001,
                target_sizes=target_sizes,
            )
            det = detections[0]

            boxes = det["boxes"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            labels = det["labels"].detach().cpu().numpy() if "labels" in det else np.zeros(len(boxes))
            if len(boxes) == 0:
                continue

            keep_cls = labels == 0
            if not np.any(keep_cls):
                continue

            boxes = boxes[keep_cls]
            scores = scores[keep_cls]

            boxes[:, 0] += tx1
            boxes[:, 2] += tx1
            boxes[:, 1] += ty1
            boxes[:, 3] += ty1
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

        image_ids_eval.append(image_id)
        for i in keep:
            x1, y1, x2, y2 = merged_boxes[i].tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0.0 or h <= 0.0:
                continue
            results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(merged_scores[i]),
                }
            )

    if not results:
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = sorted(set(image_ids_eval))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


# ──────────────────────────────────────────────────────────────────────────────
# ARGS & MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DINO/DETR training for ClearSAR")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--model-name", type=str, default="conditional_detr",
                   choices=list(DINO_MODELS.keys()),
                   help="Variante DINO/DETR a usar")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Tiles por batch. DINO usa mucha VRAM: bs=2 para 24GB con tile=1024")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile-size", type=int, default=TILE_SIZE)
    p.add_argument("--tile-overlap", type=int, default=TILE_OVERLAP)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--gradient-accumulation", type=int, default=4,
                   help="Pasos de acumulación de gradiente para simular batch mayor")
    return p.parse_args()


def split_ids(annotation_path: Path, val_fraction: float, seed: int):
    with annotation_path.open() as f:
        coco = json.load(f)
    ids = [img["id"] for img in coco["images"]]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    return ids[n_val:], ids[:n_val]


def main():
    if not HF_AVAILABLE:
        raise ImportError(
            "transformers requerido.\n"
            "Instala con: pip install transformers accelerate"
        )

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_root = Path(args.project_root)
    ann_path = project_root / "data" / "annotations" / "instances_train.json"
    train_images_dir = project_root / "data" / "images" / "train"
    output_dir = project_root / "outputs" / "dino"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    hf_model_name = DINO_MODELS[args.model_name]

    print(f"[dino] Device: {device}")
    print(f"[dino] Modelo HF: {hf_model_name}")
    print(f"[dino] Tile size: {args.tile_size}px (alta resolución)")

    # Cargar processor y modelo
    processor = AutoImageProcessor.from_pretrained(
        hf_model_name,
        size={"height": args.tile_size, "width": args.tile_size},
    )
    config = AutoConfig.from_pretrained(hf_model_name)
    config.num_labels = 1
    config.id2label = {0: "RFI"}
    config.label2id = {"RFI": 0}
    model = AutoModelForObjectDetection.from_pretrained(
        hf_model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        print(f"[dino] Resumed from {args.resume}")

    # Split
    train_ids, val_ids = split_ids(ann_path, args.val_fraction, args.seed)
    print(f"[dino] Train: {len(train_ids)} imgs | Val: {len(val_ids)} imgs")

    # Datasets
    train_ds = TiledDINODataset(ann_path, train_images_dir, processor,
                                 train_ids, args.tile_size, args.tile_overlap, augment=True)
    val_ds = TiledDINODataset(ann_path, train_images_dir, processor,
                               val_ids, args.tile_size, args.tile_overlap, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_dino)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_dino)

    coco_gt = COCO(str(ann_path))
    meta_by_id = {int(img_id): meta for img_id, meta in coco_gt.imgs.items()}

    # Optimizador con LR diferencial backbone/head (crucial para DINO)
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    other_params   = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": other_params,    "lr": args.lr},
    ], weight_decay=args.weight_decay)

    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation))
    total_steps = max(1, updates_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_map = -1.0
    patience = 0
    PATIENCE = 15

    print(f"[dino] Tiles de entrenamiento: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"[dino] Gradient accumulation: {args.gradient_accumulation} → effective batch = {args.batch_size * args.gradient_accumulation}")
    print(f"[dino] Iniciando entrenamiento por {args.epochs} épocas...")

    for epoch in range(args.epochs):
        # ── Train con gradient accumulation ────────────────────────────────
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, total=len(train_loader),
                desc=f"Epoch {epoch}", leave=False)

        for step, batch in enumerate(pbar, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            amp_ctx = (
                torch.amp.autocast(device_type=device.type, enabled=True)
                if scaler is not None
                else nullcontext()
            )
            with amp_ctx:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / args.gradient_accumulation

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * args.gradient_accumulation
            pbar.set_postfix(
                loss=f"{loss.item() * args.gradient_accumulation:.4f}",
                avg=f"{total_loss / max(step, 1):.4f}",
            )

            if step % args.gradient_accumulation == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    # Si AMP salto optimizer.step por overflow, no avanzamos scheduler.
                    if scaler.get_scale() >= prev_scale:
                        scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad()

        # Flush del ultimo micro-batch si no cae exacto en gradient_accumulation.
        if len(train_loader) % args.gradient_accumulation != 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= prev_scale:
                    scheduler.step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()

        train_loss = total_loss / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device)
        val_map = evaluate_map_tiled(
            model,
            processor,
            image_ids=val_ids,
            images_dir=train_images_dir,
            meta_by_id=meta_by_id,
            device=device,
            coco_gt=coco_gt,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            category_id=1,
        )
        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"mAP50:95={val_map:.5f} | lr={lr_now:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "val_loss": val_loss,
            "val_map_50_95": val_map,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_map > best_map:
            best_map = val_map
            patience = 0
            torch.save(ckpt, output_dir / "best.pt")
            # Guardar también en formato HuggingFace para fácil carga
            model.save_pretrained(output_dir / "best_hf")
            processor.save_pretrained(output_dir / "best_hf")
            print(f"  ✓ Nuevo best (mAP50:95={val_map:.5f}) → {output_dir / 'best.pt'}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[dino] Early stopping en época {epoch}")
                break

    print(f"[dino] Entrenamiento completo. Best mAP50:95={best_map:.5f}")
    print(f"[dino] Checkpoint: {output_dir / 'best.pt'}")
    print(f"[dino] HF format:  {output_dir / 'best_hf'}")


if __name__ == "__main__":
    main()
