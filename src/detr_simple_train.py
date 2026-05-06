"""
DETR fine-tuning with HuggingFace transformers.
Uses built-in Hungarian-matching loss and proper COCO evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODetectionDataset(Dataset):
    def __init__(
        self,
        annotation_path: Path,
        image_dir: Path,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        with open(annotation_path) as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.annotations_data = data

        self.img_to_anns: dict = {}
        for ann in data["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.img_info = {img["id"]: img for img in data["images"]}
        self.cat_to_id = {cat["id"]: i for i, cat in enumerate(data["categories"])}
        self.num_classes = len(data["categories"])

        np.random.seed(seed)
        all_ids = [img["id"] for img in data["images"]]
        np.random.shuffle(all_ids)
        split_idx = int(len(all_ids) * (1 - val_fraction))
        self.image_ids = all_ids[:split_idx] if split == "train" else all_ids[split_idx:]
        print(f"[DETR] {split} split: {len(self.image_ids)} images, {self.num_classes} classes")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.img_info[img_id]
        image = Image.open(self.image_dir / info["file_name"]).convert("RGB")
        orig_w, orig_h = image.size

        boxes, labels = [], []
        for ann in self.img_to_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            # DETR expects [cx, cy, w, h] normalized to [0, 1]
            cx = np.clip((x + w / 2) / orig_w, 0.0, 1.0)
            cy = np.clip((y + h / 2) / orig_h, 0.0, 1.0)
            nw = np.clip(w / orig_w, 0.0, 1.0)
            nh = np.clip(h / orig_h, 0.0, 1.0)
            if nw > 0 and nh > 0:
                boxes.append([cx, cy, nw, nh])
                labels.append(self.cat_to_id[ann["category_id"]])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)

        return {
            "image": image,
            "labels": labels_t,
            "boxes": boxes_t,
            "image_id": img_id,
            "orig_size": (orig_h, orig_w),
        }


def collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "class_labels": [item["labels"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "orig_sizes": [item["orig_size"] for item in batch],
    }


def train_epoch(model, processor, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    valid_steps = 0
    nan_count = 0

    pbar = tqdm(dataloader, desc="Training", leave=True)
    for batch in pbar:
        encoding = processor(images=batch["images"], return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        # HuggingFace DETR computes Hungarian-matching loss internally when labels are provided
        labels = [
            {"class_labels": cl.to(device), "boxes": bx.to(device)}
            for cl, bx in zip(batch["class_labels"], batch["boxes"])
        ]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            # DETR training is sensitive to large gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            valid_steps += 1

        avg = total_loss / max(1, valid_steps)
        pbar.set_postfix({"loss": f"{avg:.4f}", "nan": nan_count})

    if nan_count > 0:
        print(f"[WARN] {nan_count} NaN batches skipped this epoch")
    return total_loss / max(1, valid_steps)


@torch.no_grad()
def evaluate(model, processor, dataloader, val_dataset, device):
    model.eval()
    predictions = []

    pbar = tqdm(dataloader, desc="Validation", leave=True)
    for batch in pbar:
        encoding = processor(images=batch["images"], return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_sizes = torch.tensor(batch["orig_sizes"], device=device)  # [B, 2] (H, W)
        results = processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=orig_sizes
        )

        gt_cat_id = val_dataset.annotations_data["categories"][0]["id"]
        for img_id, result in zip(batch["image_ids"], results):
            scores = result["scores"].cpu()
            boxes = result["boxes"].cpu()  # [x1, y1, x2, y2] pixel coords

            for score, box in zip(scores, boxes):
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    predictions.append({
                        "image_id": int(img_id),
                        "category_id": gt_cat_id,
                        "bbox": [x1, y1, w, h],
                        "score": float(score),
                    })

    if not predictions:
        print("[WARN] No predictions above threshold — mAP=0.0")
        return 0.0

    # Build COCO GT object from val split
    gt_data = {
        "images": [{"id": img_id} for img_id in val_dataset.image_ids],
        "categories": val_dataset.annotations_data["categories"],
        "annotations": [],
    }
    ann_id = 1
    for img_id in val_dataset.image_ids:
        for ann in val_dataset.img_to_anns.get(img_id, []):
            gt_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["bbox"][2] * ann["bbox"][3],
                "iscrowd": ann.get("iscrowd", 0),
            })
            ann_id += 1

    try:
        coco_gt = COCO()
        coco_gt.dataset = gt_data
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return float(coco_eval.stats[0])
    except Exception as e:
        print(f"[WARN] Could not calculate mAP: {e}")
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DETR for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--annotation-path", type=str, default="data/annotations/instances_train.json")
    parser.add_argument("--train-images-dir", type=str, default="data/images/train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[DETR] Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    annotation_path = Path(args.annotation_path)
    if not annotation_path.is_absolute():
        annotation_path = project_root / annotation_path
    train_images_dir = Path(args.train_images_dir)
    if not train_images_dir.is_absolute():
        train_images_dir = project_root / train_images_dir

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Train images dir not found: {train_images_dir}")

    print(f"[DETR] Annotations: {annotation_path}")
    print(f"[DETR] Images: {train_images_dir}")
    print("[DETR] Loading processor and model from facebook/detr-resnet-50...")

    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    # num_labels=1: one object class; HuggingFace DETR adds no-object class automatically
    model = AutoModelForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", num_labels=1, ignore_mismatched_sizes=True
    )
    model = model.to(device)
    print("[DETR] Model loaded!")

    train_dataset = COCODetectionDataset(
        annotation_path=annotation_path,
        image_dir=train_images_dir,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    val_dataset = COCODetectionDataset(
        annotation_path=annotation_path,
        image_dir=train_images_dir,
        split="val",
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Differential LR: lower for pretrained backbone, higher for new head
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    other_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    num_warmup_steps = len(train_loader) * args.warmup_epochs
    total_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_mAP = 0.0

    for epoch in range(args.epochs):
        print(f"\n[DETR] Epoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, processor, train_loader, optimizer, scheduler, device)
        mAP = evaluate(model, processor, val_loader, val_dataset, device)

        print(f"[DETR] Train loss: {train_loss:.4f} | mAP50:95: {mAP:.4f}")

        if mAP > best_mAP:
            best_mAP = mAP
            save_path = models_dir / "detr_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"[DETR] Saved best model (mAP50:95={mAP:.4f}) -> {save_path}")

    print(f"\n[DETR] Training complete! Best mAP50:95: {best_mAP:.4f}")


if __name__ == "__main__":
    main()
