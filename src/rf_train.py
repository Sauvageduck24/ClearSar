from __future__ import annotations

"""
RF-DETR training module for ClearSAR.

Uses the rfdetr library (Roboflow) with COCO-format annotations directly.
Supported model variants: s, m, l, x, xl, xxl (via --model rf-detr-s, etc.)

Usage:
    python -m src.rf_train --project-root . --epochs 100 --model rf-detr-l
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2

from src.config import default_config
from src.dataset import split_train_val_image_ids
from src.utils.repro import set_seed


# ---------------------------------------------------------------------------
# SAR-specific augmentation helpers (shared logic with yolo_train)
# ---------------------------------------------------------------------------

def apply_sar_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE selectivo en canal VV (R) del quicklook Sentinel-1."""
    result = image.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    try:
        result[:, :, 0] = clahe.apply(result[:, :, 0])
    except Exception:
        return image
    return result


def _build_albumentations_pipeline():
    """Pipeline Albumentations SAR-específico reutilizable."""
    import albumentations as A
    return A.Compose([
        A.Lambda(image=apply_sar_clahe, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.3, 0.3),
            p=0.4,
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.GaussNoise(var_limit=(5.0, 30.0), mean=0, p=0.35),
        A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.3),
        A.OneOf([
            A.Blur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.15),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], bbox_params=A.BboxParams(
        format="coco",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ))


# ---------------------------------------------------------------------------
# COCO dataset split helpers
# ---------------------------------------------------------------------------

def _build_coco_splits(
    annotation_path: Path,
    train_images_dir: Path,
    output_dir: Path,
    val_fraction: float,
    seed: int,
    extra_images_dir: Optional[Path] = None,
    extra_annotations_path: Optional[Path] = None,
) -> tuple[Path, Path, Path, Path]:
    """
    Split the COCO annotation file into train/val JSON files and create
    images symlink directories expected by rfdetr.

    Returns (train_ann_path, val_ann_path, train_img_dir, val_img_dir).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    train_ids, val_ids = split_train_val_image_ids(
        annotation_path=annotation_path,
        val_fraction=val_fraction,
        seed=seed,
    )

    # --- Optionally merge extra pseudo-labeled data into train ---
    if extra_images_dir and extra_annotations_path and extra_annotations_path.exists():
        with extra_annotations_path.open("r", encoding="utf-8") as f:
            extra_coco = json.load(f)
        id_offset = max(img["id"] for img in coco["images"]) + 1
        ann_offset = max(a["id"] for a in coco["annotations"]) + 1 if coco["annotations"] else 1
        extra_imgs_patched = []
        id_remap: Dict[int, int] = {}
        for img in extra_coco["images"]:
            new_id = img["id"] + id_offset
            id_remap[img["id"]] = new_id
            extra_imgs_patched.append({**img, "id": new_id})
        extra_anns_patched = [
            {**a, "id": a["id"] + ann_offset, "image_id": id_remap[a["image_id"]]}
            for a in extra_coco["annotations"]
        ]
        coco["images"].extend(extra_imgs_patched)
        coco["annotations"].extend(extra_anns_patched)
        for img in extra_imgs_patched:
            train_ids.append(img["id"])
        print(f"[rf_detr] Extra pseudo dataset: {len(extra_imgs_patched)} images added to train")

    images_meta: Dict[int, Dict] = {img["id"]: img for img in coco["images"]}

    def _subset(ids: List[int]) -> dict:
        img_set = {i for i in ids}
        imgs = [images_meta[i] for i in ids if i in images_meta]
        anns = [a for a in coco["annotations"] if a["image_id"] in img_set]
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco.get("categories", [{"id": 0, "name": "RFI"}]),
            "images": imgs,
            "annotations": anns,
        }

    train_ann = _subset(train_ids)
    val_ann = _subset(val_ids)

    train_ann_path = output_dir / "train_annotations.json"
    val_ann_path = output_dir / "val_annotations.json"
    train_ann_path.write_text(json.dumps(train_ann, ensure_ascii=False, indent=2))
    val_ann_path.write_text(json.dumps(val_ann, ensure_ascii=False, indent=2))

    # Symlink image directories
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    def _link(ids: List[int], dest: Path, src_dir: Path) -> None:
        for img_id in ids:
            if img_id not in images_meta:
                continue
            fname = images_meta[img_id]["file_name"]
            src = src_dir / fname
            dst = dest / fname
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)

    _link(train_ids, train_img_dir, train_images_dir)
    _link(val_ids, val_img_dir, train_images_dir)

    print(f"[rf_detr] Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    return train_ann_path, val_ann_path, train_img_dir, val_img_dir


# ---------------------------------------------------------------------------
# Model variant mapping
# ---------------------------------------------------------------------------

_RFDETR_VARIANTS = {
    "rf-detr-s":   "RFDETRSmall",
    "rf-detr-m":   "RFDETRMedium",
    "rf-detr-l":   "RFDETRLarge",
    "rf-detr-x":   "RFDETRX",
    "rf-detr-xl":  "RFDETRXL",
    "rf-detr-xxl": "RFDETRXXL",
    # short aliases
    "s":   "RFDETRSmall",
    "m":   "RFDETRMedium",
    "l":   "RFDETRLarge",
    "x":   "RFDETRX",
    "xl":  "RFDETRXL",
    "xxl": "RFDETRXXL",
}


def _resolve_rfdetr_class(model_key: str):
    """Import and return the rfdetr model class for the requested variant."""
    try:
        import rfdetr  # noqa: F401
    except ImportError:
        raise ImportError(
            "rfdetr is required for training.\n"
            "Install with: pip install rfdetr"
        )

    class_name = _RFDETR_VARIANTS.get(model_key.lower())
    if class_name is None:
        raise ValueError(
            f"Unknown RF-DETR variant '{model_key}'. "
            f"Choose one of: {list(_RFDETR_VARIANTS.keys())}"
        )

    import importlib
    mod = importlib.import_module("rfdetr")
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF-DETR detector for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--model", type=str, default="rf-detr-l",
        help="Model variant: rf-detr-s/m/l/x/xl/xxl  (or just s/m/l/x/xl/xxl)"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--extra-images-dir", type=str, default=None)
    parser.add_argument("--extra-annotations-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = default_config(project_root=args.project_root)
    project_root = cfg.paths.project_root
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.outputs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    extra_images_dir = Path(args.extra_images_dir) if args.extra_images_dir else None
    extra_annotations_path = (
        Path(args.extra_annotations_path) if args.extra_annotations_path else None
    )

    # ── Build dataset ──────────────────────────────────────────────────────
    rfdetr_dir = project_root / "data" / "rfdetr"
    train_ann_path, val_ann_path, train_img_dir, val_img_dir = _build_coco_splits(
        annotation_path=cfg.paths.train_annotations_path,
        train_images_dir=cfg.paths.train_images_dir,
        output_dir=rfdetr_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        extra_images_dir=extra_images_dir,
        extra_annotations_path=extra_annotations_path,
    )

    # ── Resolve device ─────────────────────────────────────────────────────
    import torch
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Resolve model class and instantiate ────────────────────────────────
    ModelClass = _resolve_rfdetr_class(args.model)
    run_name = f"clearsar_{args.model.lower().replace(' ', '_')}"
    output_dir = project_root / "outputs" / "rfdetr_runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rf_detr] Instanciando {ModelClass.__name__} (device={device})")

    model_kwargs: dict = dict(
        num_classes=1,
        pretrain_weights=None if args.resume else "auto",  # descarga pesos COCO si no se resume
    )
    if args.resume:
        model_kwargs["pretrain_weights"] = args.resume
        print(f"[rf_detr] Resuming from {args.resume}")

    model = ModelClass(**model_kwargs)

    # ── Train ──────────────────────────────────────────────────────────────
    train_kwargs = dict(
        dataset_dir=str(rfdetr_dir),
        train_annotations=str(train_ann_path),
        val_annotations=str(val_ann_path),
        train_images=str(train_img_dir),
        val_images=str(val_img_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.image_size,
        workers=args.num_workers,
        lr=args.lr,
        output_dir=str(output_dir),
        device=device,
        # ── Augmentaciones geométricas conservadoras (idéntica razón que YOLO)
        # RF-DETR hereda parámetros de augmentación en su API.
        # Los flips horizontales/verticales están habilitados en el pipeline
        # Albumentations de arriba (para transforms fuera de la librería).
        # Dentro de la API nativa se deshabilita rotación:
        augment=True,
    )

    print(f"[rf_detr] Iniciando entrenamiento: {run_name}")
    model.train(**train_kwargs)

    # ── Save best checkpoint ───────────────────────────────────────────────
    # rfdetr guarda checkpoints en output_dir; buscamos el best
    best_candidates = sorted(output_dir.glob("*best*.pth")) + sorted(output_dir.glob("*best*.pt"))
    if best_candidates:
        best_ckpt = best_candidates[-1]
        dest = cfg.paths.models_dir / f"rfdetr_best_{args.model.lower().replace(' ', '_')}.pth"
        shutil.copy2(best_ckpt, dest)
        print(f"[rf_detr] Best checkpoint copiado a {dest}")
    else:
        print(f"[rf_detr] Entrenamiento completado. Checkpoints en {output_dir}")


if __name__ == "__main__":
    main()
