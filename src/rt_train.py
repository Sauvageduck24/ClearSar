from __future__ import annotations

"""
RT-DETR training module for ClearSAR.

Uses the Ultralytics RT-DETR implementation (same API as YOLO but with
transformer-based architecture). Converts COCO annotations to YOLO format
automatically, exactly as yolo_train.py does.

Supported model variants: rtdetr-s, rtdetr-m, rtdetr-l, rtdetr-x
(or short aliases: s, m, l, x)

Usage:
    python -m src.rt_train --project-root . --epochs 100 --model rtdetr-l
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np
import cv2
import torch

from src.config import default_config
from src.dataset import split_train_val_image_ids
from src.utils.repro import set_seed


# ---------------------------------------------------------------------------
# SAR-specific augmentation helpers (mirrored from yolo_train)
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


def build_sar_augment_callback():
    """
    Callback SAR-específico para RT-DETR entrenado con Ultralytics.
    Idéntico al de yolo_train: se engancha en on_train_batch_start.
    """
    import albumentations as A

    pipeline = A.Compose([
        # CLAHE es vital en SAR, manténlo pero quizás con clipLimit más bajo
        A.Lambda(image=apply_sar_clahe, p=0.4), 
        
        # Menos agresivo con el brillo
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), 
            contrast_limit=(-0.2, 0.2), 
            p=0.3
        ),
        
        # El ruido Gaussiano es excelente para SAR (simula speckle)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # ¡Importante! El Sharpen ayuda a resaltar esas rayas de 1px
        A.Sharpen(alpha=(0.2, 0.5), p=0.4),
        
        # Elimina los Blurs o ponlos testimoniales
        # A.Blur(p=0.05), 
    ])

    def on_train_batch_start(trainer, *cb_args, **cb_kwargs):
        imgs = None
        batch_container = None

        if hasattr(trainer, "batch") and isinstance(trainer.batch, dict) and "img" in trainer.batch:
            imgs = trainer.batch["img"]
            batch_container = ("trainer", None)
        else:
            if len(cb_args) >= 1:
                batch = cb_args[0]
                if isinstance(batch, dict) and "img" in batch:
                    imgs = batch["img"]
                    batch_container = ("arg", 0)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    imgs = batch[0]
                    batch_container = ("arg", 0)
            if imgs is None and "batch" in cb_kwargs:
                batch = cb_kwargs.get("batch")
                if isinstance(batch, dict) and "img" in batch:
                    imgs = batch["img"]
                    batch_container = ("kw", "batch")
                elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    imgs = batch[0]
                    batch_container = ("kw", "batch")

        if imgs is None:
            return

        device = imgs.device if hasattr(imgs, "device") else torch.device("cpu")
        imgs_np = (imgs.cpu().numpy() * 255).astype(np.uint8)

        augmented = []
        for img in imgs_np:
            img_hwc = img.transpose(1, 2, 0)
            result = pipeline(image=img_hwc)["image"]
            augmented.append(result.transpose(2, 0, 1))

        augmented_tensor = (
            torch.from_numpy(np.stack(augmented).astype(np.float32) / 255.0).to(device)
        )

        if batch_container is None:
            return
        where, key = batch_container
        if where == "trainer":
            trainer.batch["img"] = augmented_tensor
        elif where == "arg":
            try:
                cb_args[key][0] = augmented_tensor
            except Exception:
                try:
                    mutable = list(cb_args)
                    if isinstance(mutable[key], (list, tuple)):
                        inner = list(mutable[key])
                        inner[0] = augmented_tensor
                        mutable[key] = type(cb_args[key])(inner)
                except Exception:
                    pass
        elif where == "kw":
            try:
                cb_kwargs[key] = augmented_tensor
            except Exception:
                pass

    return on_train_batch_start


# ---------------------------------------------------------------------------
# COCO → YOLO conversion (identical logic to yolo_train)
# ---------------------------------------------------------------------------

def _convert_coco_to_yolo(
    annotation_path: Path,
    images_dir: Path,
    output_labels_dir: Path,
    image_ids: Optional[List[int]] = None,
) -> None:
    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    output_labels_dir.mkdir(parents=True, exist_ok=True)

    images_meta: Dict[int, Dict] = {img["id"]: img for img in coco["images"]}
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in coco["annotations"]:
        img_id = int(ann["image_id"])
        anns_by_image.setdefault(img_id, []).append(ann)

    ids_to_process = image_ids if image_ids is not None else list(images_meta.keys())

    for img_id in ids_to_process:
        meta = images_meta[img_id]
        img_w = float(meta["width"])
        img_h = float(meta["height"])
        stem = Path(meta["file_name"]).stem
        label_path = output_labels_dir / f"{stem}.txt"

        anns = anns_by_image.get(img_id, [])
        lines: List[str] = []
        for ann in anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            cx = max(0.0, min(1.0, (x + w / 2) / img_w))
            cy = max(0.0, min(1.0, (y + h / 2) / img_h))
            nw = max(0.0, min(1.0, w / img_w))
            nh = max(0.0, min(1.0, h / img_h))
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        label_path.write_text("\n".join(lines), encoding="utf-8")


def _build_rtdetr_dataset(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    val_fraction: float,
    seed: int,
    extra_images_dir: Optional[Path] = None,
    extra_annotations_path: Optional[Path] = None,
) -> Path:
    """
    Construye la estructura de dataset compatible con Ultralytics RT-DETR
    (mismo formato que YOLO) y devuelve la ruta al yaml.

    data/rtdetr/
        images/train/
        images/val/
        labels/train/
        labels/val/
        clearsar.yaml
    """
    rtdetr_dir = project_root / "data" / "rtdetr"
    img_train = rtdetr_dir / "images" / "train"
    img_val = rtdetr_dir / "images" / "val"
    lbl_train = rtdetr_dir / "labels" / "train"
    lbl_val = rtdetr_dir / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = split_train_val_image_ids(
        annotation_path=annotation_path,
        val_fraction=val_fraction,
        seed=seed,
    )

    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images_meta: Dict[int, Dict] = {img["id"]: img for img in coco["images"]}

    def _link_images(ids: List[int], dest_dir: Path) -> None:
        for img_id in ids:
            fname = images_meta[img_id]["file_name"]
            src = train_images_dir / fname
            dst = dest_dir / fname
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)

    _link_images(train_ids, img_train)
    _link_images(val_ids, img_val)

    _convert_coco_to_yolo(annotation_path, train_images_dir, lbl_train, train_ids)
    _convert_coco_to_yolo(annotation_path, train_images_dir, lbl_val, val_ids)

    if extra_images_dir and extra_annotations_path and extra_annotations_path.exists():
        with extra_annotations_path.open("r", encoding="utf-8") as f:
            extra_coco = json.load(f)
        extra_ids = [img["id"] for img in extra_coco["images"]]
        extra_meta: Dict[int, Dict] = {img["id"]: img for img in extra_coco["images"]}
        for img_id in extra_ids:
            fname = extra_meta[img_id]["file_name"]
            src = extra_images_dir / fname
            dst = img_train / fname
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)
        _convert_coco_to_yolo(extra_annotations_path, extra_images_dir, lbl_train, extra_ids)
        print(f"[rt_detr] Extra pseudo dataset: {len(extra_ids)} images añadidas al train")

    yaml_path = rtdetr_dir / "clearsar.yaml"
    dataset_cfg = {
        "path": str(rtdetr_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["RFI"],
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    print(f"[rt_detr] Dataset yaml: {yaml_path}")
    print(f"[rt_detr] Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    return yaml_path


# ---------------------------------------------------------------------------
# Model variant mapping
# ---------------------------------------------------------------------------

_RTDETR_VARIANTS = {
    "rtdetr-s": "rtdetr-s.yaml",
    "rtdetr-m": "rtdetr-m.yaml",    # community / unofficial
    "rtdetr-l": "rtdetr-l.yaml",
    "rtdetr-x": "rtdetr-x.yaml",
    # short aliases
    "s": "rtdetr-s.yaml",
    "m": "rtdetr-m.yaml",
    "l": "rtdetr-l.yaml",
    "x": "rtdetr-x.yaml",
}


def _resolve_model_name(model_key: str, project_root: Path, models_dir: Path) -> str:
    """
    Devuelve el identificador de modelo para Ultralytics RTDETR.
    Si se pasa una ruta .pt/.pth existente se usa directamente.
    """
    path = Path(model_key)
    if path.suffix in (".pt", ".pth"):
        return str(model_key)

    candidates = [
        project_root / model_key,
        project_root / f"{model_key}.pt",
        models_dir / f"{model_key}.pt",
        Path(f"{model_key}.pt"),
    ]
    found = next((c for c in candidates if c.exists()), None)
    if found:
        return str(found)

    # Intentar con el yaml de ultralytics
    variant_yaml = _RTDETR_VARIANTS.get(model_key.lower())
    if variant_yaml:
        return variant_yaml  # Ultralytics descarga pesos automáticamente

    return model_key + ".pt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR detector for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--model", type=str, default="rtdetr-l",
        help="Model variant: rtdetr-s/m/l/x  (or short: s/m/l/x)",
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from ultralytics import RTDETR
    except ImportError:
        raise ImportError(
            "ultralytics>=8.1 is required for RT-DETR training.\n"
            "Install with: pip install ultralytics"
        )

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

    yaml_path = _build_rtdetr_dataset(
        project_root=project_root,
        annotation_path=cfg.paths.train_annotations_path,
        train_images_dir=cfg.paths.train_images_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        extra_images_dir=extra_images_dir,
        extra_annotations_path=extra_annotations_path,
    )

    model_name = _resolve_model_name(args.model, project_root, cfg.paths.models_dir)
    print(f"[rt_detr] Cargando RT-DETR: {model_name}")
    model = RTDETR(model_name)

    run_name = f"clearsar_{args.model.lower().replace(' ', '_').replace('.pt', '')}"
    runs_dir = project_root / "outputs" / "rtdetr_runs"

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.num_workers,
        lr0=args.lr,
        seed=args.seed,
        patience=args.patience,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
        # ── Geométricas (misma razón que YOLO: RFI son rayas horizontales) ──
        degrees=0.0,
        scale=0.3,
        translate=0.05,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.5,
        # ── Color / intensidad ──────────────────────────────────────────────
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.5,
        # ── Mosaic ──────────────────────────────────────────────────────────
        # RT-DETR puede usar mosaic, aunque suele entrenarse sin él por
        # defecto. Lo habilitamos a 0.5 como compromiso.
        mosaic=0.5,
        close_mosaic=10,
        copy_paste=0.4,
        mixup=0.0,
        # ── Optimizador ─────────────────────────────────────────────────────
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,
        # ── Loss ────────────────────────────────────────────────────────────
        box=7.5,
        cls=0.5,
        # ── Misc ────────────────────────────────────────────────────────────
        amp=True,
        verbose=True,
        erasing=0.0,
        augment=True,
    )

    if args.device is not None:
        train_kwargs["device"] = args.device

    if args.resume:
        last_ckpt = runs_dir / run_name / "weights" / "last.pt"
        if last_ckpt.exists():
            model = RTDETR(str(last_ckpt))
            train_kwargs["resume"] = True
            print(f"[rt_detr] Resumiendo desde {last_ckpt}")
        else:
            print(f"[rt_detr] No se encontró checkpoint en {last_ckpt}, empezando desde cero")

    # Registrar callback SAR antes de entrenar
    callback_fn = build_sar_augment_callback()
    model.add_callback("on_train_batch_start", callback_fn)

    results = model.train(**train_kwargs)

    best_ckpt = runs_dir / run_name / "weights" / "best.pt"
    if best_ckpt.exists():
        dest = cfg.paths.models_dir / f"rtdetr_best_{args.model.lower().replace(' ', '_').replace('.pt', '')}.pt"
        shutil.copy2(best_ckpt, dest)
        print(f"[rt_detr] Best checkpoint copiado a {dest}")

    print(f"[rt_detr] Entrenamiento completo. Resultados: {results}")


if __name__ == "__main__":
    main()
