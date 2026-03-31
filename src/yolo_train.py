from __future__ import annotations

"""
YOLO training module for ClearSAR.

Converts the existing COCO annotations to YOLO format automatically,
then trains with Ultralytics. The trained model is saved as a standard
.pt file compatible with yolo_inference.py and the ensemble.

Usage:
    python -m src.yolo_train --project-root . --epochs 100 --model yolo11m
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import albumentations as A
from src.config import default_config
from src.dataset import split_train_val_image_ids
from src.utils.repro import set_seed
import torch
import numpy as np
import cv2


def apply_sar_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE selectivo en canal VV (R) del quicklook Sentinel-1.

    Aplica CLAHE solo al canal R (índice 0) que contiene la firma RFI.
    Espera `image` en formato HWC uint8.
    """
    result = image.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # Solo canal R = VV donde el RFI tiene mayor firma
    try:
        result[:, :, 0] = clahe.apply(result[:, :, 0])
    except Exception:
        # En caso de que la imagen no tenga 3 canales, devolver original
        return image
    return result

def build_sar_augment_callback():
    """
    Callback que inyecta augmentaciones SAR-específicas via Albumentations
    encima de las augmentaciones geométricas de YOLO.
    
    Se engancha en on_train_batch_start, donde trainer.batch['img']
    ya es un tensor (B, 3, H, W) float32 normalizado [0,1] en GPU/CPU.
    """
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
    # Nota: sin bbox_params porque las cajas ya están gestionadas por YOLO
    # Solo tocamos los píxeles, no la geometría

    def on_train_batch_start(trainer, *cb_args, **cb_kwargs):
        # Extraer imgs soportando distintas firmas de callback (trainer.batch, batch arg, kwargs)
        imgs = None
        batch_container = None

        if hasattr(trainer, "batch") and isinstance(trainer.batch, dict) and "img" in trainer.batch:
            imgs = trainer.batch["img"]
            batch_container = ("trainer", None)
        else:
            # Positional batch (común: (imgs, targets, paths, shapes))
            if len(cb_args) >= 1:
                batch = cb_args[0]
                if isinstance(batch, dict) and "img" in batch:
                    imgs = batch["img"]
                    batch_container = ("arg", 0)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    imgs = batch[0]
                    batch_container = ("arg", 0)

            # Keyword 'batch'
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

        augmented_tensor = torch.from_numpy(np.stack(augmented).astype(np.float32) / 255.0).to(device)

        # Escribir de vuelta en el contenedor original si es mutable
        if batch_container is None:
            return
        where, key = batch_container
        if where == "trainer":
            trainer.batch["img"] = augmented_tensor
        elif where == "arg":
            try:
                cb_args[key][0] = augmented_tensor
            except Exception:
                # si no mutable, intentar si es lista dentro de args
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


def _convert_coco_to_yolo(
    annotation_path: Path,
    images_dir: Path,
    output_labels_dir: Path,
    image_ids: Optional[List[int]] = None,
) -> None:
    """
    Convert COCO annotations to YOLO txt format.

    Each image gets a .txt with one line per box: class cx cy w h (normalized).
    Images with no annotations get an empty .txt file (YOLO needs this).
    """
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
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines), encoding="utf-8")


def _build_yolo_dataset(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    val_fraction: float,
    seed: int,
    extra_images_dir: Optional[Path] = None,
    extra_annotations_path: Optional[Path] = None,
) -> Path:
    """
    Build the YOLO dataset folder structure and return the path to the yaml file.

    Structure created under project_root/data/yolo/:
        images/train/  -> symlinks or copies to original train images
        images/val/    -> symlinks or copies to original val images
        labels/train/  -> converted YOLO txt labels
        labels/val/    -> converted YOLO txt labels
        clearsar.yaml  -> dataset config for Ultralytics
    """
    yolo_dir = project_root / "data" / "yolo"
    yolo_images_train = yolo_dir / "images" / "train"
    yolo_images_val = yolo_dir / "images" / "val"
    yolo_labels_train = yolo_dir / "labels" / "train"
    yolo_labels_val = yolo_dir / "labels" / "val"

    for d in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
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

    _link_images(train_ids, yolo_images_train)
    _link_images(val_ids, yolo_images_val)

    _convert_coco_to_yolo(annotation_path, train_images_dir, yolo_labels_train, train_ids)
    _convert_coco_to_yolo(annotation_path, train_images_dir, yolo_labels_val, val_ids)

    if extra_images_dir and extra_annotations_path and extra_annotations_path.exists():
        with extra_annotations_path.open("r", encoding="utf-8") as f:
            extra_coco = json.load(f)
        extra_ids = [img["id"] for img in extra_coco["images"]]
        extra_meta: Dict[int, Dict] = {img["id"]: img for img in extra_coco["images"]}
        for img_id in extra_ids:
            fname = extra_meta[img_id]["file_name"]
            src = extra_images_dir / fname
            dst = yolo_images_train / fname
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)
        _convert_coco_to_yolo(extra_annotations_path, extra_images_dir, yolo_labels_train, extra_ids)
        print(f"[yolo] Extra pseudo dataset: {len(extra_ids)} images added to train")

    yaml_path = yolo_dir / "clearsar.yaml"
    dataset_cfg = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["RFI"],
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    print(f"[yolo] Dataset yaml: {yaml_path}")
    print(f"[yolo] Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--model", type=str, default="yolo11m",
                        help="Model variant: yolo11n/s/m/l/x, yolov8m, etc.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--extra-images-dir", type=str, default=None)
    parser.add_argument("--extra-annotations-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for training.\n"
            "Install with: pip install ultralytics"
        )

    args = parse_args()
    cfg = default_config(project_root=args.project_root)
    project_root = cfg.paths.project_root
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.outputs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    extra_images_dir = Path(args.extra_images_dir) if args.extra_images_dir else None
    extra_annotations_path = Path(args.extra_annotations_path) if args.extra_annotations_path else None

    yaml_path = _build_yolo_dataset(
        project_root=project_root,
        annotation_path=cfg.paths.train_annotations_path,
        train_images_dir=cfg.paths.train_images_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        extra_images_dir=extra_images_dir,
        extra_annotations_path=extra_annotations_path,
    )

    # Resolve model path
    model_arg = args.model

    model_arg_path = Path(model_arg)
    if model_arg_path.suffix in (".pt", ".pth"):
        model_name = str(model_arg)
    else:
        candidates = [
            project_root / model_arg,
            project_root / f"{model_arg}.pt",
            cfg.paths.models_dir / f"{model_arg}.pt",
            Path(f"{model_arg}.pt"),
        ]
        found = next((c for c in candidates if c.exists()), None)
        model_name = str(found) if found else model_arg + ".pt"

    print(f"[yolo] Cargando YOLO: {model_name}")
    model = YOLO(model_name)

    run_name = f"clearsar_{args.model.replace('.pt', '')}"
    yolo_runs_dir = project_root / "outputs" / "yolo_runs"

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.num_workers,
        lr0=args.lr,
        seed=args.seed,
        patience=args.patience,
        project=str(yolo_runs_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        val=True,

        # ── Geométricas ────────────────────────────────────────────────────
        # degrees=0: el RFI son rayas físicamente horizontales en SAR.
        # Rotar 90° generaría rayas verticales que no existen en el test set
        # → el modelo aprendería patrones falsos. Mantenemos solo flips.
        degrees=0.0,
        #rect=True,  # entrenamiento rectangular (no square) para aprovechar mejor la resolución de las rayas finas

        # scale=0.3: la mediana de alto de caja es 10px. Con scale=0.5 (el
        # valor anterior) el zoom-out llevaría esa mediana a 5px → casi
        # invisible en el feature map. 0.3 reduce al 70% como mínimo → 7px.
        scale=0.3,

        # translate=0.05: las rayas ocupan casi todo el ancho de la imagen
        # (~27% del ancho normalizado). Traslaciones grandes las sacan del
        # borde → cajas eliminadas por min_visibility. 5% es suficiente.
        translate=0.05,

        # shear=0.0: shear inclina las rayas horizontales. El RFI real no
        # aparece inclinado → entrenar con shear introduciría ruido.
        shear=0.0,

        # perspective=0.0: mismo razonamiento que shear.
        perspective=0.0,

        # rect=True: entrenamiento rectangular. Las imágenes miden ~515x342
        # (aspecto 1.5:1). Con rect=True YOLO usa letterbox mínimo
        # (640x426 en lugar de 640x640) → menos padding → más resolución
        # efectiva → mejor detección de rayas finas de 1-3px de alto.

        # fliplr/flipud: el RFI puede aparecer en cualquier posición y
        # orientación espejo → ambos flips son válidos físicamente.
        fliplr=0.5,
        flipud=0.5,

        # ── Color / intensidad ─────────────────────────────────────────────
        # Los quicklooks SAR codifican polarizaciones en canales RGB
        # (típicamente VV, VH, ratio). hsv_h muy bajo para no alterar la
        # firma radiométrica. hsv_v alto porque el contraste del RFI vs fondo
        # es la clave para detectarlo.
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,

        # ── Mosaic / copy-paste ────────────────────────────────────────────
        # mosaic=1.0: combina 4 imágenes → el modelo ve RFI en muchos
        # contextos distintos. Muy útil con pocos datos (3154 imágenes).
        mosaic=1.0,

        # close_mosaic=10: desactiva mosaic en las últimas 10 épocas para
        # que el modelo se estabilice con imágenes realistas antes de
        # la evaluación final. Mejora el mAP en val.
        close_mosaic=15,

        # copy_paste=0.3: copia objetos RFI de una imagen a otra. Especialmente
        # útil aquí porque el 48.8% de las cajas son "small" (<1024px²) y hay
        # pocas por imagen (mediana=2). Multiplica ejemplos de RFI sin anotar más.
        copy_paste=0.4,

        # mixup=0.0: mixup en detección superpone dos imágenes y mezcla sus
        # bounding boxes. Con rayas finas esto genera targets ambiguos y
        # degrada el entrenamiento. Mejor desactivarlo.
        mixup=0.0,

        # ── Optimizador y scheduler ────────────────────────────────────────
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,

        # ── Loss weights ───────────────────────────────────────────────────
        # box=7.5: peso del box regression loss. Default de YOLO, bien
        # calibrado para objetos pequeños. No tocar.
        box=7.5,

        # cls=0.5: solo hay una clase (RFI) → el classification loss tiene
        # menos importancia que el box loss.
        cls=0.5,

        # ── Misc ───────────────────────────────────────────────────────────
        amp=True,
        verbose=True,
        erasing=0.0,
        augment=True,
    )

    if args.device is not None:
        train_kwargs["device"] = args.device

    if args.resume:
        last_ckpt = yolo_runs_dir / run_name / "weights" / "last.pt"
        if last_ckpt.exists():
            model = YOLO(str(last_ckpt))
            train_kwargs["resume"] = True
            print(f"[yolo] Resuming from {last_ckpt}")
        else:
            print(f"[yolo] No checkpoint found at {last_ckpt}, starting fresh")

    # Registrar callback antes de llamar a train
    callback_fn = build_sar_augment_callback()
    model.add_callback("on_train_batch_start", callback_fn)

    results = model.train(**train_kwargs)

    best_ckpt = yolo_runs_dir / run_name / "weights" / "best.pt"
    if best_ckpt.exists():
        dest = cfg.paths.models_dir / f"yolo_best_{args.model.replace('.pt', '')}.pt"
        shutil.copy2(best_ckpt, dest)
        print(f"[yolo] Best checkpoint copied to {dest}")

    print(f"[yolo] Training complete. Results: {results}")


if __name__ == "__main__":
    main()