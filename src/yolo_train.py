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
from src.config import default_config
from src.dataset import split_train_val_image_ids
from src.utils.repro import set_seed

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
            if w < 2 or h < 2:
                continue
            if w * h < 4:
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
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--extra-images-dir", type=str, default=None)
    parser.add_argument("--extra-annotations-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--evolve",
        type=int,
        nargs="?",
        const=100,
        default=0,
        help="Enable YOLO hyperparameter evolution. Optional value is generations (default: 100).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--cache",
        type=str,
        default="disk",
        choices=["disk", "ram", "none"],
        help="Cache mode for Ultralytics dataloader: disk, ram or none.",
    )
    return parser.parse_args()


def _extract_detailed_map_metrics(metrics: Dict) -> Dict[str, float]:
    """
    Normalize metric keys and extract AP split by object size when available.
    """
    normalized: Dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        nk = str(key).lower().replace(" ", "").replace("_", "").replace("-", "")
        normalized[nk] = float(value)

    def _pick(candidates: List[str]) -> Optional[float]:
        for cand in candidates:
            if cand in normalized:
                return normalized[cand]
        return None

    out: Dict[str, float] = {}

    map5095_s = _pick([
        "metrics/map5095small(b)",
        "metrics/map5095s(b)",
        "boxmap5095s",
        "map5095s",
        "aps",
    ])
    map5095_m = _pick([
        "metrics/map5095medium(b)",
        "metrics/map5095m(b)",
        "boxmap5095m",
        "map5095m",
        "apm",
    ])
    map5095_l = _pick([
        "metrics/map5095large(b)",
        "metrics/map5095l(b)",
        "boxmap5095l",
        "map5095l",
        "apl",
    ])

    map50_s = _pick([
        "metrics/map50small(b)",
        "metrics/map50s(b)",
        "boxmap50s",
        "map50s",
    ])
    map50_m = _pick([
        "metrics/map50medium(b)",
        "metrics/map50m(b)",
        "boxmap50m",
        "map50m",
    ])
    map50_l = _pick([
        "metrics/map50large(b)",
        "metrics/map50l(b)",
        "boxmap50l",
        "map50l",
    ])

    if map5095_s is not None:
        out["mAP_s"] = map5095_s
    if map5095_m is not None:
        out["mAP_m"] = map5095_m
    if map5095_l is not None:
        out["mAP_l"] = map5095_l
    if map50_s is not None:
        out["mAP50_s"] = map50_s
    if map50_m is not None:
        out["mAP50_m"] = map50_m
    if map50_l is not None:
        out["mAP50_l"] = map50_l

    return out


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
        # Para objetos pequeños conviene entrenar con resolución alta.
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.num_workers,
        lr0=args.lr,
        lrf=0.01,
        weight_decay=5e-4,
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
        # Entrenamiento rectangular para reducir padding y ganar resolución efectiva.
        rect=False,

        # scale=0.15: reduce el zoom-out máximo para preservar mejor cajas
        # pequeñas. Una caja de 10px queda en un mínimo aproximado de 8.5px.
        scale=0.15,

        # translate=0.05: las rayas ocupan casi todo el ancho de la imagen
        # (~27% del ancho normalizado). Traslaciones grandes las sacan del
        # borde → cajas eliminadas por min_visibility. 5% es suficiente.
        translate=0.05,

        # shear=0.0: shear inclina las rayas horizontales. El RFI real no
        # aparece inclinado → entrenar con shear introduciría ruido.
        shear=0.0,

        # perspective=0.0: mismo razonamiento que shear.
        perspective=0.0,

        # fliplr/flipud: el RFI puede aparecer en cualquier posición y
        # orientación espejo → ambos flips son válidos físicamente.
        fliplr=0.5,
        flipud=0.0,

        # ── Color / intensidad ─────────────────────────────────────────────
        # Los quicklooks SAR codifican polarizaciones en canales RGB
        # (típicamente VV, VH, ratio). hsv_h muy bajo para no alterar la
        # firma radiométrica. hsv_v alto porque el contraste del RFI vs fondo
        # es la clave para detectarlo.
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.4,

        auto_augment=False, # AutoAugment y RandAugment aplican transformaciones muy agresivas que pueden degradar el entrenamiento con rayas finas. Mejor desactivarlos.

        # ── Mosaic / copy-paste ────────────────────────────────────────────
        mosaic=0.0,
        close_mosaic=30,

        copy_paste=0.0,
        copy_paste_mode='flip',

        # mixup=0.0: mixup en detección superpone dos imágenes y mezcla sus
        # bounding boxes. Con rayas finas esto genera targets ambiguos y
        # degrada el entrenamiento. Mejor desactivarlo.
        mixup=0.0,

        # ── Optimizador y scheduler ────────────────────────────────────────
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        #multi_scale=0.3,

        # ── Loss weights ───────────────────────────────────────────────────
        box=15.0,

        # cls para que diferencie el box del fondo, solo hay una clase (RFI) y el fondo es fondo, no hay otras clases que confundir.
        cls=1.0,
        dfl=0.0, # Distribution Focal Loss no aporta beneficio para cajas tan pequeñas y finas, y puede introducir ruido en el entrenamiento. Mejor desactivarlo.

        # Una sola clase en ClearSAR.
        single_cls=True,

        # ── Misc ───────────────────────────────────────────────────────────
        amp=True,
        verbose=True,
        erasing=0.0,
        augment=True,
        cache=True,
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

    cache_mode = False if args.cache == "none" else args.cache
    train_kwargs["cache"] = cache_mode
    print(f"[yolo] Cache mode: {cache_mode}")

    if args.evolve > 0:
        train_kwargs["evolve"] = args.evolve
        print(f"[yolo] Evolve enabled: {args.evolve} generations")

    results = model.train(**train_kwargs)

    # Validate the model
    print("\n[yolo] ═" * 40)
    print("[yolo] Validate the model")
    print("[yolo] ═" * 40)
    metrics = model.val(data=str(yaml_path), split="val", plots=False)
    print(f"[yolo] map50-95={metrics.box.map:.4f}")
    print(f"[yolo] map50={metrics.box.map50:.4f}")
    print(f"[yolo] map75={metrics.box.map75:.4f}")

    best_ckpt = yolo_runs_dir / run_name / "weights" / "best.pt"
    if best_ckpt.exists():
        dest = cfg.paths.models_dir / f"yolo_best_{args.model.replace('.pt', '')}.pt"
        shutil.copy2(best_ckpt, dest)
        print(f"\n[yolo] Best checkpoint copied to {dest}")

    print(f"[yolo] Training complete.")


if __name__ == "__main__":
    main()