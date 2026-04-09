from __future__ import annotations

"""
YOLO training module for ClearSAR.

Builds a YOLO dataset structure from COCO annotations and trains with Ultralytics.
Supports either:
- normal dataset (data/annotations + data/images/train), or
- pre-sliced SAHI dataset (data/sliced_dataset by default).
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.utils.repro import set_seed


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".npy"}


def _parse_image_size(value: str) -> int | List[int]:
    raw = str(value).strip()
    if not raw:
        raise argparse.ArgumentTypeError("--image-size no puede estar vacio")

    if raw.isdigit():
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("--image-size debe ser > 0")
        return parsed

    cleaned = raw.strip("()[]")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        h, w = int(parts[0]), int(parts[1])
        if h <= 0 or w <= 0:
            raise argparse.ArgumentTypeError("--image-size requiere valores > 0")
        return [h, w]

    raise argparse.ArgumentTypeError(
        "Formato invalido para --image-size. Usa 960 o [512,1024] / (512,1024)."
    )


def _split_train_val_image_ids(
    annotation_path: Path,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    image_ids = [int(img["id"]) for img in coco.get("images", [])]
    if not image_ids:
        raise ValueError(f"No hay imagenes en annotation_path: {annotation_path}")

    rng = random.Random(seed)
    rng.shuffle(image_ids)

    val_count = max(1, int(len(image_ids) * val_fraction)) if len(image_ids) > 1 else 0
    val_count = min(val_count, len(image_ids) - 1) if len(image_ids) > 1 else 0

    val_ids = image_ids[:val_count]
    train_ids = image_ids[val_count:]

    if not train_ids and val_ids:
        train_ids.append(val_ids.pop())

    return train_ids, val_ids


def _convert_coco_to_yolo(
    annotation_path: Path,
    output_labels_dir: Path,
    image_ids: Optional[List[int]] = None,
) -> None:
    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    output_labels_dir.mkdir(parents=True, exist_ok=True)

    images_meta: Dict[int, Dict] = {int(img["id"]): img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        img_id = int(ann["image_id"])
        anns_by_image.setdefault(img_id, []).append(ann)

    ids_to_process = image_ids if image_ids is not None else list(images_meta.keys())

    for img_id in ids_to_process:
        if img_id not in images_meta:
            continue

        meta = images_meta[img_id]
        img_w = float(meta["width"])
        img_h = float(meta["height"])
        stem = Path(meta["file_name"]).stem
        label_path = output_labels_dir / f"{stem}.txt"

        anns = anns_by_image.get(img_id, [])
        lines: List[str] = []
        for ann in anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            if nw <= 0.0 or nh <= 0.0:
                continue

            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Quitar duplicados preservando orden.
        lines = list(dict.fromkeys(lines))
        label_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_sahi_dataset_paths(project_root: Path, dataset_root_arg: Optional[str]) -> tuple[Path, Path]:
    if dataset_root_arg:
        dataset_root = Path(dataset_root_arg)
        if not dataset_root.is_absolute():
            dataset_root = project_root / dataset_root
    else:
        dataset_root = project_root / "data" / "sliced_dataset"

    if not dataset_root.exists():
        raise FileNotFoundError(f"No existe dataset-root para SAHI: {dataset_root}")

    ann_candidates = [
        dataset_root / "instances_train.json",
        dataset_root / "instances.json",
        dataset_root / "annotations" / "instances_train.json",
        dataset_root / "annotations" / "instances.json",
    ]
    ann_candidates.extend(sorted(dataset_root.glob("instances_train*.json")))
    ann_candidates.extend(sorted(dataset_root.glob("instances*.json")))
    ann_candidates.extend(sorted((dataset_root / "annotations").glob("instances_train*.json")))
    ann_candidates.extend(sorted((dataset_root / "annotations").glob("instances*.json")))

    images_candidates = [
        dataset_root / "images",
        dataset_root / "train",
    ]
    images_candidates.extend(sorted(p for p in dataset_root.glob("instances_train_images*") if p.is_dir()))
    images_candidates.extend(sorted(p for p in dataset_root.glob("instances*_images*") if p.is_dir()))

    annotation_path = next((p for p in ann_candidates if p.exists()), None)
    images_dir = None
    for cand in images_candidates:
        if not cand.exists() or not cand.is_dir():
            continue
        has_images = any(
            x.is_file() and x.suffix.lower() in IMAGE_EXTS
            for x in cand.iterdir()
        )
        if has_images:
            images_dir = cand
            break

    if annotation_path is None:
        raise FileNotFoundError(
            f"No encontre anotaciones COCO en {dataset_root}. "
            "Busca uno de: instances_train.json, instances*.json, annotations/instances_train.json, annotations/instances*.json"
        )
    if images_dir is None:
        raise FileNotFoundError(
            f"No encontre carpeta de imagenes en {dataset_root}. "
            "Busca una carpeta images/train o instances_train_images*."
        )

    return annotation_path, images_dir


def _build_yolo_dataset(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    val_fraction: float,
    seed: int,
    extra_images_dir: Optional[Path] = None,
    extra_annotations_path: Optional[Path] = None,
) -> Path:
    yolo_dir = project_root / "data" / "yolo"
    yolo_images_train = yolo_dir / "images" / "train"
    yolo_images_val = yolo_dir / "images" / "val"
    yolo_labels_train = yolo_dir / "labels" / "train"
    yolo_labels_val = yolo_dir / "labels" / "val"

    # Limpiar dataset YOLO anterior para evitar archivos stale.
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    for d in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = _split_train_val_image_ids(
        annotation_path=annotation_path,
        val_fraction=val_fraction,
        seed=seed,
    )

    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images_meta: Dict[int, Dict] = {int(img["id"]): img for img in coco.get("images", [])}

    def _link_images(ids: list[int], dest_dir: Path) -> None:
        for img_id in ids:
            if img_id not in images_meta:
                continue
            fname = images_meta[img_id]["file_name"]
            src = train_images_dir / fname
            if not src.exists():
                continue
            dst = dest_dir / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)

    _link_images(train_ids, yolo_images_train)
    _link_images(val_ids, yolo_images_val)

    _convert_coco_to_yolo(annotation_path, yolo_labels_train, train_ids)
    _convert_coco_to_yolo(annotation_path, yolo_labels_val, val_ids)

    if extra_images_dir and extra_annotations_path and extra_annotations_path.exists():
        with extra_annotations_path.open("r", encoding="utf-8") as f:
            extra_coco = json.load(f)
        extra_ids = [int(img["id"]) for img in extra_coco.get("images", [])]
        extra_meta: Dict[int, Dict] = {int(img["id"]): img for img in extra_coco.get("images", [])}

        for img_id in extra_ids:
            if img_id not in extra_meta:
                continue
            fname = extra_meta[img_id]["file_name"]
            src = extra_images_dir / fname
            if not src.exists():
                continue
            dst = yolo_images_train / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)

        _convert_coco_to_yolo(extra_annotations_path, yolo_labels_train, extra_ids)
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
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m",
        help="Model variant: yolo11n/s/m/l/x, yolo26n/s, yolov8m, etc.",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["normal", "sahi"],
        default="normal",
        help="Dataset source for YOLO train: normal (COCO original) or sahi (sliced dataset).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Root folder for SAHI sliced dataset. If omitted, defaults to data/sliced_dataset.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--image-size",
        type=_parse_image_size,
        default=960,
        help="Tamano de imagen YOLO. Ejemplos: 960 o [512,1024] / (512,1024).",
    )
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


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for training. Install with: pip install ultralytics"
        )

    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    extra_images_dir = Path(args.extra_images_dir) if args.extra_images_dir else None
    extra_annotations_path = Path(args.extra_annotations_path) if args.extra_annotations_path else None

    if args.dataset_source == "sahi":
        annotation_path, train_images_dir = _resolve_sahi_dataset_paths(project_root, args.dataset_root)
        print(f"[yolo] Dataset source: SAHI ({train_images_dir}, {annotation_path})")
    else:
        annotation_path = project_root / "data" / "annotations" / "instances_train.json"
        train_images_dir = project_root / "data" / "images" / "train"
        if not annotation_path.exists():
            raise FileNotFoundError(f"No existe annotation path: {annotation_path}")
        if not train_images_dir.exists():
            raise FileNotFoundError(f"No existe images dir: {train_images_dir}")
        print("[yolo] Dataset source: normal")

    yaml_path = _build_yolo_dataset(
        project_root=project_root,
        annotation_path=annotation_path,
        train_images_dir=train_images_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        extra_images_dir=extra_images_dir,
        extra_annotations_path=extra_annotations_path,
    )

    model_arg = args.model
    model_arg_path = Path(model_arg)
    if model_arg_path.suffix in (".pt", ".pth"):
        model_name = str(model_arg)
    else:
        candidates = [
            project_root / model_arg,
            project_root / f"{model_arg}.pt",
            models_dir / f"{model_arg}.pt",
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
        degrees=0.0,
        rect=False,
        scale=0.15,
        translate=0.05,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.4,
        auto_augment=False,
        mosaic=0.0,
        close_mosaic=30,
        copy_paste=0.1,
        copy_paste_mode="flip",
        mixup=0.0,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=20.0,
        cls=0.3,
        dfl=0.0,
        single_cls=True,
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

    model.train(**train_kwargs)

    print("\n" + "[yolo] ═" * 40)
    print("[yolo] Validate the model")
    print("[yolo] ═" * 40)
    metrics = model.val(data=str(yaml_path), split="val", plots=False)
    print(f"[yolo] map50-95={metrics.box.map:.4f}")
    print(f"[yolo] map50={metrics.box.map50:.4f}")
    print(f"[yolo] map75={metrics.box.map75:.4f}")

    best_ckpt = yolo_runs_dir / run_name / "weights" / "best.pt"
    if best_ckpt.exists():
        dest = models_dir / f"yolo_best_{args.model.replace('.pt', '')}.pt"
        shutil.copy2(best_ckpt, dest)
        print(f"\n[yolo] Best checkpoint copied to {dest}")

    print("[yolo] Training complete.")


if __name__ == "__main__":
    main()
