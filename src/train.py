from __future__ import annotations

"""
YOLO training module for ClearSAR.

Converts COCO annotations to YOLO format, applies a stratified train/val/holdout
split, and trains with Ultralytics. Supports single-run and k-fold cross-validation.

Usage:
    python yolo_train.py --epochs 100 --model yolo11m
    python yolo_train.py --epochs 100 --kfold 5
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# COCO utilities
# ---------------------------------------------------------------------------

def _load_coco(annotation_path: Path) -> tuple[dict, Dict[int, dict]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images_meta = {int(img["id"]): img for img in coco.get("images", [])}
    return coco, images_meta


def _convert_coco_to_yolo(
    coco: dict,
    images_meta: Dict[int, dict],
    output_labels_dir: Path,
    image_ids: Optional[List[int]] = None,
) -> None:
    """Convert COCO annotations to YOLO txt format (class cx cy w h, normalized)."""
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    anns_by_image: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    ids_to_process = image_ids if image_ids is not None else list(images_meta.keys())

    for img_id in ids_to_process:
        if img_id not in images_meta:
            continue
        meta = images_meta[img_id]
        img_w, img_h = float(meta["width"]), float(meta["height"])
        stem = Path(meta["file_name"]).stem
        label_path = output_labels_dir / f"{stem}.txt"

        lines: List[str] = []
        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            cx = max(0.0, min(1.0, (x + w / 2.0) / img_w))
            cy = max(0.0, min(1.0, (y + h / 2.0) / img_h))
            nw = max(0.0, min(1.0, w / img_w))
            nh = max(0.0, min(1.0, h / img_h))
            if nw > 0 and nh > 0:
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Deduplicate while preserving order
        label_path.write_text("\n".join(dict.fromkeys(lines)), encoding="utf-8")


def _link_images(
    ids: List[int],
    images_meta: Dict[int, dict],
    src_dir: Path,
    dst_dir: Path,
) -> None:
    for img_id in ids:
        if img_id not in images_meta:
            continue
        fname = images_meta[img_id]["file_name"]
        src = src_dir / fname
        if not src.exists():
            continue
        dst = dst_dir / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)


def _write_yolo_yaml(dataset_root: Path, yaml_filename: str = "clearsar.yaml") -> Path:
    yaml_path = dataset_root / yaml_filename
    cfg = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["RFI"],
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return yaml_path


# ---------------------------------------------------------------------------
# Candidate image filtering
# ---------------------------------------------------------------------------

def _select_candidate_image_ids(
    coco: dict,
    images_meta: Dict[int, dict],
    excluded_ids: Optional[set[int]] = None,
) -> List[int]:
    """
    Filter out images that contain annotations flagged as:
      - is_extreme_ratio : bbox aspect ratio > 60
      - is_bad_resolution: image height > 1.5 × image width
    """
    bad_image_ids: set[int] = set()

    for ann in coco.get("annotations", []):
        img_id = int(ann["image_id"])
        meta = images_meta.get(img_id)
        if meta is None:
            continue
        img_w, img_h = float(meta["width"]), float(meta["height"])
        _, _, bw, bh = [float(v) for v in ann["bbox"]]

        is_extreme_ratio = (bw / max(bh, 0.1)) > 60
        is_bad_resolution = img_h > (img_w * 1.5)

        if is_extreme_ratio or is_bad_resolution:
            bad_image_ids.add(img_id)

    print(f"[filter] Excluding {len(bad_image_ids)} images with bad annotations.")

    candidate_ids = [
        img_id for img_id in images_meta
        if img_id not in bad_image_ids
        and (excluded_ids is None or img_id not in excluded_ids)
    ]

    candidate_ids = sorted(candidate_ids)
    if not candidate_ids:
        raise ValueError("No images available after filtering.")
    
    return candidate_ids


# ---------------------------------------------------------------------------
# Stratified splits
# ---------------------------------------------------------------------------

def _stratified_split(
    coco: dict,
    image_ids: List[int],
    fraction: float,
    seed: int,
) -> tuple[List[int], List[int]]:
    """Split image_ids into (keep, split_out) stratified by annotation presence."""
    positive_ids = {int(ann["image_id"]) for ann in coco.get("annotations", [])}
    pos = [i for i in image_ids if i in positive_ids]
    neg = [i for i in image_ids if i not in positive_ids]

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos = max(1, round(len(pos) * fraction)) if pos else 0
    n_neg = max(1, round(len(neg) * fraction)) if neg else 0

    split_out = sorted(pos[:n_pos] + neg[:n_neg])
    keep = sorted(pos[n_pos:] + neg[n_neg:])
    return keep, split_out


def _kfold_split(
    image_ids: List[int],
    num_folds: int,
    seed: int,
) -> List[List[int]]:
    if num_folds < 2:
        raise ValueError("--kfold must be >= 2")
    if len(image_ids) < num_folds:
        raise ValueError(f"Not enough images ({len(image_ids)}) for {num_folds} folds.")

    shuffled = image_ids[:]
    random.Random(seed).shuffle(shuffled)

    base_size, remainder = divmod(len(shuffled), num_folds)
    folds, start = [], 0
    for i in range(num_folds):
        size = base_size + (1 if i < remainder else 0)
        folds.append(shuffled[start: start + size])
        start += size
    return folds


# ---------------------------------------------------------------------------
# Dataset materialization
# ---------------------------------------------------------------------------

def _materialize_dataset(
    dataset_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    train_ids: List[int],
    val_ids: List[int],
    yaml_filename: str = "clearsar.yaml",
) -> Path:
    images_train = dataset_root / "images" / "train"
    images_val   = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val   = dataset_root / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    coco, images_meta = _load_coco(annotation_path)

    _link_images(train_ids, images_meta, train_images_dir, images_train)
    _link_images(val_ids,   images_meta, train_images_dir, images_val)

    _convert_coco_to_yolo(coco, images_meta, labels_train, train_ids)
    _convert_coco_to_yolo(coco, images_meta, labels_val,   val_ids)

    yaml_path = _write_yolo_yaml(dataset_root, yaml_filename)
    print(f"[yolo] Dataset: {yaml_path}  |  train={len(train_ids)}  val={len(val_ids)}")
    return yaml_path


def _build_holdout_dataset(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    holdout_fraction: float,
    seed: int,
) -> tuple[Path, set[int]]:
    """Reserve a holdout split from training data. Never used during training."""
    holdout_root      = project_root / "data" / "yolo" / "holdout"
    holdout_images    = holdout_root / "images" / "val"
    holdout_labels    = holdout_root / "labels" / "val"

    if holdout_root.exists():
        shutil.rmtree(holdout_root)
    holdout_images.mkdir(parents=True, exist_ok=True)
    holdout_labels.mkdir(parents=True, exist_ok=True)

    coco, images_meta = _load_coco(annotation_path)
    all_ids = sorted(images_meta.keys())
    _, holdout_ids_list = _stratified_split(coco, all_ids, holdout_fraction, seed)
    holdout_ids = set(holdout_ids_list)

    _link_images(holdout_ids_list, images_meta, train_images_dir, holdout_images)
    _convert_coco_to_yolo(coco, images_meta, holdout_labels, holdout_ids_list)

    holdout_yaml = project_root / "data" / "yolo" / "holdout.yaml"
    cfg = {
        "path": str(holdout_root.resolve()),
        "train": "images/val",
        "val":   "images/val",
        "nc": 1,
        "names": ["RFI"],
    }
    with holdout_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"[yolo] Holdout yaml: {holdout_yaml}  |  images={len(holdout_ids_list)}")
    return holdout_yaml, holdout_ids


def _build_single_dataset(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    val_fraction: float,
    seed: int,
    excluded_ids: Optional[set[int]],
) -> Path:
    coco, images_meta = _load_coco(annotation_path)
    candidate_ids = _select_candidate_image_ids(coco, images_meta, excluded_ids)
    train_ids, val_ids = _stratified_split(coco, candidate_ids, val_fraction, seed)
    return _materialize_dataset(
        dataset_root   = project_root / "data" / "yolo",
        annotation_path = annotation_path,
        train_images_dir = train_images_dir,
        train_ids = train_ids,
        val_ids   = val_ids,
    )


def _build_kfold_datasets(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    num_folds: int,
    seed: int,
    excluded_ids: Optional[set[int]],
) -> List[tuple[int, Path]]:
    coco, images_meta = _load_coco(annotation_path)
    candidate_ids = _select_candidate_image_ids(coco, images_meta, excluded_ids)
    folds = _kfold_split(candidate_ids, num_folds, seed)

    fold_specs: List[tuple[int, Path]] = []
    for fold_idx, val_ids in enumerate(folds, start=1):
        val_set   = set(val_ids)
        train_ids = [i for i in candidate_ids if i not in val_set]
        fold_root = project_root / "data" / "yolo" / "kfold" / f"fold_{fold_idx:02d}"
        yaml_path = _materialize_dataset(
            dataset_root    = fold_root,
            annotation_path = annotation_path,
            train_images_dir = train_images_dir,
            train_ids = train_ids,
            val_ids   = val_ids,
            yaml_filename = "clearsar.yaml",
        )
        fold_specs.append((fold_idx, yaml_path))

    return fold_specs


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

def _base_train_kwargs(args: argparse.Namespace, yaml_path: Path, run_name: str) -> dict:
    yolo_runs_dir = Path(args.project_root).resolve() / "outputs" / "yolo_runs" \
        if args.project_root else Path(__file__).resolve().parents[1] / "outputs" / "yolo_runs"

    kwargs = dict(
        data    = str(yaml_path),
        epochs  = args.epochs,
        imgsz   = args.image_size,
        batch   = args.batch_size,
        workers = args.num_workers,
        lr0     = args.lr,
        lrf     = args.lrf,
        seed    = args.seed,
        patience = args.patience,
        project = str(yolo_runs_dir),
        name    = run_name,
        exist_ok = True,
        save    = True,
        plots   = True,
        val     = True,
        cache   = args.cache if args.cache != "none" else False,

        # Augmentation
        scale        = 0.2,
        degrees      = 0.0,
        fliplr       = 0.5,
        shear        = 0.0,
        translate    = 0.05,
        flipud       = 0.5,
        hsv_h        = 0.01,
        hsv_s        = 0.3,
        hsv_v        = 0.5,
        mosaic       = 1.0,
        close_mosaic = 20,
        perspective  = 0.0,
        copy_paste   = 0.5,
        mixup        = 0.0,
        erasing      = 0.0,

        # Optimizer
        optimizer     = "AdamW",
        cos_lr        = True,
        warmup_epochs = 3,
        weight_decay  = 5e-4,

        # Loss weights
        box = 7.5,
        cls = 0.5,

        amp     = True,
        augment = True,
        verbose = True,
    )

    device = args.device or resolve_device()
    kwargs["device"] = device

    return kwargs, yolo_runs_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--annotation-path", type=str,
        default="data/annotations/instances_train.json",
    )
    parser.add_argument(
        "--train-images-dir", type=str,
        default="data/images/train",
    )
    parser.add_argument("--model",      type=str,   default="yolo11m")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--image-size", type=int,   default=960)
    parser.add_argument("--num-workers",type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--lrf",        type=float, default=0.01)
    parser.add_argument("--val-fraction",    type=float, default=0.1)
    parser.add_argument("--holdout-fraction",type=float, default=0.1)
    parser.add_argument(
        "--kfold", type=int, default=1,
        help="Number of folds. 1 disables k-fold and uses a single train/val split.",
    )
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--patience",type=int, default=50)
    parser.add_argument("--device",  type=str, default=None,
                        help="Device override (e.g. 'cpu', '0', 'cuda:1'). Auto-detects CUDA if omitted.")
    parser.add_argument(
        "--cache", type=str, default="disk", choices=["disk", "ram", "none"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required. Install with: pip install ultralytics")

    args = parse_args()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[1]
    )
    models_dir  = project_root / "models"
    outputs_dir = project_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.kfold < 1:
        raise ValueError("--kfold must be >= 1")

    set_seed(args.seed)

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

    # --- Resolve model path ---
    model_tag  = Path(args.model).stem
    model_path = Path(args.model)
    if model_path.suffix not in (".pt", ".pth"):
        candidates = [
            project_root / args.model,
            project_root / f"{args.model}.pt",
            models_dir   / f"{args.model}.pt",
            Path(f"{args.model}.pt"),
        ]
        found      = next((c for c in candidates if c.exists()), None)
        model_path = found if found else Path(f"{args.model}.pt")

    # --- Holdout (carved out once, never used in training) ---
    holdout_yaml, holdout_ids = _build_holdout_dataset(
        project_root     = project_root,
        annotation_path  = annotation_path,
        train_images_dir = train_images_dir,
        holdout_fraction = args.holdout_fraction,
        seed             = args.seed,
    )

    base_run_name = f"clearsar_{model_tag}"
    yolo_runs_dir = project_root / "outputs" / "yolo_runs"

    # -----------------------------------------------------------------------
    # K-fold training
    # -----------------------------------------------------------------------
    if args.kfold > 1:
        print(f"\n[yolo] K-fold training: {args.kfold} folds")
        fold_specs = _build_kfold_datasets(
            project_root     = project_root,
            annotation_path  = annotation_path,
            train_images_dir = train_images_dir,
            num_folds        = args.kfold,
            seed             = args.seed,
            excluded_ids     = holdout_ids,
        )

        fold_results = []
        for fold_idx, fold_yaml in fold_specs:
            fold_run_name = f"{base_run_name}_fold{fold_idx:02d}"
            fold_model = YOLO(str(model_path))
            kwargs, _ = _base_train_kwargs(args, fold_yaml, fold_run_name)
            print(f"\n[yolo] ===== Fold {fold_idx}/{args.kfold} =====")
            fold_model.train(**kwargs)

            metrics = fold_model.val(data=str(fold_yaml), split="val", plots=False)
            print(
                f"[yolo] fold {fold_idx} | "
                f"map50-95={metrics.box.map:.4f}  map50={metrics.box.map50:.4f}  map75={metrics.box.map75:.4f}"
            )
            fold_results.append({
                "fold_idx": fold_idx,
                "metrics":  metrics,
                "best_ckpt": yolo_runs_dir / fold_run_name / "weights" / "best.pt",
            })

        print("\n" + "=" * 55)
        print("[yolo] K-fold summary")
        print("=" * 55)
        for r in fold_results:
            m = r["metrics"]
            print(
                f"  fold {r['fold_idx']:02d}: "
                f"map50-95={m.box.map:.4f}  map50={m.box.map50:.4f}  map75={m.box.map75:.4f}"
            )

        best = max(fold_results, key=lambda r: float(r["metrics"].box.map))
        print(f"[yolo] Best fold: {best['fold_idx']} (map50-95={best['metrics'].box.map:.4f})")

        if best["best_ckpt"].exists():
            dest = models_dir / f"yolo_best_{model_tag}.pt"
            shutil.copy2(best["best_ckpt"], dest)
            print(f"[yolo] Best checkpoint → {dest}")
            final_model = YOLO(str(dest))
        else:
            print(f"[yolo] WARNING: checkpoint not found at {best['best_ckpt']}")
            final_model = fold_results[-1]["fold_model"] if fold_results else YOLO(str(model_path))

    # -----------------------------------------------------------------------
    # Single run
    # -----------------------------------------------------------------------
    else:
        yaml_path = _build_single_dataset(
            project_root     = project_root,
            annotation_path  = annotation_path,
            train_images_dir = train_images_dir,
            val_fraction     = args.val_fraction,
            seed             = args.seed,
            excluded_ids     = holdout_ids,
        )

        model = YOLO(str(model_path))
        kwargs, _ = _base_train_kwargs(args, yaml_path, base_run_name)
        model.train(**kwargs)

        print("\n" + "=" * 55)
        print("[yolo] Validation metrics")
        print("=" * 55)
        metrics = model.val(data=str(yaml_path), split="val", plots=False)
        print(
            f"[yolo] map50-95={metrics.box.map:.4f}  "
            f"map50={metrics.box.map50:.4f}  "
            f"map75={metrics.box.map75:.4f}"
        )

        best_ckpt = yolo_runs_dir / base_run_name / "weights" / "best.pt"
        if best_ckpt.exists():
            dest = models_dir / f"yolo_best_{model_tag}.pt"
            shutil.copy2(best_ckpt, dest)
            print(f"[yolo] Best checkpoint → {dest}")

        final_model = model

    # -----------------------------------------------------------------------
    # Holdout evaluation (original images, never seen during training)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("[yolo] Holdout metrics (original images, unseen during training)")
    print("=" * 55)
    holdout_metrics = final_model.val(data=str(holdout_yaml), split="val", plots=False)
    print(
        f"[yolo] holdout map50-95={holdout_metrics.box.map:.4f}  "
        f"map50={holdout_metrics.box.map50:.4f}  "
        f"map75={holdout_metrics.box.map75:.4f}"
    )

    print("\n[yolo] Training complete.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Ejemplos de ejecucion (desde la raiz del proyecto)
# ---------------------------------------------------------------------------
# Basico: entrenamiento simple con defaults.
# python src/train.py
#
# Entrenamiento con parametros comunes.
# python src/train.py --model yolo11m --epochs 120 --batch-size 8 --image-size 960
#
# K-fold (5 folds) para validacion mas robusta.
# python src/train.py --kfold 5 --epochs 80 --model yolo11m
#
# Forzar CPU (util para pruebas o entornos sin GPU).
# python src/train.py --device cpu --epochs 5 --batch-size 2
#
# Usar un checkpoint local como punto de partida.
# python src/train.py --model models/yolo_best_yolo11m.pt --epochs 30
#
# Desactivar cache en disco/RAM.
# python src/train.py --cache none

