from __future__ import annotations

"""
DEIM-D training module for ClearSAR.

DEIM (Detection with Implicit Matching) uses a COCO-compatible training loop.
This script wraps the official DEIM training pipeline, converting the dataset
to COCO format (already our native format) and launching training.

Supported model variants: deim-d-s, deim-d-m, deim-d-l, deim-d-x
(short aliases: s, m, l, x)

Installation:
    pip install deim   # or clone https://github.com/ShiqiYu/DEIM and pip install -e .

Usage:
    python -m src.deim_d_train --project-root . --epochs 100 --model deim-d-l
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2
import torch

from src.config import default_config
from src.dataset import split_train_val_image_ids
from src.utils.repro import set_seed


# ---------------------------------------------------------------------------
# SAR-specific augmentation helpers
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


def build_sar_albumentations_pipeline():
    """Pipeline de augmentación SAR reutilizable con soporte bbox COCO."""
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
        # Sin rotación: RFI son rayas horizontales físicas
    ], bbox_params=A.BboxParams(
        format="coco",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ))


# ---------------------------------------------------------------------------
# Dataset COCO split + symlink helpers
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
    Divide el COCO JSON en train/val y crea directorios de imágenes con
    symlinks. Devuelve (train_ann_path, val_ann_path, train_img_dir, val_img_dir).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    train_ids, val_ids = split_train_val_image_ids(
        annotation_path=annotation_path,
        val_fraction=val_fraction,
        seed=seed,
    )

    # Merge extra pseudo-labeled data if available
    if extra_images_dir and extra_annotations_path and extra_annotations_path.exists():
        with extra_annotations_path.open("r", encoding="utf-8") as f:
            extra_coco = json.load(f)
        id_offset = max(img["id"] for img in coco["images"]) + 1
        ann_offset = (max(a["id"] for a in coco["annotations"]) + 1) if coco["annotations"] else 1
        id_remap: Dict[int, int] = {}
        extra_imgs_patched = []
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
        print(f"[deim_d] Extra pseudo dataset: {len(extra_imgs_patched)} imágenes añadidas al train")

    images_meta: Dict[int, Dict] = {img["id"]: img for img in coco["images"]}

    def _subset_coco(ids: List[int]) -> dict:
        img_set = set(ids)
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco.get("categories", [{"id": 0, "name": "RFI"}]),
            "images": [images_meta[i] for i in ids if i in images_meta],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in img_set],
        }

    train_ann_path = output_dir / "train_annotations.json"
    val_ann_path = output_dir / "val_annotations.json"
    train_ann_path.write_text(json.dumps(_subset_coco(train_ids), ensure_ascii=False, indent=2))
    val_ann_path.write_text(json.dumps(_subset_coco(val_ids), ensure_ascii=False, indent=2))

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

    print(f"[deim_d] Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    return train_ann_path, val_ann_path, train_img_dir, val_img_dir


# ---------------------------------------------------------------------------
# Model variant mapping
# ---------------------------------------------------------------------------

_DEIM_VARIANTS = {
    "deim-d-s":  "deim_dfine_hgnetv2_s_coco",
    "deim-d-m":  "deim_dfine_hgnetv2_m_coco",
    "deim-d-l":  "deim_dfine_hgnetv2_l_coco",
    "deim-d-x":  "deim_dfine_hgnetv2_x_coco",
    # short aliases
    "s": "deim_dfine_hgnetv2_s_coco",
    "m": "deim_dfine_hgnetv2_m_coco",
    "l": "deim_dfine_hgnetv2_l_coco",
    "x": "deim_dfine_hgnetv2_x_coco",
}

_DEIM_CONFIG_URLS = {
    "deim_dfine_hgnetv2_s_coco": "configs/deim/deim_dfine_hgnetv2_s_coco.yml",
    "deim_dfine_hgnetv2_m_coco": "configs/deim/deim_dfine_hgnetv2_m_coco.yml",
    "deim_dfine_hgnetv2_l_coco": "configs/deim/deim_dfine_hgnetv2_l_coco.yml",
    "deim_dfine_hgnetv2_x_coco": "configs/deim/deim_dfine_hgnetv2_x_coco.yml",
}


def _resolve_variant(model_key: str) -> str:
    v = _DEIM_VARIANTS.get(model_key.lower())
    if v is None:
        raise ValueError(
            f"Unknown DEIM-D variant '{model_key}'. "
            f"Choose one of: {list(_DEIM_VARIANTS.keys())}"
        )
    return v


# ---------------------------------------------------------------------------
# Config patching helpers
# ---------------------------------------------------------------------------

def _patch_deim_config(
    base_config_path: Path,
    output_config_path: Path,
    train_ann_path: Path,
    val_ann_path: Path,
    train_img_dir: Path,
    val_img_dir: Path,
    num_classes: int,
    epochs: int,
    batch_size: int,
    image_size: int,
    lr: float,
    num_workers: int,
    output_dir: Path,
) -> None:
    """
    Lee el config YAML de DEIM y sobreescribe los campos de dataset, clases
    y hiperparámetros, guardando el resultado en output_config_path.

    DEIM usa configs YAML con estructura similar a D-FINE / RT-DETR.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required: pip install pyyaml")

    with base_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Dataset ─────────────────────────────────────────────────────────────
    def _set_nested(d: dict, keys: list, value) -> None:
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    _set_nested(cfg, ["train_dataloader", "dataset", "img_folder"], str(train_img_dir))
    _set_nested(cfg, ["train_dataloader", "dataset", "ann_file"], str(train_ann_path))
    _set_nested(cfg, ["val_dataloader", "dataset", "img_folder"], str(val_img_dir))
    _set_nested(cfg, ["val_dataloader", "dataset", "ann_file"], str(val_ann_path))

    # Algunos configs usan claves planas
    for key in ["img_folder", "ann_file"]:
        if key in cfg.get("train_dataloader", {}).get("dataset", {}):
            pass  # ya seteado arriba

    # ── Clases ──────────────────────────────────────────────────────────────
    _set_nested(cfg, ["model", "num_classes"], num_classes)
    # D-FINE/DEIM puede tener num_classes también en criterion
    for block in ["criterion", "postprocessor"]:
        if block in cfg.get("model", {}):
            cfg["model"][block]["num_classes"] = num_classes

    # ── Hiperparámetros ─────────────────────────────────────────────────────
    cfg["epoches"] = epochs                    # DEIM usa 'epoches' (sic)
    cfg.setdefault("optimizer", {})["lr"] = lr
    _set_nested(cfg, ["train_dataloader", "total_batch_size"], batch_size)
    _set_nested(cfg, ["val_dataloader", "total_batch_size"], max(1, batch_size // 2))
    _set_nested(cfg, ["train_dataloader", "num_workers"], num_workers)
    _set_nested(cfg, ["val_dataloader", "num_workers"], num_workers)

    # Imagen size
    for split in ["train_dataloader", "val_dataloader"]:
        _set_nested(cfg, [split, "dataset", "transforms", "size"], [image_size, image_size])

    cfg["output_dir"] = str(output_dir)

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with output_config_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"[deim_d] Config parcheado guardado en {output_config_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DEIM-D detector for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--model", type=str, default="deim-d-l",
        help="Model variant: deim-d-s/m/l/x  (or short: s/m/l/x)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-images-dir", type=str, default=None)
    parser.add_argument("--extra-annotations-path", type=str, default=None)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to .pth checkpoint to resume from",
    )
    parser.add_argument(
        "--deim-repo", type=str, default=None,
        help="Path to the local DEIM repo (needed to locate config YAMLs). "
             "If not provided, will try to import deim directly.",
    )
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

    # ── Resolve device ────────────────────────────────────────────────────
    if args.device is not None:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Build COCO dataset ────────────────────────────────────────────────
    deim_data_dir = project_root / "data" / "deim_d"
    train_ann_path, val_ann_path, train_img_dir, val_img_dir = _build_coco_splits(
        annotation_path=cfg.paths.train_annotations_path,
        train_images_dir=cfg.paths.train_images_dir,
        output_dir=deim_data_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        extra_images_dir=extra_images_dir,
        extra_annotations_path=extra_annotations_path,
    )

    # ── Resolve variant and config ────────────────────────────────────────
    variant = _resolve_variant(args.model)
    run_name = f"clearsar_{args.model.lower().replace(' ', '_')}"
    output_dir = project_root / "outputs" / "deim_d_runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate base config
    deim_repo = Path(args.deim_repo) if args.deim_repo else None
    base_config_path: Optional[Path] = None

    if deim_repo and deim_repo.exists():
        candidate = deim_repo / _DEIM_CONFIG_URLS[variant]
        if candidate.exists():
            base_config_path = candidate

    if base_config_path is None:
        # Try to find config within installed package
        try:
            import deim as _deim_pkg
            pkg_root = Path(_deim_pkg.__file__).parent
            candidate = pkg_root / _DEIM_CONFIG_URLS[variant]
            if candidate.exists():
                base_config_path = candidate
        except ImportError:
            pass

    patched_config_path = output_dir / f"{variant}_clearsar.yml"

    if base_config_path is not None:
        _patch_deim_config(
            base_config_path=base_config_path,
            output_config_path=patched_config_path,
            train_ann_path=train_ann_path,
            val_ann_path=val_ann_path,
            train_img_dir=train_img_dir,
            val_img_dir=val_img_dir,
            num_classes=1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            lr=args.lr,
            num_workers=args.num_workers,
            output_dir=output_dir,
        )
    else:
        print(
            "[deim_d] AVISO: No se encontró el config base de DEIM. "
            "Se intentará lanzar el entrenamiento con la API programática "
            "sin parchar el YAML. Considera pasar --deim-repo <ruta>."
        )
        patched_config_path = None

    # ── Launch training ───────────────────────────────────────────────────
    print(f"[deim_d] Iniciando entrenamiento DEIM-D: {variant} (device={device_str})")

    try:
        # ── Intento 1: API programática de deim (si está instalado) ─────────
        import deim
        trainer_cls = getattr(deim, "Trainer", None) or getattr(deim, "train", None)
        if trainer_cls is not None and patched_config_path is not None:
            trainer = trainer_cls(
                config=str(patched_config_path),
                resume=args.resume,
                device=device_str,
            )
            trainer.train()
        elif patched_config_path is not None:
            # ── Intento 2: lanzar via subprocess con el script train.py del repo ──
            import subprocess, sys
            cmd = [
                sys.executable,
                str(deim_repo / "train.py") if deim_repo else "train.py",
                "--config", str(patched_config_path),
                "--device", device_str,
            ]
            if args.resume:
                cmd += ["--resume", args.resume]
            print(f"[deim_d] Ejecutando: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            raise RuntimeError(
                "No se pudo localizar ni el config base ni la API de DEIM. "
                "Instala deim con 'pip install deim' o pasa --deim-repo <ruta>."
            )
    except ImportError:
        if patched_config_path is None:
            raise RuntimeError(
                "deim no está instalado y no se encontró el config base.\n"
                "Instala: pip install deim  o clona https://github.com/ShiqiYu/DEIM"
            )
        # Fallback: subprocess con el script del repo si se pasó --deim-repo
        import subprocess, sys
        train_script = (deim_repo / "train.py") if deim_repo else Path("train.py")
        cmd = [
            sys.executable, str(train_script),
            "--config", str(patched_config_path),
            "--device", device_str,
        ]
        if args.resume:
            cmd += ["--resume", args.resume]
        print(f"[deim_d] deim no importable como paquete. Ejecutando: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # ── Save best checkpoint ───────────────────────────────────────────────
    best_candidates = (
        sorted(output_dir.glob("*best*.pth"))
        + sorted(output_dir.glob("*best*.pt"))
        + sorted(output_dir.glob("best_*.pth"))
    )
    if best_candidates:
        best_ckpt = best_candidates[-1]
        dest = cfg.paths.models_dir / f"deim_d_best_{args.model.lower().replace(' ', '_')}.pth"
        shutil.copy2(best_ckpt, dest)
        print(f"[deim_d] Best checkpoint copiado a {dest}")
    else:
        print(f"[deim_d] Entrenamiento completado. Checkpoints en {output_dir}")


if __name__ == "__main__":
    main()
