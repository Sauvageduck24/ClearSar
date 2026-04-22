from __future__ import annotations

"""
YOLO training module for ClearSAR.

Converts COCO annotations to YOLO format, applies a stratified train/val/holdout
split, and trains with Ultralytics. Supports single-run and k-fold cross-validation.

Usage:
    python src/train.py --epochs 100 --model yolo11m
    python src/train.py --epochs 100 --kfold 5
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

from .dataset import _build_holdout_dataset, _build_kfold_datasets, _build_single_dataset
from .patches import _patch_channel_augmentations, _patch_tal_topk
from .utils import _str2bool, resolve_device, set_seed


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

def _base_train_kwargs(args: argparse.Namespace, yaml_path: Path, run_name: str) -> dict:
    yolo_runs_dir = (
        Path(args.project_root).resolve() / "outputs" / "yolo_runs"
        if args.project_root
        else Path(__file__).resolve().parents[1] / "outputs" / "yolo_runs"
    )

    kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.num_workers,
        lr0=args.lr,
        lrf=args.lrf,
        seed=args.seed,
        patience=args.patience,
        project=str(yolo_runs_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
        cache=args.cache if args.cache != "none" else False,
        scale=0.2,
        degrees=0.0,
        fliplr=0.5,
        shear=0.0,
        translate=0.05,
        flipud=0.5,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.5,
        mosaic=args.mosaic,
        close_mosaic=20,
        perspective=0.0,
        copy_paste=0.0 if args.small_box_copy_paste else args.copy_paste_p,
        mixup=0.0,
        erasing=0.0,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=5e-4,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        label_smoothing=args.label_smoothing,
        amp=True,
        augment=True,
        verbose=True,
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
        "--annotation-path",
        type=str,
        default="data/annotations/instances_train.json",
    )
    parser.add_argument(
        "--train-images-dir",
        type=str,
        default="data/images/train",
    )
    parser.add_argument("--model", type=str, default="yolo11m")
    parser.add_argument(
        "--model-yaml",
        type=str,
        default=None,
        help=(
            "Ruta opcional a un YAML de arquitectura de modelo. "
            "Si se provee, el modelo se crea desde este YAML y luego carga "
            "pesos preentrenados desde --model. Default=None."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument(
        "--slicing",
        type=_str2bool,
        default=False,
        help=(
            "Si true, aplica slicing horizontal (ancho completo) y estira cada slice "
            "al tamano original para ampliar objetos finos verticalmente."
        ),
    )
    parser.add_argument(
        "--slice-height",
        type=int,
        default=256,
        help="Altura (pixeles) de cada slice horizontal antes del estiramiento.",
    )
    parser.add_argument(
        "--slice-width",
        type=int,
        default=None,
        help="Ancho (pixeles) de cada slice vertical antes del estiramiento.",
    )
    parser.add_argument(
        "--slice-width-overlap",
        type=float,
        default=0.0,
        help="Solape fraccional entre slices verticales consecutivos (0.0 a <1.0).",
    )
    parser.add_argument(
        "--slice-height-overlap",
        type=float,
        default=0.2,
        help="Solape fraccional entre slices horizontales consecutivos (0.0 a <1.0).",
    )
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=8,
        help="Numero de procesos para construir dataset (copiado/resize/slicing).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument(
        "--kfold",
        type=int,
        default=1,
        help="Number of folds. 1 disables k-fold and uses a single train/val split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. 'cpu', '0', 'cuda:1'). Auto-detects CUDA if omitted.",
    )
    parser.add_argument("--cache", type=str, default="disk", choices=["disk", "ram", "none"])
    parser.add_argument(
        "--train-snr-threshold",
        type=float,
        default=None,
        help="Descarta cajas del set de train con evaluate_box_snr < este umbral.",
    )
    parser.add_argument(
        "--train-max-box-height",
        type=float,
        default=None,
        help="Descarta cajas del set de train cuando bbox height > este valor (pixeles COCO).",
    )
    parser.add_argument(
        "--train-merge-contiguous-boxes",
        type=_str2bool,
        default=False,
        help="Si true, fusiona cajas contiguas/intersectadas en train (true/false).",
    )
    parser.add_argument(
        "--remove-small",
        type=int,
        default=None,
        help="Elimina los boxes del train cuando h < este valor (pixeles COCO).",
    )
    parser.add_argument(
        "--skip-vertical-boxes",
        type=_str2bool,
        default=False,
        help="Si true, descarta cajas de train cuando h >= w (true/false).",
    )
    parser.add_argument(
        "--small-box-copy-paste",
        type=_str2bool,
        default=False,
        help="Si true, aplica copy-paste dirigido a cajas pequenas en train (true/false).",
    )
    parser.add_argument(
        "--copy-paste-p",
        type=float,
        default=0.5,
        help="Probabilidad por intento para pegar un crop pequeno en una posicion aleatoria.",
    )
    parser.add_argument(
        "--copy-paste-max-h",
        type=float,
        default=12.0,
        help="Una caja se considera pequena para copy-paste si h < este valor (pixeles COCO).",
    )
    parser.add_argument(
        "--copy-paste-n",
        type=int,
        default=3,
        help="Numero de intentos de pegado por imagen usando crops del pool pequeno.",
    )
    parser.add_argument("--box", type=float, default=7.5, help="Peso de loss box.")
    parser.add_argument("--cls", type=float, default=0.5, help="Peso de loss cls.")
    parser.add_argument("--dfl", type=float, default=1.5, help="Peso de loss dfl.")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=1.0,
        help="Valor de label smoothing para la perdida de clasificacion.",
    )
    parser.add_argument(
        "--tal-topk",
        type=int,
        default=None,
        help="Anchors positivos por box en TaskAlignedAssigner. Omitir para usar default de Ultralytics. Sube a 13-15 para RFIs elongados.",
    )
    parser.add_argument(
        "--reorder-channels",
        type=_str2bool,
        default=False,
        help=(
            "Si true, reordena los 3 canales BGR por contraste horizontal de RFI "
            "(mayor primero) antes de pasar la imagen a YOLO. "
            "Normaliza la representacion: el modelo siempre ve el RFI en canal 0. "
            "Se aplica a train, val y holdout por igual."
        ),
    )
    parser.add_argument(
        "--vv-vh-max",
        type=_str2bool,
        default=False,
        help=(
            "Si true, construye la representacion [VV, VH, max(VV, VH)] "
            "antes de pasar la imagen a YOLO. Se aplica a train, val y holdout por igual."
        ),
    )
    parser.add_argument(
        "--inject-wavelet",
        type=_str2bool,
        default=False,
        help=(
            "Si true, inyecta los detalles horizontales de la transformada wavelet Haar "
            "en el canal azul (indice 2) de las imagenes. "
            "El canal rojo (VV) se procesara con DWT Haar, extrayendo los detalles "
            "horizontales (LH) que realzan las lineas horizontales de RFI. "
            "Se aplica a train, val y holdout por igual."
        ),
    )
    parser.add_argument(
        "--std-multi-norm",
        type=_str2bool,
        default=False,
        help=(
            "Si true, aplica preprocesado por imagen como en analisis: "
            "R=VV(log+norm), G=VH(log+norm), B=STD_MULTI normalizado. "
            "Se aplica a train, val y holdout por igual."
        ),
    )
    parser.add_argument(
        "--specific-augmentations",
        type=_str2bool,
        default=False,
        help=(
            "Si true, activa augmentaciones de robustez de canal en train: "
            "ChannelShuffle y ChannelDropout. Default=false."
        ),
    )
    parser.add_argument(
        "--hard-negative-mining",
        type=_str2bool,
        default=False,
        help=(
            "Si true, genera recortes de fondo sin boxes en train para que "
            "~10-20%% del dataset final sean imagenes negativas (target interno ~15%%)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install with: pip install ultralytics") from exc

    args = parse_args()

    if args.tal_topk is not None:
        _patch_tal_topk(args.tal_topk)

    _patch_channel_augmentations(args.specific_augmentations)

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
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

    print(f"[yolo] Training images source: {train_images_dir.resolve()}")

    model_tag = Path(args.model).stem
    model_path = Path(args.model)
    if model_path.suffix not in (".pt", ".pth"):
        candidates = [
            project_root / args.model,
            project_root / f"{args.model}.pt",
            models_dir / f"{args.model}.pt",
            Path(f"{args.model}.pt"),
        ]
        found = next((c for c in candidates if c.exists()), None)
        model_path = found if found else Path(f"{args.model}.pt")

    model_yaml_path: Optional[Path] = None
    if args.model_yaml:
        model_yaml_path = Path(args.model_yaml)
        if not model_yaml_path.is_absolute():
            model_yaml_path = project_root / model_yaml_path
        if not model_yaml_path.exists():
            raise FileNotFoundError(f"Model YAML not found: {model_yaml_path}")
        print(f"[yolo] Model YAML: {model_yaml_path.resolve()}")

    holdout_path = project_root / Path("data/annotations/instances_train_og.json")

    def _make_model() -> YOLO:
        if model_yaml_path is None:
            return YOLO(str(model_path))

        model = YOLO(str(model_yaml_path))
        try:
            model.load(str(model_path))
            print(f"[yolo] Loaded pretrained weights from: {model_path}")
        except Exception as e:
            print(f"[yolo] WARNING: no se pudieron cargar pesos preentrenados ({model_path}): {e}")
        return model

    holdout_yaml, holdout_ids = _build_holdout_dataset(
        project_root=project_root,
        annotation_path=holdout_path,
        train_images_dir=train_images_dir,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        reorder_channels=args.reorder_channels,
        vv_vh_max=args.vv_vh_max,
        inject_wavelet=args.inject_wavelet,
        std_multi_norm=args.std_multi_norm,
        image_size=args.image_size,
        slicing=args.slicing,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        slice_width_overlap=args.slice_width_overlap,
        slice_height_overlap=args.slice_height_overlap,
        prep_workers=args.prep_workers,
    )

    base_run_name = f"clearsar_{model_tag}"
    yolo_runs_dir = project_root / "outputs" / "yolo_runs"

    if args.kfold > 1:
        print(f"\n[yolo] K-fold training: {args.kfold} folds")
        fold_specs = _build_kfold_datasets(
            project_root=project_root,
            annotation_path=annotation_path,
            train_images_dir=train_images_dir,
            num_folds=args.kfold,
            seed=args.seed,
            excluded_ids=holdout_ids,
            train_min_snr=args.train_snr_threshold,
            train_max_box_height=args.train_max_box_height,
            train_remove_small=args.remove_small,
            train_skip_vertical_boxes=args.skip_vertical_boxes,
            train_small_box_copy_paste=args.small_box_copy_paste,
            train_copy_paste_p=args.copy_paste_p,
            train_copy_paste_max_h=args.copy_paste_max_h,
            train_copy_paste_n=args.copy_paste_n,
            train_merge_contiguous_boxes=args.train_merge_contiguous_boxes,
            reorder_channels=args.reorder_channels,
            vv_vh_max=args.vv_vh_max,
            inject_wavelet=args.inject_wavelet,
            std_multi_norm=args.std_multi_norm,
            hard_negative_mining=args.hard_negative_mining,
            image_size=args.image_size,
            slicing=args.slicing,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            slice_width_overlap=args.slice_width_overlap,
            slice_height_overlap=args.slice_height_overlap,
            prep_workers=args.prep_workers,
        )

        fold_results = []
        for fold_idx, fold_yaml in fold_specs:
            fold_run_name = f"{base_run_name}_fold{fold_idx:02d}"
            fold_model = _make_model()
            kwargs, _ = _base_train_kwargs(args, fold_yaml, fold_run_name)
            print(f"\n[yolo] ===== Fold {fold_idx}/{args.kfold} =====")
            fold_model.train(**kwargs)

            metrics = fold_model.val(data=str(fold_yaml), split="val", plots=False)
            print(
                f"[yolo] fold {fold_idx} | "
                f"map50-95={metrics.box.map:.4f}  map50={metrics.box.map50:.4f}  map75={metrics.box.map75:.4f}"
            )
            fold_results.append(
                {
                    "fold_idx": fold_idx,
                    "metrics": metrics,
                    "best_ckpt": yolo_runs_dir / fold_run_name / "weights" / "best.pt",
                    "fold_model": fold_model,
                }
            )

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
            print(f"[yolo] Best checkpoint -> {dest}")
            final_model = YOLO(str(dest))
        else:
            print(f"[yolo] WARNING: checkpoint not found at {best['best_ckpt']}")
            final_model = fold_results[-1]["fold_model"] if fold_results else YOLO(str(model_path))

    else:
        yaml_path = _build_single_dataset(
            project_root=project_root,
            annotation_path=annotation_path,
            train_images_dir=train_images_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            excluded_ids=holdout_ids,
            train_min_snr=args.train_snr_threshold,
            train_max_box_height=args.train_max_box_height,
            train_remove_small=args.remove_small,
            train_skip_vertical_boxes=args.skip_vertical_boxes,
            train_small_box_copy_paste=args.small_box_copy_paste,
            train_copy_paste_p=args.copy_paste_p,
            train_copy_paste_max_h=args.copy_paste_max_h,
            train_copy_paste_n=args.copy_paste_n,
            train_merge_contiguous_boxes=args.train_merge_contiguous_boxes,
            reorder_channels=args.reorder_channels,
            vv_vh_max=args.vv_vh_max,
            inject_wavelet=args.inject_wavelet,
            std_multi_norm=args.std_multi_norm,
            hard_negative_mining=args.hard_negative_mining,
            image_size=args.image_size,
            slicing=args.slicing,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            slice_width_overlap=args.slice_width_overlap,
            slice_height_overlap=args.slice_height_overlap,
            prep_workers=args.prep_workers,
        )

        model = _make_model()
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
            print(f"[yolo] Best checkpoint -> {dest}")

        final_model = model

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
