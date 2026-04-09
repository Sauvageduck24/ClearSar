from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _run_slice(
    step_name: str,
    dataset_json_path: Path,
    image_dir: Path,
    output_dir: Path,
    slice_size: int,
    overlap_ratio: float,
) -> None:
    print(
        f"[{step_name}] Ejecutando con API SAHI: "
        f"dataset_json_path={dataset_json_path} image_dir={image_dir} output_dir={output_dir} "
        f"slice_size={slice_size} overlap_ratio={overlap_ratio}"
    )

    try:
        from sahi.slicing import slice_coco
    except ImportError as e:
        raise ImportError(
            "No se pudo importar SAHI. Instala con: pip install sahi"
        ) from e

    output_coco_name = dataset_json_path.name
    slice_coco(
        coco_annotation_file_path=str(dataset_json_path),
        image_dir=str(image_dir),
        output_coco_annotation_file_name=output_coco_name,
        output_dir=str(output_dir),
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        verbose=False,
    )


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def _validate_inputs(annotation_path: Path, images_dir: Path) -> None:
    if not annotation_path.exists():
        raise FileNotFoundError(f"No existe annotation path: {annotation_path}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"No existe images dir: {images_dir}")
    has_images = any(p.suffix.lower() in IMAGE_EXTS for p in images_dir.iterdir() if p.is_file())
    if not has_images:
        raise FileNotFoundError(f"No se encontraron imagenes validas en: {images_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slicing COCO con SAHI para ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)

    parser.add_argument(
        "--train-annotations",
        type=str,
        default="data/annotations/instances_train.json",
        help="Ruta al JSON COCO de train.",
    )
    parser.add_argument(
        "--train-images",
        type=str,
        default="data/images/train",
        help="Carpeta de imagenes de train.",
    )
    parser.add_argument(
        "--train-output-dir",
        type=str,
        default="data/sliced_dataset",
        help="Directorio de salida para slicing de train.",
    )

    parser.add_argument(
        "--slice-test",
        action="store_true",
        help="Si se activa, tambien ejecuta slicing para test.",
    )
    parser.add_argument(
        "--test-annotations",
        type=str,
        default=None,
        help="Ruta al JSON COCO para slicing de test (requerido con --slice-test).",
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default="data/images/test",
        help="Carpeta de imagenes de test (si --slice-test).",
    )
    parser.add_argument(
        "--test-output-dir",
        type=str,
        default="data/sliced_test",
        help="Directorio de salida para slicing de test.",
    )

    parser.add_argument("--slice-size", type=int, default=256)
    parser.add_argument("--overlap-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    if not project_root.exists():
        raise FileNotFoundError(f"No existe project-root: {project_root}")

    if args.slice_size <= 0:
        raise ValueError("--slice-size debe ser > 0")
    if not (0 <= args.overlap_ratio < 1):
        raise ValueError("--overlap-ratio debe estar en [0, 1)")

    train_annotations = _resolve_project_path(project_root, args.train_annotations)
    train_images = _resolve_project_path(project_root, args.train_images)
    train_output = _resolve_project_path(project_root, args.train_output_dir)

    _validate_inputs(train_annotations, train_images)
    train_output.mkdir(parents=True, exist_ok=True)

    _run_slice(
        step_name="slice-train",
        dataset_json_path=train_annotations,
        image_dir=train_images,
        output_dir=train_output,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap_ratio,
    )

    if args.slice_test:
        if not args.test_annotations:
            raise ValueError("Con --slice-test debes indicar --test-annotations")

        test_annotations = _resolve_project_path(project_root, args.test_annotations)
        test_images = _resolve_project_path(project_root, args.test_images)
        test_output = _resolve_project_path(project_root, args.test_output_dir)

        _validate_inputs(test_annotations, test_images)
        test_output.mkdir(parents=True, exist_ok=True)

        _run_slice(
            step_name="slice-test",
            dataset_json_path=test_annotations,
            image_dir=test_images,
            output_dir=test_output,
            slice_size=args.slice_size,
            overlap_ratio=args.overlap_ratio,
        )

    print("[slicing] Proceso completado")


if __name__ == "__main__":
    main()
