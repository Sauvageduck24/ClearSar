from __future__ import annotations

"""
YOLO training module for ClearSAR — con soporte de horizontal slicing.

Igual que yolo_train.py pero añade generación de strips horizontales para las
imágenes de entrenamiento, mejorando la detección de bounding boxes pequeños/finos.
Los strips se guardan en disco junto a las imágenes originales antes de entrenar.

Nuevos argumentos:
  --slice-height         Altura de cada strip en píxeles (default: 160)
  --slice-overlap        Solapamiento vertical entre strips (default: 40)
  --slice-min-visibility Fracción mínima de área visible para incluir un bbox (default: 0.5)
  --no-slicing           Deshabilita el slicing (equivale al yolo_train.py original)
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import os
import yaml
from PIL import Image as PILImage
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Utilidades COCO / YOLO
# ---------------------------------------------------------------------------

def _load_coco_metadata(annotation_path: Path) -> tuple[dict, Dict[int, Dict]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images_meta: Dict[int, Dict] = {int(img["id"]): img for img in coco.get("images", [])}
    return coco, images_meta

def _split_image_ids_for_val_fraction_stratified(
    coco: dict,
    image_ids: list[int],
    val_fraction: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split estratificado por presencia de anotaciones"""
    positive_ids = {int(ann["image_id"]) for ann in coco.get("annotations", [])}
    pos_group = [img_id for img_id in image_ids if img_id in positive_ids]
    neg_group = [img_id for img_id in image_ids if img_id not in positive_ids]

    rng = random.Random(seed)
    rng.shuffle(pos_group)
    rng.shuffle(neg_group)

    n_val_pos = max(1, int(round(len(pos_group) * val_fraction))) if pos_group else 0
    n_val_neg = max(1, int(round(len(neg_group) * val_fraction))) if neg_group else 0

    val_ids = sorted(pos_group[:n_val_pos] + neg_group[:n_val_neg])
    train_ids = sorted(pos_group[n_val_pos:] + neg_group[n_val_neg:])
    return train_ids, val_ids

def _split_kfold_image_ids(
    image_ids: list[int],
    num_folds: int,
    seed: int = 42,
) -> list[list[int]]:
    if num_folds < 2:
        raise ValueError("--kfold debe ser >= 2")
    if len(image_ids) < num_folds:
        raise ValueError(
            f"No hay suficientes imagenes ({len(image_ids)}) para dividir en {num_folds} folds"
        )
    shuffled_ids = image_ids[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled_ids)
    base_size, remainder = divmod(len(shuffled_ids), num_folds)
    folds: list[list[int]] = []
    start = 0
    for fold_index in range(num_folds):
        fold_size = base_size + (1 if fold_index < remainder else 0)
        fold_ids = shuffled_ids[start: start + fold_size]
        if not fold_ids:
            raise ValueError("Fallo al construir folds: un fold quedo vacio")
        folds.append(fold_ids)
        start += fold_size
    return folds


def _select_candidate_image_ids(
    coco: dict,
    images_meta: Dict[int, Dict],
    excluded_original_image_ids: Optional[set[int]] = None,
) -> list[int]:
    bad_image_ids = set()

    for ann in coco.get("annotations", []):
        img_id = int(ann["image_id"])
        meta = images_meta[img_id]
        img_w, img_h = float(meta["width"]), float(meta["height"])
        _, _, w, h = [float(v) for v in ann["bbox"]]

        is_tiny = (w * h) < 4 or w < 1 or h < 1
        is_too_big = w > 480 and h > 300
        aspect_ratio = w / max(h, 0.1)
        is_extreme_ratio = aspect_ratio > 60
        is_bad_resolution = img_h > (img_w * 1.5)

        if is_tiny or is_too_big or is_extreme_ratio or is_bad_resolution:
            bad_image_ids.add(img_id)

    print(f"\n[Filtro] Excluyendo {len(bad_image_ids)} imagenes con cajas mal etiquetadas.")
    print()

    candidate_ids: list[int] = []
    for img_id in images_meta:
        if excluded_original_image_ids and img_id in excluded_original_image_ids:
            continue
        if img_id in bad_image_ids:
            continue
        candidate_ids.append(img_id)

    candidate_ids = sorted(candidate_ids)
    if not candidate_ids:
        raise ValueError("No hay imagenes disponibles para entrenar luego de excluir hold-out.")
    return candidate_ids

def _split_train_val_image_ids(
    annotation_path: Path,
    val_fraction: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    coco, _ = _load_coco_metadata(annotation_path)
    image_ids = sorted({int(img["id"]) for img in coco.get("images", [])})
    return _split_image_ids_for_val_fraction_stratified(coco, image_ids, val_fraction, seed)


def _convert_coco_to_yolo(
    annotation_path: Path,
    output_labels_dir: Path,
    image_ids: Optional[List[int]] = None,
    skip_vertical_boxes: bool = False,
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
            if skip_vertical_boxes and h > w:
                continue

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

        lines = list(dict.fromkeys(lines))
        label_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Slicing — generación de strips horizontales
# ---------------------------------------------------------------------------

_SLICE_MARKER = "_sl"   # sufijo que identifica imágenes generadas por slicing
_IMAGE_EXTS_TRAIN = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


def _get_strip_ranges(img_h: int, strip_height: int, overlap: int) -> list[tuple[int, int]]:
    """
    Devuelve una lista de (y_start, y_end) para strips horizontales de ancho completo.
    Si el último strip es demasiado corto (<strip_height//2), se reemplaza por uno
    anclado al fondo de la imagen.
    """
    step = max(1, strip_height - overlap)
    strips: list[tuple[int, int]] = []
    y = 0
    while y < img_h:
        y_end = min(y + strip_height, img_h)
        strips.append((y, y_end))
        if y_end >= img_h:
            break
        y += step

    # Último strip demasiado delgado → anclar al fondo
    if strips and (strips[-1][1] - strips[-1][0]) < strip_height // 2:
        bottom_strip = (max(0, img_h - strip_height), img_h)
        if bottom_strip != strips[-1]:
            strips[-1] = bottom_strip

    # Eliminar duplicados preservando orden
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for s in strips:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _generate_horizontal_slices_for_image(
    img_path: Path,
    label_path: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    strip_height: int,
    overlap: int,
    min_visibility: float = 0.5,
) -> tuple[int, int]:
    """
    Genera strips horizontales de una imagen y sus etiquetas YOLO adaptadas.

    Para cada strip:
      - Recorta la imagen al rango [0, img_w] × [y_start, y_end].
      - Transforma cada bbox al sistema de coordenadas del strip.
      - Descarta bboxes cuya visibilidad (fracción de área dentro del strip) < min_visibility.
      - El bbox recortado se normaliza respecto a las dimensiones del strip.

    Parámetros
    ----------
    img_path          : ruta a la imagen original
    label_path        : ruta al archivo .txt YOLO de la imagen original
    out_images_dir    : directorio destino para las imágenes de los strips
    out_labels_dir    : directorio destino para los labels de los strips
    strip_height      : altura en píxeles de cada strip
    overlap           : solapamiento vertical en píxeles entre strips consecutivos
    min_visibility    : fracción mínima del área original que debe ser visible en el strip

    Retorna
    -------
    (n_slices, n_boxes_total) — strips generados y boxes totales escritos
    """
    img = PILImage.open(img_path)
    img_w, img_h = img.size

    # Si la imagen es más corta que medio strip, no tiene sentido slicear
    if img_h < strip_height // 2:
        return 0, 0

    # Leer labels originales en coords normalizadas
    boxes: list[tuple[int, float, float, float, float]] = []
    if label_path.exists():
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) == 5:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append((cls, cx, cy, bw, bh))

    strips = _get_strip_ranges(img_h, strip_height, overlap)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    suffix = img_path.suffix.lower()
    # Usar PNG para evitar artefactos de compresión JPEG en slices muy finos
    save_suffix = suffix if suffix in {".png", ".tif", ".tiff"} else ".jpg"

    total_slices = 0
    total_boxes = 0

    for i, (y_start, y_end) in enumerate(strips):
        strip_h = y_end - y_start

        # Recortar imagen
        crop = img.crop((0, y_start, img_w, y_end))
        out_img_path = out_images_dir / f"{stem}{_SLICE_MARKER}{i:02d}{save_suffix}"
        save_kwargs = {"quality": 95} if save_suffix == ".jpg" else {}
        crop.save(out_img_path, **save_kwargs)

        # Transformar labels al sistema del strip
        slice_lines: list[str] = []
        for cls, cx, cy, bw, bh in boxes:
            # Coordenadas absolutas en imagen original
            abs_cx = cx * img_w
            abs_cy = cy * img_h
            abs_bw = bw * img_w
            abs_bh = bh * img_h

            b_x1 = abs_cx - abs_bw / 2.0
            b_x2 = abs_cx + abs_bw / 2.0
            b_y1 = abs_cy - abs_bh / 2.0
            b_y2 = abs_cy + abs_bh / 2.0

            # Intersección con el strip (en y)
            c_y1 = max(b_y1, float(y_start))
            c_y2 = min(b_y2, float(y_end))
            if c_y2 <= c_y1:
                continue  # Bbox fuera del strip

            # Intersección con los bordes de la imagen (en x)
            c_x1 = max(b_x1, 0.0)
            c_x2 = min(b_x2, float(img_w))
            if c_x2 <= c_x1:
                continue

            # Visibilidad: área intersección / área original
            inter_area = (c_y2 - c_y1) * (c_x2 - c_x1)
            orig_area = abs_bh * abs_bw
            if orig_area <= 0:
                continue
            if inter_area / orig_area < min_visibility:
                continue

            # Convertir a coordenadas normalizadas del strip
            new_cx = (c_x1 + c_x2) / 2.0 / img_w
            new_cy = ((c_y1 - y_start) + (c_y2 - y_start)) / 2.0 / strip_h
            new_w = (c_x2 - c_x1) / img_w
            new_h = (c_y2 - c_y1) / strip_h

            new_cx = max(0.0, min(1.0, new_cx))
            new_cy = max(0.0, min(1.0, new_cy))
            new_w = max(0.0, min(1.0, new_w))
            new_h = max(0.0, min(1.0, new_h))

            if new_w <= 0.0 or new_h <= 0.0:
                continue

            slice_lines.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

        # Escribir label del strip (puede estar vacío si ningún bbox pasa el filtro)
        slice_lines = list(dict.fromkeys(slice_lines))
        out_label_path = out_labels_dir / f"{stem}{_SLICE_MARKER}{i:02d}.txt"
        out_label_path.write_text("\n".join(slice_lines), encoding="utf-8")

        total_slices += 1
        total_boxes += len(slice_lines)

    return total_slices, total_boxes


def _add_slices_to_train_split(
    images_dir: Path,
    labels_dir: Path,
    strip_height: int,
    overlap: int,
    min_visibility: float = 0.5,
) -> tuple[int, int]:
    """
    Itera sobre todas las imágenes originales del split de entrenamiento y genera
    sus strips horizontales en el mismo directorio.

    Las imágenes ya generadas (identificadas por '_sl' en el stem) se ignoran para
    evitar doble procesado en ejecuciones sucesivas.

    Retorna (total_slices, total_boxes).
    """
    image_files = [
        p for p in sorted(images_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTS_TRAIN
        and _SLICE_MARKER not in p.stem   # no re-slicear slices previos
    ]

    if not image_files:
        print("[slicing] No se encontraron imágenes en el directorio de entrenamiento.")
        return 0, 0

    total_slices = 0
    total_boxes = 0

    for img_path in tqdm(image_files, desc="[slicing] Generando strips de entrenamiento"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        n_slices, n_boxes = _generate_horizontal_slices_for_image(
            img_path=img_path,
            label_path=label_path,
            out_images_dir=images_dir,
            out_labels_dir=labels_dir,
            strip_height=strip_height,
            overlap=overlap,
            min_visibility=min_visibility,
        )
        total_slices += n_slices
        total_boxes += n_boxes

    return total_slices, total_boxes


# ---------------------------------------------------------------------------
# Materialización del dataset YOLO
# ---------------------------------------------------------------------------

def _materialize_yolo_dataset(
    dataset_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    train_ids: list[int],
    val_ids: list[int],
    yaml_filename: str = "clearsar.yaml",
    slice_cfg: Optional[Dict] = None,
    label_filter_cfg: Optional[Dict] = None,
) -> Path:
    """
    Construye la estructura de directorios YOLO y opcionalmente genera strips
    horizontales para el split de entrenamiento.

    slice_cfg puede ser None (sin slicing) o un dict con:
        strip_height      (int)
        overlap           (int)
        min_visibility    (float)
    """
    yolo_images_train = dataset_root / "images" / "train"
    yolo_images_val = dataset_root / "images" / "val"
    yolo_labels_train = dataset_root / "labels" / "train"
    yolo_labels_val = dataset_root / "labels" / "val"

    for path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
        if path.exists():
            shutil.rmtree(path)

    for d in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    _, images_meta = _load_coco_metadata(annotation_path)

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

    skip_vertical_boxes = bool(
        label_filter_cfg and label_filter_cfg.get("skip_vertical_boxes", False)
    )
    if skip_vertical_boxes:
        print(
            "[yolo/labels] Filtros activos: "
            f"skip_vertical_boxes={skip_vertical_boxes}"
        )

    _convert_coco_to_yolo(
        annotation_path,
        yolo_labels_train,
        train_ids,
        skip_vertical_boxes=skip_vertical_boxes,
    )
    _convert_coco_to_yolo(
        annotation_path,
        yolo_labels_val,
        val_ids,
        skip_vertical_boxes=skip_vertical_boxes,
    )

    # --- Slicing ---
    if slice_cfg is not None:
        print(
            f"\n[yolo/slicing] Generando strips horizontales para train "
            f"(height={slice_cfg['strip_height']}, overlap={slice_cfg['overlap']}, "
            f"min_vis={slice_cfg['min_visibility']})..."
        )
        n_slices, n_boxes = _add_slices_to_train_split(
            images_dir=yolo_images_train,
            labels_dir=yolo_labels_train,
            strip_height=slice_cfg["strip_height"],
            overlap=slice_cfg["overlap"],
            min_visibility=slice_cfg["min_visibility"],
        )
        print(f"[yolo/slicing] {n_slices} strips generados con {n_boxes} boxes totales")
    else:
        print("[yolo/slicing] Slicing deshabilitado (--no-slicing)")

    # --- YAML ---
    yaml_path = dataset_root / yaml_filename
    dataset_cfg = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["RFI"],
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    n_train_with_slices = len(train_ids) + (n_slices if slice_cfg else 0)
    print(f"[yolo] Dataset yaml: {yaml_path}")
    print(f"[yolo] Train images (orig + slices): {n_train_with_slices}, Val images: {len(val_ids)}")
    return yaml_path


def _build_yolo_dataset(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    val_fraction: float,
    seed: int = 42,
    excluded_original_image_ids: Optional[set[int]] = None,
    slice_cfg: Optional[Dict] = None,
    label_filter_cfg: Optional[Dict] = None,
) -> Path:
    dataset_root = project_root / "data" / "yolo"
    coco, images_meta = _load_coco_metadata(annotation_path)
    candidate_ids = _select_candidate_image_ids(
        coco=coco,
        images_meta=images_meta,
        excluded_original_image_ids=excluded_original_image_ids,
    )
    train_ids, val_ids = _split_image_ids_for_val_fraction_stratified(coco, candidate_ids, val_fraction, seed)
    return _materialize_yolo_dataset(
        dataset_root=dataset_root,
        annotation_path=annotation_path,
        train_images_dir=train_images_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        yaml_filename="clearsar.yaml",
        slice_cfg=slice_cfg,
        label_filter_cfg=label_filter_cfg,
    )


def _build_kfold_yolo_datasets(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    num_folds: int,
    seed: int = 42,
    excluded_original_image_ids: Optional[set[int]] = None,
    slice_cfg: Optional[Dict] = None,
    label_filter_cfg: Optional[Dict] = None,
) -> list[tuple[int, Path]]:
    coco, images_meta = _load_coco_metadata(annotation_path)
    candidate_ids = _select_candidate_image_ids(
        coco=coco,
        images_meta=images_meta,
        excluded_original_image_ids=excluded_original_image_ids,
    )
    folds = _split_kfold_image_ids(candidate_ids, num_folds, seed)

    fold_specs: list[tuple[int, Path]] = []
    for fold_index, fold_val_ids in enumerate(folds, start=1):
        fold_val_set = set(fold_val_ids)
        fold_train_ids = [img_id for img_id in candidate_ids if img_id not in fold_val_set]
        fold_root = project_root / "data" / "yolo" / "kfold" / f"fold_{fold_index:02d}"
        fold_yaml_path = _materialize_yolo_dataset(
            dataset_root=fold_root,
            annotation_path=annotation_path,
            train_images_dir=train_images_dir,
            train_ids=fold_train_ids,
            val_ids=fold_val_ids,
            yaml_filename="clearsar.yaml",
            slice_cfg=slice_cfg,
            label_filter_cfg=label_filter_cfg,
        )
        fold_specs.append((fold_index, fold_yaml_path))

    return fold_specs


def _build_holdout_dataset(
    project_root: Path,
    holdout_fraction: float,
    seed: int = 42,
) -> tuple[Path, set[int]]:
    """Hold-out siempre se evalúa sobre imágenes ORIGINALES (sin slicing)."""
    annotation_path = project_root / "data" / "annotations" / "instances_train_og.json"
    train_images_dir = project_root / "data" / "images" / "train"

    if not annotation_path.exists():
        raise FileNotFoundError(f"No existe annotation path para hold-out: {annotation_path}")
    if not train_images_dir.exists():
        raise FileNotFoundError(f"No existe images dir para hold-out: {train_images_dir}")

    holdout_root = project_root / "data" / "yolo" / "holdout"
    holdout_images_val = holdout_root / "images" / "val"
    holdout_labels_val = holdout_root / "labels" / "val"

    if holdout_root.exists():
        shutil.rmtree(holdout_root)
    holdout_images_val.mkdir(parents=True, exist_ok=True)
    holdout_labels_val.mkdir(parents=True, exist_ok=True)

    _, holdout_ids_list = _split_train_val_image_ids(
        annotation_path=annotation_path,
        val_fraction=holdout_fraction,
        seed=seed,
    )
    holdout_ids = set(holdout_ids_list)

    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images_meta: Dict[int, Dict] = {int(img["id"]): img for img in coco.get("images", [])}

    linked = 0
    for img_id in holdout_ids_list:
        meta = images_meta.get(img_id)
        if not meta:
            continue
        fname = meta["file_name"]
        src = train_images_dir / fname
        if not src.exists():
            continue
        dst = holdout_images_val / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)
        linked += 1

    _convert_coco_to_yolo(annotation_path, holdout_labels_val, holdout_ids_list)

    holdout_yaml_path = project_root / "data" / "yolo" / "holdout.yaml"
    holdout_cfg = {
        "path": str(holdout_root.resolve()),
        "train": "images/val",
        "val": "images/val",
        "nc": 1,
        "names": ["RFI"],
    }
    with holdout_yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(holdout_cfg, f, default_flow_style=False)

    print(f"[yolo] Hold-out yaml: {holdout_yaml_path}")
    print(f"[yolo] Hold-out images (imagen original, sin slicing): {linked}")
    return holdout_yaml_path, holdout_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO detector for ClearSAR (con horizontal slicing)"
    )
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--annotation-path",
        type=str,
        default="data/annotations/instances_train.json",
        help="Ruta al COCO annotations JSON para entrenamiento (relativa a --project-root o absoluta).",
    )
    parser.add_argument(
        "--train-images-dir",
        type=str,
        default="data/images/train",
        help="Ruta al directorio de imagenes de entrenamiento (relativa a --project-root o absoluta).",
    )
    parser.add_argument(
        "--model", type=str, default="yolo11m",
        help="Model variant: yolo11n/s/m/l/x, yolo26n/s, yolov8m, etc.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--image-size", type=int, default=960,
        help="Tamaño de imagen YOLO (entero). Ejemplo: 960.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help=(
            "Workers del dataloader. En Windows se fuerza modo seguro (workers=0) "
            "por defecto para evitar errores de memoria compartida."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--kfold", type=int, default=1,
        help="Numero de folds para validacion cruzada. 1 desactiva k-fold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.1,
        help="Fraccion fija para hold-out local sobre imagenes originales (sin slicing).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--cache", type=str, default="disk", choices=["disk", "ram", "none"],
        help="Cache mode for Ultralytics dataloader: disk, ram o none.",
    )
    parser.add_argument(
        "--skip-vertical-boxes",
        action="store_true",
        help="Excluye annotations con bbox vertical (h > w) durante la conversion a YOLO.",
    )

    # --- Slicing ---
    slicing_group = parser.add_argument_group("horizontal slicing")
    slicing_group.add_argument(
        "--no-slicing", action="store_true",
        help="Deshabilita la generación de strips horizontales en entrenamiento.",
    )
    slicing_group.add_argument(
        "--slice-height", type=int, default=160,
        help="Altura en píxeles de cada strip horizontal (default: 160).",
    )
    slicing_group.add_argument(
        "--slice-overlap", type=int, default=40,
        help="Solapamiento vertical en píxeles entre strips consecutivos (default: 40).",
    )
    slicing_group.add_argument(
        "--slice-min-visibility", type=float, default=0.5,
        help="Fraccion mínima del area original que debe ser visible en el strip (default: 0.5).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for training. Install with: pip install ultralytics"
        )

    args = parse_args()
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[1]
    )
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.kfold < 1:
        raise ValueError("--kfold debe ser >= 1")

    set_seed(args.seed)

    annotation_path = Path(args.annotation_path)
    if not annotation_path.is_absolute():
        annotation_path = project_root / annotation_path

    train_images_dir = Path(args.train_images_dir)
    if not train_images_dir.is_absolute():
        train_images_dir = project_root / train_images_dir

    if not annotation_path.exists():
        raise FileNotFoundError(f"No existe annotation path: {annotation_path}")
    if not train_images_dir.exists():
        raise FileNotFoundError(f"No existe images dir: {train_images_dir}")

    # Configuración de slicing
    slice_cfg: Optional[Dict] = None
    if not args.no_slicing:
        if args.slice_height <= args.slice_overlap:
            raise ValueError(
                f"--slice-height ({args.slice_height}) debe ser mayor que "
                f"--slice-overlap ({args.slice_overlap})"
            )
        slice_cfg = {
            "strip_height": args.slice_height,
            "overlap": args.slice_overlap,
            "min_visibility": args.slice_min_visibility,
        }
        print(
            f"[yolo/slicing] Configuracion: height={args.slice_height}px, "
            f"overlap={args.slice_overlap}px, min_visibility={args.slice_min_visibility}"
        )

    label_filter_cfg: Optional[Dict] = None
    if args.skip_vertical_boxes:
        label_filter_cfg = {
            "skip_vertical_boxes": args.skip_vertical_boxes,
        }
        print(
            "[yolo/labels] Configuracion: "
            f"skip_vertical_boxes={args.skip_vertical_boxes}"
        )

    print("[yolo] Dataset source: normal")

    holdout_yaml_path, holdout_ids = _build_holdout_dataset(
        project_root=project_root,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )

    yaml_path = project_root / "data" / "yolo" / "clearsar.yaml"
    if args.kfold == 1:
        yaml_path = _build_yolo_dataset(
            project_root=project_root,
            annotation_path=annotation_path,
            train_images_dir=train_images_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            excluded_original_image_ids=holdout_ids,
            slice_cfg=slice_cfg,
            label_filter_cfg=label_filter_cfg,
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
    model_tag = Path(args.model).stem
    base_run_name = f"clearsar_{model_tag}"
    yolo_runs_dir = project_root / "outputs" / "yolo_runs"

    effective_workers = args.num_workers
    effective_cache_mode = False if args.cache == "none" else args.cache

    train_kwargs_base = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=effective_workers,
        lr0=args.lr,
        lrf=args.lrf,
        weight_decay=5e-4,
        seed=args.seed,
        patience=args.patience,
        project=str(yolo_runs_dir),
        name=base_run_name,
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
        degrees=0.0,
        rect=False,
        scale=0.3,
        translate=0.05,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        auto_augment=False,
        mosaic=1.0,
        close_mosaic=20,
        copy_paste=0.5,
        copy_paste_mode="flip",
        mixup=0.0,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,
        box=7.5,
        cls=0.3,
        single_cls=True,
        amp=True,
        verbose=True,
        erasing=0.0,
        augment=True,
    )

    if args.device is not None:
        train_kwargs_base["device"] = args.device

    train_kwargs_base["cache"] = effective_cache_mode
    print(f"[yolo] Cache mode: {effective_cache_mode}")
    print(f"[yolo] DataLoader workers: {effective_workers}")

    if args.kfold > 1:
        print(f"[yolo] K-fold enabled: {args.kfold} folds")
        fold_specs = _build_kfold_yolo_datasets(
            project_root=project_root,
            annotation_path=annotation_path,
            train_images_dir=train_images_dir,
            num_folds=args.kfold,
            seed=args.seed,
            excluded_original_image_ids=holdout_ids,
            slice_cfg=slice_cfg,
            label_filter_cfg=label_filter_cfg,
        )

        fold_results: list[dict] = []
        for fold_index, fold_yaml_path in fold_specs:
            fold_run_name = f"{base_run_name}_fold{fold_index:02d}"
            fold_model = YOLO(model_name)
            fold_train_kwargs = dict(train_kwargs_base)
            fold_train_kwargs["data"] = str(fold_yaml_path)
            fold_train_kwargs["name"] = fold_run_name
            print(f"\n[yolo] ===== Fold {fold_index}/{args.kfold} =====")
            fold_model.train(**fold_train_kwargs)

            print("[yolo] Validate fold model")
            fold_metrics = fold_model.val(data=str(fold_yaml_path), split="val", plots=False)
            print(f"[yolo] fold {fold_index} map50-95={fold_metrics.box.map:.4f}")
            print(f"[yolo] fold {fold_index} map50={fold_metrics.box.map50:.4f}")
            print(f"[yolo] fold {fold_index} map75={fold_metrics.box.map75:.4f}")

            fold_best_ckpt = yolo_runs_dir / fold_run_name / "weights" / "best.pt"
            fold_results.append({
                "fold_index": fold_index,
                "metrics": fold_metrics,
                "best_ckpt": fold_best_ckpt,
            })

        best_fold = max(fold_results, key=lambda item: float(item["metrics"].box.map))
        best_fold_index = int(best_fold["fold_index"])
        best_fold_metrics = best_fold["metrics"]
        best_ckpt = best_fold["best_ckpt"]

        print("\n" + "[yolo] ═" * 40)
        print("[yolo] K-fold summary")
        print("[yolo] ═" * 40)
        for result in fold_results:
            metrics = result["metrics"]
            fi = int(result["fold_index"])
            print(
                f"[yolo] fold {fi}: map50-95={metrics.box.map:.4f}, "
                f"map50={metrics.box.map50:.4f}, map75={metrics.box.map75:.4f}"
            )
        print(f"[yolo] best fold: {best_fold_index} (map50-95={best_fold_metrics.box.map:.4f})")

        if best_ckpt.exists():
            dest = models_dir / f"yolo_best_{model_tag}.pt"
            shutil.copy2(best_ckpt, dest)
            print(f"\n[yolo] Best fold checkpoint copied to {dest}")
            model = YOLO(str(dest))
        else:
            print(f"[yolo] WARNING: no se encontro checkpoint para el mejor fold en {best_ckpt}")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)
        run_name = base_run_name

        train_kwargs = dict(train_kwargs_base)
        train_kwargs["name"] = run_name
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
            dest = models_dir / f"yolo_best_{model_tag}.pt"
            shutil.copy2(best_ckpt, dest)
            print(f"\n[yolo] Best checkpoint copied to {dest}")

    print("\n" + "[yolo] ═" * 40)
    print("[yolo] Hold-out calibration metrics (original full image)")
    print("[yolo] ═" * 40)
    holdout_metrics = model.val(data=str(holdout_yaml_path), split="val", plots=False)
    print(f"[yolo] holdout map50-95={holdout_metrics.box.map:.4f}")
    print(f"[yolo] holdout map50={holdout_metrics.box.map50:.4f}")
    print(f"[yolo] holdout map75={holdout_metrics.box.map75:.4f}")

    print("[yolo] Training complete.")

if __name__ == "__main__":
    main()