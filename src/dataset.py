from __future__ import annotations

import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from .coco_utils import _convert_coco_to_yolo, _load_coco, _random_non_overlapping_position
from .preprocessing import (
    _apply_std_multi_norm,
    _apply_vv_vh_max,
    _inject_horizontal_wavelet,
    _load_image_raw,
    _reorder_channels_by_rfi_contrast,
    _resize_image_to_shape,
    _resize_image_to_square,
    _save_image_raw,
)
from .utils import _progress_iter


def _compute_horizontal_slices(
    image_h: int,
    slice_height: int,
    slice_overlap: float,
) -> List[tuple[int, int]]:
    """Devuelve ventanas verticales [y0, y1) para slicing horizontal (ancho completo)."""
    if image_h <= 0:
        return [(0, 0)]

    sh = max(1, int(slice_height))
    ov = float(max(0.0, min(0.95, slice_overlap)))
    if sh >= image_h:
        return [(0, image_h)]

    step = max(1, int(round(sh * (1.0 - ov))))
    starts: List[int] = []
    y = 0
    while y + sh < image_h:
        starts.append(y)
        y += step

    last_start = max(0, image_h - sh)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    return [(s, min(image_h, s + sh)) for s in starts]


def _compute_vertical_slices(
    image_w: int,
    slice_width: Optional[int],
    slice_width_overlap: float,
) -> List[tuple[int, int]]:
    """Devuelve ventanas horizontales [x0, x1) para slicing vertical opcional."""
    if image_w <= 0:
        return [(0, 0)]

    if slice_width is None:
        return [(0, image_w)]

    sw = max(1, int(slice_width))
    ov = float(max(0.0, min(0.95, slice_width_overlap)))
    if sw >= image_w:
        return [(0, image_w)]

    step = max(1, int(round(sw * (1.0 - ov))))
    starts: List[int] = []
    x = 0
    while x + sw < image_w:
        starts.append(x)
        x += step

    last_start = max(0, image_w - sw)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    return [(s, min(image_w, s + sw)) for s in starts]


def _process_and_save_image(
    src: Path,
    dst: Path,
    reorder_channels: bool,
    vv_vh_max: bool,
    inject_wavelet: bool,
    std_multi_norm: bool,
    image_size: Optional[int],
) -> bool:
    """Carga, transforma y guarda una imagen individual."""
    img = _load_image_raw(src)
    if img is None or img.size == 0:
        shutil.copy2(src, dst)
        return True

    if std_multi_norm:
        img = _apply_std_multi_norm(img)
    if vv_vh_max:
        img = _apply_vv_vh_max(img)
    if reorder_channels:
        img = _reorder_channels_by_rfi_contrast(img)
    if inject_wavelet:
        img = _inject_horizontal_wavelet(img)
    if image_size is not None:
        img = _resize_image_to_square(img, int(image_size))

    if _save_image_raw(dst, img):
        return True

    shutil.copy2(src, dst)
    return True


def _link_image_worker(task: tuple) -> int:
    """Worker multiproceso para preparar una imagen (y slices si aplica)."""
    (
        fname,
        src_dir,
        dst_dir,
        reorder_channels,
        vv_vh_max,
        inject_wavelet,
        std_multi_norm,
        image_size,
        slicing,
        slice_height,
        slice_height_overlap,
        slice_width,
        slice_width_overlap,
    ) = task

    src = Path(src_dir) / fname
    if not src.exists():
        return 0

    dst_root = Path(dst_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    if slicing:
        img = _load_image_raw(src)
        if img is None or img.size == 0 or img.ndim < 2:
            return 0

        img_h = int(img.shape[0])
        img_w = int(img.shape[1])
        h_windows = _compute_horizontal_slices(img_h, slice_height, slice_height_overlap)
        v_windows = _compute_vertical_slices(img_w, slice_width, slice_width_overlap)
        stem = Path(fname).stem
        suffix = Path(fname).suffix
        created = 0

        has_vertical = len(v_windows) > 1
        for row_idx, (y0, y1) in enumerate(h_windows):
            for col_idx, (x0, x1) in enumerate(v_windows):
                if y1 <= y0 or x1 <= x0 or img_h <= 0 or img_w <= 0:
                    continue

                crop = img[y0:y1, x0:x1, ...] if img.ndim == 3 else img[y0:y1, x0:x1]
                out_img = _resize_image_to_shape(crop, img_w, img_h)
                if std_multi_norm:
                    out_img = _apply_std_multi_norm(out_img)
                if vv_vh_max:
                    out_img = _apply_vv_vh_max(out_img)
                if reorder_channels:
                    out_img = _reorder_channels_by_rfi_contrast(out_img)
                if inject_wavelet:
                    out_img = _inject_horizontal_wavelet(out_img)
                if image_size is not None:
                    out_img = _resize_image_to_square(out_img, int(image_size))

                if has_vertical:
                    slice_name = f"{stem}_sl{row_idx:03d}_sw{col_idx:03d}{suffix}"
                else:
                    slice_name = f"{stem}_sl{row_idx:03d}{suffix}"
                slice_dst = dst_root / slice_name
                if _save_image_raw(slice_dst, out_img):
                    created += 1

        return created

    dst = dst_root / fname
    if dst.exists():
        return 0

    _process_and_save_image(
        src=src,
        dst=dst,
        reorder_channels=reorder_channels,
        vv_vh_max=vv_vh_max,
        inject_wavelet=inject_wavelet,
        std_multi_norm=std_multi_norm,
        image_size=image_size,
    )
    return 1


def _link_images(
    ids: List[int],
    images_meta: dict,
    src_dir: Path,
    dst_dir: Path,
    force_copy: bool = False,
    reorder_channels: bool = False,
    vv_vh_max: bool = False,
    inject_wavelet: bool = False,
    std_multi_norm: bool = False,
    image_size: Optional[int] = None,
    slicing: bool = False,
    slice_height: int = 256,
    slice_height_overlap: float = 0.2,
    slice_width: Optional[int] = None,
    slice_width_overlap: float = 0.0,
    progress_desc: str = "images",
    prep_workers: int = 8,
) -> None:
    """
    Copia o enlaza imagenes al directorio de destino del dataset YOLO.

    Si reorder_channels=True, reordena los canales por contraste de RFI antes
    de guardar: el canal 0 siempre sera el de mayor contraste horizontal.

    Si vv_vh_max=True, construye la representacion [VV, VH, max(VV, VH)].

    Si inject_wavelet=True, inyecta los detalles horizontales de la transformada
    wavelet Haar en el canal azul (indice 2).

    Si std_multi_norm=True, aplica el preprocesado:
    R=VV(log+norm), G=VH(log+norm), B=STD_MULTI(vv_log) normalizado.

    Esto obliga a force_copy implicito (no se puede reordenar un symlink).
    """
    do_copy = (
        force_copy
        or reorder_channels
        or vv_vh_max
        or inject_wavelet
        or std_multi_norm
        or image_size is not None
        or slicing
    )

    prep_workers = max(1, int(prep_workers))

    valid_items: List[tuple[str, Path, Path]] = []
    for img_id in ids:
        if img_id not in images_meta:
            continue
        fname = images_meta[img_id]["file_name"]
        src = src_dir / fname
        dst = dst_dir / fname
        if not src.exists():
            continue
        if dst.exists() and not slicing:
            continue
        valid_items.append((fname, src, dst))

    if do_copy and prep_workers > 1:
        tasks = [
            (
                fname,
                str(src_dir),
                str(dst_dir),
                reorder_channels,
                vv_vh_max,
                inject_wavelet,
                std_multi_norm,
                image_size,
                slicing,
                slice_height,
                slice_height_overlap,
                slice_width,
                slice_width_overlap,
            )
            for (fname, _src, _dst) in valid_items
        ]

        with ProcessPoolExecutor(max_workers=prep_workers) as ex:
            futures = [ex.submit(_link_image_worker, t) for t in tasks]
            for fut in _progress_iter(as_completed(futures), total=len(futures), desc=progress_desc):
                fut.result()
        return

    for fname, src, dst in _progress_iter(valid_items, total=len(valid_items), desc=progress_desc):
        dst.parent.mkdir(parents=True, exist_ok=True)

        if do_copy:
            if slicing:
                _link_image_worker(
                    (
                        fname,
                        str(src_dir),
                        str(dst_dir),
                        reorder_channels,
                        vv_vh_max,
                        inject_wavelet,
                        std_multi_norm,
                        image_size,
                        slicing,
                        slice_height,
                        slice_height_overlap,
                        slice_width,
                        slice_width_overlap,
                    )
                )
            else:
                _process_and_save_image(
                    src=src,
                    dst=dst,
                    reorder_channels=reorder_channels,
                    vv_vh_max=vv_vh_max,
                    inject_wavelet=inject_wavelet,
                    std_multi_norm=std_multi_norm,
                    image_size=image_size,
                )
        else:
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)


def _write_yolo_yaml(
    dataset_root: Path,
    train_images_rel: str,
    val_images_rel: str,
    yaml_filename: str = "clearsar.yaml",
) -> Path:
    yaml_path = dataset_root / yaml_filename
    cfg = {
        "path": str(dataset_root.resolve()),
        "train": train_images_rel,
        "val": val_images_rel,
        "nc": 1,
        "names": ["RFI"],
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return yaml_path


def _select_candidate_image_ids(
    coco: dict,
    images_meta: dict,
    excluded_ids: Optional[set[int]] = None,
) -> List[int]:
    """
    Filtra imagenes que contienen anotaciones marcadas como:
      - is_extreme_ratio : bbox aspect ratio > 60
      - is_bad_resolution: image height > 1.5 x image width
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
        img_id
        for img_id in images_meta
        if img_id not in bad_image_ids and (excluded_ids is None or img_id not in excluded_ids)
    ]

    candidate_ids = sorted(candidate_ids)
    if not candidate_ids:
        raise ValueError("No images available after filtering.")

    return candidate_ids


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
        folds.append(shuffled[start : start + size])
        start += size
    return folds


def _build_hard_negative_crops(
    coco: dict,
    images_meta: dict,
    train_ids: List[int],
    train_images_dir: Path,
    labels_train_dir: Path,
    target_ratio: float = 0.15,
    seed: int = 42,
) -> int:
    """
    Genera recortes de fondo (sin interseccion con boxes GT) para hard negative mining.

    target_ratio es la fraccion objetivo aproximada de imagenes de fondo en el
    dataset final de train. Por ejemplo, 0.15 implica ~15% de imagenes negativas.
    """
    if not train_ids:
        return 0

    target_ratio = max(0.0, min(0.9, float(target_ratio)))
    if target_ratio <= 0.0:
        return 0

    n_base = len(train_ids)
    target_negatives = max(1, int(round((target_ratio / max(1e-6, 1.0 - target_ratio)) * n_base)))

    anns_by_image: dict[int, List[List[float]]] = {}
    for ann in coco.get("annotations", []):
        img_id = int(ann["image_id"])
        if img_id in train_ids:
            anns_by_image.setdefault(img_id, []).append([float(v) for v in ann["bbox"]])

    rng = random.Random(seed)
    train_ids_shuffled = train_ids[:]
    rng.shuffle(train_ids_shuffled)

    created = 0
    tries = 0
    max_total_tries = max(200, target_negatives * 50)
    unique_idx = 0

    while created < target_negatives and tries < max_total_tries:
        tries += 1
        img_id = rng.choice(train_ids_shuffled)
        meta = images_meta.get(img_id)
        if meta is None:
            continue

        src_name = meta["file_name"]
        src_path = train_images_dir / src_name
        if not src_path.exists():
            continue

        arr = _load_image_raw(src_path)
        if arr is None or arr.size == 0 or arr.ndim < 2:
            continue

        img_h, img_w = int(arr.shape[0]), int(arr.shape[1])
        if img_h < 8 or img_w < 8:
            continue

        min_crop_w = max(16, img_w // 8)
        max_crop_w = max(min_crop_w, img_w // 3)
        min_crop_h = max(16, img_h // 8)
        max_crop_h = max(min_crop_h, img_h // 3)

        if min_crop_w > img_w or min_crop_h > img_h:
            continue

        crop_w = rng.randint(min_crop_w, min(max_crop_w, img_w))
        crop_h = rng.randint(min_crop_h, min(max_crop_h, img_h))

        gt_boxes = anns_by_image.get(img_id, [])
        candidate = _random_non_overlapping_position(
            crop_w=crop_w,
            crop_h=crop_h,
            img_w=img_w,
            img_h=img_h,
            existing_boxes=gt_boxes,
            max_overlap=0.0,
            max_tries=80,
        )
        if candidate is None:
            continue

        x0, y0, cw, ch = [int(round(v)) for v in candidate]
        x1, y1 = x0 + cw, y0 + ch
        if x1 <= x0 or y1 <= y0:
            continue

        crop = arr[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        stem = Path(src_name).stem
        suffix = Path(src_name).suffix
        neg_stem = f"{stem}_hnm_{unique_idx:06d}"
        neg_img_path = train_images_dir / f"{neg_stem}{suffix}"
        neg_lbl_path = labels_train_dir / f"{neg_stem}.txt"
        unique_idx += 1

        if neg_img_path.exists() or neg_lbl_path.exists():
            continue

        if not _save_image_raw(neg_img_path, crop):
            continue

        neg_lbl_path.write_text("", encoding="utf-8")
        created += 1

    print(
        f"[hnm] hard_negative_mining=true target_ratio={target_ratio:.2f} "
        f"target={target_negatives} created={created} tries={tries}"
    )
    return created


def _materialize_dataset(
    dataset_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    train_ids: List[int],
    val_ids: List[int],
    yaml_filename: str = "clearsar.yaml",
    train_min_snr: Optional[float] = None,
    train_max_box_height: Optional[float] = None,
    train_remove_small: Optional[int] = None,
    train_skip_vertical_boxes: bool = False,
    train_small_box_copy_paste: bool = False,
    train_copy_paste_p: float = 0.5,
    train_copy_paste_max_h: float = 12.0,
    train_copy_paste_n: int = 3,
    train_merge_contiguous_boxes: bool = False,
    reorder_channels: bool = False,
    vv_vh_max: bool = False,
    inject_wavelet: bool = False,
    std_multi_norm: bool = False,
    hard_negative_mining: bool = False,
    seed: int = 42,
    image_size: Optional[int] = None,
    slicing: bool = False,
    slice_height: int = 256,
    slice_height_overlap: float = 0.2,
    slice_width: Optional[int] = None,
    slice_width_overlap: float = 0.0,
    prep_workers: int = 8,
) -> Path:
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    coco, images_meta = _load_coco(annotation_path)

    _link_images(
        train_ids,
        images_meta,
        train_images_dir,
        images_train,
        force_copy=train_small_box_copy_paste,
        reorder_channels=reorder_channels,
        vv_vh_max=vv_vh_max,
        inject_wavelet=inject_wavelet,
        std_multi_norm=std_multi_norm,
        image_size=image_size,
        slicing=slicing,
        slice_height=slice_height,
        slice_height_overlap=slice_height_overlap,
        slice_width=slice_width,
        slice_width_overlap=slice_width_overlap,
        progress_desc="train images",
        prep_workers=prep_workers,
    )
    _link_images(
        val_ids,
        images_meta,
        train_images_dir,
        images_val,
        reorder_channels=reorder_channels,
        vv_vh_max=vv_vh_max,
        inject_wavelet=inject_wavelet,
        std_multi_norm=std_multi_norm,
        image_size=image_size,
        slicing=slicing,
        slice_height=slice_height,
        slice_height_overlap=slice_height_overlap,
        slice_width=slice_width,
        slice_width_overlap=slice_width_overlap,
        progress_desc="val images",
        prep_workers=prep_workers,
    )

    _convert_coco_to_yolo(
        coco,
        images_meta,
        labels_train,
        train_ids,
        images_dir=images_train,
        min_snr=train_min_snr,
        max_box_height=train_max_box_height,
        min_box_height=float(train_remove_small) if train_remove_small is not None else None,
        skip_vertical_boxes=train_skip_vertical_boxes,
        small_box_copy_paste=train_small_box_copy_paste,
        copy_paste_p=train_copy_paste_p,
        copy_paste_max_h=train_copy_paste_max_h,
        copy_paste_n=train_copy_paste_n,
        merge_contiguous_boxes=train_merge_contiguous_boxes,
        resized_image_size=image_size,
        slicing=slicing,
        slice_height=slice_height,
        slice_height_overlap=slice_height_overlap,
        slice_width=slice_width,
        slice_width_overlap=slice_width_overlap,
        progress_desc="train labels",
    )

    if hard_negative_mining:
        _build_hard_negative_crops(
            coco=coco,
            images_meta=images_meta,
            train_ids=train_ids,
            train_images_dir=images_train,
            labels_train_dir=labels_train,
            target_ratio=0.15,
            seed=seed,
        )

    _convert_coco_to_yolo(
        coco,
        images_meta,
        labels_val,
        val_ids,
        resized_image_size=image_size,
        slicing=slicing,
        slice_height=slice_height,
        slice_height_overlap=slice_height_overlap,
        slice_width=slice_width,
        slice_width_overlap=slice_width_overlap,
        progress_desc="val labels",
    )

    train_images_rel = images_train.relative_to(dataset_root).as_posix()
    val_images_rel = images_val.relative_to(dataset_root).as_posix()
    yaml_path = _write_yolo_yaml(
        dataset_root=dataset_root,
        train_images_rel=train_images_rel,
        val_images_rel=val_images_rel,
        yaml_filename=yaml_filename,
    )
    print(f"[yolo] Dataset: {yaml_path}  |  train={len(train_ids)}  val={len(val_ids)}")
    return yaml_path


def _build_holdout_dataset(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    holdout_fraction: float,
    seed: int,
    reorder_channels: bool = False,
    vv_vh_max: bool = False,
    inject_wavelet: bool = False,
    std_multi_norm: bool = False,
    image_size: Optional[int] = None,
    slicing: bool = False,
    slice_height: int = 256,
    slice_height_overlap: float = 0.2,
    slice_width: Optional[int] = None,
    slice_width_overlap: float = 0.0,
    prep_workers: int = 8,
) -> tuple[Path, set[int]]:
    """Reserve a holdout split from training data. Never used during training."""
    holdout_root = project_root / "data" / "yolo" / "holdout"
    holdout_images = holdout_root / "images" / "val"
    holdout_labels = holdout_root / "labels" / "val"

    if holdout_root.exists():
        shutil.rmtree(holdout_root)
    holdout_images.mkdir(parents=True, exist_ok=True)
    holdout_labels.mkdir(parents=True, exist_ok=True)

    coco, images_meta = _load_coco(annotation_path)
    all_ids = sorted(images_meta.keys())
    _, holdout_ids_list = _stratified_split(coco, all_ids, holdout_fraction, seed)
    holdout_ids = set(holdout_ids_list)

    _link_images(
        holdout_ids_list,
        images_meta,
        train_images_dir,
        holdout_images,
        reorder_channels=reorder_channels,
        vv_vh_max=vv_vh_max,
        inject_wavelet=inject_wavelet,
        std_multi_norm=std_multi_norm,
        image_size=image_size,
        slicing=slicing,
        slice_height=slice_height,
        slice_height_overlap=slice_height_overlap,
        slice_width=slice_width,
        slice_width_overlap=slice_width_overlap,
        progress_desc="holdout images",
        prep_workers=prep_workers,
    )
    _convert_coco_to_yolo(
        coco,
        images_meta,
        holdout_labels,
        holdout_ids_list,
        resized_image_size=image_size,
        slicing=slicing,
        slice_height=slice_height,
        slice_height_overlap=slice_height_overlap,
        slice_width=slice_width,
        slice_width_overlap=slice_width_overlap,
        progress_desc="holdout labels",
    )

    holdout_yaml = project_root / "data" / "yolo" / "holdout.yaml"
    holdout_images_rel = holdout_images.relative_to(holdout_root).as_posix()
    cfg = {
        "path": str(holdout_root.resolve()),
        "train": holdout_images_rel,
        "val": holdout_images_rel,
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
    train_min_snr: Optional[float],
    train_max_box_height: Optional[float],
    train_remove_small: Optional[int],
    train_skip_vertical_boxes: bool,
    train_small_box_copy_paste: bool,
    train_copy_paste_p: float,
    train_copy_paste_max_h: float,
    train_copy_paste_n: int,
    train_merge_contiguous_boxes: bool,
    reorder_channels: bool = False,
    vv_vh_max: bool = False,
    inject_wavelet: bool = False,
    std_multi_norm: bool = False,
    hard_negative_mining: bool = False,
    image_size: Optional[int] = None,
    slicing: bool = False,
    slice_height: int = 256,
    slice_height_overlap: float = 0.2,
    slice_width: Optional[int] = None,
    slice_width_overlap: float = 0.0,
    prep_workers: int = 8,
) -> Path:
    coco, images_meta = _load_coco(annotation_path)
    candidate_ids = _select_candidate_image_ids(coco, images_meta, excluded_ids)
    train_ids, val_ids = _stratified_split(coco, candidate_ids, val_fraction, seed)
    return _materialize_dataset(
        dataset_root=project_root / "data" / "yolo",
        annotation_path=annotation_path,
        train_images_dir=train_images_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        train_min_snr=train_min_snr,
        train_max_box_height=train_max_box_height,
        train_remove_small=train_remove_small,
        train_skip_vertical_boxes=train_skip_vertical_boxes,
        train_small_box_copy_paste=train_small_box_copy_paste,
        train_copy_paste_p=train_copy_paste_p,
        train_copy_paste_max_h=train_copy_paste_max_h,
        train_copy_paste_n=train_copy_paste_n,
        train_merge_contiguous_boxes=train_merge_contiguous_boxes,
        reorder_channels=reorder_channels,
        vv_vh_max=vv_vh_max,
        inject_wavelet=inject_wavelet,
        std_multi_norm=std_multi_norm,
        hard_negative_mining=hard_negative_mining,
        seed=seed,
        image_size=image_size,
        slicing=slicing,
        slice_height=slice_height,
        slice_height_overlap=slice_height_overlap,
        slice_width=slice_width,
        slice_width_overlap=slice_width_overlap,
        prep_workers=prep_workers,
    )


def _build_kfold_datasets(
    project_root: Path,
    annotation_path: Path,
    train_images_dir: Path,
    num_folds: int,
    seed: int,
    excluded_ids: Optional[set[int]],
    train_min_snr: Optional[float],
    train_max_box_height: Optional[float],
    train_remove_small: Optional[int],
    train_skip_vertical_boxes: bool,
    train_small_box_copy_paste: bool,
    train_copy_paste_p: float,
    train_copy_paste_max_h: float,
    train_copy_paste_n: int,
    train_merge_contiguous_boxes: bool,
    reorder_channels: bool = False,
    vv_vh_max: bool = False,
    inject_wavelet: bool = False,
    std_multi_norm: bool = False,
    hard_negative_mining: bool = False,
    image_size: Optional[int] = None,
    slicing: bool = False,
    slice_height: int = 256,
    slice_height_overlap: float = 0.2,
    slice_width: Optional[int] = None,
    slice_width_overlap: float = 0.0,
    prep_workers: int = 8,
) -> List[tuple[int, Path]]:
    coco, images_meta = _load_coco(annotation_path)
    candidate_ids = _select_candidate_image_ids(coco, images_meta, excluded_ids)
    folds = _kfold_split(candidate_ids, num_folds, seed)

    fold_specs: List[tuple[int, Path]] = []
    for fold_idx, val_ids in enumerate(folds, start=1):
        val_set = set(val_ids)
        train_ids = [i for i in candidate_ids if i not in val_set]
        fold_root = project_root / "data" / "yolo" / "kfold" / f"fold_{fold_idx:02d}"
        yaml_path = _materialize_dataset(
            dataset_root=fold_root,
            annotation_path=annotation_path,
            train_images_dir=train_images_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            yaml_filename="clearsar.yaml",
            train_min_snr=train_min_snr,
            train_max_box_height=train_max_box_height,
            train_remove_small=train_remove_small,
            train_skip_vertical_boxes=train_skip_vertical_boxes,
            train_small_box_copy_paste=train_small_box_copy_paste,
            train_copy_paste_p=train_copy_paste_p,
            train_copy_paste_max_h=train_copy_paste_max_h,
            train_copy_paste_n=train_copy_paste_n,
            train_merge_contiguous_boxes=train_merge_contiguous_boxes,
            reorder_channels=reorder_channels,
            vv_vh_max=vv_vh_max,
            inject_wavelet=inject_wavelet,
            std_multi_norm=std_multi_norm,
            hard_negative_mining=hard_negative_mining,
            seed=seed + fold_idx,
            image_size=image_size,
            slicing=slicing,
            slice_height=slice_height,
            slice_height_overlap=slice_height_overlap,
            slice_width=slice_width,
            slice_width_overlap=slice_width_overlap,
            prep_workers=prep_workers,
        )
        fold_specs.append((fold_idx, yaml_path))

    return fold_specs
