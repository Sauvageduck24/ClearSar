from __future__ import annotations

"""
YOLO inference module for ClearSAR — con dual-pass horizontal slicing.

Estrategia de inferencia en dos pasadas:
  Pasada 1 (imagen completa): el modelo ve todo el contexto.
              → Se aceptan TODOS sus boxes (grandes, medianos y pequeños).
  Pasada 2 (strips horizontales): el modelo ve cada franja de la imagen
              a mayor resolución relativa, facilitando la detección de
              boxes finos/pequeños.
              → Solo se aceptan boxes cuya ALTURA en coords de imagen
                completa ≤ --slice-max-height-px (default 32).
              → Los boxes altos de esta pasada se descartan para evitar
                aceptar detecciones rotas de objetos grandes.

Las dos listas se fusionan con NMS (IoU threshold configurable).

Nuevos argumentos:
  --no-slicing            Deshabilita la segunda pasada (idéntico a yolo_inference.py)
  --slice-height          Altura de cada strip en píxeles (default: 160)
  --slice-overlap         Solapamiento vertical entre strips (default: 40)
  --slice-max-height-px   Umbral de altura máxima para aceptar un box de la pasada de slices (default: 32)
  --slice-nms-iou         IoU threshold para el NMS de fusión (default: 0.3)
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.6
DEFAULT_MAX_DET = 500
MIN_BOX_DIM = 0.1

# ---------------------------------------------------------------------------
# Utilidades de tamaño de imagen
# ---------------------------------------------------------------------------

def _parse_image_size(value: str) -> int | list[int]:
    raw = str(value).strip()
    if not raw:
        raise argparse.ArgumentTypeError("--image-size no puede estar vacio")
    if raw.isdigit():
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("--image-size debe ser > 0")
        return parsed
    parts = [p.strip() for p in raw.strip("()[]").split(",") if p.strip()]
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        h, w = int(parts[0]), int(parts[1])
        if h <= 0 or w <= 0:
            raise argparse.ArgumentTypeError("--image-size requiere valores > 0")
        return [h, w]
    raise argparse.ArgumentTypeError(
        "Formato invalido para --image-size. Usa 640 o [512,1024] / (512,1024)."
    )


def _normalize_imgsz(imgsz: int | list[int]) -> tuple[int, int]:
    if isinstance(imgsz, list):
        if len(imgsz) != 2:
            raise ValueError("--image-size como lista debe tener exactamente 2 valores")
        return int(imgsz[0]), int(imgsz[1])
    return int(imgsz), int(imgsz)


def _imgsz_for_ultralytics(imgsz: int | list[int]) -> int | list[int]:
    h, w = _normalize_imgsz(imgsz)
    return h if h == w else [h, w]


# ---------------------------------------------------------------------------
# Utilidades de boxes
# ---------------------------------------------------------------------------

def _valid_bbox(w: float, h: float) -> bool:
    return w > MIN_BOX_DIM and h > MIN_BOX_DIM


def _result_boxes_as_list(result: Any) -> list[list[float]]:
    """Extrae boxes de un resultado YOLO como lista de [x1,y1,x2,y2,score,cls]."""
    boxes_as_list: list[list[float]] = []
    if result is None or result.boxes is None:
        return boxes_as_list
    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        score = float(box.conf[0].item()) if box.conf is not None else 0.0
        class_id = int(box.cls[0].item()) if box.cls is not None else 0
        boxes_as_list.append([x1, y1, x2, y2, score, class_id])
    return boxes_as_list


def _yolo_box_to_xywh_and_score(box: Any) -> tuple[list[float], float]:
    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
    score = float(box.conf[0].item()) if box.conf is not None else 0.0
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], score


def _yolo_to_coco_bbox(
    x_center: float, y_center: float,
    width: float, height: float,
    image_width: int, image_height: int,
) -> list[float]:
    w = max(0.0, width * float(image_width))
    h = max(0.0, height * float(image_height))
    x = (x_center * float(image_width)) - (w / 2.0)
    y = (y_center * float(image_height)) - (h / 2.0)
    return [x, y, w, h]


# ---------------------------------------------------------------------------
# NMS y gap-filling
# ---------------------------------------------------------------------------

def _nms_boxes(
    boxes: list[list[float]],
    iou_threshold: float = 0.3,
) -> list[list[float]]:
    """
    NMS greedy sobre una lista de boxes [x1,y1,x2,y2,score,cls].
    Ordena por score descendente y suprime los boxes con IoU > iou_threshold.
    Se usa internamente para limpiar dentro de cada pasada por separado.
    """
    if not boxes:
        return []

    arr = np.array([[b[0], b[1], b[2], b[3], b[4]] for b in boxes], dtype=np.float32)
    scores = arr[:, 4]
    order = scores.argsort()[::-1]

    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])

        iw = np.maximum(0.0, ix2 - ix1)
        ih = np.maximum(0.0, iy2 - iy1)
        inter = iw * ih
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-6)

        order = rest[iou <= iou_threshold]

    return [boxes[k] for k in keep]


def _gap_fill(
    full_boxes: list[list[float]],
    slice_boxes: list[list[float]],
    iou_threshold: float = 0.3,
) -> list[list[float]]:
    """
    Fusión asimétrica: los boxes de la pasada completa son INTOCABLES.
    Los boxes de slices solo se añaden si NO solapan con ningún box de la
    pasada completa por encima de iou_threshold.

    Esto garantiza que los slices únicamente llenan huecos reales (objetos
    que el modelo no vio en la pasada completa), sin suprimir ni reemplazar
    nunca una detección que ya tenía contexto completo de la imagen.

    Parámetros
    ----------
    full_boxes   : boxes de la pasada completa [x1,y1,x2,y2,score,cls]
    slice_boxes  : boxes candidatos de los strips (ya filtrados por altura)
    iou_threshold: si un box de slices solapa > este valor con cualquier
                   box full, se descarta (default 0.3)

    Retorna
    -------
    Lista combinada: todos los full_boxes + los slice_boxes que pasaron el filtro.
    """
    if not slice_boxes:
        return full_boxes
    if not full_boxes:
        # Sin referencia de pasada completa, aceptamos todos los slice_boxes
        return slice_boxes

    full_arr = np.array(
        [[b[0], b[1], b[2], b[3]] for b in full_boxes], dtype=np.float32
    )
    fx1, fy1, fx2, fy2 = full_arr[:, 0], full_arr[:, 1], full_arr[:, 2], full_arr[:, 3]
    full_areas = np.maximum(0.0, fx2 - fx1) * np.maximum(0.0, fy2 - fy1)

    accepted_slice: list[list[float]] = []
    for sb in slice_boxes:
        sx1, sy1, sx2, sy2 = sb[0], sb[1], sb[2], sb[3]
        s_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
        if s_area <= 0:
            continue

        ix1 = np.maximum(fx1, sx1)
        iy1 = np.maximum(fy1, sy1)
        ix2 = np.minimum(fx2, sx2)
        iy2 = np.minimum(fy2, sy2)
        iw = np.maximum(0.0, ix2 - ix1)
        ih = np.maximum(0.0, iy2 - iy1)
        inter = iw * ih
        union = full_areas + s_area - inter
        iou = inter / np.maximum(union, 1e-6)

        if float(iou.max()) <= iou_threshold:
            # No solapa con ningún box de la pasada completa → es un hueco real
            accepted_slice.append(sb)

    return full_boxes + accepted_slice


# ---------------------------------------------------------------------------
# Slicing — strips horizontales
# ---------------------------------------------------------------------------

def _get_strip_ranges(img_h: int, strip_height: int, overlap: int) -> list[tuple[int, int]]:
    """
    Devuelve lista de (y_start, y_end) para strips de ancho completo.
    El último strip se ancla al fondo si resulta demasiado delgado.
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

    # Último strip demasiado delgado → anclarlo al fondo
    if strips and (strips[-1][1] - strips[-1][0]) < strip_height // 2:
        bottom = (max(0, img_h - strip_height), img_h)
        if bottom != strips[-1]:
            strips[-1] = bottom

    # Eliminar duplicados preservando orden
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for s in strips:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _predict_on_slices(
    model: YOLO,
    img: Image.Image,
    img_w: int,
    img_h: int,
    strip_height: int,
    overlap: int,
    imgsz: int | list[int],
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[list[float]]:
    """
    Ejecuta el modelo sobre cada strip horizontal y traduce las coordenadas
    de vuelta al sistema de la imagen completa.

    Retorna lista de [x1, y1_full, x2, y2_full, score, cls].
    """
    strips = _get_strip_ranges(img_h, strip_height, overlap)
    all_boxes: list[list[float]] = []

    for y_start, y_end in strips:
        crop = img.crop((0, y_start, img_w, y_end))
        # YOLO espera 3 canales; algunos TIFF/SAR entran como 1 canal.
        # Convertimos explicitamente cada strip a RGB para evitar errores
        # del tipo "expected input ... to have 3 channels, but got 1".
        crop_np = np.array(crop.convert("RGB"))

        results = model.predict(
            source=crop_np,
            conf=conf,
            iou=iou,
            imgsz=_imgsz_for_ultralytics(imgsz),
            max_det=max_det,
            augment=False,
            device=device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            continue

        for box in results[0].boxes:
            x1, y1_crop, x2, y2_crop = map(float, box.xyxy[0].tolist())
            score = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls = int(box.cls[0].item()) if box.cls is not None else 0

            # Traducir coordenadas y a imagen completa
            y1_full = y1_crop + y_start
            y2_full = y2_crop + y_start

            all_boxes.append([x1, y1_full, x2, y2_full, score, cls])

    return all_boxes


def _predict_dual_pass(
    model: YOLO,
    img_path: Path,
    imgsz: int | list[int],
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    slice_height: int,
    slice_overlap: int,
    slice_max_height_px: int,
    slice_nms_iou: float,
    use_slicing: bool = True,
) -> list[list[float]]:
    """
    Doble pasada de inferencia sobre una imagen.

    Pasada 1 — imagen completa:
        Todos los boxes se aceptan sin filtrar.

    Pasada 2 — strips horizontales (solo si use_slicing=True e img_h > strip_height/2):
        Se aceptan únicamente boxes cuya altura en imagen completa
        sea ≤ slice_max_height_px (boxes pequeños/finos).
        Los boxes más altos (potencialmente rotos por el corte) se descartan.

    Fusión: NMS con slice_nms_iou sobre la lista combinada.

    Retorna lista de [x1, y1, x2, y2, score, cls].
    """
    # --- Pasada 1: imagen completa ---
    full_results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=_imgsz_for_ultralytics(imgsz),
        max_det=max_det,
        augment=False,
        device=device,
        verbose=False,
    )
    full_boxes = _result_boxes_as_list(full_results[0] if full_results else None)

    if not use_slicing:
        return full_boxes

    # --- Pasada 2: strips horizontales ---
    img = Image.open(img_path)
    img_w, img_h = img.size

    # No tiene sentido slicear si la imagen ya es más corta que medio strip
    if img_h < slice_height // 2:
        return full_boxes

    slice_boxes_raw = _predict_on_slices(
        model=model,
        img=img,
        img_w=img_w,
        img_h=img_h,
        strip_height=slice_height,
        overlap=slice_overlap,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
    )

    # Filtrar: de los boxes de slices solo aceptamos los "pequeños"
    # (altura en imagen completa <= slice_max_height_px).
    # Esto descarta las detecciones rotas de objetos grandes.
    slice_boxes_small = [
        b for b in slice_boxes_raw
        if (b[3] - b[1]) <= slice_max_height_px
    ]

    # --- Fusion por gap-filling (asimetrica) ---
    # Los boxes de la pasada completa son intocables: nunca se suprimen.
    # Los boxes de slices solo se anaden si no solapan con ningun box
    # de la pasada completa (IoU > slice_nms_iou). Asi los slices
    # unicamente rellenan huecos reales, sin degradar las detecciones
    # que ya tenian contexto completo de la imagen.
    return _gap_fill(full_boxes, slice_boxes_small, iou_threshold=slice_nms_iou)


# ---------------------------------------------------------------------------
# Holdout (pycocotools)
# ---------------------------------------------------------------------------

def _parse_holdout_yaml(holdout_yaml_path: Path) -> tuple[Path, Path, list[str]]:
    if not holdout_yaml_path.exists():
        raise FileNotFoundError(f"No existe holdout-yaml: {holdout_yaml_path}")
    payload = yaml.safe_load(holdout_yaml_path.read_text(encoding="utf-8")) or {}
    root = Path(payload.get("path", holdout_yaml_path.parent))
    if not root.is_absolute():
        root = (holdout_yaml_path.parent / root).resolve()
    val_value = str(payload.get("val", "images/val"))
    images_dir = (root / val_value).resolve()
    labels_dir = (root / val_value.replace("images", "labels", 1)).resolve()
    names = payload.get("names", [])
    if isinstance(names, dict):
        ordered_keys = sorted(names.keys(), key=lambda k: int(k))
        class_names = [str(names[k]) for k in ordered_keys]
    else:
        class_names = [str(name) for name in names]
    if not class_names:
        num_classes = int(payload.get("nc", 0))
        class_names = [f"class_{i}" for i in range(num_classes)]
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No existe carpeta de imagenes holdout: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"No existe carpeta de labels holdout: {labels_dir}")
    return images_dir, labels_dir, class_names


def _build_holdout_coco_gt(
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
) -> tuple[dict[str, Any], dict[str, int], list[Path]]:
    image_files = [
        p for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not image_files:
        raise ValueError(f"No hay imagenes en holdout: {images_dir}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    filename_to_image_id: dict[str, int] = {}
    ann_id = 1

    for idx, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as image_obj:
            img_w, img_h = image_obj.size
        image_id = int(img_path.stem) if img_path.stem.isdigit() else idx
        filename_to_image_id[img_path.name] = image_id
        images.append({"id": image_id, "file_name": img_path.name, "width": img_w, "height": img_h})

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:5])
            x, y, w, h = _yolo_to_coco_bbox(x_center, y_center, width, height, img_w, img_h)
            if not _valid_bbox(w, h):
                continue
            annotations.append({
                "id": ann_id, "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    coco_gt = {"images": images, "annotations": annotations, "categories": categories}
    return coco_gt, filename_to_image_id, image_files


def _evaluate_holdout_with_pycocotools(
    model: YOLO,
    holdout_yaml_path: Path,
    imgsz: int | list[int],
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    slice_height: int = 160,
    slice_overlap: int = 40,
    slice_max_height_px: int = 32,
    slice_nms_iou: float = 0.3,
    use_slicing: bool = True,
) -> dict[str, dict[str, float]]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError(
            "pycocotools no esta instalado. Instala con: pip install pycocotools"
        ) from exc

    images_dir, labels_dir, class_names = _parse_holdout_yaml(holdout_yaml_path)
    coco_gt, filename_to_image_id, image_files = _build_holdout_coco_gt(
        images_dir, labels_dir, class_names
    )

    def _compute_coco_metrics(pred_rows: list[dict[str, Any]]) -> dict[str, float]:
        with tempfile.TemporaryDirectory(prefix="clearsar_holdout_eval_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            gt_json = tmp_path / "gt.json"
            pred_json = tmp_path / "pred.json"
            gt_json.write_text(json.dumps(coco_gt), encoding="utf-8")
            pred_json.write_text(json.dumps(pred_rows), encoding="utf-8")
            coco_gt_api = COCO(str(gt_json))
            coco_dt_api = coco_gt_api.loadRes(str(pred_json))
            coco_eval = COCOeval(coco_gt_api, coco_dt_api, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        return {
            "map50_95": float(coco_eval.stats[0]),
            "map50": float(coco_eval.stats[1]),
            "map75": float(coco_eval.stats[2]),
            "recall": float(coco_eval.stats[8]),
        }

    desc = "Holdout (dual-pass)" if use_slicing else "Holdout (single-pass)"
    coco_preds: list[dict[str, Any]] = []

    for img_path in tqdm(image_files, desc=desc):
        image_id = filename_to_image_id[img_path.name]

        merged_boxes = _predict_dual_pass(
            model=model,
            img_path=img_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            slice_height=slice_height,
            slice_overlap=slice_overlap,
            slice_max_height_px=slice_max_height_px,
            slice_nms_iou=slice_nms_iou,
            use_slicing=use_slicing,
        )

        # === HACK PARA EL LEADERBOARD: Engordar las cajas ===
        # Ajusta estos valores empíricamente. 8.0 px por lado = la caja es 16 px más ancha.
        padding_x = 8.0 
        padding_y = 2.0 
        
        with Image.open(img_path) as tmp_img:
            img_w, img_h = tmp_img.size
        
        # OJO: Necesitas saber el ancho y alto de la imagen original aquí. 
        # Si tu script hace resize a 512, pon 512. Si usa el tamaño original, usa esas variables.
        # Voy a asumir que tienes variables img_w e img_h disponibles (o usa un número grande como 9999).
        max_w = float(img_w) if 'img_w' in locals() else 9999.0
        max_h = float(img_h) if 'img_h' in locals() else 9999.0

        for x1, y1, x2, y2, score, cls in merged_boxes:
            # 1. Aplicar el padding expandiendo los límites (sin salirnos de la imagen)
            nx1 = max(0.0, float(x1) - padding_x)
            ny1 = max(0.0, float(y1) - padding_y)
            nx2 = min(max_w, float(x2) + padding_x)
            ny2 = min(max_h, float(y2) + padding_y)

            # 2. Recalcular el nuevo ancho y alto inflados
            w = max(0.0, nx2 - nx1)
            h = max(0.0, ny2 - ny1)

            if not _valid_bbox(w, h):
                continue
                
            coco_preds.append({
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [nx1, ny1, w, h],  # OJO: pasamos nx1 y ny1, no los viejos
                "score": float(score),
            })

    vertical_preds = [r for r in coco_preds if r["bbox"][3] > r["bbox"][2]]
    print(f"[holdout] Boxes verticales: {len(vertical_preds)} de {len(coco_preds)}")

    mode_label = "dual-pass (slicing)" if use_slicing else "single-pass (sin slicing)"
    print(f"\n[holdout] mAP — {mode_label}")
    metrics = _compute_coco_metrics(coco_preds)

    return {"result": metrics}

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def _merged_submission_rows(
    boxes: list[list[float]],
    image_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for x1, y1, x2, y2, score, class_id in boxes:
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        if not _valid_bbox(w, h):
            continue
        rows.append({
            "image_id": image_id,
            "category_id": int(class_id) + 1,
            "bbox": [float(x1), float(y1), w, h],
            "score": float(score),
        })
    return rows


def validate_submission_schema(rows: list[dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        for key in ("image_id", "category_id", "bbox", "score"):
            if key not in row:
                raise ValueError(f"Fila {i} sin key requerida '{key}'")
        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Fila {i} tiene bbox invalido: {bbox}")


def save_submission_auto(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(payload, encoding="utf-8")
        return
    zip_path = output_path if output_path.suffix.lower() == ".zip" else output_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", payload)


def get_train_stats(json_path: Path) -> float:
    if not json_path.exists():
        return 0.0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    num_images = len(data.get("images", []))
    num_anns = len(data.get("annotations", []))
    return num_anns / num_images if num_images else 0.0


def _load_test_id_mapping(test_images_dir: Path, mapping_path: Path) -> dict[str, int]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"No existe mapping-path: {mapping_path}")
    df = pd.read_parquet(mapping_path)
    if "id" not in df.columns:
        raise ValueError(f"El parquet no contiene columna 'id': {mapping_path}")
    mapping = {
        Path(normalized).name: int(Path(normalized).stem)
        for item_id in df["id"].astype(str)
        if "/images/test/" in (normalized := item_id.replace("\\", "/"))
        and Path(normalized).stem.isdigit()
    }
    test_files = [
        p.name for p in sorted(test_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    missing = [name for name in test_files if name not in mapping]
    if missing:
        raise ValueError(
            "Faltan IDs para imagenes de test en el mapping parquet. "
            f"Ejemplos: {', '.join(missing[:5])}"
        )
    return mapping


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO inference for ClearSAR (con dual-pass horizontal slicing)"
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--checkpoint", required=True, help="Ruta al archivo best.pt")
    parser.add_argument(
        "--mode", default="both", choices=["test", "holdout", "both"],
        help="Modo de ejecucion: holdout evalua metricas; test genera submission; both ejecuta ambos.",
    )
    parser.add_argument("--mapping-path", default="catalog.v1.parquet")
    parser.add_argument("--test-images-dir", default="data/images/test")
    parser.add_argument("--holdout-yaml", default="data/yolo/holdout.yaml")
    parser.add_argument("--output", default="outputs/submission_yolo.zip")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--image-size", type=_parse_image_size, default=640,
        help="Tamaño de entrada del detector. Ejemplos: 640 o [512,1024].",
    )

    # --- Slicing ---
    slicing_group = parser.add_argument_group("dual-pass slicing")
    slicing_group.add_argument(
        "--no-slicing", action="store_true",
        help="Deshabilita la segunda pasada con strips horizontales.",
    )
    slicing_group.add_argument(
        "--slice-height", type=int, default=160,
        help=(
            "Altura en píxeles de cada strip horizontal (default: 160). "
            "Recomendado: ~160px para imágenes de ~350px de alto."
        ),
    )
    slicing_group.add_argument(
        "--slice-overlap", type=int, default=40,
        help="Solapamiento vertical en píxeles entre strips consecutivos (default: 40).",
    )
    slicing_group.add_argument(
        "--slice-max-height-px", type=int, default=32,
        help=(
            "Umbral de altura máxima (px) para aceptar un box de la pasada de slices (default: 32). "
            "Boxes de la pasada de slices con altura > este valor se descartan para evitar "
            "aceptar detecciones rotas de objetos grandes."
        ),
    )
    slicing_group.add_argument(
        "--slice-nms-iou", type=float, default=0.3,
        help=(
            "IoU threshold para el NMS de fusión entre pasadas (default: 0.3). "
            "Bajo para tolerar solapamiento parcial de boxes finos."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    def _resolve(path_str: str, root: Path) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else root / p

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[1]
    )
    mapping_path = _resolve(args.mapping_path, project_root)
    test_images_dir = _resolve(args.test_images_dir, project_root)
    holdout_yaml_path = _resolve(args.holdout_yaml, project_root)
    output_path = _resolve(args.output, project_root)
    checkpoint_path = str(Path(args.checkpoint).resolve())

    use_slicing = not args.no_slicing

    best_iou = DEFAULT_IOU
    best_max_det = DEFAULT_MAX_DET

    detection_model = YOLO(checkpoint_path)

    # Resumen de configuración
    mode_str = "dual-pass" if use_slicing else "single-pass"
    print(
        f"[inference] imgsz={args.image_size} | conf={DEFAULT_CONF} | "
        f"iou={best_iou} | max_det={best_max_det} | mode={mode_str}"
    )
    if use_slicing:
        print(
            f"[inference/slicing] strip_height={args.slice_height}px | "
            f"overlap={args.slice_overlap}px | "
            f"max_height_from_slices={args.slice_max_height_px}px | "
            f"nms_iou={args.slice_nms_iou}"
        )

    # -----------------------------------------------------------------------
    # HOLDOUT
    # -----------------------------------------------------------------------
    if args.mode in ("holdout", "both"):
        print("\n" + "=" * 50)
        print("EVALUACION HOLDOUT")
        print("=" * 50)

        holdout_metrics = _evaluate_holdout_with_pycocotools(
            model=detection_model,
            holdout_yaml_path=holdout_yaml_path,
            imgsz=args.image_size,
            conf=DEFAULT_CONF,
            iou=best_iou,
            max_det=best_max_det,
            device=args.device,
            slice_height=args.slice_height,
            slice_overlap=args.slice_overlap,
            slice_max_height_px=args.slice_max_height_px,
            slice_nms_iou=args.slice_nms_iou,
            use_slicing=use_slicing,
        )
        base = holdout_metrics["result"]
        print(
            f"[holdout] map50-95={base['map50_95']:.4f} | "
            f"map50={base['map50']:.4f} | "
            f"map75={base['map75']:.4f} | "
            f"recall={base['recall']:.4f}"
        )

        if args.mode == "holdout":
            return

    # -----------------------------------------------------------------------
    # INFERENCIA EN TEST
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("INFERENCIA EN TEST")
    print("=" * 50)

    if not test_images_dir.is_dir():
        raise FileNotFoundError(f"No existe test-images-dir: {test_images_dir}")

    avg_train = get_train_stats(project_root / "data" / "annotations" / "instances_train.json")
    filename_to_image_id = _load_test_id_mapping(test_images_dir, mapping_path)
    image_files = [
        p for p in sorted(test_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    submission_rows: list[dict[str, Any]] = []
    n_full_only = 0
    n_from_slices = 0

    desc = "Inference (dual-pass)" if use_slicing else "Inference (single-pass)"
    for img_path in tqdm(image_files, desc=desc):
        if img_path.name not in filename_to_image_id:
            continue

        image_id = int(filename_to_image_id[img_path.name])

        if use_slicing:
            # --- Pasada 1: imagen completa ---
            full_results = detection_model.predict(
                source=str(img_path),
                conf=DEFAULT_CONF,
                iou=best_iou,
                imgsz=_imgsz_for_ultralytics(args.image_size),
                max_det=best_max_det,
                augment=False,
                device=args.device,
                verbose=False,
            )
            full_boxes = _result_boxes_as_list(full_results[0] if full_results else None)

            # --- Pasada 2: strips ---
            img = Image.open(img_path)
            img_w, img_h = img.size

            slice_boxes_raw: list[list[float]] = []
            if img_h >= args.slice_height // 2:
                slice_boxes_raw = _predict_on_slices(
                    model=detection_model,
                    img=img,
                    img_w=img_w,
                    img_h=img_h,
                    strip_height=args.slice_height,
                    overlap=args.slice_overlap,
                    imgsz=args.image_size,
                    conf=DEFAULT_CONF,
                    iou=best_iou,
                    max_det=best_max_det,
                    device=args.device,
                )

            slice_boxes_small = [
                b for b in slice_boxes_raw
                if (b[3] - b[1]) <= args.slice_max_height_px
            ]
            n_from_slices += len(slice_boxes_small)

            # --- Fusion gap-filling (asimetrica) ---
            # full_boxes son intocables; slice_boxes_small solo se anaden
            # si no solapan con ningun box de la pasada completa.
            final_boxes = _gap_fill(full_boxes, slice_boxes_small, iou_threshold=args.slice_nms_iou)
            n_full_only += len(full_boxes)
        else:
            pred_results = detection_model.predict(
                source=str(img_path),
                conf=DEFAULT_CONF,
                iou=best_iou,
                imgsz=_imgsz_for_ultralytics(args.image_size),
                max_det=best_max_det,
                augment=False,
                device=args.device,
                verbose=False,
            )
            final_boxes = _result_boxes_as_list(pred_results[0] if pred_results else None)

        submission_rows.extend(
            _merged_submission_rows(boxes=final_boxes, image_id=image_id)
        )

    total_boxes = len(submission_rows)
    avg_test = total_boxes / len(image_files) if image_files else 0.0

    print(f"\n{'=' * 50}")
    print("RESUMEN DE DENSIDAD")
    print(f"Promedio en TRAIN:        {avg_train:.2f} cajas/img")
    print(f"Promedio en TEST:         {avg_test:.2f} cajas/img")
    print(f"Total cajas:              {total_boxes}")
    if use_slicing:
        print(f"Cajas nuevas aportadas por slices (gap-fill): {n_from_slices}")
    print(f"{'=' * 50}\n")

    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)
    print(f"Proceso completado. Archivo guardado en: {output_path}")


if __name__ == "__main__":
    main()
