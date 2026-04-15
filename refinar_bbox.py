"""
refinar_bbox.py — Feature engineering de bounding boxes para datasets SAR/RFI.

Estrategia:
  Para cada anotacion aceptada, extraemos el crop, umbalizamos por percentil
  para aislar los pixeles brillantes (el RFI real), buscamos contornos y
  aplicamos cv2.boundingRect sobre todos ellos combinados.

  La bbox refinada reemplaza a la original solo si pasa todos los filtros de
  validacion. En caso contrario se conserva la original con un flag explicativo.

Filtros de validacion:
  - El contenido brillante debe ocupar al menos min_content_ratio del crop original.
  - El desplazamiento del centro no puede superar max_center_shift_ratio
    (para no ajustarse a un pixel de ruido lejos del objeto real).
  - La nueva bbox no puede ser menor que min_box_dim en ninguna dimension.

Salida:
  - JSON refinado listo para entrenamiento.
  - PNGs de debug con bbox original (azul) y bbox refinada (verde) superpuestas.
  - Resumen de estadisticas en consola.
"""

from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Tipos internos
# ---------------------------------------------------------------------------

def compute_iou_and_ioa(box1: list[float], box2: list[float]) -> tuple[float, float]:
    """
    Calcula el Intersection over Union (IoU) y el Intersection over Area (IoA)
    box = [x, y, w, h]
    Devuelve: (IoU, IoA_max) 
    IoA_max es el porcentaje de solapamiento respecto a la caja más pequeña
    (útil para detectar si una caja está 100% dentro de otra aunque la grande sea inmensa).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Coordenadas min/max
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2

    # Intersección
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    
    # ¿Qué porcentaje de la caja más pequeña está cubierto por la intersección?
    min_area = min(area1, area2)
    ioa = inter_area / min_area if min_area > 0 else 0.0

    return iou, ioa

def merge_overlapping_boxes(anns: list[dict], iou_thr: float = 0.4, ioa_thr: float = 0.8) -> list[dict]:
    """
    Toma una lista de anotaciones (de UNA imagen) y fusiona las que se solapen mucho.
    """
    if not anns:
        return []

    # Trabajar con una copia activa
    active_anns = anns.copy()
    merged_anns = []

    while active_anns:
        # Coger la primera
        curr_ann = active_anns.pop(0)
        cx, cy, cw, ch = curr_ann["bbox"]
        c_x1, c_y1, c_x2, c_y2 = cx, cy, cx + cw, cy + ch
        
        to_remove = []
        for i, other_ann in enumerate(active_anns):
            ox, oy, ow, oh = other_ann["bbox"]
            iou, ioa = compute_iou_and_ioa([cx, cy, cw, ch], [ox, oy, ow, oh])
            
            # Si se solapan mucho, o si una está casi entera dentro de otra
            if iou > iou_thr or ioa > ioa_thr:
                # FUSIONAR: expandimos las coordenadas para englobar ambas
                o_x1, o_y1, o_x2, o_y2 = ox, oy, ox + ow, oy + oh
                c_x1 = min(c_x1, o_x1)
                c_y1 = min(c_y1, o_y1)
                c_x2 = max(c_x2, o_x2)
                c_y2 = max(c_y2, o_y2)
                
                # Actualizar las variables de curr_ann
                cx, cy, cw, ch = c_x1, c_y1, c_x2 - c_x1, c_y2 - c_y1
                curr_ann["bbox"] = [cx, cy, cw, ch]
                curr_ann["area"] = cw * ch
                
                to_remove.append(i)
                
        # Eliminar las que ya hemos fusionado en curr_ann
        for i in reversed(to_remove):
            active_anns.pop(i)
            
        merged_anns.append(curr_ann)

    return merged_anns

@dataclass
class RefineResult:
    x: float
    y: float
    w: float
    h: float
    reason: str          # 'refined' | 'kept_*'
    delta_cx: float = 0.0   # desplazamiento del centro en x (px)
    delta_cy: float = 0.0   # desplazamiento del centro en y (px)
    area_ratio: float = 1.0  # area_nueva / area_original


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Normaliza a uint8 independientemente del dtype de entrada."""
    if img.dtype == np.uint8:
        return img
    if img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip((img.astype(np.float32) - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Logica central de refinamiento
# ---------------------------------------------------------------------------

def refine_single_bbox(
    img: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    threshold_percentile: float = 65.0,
    padding: int = 2,
    min_content_ratio: float = 0.05,
    max_center_shift_ratio: float = 0.40,
    min_box_dim: float = 2.0,
    morph_close_px: int = 3,
) -> RefineResult:
    """
    Refina una sola bbox usando el boundingRect de los pixeles brillantes.

    Parametros
    ----------
    threshold_percentile  : percentil del crop para binarizacion (65-80 suele funcionar bien).
    padding               : pixeles extra alrededor del contenido encontrado.
    min_content_ratio     : fraccion minima del area original que debe tener el contenido brillante.
    max_center_shift_ratio: maximo desplazamiento del centro como fraccion de la dimension mayor.
    min_box_dim           : dimension minima valida en px.
    morph_close_px        : kernel de cierre morfologico para unir fragmentos proximos (0=desactivado).
    """
    orig_area = float(max(w, 0.0) * max(h, 0.0))

    # --- Extraer crop clipeado a bordes de imagen ---
    cx1 = max(0, int(round(x)))
    cy1 = max(0, int(round(y)))
    cx2 = min(img_w, int(round(x + w)))
    cy2 = min(img_h, int(round(y + h)))

    if cx2 <= cx1 or cy2 <= cy1:
        return RefineResult(x, y, w, h, "kept_crop_empty")

    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return RefineResult(x, y, w, h, "kept_crop_empty")

    crop_8 = _to_uint8(crop)

    # --- Umbralizar para aislar el contenido brillante ---
    thresh_val = float(np.percentile(crop_8, threshold_percentile))
    thresh_val = max(thresh_val, 5.0)  # nunca umbral trivial
    _, binary = cv2.threshold(crop_8, thresh_val, 255, cv2.THRESH_BINARY)

    # Cierre morfologico opcional para unir fragmentos cercanos
    if morph_close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_close_px, morph_close_px)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # --- Buscar contornos ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return RefineResult(x, y, w, h, "kept_no_contours")

    # Bounding rect global de todos los contornos combinados
    all_pts = np.vstack(contours)
    bx, by, bw, bh = cv2.boundingRect(all_pts)

    # --- Validacion 1: el contenido debe ser suficientemente grande ---
    content_area = float(bw * bh)
    if orig_area > 0 and content_area < min_content_ratio * orig_area:
        return RefineResult(x, y, w, h, "kept_content_too_small")

    # Mapear a coordenadas de imagen completa con padding
    nx1 = float(max(0,     cx1 + bx - padding))
    ny1 = float(max(0,     cy1 + by - padding))
    nx2 = float(min(img_w, cx1 + bx + bw + padding))
    ny2 = float(min(img_h, cy1 + by + bh + padding))
    nw = nx2 - nx1
    nh = ny2 - ny1

    if nw < min_box_dim or nh < min_box_dim:
        return RefineResult(x, y, w, h, "kept_new_too_small")

    # --- Validacion 2: el centro no se puede haber desplazado demasiado ---
    orig_cx = x + w / 2.0
    orig_cy = y + h / 2.0
    new_cx  = nx1 + nw / 2.0
    new_cy  = ny1 + nh / 2.0
    dcx = abs(new_cx - orig_cx)
    dcy = abs(new_cy - orig_cy)
    max_allowed = max_center_shift_ratio * max(w, h, 1.0)
    if dcx > max_allowed or dcy > max_allowed:
        return RefineResult(
            x, y, w, h, "kept_center_shift_too_large",
            delta_cx=dcx, delta_cy=dcy,
        )

    new_area = nw * nh
    area_ratio = new_area / orig_area if orig_area > 0 else 1.0

    return RefineResult(
        nx1, ny1, nw, nh, "refined",
        delta_cx=dcx, delta_cy=dcy, area_ratio=area_ratio,
    )


# ---------------------------------------------------------------------------
# Debug visual
# ---------------------------------------------------------------------------

def _draw_debug_image(
    image_path: Path,
    boxes_before_after: list[dict],
    output_path: Path,
    title: str,
) -> bool:
    """
    Genera imagen de debug con:
      - bbox original  → azul
      - bbox refinada  → verde  (solo si cambio)
      - bbox sin cambio → naranja
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    img_viz = cv2.cvtColor(_to_uint8(img), cv2.COLOR_GRAY2BGR)
    h_img, w_img = img.shape

    def clip_rect(rx, ry, rw, rh):
        x1 = max(0, min(w_img - 1, int(round(rx))))
        y1 = max(0, min(h_img - 1, int(round(ry))))
        x2 = max(0, min(w_img - 1, int(round(rx + rw))))
        y2 = max(0, min(h_img - 1, int(round(ry + rh))))
        return x1, y1, x2, y2

    for item in boxes_before_after:
        bx, by, bw, bh = item["before"]
        ax, ay, aw, ah = item["after"]
        was_refined = item.get("refined", False)

        # Original: azul siempre
        x1, y1, x2, y2 = clip_rect(bx, by, bw, bh)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(img_viz, (x1, y1), (x2, y2), (200, 80, 0), 1)  # azul oscuro

        if was_refined:
            # Refinada: verde brillante
            ax1, ay1, ax2, ay2 = clip_rect(ax, ay, aw, ah)
            if ax2 > ax1 and ay2 > ay1:
                cv2.rectangle(img_viz, (ax1, ay1), (ax2, ay2), (40, 210, 40), 1)
        else:
            # Sin cambio: naranja
            ax1, ay1, ax2, ay2 = clip_rect(ax, ay, aw, ah)
            if ax2 > ax1 and ay2 > ay1:
                cv2.rectangle(img_viz, (ax1, ay1), (ax2, ay2), (0, 165, 255), 1)

    # Leyenda
    cv2.rectangle(img_viz, (4, 4), (390, 72), (0, 0, 0), -1)
    cv2.putText(img_viz, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img_viz, "azul=original | verde=refinada | naranja=sin_cambio", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    n_ref = sum(1 for b in boxes_before_after if b.get("refined"))
    n_tot = len(boxes_before_after)
    cv2.putText(img_viz, f"refinadas={n_ref}/{n_tot}", (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), img_viz)


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def refine_dataset_bboxes(
    input_json: str,
    output_json: str,
    images_dir: str,
    # --- parametros de refinamiento ---
    threshold_percentile: float = 65.0,
    padding: int = 2,
    min_content_ratio: float = 0.05,
    max_center_shift_ratio: float = 0.40,
    min_box_dim: float = 2.0,
    morph_close_px: int = 3,
    # --- debug ---
    examples_dir: str = "outputs/bbox_refine_debug",
    max_example_images: int = 16,
    max_boxes_per_example: int = 200,
) -> None:
    """
    Lee el JSON de anotaciones, refina cada bbox con boundingRect sobre
    el contenido brillante y escribe un JSON nuevo.

    Parametros de refinamiento
    --------------------------
    threshold_percentile  : percentil para binarizacion dentro del crop (recomendado 60-75).
    padding               : margen extra en px alrededor del boundingRect (recomendado 1-4).
    min_content_ratio     : fraccion minima del area original que debe tener el contenido.
                            Sube este valor si ves que ajusta a ruido puntual.
    max_center_shift_ratio: si el centro se mueve mas de esta fraccion de la dimension mayor
                            de la caja, se descarta el refinamiento. 0.40 = hasta 40%.
    min_box_dim           : dimension minima en px para aceptar la caja refinada.
    morph_close_px        : kernel de cierre morfologico para unir fragmentos. 0 = desactivar.
    """
    print(f"Cargando {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_dict = {img["id"]: img["file_name"] for img in coco["images"]}
    img_size_dict: dict[int, tuple[int, int]] = {}  # image_id → (w, h)
    images_path = Path(images_dir)

    # Agrupar anotaciones por imagen
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Contadores globales
    stats = {
        "total_anns": len(coco.get("annotations", [])),
        "refined": 0,
        "kept_no_contours": 0,
        "kept_content_too_small": 0,
        "kept_center_shift_too_large": 0,
        "kept_new_too_small": 0,
        "kept_crop_empty": 0,
        "kept_other": 0,
        "imgs_procesadas": 0,
        "imgs_sin_archivo": 0,
        "imgs_no_leidas": 0,
    }

    # Acumuladores para histograma de desplazamientos y ratios de area
    delta_centers: list[float] = []
    area_ratios: list[float] = []

    # Registro por imagen para debug visual
    per_image_debug: list[dict] = []

    # Mapa id_original → anotacion refinada
    refined_by_id: dict[int, dict] = {}

    print("Refinando bboxes...")
    for image_id, anns in tqdm(anns_by_img.items()):
        file_name = img_dict.get(image_id)
        if not file_name:
            continue

        img_file = images_path / file_name
        if not img_file.exists():
            stats["imgs_sin_archivo"] += 1
            continue

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats["imgs_no_leidas"] += 1
            continue

        stats["imgs_procesadas"] += 1
        img_h, img_w = img.shape
        img_size_dict[image_id] = (img_w, img_h)

        image_boxes_debug: list[dict] = []

        for ann in anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]

            result = refine_single_bbox(
                img=img,
                x=x, y=y, w=w, h=h,
                img_w=img_w, img_h=img_h,
                threshold_percentile=threshold_percentile,
                padding=padding,
                min_content_ratio=min_content_ratio,
                max_center_shift_ratio=max_center_shift_ratio,
                min_box_dim=min_box_dim,
                morph_close_px=morph_close_px,
            )

            # Actualizar contador
            if result.reason == "refined":
                stats["refined"] += 1
                delta_centers.append(max(result.delta_cx, result.delta_cy))
                area_ratios.append(result.area_ratio)
            elif result.reason in stats:
                stats[result.reason] += 1
            else:
                stats["kept_other"] += 1

            new_ann = ann.copy()
            new_ann["bbox"] = [result.x, result.y, result.w, result.h]
            # Actualizar area COCO
            new_ann["area"] = float(result.w * result.h)
            refined_by_id[ann["id"]] = new_ann

            if len(image_boxes_debug) < max_boxes_per_example:
                image_boxes_debug.append({
                    "before": [x, y, w, h],
                    "after":  [result.x, result.y, result.w, result.h],
                    "refined": result.reason == "refined",
                    "reason": result.reason,
                    "delta_cx": result.delta_cx,
                    "delta_cy": result.delta_cy,
                    "area_ratio": result.area_ratio,
                })

        if image_boxes_debug:
            n_ref_img = sum(1 for b in image_boxes_debug if b["refined"])
            per_image_debug.append({
                "image_id": image_id,
                "file_name": file_name,
                "boxes": image_boxes_debug,
                "n_refined": n_ref_img,
                "n_total": len(image_boxes_debug),
            })

    # ---------------------------------------------------------------------------
    # PASO NUEVO: Fusionar cajas superpuestas imagen por imagen
    # ---------------------------------------------------------------------------
    print("Fusionando cajas superpuestas (Limpiando errores humanos)...")
    
    # Agrupar las refinadas por imagen
    refined_anns_by_img = {}
    for ann in coco.get("annotations", []):
        final_ann = refined_by_id.get(ann["id"], ann)
        img_id = final_ann["image_id"]
        refined_anns_by_img.setdefault(img_id, []).append(final_ann)
        
    final_clean_annotations = []
    
    for img_id, img_anns in refined_anns_by_img.items():
        # Fusiona cajas que se solapen más de un 40%, o si una está un 80% dentro de otra
        merged = merge_overlapping_boxes(img_anns, iou_thr=0.40, ioa_thr=0.80)
        final_clean_annotations.extend(merged)
        
    print(f"Cajas antes de fusionar : {len(coco.get('annotations', []))}")
    print(f"Cajas despues de fusionar: {len(final_clean_annotations)}")

    coco["annotations"] = final_clean_annotations

    # Guardar JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    # ---------------------------------------------------------------------------
    # Guardar ejemplos de debug
    # ---------------------------------------------------------------------------
    examples_path = Path(examples_dir)
    examples_path.mkdir(parents=True, exist_ok=True)

    # Priorizar imagenes con mas cajas refinadas
    per_image_debug.sort(key=lambda x: x["n_refined"], reverse=True)
    selected = per_image_debug[:max_example_images]

    saved = 0
    for idx, item in enumerate(selected, start=1):
        img_path = images_path / item["file_name"]
        out_file = examples_path / f"refine_{idx:02d}_{Path(item['file_name']).stem}.png"
        title = (
            f"{item['file_name']} | ref={item['n_refined']}/{item['n_total']}"
        )
        ok = _draw_debug_image(
            image_path=img_path,
            boxes_before_after=item["boxes"],
            output_path=out_file,
            title=title,
        )
        if ok:
            saved += 1

    # ---------------------------------------------------------------------------
    # Resumen
    # ---------------------------------------------------------------------------
    total = max(stats["total_anns"], 1)
    refined = stats["refined"]
    kept = total - refined

    print("\n--- RESUMEN DEL REFINAMIENTO ---")
    print(f"Anotaciones totales      : {stats['total_anns']}")
    print(f"Imagenes procesadas      : {stats['imgs_procesadas']}")
    print(f"Imagenes sin archivo     : {stats['imgs_sin_archivo']}")
    print(f"Imagenes no leidas       : {stats['imgs_no_leidas']}")
    print()
    print(f"Bboxes refinadas         : {refined}  ({100*refined/total:.1f}%)")
    print(f"Bboxes sin cambio        : {kept}  ({100*kept/total:.1f}%)")
    print(f"  -> no_contours         : {stats['kept_no_contours']}")
    print(f"  -> content_too_small   : {stats['kept_content_too_small']}")
    print(f"  -> center_shift_large  : {stats['kept_center_shift_too_large']}")
    print(f"  -> new_too_small       : {stats['kept_new_too_small']}")
    print(f"  -> crop_empty/otro     : {stats['kept_crop_empty'] + stats['kept_other']}")

    if delta_centers:
        print()
        print(f"Desplazamiento centro (px)")
        print(f"  mean: {np.mean(delta_centers):.2f} | "
              f"p50: {np.percentile(delta_centers, 50):.2f} | "
              f"p95: {np.percentile(delta_centers, 95):.2f} | "
              f"max: {np.max(delta_centers):.2f}")
    if area_ratios:
        print()
        print(f"Ratio area nueva/original")
        print(f"  mean: {np.mean(area_ratios):.3f} | "
              f"p5: {np.percentile(area_ratios, 5):.3f} | "
              f"p95: {np.percentile(area_ratios, 95):.3f}")
        n_shrunk = sum(1 for r in area_ratios if r < 0.80)
        n_grown  = sum(1 for r in area_ratios if r > 1.20)
        print(f"  encogidas >20%: {n_shrunk} | crecidas >20%: {n_grown}")

    print()
    print(f"JSON guardado en         : {output_json}")
    print(f"Debug PNGs guardados     : {saved}/{len(selected)}")
    print(f"Carpeta debug            : {examples_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    refine_dataset_bboxes(
        input_json="data/annotations/instances_train_clean.json",
        output_json="data/annotations/instances_train_refined.json",
        images_dir="data/images/train",
        # --- tuning de refinamiento ---
        threshold_percentile=65.0,   # percentil para binarizar el crop
        padding=2,                   # px extra alrededor del contenido encontrado
        min_content_ratio=0.05,      # el contenido brillante debe ser >= 5% del area original
        max_center_shift_ratio=0.40, # el centro no puede moverse mas del 40% de la dim. mayor
        min_box_dim=2.0,             # caja refinada debe tener al menos 2px en cada eje
        morph_close_px=3,            # cierre morfologico de 3px para unir fragmentos
        # --- debug ---
        examples_dir="outputs/bbox_refine_debug",
        max_example_images=30,
        max_boxes_per_example=200,
    )
