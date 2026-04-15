import json
import cv2
import numpy as np
from pathlib import Path
import math
from tqdm import tqdm


def _to_uint8_for_viz(img_gray: np.ndarray) -> np.ndarray:
    """Convierte cualquier imagen en escala de grises a uint8 para visualizacion."""
    if img_gray.dtype == np.uint8:
        return img_gray
    if img_gray.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    mn = float(np.min(img_gray))
    mx = float(np.max(img_gray))
    if mx <= mn:
        return np.zeros_like(img_gray, dtype=np.uint8)
    img_norm = (img_gray.astype(np.float32) - mn) / (mx - mn)
    return np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)


def _draw_example_image(
    image_path: Path,
    boxes_info: list,
    output_path: Path,
    image_title: str
):
    """Dibuja ejemplos de cajas sobre la imagen y guarda un PNG de depuracion."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    img_viz = cv2.cvtColor(_to_uint8_for_viz(img), cv2.COLOR_GRAY2BGR)
    h, w = img.shape

    color_map = {
        "aceptada": (40, 210, 40),      # verde
        "rechazada": (40, 40, 230),     # rojo (BGR)
        "normal": (255, 170, 20),       # naranja
    }

    for b in boxes_info:
        x, y, bw, bh = b["bbox"]
        x1 = max(0, min(w - 1, int(round(x))))
        y1 = max(0, min(h - 1, int(round(y))))
        x2 = max(0, min(w - 1, int(round(x + bw))))
        y2 = max(0, min(h - 1, int(round(y + bh))))
        if x2 <= x1 or y2 <= y1:
            continue

        label = b.get("label", "normal")
        color = color_map.get(label, (255, 255, 255))
        cv2.rectangle(img_viz, (x1, y1), (x2, y2), color, 1)

    # Leyenda
    cv2.rectangle(img_viz, (5, 5), (380, 78), (0, 0, 0), -1)
    cv2.putText(img_viz, image_title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img_viz, "verde=aceptada | rojo=rechazada | naranja=normal", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img_viz, f"cajas dibujadas={len(boxes_info)}", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), img_viz)

def process_and_filter_dataset(
    input_json: str, 
    output_json: str, 
    images_dir: str,
    brightness_threshold: float = 0.8, # Porcentaje del percentil 95
    examples_dir: str = "outputs/bbox_debug_examples",
    max_example_images: int = 12,
    max_boxes_per_example: int = 160
):
    print(f"Cargando dataset: {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)
        
    # Crear un diccionario para acceder rápido a los nombres de archivo por image_id
    img_dict = {img['id']: img['file_name'] for img in coco['images']}
    images_path = Path(images_dir)
    
    new_annotations = []
    ann_id = 1
    
    stats = {
        'originales': len(coco.get('annotations', [])),
        'imagenes_en_json': len(coco.get('images', [])),
        'imagenes_con_anotaciones': 0,
        'imagenes_procesadas': 0,
        'imagenes_sin_archivo': 0,
        'imagenes_no_leidas': 0,
        'cajas_no_verticales': 0,
        'verticales_rotas': 0,
        'slices_generados': 0,
        'slices_rechazados_visual': 0,
        'slices_aceptados': 0,
        'crops_vacios': 0,
        'bbox_area_min': float('inf'),
        'bbox_area_max': 0.0,
        'bbox_area_sum': 0.0,
        'bbox_ratio_min': float('inf'),
        'bbox_ratio_max': 0.0,
        'bbox_ratio_sum': 0.0
    }

    p95_aceptados = []
    p95_rechazados = []
    per_image_debug = []

    # Agrupar anotaciones por imagen para no abrir la misma imagen 100 veces
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    print("Procesando imágenes y evaluando contraste visual...")
    stats['imagenes_con_anotaciones'] = len(anns_by_img)
    for image_id, anns in tqdm(anns_by_img.items()):
        file_name = img_dict.get(image_id)
        if not file_name:
            continue
            
        img_file = images_path / file_name
        if not img_file.exists():
            stats['imagenes_sin_archivo'] += 1
            continue
            
        # Leer la imagen en escala de grises (suficiente para evaluar brillo/contraste)
        # Asegúrate de usar cv2.IMREAD_UNCHANGED si son TIFFs de 16 bits
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats['imagenes_no_leidas'] += 1
            continue

        stats['imagenes_procesadas'] += 1
            
        img_h, img_w = img.shape
        image_debug_boxes = []
        img_stats = {
            'image_id': image_id,
            'file_name': file_name,
            'total_anns': len(anns),
            'verticales': 0,
            'normales': 0,
            'slices_generados': 0,
            'slices_aceptados': 0,
            'slices_rechazados': 0
        }

        for ann in anns:
            x, y, w, h = ann["bbox"]
            area = float(max(w, 0.0) * max(h, 0.0))
            ratio = float(h / max(w, 0.1))
            stats['bbox_area_min'] = min(stats['bbox_area_min'], area)
            stats['bbox_area_max'] = max(stats['bbox_area_max'], area)
            stats['bbox_area_sum'] += area
            stats['bbox_ratio_min'] = min(stats['bbox_ratio_min'], ratio)
            stats['bbox_ratio_max'] = max(stats['bbox_ratio_max'], ratio)
            stats['bbox_ratio_sum'] += ratio
            
            # Limitar coordenadas a los bordes de la imagen
            cx1 = max(0, int(x))
            cy1 = max(0, int(y))
            cx2 = min(img_w, int(x + w))
            cy2 = min(img_h, int(y + h))
            
            # Extraer el parche visual de la CAJA ENTERA
            crop = img[cy1:cy2, cx1:cx2]
            
            if crop.size == 0:
                stats['crops_vacios'] += 1
                continue
                
            # Calcular P95 de la caja completa
            p95_brightness = float(np.percentile(crop, 95))
            max_possible_val = 255.0 if img.dtype == np.uint8 else float(np.max(img))
            decision_threshold = max_possible_val * brightness_threshold
            
            if p95_brightness >= decision_threshold:
                # La caja completa contiene al menos algo de RFI brillante
                new_ann = ann.copy()
                new_ann["id"] = ann_id
                new_annotations.append(new_ann)
                ann_id += 1
                
                stats['slices_aceptados'] += 1
                img_stats['slices_aceptados'] += 1
                p95_aceptados.append(p95_brightness)

                if len(image_debug_boxes) < max_boxes_per_example:
                    image_debug_boxes.append({
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "label": "aceptada",
                        "p95": p95_brightness,
                        "threshold": decision_threshold
                    })
            else:
                # La caja completa es oscuridad/ruido. La destruimos entera.
                stats['slices_rechazados_visual'] += 1
                img_stats['slices_rechazados'] += 1
                p95_rechazados.append(p95_brightness)

                if len(image_debug_boxes) < max_boxes_per_example:
                    image_debug_boxes.append({
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "label": "rechazada",
                        "p95": p95_brightness,
                        "threshold": decision_threshold
                    })

        if image_debug_boxes:
            per_image_debug.append({
                "image_id": image_id,
                "file_name": file_name,
                "boxes_info": image_debug_boxes,
                "stats": img_stats
            })

    coco["annotations"] = new_annotations
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    examples_path = Path(examples_dir)
    examples_path.mkdir(parents=True, exist_ok=True)

    bad_candidates = [x for x in per_image_debug if x['stats']['slices_rechazados'] > 0]
    good_candidates = [x for x in per_image_debug if x['stats']['slices_rechazados'] == 0 and (x['stats']['slices_aceptados'] > 0 or x['stats']['normales'] > 0)]

    bad_candidates = sorted(
        bad_candidates,
        key=lambda x: (x['stats']['slices_rechazados'], x['stats']['slices_generados']),
        reverse=True
    )
    good_candidates = sorted(
        good_candidates,
        key=lambda x: (x['stats']['slices_aceptados'], x['stats']['normales']),
        reverse=True
    )

    n_bad = min(len(bad_candidates), max_example_images // 2)
    n_good = min(len(good_candidates), max_example_images - n_bad)
    if n_bad + n_good < max_example_images and len(bad_candidates) > n_bad:
        extra = min(max_example_images - (n_bad + n_good), len(bad_candidates) - n_bad)
        n_bad += extra

    selected_examples = bad_candidates[:n_bad] + good_candidates[:n_good]

    saved_examples = 0
    for idx, item in enumerate(selected_examples, start=1):
        img_path = images_path / item['file_name']
        out_file = examples_path / f"example_{idx:02d}_{Path(item['file_name']).stem}.png"

        st = item['stats']
        title = (
            f"{item['file_name']} | gen={st['slices_generados']} "
            f"acc={st['slices_aceptados']} rej={st['slices_rechazados']} norm={st['normales']}"
        )
        ok = _draw_example_image(
            image_path=img_path,
            boxes_info=item['boxes_info'],
            output_path=out_file,
            image_title=title
        )
        if ok:
            saved_examples += 1

    total_originales = max(stats['originales'], 1)
    total_slices = max(stats['slices_generados'], 1)

    area_min = 0.0 if stats['bbox_area_min'] == float('inf') else stats['bbox_area_min']
    ratio_min = 0.0 if stats['bbox_ratio_min'] == float('inf') else stats['bbox_ratio_min']
    area_mean = stats['bbox_area_sum'] / total_originales
    ratio_mean = stats['bbox_ratio_sum'] / total_originales

    p95_acc_mean = float(np.mean(p95_aceptados)) if p95_aceptados else 0.0
    p95_rej_mean = float(np.mean(p95_rechazados)) if p95_rechazados else 0.0

    top_bad = bad_candidates[:5]
        
    print("\n--- RESUMEN DE LA CIRUGÍA VISUAL ---")
    print(f"Cajas originales totales : {stats['originales']}")
    print(f"Imagenes en JSON         : {stats['imagenes_en_json']}")
    print(f"Imagenes c/anotaciones   : {stats['imagenes_con_anotaciones']}")
    print(f"Imagenes procesadas      : {stats['imagenes_procesadas']}")
    print(f"Imagenes sin archivo     : {stats['imagenes_sin_archivo']}")
    print(f"Imagenes no leidas       : {stats['imagenes_no_leidas']}")
    print(f"Cajas no verticales      : {stats['cajas_no_verticales']}")
    print(f"Cajas gigantes procesadas: {stats['verticales_rotas']}")
    print(f"Sub-cajas generadas      : {stats['slices_generados']}")
    print(f"  -> Aceptadas (RFI real): {stats['slices_aceptados']}")
    print(f"  -> Rechazadas (Fondo)  : {stats['slices_rechazados_visual']}")
    print(f"  -> Crops vacios        : {stats['crops_vacios']}")
    print(f"Tasa aceptacion slices   : {100.0 * stats['slices_aceptados'] / total_slices:.2f}%")
    print(f"Tasa rechazo slices      : {100.0 * stats['slices_rechazados_visual'] / total_slices:.2f}%")
    print(f"Area bbox (min/mean/max) : {area_min:.2f} / {area_mean:.2f} / {stats['bbox_area_max']:.2f}")
    print(f"Ratio h/w (min/mean/max) : {ratio_min:.3f} / {ratio_mean:.3f} / {stats['bbox_ratio_max']:.3f}")
    print(f"P95 mean aceptadas       : {p95_acc_mean:.2f}")
    print(f"P95 mean rechazadas      : {p95_rej_mean:.2f}")
    print(f"Total cajas final en JSON: {len(new_annotations)}")
    print(f"Guardado en              : {output_json}")
    print(f"Ejemplos visuales guard. : {saved_examples}/{len(selected_examples)}")
    print(f"Carpeta ejemplos         : {examples_dir}")

    if top_bad:
        print("\nTop 5 imagenes con mas rechazos:")
        for item in top_bad:
            s = item['stats']
            print(
                f"  - {item['file_name']}: rech={s['slices_rechazados']} "
                f"acc={s['slices_aceptados']} gen={s['slices_generados']}"
            )

if __name__ == "__main__":
    # ¡Ajusta estas rutas a tu proyecto!
    process_and_filter_dataset(
        input_json="data/annotations/instances_train.json",
        output_json="data/annotations/instances_train_clean.json",
        images_dir="data/images/train",  # Necesario para que OpenCV lea la imagen
        brightness_threshold=0.2, # Requiere que el percentil 95 tenga al menos el 20% del brillo maximo
        examples_dir="outputs/bbox_debug_examples",
        max_example_images=12,
        max_boxes_per_example=300
    )