from __future__ import annotations

import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

MIN_BOX_DIM = 0.001
DEFAULT_MICRO_AREA_MAX = 5000.0

def compute_iou(boxA, boxB):
    # box: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, float(xB - xA)) * max(0, float(yB - yA))
    if interArea == 0:
        return 0.0
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def parse_args():
    parser = argparse.ArgumentParser(description="Generador de Pseudo-Labels Híbrido")
    parser.add_argument("--models", nargs='+', required=True, help="Lista de modelo,imgsz. Ej: run0.pt,512")
    parser.add_argument("--input-json", default="data/annotations/instances_train_og.json")
    parser.add_argument("--output-json", default="data/annotations/instances_train_pseudo.json")
    parser.add_argument("--train-images-dir", default="data/images/train")
    parser.add_argument("--conf-thresh", type=float, default=0.15, help="Confianza mínima del Ensemble")
    parser.add_argument(
        "--micro-area-max",
        type=float,
        default=DEFAULT_MICRO_AREA_MAX,
        help="Area maxima (px^2) para considerar una caja del ensemble como micro-caja.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    
    in_json_path = project_root / args.input_json
    out_json_path = project_root / args.output_json
    img_dir = project_root / args.train_images_dir

    print(f"\n[Pseudo-Labeler] Cargando {len(args.models)} modelos...")
    models_info = []
    for m in args.models:
        path, size = m.split(',')
        models_info.append((YOLO(path), int(size)))

    print("[Pseudo-Labeler] Cargando JSON original...")
    with open(in_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Agrupar anotaciones originales por image_id
    orig_anns_by_img = {}
    for ann in coco.get("annotations", []):
        orig_anns_by_img.setdefault(ann["image_id"], []).append(ann)

    new_annotations = []
    ann_id_counter = 1
    
    stats = {"ensemble_added": 0, "ensemble_discarded": 0, "human_kept": 0}

    print("\n[Pseudo-Labeler] Procesando imágenes y fusionando etiquetas...")
    for img_meta in tqdm(coco.get("images", [])):
        img_id = img_meta["id"]
        fname = img_meta["file_name"]
        img_w, img_h = img_meta["width"], img_meta["height"]
        img_path = img_dir / fname
        
        if not img_path.exists():
            continue

        # 1. Inferencia del Ensemble
        m_boxes_list, m_scores_list, m_labels_list = [], [], []
        for model, imgsz in models_info:
            results = model.predict(
                source=str(img_path), conf=0.005, iou=0.7, imgsz=imgsz, 
                max_det=500, augment=False, device=args.device, verbose=False
            )
            b, s, l = [], [], []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    b.append([x1, y1, x2, y2])
                    s.append(float(box.conf[0]))
                    l.append(0)
            m_boxes_list.append(b)
            m_scores_list.append(s)
            m_labels_list.append(l)

        # WBF
        ens_boxes, ens_scores, _ = weighted_boxes_fusion(
            m_boxes_list, m_scores_list, m_labels_list,
            weights=[1]*len(models_info), iou_thr=0.6, skip_box_thr=0.01
        )

        # Convertir a absolutos y filtrar por confianza
        final_ens_boxes = []
        for i in range(len(ens_boxes)):
            if ens_scores[i] < args.conf_thresh:
                continue
            nx1, ny1, nx2, ny2 = ens_boxes[i]
            x1, y1 = nx1 * img_w, ny1 * img_h
            w, h = max(0.0, (nx2 - nx1) * img_w), max(0.0, (ny2 - ny1) * img_h)
            if w > MIN_BOX_DIM and h > MIN_BOX_DIM:
                final_ens_boxes.append([float(x1), float(y1), float(w), float(h)])

        # 2. Conservar SIEMPRE todas las cajas humanas (incluye gigantes/verticales)
        orig_anns = orig_anns_by_img.get(img_id, [])
        for o_ann in orig_anns:
            orig_box = o_ann["bbox"]
            if len(orig_box) < 4:
                continue
            if orig_box[2] <= MIN_BOX_DIM or orig_box[3] <= MIN_BOX_DIM:
                continue
            cloned_ann = dict(o_ann)
            cloned_ann["id"] = ann_id_counter
            cloned_ann["source"] = "human_kept"
            new_annotations.append(cloned_ann)
            ann_id_counter += 1
            stats["human_kept"] += 1

        # 3. Inyectar SOLO micro-cajas del ensemble no cubiertas por humano
        for e_box in final_ens_boxes:
            e_area = e_box[2] * e_box[3]
            if e_area > args.micro_area_max:
                stats["ensemble_discarded"] += 1
                continue

            # Si ya hay cobertura humana, no la inyectamos.
            max_iou = 0.0
            for o_ann in orig_anns:
                orig_box = o_ann["bbox"]
                if len(orig_box) < 4:
                    continue
                iou = compute_iou(orig_box, e_box)
                if iou > max_iou:
                    max_iou = iou

            if max_iou < 0.3:
                new_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": e_box,
                    "area": e_area,
                    "iscrowd": 0,
                    "source": "ensemble_micro_added",
                })
                ann_id_counter += 1
                stats["ensemble_added"] += 1
            else:
                stats["ensemble_discarded"] += 1

    # Guardar nuevo JSON
    coco["annotations"] = new_annotations
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print("\n" + "="*40)
    print("RESUMEN DEL SÚPER-DATASET")
    print(f"Cajas Humanas conservadas (incl. gigantes): {stats['human_kept']}")
    print(f"Micro-cajas nuevas del Ensemble:            {stats['ensemble_added']}")
    print(f"Cajas del Ensemble descartadas:             {stats['ensemble_discarded']}")
    print(f"Total Anotaciones: {len(new_annotations)}")
    print("="*40)
    print(f"Guardado en: {out_json_path}")

if __name__ == "__main__":
    main()