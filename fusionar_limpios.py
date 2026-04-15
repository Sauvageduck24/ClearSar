import json
from pathlib import Path
from tqdm import tqdm

def compute_iou_and_ioa(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2

    inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1, area2 = w1 * h1, w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    min_area = min(area1, area2)
    ioa = inter_area / min_area if min_area > 0 else 0.0
    return iou, ioa

def merge_overlapping_boxes(anns, iou_thr=0.4, ioa_thr=0.8):
    if not anns: return []
    active_anns = anns.copy()
    merged_anns = []

    while active_anns:
        curr_ann = active_anns.pop(0)
        cx, cy, cw, ch = curr_ann["bbox"]
        c_x1, c_y1, c_x2, c_y2 = cx, cy, cx + cw, cy + ch
        
        to_remove = []
        for i, other_ann in enumerate(active_anns):
            ox, oy, ow, oh = other_ann["bbox"]
            iou, ioa = compute_iou_and_ioa([cx, cy, cw, ch], [ox, oy, ow, oh])
            
            if iou > iou_thr or ioa > ioa_thr:
                o_x1, o_y1, o_x2, o_y2 = ox, oy, ox + ow, oy + oh
                c_x1, c_y1 = min(c_x1, o_x1), min(c_y1, o_y1)
                c_x2, c_y2 = max(c_x2, o_x2), max(c_y2, o_y2)
                cx, cy, cw, ch = c_x1, c_y1, c_x2 - c_x1, c_y2 - c_y1
                curr_ann["bbox"] = [cx, cy, cw, ch]
                curr_ann["area"] = cw * ch
                to_remove.append(i)
                
        for i in reversed(to_remove):
            active_anns.pop(i)
        merged_anns.append(curr_ann)
    return merged_anns

def main():
    input_path = "data/annotations/instances_train_clean.json"
    output_path = "data/annotations/instances_train_final.json"
    
    print(f"Cargando {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    final_annotations = []
    print("Fusionando cajas superpuestas sin alterar geometria base...")
    for img_id, img_anns in tqdm(anns_by_img.items()):
        merged = merge_overlapping_boxes(img_anns, iou_thr=0.40, ioa_thr=0.80)
        final_annotations.extend(merged)

    print(f"\nResumen:")
    print(f"Cajas antes de fusionar : {len(coco.get('annotations', []))}")
    print(f"Cajas despues de fusionar: {len(final_annotations)}")

    coco["annotations"] = final_annotations
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"¡Guardado en {output_path}!")

if __name__ == "__main__":
    main()