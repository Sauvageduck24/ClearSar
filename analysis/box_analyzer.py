import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def compute_iou(boxA, boxB):
    # box: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, float(xB - xA)) * max(0, float(yB - yA))
    if interArea == 0: return 0.0
    return interArea / float((boxA[2]*boxA[3]) + (boxB[2]*boxB[3]) - interArea)

def compute_iop(pred_box, gt_box):
    # pred_box y gt_box: [x, y, w, h]
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
    yB = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])
    
    interArea = max(0, float(xB - xA)) * max(0, float(yB - yA))
    if interArea == 0: 
        return 0.0
        
    predArea = pred_box[2] * pred_box[3]
    
    # Devolvemos qué porcentaje de nuestra predicción está DENTRO de la caja humana
    return interArea / float(predArea)

def main():
    submissions_dir = Path(__file__).resolve().parent
    project_root = submissions_dir.parent
    
    # 1. Configuración (Ajusta la ruta de tu mejor modelo aquí)
    model_path = submissions_dir / "run0.pt" # Cambia al modelo que prefieras
    holdout_yaml_dir = project_root / "data" / "yolo" / "holdout" / "images" / "val"
    gt_json_path = project_root / "data" / "annotations" / "instances_train_og.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo: {model_path}")
    if not holdout_yaml_dir.is_dir():
        raise FileNotFoundError(f"No existe carpeta holdout: {holdout_yaml_dir}")
    if not gt_json_path.exists():
        raise FileNotFoundError(f"No existe GT json: {gt_json_path}")
    
    print("Cargando modelo y Ground Truth...")
    model = YOLO(str(model_path))
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        coco_gt = json.load(f)

    # Crear diccionario de Ground Truth por nombre de archivo
    gt_by_filename = {}
    id_to_filename = {img['id']: img['file_name'] for img in coco_gt['images']}
    for ann in coco_gt['annotations']:
        fname = id_to_filename.get(ann['image_id'])
        if fname:
            gt_by_filename.setdefault(fname, []).append(ann['bbox']) # [x, y, w, h]

    image_files = list(holdout_yaml_dir.glob("*.*"))
    
    stats_horizontal = {'w_ratio': [], 'h_ratio': [], 'area_ratio': []}
    stats_vertical = {'w_ratio': [], 'h_ratio': [], 'area_ratio': []}
    stats_small = {'w_ratio': [], 'h_ratio': [], 'area_ratio': []}

    print("Emparejando predicciones con el humano...")
    for img_path in tqdm(image_files):
        fname = img_path.name
        gt_boxes = gt_by_filename.get(fname, [])
        if not gt_boxes: continue

        # Inferencia
        results = model.predict(source=str(img_path), imgsz=512, conf=0.25, iou=0.6, verbose=False)
        if not results or results[0].boxes is None: continue

        pred_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_boxes.append([x1, y1, x2-x1, y2-y1]) # Convertir a [x,y,w,h] absolutos

        # Emparejar cada caja predicha con la mejor del humano
        for p_box in pred_boxes:
            p_w, p_h = p_box[2], p_box[3]
            if p_w == 0 or p_h == 0: continue
            
            best_iop = 0
            best_gt = None
            
            for g_box in gt_boxes:
                iop = compute_iop(p_box, g_box) # Usamos la nueva función
                if iop > best_iop:
                    best_iop = iop
                    best_gt = g_box
            
            # Si al menos el 50% de nuestra raya está dentro de la caja del humano...
            if best_iop > 0.50 and best_gt is not None:
                g_w, g_h = best_gt[2], best_gt[3]
                
                w_ratio = g_w / p_w
                h_ratio = g_h / p_h
                area_ratio = (g_w * g_h) / (p_w * p_h)
                
                # Clasificar el tipo de caja que vio TU MODELO
                if (p_w * p_h) < 200:
                    stats_small['w_ratio'].append(w_ratio)
                    stats_small['h_ratio'].append(h_ratio)
                elif p_w > (p_h * 1.5): # Es claramente horizontal
                    stats_horizontal['w_ratio'].append(w_ratio)
                    stats_horizontal['h_ratio'].append(h_ratio)
                elif p_h > (p_w * 1.5): # Es claramente vertical
                    stats_vertical['w_ratio'].append(w_ratio)
                    stats_vertical['h_ratio'].append(h_ratio)

    # Imprimir resultados
    print("\n" + "="*50)
    print("ANÁLISIS ESTADÍSTICO: CÓMO DIBUJA EL HUMANO vs TU MODELO")
    print("="*50)
    
    def print_stats(name, stats_dict):
        if not stats_dict['w_ratio']: return
        print(f"\n[Cuando el modelo ve una caja {name.upper()}]")
        print(f"Muestras analizadas: {len(stats_dict['w_ratio'])}")
        # Usamos la mediana en lugar de la media para ignorar casos extremos raros
        print(f"-> El humano dibujó la ANCHURA  x{np.median(stats_dict['w_ratio']):.2f} veces más grande.")
        print(f"-> El humano dibujó la ALTURA   x{np.median(stats_dict['h_ratio']):.2f} veces más grande.")

    print_stats("Horizontal (Rayas finas anchas)", stats_horizontal)
    print_stats("Vertical (Rayas finas altas)", stats_vertical)
    print_stats("Pequeña (Micro-ruido)", stats_small)
    print("="*50)

if __name__ == "__main__":
    main()


#  & C:\Users\esteb\.conda\envs\clearsar\python.exe -m submissions.box_analyzer