"""
Ensemble de múltiples submissions de detección de objetos.
Combina predicciones de varios modelos usando diferentes estrategias de ensemble.
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar
from collections import defaultdict

import numpy as np
from torchvision.ops import box_iou, nms
import torch


T = TypeVar("T")


def progress_bar(
    iterable: Iterable[T],
    desc: str,
    total: Optional[int] = None,
    enabled: bool = True,
) -> Iterable[T]:
    """Devuelve un iterable con barra de progreso si tqdm está disponible."""
    if not enabled:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, desc=desc, total=total, unit="it")


def load_submission(zip_path: Path) -> List[Dict]:
    """Carga un submission desde un archivo ZIP.
    
    Args:
        zip_path: Ruta al archivo ZIP que contiene submission.json
        
    Returns:
        Lista de detecciones con estructura: 
        [{image_id, category_id, bbox, score}, ...]
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Buscar archivo JSON en el ZIP (podría tener diferentes nombres)
        json_files = [f for f in z.namelist() if f.endswith('.json')]
        if not json_files:
            raise ValueError(f"No se encontró archivo JSON en {zip_path}")
        
        json_file = json_files[0]
        with z.open(json_file) as f:
            return json.load(f)


def nms_detections(
    detections: List[Dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    show_progress: bool = True,
) -> List[Dict]:
    """Aplica NMS a un conjunto de detecciones.
    
    Args:
        detections: Lista de detecciones
        iou_threshold: Umbral de IoU para NMS
        score_threshold: Score mínimo para mantener una detección
        
    Returns:
        Lista de detecciones después de NMS
    """
    if not detections:
        return []
    
    # Agrupar por imagen
    by_image = defaultdict(list)
    for det in detections:
        by_image[det["image_id"]].append(det)
    
    result = []
    image_items = progress_bar(
        by_image.items(),
        desc="NMS por imagen",
        total=len(by_image),
        enabled=show_progress,
    )

    for image_id, image_dets in image_items:
        if not image_dets:
            continue
            
        # Convertir a tensores
        boxes = torch.tensor(
            [[d["bbox"][0], d["bbox"][1], 
              d["bbox"][0] + d["bbox"][2], 
              d["bbox"][1] + d["bbox"][3]] for d in image_dets],
            dtype=torch.float32
        )
        scores = torch.tensor([d["score"] for d in image_dets], dtype=torch.float32)
        
        # Aplicar NMS
        keep_idx = nms(boxes, scores, iou_threshold)
        
        # Mantener detecciones después de NMS
        for idx in keep_idx:
            det = image_dets[idx.item()]
            if det["score"] >= score_threshold:
                result.append(det)
    
    return result


def filter_detections(
    detections: List[Dict],
    min_score: float = 0.0,
    max_dets_per_image: Optional[int] = None,
) -> List[Dict]:
    """Filtra detecciones por score y limita cantidad por imagen."""
    if min_score <= 0.0 and max_dets_per_image is None:
        return detections

    by_image = defaultdict(list)
    for det in detections:
        if det["score"] >= min_score:
            by_image[det["image_id"]].append(det)

    filtered = []
    for image_dets in by_image.values():
        image_dets.sort(key=lambda d: d["score"], reverse=True)
        if max_dets_per_image is not None:
            image_dets = image_dets[:max_dets_per_image]
        filtered.extend(image_dets)

    return filtered


def ensemble_average(
    submissions: List[List[Dict]],
    nms_iou: float = 0.5,
    show_progress: bool = True,
) -> List[Dict]:
    """Ensemble por promedio de scores. Las detecciones similares se promedian.
    
    Args:
        submissions: Lista de submissions (cada uno es una lista de detecciones)
        nms_iou: Umbral de IoU para agrupar detecciones similares
        
    Returns:
        Submission con detecciones promediadas
    """
    # Agrupar por imagen
    by_image = defaultdict(list)
    for submission in submissions:
        for det in submission:
            by_image[det["image_id"]].append(det)
    
    result = []
    
    image_items = progress_bar(
        by_image.items(),
        desc="Ensembling por imagen",
        total=len(by_image),
        enabled=show_progress,
    )

    for image_id, all_dets in image_items:
        if not all_dets:
            continue
        
        # Convertir a tensores
        boxes = torch.tensor(
            [[d["bbox"][0], d["bbox"][1], 
              d["bbox"][0] + d["bbox"][2], 
              d["bbox"][1] + d["bbox"][3]] for d in all_dets],
            dtype=torch.float32
        )
        boxes_xywh = torch.tensor([d["bbox"] for d in all_dets], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in all_dets], dtype=torch.float32)
        
        # Aplicar NMS para encontrar detecciones similares
        keep_idx = nms(boxes, scores, nms_iou)
        
        # Matriz IoU vectorizada para evitar bucles Python O(N^2)
        iou_matrix = box_iou(boxes, boxes)
        similarity_threshold = nms_iou * 0.5

        # Para cada detección que se mantiene, promediar todas las similares
        for idx in keep_idx.tolist():
            similar_mask = iou_matrix[idx] > similarity_threshold
            similar_idx = torch.where(similar_mask)[0]

            if len(similar_idx) == 0:
                similar_idx = torch.tensor([idx], dtype=torch.long)

            avg_bbox = boxes_xywh[similar_idx].mean(dim=0)
            avg_score = scores[similar_idx].mean()
            
            result.append({
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [
                    float(avg_bbox[0]),
                    float(avg_bbox[1]),
                    float(avg_bbox[2]),
                    float(avg_bbox[3]),
                ],
                "score": float(avg_score.item()),
            })
    
    return result


def ensemble_weighted(
    submissions: List[Tuple[List[Dict], float]],
    nms_iou: float = 0.5,
    show_progress: bool = True,
) -> List[Dict]:
    """Ensemble ponderado por weights. Cada submission puede tener un peso diferente.
    
    Args:
        submissions: Lista de tuplas (submission, weight)
        nms_iou: Umbral de IoU para NMS
        
    Returns:
        Submission con detecciones ponderadas
    """
    # Aplicar pesos a los scores
    weighted_dets = []
    for submission, weight in submissions:
        for det in submission:
            det_copy = det.copy()
            det_copy["score"] *= weight
            weighted_dets.append(det_copy)
    
    # Usar ensemble_average sobre las detecciones ponderadas
    return ensemble_average([weighted_dets], nms_iou=nms_iou, show_progress=show_progress)


def ensemble_nms(
    submissions: List[List[Dict]],
    iou_threshold: float = 0.5,
    show_progress: bool = True,
) -> List[Dict]:
    """Ensemble simple: combinar todos y aplicar NMS.
    
    Args:
        submissions: Lista de submissions
        iou_threshold: Umbral de IoU para NMS
        
    Returns:
        Submission después de NMS
    """
    # Combinar todas las detecciones
    all_dets = []
    for submission in submissions:
        all_dets.extend(submission)
    
    return nms_detections(all_dets, iou_threshold=iou_threshold, show_progress=show_progress)


def ensemble_wbf(
    submissions: List[List[Dict]],
    weights: Optional[List[float]] = None,
    iou_threshold: float = 0.55,
    skip_box_threshold: float = 0.0,
    show_progress: bool = True,
) -> List[Dict]:
    """Ensemble con Weighted Box Fusion.

    Nota: WBF requiere coordenadas normalizadas [0, 1].
    Para no depender de metadata externa, se normaliza por escala por imagen.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as exc:
        raise ImportError(
            "Falta dependencia 'ensemble-boxes'. Instala con: pip install ensemble-boxes"
        ) from exc

    by_image_and_submission: Dict[int, List[List[Dict]]] = defaultdict(list)
    for submission in submissions:
        grouped = defaultdict(list)
        for det in submission:
            grouped[det["image_id"]].append(det)

        for image_id, dets in grouped.items():
            by_image_and_submission[image_id].append(dets)

    result: List[Dict] = []
    image_items = progress_bar(
        by_image_and_submission.items(),
        desc="WBF por imagen",
        total=len(by_image_and_submission),
        enabled=show_progress,
    )

    for image_id, image_submissions in image_items:
        if not image_submissions:
            continue

        max_x2 = 1.0
        max_y2 = 1.0
        for dets in image_submissions:
            for det in dets:
                x1, y1, w, h = det["bbox"]
                max_x2 = max(max_x2, x1 + w)
                max_y2 = max(max_y2, y1 + h)

        boxes_list: List[List[List[float]]] = []
        scores_list: List[List[float]] = []
        labels_list: List[List[int]] = []

        for dets in image_submissions:
            model_boxes: List[List[float]] = []
            model_scores: List[float] = []
            model_labels: List[int] = []

            for det in dets:
                x1, y1, w, h = det["bbox"]
                x2 = x1 + w
                y2 = y1 + h

                model_boxes.append(
                    [
                        float(x1 / max_x2),
                        float(y1 / max_y2),
                        float(x2 / max_x2),
                        float(y2 / max_y2),
                    ]
                )
                model_scores.append(float(det["score"]))
                model_labels.append(int(det.get("category_id", 1)))

            boxes_list.append(model_boxes)
            scores_list.append(model_scores)
            labels_list.append(model_labels)

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_threshold,
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1 = float(box[0] * max_x2)
            y1 = float(box[1] * max_y2)
            x2 = float(box[2] * max_x2)
            y2 = float(box[3] * max_y2)

            result.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )

    return result


def save_submission(detections: List[Dict], output_path: Path) -> None:
    """Guarda las detecciones en un archivo ZIP con JSON.
    
    Args:
        detections: Lista de detecciones
        output_path: Ruta donde guardar el archivo ZIP
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as z:
        json_data = json.dumps(detections, indent=2)
        z.writestr('submission.json', json_data)
    
    print(f"✅ Submission guardado en: {output_path}")


def print_stats(detections: List[Dict], name: str = "Submission") -> None:
    """Imprime estadísticas del submission.
    
    Args:
        detections: Lista de detecciones
        name: Nombre del submission para imprimir
    """
    if not detections:
        print(f"{name}: 0 detecciones")
        return
    
    # Agrupar por imagen
    by_image = defaultdict(list)
    for det in detections:
        by_image[det["image_id"]].append(det)
    
    num_images = len(by_image)
    total_dets = len(detections)
    avg_dets = total_dets / num_images if num_images > 0 else 0
    avg_score = np.mean([d["score"] for d in detections])
    
    print(f"\n{name}:")
    print(f"  Imágenes: {num_images}")
    print(f"  Total detecciones: {total_dets}")
    print(f"  Promedio por imagen: {avg_dets:.2f}")
    print(f"  Score promedio: {avg_score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble de múltiples submissions de detección",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Ensemble simple con NMS:
   python ensemble_submissions.py --method nms \\
     --submissions sub1.zip sub2.zip sub3.zip \\
     --output ensemble.zip

2. Ensemble con promedio de scores:
   python ensemble_submissions.py --method average \\
     --submissions sub1.zip sub2.zip sub3.zip \\
     --output ensemble.zip

3. Ensemble ponderado:
   python ensemble_submissions.py --method weighted \\
     --submissions sub1.zip sub2.zip sub3.zip \\
     --weights 0.5 0.3 0.2 \\
     --output ensemble.zip
        """,
    )
    
    parser.add_argument(
        "--submissions",
        type=str,
        nargs="+",
        required=True,
        help="Rutas a los archivos ZIP de submissions",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="ensemble_submission.zip",
        help="Ruta del archivo de salida (default: ensemble_submission.zip)",
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="average",
        choices=["nms", "average", "weighted", "wbf"],
        help="Método de ensemble (default: average)",
    )
    
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="Umbral de IoU para NMS (default: 0.5)",
    )
    
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Pesos para cada submission (con --method weighted o --method wbf)",
    )

    parser.add_argument(
        "--wbf-iou",
        type=float,
        default=0.55,
        help="Umbral de IoU para WBF (default: 0.55)",
    )

    parser.add_argument(
        "--wbf-skip-box-thr",
        type=float,
        default=0.0,
        help="Score mínimo para considerar una caja en WBF (default: 0.0)",
    )
    
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="No mostrar estadísticas",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Desactiva barras de progreso",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Filtra detecciones con score menor a este valor antes del ensemble",
    )

    parser.add_argument(
        "--max-dets-per-image",
        type=int,
        default=None,
        help="Limita detecciones por imagen y submission (top score)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    show_progress = not args.no_progress
    
    # Validar argumentos
    if args.method in {"weighted", "wbf"} and args.weights is None:
        raise ValueError("--weights es requerido cuando --method=weighted o --method=wbf")
    
    if args.method in {"weighted", "wbf"} and len(args.weights) != len(args.submissions):
        raise ValueError(
            f"El número de pesos ({len(args.weights)}) debe coincidir "
            f"con el número de submissions ({len(args.submissions)})"
        )
    
    # Normalizar pesos si se proporcionan
    if args.weights is not None:
        total_weight = sum(args.weights)
        args.weights = [w / total_weight for w in args.weights]
    
    print(f"Cargando {len(args.submissions)} submissions...")
    print("=" * 50)
    
    # Cargar submissions
    submissions = []
    submissions_iter = progress_bar(
        args.submissions,
        desc="Cargando submissions",
        total=len(args.submissions),
        enabled=show_progress,
    )

    for i, sub_path in enumerate(submissions_iter):
        sub = load_submission(Path(sub_path))
        sub = filter_detections(
            sub,
            min_score=args.min_score,
            max_dets_per_image=args.max_dets_per_image,
        )
        submissions.append(sub)
        if not args.no_stats:
            print_stats(sub, f"Submission {i+1} ({Path(sub_path).name})")
    
    print("\n" + "=" * 50)
    print(f"Ejecutando ensemble con método: {args.method}")
    print("=" * 50)
    
    # Ejecutar ensemble según el método
    if args.method == "nms":
        result = ensemble_nms(
            submissions,
            iou_threshold=args.nms_iou,
            show_progress=show_progress,
        )
    elif args.method == "average":
        result = ensemble_average(
            submissions,
            nms_iou=args.nms_iou,
            show_progress=show_progress,
        )
    elif args.method == "weighted":
        weighted_subs = list(zip(submissions, args.weights))
        result = ensemble_weighted(
            weighted_subs,
            nms_iou=args.nms_iou,
            show_progress=show_progress,
        )
    elif args.method == "wbf":
        result = ensemble_wbf(
            submissions,
            weights=args.weights,
            iou_threshold=args.wbf_iou,
            skip_box_threshold=args.wbf_skip_box_thr,
            show_progress=show_progress,
        )
    else:
        raise ValueError(f"Método desconocido: {args.method}")
    
    # Mostrar estadísticas finales
    if not args.no_stats:
        print_stats(result, "Ensemble Final")
    
    # Guardar resultado
    output_path = Path(args.output)
    save_submission(result, output_path)
    
    print(f"\n✨ Ensemble completado exitosamente")


if __name__ == "__main__":
    main()