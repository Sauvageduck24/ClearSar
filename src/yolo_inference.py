from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _parse_image_size(value: str) -> int | List[int]:
    raw = str(value).strip()
    if not raw:
        raise argparse.ArgumentTypeError("--image-size no puede estar vacio")

    if raw.isdigit():
        parsed = int(raw)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("--image-size debe ser > 0")
        return parsed

    cleaned = raw.strip("()[]")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        h, w = int(parts[0]), int(parts[1])
        if h <= 0 or w <= 0:
            raise argparse.ArgumentTypeError("--image-size requiere valores > 0")
        return [h, w]

    raise argparse.ArgumentTypeError(
        "Formato invalido para --image-size. Usa 640 o [512,1024] / (512,1024)."
    )


def _normalize_imgsz(imgsz: int | List[int]) -> tuple[int, int]:
    if isinstance(imgsz, list):
        if len(imgsz) != 2:
            raise ValueError("--image-size como lista debe tener exactamente 2 valores")
        return int(imgsz[0]), int(imgsz[1])
    return int(imgsz), int(imgsz)


def _resize_for_model(img_np: np.ndarray, imgsz: int | List[int]) -> np.ndarray:
    target_h, target_w = _normalize_imgsz(imgsz)
    pil_image = Image.fromarray(img_np)
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    return np.array(pil_image.resize((target_w, target_h), resample))


def merge_horizontal_boxes(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    row_tol_px: float = 4.0,
    gap_px: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores

    boxes = [list(map(float, box)) for box in boxes_xyxy]
    score_list = [float(score) for score in scores]
    order = sorted(range(len(boxes)), key=lambda idx: (boxes[idx][1], boxes[idx][0]))
    used = [False] * len(boxes)

    merged_boxes: List[List[float]] = []
    merged_scores: List[float] = []

    for i in order:
        if used[i]:
            continue

        x1, y1, x2, y2 = boxes[i]
        score = score_list[i]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in order:
                if used[j]:
                    continue
                jx1, jy1, jx2, jy2 = boxes[j]
                same_row = abs(jy1 - y1) <= row_tol_px and abs(jy2 - y2) <= row_tol_px
                close_horiz = (jx1 - x2) <= gap_px and (x1 - jx2) <= gap_px
                if same_row and close_horiz:
                    x1 = min(x1, jx1)
                    y1 = min(y1, jy1)
                    x2 = max(x2, jx2)
                    y2 = max(y2, jy2)
                    score = max(score, score_list[j])
                    used[j] = True
                    changed = True

        merged_boxes.append([x1, y1, x2, y2])
        merged_scores.append(score)

    return np.asarray(merged_boxes, dtype=np.float32), np.asarray(merged_scores, dtype=np.float32)


def get_train_stats(json_path: Path) -> float:
    if not json_path.exists():
        return 0.0
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    num_images = len(data.get("images", []))
    num_anns = len(data.get("annotations", []))
    return num_anns / num_images if num_images > 0 else 0.0


def _load_test_id_mapping(test_images_dir: Path, mapping_path: Path) -> dict[str, int]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"No existe mapping-path: {mapping_path}")

    df = pd.read_parquet(mapping_path)
    if "id" not in df.columns:
        raise ValueError(f"El parquet no contiene columna 'id': {mapping_path}")

    mapping: dict[str, int] = {}
    for item_id in df["id"].astype(str).tolist():
        normalized = item_id.replace("\\", "/")
        if "/images/test/" not in normalized:
            continue

        file_name = Path(normalized).name
        stem = Path(file_name).stem
        if not stem.isdigit():
            continue

        mapping[file_name] = int(stem)

    test_files = [
        p.name
        for p in sorted(test_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    missing = [name for name in test_files if name not in mapping]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Faltan IDs para imagenes de test en el mapping parquet. "
            f"Ejemplos: {preview}"
        )

    return mapping


def _predict_single(
    model,
    img_path: Path,
    conf: float,
    iou: float,
    max_det: int,
    imgsz: int | list[int],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=imgsz,
        augment=False,
        device=device,
        verbose=False,
    )
    res = results[0]

    if res.boxes is None or res.boxes.xyxy.numel() == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = res.boxes.conf.cpu().numpy().astype(np.float32)
    return boxes_xyxy, scores


def tta_predict(model, img_path, conf, iou, tta_iou, max_det, imgsz, device):
    img = np.array(Image.open(img_path).convert("RGB"))
    h_img, w_img = img.shape[:2]
    target_h, target_w = _normalize_imgsz(imgsz)

    variants = [
        (img, lambda b: b),
        (np.fliplr(img), lambda b: [target_w - b[2], b[1], target_w - b[0], b[3]]),
        (np.flipud(img), lambda b: [b[0], target_h - b[3], b[2], target_h - b[1]]),
        (
            np.fliplr(np.flipud(img)),
            lambda b: [target_w - b[2], target_h - b[3], target_w - b[0], target_h - b[1]],
        ),
    ]

    all_boxes, all_scores = [], []
    scale_x = w_img / float(target_w)
    scale_y = h_img / float(target_h)

    for aug_img, inv_fn in variants:
        aug_img_resized = _resize_for_model(aug_img, imgsz)
        res = model.predict(
            source=aug_img_resized,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        if res.boxes is None or res.boxes.xyxy.numel() == 0:
            continue

        for box, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            inv_box = inv_fn(box.tolist())
            all_boxes.append(
                [
                    inv_box[0] * scale_x,
                    inv_box[1] * scale_y,
                    inv_box[2] * scale_x,
                    inv_box[3] * scale_y,
                ]
            )
            all_scores.append(float(score))

    if not all_boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=tta_iou)

    final_boxes = boxes_tensor[keep_indices].cpu().numpy()[:max_det]
    final_scores = scores_tensor[keep_indices].cpu().numpy()[:max_det]
    return final_boxes, final_scores


def validate_submission_schema(rows: list[dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        for key in ["image_id", "category_id", "bbox", "score"]:
            if key not in row:
                raise ValueError(f"Fila {i} sin key requerida '{key}'")

        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Fila {i} tiene bbox invalido: {bbox}")


def save_submission_auto(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(rows), encoding="utf-8")
        return

    if output_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("submission.json", json.dumps(rows))
        return

    # Por defecto, guardar como .zip con submission.json
    zip_path = output_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO inference for ClearSAR")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al archivo best.pt")
    parser.add_argument("--mapping-path", type=str, default="catalog.v1.parquet")
    parser.add_argument("--test-images-dir", type=str, default="data/images/test")
    parser.add_argument("--output", type=str, default="outputs/submission_yolo.zip")
    parser.add_argument("--conf", type=float, default=0.1, help="Umbral de confianza")
    parser.add_argument("--iou", type=float, default=0.45, help="Umbral IOU para NMS")
    parser.add_argument("--tta-iou", type=float, default=0.4, help="Umbral IOU para fusion NMS en TTA")
    parser.add_argument("--max-det", type=int, default=500, help="Maximo detecciones por imagen")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--image-size",
        type=_parse_image_size,
        default=640,
        help="Tamano de entrada del modelo. Ejemplos: 640 o [512,1024] / (512,1024).",
    )

    parser.add_argument("--tta", action="store_true", help="Activar Test Time Augmentation")
    parser.add_argument("--merge-boxes", action="store_true", help="Fusionar sub-cajas horizontales adyacentes")
    parser.add_argument("--merge-row-tol", type=float, default=4.0, help="Tolerancia vertical para merge")
    parser.add_argument("--merge-gap-px", type=float, default=8.0, help="Gap horizontal maximo para merge")
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Por favor instala ultralytics: pip install ultralytics")

    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]

    mapping_path = Path(args.mapping_path)
    if not mapping_path.is_absolute():
        mapping_path = project_root / mapping_path

    test_images_dir = Path(args.test_images_dir)
    if not test_images_dir.is_absolute():
        test_images_dir = project_root / test_images_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    if not test_images_dir.exists() or not test_images_dir.is_dir():
        raise FileNotFoundError(f"No existe test-images-dir: {test_images_dir}")

    train_json = project_root / "data" / "annotations" / "instances_train.json"
    avg_train = get_train_stats(train_json)

    filename_to_image_id = _load_test_id_mapping(
        test_images_dir=test_images_dir,
        mapping_path=mapping_path,
    )

    image_files = [
        p
        for p in sorted(test_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    model = YOLO(args.checkpoint)
    submission_rows: List[Dict[str, Any]] = []

    print(
        f"[yolo] imgsz={args.image_size} | tta={args.tta} | conf={args.conf} | "
        f"iou={args.iou} | tta_iou={args.tta_iou} | merge_boxes={args.merge_boxes}"
    )

    for img_path in tqdm(image_files, desc="Inference"):
        if img_path.name not in filename_to_image_id:
            continue

        if args.tta:
            boxes_xyxy, scores = tta_predict(
                model=model,
                img_path=img_path,
                conf=args.conf,
                iou=args.iou,
                tta_iou=args.tta_iou,
                max_det=args.max_det,
                imgsz=args.image_size,
                device=args.device,
            )
        else:
            boxes_xyxy, scores = _predict_single(
                model=model,
                img_path=img_path,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                imgsz=args.image_size,
                device=args.device,
            )

        if args.merge_boxes:
            boxes_xyxy, scores = merge_horizontal_boxes(
                boxes_xyxy=boxes_xyxy,
                scores=scores,
                row_tol_px=args.merge_row_tol,
                gap_px=args.merge_gap_px,
            )

        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = [float(v) for v in box]
            w, h = x2 - x1, y2 - y1
            if w <= 0.001 or h <= 0.001:
                continue

            submission_rows.append(
                {
                    "image_id": int(filename_to_image_id[img_path.name]),
                    "category_id": 1,
                    "bbox": [x1, y1, float(w), float(h)],
                    "score": float(scores[i]),
                }
            )

    num_test_images = len(image_files)
    total_boxes = len(submission_rows)
    avg_test = total_boxes / num_test_images if num_test_images > 0 else 0.0

    print("\n" + "=" * 40)
    print("RESUMEN DE DENSIDAD")
    print(f"Promedio en TRAIN: {avg_train:.2f} cajas/img")
    print(f"Promedio en TEST:  {avg_test:.2f} cajas/img")
    print(f"Total cajas:       {total_boxes}")
    print("=" * 40 + "\n")

    validate_submission_schema(submission_rows)
    save_submission_auto(submission_rows, output_path)
    print(f"Proceso completado. Archivo guardado en: {output_path}")


if __name__ == "__main__":
    main()
