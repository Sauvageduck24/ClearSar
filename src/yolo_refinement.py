from __future__ import annotations

"""
YOLO Refinement module for ClearSAR.
Entrenamiento final de "pulido" (sprint final) sin mosaicos ni copy-paste.
Carga el best.pt del entrenamiento anterior y ajusta los pesos finales.
"""

import argparse
import shutil
from pathlib import Path
import torch
import numpy as np
import cv2
import albumentations as A
from ultralytics import YOLO

from src.config import default_config
from src.utils.repro import set_seed

# --- MISMAS FUNCIONES DE AUMENTACIÓN QUE EN YOLO_TRAIN ---

def apply_sar_clahe(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    try:
        result[:, :, 0] = clahe.apply(result[:, :, 0])
    except Exception:
        return image
    return result

def build_sar_augment_callback():
    """Mantenemos el pipeline pero puedes bajar 'p' si quieres algo más real"""
    pipeline = A.Compose([
        A.Lambda(image=apply_sar_clahe, p=0.3), 
        A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.1, 0.1), p=0.2),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.Sharpen(alpha=(0.1, 0.3), p=0.3),
    ])

    def on_train_batch_start(trainer, *cb_args, **cb_kwargs):
        # Lógica de inyección de Albumentations (idéntica a yolo_train.py)
        imgs = None
        batch_container = None
        if hasattr(trainer, "batch") and isinstance(trainer.batch, dict) and "img" in trainer.batch:
            imgs = trainer.batch["img"]
            batch_container = ("trainer", None)
        if imgs is None: return

        device = imgs.device
        imgs_np = (imgs.cpu().numpy() * 255).astype(np.uint8)
        augmented = []
        for img in imgs_np:
            img_hwc = img.transpose(1, 2, 0)
            result = pipeline(image=img_hwc)["image"]
            augmented.append(result.transpose(2, 0, 1))
        
        augmented_tensor = torch.from_numpy(np.stack(augmented).astype(np.float32) / 255.0).to(device)
        if batch_container[0] == "trainer":
            trainer.batch["img"] = augmented_tensor

    return on_train_batch_start

# --- PARSE DE ARGUMENTOS ESPECÍFICO PARA REFINAMIENTO ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine YOLO detector (Sprint Final)")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--weights", type=str, required=True, 
                        help="Path al best.pt del entrenamiento previo")
    parser.add_argument("--epochs", type=int, default=30, 
                        help="Pocas épocas para el pulido final (20-40)")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate más bajo (ej: 0.0001)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = default_config(project_root=args.project_root)
    project_root = cfg.paths.project_root
    set_seed(args.seed)

    # El dataset ya fue creado por yolo_train.py, lo reutilizamos
    yaml_path = project_root / "data" / "yolo" / "clearsar.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en {yaml_path}. Ejecuta yolo_train.py primero.")

    print(f"[refine] Cargando pesos para refinamiento: {args.weights}")
    model = YOLO(args.weights)

    # Definir nombre de la corrida (añadimos _refine para no sobreescribir el original pero estar en la misma carpeta)
    original_run_name = Path(args.weights).parents[1].name # Asume estructura runs/train/weights/best.pt
    run_name = f"{original_run_name}_refine"
    yolo_runs_dir = project_root / "outputs" / "yolo_runs"

    # ── CONFIGURACIÓN DE REFINAMIENTO (SIN PESAS) ───────────────────────
    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        lr0=args.lr,           # Learning rate reducido
        seed=args.seed,
        patience=0,            # En refinamiento solemos querer completar todas las épocas
        project=str(yolo_runs_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        
        # DESACTIVAMOS aumentaciones geométricas pesadas
        mosaic=0.0,            # <<--- CLAVE: Ya no hay mosaicos
        copy_paste=0.0,        # <<--- CLAVE: Ya no hay copy-paste
        mixup=0.0,
        
        # Mantenemos las que no ensucian la imagen
        fliplr=0.5,
        flipud=0.5,
        hsv_v=0.1,             # Bajamos un poco la agresividad del color
        
        # Optimizador estable
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=0,       # No hace falta warmup, el modelo ya sabe mucho
    )

    if args.device is not None:
        train_kwargs["device"] = args.device

    # Registrar el callback de Albumentations
    model.add_callback("on_train_batch_start", build_sar_augment_callback())

    print(f"[refine] Iniciando sprint final de {args.epochs} épocas...")
    results = model.train(**train_kwargs)

    # Guardar el resultado final en la carpeta de modelos
    best_refine = yolo_runs_dir / run_name / "weights" / "best.pt"
    if best_refine.exists():
        dest = cfg.paths.models_dir / f"yolo_refined_final.pt"
        shutil.copy2(best_refine, dest)
        print(f"[refine] Modelo final pulido guardado en: {dest}")

if __name__ == "__main__":
    main()