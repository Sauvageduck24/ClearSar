import subprocess
import csv
from pathlib import Path
from typing import Optional

# Lista de modelos a probar
MODELS = [
    # "yolo26s",
    "yolo11s",
    "yolo12s",
    # "yolov10s",
    # "yolov3s",
    "yolov5s",
    # "yolov6s",
    "yolov8s",
    "yolov9s",
]

# Configuración de entrenamiento e inferencia
CONFIG = {
    "epochs": 5,
    "batch_size": 16,
    "num_workers": 12,
    "lr": 0.0005,
    "image_size": 512,
    "mapping_path": "catalog.v1.parquet",
    "python_exe": r"C:\Users\esteb\.conda\envs\clearsar\python.exe",
    "conf": 0.05,
    "iou": 0.3,
    "max_det": 300,
}


def _best_val_map_for_model(model: str) -> Optional[float]:
    """Lee el results.csv del modelo y retorna el mejor mAP50-95 de validacion."""
    model_run_name = f"clearsar_{model.replace('.pt', '')}"
    results_csv = Path("outputs") / "yolo_runs" / model_run_name / "results.csv"

    if not results_csv.exists():
        return None

    candidate_cols = (
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
        "metrics/mAP50(B)",
        "metrics/mAP50",
    )

    best_map = None
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = None
            for col in candidate_cols:
                if col in row and row[col] not in (None, ""):
                    value = row[col]
                    break
            if value is None:
                continue
            try:
                metric = float(str(value).strip())
            except ValueError:
                continue
            best_map = metric if best_map is None else max(best_map, metric)

    return best_map

def run_test(model: str) -> bool:
    """Ejecuta el pipeline para un modelo dado."""
    print(f"\n{'='*60}")
    print(f"Probando modelo: {model} por {CONFIG['epochs']} épocas")
    print(f"{'='*60}\n")
    
    cmd = [
        CONFIG["python_exe"],
        "run_pipeline.py",
        "--yolo-model", model,
        "--skip-slicing",
        "--skip-inference",
        "--mapping-path", CONFIG["mapping_path"],
        "--yolo-extra-args", 
            "--dataset-source normal "
            f"--epochs {CONFIG['epochs']} "
            f"--batch-size {CONFIG['batch_size']} "
            f"--num-workers {CONFIG['num_workers']} "
            f"--image-size {CONFIG['image_size']} "
            f"--lr {CONFIG['lr']}"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {model} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error al procesar {model}: {e}")
        return False

def main():
    """Ejecuta todas las pruebas."""
    print("Iniciando pruebas de todos los modelos YOLO sin SAHI...")
    print(f"Modelos a probar: {', '.join(MODELS)}")
    
    results = {}
    best_map_by_model = {}
    for model in MODELS:
        results[model] = run_test(model)
        if results[model]:
            best_map_by_model[model] = _best_val_map_for_model(model)
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    
    successful = [m for m, success in results.items() if success]
    failed = [m for m, success in results.items() if not success]
    
    print(f"\n✓ Exitosos ({len(successful)}):")
    for model in successful:
        print(f"  - {model}")
    
    if failed:
        print(f"\n✗ Fallidos ({len(failed)}):")
        for model in failed:
            print(f"  - {model}")

    print("\nMejor mAP de validacion por modelo:")
    for model in MODELS:
        metric = best_map_by_model.get(model)
        if metric is None:
            print(f"  - {model}: N/A")
        else:
            print(f"  - {model}: {metric:.6f}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
