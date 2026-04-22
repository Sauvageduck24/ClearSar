# rfi_filter_search_v3_fusion.py
"""
Optimización de preprocesado SAR para detección de RFI
- Fusión por máximo de los 3 canales
- Métrica SCR local por box
- Visualizaciones antes/después y por canal individual
"""

import cv2
import json
import numpy as np
import optuna
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import pandas as pd

# ---------- Configuración ----------
ANN_PATH = Path("data/annotations/instances_train.json")
IMG_DIR = Path("data/images/train")
OUTPUT_DIR = Path("filter_optimization_results")
OUTPUT_DIR.mkdir(exist_ok=True)

N_TRIALS = 100
WORKERS = 4
MAX_IMGS_OPT = 200  # imágenes para optimización
MAX_IMGS_VIZ = 10   # imágenes para visualización final

# ---------- Carga de datos ----------
def load_dataset(ann_path, img_dir, max_imgs=None):
    """Carga pares (ruta_imagen, boxes) desde anotaciones COCO."""
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images_by_id = {img['id']: img for img in coco.get('images', [])}
    anns_by_image = {}
    for ann in coco.get('annotations', []):
        img_id = ann.get('image_id')
        bbox = ann.get('bbox')
        if img_id is None or not bbox or len(bbox) < 4:
            continue
        anns_by_image.setdefault(img_id, []).append([int(v) for v in bbox[:4]])

    img_ids = sorted(anns_by_image.keys())
    if max_imgs is not None:
        img_ids = img_ids[:max_imgs]

    dataset = []
    for img_id in img_ids:
        img_info = images_by_id.get(img_id)
        if not img_info:
            continue

        file_name = img_info.get('file_name')
        if not file_name:
            continue

        img_path = img_dir / file_name
        if not img_path.exists():
            continue

        boxes = anns_by_image.get(img_id, [])
        if not boxes:
            continue

        dataset.append((img_path, boxes))

    print(f"  → Dataset cargado: {len(dataset)} imágenes con anotaciones")
    return dataset


# ---------- Filtros SAR ----------
def lee_filter(ch, win=5):
    """Filtro Lee para reducción de speckle multiplicativo."""
    if win % 2 == 0:
        win += 1
    mean = cv2.blur(ch, (win, win))
    mean_sq = cv2.blur(ch ** 2, (win, win))
    var = mean_sq - mean ** 2
    noise_var = np.mean(var)
    k = np.clip(var / (var + noise_var + 1e-6), 0, 1)
    return mean + k * (ch - mean)


def apply_pipeline_to_channel(channel, params):
    """
    Aplica el pipeline completo a un solo canal.
    Retorna mapa de anomalía normalizado [0,255] float32.
    """
    ch = channel.astype(np.float32)
    
    # Filtro Lee
    if params.get('lee_enabled', True):
        win = params.get('lee_win', 5)
        ch = lee_filter(ch, win)
    
    # Top-hat horizontal
    kw = params.get('kernel_len', 21)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    tophat = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, kernel)
    
    # Normalización
    tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # CLAHE
    clahe_clip = params.get('clahe_clip', 1.5)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(tophat_norm)
    
    return enhanced.astype(np.float32)


def preprocess_rfi_fusion_max(img_bgr, params):
    """
    Procesa los 3 canales por separado y fusiona con máximo.
    Estrategia robusta para test sin anotaciones.
    """
    maps = []
    
    # Procesar cada canal (B=0, G=1, R=2)
    for ch_idx in range(3):
        channel = img_bgr[:, :, ch_idx]
        amap = apply_pipeline_to_channel(channel, params)
        maps.append(amap)
    
    # Fusión por máximo (el RFI es brillante en al menos un canal)
    fused = np.maximum.reduce(maps)
    
    return fused, maps  # Retorna también los mapas individuales para visualización


# ---------- Métrica SCR local ----------
def local_scr(amap, boxes, ring_width=5):
    """
    Calcula SCR local por box (contraste interior vs anillo exterior).
    Usa mediana para robustez y pondera por área del box.
    """
    scores = []
    H, W = amap.shape
    
    for x, y, w, h in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w < 2 or h < 2:
            continue
            
        # Región interior
        inner = amap[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]
        if inner.size == 0:
            continue
            
        # Anillo exterior
        y1, y2 = max(y - ring_width, 0), min(y + h + ring_width, H)
        x1, x2 = max(x - ring_width, 0), min(x + w + ring_width, W)
        outer_region = amap[y1:y2, x1:x2].copy()
        
        # Enmascarar interior
        mask = np.ones_like(outer_region, dtype=bool)
        inner_y1 = max(y - y1, 0)
        inner_y2 = min(y + h - y1, outer_region.shape[0])
        inner_x1 = max(x - x1, 0)
        inner_x2 = min(x + w - x1, outer_region.shape[1])
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
        
        outer_pixels = outer_region[mask]
        if outer_pixels.size == 0:
            continue
            
        med_in = np.median(inner)
        med_out = np.median(outer_pixels)
        
        if med_out > 0:
            scr = 20 * np.log10((med_in + 1e-6) / (med_out + 1e-6))
            weight = np.sqrt(w * h)  # Ponderar por área
            scores.append((scr, weight))
    
    if not scores:
        return -999.0
    
    total_weight = sum(w for _, w in scores)
    weighted_scr = sum(s * w for s, w in scores) / total_weight
    return weighted_scr


# ---------- Evaluación de una configuración ----------
def evaluate_params(params, dataset):
    """Evalúa una configuración sobre todo el dataset."""
    all_scores = []
    
    for img_path, boxes in tqdm(dataset, desc="Evaluando", leave=False):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        amap, _ = preprocess_rfi_fusion_max(img, params)
        scr = local_scr(amap, boxes)
        all_scores.append(scr)
    
    if not all_scores:
        return -999.0
    
    return np.mean(all_scores)


# ---------- Optuna Objective ----------
def objective(trial):
    """Función objetivo para Optuna."""
    params = {
        'kernel_len': trial.suggest_int('kernel_len', 11, 41, step=5),
        'lee_enabled': trial.suggest_categorical('lee_enabled', [True, False]),
        'lee_win': trial.suggest_int('lee_win', 3, 9, step=2),
        'clahe_clip': trial.suggest_float('clahe_clip', 1.0, 2.5),
    }
    
    # Usar dataset global cargado
    score = evaluate_params(params, _DATASET_OPT)
    
    # Guardar métricas adicionales
    trial.set_user_attr('kernel_len', params['kernel_len'])
    trial.set_user_attr('score', score)
    
    return score


# Variable global para el dataset de optimización
_DATASET_OPT = []


# ---------- Visualizaciones ----------
def visualize_results(dataset, best_params, num_samples=10):
    """
    Genera visualizaciones completas:
    - Original vs Fusionado
    - Los 3 canales procesados individualmente
    - Métricas SCR por imagen
    """
    print(f"\n[Visualización] Generando comparativas para {num_samples} imágenes...")
    
    # Seleccionar muestras aleatorias
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    samples = [dataset[i] for i in indices]
    
    # Crear figura grande
    fig = plt.figure(figsize=(20, 4 * len(samples)))
    gs = gridspec.GridSpec(len(samples), 6, figure=fig, wspace=0.05, hspace=0.3)
    
    channel_names = ['Azul (Ratio)', 'Verde (VH)', 'Rojo (VV)']
    
    for row, (img_path, boxes) in enumerate(samples):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Aplicar pipeline
        fused, individual_maps = preprocess_rfi_fusion_max(img, best_params)
        
        # Calcular SCR para cada versión
        scr_fused = local_scr(fused, boxes)
        scr_individual = [local_scr(m, boxes) for m in individual_maps]
        
        # 1. Imagen original (convertir BGR a RGB para visualización)
        ax = fig.add_subplot(gs[row, 0])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        # Dibujar boxes
        for x, y, w, h in boxes:
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='yellow', linewidth=1.5)
            ax.add_patch(rect)
        ax.set_title(f"Original\n{img_path.name[:15]}", fontsize=9)
        ax.axis('off')
        
        # 2. Resultado fusionado
        ax = fig.add_subplot(gs[row, 1])
        im = ax.imshow(fused, cmap='inferno', vmin=0, vmax=255)
        for x, y, w, h in boxes:
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='cyan', linewidth=1.5)
            ax.add_patch(rect)
        ax.set_title(f"Fusión (Max)\nSCR: {scr_fused:.1f} dB", fontsize=9, color='blue', fontweight='bold')
        ax.axis('off')
        
        # 3-5. Canales individuales procesados
        for col, (ch_map, ch_name, scr_val) in enumerate(zip(individual_maps, channel_names, scr_individual)):
            ax = fig.add_subplot(gs[row, col + 2])
            ax.imshow(ch_map, cmap='inferno', vmin=0, vmax=255)
            for x, y, w, h in boxes:
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=1.0, alpha=0.7)
                ax.add_patch(rect)
            ax.set_title(f"{ch_name}\nSCR: {scr_val:.1f} dB", fontsize=8)
            ax.axis('off')
        
        # 6. Diferencia Fusión - Mejor canal individual
        ax = fig.add_subplot(gs[row, 5])
        best_individual = individual_maps[np.argmax(scr_individual)]
        diff = np.clip(fused - best_individual, 0, 255)
        ax.imshow(diff, cmap='hot', vmin=0, vmax=100)
        ax.set_title("Diferencia\n(Fusión - Mejor canal)", fontsize=8)
        ax.axis('off')
    
    # Añadir barra de color
    plt.suptitle(f"Comparativa de Preprocesado RFI - Fusión por Máximo\n"
                 f"Params: kernel={best_params['kernel_len']}, Lee={best_params['lee_enabled']}, "
                 f"win={best_params.get('lee_win', 5)}, CLAHE={best_params['clahe_clip']:.2f}",
                 fontsize=12, y=1.02)
    
    # Guardar
    output_path = OUTPUT_DIR / f"visualizacion_comparativa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  → Visualización guardada: {output_path}")
    plt.show()
    
    return output_path


def plot_optimization_history(study):
    """Visualiza el historial de optimización."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Historial de SCR
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    best_values = np.maximum.accumulate(values)
    
    axes[0].plot(values, 'o-', alpha=0.5, markersize=4, label='Trial')
    axes[0].plot(best_values, 'r-', linewidth=2, label='Mejor hasta ahora')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('SCR medio (dB)')
    axes[0].set_title('Evolución de la Optimización')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Importancia de parámetros
    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())
        values_imp = list(importance.values())
        
        axes[1].barh(params, values_imp, color='steelblue')
        axes[1].set_xlabel('Importancia relativa')
        axes[1].set_title('Importancia de Hiperparámetros')
    except:
        axes[1].text(0.5, 0.5, 'Importancia no disponible', ha='center', va='center')
        axes[1].set_title('Importancia de Hiperparámetros')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"optimization_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  → Historial guardado: {output_path}")
    plt.show()


def plot_channel_comparison(img_path, boxes, best_params):
    """
    Visualización detallada de una imagen específica mostrando:
    - Canales originales
    - Canales procesados
    - Fusión final
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    fused, individual_maps = preprocess_rfi_fusion_max(img, best_params)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    channel_names = ['Azul (Ratio)', 'Verde (VH)', 'Rojo (VV)']
    
    for row in range(3):
        # Columna 1: Canal original
        ax = axes[row, 0]
        ch_orig = img[:, :, row]
        ax.imshow(ch_orig, cmap='gray')
        for x, y, w, h in boxes:
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1.5)
            ax.add_patch(rect)
        ax.set_title(f"{channel_names[row]} - Original", fontsize=10)
        ax.axis('off')
        
        # Columna 2: Canal procesado
        ax = axes[row, 1]
        ch_proc = individual_maps[row]
        ax.imshow(ch_proc, cmap='inferno', vmin=0, vmax=255)
        for x, y, w, h in boxes:
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='cyan', linewidth=1.5)
            ax.add_patch(rect)
        scr = local_scr(ch_proc, boxes)
        ax.set_title(f"{channel_names[row]} - Procesado\nSCR: {scr:.1f} dB", fontsize=10)
        ax.axis('off')
        
        # Columna 3: Diferencia (procesado - original normalizado)
        ax = axes[row, 2]
        ch_orig_norm = cv2.normalize(ch_orig.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        diff = np.clip(ch_proc - ch_orig_norm, 0, 255)
        ax.imshow(diff, cmap='RdBu_r', vmin=-50, vmax=50)
        ax.set_title(f"{channel_names[row]} - Diferencia", fontsize=10)
        ax.axis('off')
    
    plt.suptitle(f"Análisis por Canal - {img_path.name}\n"
                 f"Fusión Final SCR: {local_scr(fused, boxes):.1f} dB",
                 fontsize=12)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"channel_analysis_{img_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  → Análisis por canal guardado: {output_path}")
    plt.show()


# ---------- Main ----------
def main():
    global _DATASET_OPT
    
    print("=" * 70)
    print("  OPTIMIZACIÓN DE PREPROCESADO SAR-RFI CON FUSIÓN POR MÁXIMO")
    print("=" * 70)
    
    # 1. Cargar dataset completo
    print("\n[1] Cargando dataset de entrenamiento...")
    full_dataset = load_dataset(ANN_PATH, IMG_DIR, max_imgs=None)
    
    # 2. Crear subconjunto para optimización
    print(f"\n[2] Preparando subconjunto de optimización ({MAX_IMGS_OPT} imágenes)...")
    indices = np.random.choice(len(full_dataset), min(MAX_IMGS_OPT, len(full_dataset)), replace=False)
    _DATASET_OPT = [full_dataset[i] for i in indices]
    
    # 3. Ejecutar optimización con Optuna
    print(f"\n[3] Iniciando optimización Optuna ({N_TRIALS} trials, {WORKERS} workers)...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='sar_rfi_fusion_max'
    )
    
    # Callback para mostrar progreso
    def callback(study, trial):
        if trial.value is not None:
            print(f"  Trial {trial.number}: SCR = {trial.value:.2f} dB, "
                  f"Best = {study.best_value:.2f} dB")
    
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=WORKERS, callbacks=[callback])
    
    # 4. Mostrar mejores resultados
    print("\n" + "=" * 70)
    print("  MEJOR CONFIGURACIÓN ENCONTRADA")
    print("=" * 70)
    print(f"  SCR medio: {study.best_value:.2f} dB")
    print(f"  Parámetros:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    
    # 5. Evaluar en dataset completo
    print("\n[4] Evaluando mejor configuración en dataset completo...")
    best_params = study.best_params
    full_score = evaluate_params(best_params, full_dataset)
    print(f"  SCR en dataset completo: {full_score:.2f} dB")
    
    # 6. Guardar resultados
    print("\n[5] Guardando resultados...")
    
    # DataFrame con todos los trials
    df_trials = pd.DataFrame([
        {
            'trial': t.number,
            'scr': t.value,
            **t.params
        }
        for t in study.trials if t.value is not None
    ])
    df_trials.to_csv(OUTPUT_DIR / 'optimization_results.csv', index=False)
    
    # Guardar mejor configuración
    with open(OUTPUT_DIR / 'best_params.json', 'w') as f:
        json.dump({
            'params': best_params,
            'scr_full_dataset': full_score,
            'scr_optimization': study.best_value
        }, f, indent=2)
    
    print(f"  → Resultados guardados en: {OUTPUT_DIR}")
    
    # 7. Visualizaciones
    print("\n[6] Generando visualizaciones...")
    
    # Historial de optimización
    plot_optimization_history(study)
    
    # Comparativa general
    visualize_results(full_dataset, best_params, num_samples=MAX_IMGS_VIZ)
    
    # Análisis detallado de una imagen ejemplo
    sample_img, sample_boxes = full_dataset[0]
    plot_channel_comparison(sample_img, sample_boxes, best_params)
    
    # 8. Resumen final
    print("\n" + "=" * 70)
    print("  OPTIMIZACIÓN COMPLETADA")
    print("=" * 70)
    print(f"\n  ✅ Mejor configuración guardada en: {OUTPUT_DIR / 'best_params.json'}")
    print(f"  ✅ Usar en test con: preprocess_rfi_fusion_max(img, params)")
    print(f"\n  Pipeline para inferencia:")
    print(f"  ```python")
    print(f"  params = {best_params}")
    print(f"  amap, _ = preprocess_rfi_fusion_max(img_bgr, params)")
    print(f"  # amap está listo para alimentar al detector")
    print(f"  ```")


if __name__ == "__main__":
    main()