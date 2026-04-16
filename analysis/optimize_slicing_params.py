"""
Optimización de Parámetros de Slicing - FASE 4

Basado en el análisis del dataset, este script:
1. Analiza la distribución de alturas de boxes
2. Calcula percentiles críticos
3. Recomienda parámetros óptimos de slicing
4. Genera un archivo de configuración con los mejores parámetros

Uso:
    python analysis/optimize_slicing_params.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuración
ANNOTATIONS_FILE = "data/annotations/instances_train.json"
OUTPUT_DIR = "analysis/outputs"
OUTPUT_FILE = f"{OUTPUT_DIR}/slicing_params_recommendation.json"


def load_coco_data(annotations_path):
    """Carga datos COCO del dataset."""
    with open(annotations_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    return coco


def analyze_box_heights(coco):
    """Analiza la distribución de alturas de boxes."""
    print("\n" + "="*60)
    print("ANÁLISIS DE ALTURAS DE BOXES PARA OPTIMIZACIÓN DE SLICING")
    print("="*60)

    heights = []
    widths = []
    areas = []

    for ann in coco['annotations']:
        w, h = ann['bbox'][2], ann['bbox'][3]
        area = ann['area']

        widths.append(w)
        heights.append(h)
        areas.append(area)

    heights = np.array(heights)
    widths = np.array(widths)
    areas = np.array(areas)

    # Calcular percentiles
    percentiles = [10, 25, 50, 75, 80, 85, 90, 95, 99]
    height_percentiles = {p: np.percentile(heights, p) for p in percentiles}
    width_percentiles = {p: np.percentile(widths, p) for p in percentiles}
    area_percentiles = {p: np.percentile(areas, p) for p in percentiles}

    print("\nPercentiles de Altura de Boxes:")
    for p in percentiles:
        print(f"  {p}th percentile: {height_percentiles[p]:.2f} px")

    print("\nPercentiles de Ancho de Boxes:")
    for p in percentiles:
        print(f"  {p}th percentile: {width_percentiles[p]:.2f} px")

    print("\nPercentiles de Área de Boxes:")
    for p in percentiles:
        print(f"  {p}th percentile: {area_percentiles[p]:.2f} px²")

    return {
        'heights': heights,
        'widths': widths,
        'areas': areas,
        'height_percentiles': height_percentiles,
        'width_percentiles': width_percentiles,
        'area_percentiles': area_percentiles
    }


def recommend_slicing_params(analysis):
    """Recomienda parámetros óptimos de slicing."""
    print("\n" + "="*60)
    print("RECOMENDACIONES DE PARÁMETROS DE SLICING")
    print("="*60)

    heights = analysis['heights']
    height_percentiles = analysis['height_percentiles']

    # Recomendación para slice_height
    # El slice_height debe ser suficientemente grande para capturar boxes
    # pero no tan grande que pierda la ventaja de la mayor resolución

    # Opción 1: Basado en 90th percentile (captura 90% de boxes)
    slice_height_opt_1 = int(np.ceil(height_percentiles[90] * 1.5))

    # Opción 2: Basado en 75th percentile (más agresivo)
    slice_height_opt_2 = int(np.ceil(height_percentiles[75] * 2))

    # Opción 3: Basado en mediana (conservador)
    slice_height_opt_3 = int(np.ceil(height_percentiles[50] * 3))

    print(f"\nRecomendaciones para slice_height:")
    print(f"  Opción 1 (90th percentile * 1.5): {slice_height_opt_1} px")
    print(f"  Opción 2 (75th percentile * 2): {slice_height_opt_2} px")
    print(f"  Opción 3 (50th percentile * 3): {slice_height_opt_3} px")

    # Recomendación para slice_max_height_px
    # Debe capturar boxes pequeños que se pierden en la pasada completa
    # El 75th percentile es un buen umbral: captura 75% de boxes como "pequeños"

    slice_max_height_px_opt = int(np.ceil(height_percentiles[75]))

    print(f"\nRecomendación para slice_max_height_px:")
    print(f"  Basado en 75th percentile: {slice_max_height_px_opt} px")
    print(f"  (Captura {75}% de boxes como 'pequeños' en la pasada 2)")

    # Recomendación para slice_overlap
    # Debe ser suficiente para no perder boxes en los bordes
    # Un 20-30% del slice_height es típico

    overlap_opt_1 = int(slice_height_opt_1 * 0.25)
    overlap_opt_2 = int(slice_height_opt_1 * 0.30)
    overlap_opt_3 = int(slice_height_opt_1 * 0.35)

    print(f"\nRecomendaciones para slice_overlap (para slice_height={slice_height_opt_1}):")
    print(f"  Opción 1 (25%): {overlap_opt_1} px")
    print(f"  Opción 2 (30%): {overlap_opt_2} px")
    print(f"  Opción 3 (35%): {overlap_opt_3} px")

    # Configuraciones recomendadas
    configs = [
        {
            'name': 'conservative',
            'description': 'Configuración conservadora - Mayor recall, más procesamiento',
            'slice_height': slice_height_opt_3,
            'slice_overlap': int(slice_height_opt_3 * 0.25),
            'slice_max_height_px': slice_max_height_px_opt,
            'expected_coverage': '95%+ de boxes pequeños'
        },
        {
            'name': 'balanced',
            'description': 'Configuración balanceada - Buen compromiso recall/velocidad',
            'slice_height': slice_height_opt_1,
            'slice_overlap': overlap_opt_2,
            'slice_max_height_px': slice_max_height_px_opt,
            'expected_coverage': '90%+ de boxes pequeños'
        },
        {
            'name': 'aggressive',
            'description': 'Configuración agresiva - Menor procesamiento, buen recall',
            'slice_height': slice_height_opt_2,
            'slice_overlap': int(slice_height_opt_2 * 0.30),
            'slice_max_height_px': slice_max_height_px_opt,
            'expected_coverage': '85%+ de boxes pequeños'
        }
    ]

    print("\n" + "-"*60)
    print("CONFIGURACIONES RECOMENDADAS:")
    print("-"*60)
    for config in configs:
        print(f"\n{config['name'].upper()}:")
        print(f"  Descripción: {config['description']}")
        print(f"  slice_height: {config['slice_height']} px")
        print(f"  slice_overlap: {config['slice_overlap']} px")
        print(f"  slice_max_height_px: {config['slice_max_height_px']} px")
        print(f"  Cobertura esperada: {config['expected_coverage']}")

    return configs


def calculate_coverage(config, analysis):
    """Calcula la cobertura esperada de boxes pequeños para una configuración."""
    heights = analysis['heights']
    slice_max_height_px = config['slice_max_height_px']

    # Boxes que serían capturados en la pasada 2 (pequeños)
    small_boxes = heights <= slice_max_height_px
    coverage = np.sum(small_boxes) / len(heights) * 100

    return coverage


def generate_benchmark_commands(configs, model_name='yolo26n'):
    """Genera comandos de benchmark para probar las configuraciones."""
    print("\n" + "="*60)
    print("COMANDOS DE BENCHMARK PARA PROBAR CONFIGURACIONES")
    print("="*60)

    commands = []
    for config in configs:
        cmd = f"""python run_pipeline_slicing.py \\
  --yolo-model {model_name} \\
  --image-size 512 \\
  --mapping-path catalog.v1.parquet \\
  --yolo-extra-args \\
    "--batch-size 24 --lr 0.01 --lrf 0.1 --epochs 50 --num-workers 12 --kfold 1" \\
  --yolo-inference \\
    "--slice-height {config['slice_height']} --slice-overlap {config['slice_overlap']} --slice-max-height-px {config['slice_max_height_px']} --slice-nms-iou 0.3\""""
        commands.append(cmd)

        print(f"\n# {config['name'].upper()}")
        print(cmd)

    return commands


def save_recommendations(configs, analysis, output_file):
    """Guarda las recomendaciones en un archivo JSON."""
    output_data = {
        'analysis_summary': {
            'total_boxes': len(analysis['heights']),
            'height_median': float(np.median(analysis['heights'])),
            'height_75th_percentile': float(analysis['height_percentiles'][75]),
            'height_90th_percentile': float(analysis['height_percentiles'][90]),
        },
        'recommendations': [],
        'benchmark_commands': []
    }

    for config in configs:
        coverage = calculate_coverage(config, analysis)
        config_with_coverage = config.copy()
        config_with_coverage['expected_coverage_pct'] = round(coverage, 2)
        output_data['recommendations'].append(config_with_coverage)

    # Generar comandos
    commands = generate_benchmark_commands(configs)
    output_data['benchmark_commands'] = commands

    # Guardar
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nRecomendaciones guardadas en: {output_file}")


def main():
    """Función principal."""
    print("OPTIMIZACIÓN DE PARÁMETROS DE SLICING")
    print("="*60)

    # Cargar datos
    print("\nCargando datos COCO...")
    coco = load_coco_data(ANNOTATIONS_FILE)
    print(f"  Anotaciones: {len(coco['annotations'])}")

    # Analizar alturas de boxes
    analysis = analyze_box_heights(coco)

    # Recomendar parámetros
    configs = recommend_slicing_params(analysis)

    # Guardar recomendaciones
    save_recommendations(configs, analysis, OUTPUT_FILE)

    print("\n" + "="*60)
    print("OPTIMIZACIÓN COMPLETADA")
    print("="*60)
    print(f"\nRecomendaciones guardadas en: {OUTPUT_FILE}")
    print("\nPróximo paso: Ejecutar los comandos de benchmark para encontrar")
    print("la mejor configuración para tu caso de uso específico.")


if __name__ == "__main__":
    main()
