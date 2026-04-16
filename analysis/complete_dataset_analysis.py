"""
Análisis Completo del Dataset ClearSAR - FASE 1

Este script realiza un análisis exhaustivo del dataset para entender:
1. Distribución de tamaños de boxes por imagen
2. Densidad de RFI por región de la imagen
3. Patrones de orientación (horizontal vs vertical)
4. Análisis de intensidad/contraste de rayas RFI
5. Correlación entre tamaño de imagen y número de boxes
6. Identificación de casos extremos/outliers

Uso:
    python analysis/complete_dataset_analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import cv2
from tqdm import tqdm
import seaborn as sns
from scipy import stats

# Configuración
ANNOTATIONS_FILE = "data/annotations/instances_train.json"
TRAIN_IMG_DIR = "data/images/train"
OUTPUT_DIR = "analysis/outputs"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_coco_data(annotations_path):
    """Carga datos COCO del dataset."""
    with open(annotations_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    return coco


def analyze_box_sizes_by_image(coco):
    """Analiza distribución de tamaños de boxes por imagen."""
    print("\n" + "="*60)
    print("1. ANÁLISIS DE TAMAÑOS DE BOXES POR IMAGEN")
    print("="*60)

    images = {img['id']: img for img in coco['images']}
    annotations_by_image = defaultdict(list)

    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Estadísticas por imagen
    boxes_per_image = []
    avg_box_area_per_image = []
    max_box_area_per_image = []
    min_box_area_per_image = []
    small_box_ratio_per_image = []
    medium_box_ratio_per_image = []
    large_box_ratio_per_image = []

    for img_id, anns in annotations_by_image.items():
        areas = [ann['area'] for ann in anns]
        widths = [ann['bbox'][2] for ann in anns]
        heights = [ann['bbox'][3] for ann in anns]

        boxes_per_image.append(len(anns))

        if areas:
            avg_box_area_per_image.append(np.mean(areas))
            max_box_area_per_image.append(np.max(areas))
            min_box_area_per_image.append(np.min(areas))

            # Clasificación COCO
            small = sum(1 for a in areas if a < 32**2)
            medium = sum(1 for a in areas if 32**2 <= a < 96**2)
            large = sum(1 for a in areas if a >= 96**2)

            small_box_ratio_per_image.append(small / len(anns) * 100)
            medium_box_ratio_per_image.append(medium / len(anns) * 100)
            large_box_ratio_per_image.append(large / len(anns) * 100)

    # Imprimir estadísticas
    print(f"\nEstadísticas de boxes por imagen:")
    print(f"  Media: {np.mean(boxes_per_image):.2f}")
    print(f"  Mediana: {np.median(boxes_per_image):.2f}")
    print(f"  Desviación estándar: {np.std(boxes_per_image):.2f}")
    print(f"  Mínimo: {np.min(boxes_per_image)}")
    print(f"  Máximo: {np.max(boxes_per_image)}")

    print(f"\nEstadísticas de área promedio por imagen:")
    print(f"  Media: {np.mean(avg_box_area_per_image):.2f} px²")
    print(f"  Mediana: {np.median(avg_box_area_per_image):.2f} px²")

    print(f"\nRatio de boxes pequeños por imagen (<1024 px²):")
    print(f"  Media: {np.mean(small_box_ratio_per_image):.2f}%")
    print(f"  Mediana: {np.median(small_box_ratio_per_image):.2f}%")
    print(f"  Máximo: {np.max(small_box_ratio_per_image):.2f}%")

    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis de Boxes por Imagen', fontsize=16, fontweight='bold')

    # Boxes por imagen
    axes[0, 0].hist(boxes_per_image, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Número de Boxes')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución de Boxes por Imagen')
    axes[0, 0].axvline(np.mean(boxes_per_image), color='red', linestyle='--', label=f'Media: {np.mean(boxes_per_image):.2f}')
    axes[0, 0].legend()

    # Área promedio por imagen
    axes[0, 1].hist(avg_box_area_per_image, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Área Promedio (px²)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Área Promedio por Imagen')
    axes[0, 1].axvline(np.median(avg_box_area_per_image), color='red', linestyle='--', label=f'Mediana: {np.median(avg_box_area_per_image):.0f}')
    axes[0, 1].legend()

    # Ratio de boxes pequeños
    axes[1, 0].hist(small_box_ratio_per_image, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Ratio de Boxes Pequeños (%)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Ratio de Boxes Pequeños por Imagen')
    axes[1, 0].axvline(np.mean(small_box_ratio_per_image), color='red', linestyle='--', label=f'Media: {np.mean(small_box_ratio_per_image):.1f}%')
    axes[1, 0].legend()

    # Scatter: número de boxes vs área promedio
    axes[1, 1].scatter(boxes_per_image, avg_box_area_per_image, alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Número de Boxes')
    axes[1, 1].set_ylabel('Área Promedio (px²)')
    axes[1, 1].set_title('Número de Boxes vs Área Promedio')

    # Calcular correlación
    corr = np.corrcoef(boxes_per_image, avg_box_area_per_image)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlación: {corr:.3f}', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/box_sizes_by_image.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'boxes_per_image': boxes_per_image,
        'avg_box_area_per_image': avg_box_area_per_image,
        'small_box_ratio_per_image': small_box_ratio_per_image
    }


def analyze_rfi_density_by_region(coco, train_img_dir, sample_size=100):
    """Analiza densidad de RFI por región de la imagen."""
    print("\n" + "="*60)
    print("2. ANÁLISIS DE DENSIDAD DE RFI POR REGIÓN")
    print("="*60)

    images = {img['id']: img for img in coco['images']}
    annotations_by_image = defaultdict(list)

    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Crear mapa de densidad acumulado
    density_map = defaultdict(int)

    # Muestrear imágenes para análisis
    image_ids = list(annotations_by_image.keys())
    np.random.seed(42)
    sampled_ids = np.random.choice(image_ids, min(sample_size, len(image_ids)), replace=False)

    for img_id in tqdm(sampled_ids, desc="Analizando densidad por región"):
        img_info = images[img_id]
        img_w, img_h = img_info['width'], img_info['height']

        # Dividir imagen en grid 10x10
        grid_h, grid_w = 10, 10
        cell_h = img_h / grid_h
        cell_w = img_w / grid_w

        for ann in annotations_by_image[img_id]:
            x, y, w, h = ann['bbox']
            center_x = x + w / 2
            center_y = y + h / 2

            # Determinar celda del grid
            cell_i = int(min(center_y / cell_h, grid_h - 1))
            cell_j = int(min(center_x / cell_w, grid_w - 1))

            density_map[(cell_i, cell_j)] += 1

    # Normalizar por número de imágenes
    for key in density_map:
        density_map[key] /= len(sampled_ids)

    # Visualizar mapa de calor
    fig, ax = plt.subplots(figsize=(10, 8))

    # Crear matriz de densidad
    density_matrix = np.zeros((grid_h, grid_w))
    for (i, j), count in density_map.items():
        density_matrix[i, j] = count

    im = ax.imshow(density_matrix, cmap='YlOrRd', interpolation='nearest')
    ax.set_title('Densidad de RFI por Región (promedio sobre {} imágenes)'.format(len(sampled_ids)))
    ax.set_xlabel('Región Horizontal')
    ax.set_ylabel('Región Vertical')
    plt.colorbar(im, ax=ax, label='Boxes promedio por celda')

    # Añadir valores en las celdas
    for i in range(grid_h):
        for j in range(grid_w):
            text = ax.text(j, i, f'{density_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rfi_density_by_region.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Estadísticas de densidad
    densities = list(density_map.values())
    print(f"\nEstadísticas de densidad por celda (10x10 grid):")
    print(f"  Media: {np.mean(densities):.2f} boxes/celda")
    print(f"  Mediana: {np.median(densities):.2f} boxes/celda")
    print(f"  Máximo: {np.max(densities):.2f} boxes/celda")
    print(f"  Mínimo: {np.min(densities):.2f} boxes/celda")

    return density_map


def analyze_box_orientation(coco):
    """Analiza patrones de orientación de boxes (horizontal vs vertical)."""
    print("\n" + "="*60)
    print("3. ANÁLISIS DE ORIENTACIÓN DE BOXES")
    print("="*60)

    widths = []
    heights = []
    aspect_ratios = []
    is_horizontal = []
    is_vertical = []

    for ann in coco['annotations']:
        w, h = ann['bbox'][2], ann['bbox'][3]
        widths.append(w)
        heights.append(h)

        aspect_ratio = w / h if h > 0 else float('inf')
        aspect_ratios.append(aspect_ratio)

        # Clasificar orientación
        if w > h * 1.2:
            is_horizontal.append(True)
            is_vertical.append(False)
        elif h > w * 1.2:
            is_horizontal.append(False)
            is_vertical.append(True)
        else:
            is_horizontal.append(False)
            is_vertical.append(False)

    # Estadísticas
    n_horizontal = sum(is_horizontal)
    n_vertical = sum(is_vertical)
    n_square = len(widths) - n_horizontal - n_vertical

    total = len(widths)

    print(f"\nDistribución de orientaciones:")
    print(f"  Horizontal (w > 1.2h): {n_horizontal} ({n_horizontal/total*100:.1f}%)")
    print(f"  Vertical (h > 1.2w): {n_vertical} ({n_vertical/total*100:.1f}%)")
    print(f"  Cuadrado/ísimil: {n_square} ({n_square/total*100:.1f}%)")

    print(f"\nEstadísticas de aspect ratio:")
    print(f"  Media: {np.mean(aspect_ratios):.2f}")
    print(f"  Mediana: {np.median(aspect_ratios):.2f}")
    print(f"  Desviación estándar: {np.std(aspect_ratios):.2f}")
    print(f"  Mínimo: {np.min(aspect_ratios):.2f}")
    print(f"  Máximo: {np.max(aspect_ratios):.2f}")

    # Percentiles de altura (crítico para slicing)
    heights_array = np.array(heights)
    print(f"\nPercentiles de altura de boxes:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(heights_array, p):.2f} px")

    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis de Orientación de Boxes', fontsize=16, fontweight='bold')

    # Distribución de anchos y alturas
    axes[0, 0].scatter(widths, heights, alpha=0.3, s=5)
    axes[0, 0].plot([0, max(widths)], [0, max(widths)], 'r--', label='1:1 (cuadrado)')
    axes[0, 0].set_xlabel('Ancho (px)')
    axes[0, 0].set_ylabel('Alto (px)')
    axes[0, 0].set_title('Ancho vs Alto de Boxes')
    axes[0, 0].legend()

    # Histograma de aspect ratio
    axes[0, 1].hist(aspect_ratios, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Aspect Ratio (w/h)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Aspect Ratio')
    axes[0, 1].axvline(1, color='red', linestyle='--', label='Cuadrado (1:1)')
    axes[0, 1].axvline(1.2, color='green', linestyle='--', label='Umbral 1.2')
    axes[0, 1].legend()

    # Pie chart de orientaciones
    labels = ['Horizontal', 'Vertical', 'Cuadrado']
    sizes = [n_horizontal, n_vertical, n_square]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Distribución de Orientaciones')

    # Box plot de alturas
    bp = axes[1, 1].boxplot([heights], vert=False, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    axes[1, 1].set_xlabel('Alto (px)')
    axes[1, 1].set_title('Distribución de Altura de Boxes')
    axes[1, 1].set_yticks([])
    axes[1, 1].grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/box_orientation.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'n_horizontal': n_horizontal,
        'n_vertical': n_vertical,
        'n_square': n_square,
        'heights': heights,
        'widths': widths,
        'aspect_ratios': aspect_ratios
    }


def analyze_rfi_intensity_contrast(coco, train_img_dir, sample_size=50):
    """Analiza intensidad y contraste de rayas RFI."""
    print("\n" + "="*60)
    print("4. ANÁLISIS DE INTENSIDAD Y CONTRASTE DE RAYAS RFI")
    print("="*60)

    images = {img['id']: img for img in coco['images']}
    annotations_by_image = defaultdict(list)

    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Muestrear imágenes
    image_ids = list(annotations_by_image.keys())
    np.random.seed(42)
    sampled_ids = np.random.choice(image_ids, min(sample_size, len(image_ids)), replace=False)

    intensities = []
    contrasts = []

    for img_id in tqdm(sampled_ids, desc="Analizando intensidad/contraste"):
        img_info = images[img_id]
        img_path = Path(train_img_dir) / img_info['file_name']

        if not img_path.exists():
            continue

        # Cargar imagen
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for ann in annotations_by_image[img_id]:
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Asegurar que el box esté dentro de la imagen
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_gray.shape[1] - x)
            h = min(h, img_gray.shape[0] - y)

            if w <= 0 or h <= 0:
                continue

            # Extraer región del box
            box_region = img_gray[y:y+h, x:x+w]

            if box_region.size == 0:
                continue

            # Calcular intensidad media
            intensity = np.mean(box_region)
            intensities.append(intensity)

            # Calcular contraste (desviación estándar)
            contrast = np.std(box_region)
            contrasts.append(contrast)

    # Estadísticas
    if intensities:
        print(f"\nEstadísticas de intensidad de RFI:")
        print(f"  Media: {np.mean(intensities):.2f}")
        print(f"  Mediana: {np.median(intensities):.2f}")
        print(f"  Desviación estándar: {np.std(intensities):.2f}")
        print(f"  Mínimo: {np.min(intensities):.2f}")
        print(f"  Máximo: {np.max(intensities):.2f}")

    if contrasts:
        print(f"\nEstadísticas de contraste de RFI:")
        print(f"  Media: {np.mean(contrasts):.2f}")
        print(f"  Mediana: {np.median(contrasts):.2f}")
        print(f"  Desviación estándar: {np.std(contrasts):.2f}")
        print(f"  Mínimo: {np.min(contrasts):.2f}")
        print(f"  Máximo: {np.max(contrasts):.2f}")

    # Visualizaciones
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Análisis de Intensidad y Contraste de RFI', fontsize=16, fontweight='bold')

    if intensities:
        axes[0].hist(intensities, bins=30, color='lightblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Intensidad Media')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Intensidad de RFI')
        axes[0].axvline(np.mean(intensities), color='red', linestyle='--', label=f'Media: {np.mean(intensities):.1f}')
        axes[0].legend()

    if contrasts:
        axes[1].hist(contrasts, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Contraste (std)')
        axes[1].set_ylabel('Frecuencia')
        axes[1].set_title('Distribución de Contraste de RFI')
        axes[1].axvline(np.mean(contrasts), color='red', linestyle='--', label=f'Media: {np.mean(contrasts):.1f}')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rfi_intensity_contrast.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {'intensities': intensities, 'contrasts': contrasts}


def analyze_image_size_correlation(coco):
    """Analiza correlación entre tamaño de imagen y número de boxes."""
    print("\n" + "="*60)
    print("5. ANÁLISIS DE CORRELACIÓN: TAMAÑO DE IMAGEN VS NÚMERO DE BOXES")
    print("="*60)

    images = {img['id']: img for img in coco['images']}
    annotations_by_image = defaultdict(list)

    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    image_areas = []
    num_boxes = []
    box_densities = []

    for img_id, img_info in images.items():
        img_w, img_h = img_info['width'], img_info['height']
        img_area = img_w * img_h
        n_boxes = len(annotations_by_image.get(img_id, []))
        box_density = n_boxes / (img_area / 1000000)  # boxes por millón de píxeles

        image_areas.append(img_area)
        num_boxes.append(n_boxes)
        box_densities.append(box_density)

    # Calcular correlaciones
    corr_area_boxes = np.corrcoef(image_areas, num_boxes)[0, 1]
    corr_area_density = np.corrcoef(image_areas, box_densities)[0, 1]

    print(f"\nCorrelación entre área de imagen y número de boxes: {corr_area_boxes:.3f}")
    print(f"Correlación entre área de imagen y densidad de boxes: {corr_area_density:.3f}")

    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis de Correlación: Tamaño de Imagen vs Boxes', fontsize=16, fontweight='bold')

    # Área vs número de boxes
    axes[0, 0].scatter(image_areas, num_boxes, alpha=0.5, s=20)
    axes[0, 0].set_xlabel('Área de Imagen (px²)')
    axes[0, 0].set_ylabel('Número de Boxes')
    axes[0, 0].set_title(f'Área vs Número de Boxes (r={corr_area_boxes:.3f})')

    # Área vs densidad de boxes
    axes[0, 1].scatter(image_areas, box_densities, alpha=0.5, s=20)
    axes[0, 1].set_xlabel('Área de Imagen (px²)')
    axes[0, 1].set_ylabel('Densidad de Boxes (por Mpx)')
    axes[0, 1].set_title(f'Área vs Densidad (r={corr_area_density:.3f})')

    # Distribución de áreas de imagen
    axes[1, 0].hist(image_areas, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Área de Imagen (px²)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Áreas de Imagen')

    # Distribución de número de boxes
    axes[1, 1].hist(num_boxes, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Número de Boxes')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Número de Boxes por Imagen')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/image_size_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'image_areas': image_areas,
        'num_boxes': num_boxes,
        'box_densities': box_densities,
        'corr_area_boxes': corr_area_boxes,
        'corr_area_density': corr_area_density
    }


def identify_outliers(coco):
    """Identifica casos extremos/outliers en el dataset."""
    print("\n" + "="*60)
    print("6. IDENTIFICACIÓN DE CASOS EXTREMOS/OUTLIERS")
    print("="*60)

    images = {img['id']: img for img in coco['images']}
    annotations_by_image = defaultdict(list)

    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Detectar outliers por diferentes criterios
    outliers = {
        'giant_boxes': [],
        'tiny_boxes': [],
        'extreme_aspect_ratio': [],
        'many_boxes': [],
        'very_few_boxes': [],
        'high_density': [],
        'low_density': []
    }

    for img_id, img_info in images.items():
        img_w, img_h = img_info['width'], img_info['height']
        img_area = img_w * img_h
        anns = annotations_by_image.get(img_id, [])

        for ann in anns:
            w, h = ann['bbox'][2], ann['bbox'][3]
            area = ann['area']
            aspect_ratio = w / h if h > 0 else float('inf')

            # Boxes gigantes (>100000 px²)
            if area > 100000:
                outliers['giant_boxes'].append({
                    'image_id': img_id,
                    'file_name': img_info['file_name'],
                    'area': area,
                    'w': w, 'h': h
                })

            # Boxes diminutos (<10 px²)
            if area < 10:
                outliers['tiny_boxes'].append({
                    'image_id': img_id,
                    'file_name': img_info['file_name'],
                    'area': area,
                    'w': w, 'h': h
                })

            # Aspect ratio extremo (>20 o <0.05)
            if aspect_ratio > 20 or aspect_ratio < 0.05:
                outliers['extreme_aspect_ratio'].append({
                    'image_id': img_id,
                    'file_name': img_info['file_name'],
                    'aspect_ratio': aspect_ratio,
                    'w': w, 'h': h
                })

        # Imágenes con muchos boxes (>20)
        if len(anns) > 20:
            outliers['many_boxes'].append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'num_boxes': len(anns)
            })

        # Imágenes con muy pocos boxes (<2)
        if len(anns) > 0 and len(anns) < 2:
            outliers['very_few_boxes'].append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'num_boxes': len(anns)
            })

        # Densidad muy alta (>100 boxes por Mpx)
        density = len(anns) / (img_area / 1000000)
        if density > 100:
            outliers['high_density'].append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'density': density,
                'num_boxes': len(anns)
            })

        # Densidad muy baja (<1 box por Mpx, pero con boxes)
        if len(anns) > 0 and density < 1:
            outliers['low_density'].append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'density': density,
                'num_boxes': len(anns)
            })

    # Imprimir resultados
    print(f"\nCasos extremos identificados:")
    for category, items in outliers.items():
        print(f"  {category.replace('_', ' ').title()}: {len(items)}")

        if items and len(items) <= 5:
            for item in items:
                print(f"    - {item.get('file_name', 'N/A')}: {item}")

    # Guardar outliers en JSON
    with open(f'{OUTPUT_DIR}/outliers.json', 'w') as f:
        json.dump(outliers, f, indent=2)

    # Visualización
    categories = list(outliers.keys())
    n_categories = len(categories)

    # Ajustar el numero de filas segun el numero de categorias
    n_cols = 3
    n_rows = (n_categories + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    fig.suptitle('Distribución de Casos Extremos', fontsize=16, fontweight='bold')

    # Asegurar que axes sea siempre un array 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, category in enumerate(categories):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        count = len(outliers[category])

        # Crear gráfico de barras simple
        ax.bar([category.replace('_', ' ').title()], [count], color='skyblue', edgecolor='black')
        ax.set_ylabel('Cantidad')
        ax.set_title(f'{category.replace("_", " ").title()}: {count}')
        ax.tick_params(axis='x', rotation=45)

    # Ocultar subplots vacios si los hay
    for i in range(n_categories, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/outliers_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return outliers


def generate_summary_report(all_results):
    """Genera un reporte resumen con todos los hallazgos."""
    print("\n" + "="*60)
    print("REPORTE RESUMEN - ANÁLISIS COMPLETO DEL DATASET")
    print("="*60)

    report = []
    report.append("# Análisis Completo del Dataset ClearSAR\n")

    report.append("## 1. Distribución de Tamaños de Boxes\n")
    if 'boxes_per_image' in all_results:
        boxes_per_image = all_results['boxes_per_image']
        report.append(f"- Promedio de boxes por imagen: {np.mean(boxes_per_image):.2f}\n")
        report.append(f"- Mediana de boxes por imagen: {np.median(boxes_per_image):.2f}\n")

    report.append("## 2. Orientación de Boxes\n")
    if 'n_horizontal' in all_results:
        total = all_results['n_horizontal'] + all_results['n_vertical'] + all_results['n_square']
        report.append(f"- Boxes horizontales: {all_results['n_horizontal']} ({all_results['n_horizontal']/total*100:.1f}%)\n")
        report.append(f"- Boxes verticales: {all_results['n_vertical']} ({all_results['n_vertical']/total*100:.1f}%)\n")
        report.append(f"- Boxes cuadrados: {all_results['n_square']} ({all_results['n_square']/total*100:.1f}%)\n")

    report.append("## 3. Recomendaciones para Slicing\n")
    if 'heights' in all_results:
        heights = all_results['heights']
        report.append(f"- Altura mediana de boxes: {np.median(heights):.2f} px\n")
        report.append(f"- 75th percentile de altura: {np.percentile(heights, 75):.2f} px\n")
        report.append(f"- 90th percentile de altura: {np.percentile(heights, 90):.2f} px\n")
        report.append(f"- **Recomendación slice_max_height_px**: {int(np.percentile(heights, 75))} px\n")

    report.append("## 4. Recomendaciones para Augmentations\n")
    report.append("- Implementar CLAHE con clip_limit=3.0 y tileGridSize=(8, 8)\n")
    report.append("- Añadir speckle noise (sigma=0.15, p=0.3)\n")
    report.append("- Añadir patch dropout (patch_size=32, drop_prob=0.08, p=0.3)\n")
    report.append("- Añadir blur para ignorar speckle (p=0.3)\n")
    report.append("- Añadir JPEG compression (quality 80-100, p=0.2)\n")

    # Guardar reporte
    with open(f'{OUTPUT_DIR}/analysis_summary.md', 'w') as f:
        f.writelines(report)

    print("\n" + "-"*60)
    print("REPORTE GENERADO:")
    print(f"  - Resumen: {OUTPUT_DIR}/analysis_summary.md")
    print(f"  - Outliers: {OUTPUT_DIR}/outliers.json")
    print(f"  - Visualizaciones: {OUTPUT_DIR}/*.png")
    print("-"*60)


def main():
    """Función principal."""
    print("ANALISIS COMPLETO DEL DATASET CLEARSAR")
    print("="*60)

    # Cargar datos
    print("\nCargando datos COCO...")
    coco = load_coco_data(ANNOTATIONS_FILE)
    print(f"  Imágenes: {len(coco['images'])}")
    print(f"  Anotaciones: {len(coco['annotations'])}")

    all_results = {}

    # Ejecutar análisis
    try:
        # 1. Análisis de tamaños por imagen
        results = analyze_box_sizes_by_image(coco)
        all_results.update(results)

        # 2. Densidad por región
        density_map = analyze_rfi_density_by_region(coco, TRAIN_IMG_DIR)

        # 3. Orientación de boxes
        results = analyze_box_orientation(coco)
        all_results.update(results)

        # 4. Intensidad y contraste
        intensity_results = analyze_rfi_intensity_contrast(coco, TRAIN_IMG_DIR)

        # 5. Correlación tamaño imagen vs boxes
        corr_results = analyze_image_size_correlation(coco)
        all_results.update(corr_results)

        # 6. Identificación de outliers
        outliers = identify_outliers(coco)

        # Generar reporte resumen
        generate_summary_report(all_results)

        print("\n" + "="*60)
        print("ANALISIS COMPLETADO CON EXITO")
        print("="*60)

    except Exception as e:
        print(f"\nERROR durante el analisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
