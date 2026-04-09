"""
Estadísticas del Dataset - Formato COCO
Requiere: pip install matplotlib numpy pillow
"""

import json
import os
import glob
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
ANNOTATIONS_FILE = "data/annotations/instances_train.json"
TRAIN_IMG_DIR    = "data/images/train"
TEST_IMG_DIR     = "data/images/test"

# ── Carga ──────────────────────────────────────────────────────────────────────
with open(ANNOTATIONS_FILE) as f:
    data = json.load(f)

images      = data["images"]
annotations = data["annotations"]
categories  = data["categories"]

cat_id_to_name = {c["id"]: c["name"] for c in categories}

# ── Imágenes en disco ──────────────────────────────────────────────────────────
train_files = set(os.path.basename(p) for p in glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
test_files  = set(os.path.basename(p) for p in glob.glob(os.path.join(TEST_IMG_DIR,  "*.png")))

ann_filenames = set(img["file_name"] for img in images)

# ── Índices ────────────────────────────────────────────────────────────────────
anns_by_image = defaultdict(list)
for ann in annotations:
    anns_by_image[ann["image_id"]].append(ann)

img_id_to_info = {img["id"]: img for img in images}

# ═══════════════════════════════════════════════════════════════════════════════
#  1. RESUMEN GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
n_images_json = len(images)
n_annotations = len(annotations)
n_categories  = len(categories)
n_train_disk  = len(train_files)
n_test_disk   = len(test_files)
n_img_with_ann = len(anns_by_image)
n_img_no_ann   = n_images_json - n_img_with_ann

print("=" * 60)
print("  RESUMEN GENERAL DEL DATASET")
print("=" * 60)
print(f"  Categorías            : {n_categories}  →  {[c['name'] for c in categories]}")
print(f"  Imágenes en JSON      : {n_images_json}")
print(f"  Imágenes  train/disco : {n_train_disk}")
print(f"  Imágenes  test /disco : {n_test_disk}")
print(f"  Anotaciones totales   : {n_annotations}")
print(f"  Imágenes con anots.   : {n_img_with_ann}")
print(f"  Imágenes sin anots.   : {n_img_no_ann}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  2. ANOTACIONES POR CATEGORÍA
# ═══════════════════════════════════════════════════════════════════════════════
cat_counts = Counter(ann["category_id"] for ann in annotations)
print("  ANOTACIONES POR CATEGORÍA")
print("  " + "-" * 40)
for cat_id, count in cat_counts.most_common():
    print(f"    {cat_id_to_name[cat_id]:20s}: {count:6d}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  3. ANOTACIONES POR IMAGEN (distribución)
# ═══════════════════════════════════════════════════════════════════════════════
anns_per_image = [len(v) for v in anns_by_image.values()]

# Incluir imágenes sin anotaciones
all_anns_per_image = anns_per_image + [0] * n_img_no_ann

print("  ANOTACIONES POR IMAGEN")
print("  " + "-" * 40)
print(f"    Min   : {min(all_anns_per_image)}")
print(f"    Max   : {max(all_anns_per_image)}")
print(f"    Media : {np.mean(all_anns_per_image):.2f}")
print(f"    Mediana: {np.median(all_anns_per_image):.1f}")
print(f"    Std   : {np.std(all_anns_per_image):.2f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  4. TAMAÑOS DE IMÁGENES
# ═══════════════════════════════════════════════════════════════════════════════
widths  = [img["width"]  for img in images]
heights = [img["height"] for img in images]
areas_img = [w * h for w, h in zip(widths, heights)]

print("  TAMAÑOS DE IMÁGENES")
print("  " + "-" * 40)
print(f"    Ancho  — min:{min(widths)}  max:{max(widths)}  media:{np.mean(widths):.1f}")
print(f"    Alto   — min:{min(heights)} max:{max(heights)} media:{np.mean(heights):.1f}")
unique_sizes = Counter(zip(widths, heights))
print(f"    Resoluciones únicas  : {len(unique_sizes)}")
print(f"    Todas las resoluciones:")
for (w, h), cnt in unique_sizes.most_common():
    print(f"      {w}×{h}  →  {cnt} imágenes")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  5. TAMAÑOS DE BOUNDING BOXES
# ═══════════════════════════════════════════════════════════════════════════════
bb_widths  = [ann["bbox"][2] for ann in annotations]
bb_heights = [ann["bbox"][3] for ann in annotations]
bb_areas   = [ann["area"]    for ann in annotations]

# Aspect ratio de cada bbox
bb_aspects = [w / h if h > 0 else 0 for w, h in zip(bb_widths, bb_heights)]

# Tamaño relativo bbox vs imagen
rel_areas = []
for ann in annotations:
    img_info = img_id_to_info.get(ann["image_id"])
    if img_info:
        img_area = img_info["width"] * img_info["height"]
        rel_areas.append(ann["area"] / img_area * 100)

print("  BOUNDING BOXES")
print("  " + "-" * 40)
print(f"    Ancho  — min:{min(bb_widths):.0f}  max:{max(bb_widths):.0f}  media:{np.mean(bb_widths):.1f}")
print(f"    Alto   — min:{min(bb_heights):.0f} max:{max(bb_heights):.0f} media:{np.mean(bb_heights):.1f}")
print(f"    Área   — min:{min(bb_areas):.0f}  max:{max(bb_areas):.0f}  media:{np.mean(bb_areas):.1f}")
print(f"    Área relativa (% img) — media:{np.mean(rel_areas):.2f}%  mediana:{np.median(rel_areas):.2f}%")
print(f"    Aspect ratio (w/h)    — media:{np.mean(bb_aspects):.2f}  mediana:{np.median(bb_aspects):.2f}")
print()

# Clasificar por tamaño COCO: small<32², medium<96², large>=96²
small  = sum(1 for a in bb_areas if a < 32**2)
medium = sum(1 for a in bb_areas if 32**2 <= a < 96**2)
large  = sum(1 for a in bb_areas if a >= 96**2)
print(f"    Pequeño  (<1024 px²) : {small:5d}  ({small/n_annotations*100:.1f}%)")
print(f"    Mediano  (1024–9216) : {medium:5d}  ({medium/n_annotations*100:.1f}%)")
print(f"    Grande   (>9216 px²) : {large:5d}  ({large/n_annotations*100:.1f}%)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Estadísticas del Dataset (COCO - RFI)", fontsize=16, fontweight="bold")

BINS = 40
COLOR = "#4C72B0"

# — Histograma: anotaciones por imagen —
ax = axes[0, 0]
ax.hist(all_anns_per_image, bins=BINS, color=COLOR, edgecolor="white")
ax.set_title("Anotaciones por imagen")
ax.set_xlabel("Número de anotaciones")
ax.set_ylabel("Frecuencia (imágenes)")

# — Histograma: área de bboxes —
ax = axes[0, 1]
ax.hist(bb_areas, bins=BINS, color="#DD8452", edgecolor="white")
ax.set_title("Distribución de área de BBoxes")
ax.set_xlabel("Área (px²)")
ax.set_ylabel("Frecuencia")

# — Scatter: ancho vs alto de bboxes —
ax = axes[0, 2]
ax.scatter(bb_widths, bb_heights, alpha=0.3, s=5, color="#55A868")
ax.set_title("Ancho vs Alto de BBoxes")
ax.set_xlabel("Ancho (px)")
ax.set_ylabel("Alto (px)")

# — Histograma: aspect ratio —
ax = axes[1, 0]
ax.hist(bb_aspects, bins=BINS, color="#C44E52", edgecolor="white")
ax.axvline(1, color="black", linestyle="--", linewidth=1, label="cuadrado (1:1)")
ax.set_title("Aspect Ratio de BBoxes (w/h)")
ax.set_xlabel("Ratio")
ax.set_ylabel("Frecuencia")
ax.legend()

# — Pastel: tamaño COCO —
ax = axes[1, 1]
labels_pie = [f"Pequeño\n(<1024)", f"Mediano\n(1024-9216)", f"Grande\n(>9216)"]
sizes_pie  = [small, medium, large]
colors_pie = ["#4C72B0", "#DD8452", "#55A868"]
wedges, texts, autotexts = ax.pie(
    sizes_pie, labels=labels_pie, colors=colors_pie,
    autopct="%1.1f%%", startangle=90, pctdistance=0.8
)
ax.set_title("Distribución por tamaño COCO")

# — Scatter: dimensiones de imágenes —
ax = axes[1, 2]
ax.scatter(widths, heights, alpha=0.4, s=8, color="#8172B2")
ax.set_title("Resoluciones de imágenes")
ax.set_xlabel("Ancho (px)")
ax.set_ylabel("Alto (px)")

plt.tight_layout()
plt.show()
print("\n  ✓ Análisis completado.")