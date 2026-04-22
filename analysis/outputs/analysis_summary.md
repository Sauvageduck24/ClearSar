# Análisis Completo del Dataset ClearSAR
## 1. Distribución de Tamaños de Boxes
- Promedio de boxes por imagen: 2.98
- Mediana de boxes por imagen: 2.00
## 2. Orientación de Boxes
- Boxes horizontales: 8215 (88.4%)
- Boxes verticales: 788 (8.5%)
- Boxes cuadrados: 285 (3.1%)
## 3. Recomendaciones para Slicing
- Altura mediana de boxes: 10.00 px
- 75th percentile de altura: 13.00 px
- 90th percentile de altura: 34.00 px
- **Recomendación slice_max_height_px**: 13 px
## 4. Recomendaciones para Augmentations
- Implementar CLAHE con clip_limit=3.0 y tileGridSize=(8, 8)
- Añadir speckle noise (sigma=0.15, p=0.3)
- Añadir patch dropout (patch_size=32, drop_prob=0.08, p=0.3)
- Añadir blur para ignorar speckle (p=0.3)
- Añadir JPEG compression (quality 80-100, p=0.2)
