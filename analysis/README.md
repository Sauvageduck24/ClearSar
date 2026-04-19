# ClearSAR — Pipeline de Detección de RFI

> Competición de detección de interferencias de radiofrecuencia (RFI) en imágenes SAR.
> Las RFI aparecen como rayas horizontales brillantes que contaminan la imagen de reflectividad.
> El objetivo es localizar esas rayas con bounding boxes (formato COCO, métrica mAP).

---

## Fase 1 — Exploración del dataset
*Antes de tocar el modelo, entender qué hay en los datos.*

Las imágenes SAR son muy distintas a imágenes RGB estándar: el ruido es multiplicativo
(speckle), el rango dinámico es enorme y las RFI se manifiestan como líneas de alta energía
que cruzan toda la escena. Conocer la distribución de tamaños, densidades y orientaciones
es crítico para elegir bien el modelo y la estrategia de slicing.

```bash
python -m analysis.complete_dataset_analysis   # EDA completo: tamaños, orientación, outliers
python -m analysis.optimize_slicing_params     # Calcula slice_height/overlap óptimos para inferencia
python -m analysis.image_browser_simple        # Inspección visual imagen a imagen con anotaciones
python -m analysis.filter_optimization         # Compara filtros (CLAHE, Top-hat, LoG, Lee) visualmente
```

**Qué buscar:**
- Ratio de boxes pequeños (< 32×32 px) → si es alto, el slicing es imprescindible
- Orientación dominante → las RFI son casi siempre horizontales, confirmar en el dataset
- Imágenes con muchas boxes (escenas densas) → afecta al umbral NMS
- `optimize_slicing_params.py` genera un JSON con 3 configs (conservative / balanced / aggressive)
  que luego se pasan directamente a `src/inference.py`

---

## Fase 2 — Preparación de datos
*Convertir las anotaciones COCO a formato YOLO y crear los splits.*

`src/train.py` hace esto automáticamente al arrancar: convierte el JSON COCO a labels `.txt`,
aplica un split estratificado train/val/holdout y genera el `data.yaml` que necesita YOLO.

- **Holdout**: subconjunto reservado que nunca ve el modelo durante el entrenamiento.
  Se usa exclusivamente para evaluar el modelo entrenado antes de subir al leaderboard.
- Si se usan pseudo-labels (Fase 5), se parte de `instances_train_og.json` (original)
  y se genera `instances_train_pseudo.json`.

```bash
# El split se genera solo al lanzar el entrenamiento, no hay paso separado
```

---

## Fase 3 — Entrenamiento
*Entrenar el detector YOLO sobre las imágenes SAR.*

```bash
# Run básico (un fold)
python -m src.train --model yolo26n --epochs 100 --batch-size 32 \
                    --image-size 512 --lr 0.001 --lrf 0.01 --kfold 1

# K-fold cross-validation (más robusto, más lento)
python -m src.train --model yolo26n --epochs 100 --kfold 5
```

**Consideraciones SAR:**
- `image-size 512` es el punto de partida razonable; las RFI pequeñas se pierden a menor resolución
- Augmentaciones: evitar flips verticales (las RFI son horizontales), mantener flips horizontales
- Modelos más pequeños (yolo26n) generalizan mejor en este dominio que modelos grandes sobreentrenados
- Registrar cada experimento en `experiments.md` con el comando exacto y el mAP resultante

---

## Fase 4 — Inferencia sobre test
*Generar la submission para el leaderboard.*

```bash
# Evaluación en holdout + submission en test (modo por defecto)
python -m src.inference --checkpoint models/best.pt --mode both --image-size 512

# Solo submission
python -m src.inference --checkpoint models/best.pt --mode test
```

**Slicing**: para imágenes grandes, la inferencia en un solo paso pierde boxes pequeños.
Usar los parámetros de slicing calculados en Fase 1:
```bash
python -m src.inference --checkpoint models/best.pt \
       --slice-height 128 --slice-overlap 32 --slice-max-height-px 48
```

**Ensemble** (si hay varios modelos entrenados):
```bash
python submissions/ensemble.py --models run0.pt,512 run1.pt,640 \
       --input-json data/annotations/instances_train_og.json
```

---

## Fase 5 — Evaluación del modelo
*Entender por qué el modelo falla y dónde hay margen de mejora.*

Esta fase viene **después del entrenamiento**, usando el holdout como conjunto de evaluación.

```bash
# Parts 1+2: estadísticas del GT + visualizaciones (no necesita modelo)
python -m analysis.model_eval                              # run_model=False por defecto

# Part 3: evaluación completa con el modelo entrenado
# Editar model_eval.py: main(run_model=True, model_path="models/best.pt")
python -m analysis.model_eval
```

| Parte | Contenido | Cuándo |
|---|---|---|
| Part 1 · Dataset stats | Imágenes densas, boxes pequeños/grandes, % por categoría | Antes o después de entrenar |
| Part 2 · Visualization | Distribución de áreas, alturas, boxes/imagen | Antes o después de entrenar |
| Part 3 · Model evaluation | Precision/Recall/F1 por tamaño, ejemplos FP y FN visuales | Solo después de entrenar |

> **Nota**: Parts 1 y 2 pueden correrse también en Fase 1 como análisis previo al entrenamiento.
> Part 3 requiere un `.pt` entrenado y el conjunto holdout.

**Qué buscar en los errores:**
- FN en boxes pequeños → bajar `slice_max_height_px`, aumentar resolución de inferencia
- FP en zonas de alto contraste → ajustar umbral de confianza o añadir augmentaciones de ruido
- FN en escenas densas → bajar NMS IoU threshold

---

## Fase 6 — Iteración avanzada
*Técnicas para exprimir más mAP una vez que el baseline funciona.*

### Pseudo-labeling
Usar el modelo actual para etiquetar imágenes sin label (o con baja confianza)
y añadirlas al entrenamiento. Solo vale la pena si el modelo ya supera ~0.35 mAP.

```bash
python submissions/pseudo_labeler.py \
       --models run0.pt,512 run1.pt,640 \
       --conf-thresh 0.15 \
       --output-json data/annotations/instances_train_pseudo.json
```

### Ensemble de modelos
Combinar predicciones de varios modelos con Weighted Box Fusion (WBF).
Mejora la robustez especialmente en boxes pequeños y escenas densas.

```bash
python submissions/ensemble.py --models run0.pt,512 run1.pt,640 run2.pt,512
```

---

## Registro de experimentos

Ver `experiments.md` para el historial completo de runs con comandos y métricas.

| Run | Modelo | mAP val | mAP leaderboard | Notas |
|---|---|---|---|---|
| run0 | yolo26n | 0.325 | 0.356 | Baseline, 50 épocas |

---

## Referencia rápida de scripts

| Script | Fase | Descripción |
|---|---|---|
| `analysis/complete_dataset_analysis.py` | 1 | EDA completo del dataset |
| `analysis/optimize_slicing_params.py` | 1 | Parámetros óptimos de slicing |
| `analysis/image_browser_simple.py` | 1 | Inspector visual de imágenes |
| `analysis/filter_optimization.py` | 1 | Comparación de filtros de preprocesado |
| `src/train.py` | 3 | Entrenamiento YOLO (incluye conversión y splits) |
| `src/inference.py` | 4 | Inferencia en holdout y test, genera submission |
| `analysis/model_eval.py` | 1/5 | Análisis de errores y métricas del modelo |
| `submissions/ensemble.py` | 4/6 | Ensemble con WBF sobre múltiples modelos |
| `submissions/pseudo_labeler.py` | 6 | Generación de pseudo-labels con ensemble |
