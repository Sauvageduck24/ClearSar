# PLANNING — ClearSAR Filter Stacking Experiment

## Situación actual

Baseline funcional: `yolo26n`, 50 épocas, mAP leaderboard **0.356**.

Las imágenes del dataset son imágenes SAR de intensidad (escala de grises) exportadas
como RGB con un colormap. Los 3 canales RGB dicen esencialmente lo mismo — no hay
información SAR independiente por canal.

**Hipótesis**: construir una imagen de 3 canales donde cada uno tenga una señal útil
distinta le da al modelo más información para detectar RFI, especialmente los débiles.

**Riesgo conocido**: YOLO preentrenado en ImageNet tarda más en adaptarse a canales
sintéticos. Solución: mismas épocas en todos los experimentos para comparación justa.

---

## Ranking de filtros (filter_optimization.py — 500 imágenes)

| # | Filtro | Score | Parámetros óptimos |
|---|---|---|---|
| 1 | Gabor Horiz | 7.18 | ksize=13, sigma=3.5, lam=8.0, gamma=0.5 |
| 2 | Bilateral + Top-Hat | 1.35 | d=7, sigma_color=50, sigma_space=50, kernel_w=17 |
| 3 | Log Transf | 0.61 | gain=10.0 |
| 4 | Top-Hat (Horiz) | 0.46 | kernel_w=25 |
| 5 | CLAHE | 0.44 | clip_limit=1.5, tile_size=6 |
| 6 | Original | 0.39 | — |

Gabor y Bilateral+TopHat devuelven **máscaras binarias** (0/255) — buenos cuando el RFI
es fuerte, pero pierden señal en RFI débil o sutil. CLAHE y Log son continuos — preservan
RFI débil aunque con menos contraste que los binarios.

---

## Experimentos planificados

Todos con: `yolo26n`, 50 épocas, batch 32, image-size 512, lr 0.001, kfold 1.
Punto de comparación: run0 (original RGB) → mAP val 0.325 / leaderboard 0.356.

| Run | Ch1 | Ch2 | Ch3 | Razonamiento |
|---|---|---|---|---|
| run0 | R | G | B | Baseline — original tal cual |
| run1 | Gray | CLAHE | B+TopHat | **Empezamos aquí** — continuo mejorado + detector binario |
| run2 | Gray | CLAHE | Gabor | Continuo + el mejor detector binario |
| run3 | Gray | CLAHE | Log | Todo continuo — ideal para RFI débil |
| run4 | Gray | Gabor | B+TopHat | Los 2 mejores del ranking + contexto |
| run5 | CLAHE | Gabor | B+TopHat | Sin gray crudo, máxima señal filtrada |

El ganador de run1-5 se lleva 100 épocas y genera submission al leaderboard.

---

## TODO

### Paso 1 — Decidir qué filtros usar ✅
- [x] Ejecutar `analysis/filter_optimization.py` y analizar resultados visualmente
- [x] Definir las 5 combinaciones a probar
- [x] Decidir orden: empezar por run1 [Gray, CLAHE, B+TopHat]

### Paso 2 — Generar imágenes stackeadas en disco ✅
- [x] Crear `src/preprocess.py` — genera todas las combinaciones en disco con multiprocessing
  - Fuente: `data/images/train/` y `data/images/test/`
  - Destino: `data/images/train_<combo>/` y `data/images/test_<combo>/`
  - Una carpeta por combinación (run1 … run5)
- [x] Ejecutar y verificar que las imágenes se generan correctamente

### Paso 3 — Verificación visual antes de entrenar ✅
- [x] Añadir modo `--combo` a `analysis/image_browser_simple.py`:
  - Panel superior: Original + boxes | Stacked + boxes
  - Panel inferior: Ch1 | Ch2 | Ch3 (cada uno en escala de grises con boxes)
  - Lee directamente de `data/images/train_<combo>/`, no calcula nada on-the-fly
- [x] Verificado visualmente — los canales contienen la señal esperada

### Paso 4 — Entrenamientos comparativos (run1 → run5)

Todos los runs usan exactamente los mismos hiperparámetros que el baseline (run0).
Solo cambia `--train-images-dir`. Comando:

```powershell
& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --train-images-dir data/images/train_<runX> --run-name <runX>
```

| Run | Ch1 | Ch2 | Ch3 | mAP50-95 val (best) |
|---|---|---|---|---|
| run1 | Gray | CLAHE | B+TopHat | 0.302 |
| run2 | Gray | CLAHE | Gabor | 0.294 |
| run3 | Gray | CLAHE | Log | **0.309** |
| run4 | Gray | Gabor | B+TopHat | 0.268 |

- [x] run1 — mAP50-95=0.302
- [x] run2 — mAP50-95=0.294
- [x] run3 — mAP50-95=0.309 ← mejor de los 4
- [x] run4 — mAP50-95=0.268
- [x] Registrar cada resultado en `experiments.md`

**Nota**: todos los runs de stacking están por debajo del val baseline (0.325), pero el holdout de run3 (0.331) y run1 (0.326) superan o igualan el val del baseline. Run3 es el candidato a submit.

### Paso 5 — Inferencia con preprocessing ✅ (parcial)
- [x] Modificar `src/inference.py`: añadido `--holdout-images-dir` y `--test-images-dir` para cargar imágenes pre-stackeadas desde disco (sin on-the-fly)
- [x] Añadir split `holdout` a `src/preprocess.py` → genera `data/images/holdout_<combo>/`
- [ ] Ejecutar inferencia para cada run entrenado (ver comandos en `experiments.md`)
- [ ] Generar submission del ganador y subir al leaderboard
- [ ] Comparar con baseline (run0: 0.356)

### Paso 6 — Iteración final ✅ (conclusión de stacking)

Resultado: el filter stacking **no mejora** el baseline con yolo26n a 50 épocas.

| Run | val | holdout | leaderboard |
|-----|-----|---------|-------------|
| run0 (baseline) | 0.329 | **0.358** | **0.356** |
| run3 Gray+CLAHE+Log | 0.309 | 0.331 | — |
| run1 Gray+CLAHE+B+TopHat | 0.302 | 0.326 | — |

Gap holdout→leaderboard en run0 = -0.002. Run3 leaderboard estimado ~0.329 < 0.356 → no vale submitir.

---

## Fase 2 — Próximos pasos (por decidir)

### Opción A — Modelo más grande (mayor ROI esperado)
Entrenar yolo26s o yolo11m sobre imágenes originales. yolo26n es nano — el backbone simplemente no tiene capacidad para features complejas de RFI. Un modelo medium probablemente suba 3-6 puntos solo por capacidad.

### Opción B — TTA en run0 (gratis, sin reentrenar)
Test Time Augmentation con `augment=True` en inference. Puede sumar 1-2 puntos sin coste. Solo hay que añadir el flag y re-submitir el run0 actual.

### Opción C — Run3 con 100 épocas
Apuesta a que el backbone necesita más tiempo para adaptarse a canales sintéticos. Riesgo: el gap (0.358 vs 0.331) es grande para cerrar solo con épocas. ROI incierto.

---

## Decisiones tomadas

| Decisión | Motivo |
|---|---|
| Guardar en disco, no on-the-fly | Entrenamiento más rápido, verificable visualmente, idéntico en inferencia |
| Mismas épocas en todos los runs | Comparación justa — stacking necesita tiempo para que el backbone se adapte |
| Empezar con run1 [Gray, CLAHE, B+TopHat] | Equilibrio continuo+binario, cubre tanto RFI débil como fuerte |
| yolo26n en todos | Aislar el efecto del stacking, no mezclar variables |
| Generar todas las combos antes de entrenar | El preprocessing es barato, así no se interrumpe el flujo de experimentos |
