# Experiments — ClearSAR

Formato de cada run:
- **comando**: exactamente lo ejecutado
- **mAP val**: métrica sobre el split de validación al final del entrenamiento
- **mAP test teórico**: estimación holdout antes de subir al leaderboard
- **mAP leaderboard**: resultado oficial tras la submission
- **explicacion**: qué se aprendió, por qué subió o bajó

---

# run0

Baseline. yolo26n, imágenes originales RGB (colormap SAR), 50 épocas.

## comando

```powershell
& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --run-name run0

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.inference --checkpoint models/yolo_run0.pt --mode both --image-size 512 --save-holdout-viz
```

## resultado mAP val

0.329 (mAP50-95 best epoch 43 | mAP50=0.618 | prec=0.652 | rec=0.575)
*(run anterior: 0.325 — ligera variación por aleatoriedad)*

## resultado mAP test teórico

0.358 (holdout pycocotools | mAP50=0.626 | mAP75=0.378 | recall=0.625)

## resultado mAP leaderboard

0.3561

## explicacion

Baseline re-entrenado para comparación justa con holdout pycocotools. Test: 478.52 boxes/img a conf=0.0 — similar a los runs de stacking (~493), confirma que el over-detection es comportamiento general del modelo a conf=0.0 y no artefacto del stacking. Holdout (0.358) > todos los runs de stacking (run3=0.331, run1=0.326).

--------------------

# run1 — Stacking [Gray | CLAHE | Bilateral+TopHat] [TRAINED for 50 epochs]

Primer experimento de filter stacking. Canal continuo + mejorado + detector binario.

**Canales**:
- Ch1 (R): Gray — intensidad SAR bruta
- Ch2 (G): CLAHE — clip_limit=1.5, tile_size=6
- Ch3 (B): Bilateral+TopHat — d=7, sigma_color=50, sigma_space=50, kernel_w=17 (binario)

## comando

```powershell
& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --train-images-dir data/images/train_run1 --run-name run1

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.preprocess --combo run1 --split holdout

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.inference --checkpoint models/yolo_run1.pt --run-name run1 --mode both --image-size 512 --test-images-dir data/images/test_run1 --holdout-images-dir data/images/holdout_run1 --save-holdout-viz
```

## resultado mAP val

0.302 (mAP50-95 best epoch 41 | mAP50=0.580 | prec=0.631 | rec=0.555)

## resultado mAP test teórico

0.326 (holdout pycocotools | mAP50=0.583 | mAP75=0.337 | recall=0.598)

## resultado mAP leaderboard



## explicacion

Holdout coherente con val (0.302 val → 0.326 holdout). Similar al baseline run0 (val 0.325). Test: 493.99 boxes/img a conf=0.0 — desconocido si run0 también lo tenía. Pendiente comparar con run3 antes de decidir si submitir.

--------------------

# run2 — Stacking [Gray | CLAHE | Gabor] [TRAINED for 50 epochs]

**Canales**:
- Ch1 (R): Gray
- Ch2 (G): CLAHE — clip_limit=1.5, tile_size=6
- Ch3 (B): Gabor Horiz — ksize=13, sigma=3.5, lam=8.0, gamma=0.5 (binario, #1 ranking)

## comando

```powershell
& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --train-images-dir data/images/train_run2 --run-name run2

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.preprocess --combo run2 --split holdout

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.inference --checkpoint models/yolo_run2.pt --run-name run2 --mode both --image-size 512 --test-images-dir data/images/test_run2 --holdout-images-dir data/images/holdout_run2 --save-holdout-viz
```

## resultado mAP val

0.294 (mAP50-95 best epoch 43 | mAP50=0.551 | prec=0.648 | rec=0.482)

## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion

El peor de los runs con canal continuo. Gabor binario como canal 3 es demasiado agresivo — binariza el 80% de la imagen en negro y el backbone pierde contexto. Menor recall que run1/run3.

--------------------

# run3 — Stacking [Gray | CLAHE | Log] [TRAINED for 50 epochs]

Todo continuo, sin máscaras binarias. Ideal para RFI débil o sutil.

**Canales**:
- Ch1 (R): Gray
- Ch2 (G): CLAHE — clip_limit=1.5, tile_size=6
- Ch3 (B): Log Transform — gain=10.0

## comando

```powershell
& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --train-images-dir data/images/train_run3 --run-name run3

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.preprocess --combo run3 --split holdout

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.inference --checkpoint models/yolo_run3.pt --run-name run3 --mode both --image-size 512 --test-images-dir data/images/test_run3 --holdout-images-dir data/images/holdout_run3 --save-holdout-viz
```

## resultado mAP val

0.309 (mAP50-95 best epoch 43 | mAP50=0.578 | prec=0.671 | rec=0.549)

## resultado mAP test teórico

0.331 (holdout pycocotools | mAP50=0.581 | mAP75=0.353 | recall=0.612)

## resultado mAP leaderboard



## explicacion

Mejor holdout de todos los runs de stacking (0.331 > run1 0.326). Canales 100% continuos sin binarización. Test: 493.38 boxes/img — igual que run1, confirma que no es específico del canal binario sino del conf=0.0 general. Candidato principal para submit.

--------------------

# run4 — Stacking [Gray | Gabor | Bilateral+TopHat] [TRAINED for 50 epochs]

Los 2 mejores filtros del ranking + grayscale de contexto. Sin CLAHE.

**Canales**:
- Ch1 (R): Gray
- Ch2 (G): Gabor Horiz — ksize=13, sigma=3.5, lam=8.0, gamma=0.5 (#1 ranking)
- Ch3 (B): Bilateral+TopHat — d=7, sigma_color=50, sigma_space=50, kernel_w=17 (#2 ranking)

## comando

```powershell
& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --train-images-dir data/images/train_run4 --run-name run4

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.preprocess --combo run4 --split holdout

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.inference --checkpoint models/yolo_run4.pt --run-name run4 --mode both --image-size 512 --test-images-dir data/images/test_run4 --holdout-images-dir data/images/holdout_run4 --save-holdout-viz
```

## resultado mAP val

0.268 (mAP50-95 best epoch 43 | mAP50=0.527 | prec=0.609 | rec=0.524)

## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion

Peor resultado de todos. Dos canales binarios (Gabor + B+TopHat) dominan la imagen — el backbone ve casi todo en negro/blanco sin gradiente. Confirma que los filtros binarios en mayoría son contraproducentes.

--------------------
