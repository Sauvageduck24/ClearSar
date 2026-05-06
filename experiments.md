# run0

baseline run , yolo26n, 50 epocas

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.inference --checkpoint models/yolo_baseline.pt --mode both --image-size 512 --save-holdout-viz

## resultado mAP val

0.325

## resultado mAP test teórico

0.357

## resultado mAP leaderboard

0.3561

## explicacion resultado


--------------------


# run1

baseline + altura minima 8 px

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --annotation-path data/annotations/instances_train_forzar_minimo.json

## resultado mAP val

0.3931

## resultado mAP test teórico

0.319

## resultado mAP leaderboard



## explicacion resultado

--------------------


# run2

ejecutar con feature injection

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1

## resultado mAP val

0.317

## resultado mAP test teórico

0.32

## resultado mAP leaderboard



## explicacion resultado

--------------------

# run3

baseline run + eliminar boxes que score snr < 0.1

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --train-snr-threshold 0.1

## resultado mAP val

0.325

## resultado mAP test teórico

0.349

## resultado mAP leaderboard



## explicacion resultado

--------------------

# run4

baseline + h < 40

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --train-max-box-height 40

## resultado mAP val

0.322

## resultado mAP test teórico

0.336

## resultado mAP leaderboard



## explicacion resultado

--------------------

# run5

baseline + juntar boxes

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --train-merge-contiguous-boxes True

## resultado mAP val

0.28

## resultado mAP test teórico

0.29

## resultado mAP leaderboard



## explicacion resultado

--------------------

# run6

baseline + eliminar boxes de menos de 6 px

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --remove-small 6

## resultado mAP val

0.324

## resultado mAP test teórico

0.355

## resultado mAP leaderboard



## explicacion resultado

--------------------

# run7

baseline + eliminar h>=w

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --skip-vertical-boxes True

## resultado mAP val

0.317

## resultado mAP test teórico

0.318

## resultado mAP leaderboard



## explicacion resultado

--------------------


# run8

filtro extra

actual

r => radar vv (vertical vertical)
g => radar vh (vertical horizontal)
b => vv/vh

cambiar

b por => (vv-vh) en absoluto

## comando



## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado

--------------------

# run9

copy paste dirigido

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --copy-paste-n 2

## resultado mAP val

0.339

## resultado mAP test teórico

0.358

## resultado mAP leaderboard



## explicacion resultado

--------------------

# run10

baseline + cls = 0.3 + box loss = 10.0 + dfl_loss_weight=2.0

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --box 10.0 --cls 0.3 --dfl 2.0

## resultado mAP val

0.325

## resultado mAP test teórico

0.355

## resultado mAP leaderboard



## explicacion resultado

--------------------

# run11

modelo11n con copy paste especifico

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo11n --epochs 50 --image-size 512 --batch-size 32 --num-workers 8 --lr 0.001 --lrf 0.01 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.3 --dfl 2.0 --kfold 1

## resultado mAP val

0.338

## resultado mAP test teórico

0.363

## resultado mAP leaderboard



## explicacion resultado



----------------------


# run12

baseline + std multi norm

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --std-multi-norm true

## resultado mAP val

0.305

## resultado mAP test teórico

0.334

## resultado mAP leaderboard



## explicacion resultado

----------------------------

# run13

baseline + --tal-topk 15

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --tal-topk 15

## resultado mAP val

0.324

## resultado mAP test teórico

0.346

## resultado mAP leaderboard



## explicacion resultado

-------------------

# run14

baseline + --reorder-channels

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --reorder-channels True

## resultado mAP val

0.321

## resultado mAP test teórico

0.343

## resultado mAP leaderboard



## explicacion resultado

--------------------

# run15

baseline + --inject-wavelet

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --inject-wavelet True

## resultado mAP val

0.324

## resultado mAP test teórico

0.345

## resultado mAP leaderboard



## explicacion resultado

-------------

# run16

baseline + cbam (cambiando yolo26.yaml)

## comando



## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado

--------

# run17

baseline + augmentaciones especificas

channel shuffle
channel dropout
así deja de confiar en “verde = RFI”

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --specific-augmentations true

## resultado mAP val

0.322

## resultado mAP test teórico

0.356

## resultado mAP leaderboard



## explicacion resultado


----------

# run18

modelo yolo11s y mejor configuracion, es decir, custom copy paste y loss

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo11s --epochs 100 --image-size 640 --batch-size 16 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.3 --dfl 2.0  --kfold 1

## resultado mAP val

0.379

## resultado mAP test teórico

0.395

## resultado mAP leaderboard



## evolución entrenamiento


--------

# run19

igual que run18 pero con yolo26s

modelo yolo11s y mejor configuracion, es decir, custom copy paste y loss

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolov26s --epochs 100 --image-size 640 --batch-size 16 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.3 --dfl 2.0  --kfold 1

## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## evolución entrenamiento


----------

# run20

yolo11n pero con reg_max 64 modificando el head con un yaml especifico. yolo11.yaml

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo11n --model-yaml yolo11.yaml --epochs 50 --image-size 512 --batch-size 32 --num-workers 8 --lr 0.001 --lrf 0.01 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.3 --dfl 2.0 --kfold 1

## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## evolución entrenamiento


--------

# run21

yolo11n pero con p2 modificando el head con un yaml especifico. yolo11-p2.yaml

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo11n --model-yaml yolo11-p2.yaml --epochs 50 --image-size 512 --batch-size 16 --num-workers 8 --lr 0.001 --lrf 0.01 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.3 --dfl 2.0 --kfold 1

## resultado mAP val

0.335

## resultado mAP test teórico

0.37

## resultado mAP leaderboard



## explicacion resultado


----------

# run22

yolo11n pero con resnet18. usando yolo-resnet18.yaml

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo11n --model-yaml yolo11-resnet18.yaml --epochs 50 --image-size 512 --batch-size 32 --num-workers 8 --lr 0.001 --lrf 0.01 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.3 --dfl 2.0 --kfold 1

## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado

--------

# run23



## comando



## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado


----------

# run24

probar rtdetr preentrenado

"from ultralytics import RTDETR
from torchgeo.models import resnet50, ResNet50_Weights
import torch

# 1. Cargar backbone con pesos SAR
sar_backbone = resnet50(weights=ResNet50_Weights.SENTINEL1_ALL_MOCO)

# 2. Crear modelo RT-DETR
model = RTDETR('rtdetr-resnet50.yaml')

# 3. Sustituir el backbone con los pesos SAR
# Los primeros layers del backbone son compatibles
backbone_state = {
    k: v for k, v in sar_backbone.state_dict().items()
    if not k.startswith('fc')  # excluir cabeza de clasificación
}
model.model.backbone.load_state_dict(backbone_state, strict=False)

print("Backbone SAR cargado correctamente")

weights = ResNet50_Weights.SENTINEL1_ALL_MOCO
transform = weights.transforms()
# Esto incluye mean/std específicos de Sentinel-1
# Aplícalo en _link_images o en _convert_coco_to_yolo
"

## comando



## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado


--------------------


# run25

yolo11s con pesos de torchgeo

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo11s-torchgeo.pt --epochs 100 --image-size 640 --batch-size 16 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.3 --dfl 2.0  --kfold 1

## resultado mAP val

peor que normal. malo malo.

## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado

--------------------


# run26

baseline + label smoothing

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --label-smoothing 0.1

## resultado mAP val

0.33

## resultado mAP test teórico

0.364

## resultado mAP leaderboard



## explicacion resultado

--------------------


# run27

baseline + [vv,vh,max(vv,vh)]

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --vv-vh-max true

## resultado mAP val

0.319

## resultado mAP test teórico

0.33

## resultado mAP leaderboard



## explicacion resultado


----------------------


# run28

baseline + hard negative mining

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --hard-negative-mining true

## resultado mAP val

0.324

## resultado mAP test teórico

0.359

## resultado mAP leaderboard



## explicacion resultado


-----------------


# run29

baseline sin mosaic

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --mosaic 0.0

## resultado mAP val

0.319

## resultado mAP test teórico

0.337

## resultado mAP leaderboard



## explicacion resultado


-----------------


# run30

baseline pero sin letterboxing, es decir haciendo resize antes, y sin importar la perdida de proporciones

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1

## resultado mAP val

0.335

## resultado mAP test teórico

0.37

## resultado mAP leaderboard



## explicacion resultado


-----------------


# run31

baseline + con slicing y estiramiento

## comando

/usr/bin/python -m src.train --model yolo26n --epochs 50 --batch-size 50 --image-size 512 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --slicing true --slice-height 64 --slice-height-overlap 0.25 --slice-width 512 --slice-width-overlap 0

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.inference --checkpoint models/yolo_best_yolo26n.pt --mode both --image-size 512 --slicing true --slice-height 64 --slice-overlap 0.25 --batch-size 32 --postprocess-workers 8

## resultado mAP val

0.368

## resultado mAP test teórico

0.403

## resultado mAP leaderboard



## explicacion resultado


-----------------


# run32

run33 pero con yolo26s

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26s --epochs 50 --batch-size 8 --image-size 736 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1

## resultado mAP val

0.392

## resultado mAP test teórico

0.411

## resultado mAP leaderboard



## explicacion resultado


-----------------


# run33

baseline con resize y estiramiento. image size 736

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 32 --image-size 736 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.inference --checkpoint models/yolo_best_yolo26n.pt --mode both --image-size 736 --save-holdout-viz

## resultado mAP val

0.365

## resultado mAP test teórico

0.3897 - 0.4

## resultado mAP leaderboard

0.3805

## explicacion resultado


-----------------


# run33

baseline con y*2 y con 768 sin letterboxing

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 8 --image-size 768 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --apply-letterboxing false --y-component-multiplier 2.0

## resultado mAP val

0.371

## resultado mAP test teórico

0.396

## resultado mAP leaderboard



## explicacion resultado


-----------------


# run34

baseline con y*2 y con 768 con letterboxing

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 8 --image-size 768 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --apply-letterboxing true --y-component-multiplier 2.0

## resultado mAP val

0.377

## resultado mAP test teórico

0.388

## resultado mAP leaderboard



## explicacion resultado


-----------------



# run35

run 33 pero con yolo26s

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 8 --image-size 768 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --apply-letterboxing false --y-component-multiplier 2.0

## resultado mAP val

0.387

## resultado mAP test teórico

0.416

## resultado mAP leaderboard



## explicacion resultado


-----------------


# run36

sin component y multiplier

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26n --epochs 50 --batch-size 8 --image-size 768 --num-workers 8 --lr 0.001 --lrf 0.01 --kfold 1 --apply-letterboxing false --y-component-multiplier 1.0

## resultado mAP val

0.363

## resultado mAP test teórico

0.39

## resultado mAP leaderboard



## explicacion resultado

-------------------

# run37

intento de mejora de run36 pero usando copy paste mas definido, mas epocas, mayor dfl

pasarle a claude la salida de consola a ver que le parece si se podria usar 150 epocas o algo o cambiar el lr?

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26s --epochs 100 --image-size 768 --batch-size 8 --num-workers 8 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.3 --dfl 2.5  --kfold 1 --lr 0.001 --lrf 0.01 --apply-letterboxing false --y-component-multiplier 2.0

## resultado mAP val

0.408

## resultado mAP test teórico

0.427

## resultado mAP leaderboard



## explicacion resultado


-------------------

# run38

run37 pero con hard negative mining y label smoothing

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo26s --epochs 100 --image-size 768 --batch-size 8 --num-workers 8 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.3 --dfl 2.5  --kfold 1 --lr 0.001 --lrf 0.01 --apply-letterboxing false --y-component-multiplier 2.0 --hard-negative-mining true --label-smoothing 0.0

## resultado mAP val

0.395

## resultado mAP test teórico

0.427

## resultado mAP leaderboard



## explicacion resultado

-------------------

# run39

run38 pero con --mosaic 0.8 y --cls 0.2 --remove-small 3 y image size 1280

## comando

/usr/bin/python -m src.train --model yolo26s --epochs 100 --image-size 1280 --batch-size 8 --num-workers 8 --small-box-copy-paste True --copy-paste-p 0.5 --copy-paste-max-h 10 --box 10.0 --cls 0.2 --dfl 2.5  --kfold 1 --lr 0.001 --lrf 0.01 --apply-letterboxing false --y-component-multiplier 2.0 --hard-negative-mining true --label-smoothing 0.0 --mosaic 0.8 --remove-small 3

## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado



-------------------

# run40



## comando



## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado

-------------------

# run41



## comando



## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado

-------------------

# run42



## comando



## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado


-----------------

probar 1280

probar modelos mas grandes ademas del s

probar otros modelos como yolo11 o yolov9

probar --multi-scale true



& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.train --model yolo11s --epochs 120 --image-size 736 --batch-size 8 --num-workers 8 --lr 0.002 --lrf 0.01 --apply-letterboxing false --y-component-multiplier 2.0 --kfold 1 --scale 0.02 --translate 0.01 --mosaic 1.0 --close-mosaic 20 --box 7.5 --cls 0.5 --dfl 1.5 --label-smoothing 0.0 --small-box-copy-paste true --copy-paste-p 0.3 --copy-paste-max-h 12.0 --copy-paste-n 3 --hard-negative-mining false 


& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.detr_simple_train --epochs 30 --batch-size 1 --image-size 512 --lr 1e-4 --num-workers 2 --val-fraction 0.1