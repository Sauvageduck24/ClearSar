# run0

yolov9s 50 epocas sin kfold y con slicing

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline_slicing.py --yolo-model yolov9s --image-size 512 --mapping-path catalog.v1.parquet --yolo-extra-args "--batch-size 20 --lr 0.0005 --epochs 50 --num-workers 12 --kfold 1 --slice-height 160 --slice-overlap 40" --yolo-inference "--slice-height 160 --slice-overlap 40 --slice-max-height-px 32 --slice-nms-iou 0.3"

## resultado mAP val

0.377

## resultado mAP test teórico

0.4016 - 0.421 = 0.4113

## resultado mAP leaderboard

0.4147

## explicacion resultado

ha estado bien pero se ha quedado estancado el modelo y no progresaba, aunque con mas entrenamiento se podria conseguir mas

----------

# run1

yolov9s 50 epocas sin kfold y con slicing y con slice hiehgt y slice overlap distintos

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline_slicing.py --yolo-model yolov9s --image-size 512 --mapping-path catalog.v1.parquet --yolo-extra-args "--batch-size 20 --lr 0.0005 --epochs 10 --num-workers 12 --kfold 1 --slice-height 160 --slice-overlap 40" --yolo-inference "--slice-height 256 --slice-overlap 64 --slice-max-height-px 32 --slice-nms-iou 0.3"

## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado



--------------------



# run2

yolo11s 20 epocas sin kfold ni slices ni preentrenamiento con instances train clean brightness_threshold=0.25

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline_slicing.py --yolo-model yolo11s --image-size 512 --mapping-path catalog.v1.parquet --yolo-extra-args "--batch-size 20 --lr 0.0003 --lrf 0.03 --epochs 20 --num-workers 12 --kfold 1 --no-slicing --annotation-path data/annotations/instances_train_clean.json" --yolo-inference="--no-slicing"

## resultado mAP val

0.317

## resultado mAP test teórico

0.35-0.36

## resultado mAP leaderboard



## explicacion resultado


--------------------



# run3

yolov9s 20 epocas sin kfold ni slices ni preentrenamiento

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline_slicing.py --yolo-model yolo11s --image-size 512 --mapping-path catalog.v1.parquet --yolo-extra-args "--batch-size 20 --lr 0.0003 --lrf 0.03 --epochs 20 --num-workers 12 --kfold 1 --no-slicing" --yolo-inference="--no-slicing"

## resultado mAP val

0.3405

## resultado mAP test teórico

0.3755 - 0.3819

## resultado mAP leaderboard



## explicacion resultado



--------------------


# run4



## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline_slicing.py --yolo-model yolo26n --image-size 512 --mapping-path catalog.v1.parquet --yolo-extra-args "--batch-size 32 --lr 0.006 --lrf 0.03 --epochs 150 --num-workers 12 --kfold 1 --no-slicing --annotation-path data/annotations/instances_train_og.json" --yolo-inference="--no-slicing"

## resultado mAP val

0.287 

## resultado mAP test teórico

0.31-0.32 

## resultado mAP leaderboard



## explicacion resultado



----------



# run5

descripcion

## comando



## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado



--------------------



# run6

descripcion

## comando



## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado



-------------


# ensemble0

ensemble de modelos, dados modelos y tamaños de imagenes, aunque ahora se que lo recomendado es que todos sean de 512, mas no merece la pena y menos es a peor

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m submissions.ensemble --models submissions/run0.pt,512 submissions/run1.pt,640

## map teorico 

0.391

## map real

0.4078

-----------

# pseudo labeler

por si acaso queremos intentar subir un modelo, o un ensemble, generamos boxes para los datos de train, los juntamos con los originales y asi podemos conseguir un extra score

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.pseudo_labeler --models submissions/run0.pt,512 submissions/run1.pt,640

# probar cualquier .pt

para probar una inferencia de cualquier modelo y ver lo que saca en el set de holdout podemos probar esto

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline_slicing.py --skip-train --yolo-model yolo11s --image-size 512 --yolo-checkpoint submissions/run0.pt --yolo-inference=" --no-slicing --mode 'holdout' "

# ssl pretrain

este sirve para poder preentrenar el modelo yolo a los datos de train y de test, asi se acostumbra el backbone a las texturas del dataset

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.ssl_pretrain_yolo --model yolov9s --images-dirs data/images/train data/images/test --output models/yolov9s_ssl.pt --epochs 100 --batch-size 60 --image-size 512 --num-workers 6

-------

# curriculum learning

curriculum learning, un primer entrenamiento con las imagenes limpias sin problemas, y otro segundo entrenamiento con las imagenes problematicas

## comando

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline_slicing.py  --yolo-model models/yolov9s_ssl.pt --image-size 512 --mapping-path catalog.v1.parquet --yolo-extra-args "--batch-size 20 --lr0_fase1 0.0004 --lrf_fase1 0.05 --lr0_fase2 0.00015 --lrf_fase2 0.02 --epochs 150 --num-workers 12 --kfold 1 --no-slicing" --yolo-inference=--no-slicing --curriculum-learning --curriculum-epochs 40

## resultado mAP val



## resultado mAP test teórico



## resultado mAP leaderboard



## explicacion resultado

