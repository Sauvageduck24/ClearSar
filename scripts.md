# Leaderboard de modelos que poder usar
https://leaderboard.roboflow.com/

# Primero entrena Faster R-CNN

## big model
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window1 --ensemble-top-k 3 --train-extra-args "--image-size 1024 --epochs 50 --batch-size 4 --num-workers 8 --grad-accum-steps 4 --lr 1e-4" --mapping-path catalog.v1.parquet

### resultado

"python run_pipeline.py --window1 --ensemble-top-k 3 \
--train-extra-args "--arch fasterrcnn_resnet50_fpn_v2 --image-size 640 --epochs 50 --batch-size 8 --num-workers 12 --grad-accum-steps 2 --lr 2e-4" \
--mapping-path catalog.v1.parquet"

se queda estancado entorno a 0.3 no escala bien

mAP val => 
mAP leaderboard => 

## small model
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window1 --ensemble-top-k 3 --train-extra-args "--arch fasterrcnn_mobilenet_v3_large_fpn --epochs 50 --batch-size 4 --num-workers 4 --grad-accum-steps 4" --mapping-path catalog.v1.parquet

# YOLO26n
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window-yolo --yolo-extra-args " --image-size 640 --epochs 400 --batch-size 24 --num-workers 8" --use-tta False --yolo-model yolo26n --mapping-path catalog.v1.parquet

# YOLO26n with synthetic data
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window-yolo --yolo-extra-args "--image-size 640 --epochs 400 --batch-size 12 --num-workers 4 --extra-images-dir data/synthetic/images --extra-annotations-path data/synthetic/instances.json" --use-tta False --yolo-model yolo26n --mapping-path catalog.v1.parquet

# YOLO26s
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window-yolo --yolo-extra-args " --image-size 640 --epochs 400 --batch-size 12 --num-workers 8" --yolo-model yolo26s --mapping-path catalog.v1.parquet --yolo-inference "--conf 0.0 --max-det 500"

# YOLO26x
run_pipeline.py --window-yolo --yolo-extra-args " --image-size 640 --epochs 400 --batch-size 32 --num-workers 8" --yolo-model yolo26x --mapping-path catalog.v1.parquet --yolo-inference "--conf 0.0 --max-det 500"

# rtdetrv2-s
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window-yolo --yolo-extra-args " --image-size 512 --epochs 200 --batch-size 24 --num-workers 6" --use-tta False --yolo-model rtdetr-l --mapping-path catalog.v1.parquet

# Resultados

## YOLO26n

COCO Dataset

mAPval 50-95(e2e) = 0.401

val mAP = 0.38 (110 epochs)
leaderboard = 0.3857

diferencia entre mAP leaderboard y mAP coco = 0.96 = -4%

## YOLO26s

COCO Dataset

mAPval 50-95(e2e) = 0.478

val mAP = 0.426 (epochs 170)
leaderboard = 0.4290

diferencia entre mAP leaderboard y mAP coco = 0.89 = -11%

# Retinanet

### resultados experimento

/usr/local/bin/python -m src.retinanet_train --project-root . --backbone resnet50 --epochs 80 --batch-size 16 --num-workers 16 --tile-size 512 --tiles True --lr 5e-4 --backbone-lr 1e-4 --trainable-layers 4

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.retinanet_inference --checkpoint outputs/retinanet/best.pt --backbone resnet101

# Dino

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.dino_train --project-root . --model-name dino_r50 --epochs 50 --batch-size 2 --gradient-accumulation 4 --tile-size 512 --num-workers 8

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.dino_inference --checkpoint outputs/dino/best_hf

### resultados experimento

/usr/local/bin/python -m src.dino_train --project-root . --model-name dino_r101_detr --epochs 50 --batch-size 16 --gradient-accumulation 4 --tile-size 512 --num-workers 16

# Focus Det

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.focusdet_train --project-root . --epochs 80 --batch-size 1 --imgsz 640

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.focusdet_inference --checkpoint outputs/focusdet/best.pt --imgsz 640

### resultados experimento

/usr/local/bin/python -m src.focusdet_train --project-root . --epochs 80 --batch-size 8 --imgsz 640 --num-workers 16

# Cascade r cnn 

cascade_rcnn_dcnv2
cascade_rcnn_swin_l
cascade_rcnn_hrnet
cascade_rcnn_convnext_xl
cascade_rcnn_resnet101
cascade_rcnn_resnet50

# ResNet-50 — rápido, baseline limpio
/usr/bin/python -m src.train --arch cascade_rcnn_resnet50 \
  --epochs 20 --batch-size 24 --num-workers 8 \
  --image-size 640 --save-top-k 1 --lr 6e-4

# ResNet-101 — mejor que R50, buena relación coste/mAP
/usr/bin/python -m src.train --arch cascade_rcnn_resnet101 \
  --epochs 20 --batch-size 20 --num-workers 8 \
  --image-size 640 --save-top-k 1 --lr 5e-4

# DCNv2 — el más adecuado para rayas deformables, prueba éste primero
/usr/bin/python -m src.train --arch cascade_rcnn_dcnv2 \
  --epochs 20 --batch-size 8 --num-workers 8 \
  --image-size 1024 --save-top-k 1 --lr 2e-4

# HRNet — bueno para mAP_s por alta resolución espacial
/usr/bin/python -m src.train --arch cascade_rcnn_hrnet \
  --epochs 20 --batch-size 8 --num-workers 8 \
  --image-size 1024 --save-top-k 1 --lr 3.5e-4

# Swin-L — potente pero lento, solo si tienes tiempo
/usr/bin/python -m src.train --arch cascade_rcnn_swin_l \
  --epochs 20 --batch-size 6 --num-workers 8 \
  --image-size 640 --save-top-k 1 --lr 1.5e-4

# ConvNeXt-XL — tu baseline actual
/usr/bin/python -m src.train --arch cascade_rcnn_convnext_xl \
  --epochs 20 --batch-size 6 --num-workers 8 \
  --image-size 640 --save-top-k 1 --lr 1.5e-4

# ENSEMBLE

& C:\Users\esteb\.conda\envs\clearsar\python.exe e:/Sauvageduck24/ClearSar/submissions/ensemble_submissions.py --submissions e:/Sauvageduck24/ClearSar/submissions/submission_yolo_0_29.zip e:/Sauvageduck24/ClearSar/submissions/submission_yolo_0_3857.zip --output e:/Sauvageduck24/ClearSar/submissions/ensemble_submission.zip --method wbf --max-dets-per-image 300 --min-score 0.0 --weights 0.4 0.6

# GENERATE SYNTHETIC DATA
& C:\Users\esteb\.conda\envs\clearsar\python.exe e:/Sauvageduck24/ClearSar/synthetic_data/generate_synthetic_sar.py --num-images 1000 --mix-ratio 0.25

# Estadisticas del dataset
total=9288
area<=1: 0
area<=2: 0
area<=4: 
area<=8: 6
ratio>=5: 5673
ratio>=10: 4218
ratio>=20: 1376
min_area=3
area_p10=108
area_p50=1056
area_p90=5678