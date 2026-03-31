# Leaderboard de modelos que poder usar
https://leaderboard.roboflow.com/

# Primero entrena Faster R-CNN

## big model

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window1 --train-extra-args "--image-size 512 --epochs 30 --batch-size 1 --num-workers 6 --grad-accum-steps 1" --mapping-path catalog.v1.parquet

## small model
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window1 --train-extra-args "--arch fasterrcnn_mobilenet_v3_large_fpn --epochs 50 --batch-size 4 --num-workers 4 --grad-accum-steps 4" --mapping-path catalog.v1.parquet

# YOLO26n (LEADERBOARD MAP = 0.3857)
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window-yolo --yolo-extra-args " --image-size 640 --epochs 400 --batch-size 24 --num-workers 8" --use-tta False --yolo-model yolo26n --mapping-path catalog.v1.parquet

# YOLO26s
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window-yolo --yolo-extra-args " --image-size 640 --epochs 400 --batch-size 12 --num-workers 8" --use-tta False --yolo-model yolo26s --mapping-path catalog.v1.parquet

# rtdetrv2-s
& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --window-yolo --yolo-extra-args " --image-size 512 --epochs 200 --batch-size 24 --num-workers 6" --use-tta False --yolo-model rtdetr-l --mapping-path catalog.v1.parquet


# Resultados

## YOLO26n

COCO Dataset

mAPval 50-95(e2e) = 40.1

val mAP = 0.38 (110 epochs)
leaderboard = 0.3857

## YOLO26s

COCO Dataset

mAPval 50-95(e2e) = 47.8

val mAP = 0. (epochs)
leaderboard = 0.