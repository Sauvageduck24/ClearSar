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

inference

C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.yolo_inference --project-root 'E:\Sauvageduck24\ClearSar' --checkpoint 'E:\Sauvageduck24\ClearSar\models\yolo_best_yolo26s.pt' --output 'E:\Sauvageduck24\ClearSar\outputs\submission_yolo.zip' --image-size 640 --no-tta --mapping-path catalog.v1.parquet --conf 0.0 --max-det 500

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

# **!** Para sacar mejores resultados hay que probar a entrenar con tta, y a poner un numero de epochs cercano para asi poder ajustar close mosaic y patience, por ejemplo, en YOLO26n, tendria que haber puesto 120 epochs y close_mosaic=20 y patience 20 o algo asi.

- esto no mejora todavia le pasa algo

& C:\Users\esteb\.conda\envs\clearsar\python.exe -m src.yolo_refinement --weights 'E:\Sauvageduck24\ClearSar\models\yolo_best_yolo26s.pt' --epochs 30 --lr 0.0001 --batch-size 12