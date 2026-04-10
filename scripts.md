# Yolo11s con sahi (entrenamiento e inference)

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --yolo-model yolo11s --mapping-path catalog.v1.parquet --yolo-extra-args "--epochs 10 --batch-size 20 --num-workers 12 --lr 0.0005" --yolo-inference "--conf 0.05 --iou 0.3 --max-det 300" --cache disk --slice-size 512 --overlap-ratio 0.2

# Yolo11s con sahi (solo entrenamiento) e inference con imagen original

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --yolo-model yolo11s --mapping-path catalog.v1.parquet --yolo-extra-args "--epochs 30 --batch-size 20 --num-workers 12 --lr 0.0005" --yolo-inference "--inference-mode original --image-size 512 --conf 0.05 --iou 0.3 --max-det 300" --cache disk --slice-size 512 --overlap-ratio 0.2

# Yolo11s sin sahi

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --yolo-model yolo11s --skip-slicing --mapping-path catalog.v1.parquet  --yolo-extra-args "--dataset-source normal --epochs 150 --batch-size 20 --num-workers 12 --image-size 512 --lr 0.0005" --yolo-inference "--conf 0.05 --iou 0.3 --max-det 300"