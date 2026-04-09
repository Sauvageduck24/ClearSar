# Yolo11s con sahi y tta

& C:\Users\esteb\.conda\envs\clearsar\python.exe run_pipeline.py --yolo-model yolo11s --mapping-path catalog.v1.parquet --yolo-extra-args "--epochs 10 --batch-size 32 --num-workers 12 --lr 0.0005" --yolo-inference "--conf 0.001 --max-det 500 --tta --merge-boxes" --cache disk --slice-size 256 --overlap-ratio 0.25