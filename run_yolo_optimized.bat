@echo off
REM YOLO optimizado para SAR - Objetivo: 0.45-0.50+ mAP
REM Basado en mejoras específicas para objetos pequeños/finos en imágenes SAR

setlocal enabledelayedexpansion

set PYTHON=C:\Users\esteb\.conda\envs\clearsar\python.exe

REM Versión mejorada: mejor learning rate, sin augmentaciones brutales
%PYTHON% -m src.train ^
  --model yolo11s ^
  --epochs 120 ^
  --batch-size 8 ^
  --image-size 960 ^
  --num-workers 8 ^
  --lr 0.002 ^
  --lrf 0.01 ^
  --apply-letterboxing false ^
  --y-component-multiplier 2.0 ^
  --kfold 1 ^
  --warmup-epochs 10 ^
  --scale 0.02 ^
  --translate 0.01 ^
  --mosaic 1.0 ^
  --close-mosaic 20 ^
  --box 7.5 ^
  --cls 0.5 ^
  --dfl 1.5 ^
  --label-smoothing 0.0 ^
  --small-box-copy-paste true ^
  --copy-paste-p 0.3 ^
  --copy-paste-max-h 12.0 ^
  --copy-paste-n 3 ^
  --hard-negative-mining false

pause
