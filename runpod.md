rtx 4090

runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

which python

--------------------------------------------------------------------

# 1. Purgar NumPy y PyTorch por completo
/usr/bin/python -m pip uninstall -y numpy torch torchvision
/usr/bin/python -m pip uninstall -y numpy  # Lo lanzamos dos veces para limpiar instalaciones fantasma (ej. en ~/.local)

# 2. Instalar una versión de NumPy a prueba de balas para PyTorch 2.1
/usr/bin/python -m pip install "numpy==1.26.3"

# 3. Instalar PyTorch DESPUÉS de NumPy para que detecte la librería correctamente
/usr/bin/python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

-----------------------------------------------------------------------

/usr/bin/python -m src.train --arch cascade_rcnn_convnext_xl --epochs 50 --batch-size 4 --num-workers 6 --image-size 512
