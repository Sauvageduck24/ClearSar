rtx 4090

runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

which python

--------------------------------------------------------------------

# 1. Purgar NumPy y PyTorch por completo
/usr/bin/python -m pip uninstall -y numpy torch torchvision
/usr/bin/python -m pip uninstall -y numpy  # Lo lanzamos dos veces para limpiar instalaciones fantasma (ej. en ~/.local)

# 2. Instalar una versión de NumPy a prueba de balas para PyTorch 2.1
/usr/bin/python -m pip install "numpy==1.26.3"

/usr/bin/python -m pip install pycocotools tqdm

# 3. Instalar PyTorch DESPUÉS de NumPy para que detecte la librería correctamente
/usr/bin/python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 1. Instalar la herramienta de gestión de OpenMMLab (MIM)
/usr/bin/python -m pip install -U openmim

# 2. Instalar los componentes base usando MIM 
# MIM se encarga de buscar la versión que coincida con tu PyTorch 2.1.0 y CUDA 11.8
/usr/bin/python -m mim install mmengine
/usr/bin/python -m mim install "mmcv>=2.1.0"
/usr/bin/python -m mim install mmdet

/usr/bin/python -m pip install mmpretrain

# Forzamos la reinstalación de la última versión estable de la rama 1.x
/usr/bin/python -m pip install "numpy<2.0.0" "setuptools<70.0.0" --force-reinstall

/usr/bin/python -c "import torch; import numpy; print(f'Torch OK. NumPy version: {numpy.__version__}')"