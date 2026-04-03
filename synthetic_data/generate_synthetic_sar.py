#!/usr/bin/env python3
"""
generate_synthetic_sar.py

Script para generar datos sintéticos de SAR con RFI para ClearSAR competition.

Uso:
    python generate_synthetic_sar.py --output ./data/synthetic --num-images 1000

O en Jupyter:
    from generate_synthetic_sar import *
    generate_and_mix_dataset()
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import random

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class SARSyntheticGenerator:
    """Generador de imágenes SAR sintéticas con RFI."""
    
    def __init__(self, height: int = 512, width: int = 512):
        self.height = height
        self.width = width
    
    def generate_background(self, style: str = "ocean") -> np.ndarray:
        """
        Genera background SAR realista.
        
        Estilos:
        - ocean: suave, poco texturado (agua calma)
        - land: texturado (bosque, terreno)
        - mixed: océano arriba, tierra abajo
        """
        if style == "ocean":
            # Océano: coherencia alta, speckle suave
            bg = np.random.exponential(scale=0.4, size=(self.height, self.width))
            bg = cv2.GaussianBlur(bg.astype(np.float32), (7, 7), 2.0)
        
        elif style == "land":
            # Tierra: coherencia baja, más ruido
            bg = np.random.exponential(scale=0.35, size=(self.height, self.width))
            
            # Agregar textura fractal (simula árboles/topografía)
            for octave in range(1, 5):
                scale = 2 ** octave
                noise = np.random.random((self.height // scale, self.width // scale))
                noise = cv2.resize(noise, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                bg += 0.15 * noise / octave
        
        else:  # mixed
            bg = np.zeros((self.height, self.width))
            ocean = np.random.exponential(scale=0.4, size=(self.height // 2, self.width))
            land = np.random.exponential(scale=0.35, size=(self.height // 2, self.width))

            half = self.height // 2
            bg[:half] = ocean
            bg[half:] = land

            # Transición suave centrada en la frontera océano/tierra
            transition = np.linspace(0, 1, self.height // 4)
            start = max(0, half - len(transition) // 2)
            end = min(self.height, start + len(transition))
            for i, idx in enumerate(range(start, end)):
                t = transition[i]
                ocean_row = min(max(idx, 0), ocean.shape[0] - 1)
                land_row = min(max(idx - half, 0), land.shape[0] - 1)
                bg[idx] = ocean[ocean_row] * (1 - t) + land[land_row] * t
        
        # Normalizar
        bg = np.clip((bg - bg.min()) / (bg.max() - bg.min() + 1e-6), 0, 1)
        return bg
    
    def generate_rfi(self, bg_image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Añade RFIs (rayas de interferencia) a la imagen.
        
        Características realistas del RFI:
        - Altura: 1-15 píxeles (típico 3-10)
        - Ancho: 20-90% del ancho imagen
        - Posición: aleatoria horizontal
        - Intensidad: 0.4-1.0 brightness
        - Pattern: rayas pueden tener modulación (no uniforme)
        """
        
        image = bg_image.copy()
        annotations = []
        
        # Número aleatorio de RFIs por imagen
        num_rfi = np.random.choice(
            [0, 0, 1, 1, 1, 2, 2, 3],
            p=[0.1, 0.1, 0.3, 0.2, 0.15, 0.1, 0.04, 0.01],
        )
        
        for _ in range(num_rfi):
            # Dimensiones del RFI
            height = np.random.randint(1, 16)  # 1-15px
            width_frac = np.random.uniform(0.15, 0.95)
            width = int(self.width * width_frac)
            
            # Posición
            x_start = np.random.randint(0, max(1, self.width - width))
            y_start = np.random.randint(0, max(1, self.height - height))
            
            # Intensidad y tipo de RFI
            intensity = np.random.uniform(0.4, 1.0)
            
            # Crear patrón de RFI
            rfi = np.ones((height, width)) * intensity
            
            # Variación: algunos RFIs tienen modulación
            if np.random.random() < 0.4:
                # Modulación en amplitud (simula variación en la interferencia)
                modulation = np.linspace(0.6, 1.0, width)
                rfi = rfi * modulation

            # Pequeño desenfoque horizontal para suavizar la raya
            rfi = cv2.GaussianBlur(rfi, (3, 1), 0)
            
            # Suavizar bordes verticales (gaussian falloff)
            for y in range(height):
                edge_fade = 1.0 - np.abs(y - height/2) / (height/2 + 1)
                rfi[y, :] *= edge_fade ** 0.5
            
            # Mezclar RFI con background
            x_end = x_start + width
            y_end = y_start + height
            
            # Asegurar que no se sale de los bordes
            x_end = min(x_end, self.width)
            y_end = min(y_end, self.height)
            actual_width = x_end - x_start
            actual_height = y_end - y_start
            
            if actual_width > 0 and actual_height > 0:
                rfi_clipped = rfi[:actual_height, :actual_width]
                
                # Blending: no sobreescribir, mezclar
                blend_factor = np.random.uniform(0.1, 0.4) # Más sutil
                image[y_start:y_end, x_start:x_end] = np.clip(
                    image[y_start:y_end, x_start:x_end] + rfi_clipped * blend_factor, 0, 255
                )
                
                # Guardar anotación
                annotations.append({
                    "bbox": [x_start, y_start, actual_width, actual_height],
                    "area": actual_width * actual_height,
                })
        
        return image, annotations
    
    def generate_image(
        self,
        noise_level: float = 0.15,
        background_style: str = "mixed",
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Genera una imagen SAR completa con RFI."""
        
        # Background
        bg = self.generate_background(background_style)
        
        # Convertir a 3 canales (VV, VH, ratio)
        image = np.stack([
            bg,
            bg * np.random.uniform(0.5, 0.9),  # VH típicamente más débil
            bg * np.random.uniform(0.4, 0.8),  # Ratio variable
        ], axis=2)
        
        # Speckle noise (característica SAR)
        speckle = np.random.exponential(scale=noise_level, size=image.shape)
        image = image * (1 + speckle)
        image = np.clip(image, 0, 1)
        
        # Agregar RFIs
        image_with_rfi = image.copy()
        for c in range(3):
            image_with_rfi[:, :, c], annotations = self.generate_rfi(image[:, :, c])
        
        return (image_with_rfi * 255).astype(np.uint8), annotations


def generate_synthetic_dataset(
    output_dir: Path,
    num_images: int = 500,
    image_size: int = 512,
) -> Tuple[Path, Path]:
    """
    Genera dataset sintético completo.
    
    Retorna: (images_dir, annotations_path)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    generator = SARSyntheticGenerator(height=image_size, width=image_size)
    
    all_images = []
    all_annotations = []
    
    print(f"\n{'='*60}")
    print(f"Generating {num_images} synthetic SAR images")
    print(f"{'='*60}")
    
    backgrounds = ["ocean", "land", "mixed"]
    
    iterator = range(num_images)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Generating", unit="img")

    for img_id in iterator:
        # Variar parámetros para diversidad
        bg_style = random.choice(backgrounds)
        noise = np.random.uniform(0.08, 0.25)
        
        # Generar imagen
        image, annots = generator.generate_image(
            noise_level=noise,
            background_style=bg_style,
        )
        
        # Guardar imagen
        img_filename = f"synthetic_{img_id}.png"
        img_path = images_dir / img_filename
        
        # OpenCV espera BGR
        cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Metadatos
        all_images.append({
            "id": img_id,
            "file_name": img_filename,
            "width": image_size,
            "height": image_size,
        })
        
        # Anotaciones
        for ann in annots:
            ann["image_id"] = img_id
            ann["id"] = len(all_annotations)
            ann["category_id"] = 1
            ann["iscrowd"] = 0
            all_annotations.append(ann)
        
        # Progress
        if tqdm is None and (img_id + 1) % 100 == 0:
            print(f"  ✓ Generated {img_id + 1}/{num_images} images ({len(all_annotations)} RFIs)")
    
    # Guardar anotaciones en formato COCO
    coco_data = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": [{"id": 1, "name": "RFI"}],
    }
    
    anno_path = output_dir / "instances.json"
    with open(anno_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✓ Generated {len(all_images)} images")
    print(f"✓ Generated {len(all_annotations)} RFI annotations")
    print(f"✓ Saved to {output_dir}")
    
    return images_dir, anno_path


def mix_with_real_data(
    real_anno_path: Path,
    synthetic_anno_path: Path,
    real_images_dir: Path,
    synthetic_images_dir: Path,
    output_anno_path: Path,
    synthetic_ratio: float = 0.25,
) -> Path:
    """
    Combina datos sintéticos con reales.
    
    synthetic_ratio: qué fracción del dataset debe ser sintético
    - 0.2-0.3: recomendado (+0.03-0.05 mAP)
    - 0.5: riesgo de overfitting a sintéticos
    - >0.6: generalmente peor
    """
    
    print(f"\n{'='*60}")
    print(f"Mixing synthetic ({synthetic_ratio*100:.0f}%) with real data")
    print(f"{'='*60}")
    
    # Cargar datos
    with open(real_anno_path, 'r') as f:
        real_coco = json.load(f)
    
    with open(synthetic_anno_path, 'r') as f:
        synthetic_coco = json.load(f)
    
    real_count = len(real_coco['images'])
    synthetic_to_add = int(real_count * synthetic_ratio / (1 - synthetic_ratio))
    synthetic_to_add = min(synthetic_to_add, len(synthetic_coco['images']))
    
    print(f"  Real images: {real_count}")
    print(f"  Adding synthetic: {synthetic_to_add}")
    
    # Crear dataset combinado
    combined_images = real_coco['images'].copy()
    combined_annotations = real_coco['annotations'].copy()
    
    # Muestrear sintéticas
    synthetic_indices = random.sample(
        range(len(synthetic_coco['images'])),
        synthetic_to_add
    )
    
    # IDs
    next_image_id = real_count
    next_ann_id = len(combined_annotations)
    id_map = {}
    
    # Agregar imágenes sintéticas
    for old_idx in synthetic_indices:
        old_image = synthetic_coco['images'][old_idx]
        new_image = old_image.copy()
        new_image['id'] = next_image_id
        combined_images.append(new_image)
        id_map[old_image['id']] = next_image_id
        next_image_id += 1
    
    # Agregar anotaciones
    for old_ann in synthetic_coco['annotations']:
        if old_ann['image_id'] in id_map:
            new_ann = old_ann.copy()
            new_ann['image_id'] = id_map[old_ann['image_id']]
            new_ann['id'] = next_ann_id
            combined_annotations.append(new_ann)
            next_ann_id += 1
    
    # Guardar
    combined_coco = {
        'images': combined_images,
        'annotations': combined_annotations,
        'categories': [{'id': 1, 'name': 'RFI'}],
    }
    
    output_anno_path = Path(output_anno_path)
    output_anno_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_anno_path, 'w') as f:
        json.dump(combined_coco, f, indent=2)
    
    print(f"\n✓ Combined dataset:")
    print(f"  Total images: {len(combined_images)}")
    print(f"  Total annotations: {len(combined_annotations)}")
    print(f"  Saved to {output_anno_path}")
    
    return output_anno_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SAR/RFI data for ClearSAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

1. Generar 500 imágenes sintéticas:
   python generate_synthetic_sar.py --output ./data/synthetic --num-images 500

2. Generar y mezclar con datos reales:
   python generate_synthetic_sar.py \\
     --output ./data/synthetic \\
     --num-images 1000 \\
     --real-annotations ./data/annotations/instances_train.json \\
     --real-images ./data/train/images \\
     --mix-ratio 0.25
        """,
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./data/synthetic",
        help="Output directory for synthetic data",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=500,
        help="Number of synthetic images to generate (default: 500)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size (square, default: 512)",
    )
    parser.add_argument(
        "--real-annotations",
        type=str,
        default=None,
        help="Path to real data annotations (COCO format) for mixing",
    )
    parser.add_argument(
        "--real-images",
        type=str,
        default=None,
        help="Path to real data images directory",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.25,
        help="Ratio of synthetic data to include (0.25 = 25%% synthetic)",
    )
    
    args = parser.parse_args()
    
    # Generar datos sintéticos
    synthetic_images_dir, synthetic_anno_path = generate_synthetic_dataset(
        output_dir=args.output,
        num_images=args.num_images,
        image_size=args.image_size,
    )
    
    # Mezclar con datos reales si se proporcionan
    if args.real_annotations and args.real_images:
        output_anno = Path(args.output) / "instances_mixed.json"
        mix_with_real_data(
            real_anno_path=Path(args.real_annotations),
            synthetic_anno_path=synthetic_anno_path,
            real_images_dir=Path(args.real_images),
            synthetic_images_dir=synthetic_images_dir,
            output_anno_path=output_anno,
            synthetic_ratio=args.mix_ratio,
        )
        
        print(f"\n{'='*60}")
        print("Next steps:")
        print(f"{'='*60}")
        print(f"\nPara YOLO:")
        print(f"  python -m src.yolo_train \\")
        print(f"    --extra-annotations-path {output_anno} \\")
        print(f"    --extra-images-dir {synthetic_images_dir}")
        print(f"\nPara Faster R-CNN:")
        print(f"  Actualiza el config para usar: {output_anno}")
    else:
        print(f"\n{'='*60}")
        print("Datos sintéticos generados correctamente.")
        print(f"Para mezclar con datos reales, usa --real-annotations y --real-images")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

