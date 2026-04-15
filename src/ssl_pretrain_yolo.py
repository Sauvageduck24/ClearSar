from __future__ import annotations

"""
ssl_pretrain_sar.py — Pre-entrenamiento SSL (SimCLR) del backbone de YOLO sobre imágenes SAR.

Flujo completo:
  1. Carga el modelo YOLO indicado (yolov9s / yolo11s / yolo12s u otro .pt).
  2. Extrae su backbone (capas hasta el final del bloque backbone según el YAML).
  3. Entrena con SimCLR sobre TODAS las imágenes disponibles (train + test, sin etiquetas).
  4. Guarda un nuevo .pt con los pesos del backbone reemplazados, listo para usar en
     yolo_train_slicing.py con --model <output>.

Augmentations específicas para SAR:
  - Flip horizontal / vertical
  - Rotación leve (±10°)
  - Ruido speckle multiplicativo (log-normal)
  - Dropout de parches (simula pérdida de coherencia)
  - SIN transformaciones de color (HSV, brillo, contraste son irrelevantes en SAR)

Uso básico:
  python ssl_pretrain_sar.py \\
      --model yolo11s \\
      --images-dirs data/train/images data/test/images \\
      --output models/yolo11s_sar_ssl.pt \\
      --epochs 100 \\
      --batch-size 64 \\
      --image-size 640

Uso con .pt existente:
  python ssl_pretrain_sar.py \\
      --model path/to/yolov9s.pt \\
      --images-dirs data/images \\
      --output models/yolov9s_sar_ssl.pt
"""

import argparse
import copy
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constantes por arquitectura
# ---------------------------------------------------------------------------

# Para cada familia de modelo: número de capas del backbone en m.model.model
# y canales de salida del último bloque backbone.
# Verificado contra los YAML de ultralytics 8.4.x:
#   yolov9s  → backbone[0:10],  out_ch=256  (SPPELAN)
#   yolo11s  → backbone[0:11],  out_ch=1024 (C2PSA)
#   yolo12s  → backbone[0:9],   out_ch=1024 (A2C2f)

BACKBONE_CONFIG = {
    # clave: (num_backbone_layers, out_channels)
    "yolov9t":  (10, 256),
    "yolov9s":  (10, 256),
    "yolov9m":  (10, 512),
    "yolov9c":  (10, 512),
    "yolov9e":  (10, 512),
    "yolo11n":  (11, 1024),
    "yolo11s":  (11, 1024),
    "yolo11m":  (11, 1024),
    "yolo11l":  (11, 1024),
    "yolo11x":  (11, 1024),
    "yolo12n":  (9,  1024),
    "yolo12s":  (9,  1024),
    "yolo12m":  (9,  1024),
    "yolo12l":  (9,  1024),
    "yolo12x":  (9,  1024),
    # yolov8 por si acaso
    "yolov8n":  (10, 1024),
    "yolov8s":  (10, 1024),
    "yolov8m":  (10, 1024),
    "yolov8l":  (10, 1024),
    "yolov8x":  (10, 1024),
}


def _infer_backbone_config(model_arg: str) -> Tuple[Optional[int], Optional[int]]:
    """Busca en BACKBONE_CONFIG usando el nombre del modelo (sin extensión ni ruta)."""
    stem = Path(model_arg).stem.lower()
    # búsqueda exacta primero
    if stem in BACKBONE_CONFIG:
        return BACKBONE_CONFIG[stem]
    # búsqueda por prefijo (p.ej. "yolov9s_custom" → "yolov9s")
    for key, val in BACKBONE_CONFIG.items():
        if stem.startswith(key):
            return val
    return None, None


# ---------------------------------------------------------------------------
# Augmentations SAR
# ---------------------------------------------------------------------------

class SpeckleNoise:
    """
    Ruido speckle multiplicativo log-normal.
    SAR produce ruido coherente con distribución ~Rayleigh; aproximar con
    log-normal es estándar en la literatura.
    """
    def __init__(self, sigma: float = 0.15):
        self.sigma = sigma

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: [C, H, W] en [0, 1]
        noise = torch.exp(torch.randn_like(img) * self.sigma)
        return (img * noise).clamp(0.0, 1.0)


class PatchDropout:
    """
    Pone a cero parches aleatorios del tensor imagen.
    Simula pérdida de coherencia / sombras de apuntamiento en SAR.
    """
    def __init__(self, patch_size: int = 32, drop_prob: float = 0.1):
        self.patch_size = patch_size
        self.drop_prob = drop_prob

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        p = self.patch_size
        out = img.clone()
        for y in range(0, H, p):
            for x in range(0, W, p):
                if random.random() < self.drop_prob:
                    out[:, y:y+p, x:x+p] = 0.0
        return out


class SARSimCLRAugmentation:
    """
    Par de augmentations para SimCLR orientadas a imágenes SAR.
    Devuelve (view1, view2) a partir de un tensor [C,H,W].
    """
    def __init__(
        self,
        image_size: int = 640,
        speckle_sigma: float = 0.15,
        patch_drop_prob: float = 0.08,
        rotation_degrees: float = 10.0,
    ):
        self.image_size = image_size
        self.speckle = SpeckleNoise(sigma=speckle_sigma)
        self.patch_drop = PatchDropout(patch_size=max(16, image_size // 20), drop_prob=patch_drop_prob)
        self.rot_deg = rotation_degrees

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """Aplica un conjunto aleatorio de augmentations a un único tensor [C,H,W]."""
        import torchvision.transforms.functional as TF

        # 1. Random crop + resize (SimCLR estándar, bueno para localizar RFI parciales)
        _, H, W = img.shape
        scale = random.uniform(0.5, 1.0)
        ch = int(H * scale)
        cw = int(W * scale)
        top  = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        img = img[:, top:top+ch, left:left+cw]
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size),
                            mode="bilinear", align_corners=False).squeeze(0)

        # 2. Flips (muy comunes en SAR, no cambian la física)
        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])  # horizontal
        if random.random() > 0.5:
            img = torch.flip(img, dims=[1])  # vertical

        # 3. Rotación leve (SAR puede tener orientación de vuelo variable)
        if random.random() > 0.5:
            angle = random.uniform(-self.rot_deg, self.rot_deg)
            img = TF.rotate(img, angle)

        # 4. Ruido speckle multiplicativo (siempre presente en SAR)
        img = self.speckle(img)

        # 5. Patch dropout (dropout de coherencia)
        img = self.patch_drop(img)

        return img.clamp(0.0, 1.0)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._augment(img.clone()), self._augment(img.clone())


# ---------------------------------------------------------------------------
# Dataset sin etiquetas
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def _load_image_as_tensor(path: Path, image_size: int) -> Optional[torch.Tensor]:
    """
    Carga una imagen SAR como tensor [C, H, W] normalizado a [0, 1].
    Soporta:
      - PNG/JPG/TIF con 1, 2 o 3 canales
      - .npy con shape (H,W), (H,W,C) o (C,H,W)
    Si la imagen tiene más de 3 canales, usa los primeros 3.
    Si tiene 1 o 2 canales, replica hasta 3 (para compatibilidad con el backbone).
    """
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(str(path)).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]           # (1, H, W)
            elif arr.ndim == 3 and arr.shape[2] < arr.shape[0]:
                arr = arr.transpose(2, 0, 1)    # (H, W, C) → (C, H, W)
            t = torch.from_numpy(arr)
        else:
            pil = Image.open(str(path))
            if pil.mode == "I;16":
                arr = np.array(pil, dtype=np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                t = torch.from_numpy(arr).unsqueeze(0)
            else:
                pil = pil.convert("RGB")
                arr = np.array(pil, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)

        # Normalizar rango a [0, 1] si no lo está
        if t.max() > 1.01:
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)

        # Asegurar exactamente 3 canales
        C = t.shape[0]
        if C == 1:
            t = t.repeat(3, 1, 1)
        elif C == 2:
            t = torch.cat([t, t[:1]], dim=0)
        elif C > 3:
            t = t[:3]

        # Resize
        t = F.interpolate(t.unsqueeze(0), size=(image_size, image_size),
                          mode="bilinear", align_corners=False).squeeze(0)
        return t.clamp(0.0, 1.0)

    except Exception as e:
        print(f"  [ssl/warn] No se pudo cargar {path.name}: {e}")
        return None


class UnlabeledSARDataset(Dataset):
    """Dataset de imágenes SAR sin etiquetas para SSL."""

    def __init__(
        self,
        image_dirs: List[Path],
        augmentation: SARSimCLRAugmentation,
        image_size: int = 640,
    ):
        self.augmentation = augmentation
        self.image_size = image_size
        self.paths: List[Path] = []
        for d in image_dirs:
            for ext in IMAGE_EXTENSIONS:
                self.paths.extend(sorted(d.rglob(f"*{ext}")))
        if not self.paths:
            raise ValueError(
                f"No se encontraron imágenes en: {[str(d) for d in image_dirs]}\n"
                f"Extensiones buscadas: {IMAGE_EXTENSIONS}"
            )
        print(f"[ssl/data] {len(self.paths)} imágenes encontradas en {len(image_dirs)} directorios.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = _load_image_as_tensor(self.paths[idx], self.image_size)
        if tensor is None:
            # fallback: imagen en negro si hay error de carga
            tensor = torch.zeros(3, self.image_size, self.image_size)
        return self.augmentation(tensor)


# ---------------------------------------------------------------------------
# Modelo SimCLR
# ---------------------------------------------------------------------------

class SimCLR(nn.Module):
    """
    Wrapper SimCLR sobre un backbone YOLO.

    El backbone devuelve una lista de feature maps (salida típica de ultralytics).
    Tomamos el último feature map, hacemos GAP y proyectamos.
    """

    def __init__(self, backbone: nn.Module, backbone_out_ch: int, proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        hidden = max(proj_dim * 4, backbone_out_ch // 2)
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out_ch, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Pasa x por el backbone y devuelve el último feature map."""
        out = x
        last_tensor = None
        for layer in self.backbone:
            # Ultralytics usa f=-1 para secuencial; algunos layers esperan lista.
            # La mayoría en el backbone son secuenciales.
            try:
                out = layer(out)
            except Exception:
                # Capa con concat (poco probable en backbone puro, pero por seguridad)
                break
            if isinstance(out, torch.Tensor):
                last_tensor = out
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                last_tensor = out[-1]
        return last_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._forward_backbone(x)
        return self.projector(feat)


# ---------------------------------------------------------------------------
# NT-Xent Loss (SimCLR)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss de SimCLR.
    Espera dos tensores [N, D] (proyecciones de las dos vistas).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # [2N, D]
        z = torch.cat([z1, z2], dim=0)

        # Matriz de similitud coseno [2N, 2N]
        sim = torch.mm(z, z.T) / self.temperature

        # Máscara para ignorar la diagonal (similitud consigo mismo)
        mask = torch.eye(2 * N, device=z.device).bool()
        sim = sim.masked_fill(mask, float("-inf"))

        # Las parejas positivas están en las posiciones [i, i+N] y [i+N, i]
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# Extracción y restauración del backbone
# ---------------------------------------------------------------------------

def _extract_backbone(yolo_model, num_backbone_layers: int) -> nn.Sequential:
    """
    Devuelve las primeras `num_backbone_layers` capas del modelo YOLO
    como nn.Sequential (para poder hacer for layer in backbone).
    """
    layers = list(yolo_model.model.model)
    backbone_layers = layers[:num_backbone_layers]
    return nn.Sequential(*backbone_layers)


def _restore_backbone(yolo_model, trained_backbone: nn.Module, num_backbone_layers: int) -> None:
    """Reemplaza en el modelo YOLO las capas backbone con las entrenadas en SSL."""
    layers = list(yolo_model.model.model)
    trained_layers = list(trained_backbone.children())  # nn.Sequential
    for i, trained_layer in enumerate(trained_layers):
        layers[i] = trained_layer
    # Reasignar el ModuleList/Sequential interno
    yolo_model.model.model = nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Entrenamiento SSL
# ---------------------------------------------------------------------------

def _build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_ssl_pretraining(
    yolo_model,
    image_dirs: List[Path],
    output_path: Path,
    num_backbone_layers: int,
    backbone_out_ch: int,
    image_size: int = 640,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    proj_dim: int = 128,
    num_workers: int = 4,
    warmup_epochs: int = 10,
    speckle_sigma: float = 0.15,
    patch_drop_prob: float = 0.08,
    seed: int = 42,
    device: Optional[str] = None,
) -> None:

    # ---- Dispositivo ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"[ssl] Usando dispositivo: {dev}")

    # ---- Reproducibilidad ----
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in str(dev):
        torch.cuda.manual_seed_all(seed)

    # ---- Augmentations y Dataset ----
    aug = SARSimCLRAugmentation(
        image_size=image_size,
        speckle_sigma=speckle_sigma,
        patch_drop_prob=patch_drop_prob,
    )
    dataset = UnlabeledSARDataset(image_dirs, augmentation=aug, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=("cuda" in str(dev)),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    print(f"[ssl] Batches por época: {len(loader)}  |  Batch size: {batch_size}")

    # ---- Modelo SSL ----
    backbone = _extract_backbone(yolo_model, num_backbone_layers)
    model = SimCLR(backbone=backbone, backbone_out_ch=backbone_out_ch, proj_dim=proj_dim)
    model = model.to(dev)
    print(f"[ssl] Backbone: {num_backbone_layers} capas  |  out_ch={backbone_out_ch}  |  proj_dim={proj_dim}")

    # ---- Optimizador y scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _build_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)
    criterion = NTXentLoss(temperature=temperature).to(dev)
    use_cuda_amp = ("cuda" in str(dev))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    # ---- Loop de entrenamiento ----
    print(f"\n[ssl] Iniciando SSL — {epochs} épocas\n")
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Época {epoch:>4}/{epochs}", leave=False, unit="batch")

        for batch in pbar:
            view1, view2 = batch                    # cada uno [B, C, H, W]
            view1 = view1.to(dev, non_blocking=True)
            view2 = view2.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                z1 = model(view1)
                z2 = model(view2)
                loss = criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(loader))

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.backbone.state_dict())

        print(
            f"[ssl] Época {epoch:>4}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"best={best_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    # ---- Restaurar backbone y guardar .pt ----
    print("\n[ssl] Restaurando backbone con pesos SSL en el modelo YOLO...")
    if best_state is not None:
        model.backbone.load_state_dict(best_state)

    _restore_backbone(yolo_model, model.backbone, num_backbone_layers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Guardar exactamente como lo haría ultralytics (model.model + metadatos)
    ckpt = {
        "model": yolo_model.model,
        "train_args": {},
        "date": __import__("datetime").datetime.now().isoformat(),
        "ssl_pretrained": True,
        "ssl_epochs": epochs,
        "ssl_best_loss": float(best_loss),
    }
    torch.save(ckpt, str(output_path))
    print(f"[ssl] ✅  Modelo guardado en: {output_path}")
    print(f"[ssl]    Úsalo con: --model {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-entrenamiento SSL (SimCLR) del backbone YOLO sobre imágenes SAR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Modelo ---
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s",
        help=(
            "Modelo base. Puede ser:\n"
            "  - Nombre sin extensión: yolov9s, yolo11s, yolo12s, yolov8m, ...\n"
            "  - Ruta a un .pt existente: path/to/yolov9s.pt\n"
            "Si se pasa un nombre, ultralytics descarga los pesos oficiales."
        ),
    )
    parser.add_argument(
        "--backbone-layers",
        type=int,
        default=None,
        help=(
            "Número de capas del backbone a extraer. Si no se indica, se infiere "
            "automáticamente desde el nombre del modelo. Úsalo si tienes un .pt "
            "customizado con arquitectura modificada."
        ),
    )
    parser.add_argument(
        "--backbone-out-ch",
        type=int,
        default=None,
        help=(
            "Canales de salida del backbone. Si no se indica, se infiere automáticamente. "
            "Necesario si usas un modelo no incluido en la tabla interna."
        ),
    )

    # --- Datos ---
    parser.add_argument(
        "--images-dirs",
        nargs="+",
        required=True,
        metavar="DIR",
        help=(
            "Uno o más directorios con imágenes SAR (sin etiquetas). "
            "Puedes pasar train y test juntos, es SSL. "
            "Formatos: PNG, JPG, TIF, TIFF, BMP, NPY."
        ),
    )

    # --- Output ---
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Ruta del .pt de salida. Por defecto: <model>_sar_ssl.pt "
            "en el mismo directorio donde se ejecuta el script."
        ),
    )

    # --- Entrenamiento ---
    parser.add_argument("--epochs",      type=int,   default=100,  help="Épocas de SSL.")
    parser.add_argument("--batch-size",  type=int,   default=32,   help="Batch size.")
    parser.add_argument("--image-size",  type=int,   default=640,  help="Tamaño al que redimensionar imágenes.")
    parser.add_argument("--lr",          type=float, default=3e-4, help="Learning rate inicial.")
    parser.add_argument("--weight-decay",type=float, default=1e-4, help="Weight decay AdamW.")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperatura NT-Xent.")
    parser.add_argument("--proj-dim",    type=int,   default=128,  help="Dimensión del proyector SimCLR.")
    parser.add_argument("--warmup-epochs",type=int,  default=10,   help="Épocas de warmup cosine.")
    parser.add_argument("--num-workers", type=int,   default=4,    help="Workers DataLoader.")
    parser.add_argument("--seed",        type=int,   default=42,   help="Semilla de reproducibilidad.")
    parser.add_argument("--device",      type=str,   default=None,
                        help="Dispositivo: cuda, cuda:0, cpu. Por defecto: cuda si disponible.")

    # --- Augmentations SAR ---
    parser.add_argument("--speckle-sigma",    type=float, default=0.15,
                        help="Sigma del ruido speckle log-normal. 0 para deshabilitar.")
    parser.add_argument("--patch-drop-prob",  type=float, default=0.08,
                        help="Probabilidad de dropout por parche. 0 para deshabilitar.")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- Importar YOLO (tarde para no ralentizar --help) ----
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics no está instalado.\n"
            "Instálalo con:  pip install ultralytics"
        )

    # ---- Cargar modelo YOLO ----
    model_arg = args.model
    model_path = Path(model_arg)
    if model_path.suffix in (".pt", ".pth") and model_path.exists():
        print(f"[ssl] Cargando pesos desde: {model_path}")
        yolo_model = YOLO(str(model_path))
    else:
        # Nombre sin extensión: ultralytics lo descarga si es necesario
        pt_name = model_arg if model_arg.endswith(".pt") else model_arg + ".pt"
        print(f"[ssl] Cargando modelo: {pt_name}")
        yolo_model = YOLO(pt_name)

    # ---- Inferir config del backbone ----
    num_layers, out_ch = _infer_backbone_config(model_arg)

    if args.backbone_layers is not None:
        num_layers = args.backbone_layers
    if args.backbone_out_ch is not None:
        out_ch = args.backbone_out_ch

    if num_layers is None or out_ch is None:
        # Intentar inferir automáticamente del YAML del modelo
        try:
            yaml_cfg = yolo_model.model.yaml
            n = len(yaml_cfg.get("backbone", []))
            layers_list = list(yolo_model.model.model)
            if n > 0:
                num_layers = n
                # intentar sacar out_ch del último bloque backbone
                last_bb = layers_list[n - 1]
                for attr in ("cv2", "conv", "m"):
                    sub = getattr(last_bb, attr, None)
                    if sub is not None:
                        conv_sub = getattr(sub, "conv", sub)
                        ch = getattr(conv_sub, "out_channels", None)
                        if ch:
                            out_ch = ch
                            break
        except Exception:
            pass

    if num_layers is None or out_ch is None:
        raise SystemExit(
            f"No se pudo inferir la configuración del backbone para '{model_arg}'.\n"
            f"Especifica manualmente con --backbone-layers y --backbone-out-ch.\n"
            f"Modelos soportados automáticamente: {sorted(BACKBONE_CONFIG.keys())}"
        )

    print(f"[ssl] Backbone: {num_layers} capas, out_ch={out_ch}")

    # ---- Directorios de imágenes ----
    image_dirs = [Path(d) for d in args.images_dirs]
    for d in image_dirs:
        if not d.exists():
            raise SystemExit(f"Directorio no encontrado: {d}")

    # ---- Output path ----
    if args.output is not None:
        output_path = Path(args.output)
    else:
        stem = Path(model_arg).stem
        output_path = Path(f"{stem}_sar_ssl.pt")

    # ---- Lanzar SSL ----
    run_ssl_pretraining(
        yolo_model=yolo_model,
        image_dirs=image_dirs,
        output_path=output_path,
        num_backbone_layers=num_layers,
        backbone_out_ch=out_ch,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        speckle_sigma=args.speckle_sigma,
        patch_drop_prob=args.patch_drop_prob,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()