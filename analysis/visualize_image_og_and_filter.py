import argparse
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    if max_v - min_v < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - min_v) / (max_v - min_v)


def local_std(channel: np.ndarray, k: int = 7) -> np.ndarray:
    mean = cv2.GaussianBlur(channel, (k, k), 0)
    mean_sq = cv2.GaussianBlur(channel ** 2, (k, k), 0)
    return np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))


def parse_yolo_boxes(label_path: Path, h: int, w: int):
    boxes = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            cls, x, y, bw, bh = map(float, line.split())
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

    return boxes


def load_image(image_path: Path) -> np.ndarray:
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        img = np.load(str(image_path))
    else:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"No se pudo leer la imagen: {image_path}")
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.ndim == 2:
        img = img[:, :, None]

    return img.astype(np.float32)


def choose_random_image_with_boxes(img_dir: Path, label_dir: Path, seed: int | None):
    img_paths = [Path(p) for p in glob(str(img_dir / "*.*"))]
    if not img_paths:
        raise RuntimeError(f"No hay imagenes en {img_dir}")

    img_by_stem = {p.stem: p for p in img_paths}
    label_paths = [Path(p) for p in glob(str(label_dir / "*.txt"))]

    candidate_stems = []
    for lp in label_paths:
        if lp.stem not in img_by_stem:
            continue
        if lp.stat().st_size == 0:
            continue
        candidate_stems.append(lp.stem)

    if not candidate_stems:
        raise RuntimeError("No se encontraron pares imagen/label con boxes.")

    rng = np.random.default_rng(seed)
    selected_stem = candidate_stems[int(rng.integers(0, len(candidate_stems)))]
    return img_by_stem[selected_stem], label_dir / f"{selected_stem}.txt"


def build_visualizations(img: np.ndarray):
    if img.shape[2] >= 2:
        vv_raw = img[:, :, 0]
        vh_raw = img[:, :, 1]
    else:
        vv_raw = img[:, :, 0]
        vh_raw = img[:, :, 0]

    vv_raw_norm = normalize01(vv_raw)
    vh_raw_norm = normalize01(vh_raw)
    original_rgb = np.dstack([vv_raw_norm, vh_raw_norm, (vv_raw_norm + vh_raw_norm) / 2.0])

    vv_log = np.log1p(np.clip(vv_raw, a_min=0, a_max=None))
    vh_log = np.log1p(np.clip(vh_raw, a_min=0, a_max=None))

    vv_log_norm = normalize01(vv_log)
    vh_log_norm = normalize01(vh_log)

    std7 = local_std(vv_log, 7)
    std15 = local_std(vv_log, 15)
    std31 = local_std(vv_log, 31)
    std_multi = (std7 + std15 + std31) / 3.0
    #std_multi = 0.5 * std7 + 0.3 * std15 + 0.2 * std31
    std_multi = std_multi*2
    std_multi_norm = normalize01(std_multi)
    filtered_rgb = np.dstack([vv_log_norm, vh_log_norm, std_multi_norm])
    return original_rgb, filtered_rgb


def draw_boxes(ax, boxes):
    for (x1, y1, x2, y2) in boxes:
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)


def main():
    parser = argparse.ArgumentParser(
        description="Visualiza una imagen OG y su version filtrada (VV, VH, STD_MULTI)."
    )
    parser.add_argument("--img-dir", default="data/yolo/images/train", help="Carpeta de imagenes")
    parser.add_argument("--label-dir", default="data/yolo/labels/train", help="Carpeta de labels YOLO")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para seleccionar imagen aleatoria")
    parser.add_argument(
        "--save-path",
        default="analysis/outputs/visualize_image_og_and_filter.png",
        help="Ruta para guardar la figura",
    )
    parser.add_argument("--no-show", action="store_true", help="No mostrar ventana interactiva")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)

    img_path, label_path = choose_random_image_with_boxes(img_dir, label_dir, args.seed)
    img = load_image(img_path)

    h, w = img.shape[:2]
    boxes = parse_yolo_boxes(label_path, h, w)

    original_rgb, filtered_rgb = build_visualizations(img)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(original_rgb)
    draw_boxes(axes[0], boxes)
    axes[0].set_title(f"Original ({img_path.name})")
    axes[0].axis("off")

    axes[1].imshow(filtered_rgb)
    draw_boxes(axes[1], boxes)
    axes[1].set_title("Filtrada: R=VV(log+norm), G=VH(log+norm), B=STD_MULTI")
    axes[1].axis("off")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)

    print(f"Imagen seleccionada: {img_path}")
    print(f"Label usada: {label_path}")
    print(f"Boxes dibujadas: {len(boxes)}")
    print(f"Figura guardada en: {save_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
