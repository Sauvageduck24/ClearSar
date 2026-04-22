"""
Browser de imágenes con anotaciones GT.

Modo normal   : python -m analysis.image_browser_simple
Modo stacked  : python -m analysis.image_browser_simple --combo run1

En modo stacked muestra 5 paneles por imagen:
  [Original + boxes]  [Compuesta (stacked) + boxes]
  [Canal 1]           [Canal 2]           [Canal 3]
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from src.preprocess import COMBOS, COMBO_LABELS

BASE_DIR = Path(__file__).resolve().parents[1]
ANN_PATH = BASE_DIR / "data" / "annotations" / "instances_train.json"
IMG_DIR  = BASE_DIR / "data" / "images" / "train"

CHANNEL_NAMES = {
    "gray":             "Gray",
    "clahe":            "CLAHE",
    "log":              "Log",
    "gabor":            "Gabor",
    "bilateral_tophat": "Bilateral+TopHat",
}


def load_images_and_annotations(img_dir: Path):
    with ANN_PATH.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    files = []
    for img in coco["images"]:
        img_path = img_dir / img["file_name"]
        if img_path.exists():
            files.append((img_path, img["id"]))

    return files, anns_by_image


def _draw_boxes(ax, img_id, anns_by_image):
    for ann in anns_by_image.get(img_id, []):
        x, y, w, h = ann["bbox"]
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor="lime", facecolor="none"))


# ---------------------------------------------------------------------------
# Normal browser (original images)
# ---------------------------------------------------------------------------

class ImageBrowser:
    def __init__(self, files, anns_by_image):
        self.files = files
        self.anns_by_image = anns_by_image
        self.index = 0

        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.subplots_adjust(bottom=0.12)

        ax_prev = self.fig.add_axes([0.1, 0.03, 0.12, 0.05])
        ax_next = self.fig.add_axes([0.78, 0.03, 0.12, 0.05])
        self.btn_prev = Button(ax_prev, "Anterior")
        self.btn_next = Button(ax_next, "Siguiente")
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)

    def draw(self):
        self.ax.clear()
        img_path, img_id = self.files[self.index]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            self.ax.set_title(f"No se pudo abrir: {img_path.name}")
            self.ax.axis("off")
            self.fig.canvas.draw_idle()
            return

        self.ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        _draw_boxes(self.ax, img_id, self.anns_by_image)
        self.ax.set_title(f"{img_path.name}  ({self.index + 1}/{len(self.files)})")
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def next_image(self, event):
        if self.index < len(self.files) - 1:
            self.index += 1
            self.draw()

    def prev_image(self, event):
        if self.index > 0:
            self.index -= 1
            self.draw()


# ---------------------------------------------------------------------------
# Stacked browser — 5 panels: original, composite, ch1, ch2, ch3
# ---------------------------------------------------------------------------

class StackedBrowser:
    def __init__(self, files_orig, files_stacked, anns_by_image, combo: str):
        self.files_orig    = files_orig     # (path, img_id) desde train/
        self.files_stacked = files_stacked  # (path, img_id) desde train_<combo>/
        self.anns_by_image = anns_by_image
        self.combo         = combo
        self.ch_names      = [CHANNEL_NAMES[k] for k in COMBOS[combo]]
        self.index         = 0

        self.fig = plt.figure(figsize=(18, 10))
        self.fig.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.05)

        self.ax_orig  = self.fig.add_subplot(2, 3, 1)
        self.ax_stack = self.fig.add_subplot(2, 3, 2)
        self.ax_ch    = [self.fig.add_subplot(2, 3, 4 + i) for i in range(3)]

        ax_prev = self.fig.add_axes([0.1,  0.03, 0.12, 0.05])
        ax_next = self.fig.add_axes([0.78, 0.03, 0.12, 0.05])
        self.btn_prev = Button(ax_prev, "Anterior")
        self.btn_next = Button(ax_next, "Siguiente")
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)

    def draw(self):
        orig_path,    img_id    = self.files_orig[self.index]
        stacked_path, _         = self.files_stacked[self.index]

        orig_bgr    = cv2.imread(str(orig_path))
        stacked_bgr = cv2.imread(str(stacked_path))
        if orig_bgr is None or stacked_bgr is None:
            return

        orig_rgb    = cv2.cvtColor(orig_bgr,    cv2.COLOR_BGR2RGB)
        stacked_rgb = cv2.cvtColor(stacked_bgr, cv2.COLOR_BGR2RGB)

        # Canales individuales del stacked (R=Ch1, G=Ch2, B=Ch3)
        channels = [stacked_rgb[:, :, i] for i in range(3)]

        for ax in [self.ax_orig, self.ax_stack] + self.ax_ch:
            ax.clear()
            ax.axis("off")

        self.ax_orig.imshow(orig_rgb)
        _draw_boxes(self.ax_orig, img_id, self.anns_by_image)
        self.ax_orig.set_title("Original", fontsize=10)

        self.ax_stack.imshow(stacked_rgb)
        _draw_boxes(self.ax_stack, img_id, self.anns_by_image)
        self.ax_stack.set_title(
            f"Stacked [{' | '.join(self.ch_names)}]\n(lo que ve el modelo)", fontsize=9)

        for i, (ax, name, ch) in enumerate(zip(self.ax_ch, self.ch_names, channels)):
            ax.imshow(ch, cmap="gray")
            _draw_boxes(ax, img_id, self.anns_by_image)
            ax.set_title(f"Ch{i+1}: {name}", fontsize=10)

        self.fig.suptitle(
            f"[{self.combo.upper()}]  {orig_path.name}  "
            f"({self.index + 1}/{len(self.files_orig)})",
            fontsize=12, fontweight="bold"
        )
        self.fig.canvas.draw_idle()

    def next_image(self, event):
        if self.index < len(self.files_orig) - 1:
            self.index += 1
            self.draw()

    def prev_image(self, event):
        if self.index > 0:
            self.index -= 1
            self.draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", type=str, default=None,
                        choices=list(COMBOS.keys()),
                        help="Mostrar modo stacked para este combo (ej: run1)")
    args = parser.parse_args()

    if args.combo:
        stacked_dir = BASE_DIR / "data" / "images" / f"train_{args.combo}"
        if not stacked_dir.exists():
            raise RuntimeError(
                f"No existe {stacked_dir}. Ejecuta primero:\n"
                f"  python -m src.preprocess --combo {args.combo}"
            )
        files_orig,    anns_by_image = load_images_and_annotations(IMG_DIR)
        files_stacked, _             = load_images_and_annotations(stacked_dir)
        if not files_orig or not files_stacked:
            raise RuntimeError("No se encontraron imágenes.")
        print(f"Combo: {args.combo} — {COMBO_LABELS[args.combo]}")
        print(f"Cargando desde: {stacked_dir}")
        browser = StackedBrowser(files_orig, files_stacked, anns_by_image, args.combo)
    else:
        files, anns_by_image = load_images_and_annotations(IMG_DIR)
        if not files:
            raise RuntimeError("No se encontraron imágenes.")
        browser = ImageBrowser(files, anns_by_image)

    browser.draw()
    plt.show()


if __name__ == "__main__":
    main()
