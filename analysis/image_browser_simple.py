import json
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


BASE_DIR = Path(__file__).resolve().parents[1]
ANN_PATH = BASE_DIR / "data" / "annotations" / "instances_train.json"
IMG_DIR = BASE_DIR / "data" / "images" / "train"


def load_images_and_annotations():
    with ANN_PATH.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    files = []
    for img in coco["images"]:
        img_path = IMG_DIR / img["file_name"]
        if img_path.exists():
            files.append((img_path, img["id"]))

    return files, anns_by_image


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

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img_rgb)

        for ann in self.anns_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            self.ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=1.5,
                    edgecolor="lime",
                    facecolor="none",
                )
            )

        self.ax.set_title(f"{img_path.name} ({self.index + 1}/{len(self.files)})")
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


def main():
    files, anns_by_image = load_images_and_annotations()
    if not files:
        raise RuntimeError("No se encontraron imágenes para mostrar.")

    browser = ImageBrowser(files, anns_by_image)
    browser.draw()
    plt.show()


if __name__ == "__main__":
    main()