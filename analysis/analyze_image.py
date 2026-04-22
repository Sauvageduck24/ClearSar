import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

# --- paths ---
img_dir = "data/yolo/images/train/"
label_dir = "data/yolo/labels/train/"
out_csv = "stats.csv"

rows = []

img_paths = glob(os.path.join(img_dir, "*.*"))

for img_path in img_paths:
    name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, name + ".txt")

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue

    h, w = img.shape[:2]
    img = img.astype(np.float32)

    # --- stats globales por canal ---
    global_stats = {}
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        global_stats[f"ch{c}_mean"] = ch.mean()
        global_stats[f"ch{c}_std"] = ch.std()
        global_stats[f"ch{c}_min"] = ch.min()
        global_stats[f"ch{c}_max"] = ch.max()

    # --- leer boxes ---
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)
                boxes.append((x1, y1, x2, y2))

    # --- stats dentro de boxes ---
    box_stats = {}
    if len(boxes) > 0:
        pixels = []
        for (x1, y1, x2, y2) in boxes:
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                pixels.append(crop.reshape(-1, img.shape[2]))

        if len(pixels) > 0:
            pixels = np.concatenate(pixels, axis=0)

            for c in range(img.shape[2]):
                ch = pixels[:, c]
                box_stats[f"box_ch{c}_mean"] = ch.mean()
                box_stats[f"box_ch{c}_std"] = ch.std()
                box_stats[f"box_ch{c}_min"] = ch.min()
                box_stats[f"box_ch{c}_max"] = ch.max()
    else:
        for c in range(img.shape[2]):
            box_stats[f"box_ch{c}_mean"] = np.nan
            box_stats[f"box_ch{c}_std"] = np.nan
            box_stats[f"box_ch{c}_min"] = np.nan
            box_stats[f"box_ch{c}_max"] = np.nan

    row = {
        "image": name,
        "num_boxes": len(boxes),
        **global_stats,
        **box_stats
    }

    rows.append(row)

# --- guardar ---
df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)

print(f"Guardado en {out_csv}")