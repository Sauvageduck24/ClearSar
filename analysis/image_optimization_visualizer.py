import json
import ast
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import Button

# --- CONFIGURACIÓN ---
ANN_PATH = Path("data/annotations/instances_train.json")
IMG_DIR  = Path("data/images/train")
CSV_PATH = Path("filter_search_results/grid_search_results.csv")

# ---------------------------------------------------------------------------
# 1. LECTOR AUTOMÁTICO DE LA MEJOR CONFIGURACIÓN
# ---------------------------------------------------------------------------
def get_best_params():
    """Lee el CSV del optimizador y devuelve el diccionario Top 1."""
    default_params = {
        "kernel_w": 35, "channels": (0, 1, 2), "fusion": "max",
        "direction": "both", "pre_smooth": "bilateral"
    }
    
    if not CSV_PATH.exists():
        print(f"⚠️ No se encontró {CSV_PATH}. Usando parámetros por defecto.")
        return default_params
        
    try:
        df = pd.read_csv(CSV_PATH)
        best_row = df.iloc[0] # Cogemos la fila 0 (la mejor)
        
        params = {
            "kernel_w": int(best_row["kernel_w"]),
            "channels": ast.literal_eval(str(best_row["channels"])),
            "fusion": str(best_row["fusion"]),
            "direction": str(best_row["direction"]),
            "pre_smooth": str(best_row["pre_smooth"])
        }
        print(f"🏆 Mejor config cargada automáticamente:\n{params}")
        return params
    except Exception as e:
        print(f"⚠️ Error leyendo CSV: {e}. Usando defaults.")
        return default_params

# ---------------------------------------------------------------------------
# 2. GENERADOR DEL STACK YOLO (USANDO LOS PARÁMETROS TOP 1)
# ---------------------------------------------------------------------------
def _tophat_1ch(ch: np.ndarray, kw: int, direction: str) -> np.ndarray:
    """Aplica TopHat horizontal, vertical o ambos según el optimizador."""
    results = []
    if direction in ("horizontal", "both"):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        results.append(cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, k).astype(np.float32))
    if direction in ("vertical", "both"):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kw))
        results.append(cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, k).astype(np.float32))
    return np.maximum.reduce(results)

def create_optimized_yolo_stack(img_bgr: np.ndarray, params: dict) -> np.ndarray:
    # 1. CANAL ROJO (Contexto Limpio)
    # Cogemos el mínimo y le borramos rayas horizontales y verticales
    min_c = np.min(img_bgr, axis=2).astype(np.uint8)
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    bg_clean = cv2.morphologyEx(min_c, cv2.MORPH_OPEN, k_v)
    bg_clean = cv2.morphologyEx(bg_clean, cv2.MORPH_OPEN, k_h)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c_rojo = clahe.apply(bg_clean)

    # 2. EXTRACCIÓN DE SEÑAL (Con parámetros óptimos)
    split = cv2.split(img_bgr)
    anomaly_maps = []
    
    for idx in params["channels"]:
        ch = split[idx]
        if params["pre_smooth"] == "gaussian3":
            ch = cv2.GaussianBlur(ch, (3, 3), 0)
        elif params["pre_smooth"] == "gaussian5":
            ch = cv2.GaussianBlur(ch, (5, 5), 0)
        elif params["pre_smooth"] == "bilateral":
            ch = cv2.bilateralFilter(ch, 5, 50, 50)
            
        anomaly_maps.append(_tophat_1ch(ch, params["kernel_w"], params["direction"]))

    if params["fusion"] == "max":
        amap = np.maximum.reduce(anomaly_maps)
    elif params["fusion"] == "mean":
        amap = np.mean(anomaly_maps, axis=0)
    else:  # sum
        amap = np.sum(anomaly_maps, axis=0)

    # 3. CANALES VERDE Y AZUL PARA YOLO
    # Verde: Geometría pura (Normalizado)
    c_verde = cv2.normalize(amap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Azul: Resplandor para RFI tenue (Log Transform)
    f = amap.astype(np.float32) / 255.0
    c_azul = ((np.log1p(f * 100) / np.log1p(100)) * 255).astype(np.uint8)

    # Devolvemos imagen RGB (OpenCV usa BGR internamente)
    return cv2.merge([c_azul, c_verde, c_rojo])

# ---------------------------------------------------------------------------
# 3. INTERFAZ GRÁFICA (MATPLOTLIB)
# ---------------------------------------------------------------------------
class YoloVisionBrowser:
    def __init__(self):
        # Cargar cajas del JSON
        with open(ANN_PATH, "r") as f:
            coco = json.load(f)
        
        self.id_to_name = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        self.boxes_by_file = {}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0): continue
            fname = Path(self.id_to_name.get(ann.get("image_id", -1))).name
            bbox = ann.get("bbox")
            if fname and bbox:
                self.boxes_by_file.setdefault(fname, []).append([int(v) for v in bbox])

        # Cargar imágenes y parámetros
        self.files = sorted(list(IMG_DIR.glob("*.png")))
        self.best_params = get_best_params()
        self.index = 0
        
        # Configurar Figura (1 fila, 2 columnas)
        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(bottom=0.15) # Espacio para el botón
        
        # Botón
        ax_next = plt.axes([0.45, 0.05, 0.1, 0.05])
        self.btn_next = Button(ax_next, 'Siguiente Imagen')
        self.btn_next.on_clicked(self.next_image)
        
        self.draw()

    def draw(self):
        for ax in self.axs: ax.clear()
        
        img_path = self.files[self.index]
        fname = img_path.name
        img_bgr = cv2.imread(str(img_path))
        boxes = self.boxes_by_file.get(fname, [])
        
        if img_bgr is None: return

        # 1. Generar la imagen de YOLO
        img_yolo_bgr = create_optimized_yolo_stack(img_bgr, self.best_params)
        
        # Convertir a RGB para Matplotlib
        img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_yolo = cv2.cvtColor(img_yolo_bgr, cv2.COLOR_BGR2RGB)

        # 2. Dibujar Original
        self.axs[0].imshow(img_rgb_orig)
        self.axs[0].set_title(f"Original: {fname}", fontsize=12, fontweight='bold')
        self.axs[0].axis('off')

        # 3. Dibujar YOLO Stack
        self.axs[1].imshow(img_rgb_yolo)
        self.axs[1].set_title("Visión YOLO 3-Canales (Optimizada)", fontsize=12, fontweight='bold')
        self.axs[1].axis('off')

        # 4. Dibujar Cajas de Anotación (Ground Truth)
        for x, y, w, h in boxes:
            # Caja en la original (Rojo)
            rect1 = plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=2)
            self.axs[0].add_patch(rect1)
            # Caja en YOLO (Amarillo para que contraste con el fondo azul/rojo)
            rect2 = plt.Rectangle((x, y), w, h, fill=False, edgecolor="yellow", lw=2)
            self.axs[1].add_patch(rect2)

        plt.draw()

    def next_image(self, event):
        self.index = (self.index + 1) % len(self.files)
        self.draw()

if __name__ == "__main__":
    print("🚀 Iniciando Visualizador YOLO-Stack...")
    browser = YoloVisionBrowser()
    plt.show()