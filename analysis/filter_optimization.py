import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from pathlib import Path
from scipy.stats import entropy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- CONFIGURACIÓN ---
ANN_PATH = Path("data/annotations/instances_train.json")
TRAIN_IMG_DIR = Path("data/images/train")

# --- FILTROS ---
def filter_clahe(img, clip_limit=3.0, tile_size=8):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def filter_tophat(img, kernel_w=25, kernel_h=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    res_visible = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(res_visible, cv2.COLOR_GRAY2RGB)

def filter_log(img, gain=20.0):
    img_f = img.astype(np.float32) / 255.0
    res = np.log1p(img_f * gain) / np.log1p(gain)
    return (res * 255).astype(np.uint8)

def filter_lee_sigma(img, window_size=5):
    img_f = img.astype(np.float32) / 255.0
    kernel = np.ones((window_size, window_size), np.float32) / (window_size**2)
    local_mean = cv2.filter2D(img_f, -1, kernel)
    local_mean_sq = cv2.filter2D(img_f**2, -1, kernel)
    local_var = np.maximum(local_mean_sq - local_mean**2, 0)
    noise_var = np.mean(local_var)
    weights = local_var / (local_var + noise_var + 1e-6)
    res = local_mean + weights * (img_f - local_mean)
    return (np.clip(res, 0, 1) * 255).astype(np.uint8)

def filter_unsharp(img, sigma=2.0, amount=1.5):
    gauss = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, amount, gauss, -(amount - 1.0), 0)

def filter_gabor(img, ksize=15, sigma=3.0, lam=10.0, gamma=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    theta = np.pi / 2
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, -1, kernel)
    filtered_8u = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, mask = cv2.threshold(filtered_8u, 10, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

def filter_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def filter_bilateral_tophat(img, d=9, sigma_color=50, sigma_space=50, kernel_w=25):
    smooth = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    res = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(res, 5, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

FILTER_NAMES = [
    "Original",
    "CLAHE",
    "Top-Hat (Horiz)",
    "Log Transf",
    "Lee Sigma",
    "Unsharp Mask",
    "Gabor Horiz",
    "Bilateral",
    "Bilateral + Top-Hat",
]

def _score_filtered_gray(filtered_gray, bboxes):
    if not bboxes:
        return 0.0

    h_img, w_img = filtered_gray.shape
    roi_means = []
    for x, y, w, h in bboxes:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w_img, int(x + w))
        y1 = min(h_img, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        roi = filtered_gray[y0:y1, x0:x1]
        roi_means.append(float(np.mean(roi)))

    if not roi_means:
        return 0.0

    mean_signal = float(np.mean(roi_means))
    mean_background = float(np.mean(filtered_gray))

    snr = mean_signal / (mean_background + 1e-6)
    signal_strength = mean_signal / 255.0 
    score_final = snr * signal_strength
    return score_final

def optimize_filter_params(img_rgb, bboxes):
    best = {
        "CLAHE": {"clip_limit": 3.0, "tile_size": 8, "score": 0.0},
        "Top-Hat (Horiz)": {"kernel_w": 25, "kernel_h": 1, "score": 0.0},
        "Log Transf": {"gain": 20.0, "score": 0.0},
        "Lee Sigma": {"window_size": 5, "score": 0.0},
        "Unsharp Mask": {"sigma": 2.0, "amount": 1.5, "score": 0.0},
        "Gabor Horiz": {"ksize": 15, "sigma": 3.0, "lam": 10.0, "gamma": 0.5, "score": 0.0},
        "Bilateral": {"d": 9, "sigma_color": 75, "sigma_space": 75, "score": 0.0},
        "Bilateral + Top-Hat": {"d": 9, "sigma_color": 50, "sigma_space": 50, "kernel_w": 25, "score": 0.0},
    }

    if not bboxes:
        return best

    for clip_limit in [1.5, 3.0, 5.0]:
        for tile_size in [6, 8, 12]:
            gray = cv2.cvtColor(filter_clahe(img_rgb, clip_limit, tile_size), cv2.COLOR_RGB2GRAY)
            score = _score_filtered_gray(gray, bboxes)
            if score > best["CLAHE"]["score"]:
                best["CLAHE"] = {"clip_limit": clip_limit, "tile_size": tile_size, "score": score}

    for kernel_w in [17, 25, 35]:
        gray = cv2.cvtColor(filter_tophat(img_rgb, kernel_w, 1), cv2.COLOR_RGB2GRAY)
        score = _score_filtered_gray(gray, bboxes)
        if score > best["Top-Hat (Horiz)"]["score"]:
            best["Top-Hat (Horiz)"] = {"kernel_w": kernel_w, "kernel_h": 1, "score": score}

    for gain in [10.0, 20.0, 35.0]:
        gray = cv2.cvtColor(filter_log(img_rgb, gain), cv2.COLOR_RGB2GRAY)
        score = _score_filtered_gray(gray, bboxes)
        if score > best["Log Transf"]["score"]:
            best["Log Transf"] = {"gain": gain, "score": score}

    for window_size in [5, 7, 9]:
        gray = cv2.cvtColor(filter_lee_sigma(img_rgb, window_size), cv2.COLOR_RGB2GRAY)
        score = _score_filtered_gray(gray, bboxes)
        if score > best["Lee Sigma"]["score"]:
            best["Lee Sigma"] = {"window_size": window_size, "score": score}

    for sigma in [1.2, 2.4, 3.0]:
        for amount in [1.3, 1.5, 1.8]:
            gray = cv2.cvtColor(filter_unsharp(img_rgb, sigma, amount), cv2.COLOR_RGB2GRAY)
            score = _score_filtered_gray(gray, bboxes)
            if score > best["Unsharp Mask"]["score"]:
                best["Unsharp Mask"] = {"sigma": sigma, "amount": amount, "score": score}

    for ksize in [13, 15, 19]:
        for sigma in [2.5, 3.5, 5.0]:
            for lam in [8.0, 10.0, 14.0]:
                gray = cv2.cvtColor(filter_gabor(img_rgb, ksize, sigma, lam, 0.5), cv2.COLOR_RGB2GRAY)
                score = _score_filtered_gray(gray, bboxes)
                if score > best["Gabor Horiz"]["score"]:
                    best["Gabor Horiz"] = {"ksize": ksize, "sigma": sigma, "lam": lam, "gamma": 0.5, "score": score}

    for d in [7, 9, 11]:
        for sigma_color in [50, 75, 100]:
            for sigma_space in [50, 75, 100]:
                gray = cv2.cvtColor(filter_bilateral(img_rgb, d, sigma_color, sigma_space), cv2.COLOR_RGB2GRAY)
                score = _score_filtered_gray(gray, bboxes)
                if score > best["Bilateral"]["score"]:
                    best["Bilateral"] = {"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space, "score": score}

    for d in [7, 9, 11]:
        for sigma_color in [50, 75, 100]:
            for sigma_space in [50, 75, 100]:
                for kernel_w in [17, 25, 35]:
                    gray = cv2.cvtColor(filter_bilateral_tophat(img_rgb, d, sigma_color, sigma_space, kernel_w), cv2.COLOR_RGB2GRAY)
                    score = _score_filtered_gray(gray, bboxes)
                    if score > best["Bilateral + Top-Hat"]["score"]:
                        best["Bilateral + Top-Hat"] = {"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space, "kernel_w": kernel_w, "score": score}

    return best

def apply_filter(name, img_rgb, best_params):
    if name == "Original":
        return img_rgb
    if name == "CLAHE":
        p = best_params["CLAHE"]
        return filter_clahe(img_rgb, p["clip_limit"], p["tile_size"])
    if name == "Top-Hat (Horiz)":
        p = best_params["Top-Hat (Horiz)"]
        return filter_tophat(img_rgb, p["kernel_w"], p["kernel_h"])
    if name == "Log Transf":
        p = best_params["Log Transf"]
        return filter_log(img_rgb, p["gain"])
    if name == "Lee Sigma":
        p = best_params["Lee Sigma"]
        return filter_lee_sigma(img_rgb, p["window_size"])
    if name == "Unsharp Mask":
        p = best_params["Unsharp Mask"]
        return filter_unsharp(img_rgb, p["sigma"], p["amount"])
    if name == "Gabor Horiz":
        p = best_params["Gabor Horiz"]
        return filter_gabor(img_rgb, p["ksize"], p["sigma"], p["lam"], p["gamma"])
    if name == "Bilateral":
        p = best_params["Bilateral"]
        return filter_bilateral(img_rgb, p["d"], p["sigma_color"], p["sigma_space"])
    if name == "Bilateral + Top-Hat":
        p = best_params["Bilateral + Top-Hat"]
        return filter_bilateral_tophat(img_rgb, p["d"], p["sigma_color"], p["sigma_space"], p["kernel_w"])
    return img_rgb

# --- MULTIPROCESSING WORKERS ---
def _worker_optimize_image(args):
    img_path, img_id, anns_data = args
    if not Path(img_path).exists(): return None
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: return None
    bboxes = anns_data.get(img_id, [])
    if not bboxes: return None
    return optimize_filter_params(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bboxes)

def _worker_evaluate_image(args):
    """Evalúa los parámetros optimizados en una imagen para sacar la nota media real"""
    img_path, img_id, anns_data, global_params = args
    if not Path(img_path).exists(): return None
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: return None
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bboxes = anns_data.get(img_id, [])
    if not bboxes: return None

    scores = {}
    for name in FILTER_NAMES:
        t_img = apply_filter(name, img_rgb, global_params)
        gray = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)
        scores[name] = _score_filtered_gray(gray, bboxes)
    
    return scores

def optimize_filter_params_global(all_files, anns_by_image, max_images=None):
    if max_images is not None: all_files = all_files[:max_images]
    
    global_best = {k: {"score": 0.0, "count": 0} for k in FILTER_NAMES if k != "Original"}
    # Inicializar campos necesarios
    for k, v in optimize_filter_params(np.zeros((10,10,3), dtype=np.uint8), []).items():
        if k != "Original": global_best[k].update({pk: pv for pk, pv in v.items() if pk != "score"})

    num_workers = min(8, cpu_count())
    print(f"\n[FASE 1] Optimizando parámetros con {num_workers} núcleos...")
    worker_inputs = [(str(p), idx, {idx: [a["bbox"] for a in anns_by_image.get(idx, [])]}) for p, idx in all_files if anns_by_image.get(idx, [])]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(_worker_optimize_image, worker_inputs), total=len(worker_inputs), desc="Optimizando", unit="img"))

    for img_best in results:
        if img_best is None: continue
        for filter_name in global_best:
            if img_best[filter_name]["score"] > global_best[filter_name].get("best_score", 0):
                global_best[filter_name]["best_score"] = img_best[filter_name]["score"]
                for key in img_best[filter_name]:
                    if key not in ["score", "count", "best_score"]:
                        global_best[filter_name][key] = img_best[filter_name][key]

    return {k: {pk: pv for pk, pv in v.items() if pk not in ["score", "count", "best_score"]} for k, v in global_best.items()}

def evaluate_filters_global(all_files, anns_by_image, global_params, max_images=None):
    if max_images is not None: all_files = all_files[:max_images]
    
    num_workers = min(8, cpu_count())
    print(f"\n[FASE 2] Evaluando el Ranking Global en {len(all_files)} imágenes...")
    
    worker_inputs = [(str(p), idx, {idx: [a["bbox"] for a in anns_by_image.get(idx, [])]}, global_params) for p, idx in all_files if anns_by_image.get(idx, [])]
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(_worker_evaluate_image, worker_inputs), total=len(worker_inputs), desc="Evaluando Ranking", unit="img"))

    # Agregar resultados y guardar detallados
    aggregated = {name: [] for name in FILTER_NAMES}
    detailed_results = []
    
    for worker_input, res in zip(worker_inputs, results):
        if res is None: continue
        img_path, img_id = worker_input[0], worker_input[1]
        for name, score in res.items():
            aggregated[name].append(score)
            detailed_results.append({
                "image_id": img_id,
                "image_path": Path(img_path).name,
                "filter": name,
                "score": score
            })
    
    # Guardar CSV detallado
    if detailed_results:
        df_detailed = pd.DataFrame(detailed_results)
        csv_path = Path("filter_statistics.csv")
        df_detailed.to_csv(csv_path, index=False)
        print(f"\n✅ Estadísticas guardadas en: {csv_path}")

    df = pd.DataFrame([{"Filter": name, "Score": np.mean(scores)} for name, scores in aggregated.items() if scores])
    return df

# --- INTERFAZ GRÁFICA ---
class MultiFilterBrowser:
    def __init__(self, all_files, anns_by_image, global_params):
        self.all_files = all_files
        self.anns_by_image = anns_by_image
        self.global_params = global_params
        self.index = 0

        self.fig, self.axes = plt.subplots(3, 3, figsize=(19, 12))
        self.axes = self.axes.flatten()
        self.fig.subplots_adjust(bottom=0.15, hspace=0.3)

        ax_next = self.fig.add_axes([0.8, 0.03, 0.1, 0.05])
        ax_prev = self.fig.add_axes([0.1, 0.03, 0.1, 0.05])
        self.btn_next = Button(ax_next, 'Siguiente')
        self.btn_prev = Button(ax_prev, 'Anterior')
        self.btn_next.on_clicked(self.next_img)
        self.btn_prev.on_clicked(self.prev_img)

    def draw(self):
        img_path, img_id = self.all_files[self.index]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bboxes = [ann['bbox'] for ann in self.anns_by_image.get(img_id, [])]
        
        for i, name in enumerate(FILTER_NAMES):
            ax = self.axes[i]
            ax.clear()
            t_img = apply_filter(name, img_rgb, self.global_params)
            
            gray = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)
            score = _score_filtered_gray(gray, bboxes) if bboxes else 0.0
            
            ax.imshow(t_img)
            ax.set_title(f"{name}\nScore: {score:.3f}", fontsize=11)
            ax.axis('off')
            
            if img_id in self.anns_by_image:
                for ann in self.anns_by_image[img_id]:
                    x, y, w, h = ann['bbox']
                    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='lime', facecolor='none'))

        self.fig.suptitle(f"Visualizador - {img_path.name} ({self.index+1}/{len(self.all_files)})", fontsize=16)
        self.fig.canvas.draw_idle()

    def next_img(self, event):
        if self.index < len(self.all_files) - 1: self.index += 1; self.draw()

    def prev_img(self, event):
        if self.index > 0: self.index -= 1; self.draw()

def run(max_opt_images=50, max_eval_images=300):
    with open(ANN_PATH, 'r') as f: coco = json.load(f)
    anns = {}
    for a in coco['annotations']: anns.setdefault(a['image_id'], []).append(a)
    files = [(TRAIN_IMG_DIR / i['file_name'], i['id']) for i in coco['images']]
    
    # 1. Optimización (Muestra pequeña para velocidad)
    global_params = optimize_filter_params_global(files, anns, max_images=max_opt_images)
    
    # 2. Evaluación (Muestra grande para sacar el ranking real)
    ranking_df = evaluate_filters_global(files, anns, global_params, max_images=max_eval_images)
    
    # 3. Mostrar el Veredicto por consola
    print("\n" + "="*60)
    print("🏆 RANKING GLOBAL DE FILTROS (Basado en 300 imágenes) 🏆")
    print("="*60)
    ranking_df_sorted = ranking_df.sort_values(by="Score", ascending=False)
    print(ranking_df_sorted.to_string(index=False))
    print("="*60)
    
    # Extraer el mejor filtro excluyendo el "Original" (por si acaso diera alto)
    best_filter = ranking_df_sorted[ranking_df_sorted["Filter"] != "Original"].iloc[0]["Filter"]
    print(f"\n🚀 EL GANADOR ABSOLUTO ES: [ {best_filter.upper()} ] 🚀")
    print("Parámetros que debes inyectar en YOLO:")
    print(global_params.get(best_filter, "N/A"))
    print("\nParámetros optimizados de todos los filtros:")
    for filter_name in FILTER_NAMES:
        if filter_name == "Original":
            print(f"- {filter_name}: N/A (sin parámetros)")
            continue
        print(f"- {filter_name}: {global_params.get(filter_name, 'N/A')}")
    print("="*60 + "\n")
    
    # 4. Abrir la interfaz
    browser = MultiFilterBrowser(files[:max_eval_images], anns, global_params)
    browser.draw()
    plt.show()

if __name__ == "__main__":
    run(max_opt_images=None, max_eval_images=500)