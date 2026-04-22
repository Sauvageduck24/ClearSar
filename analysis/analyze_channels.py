import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# --- CONFIGURACIÓN ---
# Puedes añadir más carpetas a la lista si quieres
FOLDERS_TO_ANALYZE = [
    Path("data/images/train"),
    Path("data/images/test")
]

# Consideramos que un píxel es "RFI Brillante" si supera este valor (de 0 a 255)
RFI_THRESHOLD = 200

def analyze_image(img_path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None
        
    # OpenCV lee en BGR, separamos para mayor claridad
    b = img_bgr[:, :, 0]
    g = img_bgr[:, :, 1]
    r = img_bgr[:, :, 2]
    
    # 1. Estadísticas básicas (Media de brillo de todo el mar/imagen)
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    
    # 2. Picos de intensidad (¿Hasta dónde llega el RFI en cada canal?)
    max_r, max_g, max_b = np.max(r), np.max(g), np.max(b)
    
    # 3. Análisis de Dominancia RFI (¿Quién tiene los píxeles más brillantes?)
    # Contamos cuántos píxeles en cada canal superan el umbral de RFI
    rfi_pixels_r = np.sum(r > RFI_THRESHOLD)
    rfi_pixels_g = np.sum(g > RFI_THRESHOLD)
    rfi_pixels_b = np.sum(b > RFI_THRESHOLD)
    
    # Determinamos el "Color" del RFI en esta imagen concreta
    dominant_channel = "Ninguno (Sin RFI brillante)"
    if max(rfi_pixels_r, rfi_pixels_g, rfi_pixels_b) > 0:
        if rfi_pixels_r > rfi_pixels_g and rfi_pixels_r > rfi_pixels_b:
            dominant_channel = "Rojo (VV)"
        elif rfi_pixels_g > rfi_pixels_r and rfi_pixels_g > rfi_pixels_b:
            dominant_channel = "Verde (VH)"
        elif rfi_pixels_b > rfi_pixels_r and rfi_pixels_b > rfi_pixels_g:
            dominant_channel = "Azul (Ratio)"
        else:
            dominant_channel = "Mixto (Cian/Magenta/Blanco)"

    return {
        "Archivo": img_path.name,
        "Carpeta": img_path.parent.name,
        "Mean_R": mean_r, "Mean_G": mean_g, "Mean_B": mean_b,
        "Max_R": max_r, "Max_G": max_g, "Max_B": max_b,
        "RFI_Pixels_R": rfi_pixels_r,
        "RFI_Pixels_G": rfi_pixels_g,
        "RFI_Pixels_B": rfi_pixels_b,
        "Dominante": dominant_channel
    }

def main():
    all_results = []
    
    for folder in FOLDERS_TO_ANALYZE:
        if not folder.exists():
            print(f"⚠️ Carpeta no encontrada: {folder}")
            continue
            
        files = list(folder.glob("*.*"))
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif'}
        files = [f for f in files if f.suffix.lower() in valid_extensions]
        
        print(f"\n🔍 Analizando {len(files)} imágenes en {folder}...")
        for f in tqdm(files):
            res = analyze_image(f)
            if res:
                all_results.append(res)
                
    if not all_results:
        print("No se analizaron imágenes.")
        return
        
    # --- ANÁLISIS GLOBAL CON PANDAS ---
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("📊 REPORTE GLOBAL DE CANALES SAR 📊")
    print("="*60)
    
    print("\n1. BRILLO MEDIO GLOBAL (Contexto del Mar):")
    print(f"   Canal Rojo (R):  {df['Mean_R'].mean():.2f}")
    print(f"   Canal Verde (G): {df['Mean_G'].mean():.2f}")
    print(f"   Canal Azul (B):  {df['Mean_B'].mean():.2f}")
    
    print("\n2. PROMEDIO DE LOS PICOS MÁXIMOS (Brillo del RFI):")
    print(f"   Canal Rojo (R):  {df['Max_R'].mean():.2f}")
    print(f"   Canal Verde (G): {df['Max_G'].mean():.2f}")
    print(f"   Canal Azul (B):  {df['Max_B'].mean():.2f}")
    
    print("\n3. ¿DÓNDE VIVE EL RFI? (Imágenes por canal dominante):")
    counts = df['Dominante'].value_counts()
    for canal, cantidad in counts.items():
        porcentaje = (cantidad / len(df)) * 100
        print(f"   {canal}: {cantidad} imágenes ({porcentaje:.1f}%)")
        
    print("\n" + "="*60)
    
    # Guardar a CSV por si quieres investigarlo en Excel
    df.to_csv("analisis_canales_sar.csv", index=False)
    print("📁 Reporte detallado guardado en 'analisis_canales_sar.csv'")

if __name__ == "__main__":
    main()