import cv2
from pathlib import Path

# Pon aquí la ruta de UNA sola imagen donde se vea el RFI azul clarito
IMG_PATH = "data/images/train/4.png"

def test_channels():
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        print("No se pudo cargar la imagen")
        return
        
    # OpenCV carga en orden B, G, R (Azul, Verde, Rojo)
    canal_azul = img_bgr[:, :, 0]
    canal_verde = img_bgr[:, :, 1]
    canal_rojo = img_bgr[:, :, 2]
    
    cv2.imwrite("test_1_CANAL_AZUL.jpg", canal_azul)
    cv2.imwrite("test_2_CANAL_VERDE.jpg", canal_verde)
    cv2.imwrite("test_3_CANAL_ROJO.jpg", canal_rojo)
    
    print("¡Imágenes guardadas! Ábrelas y mira en cuál brilla más la raya blanca.")

if __name__ == "__main__":
    test_channels()