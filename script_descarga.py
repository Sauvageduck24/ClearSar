import os
import geopandas as gpd
from eotdl.datasets import stage_dataset_file
from tqdm import tqdm

# Carpeta base donde está tu dataset ClearSAR
# Cambiado para usar el directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))
download_base = os.path.join(current_dir, "ClearSAR")

# Carga el catálogo
catalog_path = os.path.join(download_base, "catalog.v1.parquet")
catalog = gpd.read_parquet(catalog_path)

# Recorre todos los assets con tqdm
for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Descargando assets"):
    asset_info = row['assets']['asset']
    href = asset_info['href']
    
    # Construye la ruta local dentro de ClearSAR
    local_path = os.path.join(download_base, row['id'].replace("/", os.sep))
    
    # Crea carpetas si no existen
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Solo descarga si no existe
    if not os.path.exists(local_path):
        stage_dataset_file(href, os.path.dirname(local_path))