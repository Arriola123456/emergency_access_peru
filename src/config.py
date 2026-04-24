"""Rutas, CRS y constantes del proyecto."""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MAPS = OUTPUTS / "maps"
TABLES = OUTPUTS / "tables"

for p in (DATA_RAW, DATA_PROCESSED, FIGURES, MAPS, TABLES):
    p.mkdir(parents=True, exist_ok=True)

# --- CRS ---
CRS_WGS84 = "EPSG:4326"       # lat/lon, para folium y datos de origen
CRS_PERU_UTM = "EPSG:32718"   # UTM zona 18S, métrico; usado para distancias

# --- Dataset URLs (portales CKAN; resueltos a los archivos directos en ingest.py) ---
URL_CENTROS_POBLADOS_PAGE = "https://www.datosabiertos.gob.pe/dataset/dataset-centros-poblados"
URL_IPRESS_MINSA_PAGE = "https://www.datosabiertos.gob.pe/dataset/minsa-ipress"
URL_EMERGENCIAS_SUSALUD_PAGE = "http://datos.susalud.gob.pe/dataset/consulta-c1-produccion-asistencial-en-emergencia-por-ipress"
URL_DISTRITOS_SHP_BASE = "https://github.com/d2cml-ai/Data-Science-Python/raw/main/_data/Folium"

# --- Parámetros de análisis ---
EMERGENCY_DISTANCE_THRESHOLD_KM = 30.0  # centros poblados "con acceso" si IPRESS de emergencia <= este umbral
ALT_EMERGENCY_DISTANCE_THRESHOLD_KM = 15.0  # especificación alternativa (sensibilidad)
