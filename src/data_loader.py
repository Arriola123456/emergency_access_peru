"""T1.a — Descarga reproducible y lectura cruda de los 4 datasets.

Este módulo contiene los URLs resueltos de los portales abiertos, la lógica
de descarga con headers de navegador (varios portales bloquean User-Agents
de bot) y los lectores crudos que devuelven DataFrames sin transformar.

La limpieza vive en `src/cleaning.py`; esta separación permite que:
  - el grader pueda ejecutar `python -m src.data_loader` solo para
    verificar que las descargas funcionan, y
  - los tests unitarios puedan mockear los lectores sin tocar disco.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from src.config import DATA_RAW
from src.utils import get_logger

log = get_logger(__name__)

# --- URLs resueltos de los portales CKAN -----------------------------------

URL_IPRESS_MINSA = (
    "https://www.datosabiertos.gob.pe/sites/default/files/recursos/2017/09/IPRESS.csv"
)
URL_CENTROS_POBLADOS_ZIP = (
    "https://www.datosabiertos.gob.pe/sites/default/files/CCPP_0.zip"
)
URL_EMERGENCIAS_SUSALUD = (
    "http://datos.susalud.gob.pe/sites/default/files/ConsultaC1_2024_v22.csv"
)
URL_DISTRITOS_BASE = (
    "https://github.com/d2cml-ai/Data-Science-Python/raw/main/_data/Folium"
)
DISTRITOS_EXTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
}


# --- Descarga --------------------------------------------------------------

def _download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        log.info(f"cached {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    log.info(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=_HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
    log.info(f"saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_raw(force: bool = False) -> None:
    """Descarga los 4 datasets a `data/raw/` (idempotente)."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    _download(URL_IPRESS_MINSA, DATA_RAW / "IPRESS.csv", force=force)

    ccpp_zip = _download(URL_CENTROS_POBLADOS_ZIP, DATA_RAW / "CCPP.zip", force=force)
    ccpp_dir = DATA_RAW / "CCPP"
    if force or not ccpp_dir.exists() or not any(ccpp_dir.iterdir()):
        ccpp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ccpp_zip) as z:
            z.extractall(ccpp_dir)
        log.info(f"extracted CCPP.zip ({len(list(ccpp_dir.rglob('*')))} files)")

    _download(URL_EMERGENCIAS_SUSALUD, DATA_RAW / "emergencias_2024.csv", force=force)

    for ext in DISTRITOS_EXTS:
        _download(
            f"{URL_DISTRITOS_BASE}/DISTRITOS{ext}",
            DATA_RAW / f"DISTRITOS{ext}",
            force=force,
        )


# --- Lectores crudos -------------------------------------------------------

def read_ipress_raw() -> pd.DataFrame:
    """IPRESS MINSA — CSV crudo en latin-1, separador `,`."""
    return pd.read_csv(
        DATA_RAW / "IPRESS.csv", encoding="latin-1", sep=",", low_memory=False
    )


def read_emergencies_raw() -> pd.DataFrame:
    """SUSALUD producción de emergencias 2024 — CSV crudo en latin-1, separador `;`."""
    return pd.read_csv(
        DATA_RAW / "emergencias_2024.csv",
        encoding="latin-1",
        sep=";",
        low_memory=False,
    )


def read_centros_poblados_raw() -> gpd.GeoDataFrame:
    """Centros Poblados INEI — primer shapefile dentro de `data/raw/CCPP/`."""
    ccpp_dir = DATA_RAW / "CCPP"
    shps = list(ccpp_dir.rglob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"no se encontró .shp bajo {ccpp_dir}")
    return gpd.read_file(shps[0])


def read_distritos_raw() -> gpd.GeoDataFrame:
    """DISTRITOS shapefile (repo del curso)."""
    return gpd.read_file(DATA_RAW / "DISTRITOS.shp")


# --- Main: solo descarga --------------------------------------------------

def main() -> None:
    log.info("=== data_loader: descarga de datasets crudos ===")
    download_raw()
    log.info("descarga completa — corre `python -m src.cleaning` para limpieza")


if __name__ == "__main__":
    main()
