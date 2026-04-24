"""T1 — Descarga reproducible y limpieza de los 4 datasets a data/processed/."""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from src.config import CRS_WGS84, DATA_PROCESSED, DATA_RAW
from src.utils import first_existing, get_logger, normalize_ubigeo

log = get_logger(__name__)

# --- URLs resueltos de los portales CKAN ---
URL_IPRESS_MINSA = "https://www.datosabiertos.gob.pe/sites/default/files/recursos/2017/09/IPRESS.csv"
URL_CENTROS_POBLADOS_ZIP = "https://www.datosabiertos.gob.pe/sites/default/files/CCPP_0.zip"
URL_EMERGENCIAS_SUSALUD = "http://datos.susalud.gob.pe/sites/default/files/ConsultaC1_2024_v22.csv"
URL_DISTRITOS_BASE = "https://github.com/d2cml-ai/Data-Science-Python/raw/main/_data/Folium"
DISTRITOS_EXTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")

# Rango de coordenadas válido para Perú continental (+ margen amazónico)
PERU_LAT = (-18.5, 0.5)
PERU_LON = (-81.5, -68.0)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
}


# --- Descarga -------------------------------------------------------------------

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
    """Descarga los 4 datasets a data/raw/."""
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
        _download(f"{URL_DISTRITOS_BASE}/DISTRITOS{ext}", DATA_RAW / f"DISTRITOS{ext}", force=force)


# --- Helpers --------------------------------------------------------------------

def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    import unicodedata

    def clean(c: str) -> str:
        c = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode("ascii")
        return c.strip().upper().replace(" ", "_").replace(".", "_")

    out = df.copy()
    out.columns = [clean(c) for c in out.columns]
    return out


def _find_shapefile(folder: Path, name_hint: str | None = None) -> Path:
    shps = list(folder.rglob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"no .shp found under {folder}")
    if name_hint:
        filtered = [p for p in shps if name_hint.lower() in p.name.lower()]
        if filtered:
            return filtered[0]
    return shps[0]


# --- Loaders --------------------------------------------------------------------

def load_ipress() -> pd.DataFrame:
    """IPRESS MINSA — establecimientos de salud con coords.

    Nota: en el dataset fuente las columnas NORTE y ESTE están invertidas
    respecto a su nombre — NORTE contiene longitud (rango -81..-68) y ESTE
    contiene latitud (rango -18..0). Las remapeamos explícitamente.
    """
    df = pd.read_csv(DATA_RAW / "IPRESS.csv", encoding="latin-1", sep=",", low_memory=False)
    df = _normalize_colnames(df)
    log.info(f"IPRESS raw: {len(df):,} filas × {len(df.columns)} cols")

    df = df.rename(columns={
        "CODIGO_UNICO": "codigo",
        "NOMBRE_DEL_ESTABLECIMIENTO": "nombre",
        "UBIGEO": "ubigeo",
        "INSTITUCION": "institucion",
        "CLASIFICACION": "clasificacion",
        "CATEGORIA": "categoria",
        "NORTE": "lon",  # <-- invertido en fuente
        "ESTE": "lat",   # <-- invertido en fuente
    })

    keep = ["codigo", "nombre", "ubigeo", "institucion", "clasificacion", "categoria", "lat", "lon"]
    df = df[[c for c in keep if c in df.columns]].copy()

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["ubigeo"] = normalize_ubigeo(df["ubigeo"])
    df["codigo"] = df["codigo"].astype("string").str.strip()

    before = len(df)
    df = df.dropna(subset=["lat", "lon", "ubigeo"])
    df = df[df["lat"].between(*PERU_LAT) & df["lon"].between(*PERU_LON)]
    df = df.drop_duplicates(subset=["codigo"])
    log.info(f"IPRESS cleaned: {len(df):,} facilities (dropped {before - len(df):,})")
    return df.reset_index(drop=True)


def load_emergencies() -> pd.DataFrame:
    """Producción de emergencias SUSALUD 2024.

    Estructura: 1 fila por IPRESS × mes × sexo × edad, con NRO_TOTAL_ATENCIONES.
    SUSALUD anonimiza valores pequeños con el marcador 'NE_0001' (< umbral de
    privacidad). Se parsean como NaN y no contribuyen al total — equivalente
    a tratarlos como 0 para agregación anual.
    """
    df = pd.read_csv(DATA_RAW / "emergencias_2024.csv", encoding="latin-1", sep=";", low_memory=False)
    df = _normalize_colnames(df)
    log.info(f"EMERGENCIES raw: {len(df):,} filas × {len(df.columns)} cols")

    df = df.rename(columns={
        "CO_IPRESS": "codigo",
        "UBIGEO": "ubigeo",
        "NRO_TOTAL_ATENCIONES": "atenciones",
        "NRO_TOTAL_ATENDIDOS": "atendidos",
        "ANHO": "anho",
        "MES": "mes",
    })

    df["atenciones"] = pd.to_numeric(df["atenciones"], errors="coerce")
    df["atendidos"] = pd.to_numeric(df["atendidos"], errors="coerce")
    df["codigo"] = df["codigo"].astype("string").str.strip()
    df["ubigeo"] = normalize_ubigeo(df["ubigeo"])

    # NE_xxxx → NaN después de to_numeric; para la agregación anual, los
    # registros anonimizados se cuentan como 0 (fill antes de sumar).
    agg = (
        df.assign(atenciones=df["atenciones"].fillna(0), atendidos=df["atendidos"].fillna(0))
        .groupby(["codigo", "ubigeo"], dropna=False, as_index=False)[["atenciones", "atendidos"]]
        .sum()
    )
    log.info(f"EMERGENCIES aggregated: {len(agg):,} IPRESS-year; total atenciones: {agg['atenciones'].sum():,.0f}")
    return agg


def load_centros_poblados() -> gpd.GeoDataFrame:
    """Centros poblados INEI (shapefile).

    El shapefile no trae UBIGEO explícito; lo derivamos de los primeros 6
    dígitos del campo CÓDIGO (código INEI compuesto DEP+PROV+DIST+CCPP).
    Tampoco trae población; usamos peso uniforme en el baseline y el PR con
    otra fuente queda para la especificación alternativa si se integra.
    """
    ccpp_dir = DATA_RAW / "CCPP"
    shp = _find_shapefile(ccpp_dir)
    gdf = gpd.read_file(shp)
    geom_col = gdf.geometry.name
    gdf.columns = [c if c == geom_col else c for c in gdf.columns]
    log.info(f"CCPP raw: {len(gdf):,} centros poblados, geom CRS {gdf.crs}")

    codigo_col = first_existing(gdf, ["CÓDIGO", "CODIGO", "CÓD_INT", "COD_INT"])
    if codigo_col is None:
        raise RuntimeError(f"CCPP: no se encontró columna CÓDIGO en {list(gdf.columns)}")

    gdf = gdf.rename(columns={codigo_col: "codigo", "NOM_POBLAD": "nombre", "CAT_POBLAD": "categoria"})
    gdf["codigo"] = gdf["codigo"].astype("string").str.strip()
    gdf["ubigeo"] = normalize_ubigeo(gdf["codigo"].str.slice(0, 6))
    gdf["poblacion"] = 1  # peso uniforme — ver methodology.md

    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    gdf = gdf.to_crs(CRS_WGS84)

    keep = ["ubigeo", "codigo", "nombre", "categoria", "poblacion", geom_col]
    gdf = gdf[[c for c in keep if c in gdf.columns]].copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    log.info(f"CCPP final: {len(gdf):,} centros poblados con ubigeo")
    return gdf.reset_index(drop=True)


def load_distritos() -> gpd.GeoDataFrame:
    """DISTRITOS shapefile (repo del curso) — 1,873 polígonos distritales."""
    gdf = gpd.read_file(DATA_RAW / "DISTRITOS.shp")
    geom_col = gdf.geometry.name
    gdf = gdf.rename(columns={
        "IDDIST": "ubigeo",
        "DISTRITO": "distrito",
        "PROVINCIA": "provincia",
        "DEPARTAMEN": "departamento",
    })

    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    gdf = gdf.to_crs(CRS_WGS84)

    gdf["ubigeo"] = normalize_ubigeo(gdf["ubigeo"])

    keep = ["ubigeo", "distrito", "provincia", "departamento", geom_col]
    gdf = gdf[[c for c in keep if c in gdf.columns]].copy()
    log.info(f"DISTRITOS final: {len(gdf):,} distritos")
    return gdf.reset_index(drop=True)


# --- Persistencia ---------------------------------------------------------------

def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log.info(f"wrote {path.name} ({len(df):,} rows)")


def main(force_download: bool = False) -> None:
    log.info("=== T1: INGEST & CLEAN ===")
    download_raw(force=force_download)

    ipress = load_ipress()
    emergencies = load_emergencies()
    ccpp = load_centros_poblados()
    distritos = load_distritos()

    _to_parquet(ipress, DATA_PROCESSED / "ipress.parquet")
    _to_parquet(emergencies, DATA_PROCESSED / "emergencies.parquet")
    _to_parquet(ccpp, DATA_PROCESSED / "centros_poblados.parquet")
    _to_parquet(distritos, DATA_PROCESSED / "distritos.parquet")

    log.info("=== T1 complete ===")


if __name__ == "__main__":
    main()
