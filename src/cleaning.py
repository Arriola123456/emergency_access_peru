"""T1.b — Limpieza, normalización y persistencia de los 4 datasets.

Lee los archivos crudos vía `src.data_loader`, aplica la limpieza
documentada en `docs/methodology.md` y `docs/data_dictionary.md`, y
persiste los resultados a `data/processed/` como parquet/geoparquet.

Ejecutado con `python -m src.cleaning` corre el pipeline completo de T1
(descarga si no está cacheada → lectura → limpieza → persistencia).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import CRS_WGS84, DATA_PROCESSED
from src.data_loader import (
    download_raw,
    read_centros_poblados_raw,
    read_distritos_raw,
    read_emergencies_raw,
    read_ipress_raw,
)
from src.utils import first_existing, get_logger, normalize_ubigeo

log = get_logger(__name__)

# Rango de coordenadas válido para Perú continental + margen amazónico
PERU_LAT = (-18.5, 0.5)
PERU_LON = (-81.5, -68.0)


# --- Helper internos -------------------------------------------------------

def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    def clean(c: str) -> str:
        c = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode("ascii")
        return c.strip().upper().replace(" ", "_").replace(".", "_")

    out = df.copy()
    out.columns = [clean(c) for c in out.columns]
    return out


# --- Limpiezas por dataset -------------------------------------------------

def clean_ipress(raw: pd.DataFrame) -> pd.DataFrame:
    """IPRESS MINSA — fix del bug NORTE/ESTE invertido + filtros + dedup."""
    df = _normalize_colnames(raw)
    log.info(f"IPRESS raw: {len(df):,} filas × {len(df.columns)} cols")

    df = df.rename(columns={
        "CODIGO_UNICO": "codigo",
        "NOMBRE_DEL_ESTABLECIMIENTO": "nombre",
        "UBIGEO": "ubigeo",
        "INSTITUCION": "institucion",
        "CLASIFICACION": "clasificacion",
        "CATEGORIA": "categoria",
        "NORTE": "lon",  # <-- NORTE contiene longitud en el CSV fuente
        "ESTE": "lat",   # <-- ESTE contiene latitud en el CSV fuente
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


def clean_emergencies(raw: pd.DataFrame) -> pd.DataFrame:
    """SUSALUD 2024 — parsea NE_xxxx como NaN, suma anual por (codigo, ubigeo)."""
    df = _normalize_colnames(raw)
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

    agg = (
        df.assign(
            atenciones=df["atenciones"].fillna(0),
            atendidos=df["atendidos"].fillna(0),
        )
        .groupby(["codigo", "ubigeo"], dropna=False, as_index=False)[["atenciones", "atendidos"]]
        .sum()
    )
    log.info(
        f"EMERGENCIES aggregated: {len(agg):,} IPRESS-year; "
        f"total atenciones: {agg['atenciones'].sum():,.0f}"
    )
    return agg


def clean_centros_poblados(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Centros Poblados — deriva UBIGEO desde CÓDIGO[:6], peso uniforme = 1."""
    gdf = raw.copy()
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


def clean_distritos(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """DISTRITOS — UBIGEO desde IDDIST + reproyección a EPSG:4326."""
    gdf = raw.copy()
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


# --- Persistencia y orquestación -------------------------------------------

def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log.info(f"wrote {path.name} ({len(df):,} rows)")


def main(force_download: bool = False) -> None:
    """Pipeline completo de T1: descarga + lectura + limpieza + persistencia."""
    log.info("=== T1: INGEST & CLEAN ===")
    download_raw(force=force_download)

    ipress = clean_ipress(read_ipress_raw())
    emergencies = clean_emergencies(read_emergencies_raw())
    ccpp = clean_centros_poblados(read_centros_poblados_raw())
    distritos = clean_distritos(read_distritos_raw())

    _to_parquet(ipress, DATA_PROCESSED / "ipress.parquet")
    _to_parquet(emergencies, DATA_PROCESSED / "emergencies.parquet")
    _to_parquet(ccpp, DATA_PROCESSED / "centros_poblados.parquet")
    _to_parquet(distritos, DATA_PROCESSED / "distritos.parquet")

    log.info("=== T1 complete ===")


if __name__ == "__main__":
    main()
