"""T2 — Integración geoespacial.

Lee los parquet limpios de T1, construye GeoDataFrames, reproyecta a UTM 18S
para cálculos de distancia en metros, hace un spatial join IPRESS↔distritos
para validar ubigeos, y calcula la distancia desde cada centro poblado al
IPRESS de emergencia más cercano vía KDTree. Finalmente agrega todas las
métricas a nivel distrito.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import CRS_PERU_UTM, CRS_WGS84, DATA_PROCESSED
from src.utils import get_logger

log = get_logger(__name__)


# --- Carga ---------------------------------------------------------------------

def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    ipress = pd.read_parquet(DATA_PROCESSED / "ipress.parquet")
    emergencies = pd.read_parquet(DATA_PROCESSED / "emergencies.parquet")
    ccpp = gpd.read_parquet(DATA_PROCESSED / "centros_poblados.parquet")
    distritos = gpd.read_parquet(DATA_PROCESSED / "distritos.parquet")
    return ipress, emergencies, ccpp, distritos


# --- Construcción GeoDataFrame IPRESS -----------------------------------------

def ipress_to_gdf(ipress_df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        ipress_df.copy(),
        geometry=gpd.points_from_xy(ipress_df["lon"], ipress_df["lat"]),
        crs=CRS_WGS84,
    )
    return gdf


# --- Merge de actividad de emergencias ----------------------------------------

def attach_emergency_activity(
    ipress_gdf: gpd.GeoDataFrame, emergencies_df: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Une la actividad de emergencias a IPRESS. Flag is_emergency = actividad>0."""
    # El join se hace por codigo; limpiamos a string stripped para evitar
    # mismatches por padding (IPRESS tiene '00016618', SUSALUD tiene '16618' o viceversa)
    ipress = ipress_gdf.copy()
    ipress["codigo_key"] = ipress["codigo"].astype("string").str.lstrip("0")
    em = emergencies_df.copy()
    em["codigo_key"] = em["codigo"].astype("string").str.lstrip("0")
    em_agg = em.groupby("codigo_key", as_index=False)["atenciones"].sum()

    merged = ipress.merge(em_agg, on="codigo_key", how="left")
    merged["atenciones"] = merged["atenciones"].fillna(0).astype(int)
    merged["is_emergency"] = merged["atenciones"] > 0
    return merged.drop(columns=["codigo_key"])


# --- Spatial join para validar UBIGEO -----------------------------------------

def validate_ubigeo_via_sjoin(
    ipress_gdf: gpd.GeoDataFrame, distritos_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Asigna el UBIGEO del distrito en que cae geométricamente cada IPRESS.
    Cuando difiere del UBIGEO declarado, se usa el derivado de geometría."""
    right = distritos_gdf[["ubigeo", "geometry"]].rename(columns={"ubigeo": "ubigeo_geom"})
    joined = gpd.sjoin(ipress_gdf, right, how="left", predicate="within").drop(columns=["index_right"])
    # sjoin puede devolver duplicados si geometría coincide con múltiples polígonos (no debería
    # pasar con distritos disjoint, pero por seguridad mantenemos el primero)
    joined = joined.loc[~joined.index.duplicated(keep="first")]
    joined["ubigeo_final"] = joined["ubigeo_geom"].fillna(joined["ubigeo"])
    return joined


# --- Distancia CP → IPRESS de emergencia más cercano ---------------------------

def nearest_emergency_distance(
    ccpp_gdf: gpd.GeoDataFrame, ipress_gdf: gpd.GeoDataFrame, emergency_only: bool = True
) -> gpd.GeoDataFrame:
    """Distancia en km al IPRESS (de emergencia si emergency_only=True) más cercano.
    Usa UTM 18S para distancias euclídeas en metros."""
    pool = ipress_gdf[ipress_gdf["is_emergency"]] if emergency_only else ipress_gdf
    if len(pool) == 0:
        raise RuntimeError("No IPRESS in pool for nearest computation")

    pool_utm = pool.to_crs(CRS_PERU_UTM)
    ccpp_utm = ccpp_gdf.to_crs(CRS_PERU_UTM)

    tree_coords = np.column_stack([pool_utm.geometry.x.values, pool_utm.geometry.y.values])
    query_coords = np.column_stack([ccpp_utm.geometry.x.values, ccpp_utm.geometry.y.values])
    tree = cKDTree(tree_coords)
    dists_m, idx = tree.query(query_coords, k=1)

    out = ccpp_gdf.copy()
    out["dist_to_emergency_km"] = dists_m / 1000.0
    out["nearest_emergency_codigo"] = pool_utm["codigo"].values[idx]
    return out


# --- Agregación a nivel distrito -----------------------------------------------

def district_aggregates(
    ipress_gdf: gpd.GeoDataFrame,
    ccpp_with_dist: gpd.GeoDataFrame,
    distritos_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    # métricas IPRESS
    ipress_key = "ubigeo_final" if "ubigeo_final" in ipress_gdf.columns else "ubigeo"
    ipress_metrics = (
        ipress_gdf.groupby(ipress_key, dropna=True)
        .agg(
            n_facilities=("codigo", "count"),
            n_emergency_facilities=("is_emergency", "sum"),
            total_atenciones=("atenciones", "sum"),
        )
        .reset_index()
        .rename(columns={ipress_key: "ubigeo"})
    )

    # métricas desde centros poblados
    ccpp_metrics = (
        ccpp_with_dist.groupby("ubigeo", dropna=True)
        .agg(
            n_ccpp=("codigo", "count"),
            mean_dist_km=("dist_to_emergency_km", "mean"),
            median_dist_km=("dist_to_emergency_km", "median"),
            max_dist_km=("dist_to_emergency_km", "max"),
        )
        .reset_index()
    )

    out = distritos_gdf.merge(ipress_metrics, on="ubigeo", how="left")
    out = out.merge(ccpp_metrics, on="ubigeo", how="left")

    # rellenar ceros donde no hay IPRESS / CP
    for c in ["n_facilities", "n_emergency_facilities", "total_atenciones", "n_ccpp"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    return out


# --- Main ----------------------------------------------------------------------

def main() -> None:
    log.info("=== T2: GEO INTEGRATION ===")
    ipress_df, emergencies_df, ccpp_gdf, distritos_gdf = load_processed()
    log.info(
        f"loaded: ipress={len(ipress_df):,}, emergencies={len(emergencies_df):,}, "
        f"ccpp={len(ccpp_gdf):,}, distritos={len(distritos_gdf):,}"
    )

    ipress_gdf = ipress_to_gdf(ipress_df)
    ipress_gdf = attach_emergency_activity(ipress_gdf, emergencies_df)
    log.info(
        f"IPRESS con actividad emergencia: {int(ipress_gdf['is_emergency'].sum()):,}  "
        f"(total atenciones: {int(ipress_gdf['atenciones'].sum()):,})"
    )

    ipress_gdf = validate_ubigeo_via_sjoin(ipress_gdf, distritos_gdf)
    n_differ = int((ipress_gdf["ubigeo_geom"].notna() & (ipress_gdf["ubigeo_geom"] != ipress_gdf["ubigeo"])).sum())
    n_unmatched = int(ipress_gdf["ubigeo_geom"].isna().sum())
    log.info(f"sjoin: {n_differ:,} IPRESS con ubigeo reasignado por geometría; {n_unmatched:,} fuera de polígonos")

    ccpp_with_dist = nearest_emergency_distance(ccpp_gdf, ipress_gdf, emergency_only=True)
    log.info(
        f"CP → IPRESS emergencia: mean={ccpp_with_dist['dist_to_emergency_km'].mean():.1f} km, "
        f"median={ccpp_with_dist['dist_to_emergency_km'].median():.1f} km, "
        f"max={ccpp_with_dist['dist_to_emergency_km'].max():.1f} km"
    )

    district_df = district_aggregates(ipress_gdf, ccpp_with_dist, distritos_gdf)
    no_facilities = int((district_df["n_facilities"] == 0).sum())
    no_emergency = int((district_df["n_emergency_facilities"] == 0).sum())
    log.info(f"distritos sin IPRESS: {no_facilities:,}; sin IPRESS de emergencia: {no_emergency:,}")

    ipress_gdf.to_parquet(DATA_PROCESSED / "ipress_geo.parquet")
    ccpp_with_dist.to_parquet(DATA_PROCESSED / "ccpp_with_distance.parquet")
    district_df.to_parquet(DATA_PROCESSED / "district_integrated.parquet")
    log.info(
        f"wrote ipress_geo ({len(ipress_gdf):,}), ccpp_with_distance ({len(ccpp_with_dist):,}), "
        f"district_integrated ({len(district_df):,})"
    )
    log.info("=== T2 complete ===")


if __name__ == "__main__":
    main()
