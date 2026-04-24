"""T3 — Métricas distritales: baseline + especificación alternativa + índice compuesto + sensibilidad.

Construye dos conjuntos de métricas para cada distrito:

  * BASELINE: densidad por km², umbral de acceso 30 km, actividad log.
  * ALTERNATIVA: oferta por centro poblado, umbral estricto 15 km, actividad log.

Ambas se combinan en un índice de subatención via z-scores invertidos (+mayor
= más desatendido). La sensibilidad se reporta como correlación de Spearman
entre rankings + tabla de diferencias de posición por distrito.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import CRS_PERU_UTM, DATA_PROCESSED, TABLES
from src.utils import get_logger

log = get_logger(__name__)

ACCESS_THRESHOLD_KM_BASELINE = 30.0
ACCESS_THRESHOLD_KM_ALT = 15.0


# --- Helpers ------------------------------------------------------------------

def _area_km2(gdf: gpd.GeoDataFrame) -> pd.Series:
    return gdf.to_crs(CRS_PERU_UTM).geometry.area / 1e6


def _zscore_safe(x: pd.Series) -> pd.Series:
    x = pd.Series(x, dtype="float64")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - mu) / sd).fillna(0)


def _cp_access_share(ccpp_with_dist: gpd.GeoDataFrame, threshold_km: float, col_name: str) -> pd.DataFrame:
    df = ccpp_with_dist[["ubigeo", "dist_to_emergency_km"]].copy()
    df["within"] = df["dist_to_emergency_km"] <= threshold_km
    out = df.groupby("ubigeo", dropna=True, as_index=False)["within"].mean()
    return out.rename(columns={"within": col_name})


# --- Baseline -----------------------------------------------------------------

def compute_baseline(
    district_integrated: gpd.GeoDataFrame, ccpp_with_dist: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    out = district_integrated.copy()
    out["area_km2"] = _area_km2(out)

    # Supply densities per 100 km²
    safe_area = out["area_km2"].replace(0, np.nan)
    out["n_facilities_per_100km2"] = 100 * out["n_facilities"] / safe_area
    out["n_emergency_per_100km2"] = 100 * out["n_emergency_facilities"] / safe_area

    # Access share (30 km baseline)
    access = _cp_access_share(ccpp_with_dist, ACCESS_THRESHOLD_KM_BASELINE, "share_cp_within_30km")
    out = out.merge(access, on="ubigeo", how="left")

    # Activity (log)
    out["log_atenciones"] = np.log1p(out["total_atenciones"].fillna(0))
    return out


def add_composite_underservice(baseline: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Índice compuesto: mayor valor = más desatendido.
    Promedio de z-scores invertidos (mult por -1) en 3 dimensiones."""
    df = baseline.copy()
    z_supply = -_zscore_safe(df["n_emergency_per_100km2"])
    z_activity = -_zscore_safe(df["log_atenciones"])
    z_access = -_zscore_safe(df["share_cp_within_30km"])
    df["z_supply_baseline"] = z_supply
    df["z_activity_baseline"] = z_activity
    df["z_access_baseline"] = z_access
    df["underservice_index_baseline"] = (z_supply + z_activity + z_access) / 3
    return df


# --- Alternativa --------------------------------------------------------------

def compute_alternative(
    district_integrated: gpd.GeoDataFrame, ccpp_with_dist: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    out = district_integrated.copy()

    # Supply per CCPP (no por área): más sensible a distritos rurales con
    # muchos CPs pequeños vs urbanos densos de pocos CPs.
    safe_ccpp = out["n_ccpp"].replace(0, np.nan)
    out["n_facilities_per_ccpp"] = out["n_facilities"] / safe_ccpp
    out["n_emergency_per_ccpp"] = out["n_emergency_facilities"] / safe_ccpp

    # Access share (15 km — umbral más exigente)
    access = _cp_access_share(ccpp_with_dist, ACCESS_THRESHOLD_KM_ALT, "share_cp_within_15km")
    out = out.merge(access, on="ubigeo", how="left")

    out["log_atenciones"] = np.log1p(out["total_atenciones"].fillna(0))

    # Índice compuesto alternativo
    z_supply = -_zscore_safe(out["n_emergency_per_ccpp"])
    z_activity = -_zscore_safe(out["log_atenciones"])
    z_access = -_zscore_safe(out["share_cp_within_15km"])
    out["z_supply_alt"] = z_supply
    out["z_activity_alt"] = z_activity
    out["z_access_alt"] = z_access
    out["underservice_index_alt"] = (z_supply + z_activity + z_access) / 3
    return out


# --- Sensibilidad --------------------------------------------------------------

def sensitivity_table(
    baseline: gpd.GeoDataFrame, alternative: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, float]:
    b = baseline[["ubigeo", "departamento", "provincia", "distrito", "underservice_index_baseline"]].copy()
    a = alternative[["ubigeo", "underservice_index_alt"]].copy()
    joined = b.merge(a, on="ubigeo", how="inner").dropna(
        subset=["underservice_index_baseline", "underservice_index_alt"]
    )
    joined["rank_baseline"] = joined["underservice_index_baseline"].rank(ascending=False, method="min")
    joined["rank_alternative"] = joined["underservice_index_alt"].rank(ascending=False, method="min")
    joined["rank_diff"] = (joined["rank_alternative"] - joined["rank_baseline"]).astype(int)
    joined = joined.sort_values("rank_baseline").reset_index(drop=True)

    rho, _ = spearmanr(joined["rank_baseline"], joined["rank_alternative"])
    return joined, float(rho)


# --- Main ----------------------------------------------------------------------

def main() -> None:
    log.info("=== T3: DISTRICT METRICS ===")
    district = gpd.read_parquet(DATA_PROCESSED / "district_integrated.parquet")
    ccpp = gpd.read_parquet(DATA_PROCESSED / "ccpp_with_distance.parquet")

    baseline = compute_baseline(district, ccpp)
    baseline = add_composite_underservice(baseline)
    alternative = compute_alternative(district, ccpp)

    sensitivity, rho = sensitivity_table(baseline, alternative)
    log.info(f"Spearman(ranking baseline, ranking alternativa) = {rho:.4f}")

    # Guardar
    TABLES.mkdir(parents=True, exist_ok=True)
    baseline_cols = [c for c in baseline.columns if c != "geometry"]
    alt_cols = [c for c in alternative.columns if c != "geometry"]
    baseline[baseline_cols].to_csv(TABLES / "district_metrics_baseline.csv", index=False)
    alternative[alt_cols].to_csv(TABLES / "district_metrics_alternative.csv", index=False)
    sensitivity.to_csv(TABLES / "sensitivity_ranking.csv", index=False)

    # Parquet combinado (para T4, T5, T6)
    overlap = [c for c in alternative.columns if c in baseline.columns and c not in ("ubigeo", "geometry")]
    combined = baseline.merge(
        alternative.drop(columns=overlap + (["geometry"] if "geometry" in alternative.columns else [])),
        on="ubigeo",
        how="left",
    )
    combined.to_parquet(DATA_PROCESSED / "district_metrics.parquet")
    log.info(f"wrote district_metrics.parquet ({len(combined):,} filas × {len(combined.columns)} cols)")

    # Resumen ejecutivo
    log.info("TOP 10 más subatendidos (baseline):")
    top = baseline.nlargest(10, "underservice_index_baseline")[
        ["departamento", "provincia", "distrito", "underservice_index_baseline",
         "n_emergency_facilities", "share_cp_within_30km"]
    ]
    print(top.to_string(index=False))

    log.info("BOTTOM 10 mejor atendidos (baseline):")
    bottom = baseline.nsmallest(10, "underservice_index_baseline")[
        ["departamento", "provincia", "distrito", "underservice_index_baseline",
         "n_emergency_facilities", "share_cp_within_30km"]
    ]
    print(bottom.to_string(index=False))

    # Movers más fuertes entre baseline y alternativa
    top_movers = sensitivity.reindex(sensitivity["rank_diff"].abs().sort_values(ascending=False).index).head(10)
    log.info("Distritos con mayor cambio de ranking (|baseline - alternativa|):")
    print(top_movers[
        ["departamento", "provincia", "distrito", "rank_baseline", "rank_alternative", "rank_diff"]
    ].to_string(index=False))

    log.info("=== T3 complete ===")


if __name__ == "__main__":
    main()
