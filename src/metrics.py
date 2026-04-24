"""T3 — Índice de cobertura de emergencia en escala [0, 1].

Construye dos especificaciones del mismo índice:

  * BASELINE: oferta por km², umbral de acceso 30 km.
  * ALTERNATIVA: oferta por centro poblado, umbral estricto 15 km.

Cada componente se normaliza a [0, 1] (min-max con log-transform para las
dimensiones sesgadas) y se promedia. Resultado: un único escalar por distrito
donde **1 = mejor atendido** y **0 = peor atendido**. La sensibilidad se
reporta como Spearman entre los dos rankings + tabla de diferencias por
distrito.
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


def _min_max_01(x: pd.Series) -> pd.Series:
    """Escala una serie a [0, 1] con min-max. Devuelve 0 si es constante."""
    x = pd.Series(x, dtype="float64")
    lo = x.min(skipna=True)
    hi = x.max(skipna=True)
    if not np.isfinite(hi - lo) or (hi - lo) == 0:
        return pd.Series(0.0, index=x.index)
    out = (x - lo) / (hi - lo)
    return out.clip(lower=0.0, upper=1.0).fillna(0.0)


def _cp_access_share(
    ccpp_with_dist: gpd.GeoDataFrame, threshold_km: float, col_name: str
) -> pd.DataFrame:
    df = ccpp_with_dist[["ubigeo", "dist_to_emergency_km"]].copy()
    df["within"] = df["dist_to_emergency_km"] <= threshold_km
    out = df.groupby("ubigeo", dropna=True, as_index=False)["within"].mean()
    return out.rename(columns={"within": col_name})


# --- Baseline -----------------------------------------------------------------

def compute_baseline(
    district_integrated: gpd.GeoDataFrame, ccpp_with_dist: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Baseline: oferta por km², acceso 30 km, actividad log."""
    out = district_integrated.copy()
    out["area_km2"] = _area_km2(out)

    safe_area = out["area_km2"].replace(0, np.nan)
    out["n_facilities_per_100km2"] = 100 * out["n_facilities"] / safe_area
    out["n_emergency_per_100km2"] = 100 * out["n_emergency_facilities"] / safe_area

    access = _cp_access_share(ccpp_with_dist, ACCESS_THRESHOLD_KM_BASELINE, "share_cp_within_30km")
    out = out.merge(access, on="ubigeo", how="left")

    out["log_atenciones"] = np.log1p(out["total_atenciones"].fillna(0))

    # Componentes [0, 1] — mayor = mejor atendido
    out["supply_01_baseline"] = _min_max_01(
        np.log1p(out["n_emergency_per_100km2"].fillna(0))
    )
    out["activity_01"] = _min_max_01(np.log1p(out["total_atenciones"].fillna(0)))
    out["access_01_baseline"] = out["share_cp_within_30km"].fillna(0).clip(0, 1)

    out["coverage_index_baseline"] = (
        out["supply_01_baseline"]
        + out["activity_01"]
        + out["access_01_baseline"]
    ) / 3
    return out


# --- Alternativa --------------------------------------------------------------

def compute_alternative(
    district_integrated: gpd.GeoDataFrame, ccpp_with_dist: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Alternativa: oferta por CP, acceso 15 km, actividad log."""
    out = district_integrated.copy()

    safe_ccpp = out["n_ccpp"].replace(0, np.nan)
    out["n_facilities_per_ccpp"] = out["n_facilities"] / safe_ccpp
    out["n_emergency_per_ccpp"] = out["n_emergency_facilities"] / safe_ccpp

    access = _cp_access_share(ccpp_with_dist, ACCESS_THRESHOLD_KM_ALT, "share_cp_within_15km")
    out = out.merge(access, on="ubigeo", how="left")

    out["log_atenciones"] = np.log1p(out["total_atenciones"].fillna(0))

    # Componentes [0, 1] — mismas reglas que baseline
    out["supply_01_alt"] = _min_max_01(
        np.log1p(out["n_emergency_per_ccpp"].fillna(0))
    )
    out["activity_01_alt"] = _min_max_01(np.log1p(out["total_atenciones"].fillna(0)))
    out["access_01_alt"] = out["share_cp_within_15km"].fillna(0).clip(0, 1)

    out["coverage_index_alt"] = (
        out["supply_01_alt"] + out["activity_01_alt"] + out["access_01_alt"]
    ) / 3
    return out


# --- Sensibilidad --------------------------------------------------------------

def sensitivity_table(
    baseline: gpd.GeoDataFrame, alternative: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, float]:
    b = baseline[
        ["ubigeo", "departamento", "provincia", "distrito", "coverage_index_baseline"]
    ].copy()
    a = alternative[["ubigeo", "coverage_index_alt"]].copy()
    joined = b.merge(a, on="ubigeo", how="inner").dropna(
        subset=["coverage_index_baseline", "coverage_index_alt"]
    )
    # rank ascending: rank 1 = menor cobertura = peor atendido
    joined["rank_baseline"] = joined["coverage_index_baseline"].rank(
        ascending=True, method="min"
    )
    joined["rank_alternative"] = joined["coverage_index_alt"].rank(
        ascending=True, method="min"
    )
    joined["rank_diff"] = (joined["rank_alternative"] - joined["rank_baseline"]).astype(int)
    joined = joined.sort_values("rank_baseline").reset_index(drop=True)

    rho, _ = spearmanr(joined["rank_baseline"], joined["rank_alternative"])
    return joined, float(rho)


# --- Main ----------------------------------------------------------------------

def main() -> None:
    log.info("=== T3: COVERAGE INDEX [0, 1] ===")
    district = gpd.read_parquet(DATA_PROCESSED / "district_integrated.parquet")
    ccpp = gpd.read_parquet(DATA_PROCESSED / "ccpp_with_distance.parquet")

    baseline = compute_baseline(district, ccpp)
    alternative = compute_alternative(district, ccpp)

    sensitivity, rho = sensitivity_table(baseline, alternative)
    log.info(f"Spearman(ranking baseline, ranking alternativa) = {rho:.4f}")

    # Sanity check: ambos índices en [0, 1]
    for c in ("coverage_index_baseline", "coverage_index_alt"):
        df = baseline if c == "coverage_index_baseline" else alternative
        v = df[c].dropna()
        assert v.between(0.0, 1.0).all(), (
            f"{c} fuera de rango [0, 1]: min={v.min()}, max={v.max()}"
        )
    log.info("Sanity check OK — ambos índices en [0, 1]")

    TABLES.mkdir(parents=True, exist_ok=True)
    baseline_cols = [c for c in baseline.columns if c != "geometry"]
    alt_cols = [c for c in alternative.columns if c != "geometry"]
    baseline[baseline_cols].to_csv(TABLES / "district_metrics_baseline.csv", index=False)
    alternative[alt_cols].to_csv(TABLES / "district_metrics_alternative.csv", index=False)
    sensitivity.to_csv(TABLES / "sensitivity_ranking.csv", index=False)

    overlap = [c for c in alternative.columns if c in baseline.columns and c not in ("ubigeo", "geometry")]
    combined = baseline.merge(
        alternative.drop(columns=overlap + (["geometry"] if "geometry" in alternative.columns else [])),
        on="ubigeo",
        how="left",
    )
    combined.to_parquet(DATA_PROCESSED / "district_metrics.parquet")
    log.info(f"wrote district_metrics.parquet ({len(combined):,} filas × {len(combined.columns)} cols)")

    # Resumen ejecutivo — ahora rank 1 = PEOR atendido, última fila = MEJOR
    log.info("TOP 10 PEOR atendidos (menor coverage_index_baseline):")
    worst = (
        baseline.sort_values(["coverage_index_baseline", "n_ccpp"], ascending=[True, False])
        .head(10)[[
            "departamento", "provincia", "distrito", "coverage_index_baseline",
            "supply_01_baseline", "activity_01", "access_01_baseline",
        ]]
    )
    print(worst.to_string(index=False))

    log.info("TOP 10 MEJOR atendidos (mayor coverage_index_baseline):")
    best = baseline.nlargest(10, "coverage_index_baseline")[[
        "departamento", "provincia", "distrito", "coverage_index_baseline",
        "supply_01_baseline", "activity_01", "access_01_baseline",
    ]]
    print(best.to_string(index=False))

    log.info("=== T3 complete ===")


if __name__ == "__main__":
    main()
