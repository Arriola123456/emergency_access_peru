"""T4 — Visualizaciones estáticas con matplotlib/seaborn."""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATA_PROCESSED, FIGURES
from src.utils import get_logger

log = get_logger(__name__)

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out.name}")


# --- Plots --------------------------------------------------------------------

def plot_top_bottom_underservice(metrics: gpd.GeoDataFrame, n: int = 20) -> None:
    df = metrics.dropna(subset=["underservice_index_baseline"]).copy()
    df["label"] = df["distrito"] + " (" + df["departamento"].str[:3] + ")"

    top = df.nlargest(n, "underservice_index_baseline").sort_values("underservice_index_baseline")
    bot = df.nsmallest(n, "underservice_index_baseline").sort_values("underservice_index_baseline")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    sns.barplot(data=top, y="label", x="underservice_index_baseline", color="#c0392b", ax=axes[0])
    axes[0].set_title(f"Top {n} distritos más subatendidos (baseline)")
    axes[0].set_xlabel("Índice compuesto de subatención (z-score, mayor = peor)")
    axes[0].set_ylabel("")

    sns.barplot(data=bot, y="label", x="underservice_index_baseline", color="#27ae60", ax=axes[1])
    axes[1].set_title(f"Bottom {n} distritos mejor atendidos (baseline)")
    axes[1].set_xlabel("Índice compuesto de subatención")
    axes[1].set_ylabel("")
    fig.suptitle("Distritos en los extremos del índice de subatención", fontsize=14, y=1.02)
    _save(fig, "01_top_bottom_underservice.png")


def plot_scatter_facilities_vs_atenciones(metrics: gpd.GeoDataFrame) -> None:
    df = metrics[metrics["n_facilities"] > 0].copy()
    df["atenciones_k"] = df["total_atenciones"] / 1000.0

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="n_facilities",
        y="atenciones_k",
        hue="departamento",
        size="n_ccpp",
        sizes=(20, 200),
        alpha=0.7,
        legend=False,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("Nº de establecimientos de salud (log)")
    ax.set_ylabel("Atenciones de emergencia 2024 (miles, symlog)")
    ax.set_title("Oferta vs actividad de emergencias por distrito")
    _save(fig, "02_scatter_facilities_vs_atenciones.png")


def plot_correlation_heatmap(metrics: gpd.GeoDataFrame) -> None:
    cols = [
        "n_facilities",
        "n_emergency_facilities",
        "total_atenciones",
        "mean_dist_km",
        "share_cp_within_30km",
        "share_cp_within_15km",
        "n_facilities_per_100km2",
        "n_emergency_per_ccpp",
        "underservice_index_baseline",
        "underservice_index_alt",
    ]
    cols = [c for c in cols if c in metrics.columns]
    corr = metrics[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlación Spearman entre indicadores distritales")
    _save(fig, "03_correlation_heatmap.png")


def plot_distance_histogram(ccpp_with_dist: gpd.GeoDataFrame) -> None:
    d = ccpp_with_dist["dist_to_emergency_km"].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(d, bins=60, color="#2980b9", kde=True, ax=ax)
    ax.axvline(15, color="#e67e22", linestyle="--", label="umbral alternativa (15 km)")
    ax.axvline(30, color="#c0392b", linestyle="--", label="umbral baseline (30 km)")
    ax.set_xlim(0, d.quantile(0.99))
    ax.set_xlabel("Distancia al IPRESS de emergencia más cercano (km)")
    ax.set_ylabel("Nº de centros poblados")
    ax.set_title(
        f"Distribución de distancias CP → IPRESS emergencia\n"
        f"(n={len(d):,}; mediana {d.median():.1f} km; media {d.mean():.1f} km)"
    )
    ax.legend()
    _save(fig, "04_distance_histogram.png")


def plot_underservice_by_departamento(metrics: gpd.GeoDataFrame) -> None:
    df = metrics.dropna(subset=["underservice_index_baseline", "departamento"]).copy()
    # Ordenar departamentos por mediana de subatención (peor arriba)
    order = (
        df.groupby("departamento")["underservice_index_baseline"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.boxplot(data=df, y="departamento", x="underservice_index_baseline", order=order, color="#3498db", ax=ax)
    sns.stripplot(data=df, y="departamento", x="underservice_index_baseline", order=order,
                  color="black", size=1.5, alpha=0.4, ax=ax)
    ax.set_xlabel("Índice compuesto de subatención (baseline)")
    ax.set_ylabel("")
    ax.set_title("Distribución de subatención por departamento")
    _save(fig, "05_underservice_by_departamento.png")


def plot_baseline_vs_alternative(metrics: gpd.GeoDataFrame) -> None:
    df = metrics.dropna(subset=["underservice_index_baseline", "underservice_index_alt"])
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.scatterplot(data=df, x="underservice_index_baseline", y="underservice_index_alt",
                    alpha=0.35, s=15, ax=ax)
    lim_min = min(df["underservice_index_baseline"].min(), df["underservice_index_alt"].min())
    lim_max = max(df["underservice_index_baseline"].max(), df["underservice_index_alt"].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="red", linestyle="--", label="y = x")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(df["underservice_index_baseline"], df["underservice_index_alt"])
    ax.set_xlabel("Subatención BASELINE (30 km, densidad/km²)")
    ax.set_ylabel("Subatención ALTERNATIVA (15 km, oferta/CP)")
    ax.set_title(f"Baseline vs Alternativa — Spearman ρ = {rho:.3f}")
    ax.legend()
    _save(fig, "06_baseline_vs_alternative.png")


# --- Main ----------------------------------------------------------------------

def main() -> None:
    log.info("=== T4: STATIC VIZ ===")
    metrics = gpd.read_parquet(DATA_PROCESSED / "district_metrics.parquet")
    ccpp = gpd.read_parquet(DATA_PROCESSED / "ccpp_with_distance.parquet")

    plot_top_bottom_underservice(metrics, n=20)
    plot_scatter_facilities_vs_atenciones(metrics)
    plot_correlation_heatmap(metrics)
    plot_distance_histogram(ccpp)
    plot_underservice_by_departamento(metrics)
    plot_baseline_vs_alternative(metrics)

    log.info("=== T4 complete ===")


if __name__ == "__main__":
    main()
