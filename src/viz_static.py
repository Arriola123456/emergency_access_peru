"""T4 — Visualizaciones estáticas con matplotlib/seaborn.

Seis gráficos diseñados para leerse sin conocer el código:
  1. Diverging barplot de extremos del ranking (20 peores / 20 mejores)
  2. Scatter log-log oferta vs actividad (con anotaciones y tamaño = CPs)
  3. Heatmap de correlaciones Spearman con etiquetas en lenguaje natural
  4. Histograma de distancias CP→emergencia con umbrales marcados
  5. Boxplot + strip de subatención por departamento
  6. Scatter baseline vs alternativa con línea y = x y ρ de Spearman

Todas las etiquetas (ejes, títulos, leyendas) están en lenguaje natural.
Las leyendas se ubican fuera del área de trazado para evitar superposición.
"""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from src.config import DATA_PROCESSED, FIGURES
from src.utils import get_logger

log = get_logger(__name__)

sns.set_theme(style="whitegrid", context="notebook")

# Diccionario único de traducción var_nombre → etiqueta legible
LABEL_MAP: dict[str, str] = {
    "n_facilities": "N° de establecimientos (IPRESS)",
    "n_emergency_facilities": "N° con emergencia activa",
    "total_atenciones": "Atenciones de emergencia 2024",
    "mean_dist_km": "Distancia media CP→emergencia (km)",
    "median_dist_km": "Distancia mediana CP→emergencia (km)",
    "max_dist_km": "Distancia máxima CP→emergencia (km)",
    "share_cp_within_30km": "Acceso ≤ 30 km (share de CPs)",
    "share_cp_within_15km": "Acceso ≤ 15 km (share de CPs)",
    "n_facilities_per_100km2": "Densidad (IPRESS / 100 km²)",
    "n_emergency_per_100km2": "Densidad de emergencia (/100 km²)",
    "n_emergency_per_ccpp": "Emergencia por centro poblado",
    "underservice_index_baseline": "Índice subatención — baseline",
    "underservice_index_alt": "Índice subatención — alternativa",
    "log_atenciones": "log(1 + atenciones)",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"wrote {out.name}")


# --- Gráfico 1 · Diverging barplot de extremos del ranking -------------------

def plot_top_bottom_underservice(metrics: gpd.GeoDataFrame, n: int = 20) -> None:
    df = metrics.dropna(subset=["underservice_index_baseline"]).copy()
    df["label"] = (
        df["distrito"].str.title() + " (" + df["departamento"].str[:4].str.title() + ")"
    )

    top = df.nlargest(n, "underservice_index_baseline").copy()
    top["grupo"] = f"{n} más desatendidos"
    bot = df.nsmallest(n, "underservice_index_baseline").copy()
    bot["grupo"] = f"{n} mejor atendidos"

    plot_df = pd.concat([top, bot], ignore_index=True).sort_values(
        "underservice_index_baseline", ascending=False
    )

    palette = {f"{n} más desatendidos": "#c0392b", f"{n} mejor atendidos": "#27ae60"}

    fig, ax = plt.subplots(figsize=(11, 11))
    sns.barplot(
        data=plot_df,
        y="label",
        x="underservice_index_baseline",
        hue="grupo",
        palette=palette,
        dodge=False,
        ax=ax,
    )
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Índice compuesto de subatención (mayor = peor atendido)", fontsize=11)
    ax.set_ylabel("Distrito (departamento abreviado)", fontsize=11)
    ax.set_title(
        f"Extremos del ranking de subatención — {n} peores y {n} mejores distritos",
        fontsize=13,
        pad=14,
    )
    # Leyenda fuera del área de trazado (abajo centrada, dos columnas)
    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, "01_top_bottom_underservice.png")


# --- Gráfico 2 · Scatter oferta vs actividad ---------------------------------

def plot_scatter_facilities_vs_atenciones(metrics: gpd.GeoDataFrame) -> None:
    df = metrics[metrics["n_facilities"] > 0].copy()
    df["atenciones_miles"] = df["total_atenciones"] / 1_000.0

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=df,
        x="n_facilities",
        y="atenciones_miles",
        size="n_ccpp",
        sizes=(15, 300),
        alpha=0.5,
        color="#2980b9",
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Número de establecimientos (IPRESS) por distrito — escala logarítmica", fontsize=11)
    ax.set_ylabel(
        "Atenciones de emergencia en 2024 (miles) — escala symlog",
        fontsize=11,
    )
    ax.set_title(
        "Relación entre oferta de establecimientos y actividad de emergencias\n"
        "(tamaño del punto proporcional al número de centros poblados del distrito)",
        fontsize=12,
        pad=12,
    )

    # Anotaciones para los 5 distritos con más atenciones
    top5 = df.nlargest(5, "total_atenciones")
    for _, row in top5.iterrows():
        ax.annotate(
            row["distrito"].title(),
            (row["n_facilities"], row["atenciones_miles"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#7f8c8d", alpha=0.9, linewidth=0.5),
        )

    # Leyenda fuera del plot (derecha)
    ax.legend(
        title="Centros poblados\nen el distrito",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    fig.tight_layout()
    _save(fig, "02_scatter_facilities_vs_atenciones.png")


# --- Gráfico 3 · Heatmap de correlaciones ------------------------------------

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

    # Reemplazar nombres crudos por etiquetas en lenguaje natural
    pretty = [LABEL_MAP.get(c, c) for c in cols]
    corr.index = pretty
    corr.columns = pretty

    fig, ax = plt.subplots(figsize=(11, 9.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Correlación de Spearman", "shrink": 0.7},
        ax=ax,
    )
    ax.set_title(
        "Correlación Spearman entre los indicadores distritales",
        fontsize=13,
        pad=12,
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    _save(fig, "03_correlation_heatmap.png")


# --- Gráfico 4 · Histograma de distancias ------------------------------------

def plot_distance_histogram(ccpp_with_dist: gpd.GeoDataFrame) -> None:
    d = ccpp_with_dist["dist_to_emergency_km"].dropna()

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(
        d,
        bins=60,
        color="#2980b9",
        alpha=0.75,
        kde=True,
        line_kws={"color": "#1f4e79", "linewidth": 2},
        ax=ax,
    )
    ax.axvline(15, color="#e67e22", linestyle="--", linewidth=2,
               label="Umbral alternativo (15 km)")
    ax.axvline(30, color="#c0392b", linestyle="--", linewidth=2,
               label="Umbral baseline (30 km)")
    ax.set_xlim(0, d.quantile(0.99))
    ax.set_xlabel("Distancia al establecimiento con emergencia más cercano (km)", fontsize=11)
    ax.set_ylabel("Número de centros poblados", fontsize=11)
    ax.set_title(
        "Distribución de la distancia desde centros poblados al IPRESS con emergencia\n"
        f"n = {len(d):,} CPs · mediana {d.median():.1f} km · media {d.mean():.1f} km · "
        f"máximo {d.max():.0f} km",
        fontsize=12,
        pad=12,
    )
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    fig.tight_layout()
    _save(fig, "04_distance_histogram.png")


# --- Gráfico 5 · Boxplot por departamento ------------------------------------

def plot_underservice_by_departamento(metrics: gpd.GeoDataFrame) -> None:
    df = metrics.dropna(subset=["underservice_index_baseline", "departamento"]).copy()
    df["departamento"] = df["departamento"].str.title()

    order = (
        df.groupby("departamento")["underservice_index_baseline"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.boxplot(
        data=df,
        y="departamento",
        x="underservice_index_baseline",
        order=order,
        color="#3498db",
        fliersize=3,
        linewidth=1.2,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        y="departamento",
        x="underservice_index_baseline",
        order=order,
        color="black",
        size=1.8,
        alpha=0.4,
        ax=ax,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.set_xlabel("Índice compuesto de subatención (mayor = peor atendido)", fontsize=11)
    ax.set_ylabel("Departamento", fontsize=11)
    ax.set_title(
        "Distribución distrital del índice de subatención por departamento\n"
        "(ordenado de peor mediana arriba a mejor mediana abajo)",
        fontsize=12,
        pad=12,
    )
    fig.tight_layout()
    _save(fig, "05_underservice_by_departamento.png")


# --- Gráfico 6 · Baseline vs alternativa -------------------------------------

def plot_baseline_vs_alternative(metrics: gpd.GeoDataFrame) -> None:
    df = metrics.dropna(subset=["underservice_index_baseline", "underservice_index_alt"])

    fig, ax = plt.subplots(figsize=(9, 9))
    sns.scatterplot(
        data=df,
        x="underservice_index_baseline",
        y="underservice_index_alt",
        alpha=0.45,
        s=20,
        color="#2c3e50",
        edgecolor="white",
        linewidth=0.3,
        ax=ax,
    )

    lim = 1.05 * max(
        abs(df["underservice_index_baseline"].min()),
        df["underservice_index_baseline"].max(),
        abs(df["underservice_index_alt"].min()),
        df["underservice_index_alt"].max(),
    )
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        label="Línea de acuerdo perfecto (y = x)",
    )

    rho, _ = spearmanr(df["underservice_index_baseline"], df["underservice_index_alt"])
    ax.set_xlabel("Índice subatención — especificación BASELINE\n(umbral 30 km · oferta por km²)", fontsize=11)
    ax.set_ylabel("Índice subatención — especificación ALTERNATIVA\n(umbral 15 km · oferta por centro poblado)", fontsize=11)
    ax.set_title(
        f"Sensibilidad del índice al cambio de especificación\n"
        f"Spearman ρ = {rho:.3f} · puntos lejos de y = x son los distritos más sensibles",
        fontsize=12,
        pad=12,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()
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
