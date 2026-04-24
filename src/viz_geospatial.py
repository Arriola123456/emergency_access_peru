"""T5 — Salidas geoespaciales: choropleths estáticos + mapa folium interactivo.

Usa el índice de cobertura [0, 1] (1 = mejor atendido). Paleta RdYlGn:
rojo = baja cobertura, verde = alta cobertura.
"""

from __future__ import annotations

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from folium.plugins import MarkerCluster
from matplotlib.colors import ListedColormap

from src.config import CRS_WGS84, DATA_PROCESSED, MAPS
from src.utils import get_logger

log = get_logger(__name__)


def _save_fig(fig: plt.Figure, name: str) -> None:
    MAPS.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAPS / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"wrote {name}")


# --- Choropleth estático -------------------------------------------------------

def choropleth_static(
    metrics: gpd.GeoDataFrame,
    column: str,
    title: str,
    filename: str,
    cmap: str = "RdYlGn",
    legend_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 12))
    metrics.plot(
        column=column,
        cmap=cmap,
        linewidth=0.1,
        edgecolor="grey",
        legend=True,
        legend_kwds={"label": legend_label or column, "shrink": 0.5},
        missing_kwds={"color": "lightgrey", "label": "sin dato"},
        vmin=0 if "coverage" in column or "share" in column else None,
        vmax=1 if "coverage" in column or "share" in column else None,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_axis_off()
    _save_fig(fig, filename)


# --- Mapa bivariado ------------------------------------------------------------

def bivariate_map(metrics: gpd.GeoDataFrame) -> None:
    """Mapa bivariado: oferta normalizada × acceso normalizado (baseline).
    Paleta 3×3: filas oferta baja→alta, columnas acceso bajo→alto."""
    df = metrics.copy()
    df["q_oferta"] = pd.qcut(
        df["supply_01_baseline"].rank(method="first"), q=3, labels=[0, 1, 2]
    )
    df["q_acceso"] = pd.qcut(
        df["access_01_baseline"].fillna(0).rank(method="first"), q=3, labels=[0, 1, 2]
    )
    df["biv"] = df["q_oferta"].astype(int) * 3 + df["q_acceso"].astype(int)

    palette = [
        "#e8e8e8", "#ace4e4", "#5ac8c8",   # oferta baja
        "#dfb0d6", "#a5add3", "#5698b9",   # oferta media
        "#be64ac", "#8c62aa", "#3b4994",   # oferta alta
    ]
    cmap = ListedColormap(palette)

    fig, (ax, ax_leg) = plt.subplots(
        1, 2, figsize=(14, 11), gridspec_kw={"width_ratios": [4, 1]}
    )
    df.plot(column="biv", categorical=True, cmap=cmap, linewidth=0.1, edgecolor="grey",
            ax=ax, legend=False)
    ax.set_title(
        "Mapa bivariado: oferta de emergencia × acceso desde CPs\n"
        "(filas: oferta baja→alta; columnas: acceso bajo→alto)",
        fontsize=12,
        pad=10,
    )
    ax.set_axis_off()

    ax_leg.set_xlim(-0.5, 2.5)
    ax_leg.set_ylim(-0.5, 2.5)
    for i in range(3):
        for j in range(3):
            ax_leg.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=palette[i * 3 + j], edgecolor="white"))
    ax_leg.set_xticks([0, 1, 2])
    ax_leg.set_xticklabels(["bajo", "medio", "alto"])
    ax_leg.set_yticks([0, 1, 2])
    ax_leg.set_yticklabels(["bajo", "medio", "alto"])
    ax_leg.set_xlabel("Acceso (share CP ≤ 30 km)")
    ax_leg.set_ylabel("Oferta (emergencias/100 km², log-normalizada)")
    ax_leg.set_aspect("equal")
    ax_leg.set_title("Leyenda 3×3")
    _save_fig(fig, "bivariate_supply_access.png")


# --- Mapa folium interactivo ---------------------------------------------------

def interactive_folium_map(metrics: gpd.GeoDataFrame, ipress_gdf: gpd.GeoDataFrame) -> None:
    center = [-9.19, -75.01]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron", control_scale=True)

    column = "coverage_index_baseline"
    df = metrics.dropna(subset=[column]).copy().to_crs(CRS_WGS84)

    # Paleta rojo → amarillo → verde (bajo → alto = peor → mejor atendido)
    cmap = LinearColormap(
        ["#c0392b", "#ffffbf", "#27ae60"],
        vmin=0.0,
        vmax=1.0,
        caption="Índice de cobertura de emergencia (0 = peor · 1 = mejor)",
    )
    cmap.add_to(m)

    def style_fn(feature):
        v = feature["properties"].get(column)
        return {
            "fillColor": cmap(v) if v is not None else "#cccccc",
            "color": "#555555",
            "weight": 0.3,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        data=df[[
            "ubigeo", "distrito", "provincia", "departamento", column,
            "n_facilities", "n_emergency_facilities", "share_cp_within_30km", "geometry",
        ]].to_json(),
        name="Distritos — cobertura (baseline)",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "distrito", "provincia", "departamento", column,
                "n_facilities", "n_emergency_facilities", "share_cp_within_30km",
            ],
            aliases=[
                "Distrito:", "Provincia:", "Dpto:", "Cobertura [0-1]:",
                "IPRESS:", "IPRESS emergencia:", "Acceso 30 km:",
            ],
            localize=True,
            sticky=False,
        ),
    ).add_to(m)

    em = ipress_gdf[ipress_gdf["is_emergency"]].to_crs(CRS_WGS84)
    cluster = MarkerCluster(name=f"IPRESS con emergencia (n={len(em):,})").add_to(m)
    for _, row in em.iterrows():
        popup = folium.Popup(
            f"<b>{row.get('nombre', '')}</b><br>"
            f"Ubigeo: {row.get('ubigeo_final') or row.get('ubigeo', '')}<br>"
            f"Atenciones 2024: {int(row.get('atenciones', 0)):,}<br>"
            f"Institución: {row.get('institucion', '')}",
            max_width=300,
        )
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color="#c0392b",
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    out = MAPS / "interactive_map.html"
    MAPS.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    log.info(f"wrote {out.name}")


# --- Main ----------------------------------------------------------------------

def main() -> None:
    log.info("=== T5: GEOSPATIAL OUTPUTS ===")
    metrics = gpd.read_parquet(DATA_PROCESSED / "district_metrics.parquet")
    ipress = gpd.read_parquet(DATA_PROCESSED / "ipress_geo.parquet")

    choropleth_static(
        metrics,
        "coverage_index_baseline",
        "Índice de cobertura de emergencia — BASELINE (30 km · oferta por km²)",
        "choropleth_coverage_baseline.png",
        cmap="RdYlGn",
        legend_label="Cobertura [0, 1]",
    )
    choropleth_static(
        metrics,
        "coverage_index_alt",
        "Índice de cobertura de emergencia — ALTERNATIVA (15 km · oferta por CP)",
        "choropleth_coverage_alternative.png",
        cmap="RdYlGn",
        legend_label="Cobertura [0, 1]",
    )
    choropleth_static(
        metrics,
        "share_cp_within_30km",
        "Share de centros poblados con emergencia ≤ 30 km",
        "choropleth_access_share.png",
        cmap="YlGnBu",
        legend_label="Share CP ≤ 30 km",
    )

    bivariate_map(metrics)
    interactive_folium_map(metrics, ipress)
    log.info("=== T5 complete ===")


if __name__ == "__main__":
    main()
