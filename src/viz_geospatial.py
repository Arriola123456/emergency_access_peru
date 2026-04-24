"""T5 — Salidas geoespaciales: choropleth estáticos + mapa folium interactivo."""

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
    fig.savefig(MAPS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {name}")


# --- Choropleth estático -------------------------------------------------------

def choropleth_static(metrics: gpd.GeoDataFrame, column: str, title: str, filename: str, cmap: str = "YlOrRd") -> None:
    fig, ax = plt.subplots(figsize=(10, 12))
    metrics.plot(
        column=column,
        cmap=cmap,
        linewidth=0.1,
        edgecolor="grey",
        legend=True,
        legend_kwds={"label": column, "shrink": 0.5},
        missing_kwds={"color": "lightgrey", "label": "sin dato"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()
    _save_fig(fig, filename)


# --- Mapa bivariado ------------------------------------------------------------

def bivariate_map(metrics: gpd.GeoDataFrame) -> None:
    """Mapa bivariado: oferta (n_emergency_facilities) vs acceso (share_cp_within_30km).
    Paleta 3x3: filas oferta baja→alta, columnas acceso bajo→alto."""
    df = metrics.copy()
    # cuantiles 3x3
    df["q_oferta"] = pd.qcut(df["n_emergency_per_100km2"].rank(method="first"), q=3, labels=[0, 1, 2])
    df["q_acceso"] = pd.qcut(df["share_cp_within_30km"].fillna(0).rank(method="first"), q=3, labels=[0, 1, 2])
    df["biv"] = df["q_oferta"].astype(int) * 3 + df["q_acceso"].astype(int)

    # Paleta 3x3 (Joshua Stevens bivariate)
    palette = [
        "#e8e8e8", "#ace4e4", "#5ac8c8",   # oferta baja
        "#dfb0d6", "#a5add3", "#5698b9",   # oferta media
        "#be64ac", "#8c62aa", "#3b4994",   # oferta alta
    ]
    cmap = ListedColormap(palette)

    fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(14, 11), gridspec_kw={"width_ratios": [4, 1]})
    df.plot(column="biv", categorical=True, cmap=cmap, linewidth=0.1, edgecolor="grey", ax=ax, legend=False)
    ax.set_title("Mapa bivariado: oferta de emergencia × acceso de CPs\n(filas oferta ↑, columnas acceso →)", fontsize=12)
    ax.set_axis_off()

    # Leyenda 3x3
    ax_leg.set_xlim(-0.5, 2.5)
    ax_leg.set_ylim(-0.5, 2.5)
    for i in range(3):
        for j in range(3):
            ax_leg.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=palette[i * 3 + j], edgecolor="white"))
    ax_leg.set_xticks([0, 1, 2])
    ax_leg.set_xticklabels(["bajo", "medio", "alto"])
    ax_leg.set_yticks([0, 1, 2])
    ax_leg.set_yticklabels(["bajo", "medio", "alto"])
    ax_leg.set_xlabel("Acceso (share CP ≤ 30 km)")
    ax_leg.set_ylabel("Oferta (emergencias/100 km²)")
    ax_leg.set_aspect("equal")
    ax_leg.set_title("Leyenda 3×3")
    _save_fig(fig, "bivariate_supply_access.png")


# --- Mapa folium interactivo ---------------------------------------------------

def interactive_folium_map(metrics: gpd.GeoDataFrame, ipress_gdf: gpd.GeoDataFrame) -> None:
    # Centro en Perú
    center = [-9.19, -75.01]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron", control_scale=True)

    # Elegir columna para el choropleth
    column = "underservice_index_baseline"
    df = metrics.dropna(subset=[column]).copy()
    df = df.to_crs(CRS_WGS84)

    # Rango de valores para colorbar
    vmin, vmax = float(df[column].min()), float(df[column].max())
    cmap = LinearColormap(["#2ca25f", "#ffffbf", "#c0392b"], vmin=vmin, vmax=vmax,
                          caption="Índice de subatención (baseline)")
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
        data=df[["ubigeo", "distrito", "provincia", "departamento", column,
                 "n_facilities", "n_emergency_facilities", "share_cp_within_30km", "geometry"]].to_json(),
        name="Distritos — subatención (baseline)",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["distrito", "provincia", "departamento", column,
                    "n_facilities", "n_emergency_facilities", "share_cp_within_30km"],
            aliases=["Distrito:", "Provincia:", "Dpto:", "Subatención:",
                     "IPRESS:", "IPRESS emergencia:", "Acceso 30 km:"],
            localize=True,
            sticky=False,
        ),
    ).add_to(m)

    # Capa IPRESS de emergencia (clusterizados)
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

    choropleth_static(metrics, "underservice_index_baseline",
                      "Índice de subatención (BASELINE — 30 km, densidad/km²)",
                      "choropleth_underservice_baseline.png")
    choropleth_static(metrics, "underservice_index_alt",
                      "Índice de subatención (ALTERNATIVA — 15 km, oferta/CP)",
                      "choropleth_underservice_alternative.png")
    choropleth_static(metrics, "share_cp_within_30km",
                      "Share de centros poblados con emergencia ≤ 30 km",
                      "choropleth_access_share.png",
                      cmap="YlGnBu")

    bivariate_map(metrics)
    interactive_folium_map(metrics, ipress)
    log.info("=== T5 complete ===")


if __name__ == "__main__":
    main()
