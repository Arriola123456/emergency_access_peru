"""T6 — Streamlit app: Emergency Access Peru.

Cuatro tabs obligatorios (rubric):
  1. Data & Methodology
  2. Static Analysis
  3. GeoSpatial Results
  4. Interactive Exploration

Corre desde la raíz del repo con:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite `from src...` cuando Streamlit arranca desde la raíz
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from src.config import DATA_PROCESSED, FIGURES, MAPS

# ---------------------------------------------------------------- Page config --

st.set_page_config(
    page_title="Emergency Access Peru",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- Data loaders -

@st.cache_data(show_spinner="Cargando métricas distritales...")
def load_metrics() -> gpd.GeoDataFrame:
    return gpd.read_parquet(DATA_PROCESSED / "district_metrics.parquet")


@st.cache_data(show_spinner="Cargando IPRESS...")
def load_ipress() -> gpd.GeoDataFrame:
    return gpd.read_parquet(DATA_PROCESSED / "ipress_geo.parquet")


@st.cache_data(show_spinner="Cargando centros poblados con distancias...")
def load_ccpp() -> gpd.GeoDataFrame:
    return gpd.read_parquet(DATA_PROCESSED / "ccpp_with_distance.parquet")


@st.cache_data
def load_sensitivity() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "tables" / "sensitivity_ranking.csv")


# ------------------------------------------------------------------- Sidebar --

st.sidebar.title("Emergency Access Peru")
st.sidebar.markdown(
    "Análisis geoespacial de la desigualdad en el acceso a servicios de emergencia "
    "en salud entre distritos del Perú (Tarea 2 — Data Science, 2026-04)."
)
st.sidebar.info(
    "Fuentes: IPRESS MINSA · Emergencias SUSALUD 2024 · Centros Poblados INEI · Shapefile distritos (curso)."
)

# ------------------------------------------------------------------- Title ----

st.title(":hospital: Acceso a Emergencias — Desigualdad Distrital en Perú")

# -------------------------------------------------------------------- Tabs ----

tab_data, tab_static, tab_geo, tab_explore = st.tabs(
    ["📚 Data & Methodology", "📊 Static Analysis", "🗺️ GeoSpatial Results", "🔍 Interactive Exploration"]
)


# ------------------------------------------------------- Tab 1: Data & Method -

with tab_data:
    st.header("Datasets, limpieza y definiciones")

    st.subheader("Preguntas de investigación")
    st.markdown(
        """
1. ¿Qué distritos tienen menor / mayor disponibilidad de establecimientos y emergencias?
2. ¿Qué distritos muestran un acceso espacial más débil desde sus centros poblados?
3. ¿Qué distritos están más / menos desatendidos combinando presencia, actividad y acceso?
4. ¿Qué tan sensibles son los resultados al cambiar la definición analítica de "acceso"?
        """
    )

    st.subheader("Fuentes")
    st.dataframe(
        pd.DataFrame(
            {
                "Dataset": [
                    "IPRESS MINSA",
                    "Emergencias SUSALUD 2024",
                    "Centros Poblados INEI",
                    "Límites distritales (DISTRITOS.shp)",
                ],
                "Fuente": [
                    "datosabiertos.gob.pe",
                    "datos.susalud.gob.pe",
                    "datosabiertos.gob.pe",
                    "repo d2cml-ai/Data-Science-Python",
                ],
                "Formato": ["CSV", "CSV", "Shapefile (en ZIP)", "Shapefile (repo)"],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Métricas — Baseline vs Alternativa")
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            """
            **Baseline**
            - Oferta: emergencias / 100 km²
            - Actividad: log(1 + atenciones)
            - Acceso: share CP ≤ **30 km**
            - Índice = media de z-scores invertidos (3 dimensiones)
            """
        )
    with colB:
        st.markdown(
            """
            **Alternativa (sensibilidad)**
            - Oferta: emergencias / nº centros poblados
            - Actividad: log(1 + atenciones)
            - Acceso: share CP ≤ **15 km** (más estricto)
            - Índice equivalente con las 3 dimensiones alternativas
            """
        )

    st.subheader("Quirks del dato fuente")
    st.markdown(
        """
        - **IPRESS NORTE/ESTE invertidos** en origen: `NORTE` tiene longitud, `ESTE` tiene latitud. El loader los remapea.
        - **SUSALUD anonimiza** celdas pequeñas con `NE_0001` (< umbral de privacidad) → se parsean como NaN y no suman.
        - **Centros Poblados INEI** no trae UBIGEO explícito → se deriva de `CÓDIGO[:6]`. Tampoco trae población → peso uniforme en baseline.
        - **Separadores heterogéneos**: IPRESS usa `,`, SUSALUD usa `;`, ambos en `latin-1`.
        """
    )

    st.subheader("Números clave")
    metrics = load_metrics()
    ipress = load_ipress()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distritos analizados", f"{len(metrics):,}")
    c2.metric("IPRESS totales", f"{len(ipress):,}")
    c3.metric("IPRESS de emergencia", f"{int(ipress['is_emergency'].sum()):,}")
    c4.metric("Distritos sin emergencia",
              f"{int((metrics['n_emergency_facilities'] == 0).sum()):,}",
              help="Distritos donde ningún IPRESS reportó atención de emergencia en 2024")


# --------------------------------------------------- Tab 2: Static Analysis ---

with tab_static:
    st.header("Análisis estático")
    figs = sorted(FIGURES.glob("*.png"))
    if not figs:
        st.warning("No hay figuras generadas. Ejecuta `python -m src.viz_static`.")
    else:
        for p in figs:
            st.markdown(f"#### {p.stem.replace('_', ' ').title()}")
            st.image(str(p), use_column_width=True)

    st.divider()
    st.subheader("Tabla de sensibilidad (ranking baseline vs alternativa)")
    sens = load_sensitivity()
    st.dataframe(
        sens.sort_values("rank_baseline").head(50),
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------- Tab 3: GeoSpatial Results -

with tab_geo:
    st.header("Resultados geoespaciales")

    st.subheader("Choropleths estáticos")
    cols = st.columns(2)
    png_maps = sorted(MAPS.glob("*.png"))
    for i, p in enumerate(png_maps):
        with cols[i % 2]:
            st.image(str(p), caption=p.stem.replace("_", " ").title(), use_column_width=True)

    st.divider()
    st.subheader("Mapa interactivo embebido")
    interactive = MAPS / "interactive_map.html"
    if interactive.exists():
        html_content = interactive.read_text(encoding="utf-8")
        st.components.v1.html(html_content, height=650, scrolling=False)
    else:
        st.warning("Mapa interactivo no generado. Ejecuta `python -m src.viz_geospatial`.")


# ----------------------------------------------- Tab 4: Interactive Exploration

with tab_explore:
    st.header("Exploración interactiva por distrito")

    metrics = load_metrics()
    ipress = load_ipress()

    # Filtros
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        spec = st.radio("Especificación", ["Baseline", "Alternativa"], horizontal=True)
    with col2:
        dpto_options = sorted(metrics["departamento"].dropna().unique().tolist())
        dpto = st.selectbox("Departamento", ["(todos)"] + dpto_options)
    with col3:
        metric_col = (
            "underservice_index_baseline" if spec == "Baseline" else "underservice_index_alt"
        )
        top_n = st.slider("Top N a resaltar", 5, 50, 10)

    view = metrics.copy() if dpto == "(todos)" else metrics[metrics["departamento"] == dpto]

    # Métricas agregadas del filtro
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Distritos en vista", f"{len(view):,}")
    k2.metric("Sin IPRESS emergencia", f"{int((view['n_emergency_facilities'] == 0).sum()):,}")
    k3.metric("IPRESS totales", f"{int(view['n_facilities'].sum()):,}")
    k4.metric("Atenciones 2024 (M)", f"{view['total_atenciones'].sum() / 1e6:.2f}")

    st.divider()

    # Mini-mapa folium filtrado
    st.subheader(f"Mapa — subatención {spec.lower()}")
    view_wgs = view.to_crs("EPSG:4326").dropna(subset=[metric_col])
    if len(view_wgs) > 0:
        bounds = view_wgs.total_bounds  # minx, miny, maxx, maxy
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        fm = folium.Map(location=center, zoom_start=7 if dpto != "(todos)" else 5,
                        tiles="CartoDB positron")

        vmin = float(view_wgs[metric_col].min())
        vmax = float(view_wgs[metric_col].max())
        cmap = LinearColormap(["#2ca25f", "#ffffbf", "#c0392b"], vmin=vmin, vmax=vmax,
                              caption=metric_col)
        cmap.add_to(fm)

        # Resaltar top-N más subatendidos
        top_ubigeos = set(view_wgs.nlargest(top_n, metric_col)["ubigeo"].tolist())

        def style_fn(feat):
            v = feat["properties"].get(metric_col)
            is_top = feat["properties"]["ubigeo"] in top_ubigeos
            return {
                "fillColor": cmap(v) if v is not None else "#cccccc",
                "color": "#000000" if is_top else "#888888",
                "weight": 2.0 if is_top else 0.4,
                "fillOpacity": 0.75,
            }

        folium.GeoJson(
            data=view_wgs[["ubigeo", "distrito", "provincia", "departamento",
                           metric_col, "n_facilities", "n_emergency_facilities",
                           "share_cp_within_30km", "geometry"]].to_json(),
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["distrito", "provincia", "departamento", metric_col,
                        "n_facilities", "n_emergency_facilities", "share_cp_within_30km"],
                aliases=["Distrito:", "Provincia:", "Dpto:", "Score:", "IPRESS:",
                         "Emergencia:", "Acceso 30km:"],
                localize=True,
            ),
        ).add_to(fm)

        st_folium(fm, use_container_width=True, height=550, returned_objects=[])
    else:
        st.info("No hay distritos con score en esta vista.")

    st.divider()
    st.subheader(f"Top {top_n} distritos más subatendidos — {spec}")
    cols_show = ["departamento", "provincia", "distrito", metric_col,
                 "n_facilities", "n_emergency_facilities", "total_atenciones",
                 "share_cp_within_30km", "mean_dist_km"]
    cols_show = [c for c in cols_show if c in view.columns]
    top_df = (
        view.dropna(subset=[metric_col])
        .nlargest(top_n, metric_col)[cols_show]
        .reset_index(drop=True)
    )
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    # Comparar baseline vs alternativa
    st.subheader("Comparación baseline vs alternativa (cambio de ranking)")
    sens = load_sensitivity()
    if dpto != "(todos)":
        sens = sens[sens["departamento"] == dpto]
    most_movers = sens.reindex(sens["rank_diff"].abs().sort_values(ascending=False).index).head(top_n)
    st.dataframe(
        most_movers[["departamento", "provincia", "distrito", "rank_baseline",
                     "rank_alternative", "rank_diff"]],
        use_container_width=True,
        hide_index=True,
    )
