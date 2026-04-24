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
    ["📚 Datos y Metodología", "📊 Análisis Estático", "🗺️ Resultados Geoespaciales", "🔍 Exploración Interactiva"]
)


# ------------------------------------------------------- Tab 1: Datos y Método

with tab_data:
    st.header("Datos y Metodología")
    st.caption(
        "Documentación extensa del proyecto: motivación, preguntas de investigación, "
        "fuentes, limpieza, decisiones metodológicas, fórmulas, quirks y limitaciones."
    )

    # ────────────────────────────────────────────────────────── 1. MOTIVACIÓN ──
    st.markdown("## 1. Motivación")
    st.markdown(
        """
El Perú cubre **1 285 216 km²** repartidos entre costa, sierra y selva, con marcadas
desigualdades geográficas en el acceso a servicios públicos. La atención médica de
**emergencia** es especialmente sensible al tiempo: eventos como infartos, accidentes
cerebrovasculares, traumatismos severos o emergencias obstétricas tienen un desenlace
drásticamente peor cuando la atención se demora más de la llamada "hora dorada".

Cuando un distrito carece de un establecimiento con servicio de emergencia efectivo, los
pacientes enfrentan viajes largos por trochas, dependencia del transporte informal y
riesgos clínicos evitables. La pregunta de política pública — *¿qué distritos están más
desprotegidos?* — suele responderse con indicadores de **una sola dimensión** (por
ejemplo, "nº de postas por distrito"), lo que esconde tres problemas:

1. Tener un establecimiento **formalmente existente** no garantiza que atienda emergencias.
2. Tener atención de emergencia **en el distrito** no garantiza que esté cerca de todos
   los centros poblados.
3. Los **thresholds** y las agregaciones son arbitrarias; una conclusión basada en un
   solo corte puede invertirse con otro.

Este proyecto combina oferta, actividad y acceso espacial en un **único índice
de cobertura de emergencia por distrito**, construido en la escala **[0, 1]** (donde
**1 = mejor atendido** y **0 = peor atendido**), y compara dos especificaciones
para medir la robustez de los resultados.
        """
    )

    # ───────────────────────────────────────────── 2. PREGUNTAS DE INVESTIGACIÓN
    st.markdown("## 2. Preguntas de investigación")
    st.markdown(
        """
A continuación, cada pregunta del enunciado con (a) la interpretación específica que
usamos y (b) cómo la operacionalizamos en el pipeline.
        """
    )

    with st.expander("Pregunta 1 — Disponibilidad de establecimientos y atención de emergencias"):
        st.markdown(
            """
> **¿Qué distritos tienen menor / mayor disponibilidad de establecimientos de salud y de atención de emergencias?**

**Qué responde:** el inventario básico. Pone una lupa sobre la oferta *formal* (cuántos
IPRESS existen administrativamente) y sobre la oferta *efectiva* (cuántos reportaron al
menos una atención de emergencia en 2024).

**Cómo la abordamos:**
- Contamos IPRESS por distrito, corrigiendo UBIGEOs inconsistentes con un *spatial join*
  contra los polígonos distritales (600 establecimientos tenían el UBIGEO declarado
  distinto al polígono donde caen geográficamente).
- Marcamos `is_emergency = True` para un IPRESS si figura en el dataset SUSALUD de
  producción de emergencias 2024 con al menos 1 atención sumada.
- Agregamos al distrito: `n_facilities`, `n_emergency_facilities`, `total_atenciones`.
            """
        )

    with st.expander("Pregunta 2 — Acceso espacial desde los centros poblados"):
        st.markdown(
            """
> **¿Qué distritos muestran un acceso espacial más débil desde sus centros poblados hacia servicios de emergencia?**

**Qué responde:** incluso si el distrito tiene un IPRESS que atiende emergencias, los
centros poblados (CP) del mismo distrito pueden estar a decenas o cientos de km de
distancia real.

**Cómo la abordamos:**
- Para cada uno de los **136 587 centros poblados** del Perú, calculamos la distancia
  euclidiana (proyección UTM 18S) al IPRESS con actividad de emergencia más cercano,
  usando un *KDTree* (complejidad O(n log m)).
- Agregamos a nivel distrital: promedio, mediana, máximo, y la *share* de CPs cuya
  distancia cae bajo el umbral de acceso (30 km en baseline, 15 km en alternativa).
            """
        )

    with st.expander("Pregunta 3 — Combinación de presencia, actividad y acceso"):
        st.markdown(
            """
> **¿Qué distritos están más / menos desatendidos combinando presencia, actividad y acceso?**

**Qué responde:** ninguna dimensión por sí sola basta. Un distrito con un IPRESS
*fantasma* (existe en papel pero no atendió a nadie) no está servido aunque figure en
el inventario. Un distrito con mucha actividad pero en un solo punto puede tener CPs
muy alejados. El índice compuesto equilibra las tres.

**Cómo la abordamos:**
- Normalizamos cada dimensión con z-score sobre los 1 873 distritos.
- Invertimos el signo (multiplicamos por −1) para que el índice se lea de forma
  intuitiva: **mayor = más subatendido**.
- Promediamos las 3 componentes estandarizadas.
            """
        )

    with st.expander("Pregunta 4 — Sensibilidad a la definición de 'acceso'"):
        st.markdown(
            """
> **¿Qué tan sensibles son los resultados al cambiar la definición analítica de "acceso"?**

**Qué responde:** los thresholds (30 km, 15 km) y la forma de agregar oferta
(por km² vs por CP) son decisiones analíticas. ¿Cuánto cambia la foto si las
ajustamos?

**Cómo la abordamos:**
- Construimos una especificación **alternativa** que altera dos componentes
  (oferta por CP en vez de por km²; acceso con umbral 15 km en vez de 30 km) y
  deja la tercera igual.
- Comparamos los rankings resultantes con el **coeficiente de correlación de
  Spearman** y publicamos la tabla de diferencias de ranking por distrito
  (`sensitivity_ranking.csv`).
- **Resultado:** ρ = **0.876** — alto, pero con movimientos notables en el
  tramo medio del ranking, como se esperaba.
            """
        )

    # ─────────────────────────────────────────────── 3. FUENTES DE DATOS ──
    st.markdown("## 3. Fuentes de datos")
    st.dataframe(
        pd.DataFrame(
            {
                "Dataset": [
                    "IPRESS MINSA",
                    "Emergencias SUSALUD 2024",
                    "Centros Poblados INEI",
                    "Límites distritales (DISTRITOS.shp)",
                ],
                "Descripción": [
                    "Directorio nacional de Instituciones Prestadoras de Servicios de Salud (código único, UBIGEO, categoría, coordenadas).",
                    "Producción de atenciones de emergencia por IPRESS, desagregada por mes, sexo y edad. Año de análisis: 2024.",
                    "Shapefile de puntos INEI con los 136 587 centros poblados del Perú.",
                    "Shapefile de 1 873 polígonos distritales usado en la clase de GeoPandas.",
                ],
                "Fuente": [
                    "datosabiertos.gob.pe",
                    "datos.susalud.gob.pe",
                    "datosabiertos.gob.pe",
                    "d2cml-ai/Data-Science-Python",
                ],
                "Formato": ["CSV", "CSV", "Shapefile (en ZIP)", "Shapefile (raw)"],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ───────────────────────────────────────────── 4. RESUMEN DE LIMPIEZA ──
    st.markdown("## 4. Resumen de limpieza (data cleaning)")
    st.markdown(
        "El pipeline de ingesta (`src/ingest.py`) aplica los siguientes pasos por "
        "dataset, de forma reproducible y cacheada (no descarga si el archivo ya existe)."
    )

    st.markdown("#### 4.1 · IPRESS MINSA")
    st.markdown(
        """
- **Encoding:** `latin-1` · **Separador:** `,`
- Normalización de nombres de columnas (strip de tildes, uppercase, espacios → `_`).
- **Crítico:** en el CSV fuente las columnas `NORTE` y `ESTE` están *invertidas* — la
  que se llama `NORTE` tiene valores de **longitud** (rango −81..−68) y la que se llama
  `ESTE` tiene **latitud** (rango −18..0). El loader las remapea explícitamente a
  `lon` y `lat`.
- Filtro de coordenadas dentro del rango continental del Perú
  (`lat ∈ [−18.5, 0.5]`, `lon ∈ [−81.5, −68.0]`).
- UBIGEO normalizado a 6 dígitos zero-padded (p. ej. `80909` → `080909`).
- Deduplicación por `CODIGO_UNICO`.
- **Resultado:** de 20 819 IPRESS originales quedan **7 951** con coordenadas válidas
  (≈ 38 %). Los 12 868 restantes no tienen coordenadas en el directorio publicado y
  se descartan únicamente para el análisis espacial.
        """
    )

    st.markdown("#### 4.2 · Producción de emergencias SUSALUD 2024")
    st.markdown(
        """
- **Encoding:** `latin-1` · **Separador:** `;`
- Cada fila es IPRESS × mes × sexo × edad.
- SUSALUD **anonimiza** los valores pequeños con el marcador textual `NE_0001`
  (cuando `atenciones < umbral de privacidad`). Se parsean como `NaN` y se cuentan
  como `0` al agregar anualmente — decisión documentada en `methodology.md`.
- Se agregan las atenciones por `(codigo, ubigeo)` → **4 293** IPRESS-año con
  aproximadamente **15.8 M atenciones** efectivas no-anonimizadas.
        """
    )

    st.markdown("#### 4.3 · Centros Poblados INEI")
    st.markdown(
        """
- Shapefile descargado como ZIP y extraído.
- **El shapefile no trae UBIGEO explícito;** se deriva de los primeros 6 dígitos del
  campo `CÓDIGO` (código INEI compuesto por DEP + PROV + DIST + CCPP).
- Tampoco trae población del centro poblado; se asume **peso uniforme = 1 por CP**
  en el baseline.
- Proyección asumida WGS 84 (EPSG:4326) si el `.prj` no la declara.
- **Resultado:** **136 587** CPs con UBIGEO derivado y geometría válida.
        """
    )

    st.markdown("#### 4.4 · Límites distritales (`DISTRITOS.shp`)")
    st.markdown(
        """
- Shapefile del repo del curso (`d2cml-ai/Data-Science-Python/_data/Folium`).
- Se descargan los 5 archivos companion (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`) para
  que GeoPandas pueda leerlo.
- UBIGEO se toma de `IDDIST` y se normaliza a 6 dígitos.
- Reproyección a EPSG:4326 para compatibilidad con Folium.
- **Resultado:** **1 873** polígonos distritales.
        """
    )

    # ──────────────────────────────────────── 5. DECISIONES METODOLÓGICAS ──
    st.markdown("## 5. Decisiones metodológicas")
    st.markdown(
        """
| # | Decisión | Justificación |
|---|---|---|
| D1 | Año de análisis: **2024** | Último año **completo** disponible en SUSALUD al momento del análisis (abril 2026). 2025/2026 aún se publican parcialmente. |
| D2 | Proyección para distancias: **EPSG:32718** (UTM 18S) | Distancias en metros sobre un elipsoide válido para la mayor parte del Perú. EPSG:4326 no sirve porque son grados. |
| D3 | Cálculo de vecino más cercano con **KDTree** | Complejidad O(n log m) para 136 587 CPs × 787 IPRESS emergencia, en ~1 s. |
| D4 | *Spatial join* IPRESS ↔ distritos (within) | Arregla **600** UBIGEOs declarados inconsistentes con el polígono real donde cae el IPRESS. |
| D5 | "IPRESS de emergencia" = `atenciones > 0` en SUSALUD 2024 | Se filtra por actividad *efectiva* en vez de categoría administrativa — captura mejor la oferta real. |
| D6 | `log(1 + atenciones)` | La distribución de atenciones es muy sesgada (hospitales grandes de Lima vs rurales); `log1p` estabiliza y maneja ceros. |
| D7 | Umbral de acceso **30 km** (baseline) | Benchmark común en estudios de acceso rural en Perú (aprox. 1 hora por trocha en condiciones razonables). |
| D8 | Umbral **15 km** en alternativa | Stresstest: ¿qué pasa si se exige el doble de cercanía? |
| D9 | Índice compuesto = **promedio simple** de 3 componentes normalizadas a [0, 1] | Garantiza que el índice total viva en [0, 1]; interpretación directa como *share de cobertura*. |
| D10 | Normalización min-max con **log-transform** para oferta y actividad | Ambas dimensiones son altamente sesgadas (Lima vs resto). El `log1p` comprime la cola larga antes del min-max, evitando que un megahospital aplaste la escala. |
| D11 | Agregación a distrito por el **UBIGEO geométrico** | Consistente con el shapefile, no con el UBIGEO declarado. |
        """
    )

    # ─────────────────────────────── 6. BASELINE vs ALTERNATIVA (FÓRMULAS) ──
    st.markdown("## 6. Cálculo del índice de cobertura (fórmulas explícitas)")
    st.markdown(
        r"""
El índice resumen está diseñado para vivir en **[0, 1]** con una lectura directa:
**1 = mejor atendido**, **0 = peor atendido**. Es el promedio simple de tres
componentes, cada una normalizada a [0, 1].

Para un distrito $d$ (identificado por su UBIGEO), definimos:

- $N^{\text{emerg}}_d$ = número de IPRESS con actividad de emergencia en $d$.
- $A_d$ = área del distrito en $\text{km}^2$ (geometría reproyectada a UTM 18S).
- $T_d$ = total de atenciones de emergencia reportadas en 2024 en $d$.
- $C_d$ = conjunto de centros poblados cuyo UBIGEO cae en $d$.
- $\text{dist}(c)$ = distancia (km) del centro poblado $c$ al IPRESS de emergencia más cercano.
        """
    )

    st.markdown("### 6.1 · Especificación BASELINE")

    st.markdown("**Dimensión 1 — Oferta** (densidad espacial):")
    st.latex(r"\text{oferta}^{\text{base}}_d \;=\; \frac{100 \cdot N^{\text{emerg}}_d}{A_d}")

    st.markdown("**Dimensión 2 — Actividad** (uso efectivo, log-transformado):")
    st.latex(r"\text{actividad}_d \;=\; \ln(1 + T_d)")

    st.markdown("**Dimensión 3 — Acceso espacial** (cobertura 30 km desde CPs, ya en [0, 1]):")
    st.latex(
        r"\text{acceso}^{30}_d \;=\; \frac{\bigl|\{\, c \in C_d \,:\, \text{dist}(c) \leq 30 \,\}\bigr|}{|C_d|}"
    )

    st.markdown(
        "**Normalización min-max a [0, 1]** (con log previo para las dimensiones "
        "sesgadas *oferta* y *actividad*). Para una variable $X$ definimos:"
    )
    st.latex(
        r"\widetilde{X}_d \;=\; \frac{\ln(1 + X_d) - \min_d \ln(1 + X_d)}{\max_d \ln(1 + X_d) - \min_d \ln(1 + X_d)} \;\in\; [0, 1]"
    )
    st.markdown(
        "Para *acceso*, que ya es una proporción en [0, 1], se usa directamente: "
        r"$\widetilde{\text{acceso}}^{30}_d = \text{acceso}^{30}_d$."
    )

    st.markdown("**Índice de cobertura baseline** (promedio simple de las 3 componentes normalizadas):")
    st.latex(
        r"I^{\text{base}}_d \;=\; \frac{\widetilde{\text{oferta}}^{\text{base}}_d"
        r" \;+\; \widetilde{\text{actividad}}_d"
        r" \;+\; \widetilde{\text{acceso}}^{30}_d}{3} \;\in\; [0, 1]"
    )

    st.success(
        "**Lectura directa:** $I^{\\text{base}}_d = 0$ significa que el distrito "
        "está en el **piso** simultáneamente en oferta, actividad y acceso "
        "(peor atendido posible); $I^{\\text{base}}_d = 1$ requeriría que el distrito "
        "fuera el **máximo observado** en las tres dimensiones a la vez (inalcanzable "
        "en la práctica)."
    )

    st.divider()
    st.markdown("### 6.2 · Especificación ALTERNATIVA (sensibilidad)")

    st.markdown(
        "Mantiene la estructura del índice pero **modifica dos dimensiones** "
        "para someter los resultados a estrés:"
    )

    st.markdown("**Dimensión 1' — Oferta por centro poblado** (no por área):")
    st.latex(r"\text{oferta}^{\text{alt}}_d \;=\; \frac{N^{\text{emerg}}_d}{|C_d|}")

    st.markdown("**Dimensión 2' — Actividad:** _igual al baseline._")
    st.latex(r"\text{actividad}_d \;=\; \ln(1 + T_d)")

    st.markdown("**Dimensión 3' — Acceso con umbral 15 km** (más estricto):")
    st.latex(
        r"\text{acceso}^{15}_d \;=\; \frac{\bigl|\{\, c \in C_d \,:\, \text{dist}(c) \leq 15 \,\}\bigr|}{|C_d|}"
    )

    st.markdown(
        "**Normalización min-max a [0, 1]:** idéntica regla que en baseline "
        "(log-transform para oferta y actividad; acceso ya en [0, 1])."
    )

    st.markdown("**Índice de cobertura alternativo:**")
    st.latex(
        r"I^{\text{alt}}_d \;=\; \frac{\widetilde{\text{oferta}}^{\text{alt}}_d"
        r" \;+\; \widetilde{\text{actividad}}_d"
        r" \;+\; \widetilde{\text{acceso}}^{15}_d}{3} \;\in\; [0, 1]"
    )

    st.divider()
    st.markdown("### 6.3 · Comparación lado a lado")
    st.markdown(
        r"""
| Dimensión | Baseline | Alternativa |
| --- | --- | --- |
| **Oferta** | $N^{\text{emerg}} / \text{área}$ (por 100 km²) | $N^{\text{emerg}} / |C|$ (por CP) |
| **Actividad** | $\ln(1 + T)$ | $\ln(1 + T)$ *(igual)* |
| **Acceso** | share de CPs con dist ≤ **30 km** | share de CPs con dist ≤ **15 km** |
| **Normalización** | min-max (con log para oferta y actividad) | idéntica |
| **Fórmula del índice** | $I^{\text{base}} = \tfrac{1}{3}(\widetilde{x}_1 + \widetilde{x}_2 + \widetilde{x}_3)$ | $I^{\text{alt}} = \tfrac{1}{3}(\widetilde{x}'_1 + \widetilde{x}_2 + \widetilde{x}'_3)$ |
| **Rango** | $[0, 1]$ | $[0, 1]$ |
        """
    )

    st.markdown("### 6.4 · Medida de sensibilidad")
    st.latex(
        r"\rho_S \;=\; \text{Spearman}\bigl(\text{rank}(I^{\text{base}}),\;\text{rank}(I^{\text{alt}})\bigr) \;=\; 0{.}891"
    )
    st.markdown(
        "La correlación es **alta pero no perfecta** — las conclusiones cualitativas "
        "se mantienen (los distritos amazónicos y altoandinos siguen como los peor "
        "atendidos), pero hay **reordenamientos** notables en el tramo medio del "
        "ranking. La tabla completa de `rank_diff` por distrito está en "
        "`outputs/tables/sensitivity_ranking.csv` y en el tab *Exploración Interactiva*."
    )

    # ───────────────────────────────────────── 7. QUIRKS DEL DATO FUENTE ──
    st.markdown("## 7. Quirks del dato fuente")
    st.markdown(
        """
Comportamientos inesperados encontrados durante la limpieza, que merecen mención:

1. **Inversión NORTE/ESTE en IPRESS.** La columna `NORTE` del CSV contiene longitudes
   y `ESTE` contiene latitudes. Probablemente un error de publicación histórica del
   MINSA. El loader los remapea explícitamente.
2. **Anonimización textual `NE_0001` en SUSALUD.** No es un error de tipado: es un
   código de privacidad para celdas con <5 observaciones. Se parsean como nulos.
3. **Sin UBIGEO en Centros Poblados.** El shapefile de INEI trae un `CÓDIGO`
   compuesto de 10+ dígitos; el UBIGEO distrital se reconstruye con los 6 primeros.
4. **Separadores heterogéneos.** IPRESS usa `,` como separador, SUSALUD usa `;`.
   Ambos en `latin-1`.
5. **Encoding con tildes rotas en nombres de columnas.** Después del decode a
   `latin-1` aparecen caracteres como `Código único`, que el pipeline normaliza con
   `unicodedata.NFKD`.
        """
    )

    # ───────────────────────────────────────────────── 8. LIMITACIONES ──
    st.markdown("## 8. Limitaciones")
    st.markdown(
        """
El análisis tiene varias limitaciones que conviene declarar explícitamente antes de
usar los resultados para política pública:

- **Subregistro SUSALUD en IPRESS rurales.** Establecimientos pequeños pueden no
  reportar completamente sus atenciones. La actividad medida está *sesgada hacia
  centros urbanos* que tienen sistemas de registro más robustos.
- **Peso uniforme por centro poblado.** Sin población real por CP, un caserío de 5
  personas pesa igual que un pueblo de 5 000. La integración con el censo CPV 2017
  del INEI queda como extensión natural.
- **Distancia euclidiana, no de viaje.** Calculamos línea recta en UTM. En la
  realidad, un CP a 20 km en línea recta puede estar a 4 horas por trocha; en la
  sierra alta la distancia euclidiana subestima severamente el tiempo real.
- **Umbrales de 15/30 km razonables pero convencionales.** No provienen de un
  estándar técnico peruano; son proxies. Por eso la especificación alternativa
  usa 15 km como stresstest.
- **IPRESS 2017.** El directorio publicado en `datos.gob.pe` es del snapshot 2017.
  Establecimientos creados o cerrados después pueden no aparecer. La actividad
  SUSALUD, en cambio, es 2024.
- **38 % de IPRESS sin coordenadas.** Descartados para cualquier análisis espacial.
  Si esa fracción no es aleatoria (por ejemplo, más IPRESS rurales sin coord que
  urbanos), hay sesgo en la densidad distrital.
- **Cobertura de centros poblados.** INEI registra los CPs censados; asentamientos
  informales recientes o caseríos dispersos no aparecen.
- **Índice compuesto = promedio simple de 3 componentes en [0, 1].** No hay
  justificación técnica para pesos iguales más allá de simplicidad. Pesos
  distintos (por ejemplo, más peso al acceso) cambiarían el ranking.
- **Normalización min-max dependiente de la muestra.** El máximo observado entre
  los 1 873 distritos define el "1" de la escala. Incorporar o excluir distritos
  (por ejemplo, si aparecieran datos mejorados de Lima) podría mover el 1 para
  todos. Una alternativa sería usar "goalposts" teóricos fijos como en el HDI de PNUD.
- **Un solo año (2024).** No capturamos tendencias ni shocks temporales
  (post-pandemia, feriados, eventos climáticos).
        """
    )

    # ────────────────────────────────────────────────── 9. NÚMEROS CLAVE ──
    st.markdown("## 9. Números clave")
    metrics = load_metrics()
    ipress = load_ipress()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distritos analizados", f"{len(metrics):,}")
    c2.metric("IPRESS con coords", f"{len(ipress):,}")
    c3.metric("IPRESS de emergencia", f"{int(ipress['is_emergency'].sum()):,}")
    c4.metric(
        "Distritos sin emergencia",
        f"{int((metrics['n_emergency_facilities'] == 0).sum()):,}",
        help="Distritos donde ningún IPRESS reportó atención de emergencia en 2024",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Atenciones 2024 (M)", f"{int(ipress['atenciones'].sum()) / 1e6:.1f}")
    c6.metric("Centros poblados", "136,587")
    c7.metric("Mediana dist. CP→emergencia", "22.8 km")
    c8.metric("Spearman(baseline, alt)", "0.891",
              help="Correlación de rangos entre los índices baseline y alternativo — mide sensibilidad")


# ─────────────────────────────────────────── Tab 2: Análisis Estático ───

with tab_static:
    st.header("Análisis Estático")
    st.caption(
        "Seis visualizaciones que, en conjunto, responden a las cuatro preguntas "
        "de investigación mediante cortes complementarios del mismo dataset."
    )

    # --- Índice de gráficos -----------------------------------------------
    st.markdown("## Índice de visualizaciones")
    st.markdown(
        """
| # | Gráfico | Pregunta que aborda | Qué se decide al verlo |
| --- | --- | --- | --- |
| 1 | **Extremos del ranking** (top 20 / bottom 20) | Q1, Q3 | Identificar distritos prioritarios y mejor cubiertos en una sola vista comparable. |
| 2 | **Oferta vs actividad** (scatter log-log) | Q1 | Separar establecimientos *formalmente existentes* de los que realmente atienden emergencias. |
| 3 | **Matriz de correlación Spearman** | Q3 | Validar que las 3 dimensiones del índice no son redundantes entre sí. |
| 4 | **Distribución de distancias CP → emergencia** | Q2 | Ver qué fracción de centros poblados queda bajo los umbrales de 15 km y 30 km. |
| 5 | **Boxplot por departamento** | Q1, Q3 | Explorar si la desigualdad es fenómeno departamental o distrital. |
| 6 | **Baseline vs alternativa** (scatter) | Q4 | Localizar qué distritos son *sensibles* al cambio de definición. |
        """
    )

    with st.expander("¿Por qué estos 6 gráficos y no otros? — justificación técnica"):
        st.markdown(
            """
La elección apunta a **cubrir las 4 preguntas con el menor número de vistas**,
priorizando comprensibilidad sobre sofisticación:

- **Diverging barplot (1)** en vez de un barplot con los 1 873 distritos: un
  gráfico ilegible no genera decisiones. Destacar los extremos es suficiente
  para priorización política.
- **Scatter log-log (2)** en vez de una regresión lineal: la distribución de
  atenciones está severamente sesgada (megahospital de Lima vs posta amazónica).
  El log-log estabiliza la varianza y expone outliers sin distorsionar la relación.
- **Heatmap Spearman (3)** en vez de *scatter matrix*: con 10 variables, un
  *scatter matrix* tiene 45 paneles inmanejables. Spearman (no Pearson) es
  robusto a no-linealidades y resume todo en una grilla.
- **Histograma con umbrales (4)** en vez de boxplot: conserva la **forma** de la
  distribución y permite ver visualmente qué porcentaje cae bajo cada umbral —
  información que un boxplot oculta.
- **Boxplot + strip (5)** en vez de violinplot: el violinplot es más informativo
  pero menos legible con 25 categorías. El strip superpuesto muestra puntos
  individuales, rescatando parte de esa densidad.
- **Scatter de sensibilidad (6)** en vez de reportar solo ρ: un número único
  (ρ = 0.876) es correcto pero **no muestra dónde están las discrepancias**.
  El scatter con y = x señala los casos sensibles a corregir en análisis futuros.
            """
        )

    st.divider()

    # --- Gráfico 1 --------------------------------------------------------
    st.markdown("### Gráfico 1 · Extremos del ranking de cobertura")
    img1 = FIGURES / "01_top_bottom_coverage.png"
    if img1.exists():
        st.image(str(img1), width="stretch")
    st.markdown(
        """
**Análisis:** los 20 distritos **peor atendidos** (barras rojas, cobertura ≈ 0) están
concentrados en provincias como **Ayabaca (Piura)**, **Chucuito (Puno)**, **Huamalíes
(Huánuco)**, **Huaytará (Huancavelica)** y **Atalaya (Ucayali)** — geografía
amazónica y altoandina donde la cobertura es **cero en oferta, actividad y acceso
simultáneamente**. El desempate se hizo por número de CPs (distritos con más CPs
pesan primero). Los 20 **mejor atendidos** (barras verdes, cobertura ≈ 0.87-0.97)
son esencialmente **Lima Metropolitana y Callao** (Jesús María **0.97**, Bellavista
**0.96**, Miraflores **0.90**), con capitales departamentales colándose (Arequipa
cercado **0.95**, Wanchaq-Cusco **0.87**). Ninguna alcanza el 1 perfecto: nadie es
simultáneamente el #1 en las tres dimensiones. La brecha entre extremos es
**prácticamente el ancho completo de la escala** [0, 1].
        """
    )
    st.divider()

    # --- Gráfico 2 --------------------------------------------------------
    st.markdown("### Gráfico 2 · Oferta vs actividad de emergencias")
    img2 = FIGURES / "02_scatter_facilities_vs_atenciones.png"
    if img2.exists():
        st.image(str(img2), width="stretch")
    st.markdown(
        """
**Análisis:** existe una relación positiva en escala log-log (más establecimientos →
más atenciones), pero con dos fenómenos interesantes. Los **outliers hacia arriba**
son los megahospitales limeños que concentran **millones** de atenciones en pocos
IPRESS (ver anotaciones). La banda de puntos en el **piso (atenciones ≈ 0)** revela
el problema central: muchos distritos tienen IPRESS *formalmente existentes* pero
**sin actividad de emergencia reportada a SUSALUD**. Este "fantasma administrativo"
es exactamente lo que motiva separar oferta y actividad como dimensiones independientes
del índice compuesto — un conteo simple de establecimientos sobreestima la cobertura.
        """
    )
    st.divider()

    # --- Gráfico 3 --------------------------------------------------------
    st.markdown("### Gráfico 3 · Matriz de correlación Spearman")
    img3 = FIGURES / "03_correlation_heatmap.png"
    if img3.exists():
        st.image(str(img3), width="stretch")
    st.markdown(
        """
**Análisis:** las correlaciones más fuertes son **estructurales**, no informativas:
*share 30 km* con *share 15 km* (ρ ≈ 0.95, mismo indicador con distinto umbral)
y *n establecimientos* con *n de emergencia activa* (ρ ≈ 0.77). Más revelador:
el **índice de cobertura baseline** correlaciona **positivamente** con oferta,
actividad y acceso (sus componentes por construcción), pero la magnitud **no es
+1**, lo que confirma que el promedio **no está dominado por una sola dimensión**
— las tres piezas aportan información distinta. La correlación del índice
baseline con el alternativo (≈ 0.89) anticipa el resultado del gráfico 6.
        """
    )
    st.divider()

    # --- Gráfico 4 --------------------------------------------------------
    st.markdown("### Gráfico 4 · Distribución de distancias CP → emergencia")
    img4 = FIGURES / "04_distance_histogram.png"
    if img4.exists():
        st.image(str(img4), width="stretch")
    st.markdown(
        """
**Análisis:** la distribución es **fuertemente asimétrica con cola larga**. La
mediana cae en ≈ 23 km, justo por encima del umbral alternativo de 15 km (línea
naranja) y por debajo del baseline de 30 km (línea roja). Una masa densa de
centros poblados (mayormente urbanos y periurbanos) cae bajo los 10 km, pero la
cola persistente llega hasta **377 km** — centros poblados remotos de la
Amazonía profunda. El gráfico justifica por qué la elección del umbral es una
**decisión crítica**: exigir 15 km deja fuera mucho más que exigir 30 km, y
ese gap es la fuente principal de la sensibilidad reportada en el gráfico 6.
        """
    )
    st.divider()

    # --- Gráfico 5 --------------------------------------------------------
    st.markdown("### Gráfico 5 · Cobertura por departamento")
    img5 = FIGURES / "05_coverage_by_departamento.png"
    if img5.exists():
        st.image(str(img5), width="stretch")
    st.markdown(
        """
**Análisis:** **Lima y Callao** son los únicos departamentos con mediana
claramente **superior** a la mediana nacional (mejor atendidos). Los peores
medianos son amazónicos (**Amazonas, Loreto, Ucayali, Madre de Dios**) y
altoandinos (**Apurímac, Huancavelica**). El hallazgo más importante es la
**alta varianza intra-departamental**: cada departamento tiene distritos buenos
y malos, con rangos intercuartílicos que se superponen fuertemente. La
desigualdad **no es un fenómeno puramente regional** — implicando que las
intervenciones de política deben operarse a **nivel distrital**, no por bloques
departamentales.
        """
    )
    st.divider()

    # --- Gráfico 6 --------------------------------------------------------
    st.markdown("### Gráfico 6 · Baseline vs alternativa (análisis de sensibilidad)")
    img6 = FIGURES / "06_baseline_vs_alternative.png"
    if img6.exists():
        st.image(str(img6), width="stretch")
    st.markdown(
        """
**Análisis:** la nube se pega a la línea **y = x**, reflejando la alta correlación
de Spearman (ρ = 0.891). Los puntos **alejados de la recta** son los distritos
cuyo ranking **cambia materialmente al modificar la definición de acceso** — son
los *movers*. Los puntos **sobre la recta** son robustos: están mal (o bien)
parados bajo ambas especificaciones, y deben ser el foco de priorización con
**mayor confianza**. Los *movers* ameritan un análisis caso por caso antes
de incluirlos en una lista de focalización, porque el resultado depende
materialmente del umbral y de la métrica de oferta que se elija.
        """
    )

    st.divider()

    # --- Tabla de sensibilidad -------------------------------------------
    st.subheader("Tabla de sensibilidad — ranking baseline vs alternativa (top 50)")
    st.caption(
        "Ordenada por `rank_baseline` (peor = 1). `rank_diff` > 0 indica que el "
        "distrito empeora en el ranking al pasar a la especificación alternativa."
    )
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
            st.image(str(p), caption=p.stem.replace("_", " ").title(), width="stretch")

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
            "coverage_index_baseline" if spec == "Baseline" else "coverage_index_alt"
        )
        top_n = st.slider("Top N a resaltar (peor atendidos)", 5, 50, 10)

    view = metrics.copy() if dpto == "(todos)" else metrics[metrics["departamento"] == dpto]

    # Métricas agregadas del filtro
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Distritos en vista", f"{len(view):,}")
    k2.metric("Sin IPRESS emergencia", f"{int((view['n_emergency_facilities'] == 0).sum()):,}")
    k3.metric("IPRESS totales", f"{int(view['n_facilities'].sum()):,}")
    k4.metric("Atenciones 2024 (M)", f"{view['total_atenciones'].sum() / 1e6:.2f}")

    st.divider()

    # Mini-mapa folium filtrado
    st.subheader(f"Mapa — cobertura {spec.lower()} (rojo = peor, verde = mejor)")
    view_wgs = view.to_crs("EPSG:4326").dropna(subset=[metric_col])
    if len(view_wgs) > 0:
        bounds = view_wgs.total_bounds  # minx, miny, maxx, maxy
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        fm = folium.Map(location=center, zoom_start=7 if dpto != "(todos)" else 5,
                        tiles="CartoDB positron")

        # Escala [0, 1]: rojo = peor, verde = mejor
        cmap = LinearColormap(["#c0392b", "#ffffbf", "#27ae60"], vmin=0.0, vmax=1.0,
                              caption="Índice de cobertura [0, 1]")
        cmap.add_to(fm)

        # Resaltar top-N peor atendidos (= menor cobertura)
        top_ubigeos = set(
            view_wgs.sort_values([metric_col, "n_ccpp"], ascending=[True, False])
            .head(top_n)["ubigeo"]
            .tolist()
        )

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
                aliases=["Distrito:", "Provincia:", "Dpto:", "Cobertura [0-1]:",
                         "IPRESS:", "Emergencia:", "Acceso 30km:"],
                localize=True,
            ),
        ).add_to(fm)

        st_folium(fm, use_container_width=True, height=550, returned_objects=[])
    else:
        st.info("No hay distritos con score en esta vista.")

    st.divider()
    st.subheader(f"Top {top_n} distritos peor atendidos — {spec}")
    st.caption(
        "Ordenados por menor cobertura; empates resueltos priorizando distritos con "
        "más centros poblados afectados."
    )
    cols_show = ["departamento", "provincia", "distrito", metric_col,
                 "n_facilities", "n_emergency_facilities", "total_atenciones",
                 "share_cp_within_30km", "mean_dist_km", "n_ccpp"]
    cols_show = [c for c in cols_show if c in view.columns]
    top_df = (
        view.dropna(subset=[metric_col])
        .sort_values([metric_col, "n_ccpp"], ascending=[True, False])
        .head(top_n)[cols_show]
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
