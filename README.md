# emergency_access_peru

Tarea 2 — Data Science. Análisis geoespacial de la desigualdad en el acceso a servicios de emergencia en salud entre distritos del Perú.

---

## Índice

- [¿Qué hace este proyecto?](#qué-hace-este-proyecto)
- [¿Cuál es el objetivo analítico principal?](#cuál-es-el-objetivo-analítico-principal)
- [¿Qué datasets se utilizaron?](#qué-datasets-se-utilizaron)
- [¿Cómo se limpiaron los datos?](#cómo-se-limpiaron-los-datos)
- [¿Cómo se construyeron las métricas distritales?](#cómo-se-construyeron-las-métricas-distritales)
- [¿Cómo instalar las dependencias?](#cómo-instalar-las-dependencias)
- [¿Cómo ejecutar el pipeline?](#cómo-ejecutar-el-pipeline)
- [¿Cómo correr la app Streamlit?](#cómo-correr-la-app-streamlit)
- [Hallazgos principales](#hallazgos-principales)
- [Limitaciones](#limitaciones)
- [Estrategia de branches](#estrategia-de-branches)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Video explicativo](#video-explicativo)

---

## ¿Qué hace este proyecto?

Construye un **pipeline reproducible de análisis geoespacial** que mide la desigualdad en el acceso a servicios de emergencia en salud entre los **1 873 distritos del Perú**. Integra cuatro fuentes de datos abiertos (MINSA, SUSALUD, INEI y el shapefile distrital del curso), limpia y reconcilia los UBIGEOs, calcula la distancia de cada uno de los 136 587 centros poblados al IPRESS de emergencia más cercano, y produce:

- un **índice de cobertura de emergencia** por distrito en escala `[0, 1]` (1 = mejor atendido, 0 = peor), con dos especificaciones para medir sensibilidad;
- **6 figuras estáticas** en `output/figures/` (histograma bimodal con tabla integrada de extremos, ranking de peor acceso, descomposición del índice en 3 dimensiones, histograma de distancias, boxplot por departamento, scatter baseline-vs-alternativa);
- **3 choropleths** estáticos + **mapa bivariado** + **mapa folium interactivo** con los 787 IPRESS de emergencia clusterizados;
- una **app Streamlit con 4 tabs**: *Datos y Metodología*, *Análisis Estático*, *Resultados Geoespaciales* y *Exploración Interactiva*.

## ¿Cuál es el objetivo analítico principal?

Responder cuatro preguntas de investigación:

1. **Disponibilidad** — ¿qué distritos tienen menor/mayor disponibilidad de establecimientos de salud y de atención de emergencias?
2. **Acceso espacial** — ¿qué distritos muestran acceso más débil desde sus centros poblados hacia servicios de emergencia?
3. **Desatención combinada** — ¿qué distritos están más/menos desatendidos combinando presencia, actividad y acceso?
4. **Sensibilidad** — ¿qué tan sensibles son los resultados al cambiar la definición analítica de "acceso"?

La respuesta ejecutiva es un **índice distrital único** en `[0, 1]` que agrega las tres dimensiones (oferta, actividad, acceso) con pesos iguales, más una comparación contra una especificación alternativa (umbral 15 km en vez de 30 km; oferta por centro poblado en vez de por km²).

## ¿Qué datasets se utilizaron?

| Dataset | Fuente | Formato | Cobertura |
| --- | --- | --- | --- |
| IPRESS MINSA | datosabiertos.gob.pe | CSV (`latin-1`, sep `,`) | Directorio 2017 de establecimientos de salud — 20 819 filas, 7 951 con coordenadas |
| Producción de Atención de Emergencias por IPRESS | datos.susalud.gob.pe | CSV (`latin-1`, sep `;`) | Atenciones de emergencia 2024 — 250 000 filas agregables a 4 293 IPRESS-año |
| Centros Poblados INEI | datosabiertos.gob.pe | Shapefile en ZIP | 136 587 centros poblados nacionales |
| Límites distritales (`DISTRITOS.shp`) | Repo `d2cml-ai/Data-Science-Python` | Shapefile (5 archivos companion) | 1 873 polígonos distritales |

Todos se **descargan automáticamente** con `python -m src.cleaning` a `data/raw/` (no versionados). Los derivados limpios quedan en `data/processed/` como `.parquet` / geoparquet.

## ¿Cómo se limpiaron los datos?

El pipeline `src/ingest.py` aplica cuatro limpiezas por dataset, más decisiones globales:

**IPRESS MINSA**
- Encoding `latin-1`, separador `,`.
- **Fix crítico:** en el CSV fuente las columnas `NORTE` y `ESTE` están **invertidas** — `NORTE` contiene longitudes (rango −81..−68) y `ESTE` contiene latitudes (−18..0). El loader las remapea a `lon` y `lat`.
- Filtro de coordenadas dentro del rango continental del Perú.
- UBIGEO normalizado a 6 dígitos con zero-padding.
- Deduplicación por `CODIGO_UNICO`.
- **Resultado**: 7 951 IPRESS con coords válidas (de 20 819).

**Producción de emergencias SUSALUD 2024**
- Encoding `latin-1`, separador `;`.
- El marcador **`NE_0001`** (anonimización de SUSALUD cuando `atenciones < umbral de privacidad`) se parsea como `NaN` y se cuenta como 0 al agregar anualmente.
- Suma de atenciones por `(codigo, ubigeo)` → **4 293 IPRESS-año** con ~**15.8 M atenciones** efectivas no-anonimizadas.

**Centros Poblados INEI**
- Shapefile extraído del ZIP.
- **El shapefile no trae UBIGEO explícito**; se deriva de los primeros 6 dígitos del campo `CÓDIGO` (código INEI compuesto DEP+PROV+DIST+CCPP).
- No trae población del CP; se usa **peso uniforme (1)** en el índice.

**DISTRITOS.shp**
- UBIGEO desde `IDDIST`, normalizado a 6 dígitos.
- Reproyección a `EPSG:4326` para Folium; a `EPSG:32718` (UTM 18S) para cálculos de distancia en metros.

**Decisiones globales**
- **Spatial join `within`** IPRESS ↔ polígonos distritales → reasigna UBIGEOs declarados inconsistentes (**corrige 600 de los 7 951 IPRESS**).
- "IPRESS de emergencia" = `atenciones > 0` en SUSALUD 2024 (actividad efectiva, no categoría administrativa).
- KDTree sobre coordenadas UTM 18S para calcular distancia CP → IPRESS emergencia más cercano (**O(n log m)** en ~1 s para 136 587 CPs × 787 IPRESS).

Detalle completo en `docs/methodology.md` y en el tab *Datos y Metodología* de la app.

## ¿Cómo se construyeron las métricas distritales?

El indicador principal es un **índice de cobertura de emergencia** en escala `[0, 1]`, construido como promedio simple de **tres componentes normalizadas a [0, 1]**:

| Dimensión | Variable cruda | Normalización |
| --- | --- | --- |
| **Oferta** | `n_emergency_per_100km2` (densidad emergencia por 100 km²) | `log1p` → min-max |
| **Actividad** | `total_atenciones` (sumatoria anual) | `log1p` → min-max |
| **Acceso** | `share_cp_within_30km` (fracción de CPs con IPRESS ≤ 30 km) | ninguna (ya en [0, 1]) |

**Fórmula baseline**:

$$I^{\text{base}}_d \;=\; \frac{\widetilde{\text{oferta}}_d \;+\; \widetilde{\text{actividad}}_d \;+\; \text{acceso}^{30}_d}{3} \;\in\; [0, 1]$$

**Especificación alternativa** (sensibilidad): oferta por centro poblado (no por km²) + umbral de acceso estricto de 15 km.

**Interpretación:**
- `I = 0` → distrito en el piso simultáneamente en oferta, actividad y acceso.
- `I = 1` → distrito es el máximo observado en las 3 dimensiones a la vez (inalcanzable en la práctica).

**Sensibilidad:** Spearman entre los rankings baseline y alternativo = **0.891**.

## ¿Cómo instalar las dependencias?

**Requisitos:** Python ≥ 3.11 (probado con 3.14.3 usando `pyogrio` como motor geo en lugar de `fiona` por ausencia de wheel de GDAL para 3.14).

```bash
cd <ruta-al-repo>
python -m venv .venv

# Windows CMD
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **Nota Windows PowerShell:** si al activar aparece `la ejecución de scripts está deshabilitada`, usa el binario directo del venv sin activar (`.venv\Scripts\streamlit.exe ...`) o habilítalo permanentemente con `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

## ¿Cómo ejecutar el pipeline?

```bash
python -m src.data_loader         # T1.a: solo descarga los 4 datasets a data/raw/ (opcional)
python -m src.cleaning            # T1: descarga (cacheada) + limpieza + parquets en data/processed/
python -m src.geospatial          # T2: GeoDataFrames + sjoin + distancias KDTree
python -m src.metrics             # T3: índice de cobertura [0, 1] + sensibilidad
python -m src.visualization       # T4 + T5: figuras en output/figures/ y mapas en output/maps/
```

El pipeline es **idempotente y cacheado**: `data_loader.download_raw()` no re-descarga si los archivos ya existen en `data/raw/`. Los outputs intermedios en `data/processed/` se regeneran deterministamente.

## ¿Cómo correr la app Streamlit?

**Windows (PowerShell)**:
```powershell
cd C:\Users\manue\OneDrive\Documentos\GitHub\emergency_access_peru
.venv\Scripts\streamlit.exe run app.py
```

**Windows (CMD) / macOS / Linux**:
```bash
streamlit run app.py
```

Se abre automáticamente en `http://localhost:8501`. La app tiene **cuatro tabs**:

1. **📚 Datos y Metodología** — contexto, 4 preguntas comentadas, datasets, resumen de limpieza, 12 decisiones metodológicas, fórmulas LaTeX, quirks, limitaciones y números clave.
2. **📊 Análisis Estático** — índice de visualizaciones + 6 figuras con narrativa interpretativa debajo de cada una + tabla de sensibilidad top 50.
3. **🗺️ Resultados Geoespaciales** — 4 choropleths + mapa folium interactivo embebido.
4. **🔍 Exploración Interactiva** — filtros por departamento + toggle baseline/alt + slider top-N; mapa folium **cacheado** (primer render 1–3 s; cambios instantáneos) y tabla de distritos peor atendidos.

## Hallazgos principales

- **78 % de los 1 873 distritos (1 457)** **no tienen IPRESS con actividad de emergencia** reportada en 2024. Dependen de establecimientos vecinos para cualquier cobertura.
- **Distribución bimodal del índice**: 479 distritos en `x = 0` (aislados totales) + 632 en `x = 1/3` (sin emergencia propia pero con acceso vecinal pleno) = **59 % del país concentrado en dos valores exactos**.
- **Mediana de distancia CP → IPRESS de emergencia más cercano: 22.8 km**; media 30 km; **máximo 377 km** (Amazonía profunda).
- **Peor atendidos (cobertura = 0)**: Ayabaca (Piura), Pisacoma (Puno), Jacas Grande (Huánuco), Ocoyo (Huancavelica), Raimondi (Ucayali).
- **Mejor atendidos**: Jesús María (Lima, **0.966**), Bellavista (Callao, 0.959), Arequipa cercado (**0.946**), Lima cercado (0.923), Miraflores (0.897). **Ninguno alcanza el 1 perfecto**.
- **Sensibilidad al cambio de especificación**: Spearman ρ = **0.891** entre rankings baseline (30 km, densidad/km²) y alternativo (15 km, oferta/CP). Las conclusiones cualitativas se mantienen; hay reordenamientos notables en el tramo medio del ranking.
- **Alta varianza intra-departamental**: incluso Lima tiene distritos con baja cobertura y departamentos amazónicos tienen excepciones con cobertura decente. **Las intervenciones de política deben ser distritales, no por bloques departamentales**.

## Limitaciones

Declaradas explícitamente antes de usar los resultados para política pública:

- **Subregistro SUSALUD en IPRESS rurales.** La actividad medida está sesgada hacia centros urbanos con registro más robusto.
- **Peso uniforme por centro poblado (1).** Sin integración con datos censales, un caserío de 5 personas pesa igual que un pueblo de 5 000.
- **Distancia euclidiana, no de viaje.** Un CP a 20 km en línea recta puede estar a 4 horas por trocha; la euclidiana subestima severamente el tiempo real en sierra y selva.
- **Umbrales 15/30 km convencionales**, no provienen de un estándar técnico formal peruano.
- **Snapshot IPRESS 2017 vs actividad SUSALUD 2024.** Establecimientos creados o cerrados después de 2017 no aparecen en el directorio.
- **38 % de IPRESS sin coordenadas** (12 868 de 20 819). Se descartan para análisis espacial — si esa fracción no es aleatoria respecto a ruralidad, hay sesgo en la densidad distrital.
- **Cobertura INEI de centros poblados.** Asentamientos informales recientes o caseríos no censados no aparecen.
- **Índice compuesto = promedio simple de 3 componentes.** No hay justificación técnica para pesos iguales; pesos distintos cambiarían el ranking.
- **Normalización min-max dependiente de la muestra.** El máximo observado define el "1"; un outlier nuevo re-escalaría a todos.
- **Pileups estructurales** en `x = 0` y `x = 1/3` (~59 % de distritos). Es consecuencia del diseño del índice + floor de 1 457 distritos sin emergencia propia; no es artefacto pero sí una limitación de granularidad en ese régimen.
- **Un solo año (2024).** No capturamos tendencias ni shocks temporales (post-pandemia, feriados, eventos climáticos).

## Estrategia de branches

Todo el desarrollo siguió un flujo **`feature/*` → `develop` → `main`**. La rama `main` solo recibe el PR final con la submission completa; **nunca se commitea directamente en ella** (evita la penalidad de −2 del rubric). Las feature branches se mergearon a `develop` con `--no-ff` para preservar la estructura en el log.

| Branch | Tarea | Descripción |
| --- | --- | --- |
| `main` | — | Rama estable; solo recibe el merge final vía PR |
| `develop` | — | Integración de todas las features antes del merge a `main` |
| `feature/00-setup` | — | Scaffolding: carpetas, `.gitignore`, `requirements.txt`, README, skeletons de `src/` |
| `feature/t1-data-ingestion` | T1 (3 pts) | Descarga reproducible + limpieza + persistencia a parquet de los 4 datasets |
| `feature/t2-geo-integration` | T2 (3 pts) | GeoDataFrames, reproyecciones, `sjoin` IPRESS↔distritos, distancia nearest-facility via KDTree |
| `feature/t3-district-metrics` | T3 (3 pts) | Métricas baseline + alternativa + índice compuesto + sensibilidad |
| `feature/t4-static-viz` | T4 (2 pts) | Figuras con matplotlib/seaborn |
| `feature/t5-geospatial-outputs` | T5 (2 pts) | 3 choropleths + mapa bivariado + folium interactivo |
| `feature/t6-streamlit-app` | T6 (4 pts) | App Streamlit con los 4 tabs obligatorios |
| `feature/video-and-docs` | Video (3 pts) | README inicial + methodology.md + placeholder del video |
| `chore/qa-reproduce` | — | Verificación end-to-end antes del merge a main |
| `feature/streamlit-docs-expansion` | — | Traducción al español de los 4 tabs + expansión del tab *Datos y Metodología* con fórmulas LaTeX |
| `feature/static-analysis-polish` | — | Fix de deprecation `use_column_width` + índice de visualizaciones + análisis narrativo bajo cada gráfico |
| `feature/coverage-index-01-scale` | — | Refactor del índice de subatención (z-scores invertidos) al **índice de cobertura `[0, 1]`** (1 = mejor, 0 = peor) vía min-max con log-transform |

Los cambios incrementales posteriores al último merge (chart 1 como histograma superpuesto, documentación de los pileups estructurales en D12, fix del freeze en el tab *Exploración Interactiva*, rediseño de charts 2 y 3 para responder Q2 y Q3 directamente, tabla integrada dentro del chart 1, y esta reescritura del README) fueron **commits directos sobre `develop`**, sin feature branches adicionales — por instrucción explícita del usuario.

## Estructura del proyecto

```
emergency_access_peru/
├── app.py                        # T6 — Streamlit app con los 4 tabs (raíz)
├── README.md                     # Este archivo
├── requirements.txt              # Dependencias Python pinneadas
├── src/                          # Librería del proyecto
│   ├── config.py                 # Rutas, CRS, URLs, umbrales
│   ├── utils.py                  # Logger, normalización UBIGEO, helpers
│   ├── data_loader.py            # T1.a — descarga + lectores crudos
│   ├── cleaning.py               # T1.b — limpieza + persistencia a parquet
│   ├── geospatial.py             # T2 — GeoDataFrames, sjoin, distancias KDTree
│   ├── metrics.py                # T3 — índice de cobertura [0, 1] + sensibilidad
│   └── visualization.py          # T4 + T5 — figuras matplotlib + choropleths + folium
├── data/
│   ├── raw/                      # Descargado por data_loader (gitignored)
│   └── processed/                # Parquet/GeoParquet limpios
├── output/
│   ├── figures/                  # PNGs de T4 (6 figuras)
│   ├── maps/                     # PNG + HTML de T5
│   └── tables/                   # CSVs de métricas baseline/alt + sensibilidad
├── docs/
│   ├── methodology.md            # Supuestos, decisiones, limitaciones detalladas
│   └── data_dictionary.md        # Diccionario de las columnas de cada parquet
├── notebooks/                    # EDA exploratorio
└── video/
    └── link.txt                  # URL del video explicativo (≤ 4 min)
```

## Video explicativo

Ver `video/link.txt` para el URL del video explicativo (≤ 4 minutos).
