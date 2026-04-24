# emergency_access_peru

Tarea 2 — Data Science. Análisis geoespacial de la desigualdad en el acceso a servicios de emergencia en salud entre distritos del Perú.

## Objetivo

Construir un pipeline reproducible que responda a estas preguntas a nivel distrital:

1. ¿Qué distritos tienen menor / mayor disponibilidad de establecimientos de salud y de atención de emergencias?
2. ¿Qué distritos muestran un acceso espacial más débil desde sus centros poblados hacia servicios de emergencia?
3. ¿Qué distritos están más / menos desatendidos combinando presencia, actividad y acceso?
4. ¿Qué tan sensibles son los resultados al cambiar la definición analítica de "acceso"?

## Datasets

| Dataset | Fuente |
| --- | --- |
| Centros Poblados | datosabiertos.gob.pe |
| Límites distritales (`DISTRITOS.shp`) | Repo del curso d2cml-ai/Data-Science-Python |
| Producción de Atención de Emergencias por IPRESS | datos.susalud.gob.pe |
| IPRESS MINSA (establecimientos de salud) | datosabiertos.gob.pe |

Los datos en bruto se descargan con `python -m src.ingest` a `data/raw/` (no versionados).

## Estructura

```
emergency_access_peru/
├── src/                    # Librería del proyecto
│   ├── config.py           # Rutas, CRS, URLs, umbrales
│   ├── utils.py            # Logger, normalización UBIGEO
│   ├── ingest.py           # T1 — descarga y limpieza
│   ├── geo_integration.py  # T2 — GeoDataFrames, joins, distancias
│   ├── metrics.py          # T3 — métricas baseline + alternativa
│   ├── viz_static.py       # T4 — plots matplotlib/seaborn
│   └── viz_geospatial.py   # T5 — choropleth + folium
├── app/
│   └── streamlit_app.py    # T6 — app con 4 tabs
├── data/
│   ├── raw/                # Descargado por ingest.py (gitignored)
│   └── processed/          # Parquet/GeoParquet limpios
├── outputs/
│   ├── figures/            # PNGs de T4
│   ├── maps/               # HTML y PNG de T5
│   └── tables/             # CSVs de ranking y sensibilidad
├── notebooks/              # EDA exploratorio
├── docs/
│   └── methodology.md      # Supuestos, definiciones, limitaciones
└── video/
    └── README.md           # Link al video explicativo (≤4 min)
```

## Setup

Requiere Python ≥ 3.11. Recomendado venv aislado:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Reproducción end-to-end

```bash
# 1. Descargar los 4 datasets a data/raw/
python -m src.ingest

# 2. Integración geoespacial (GeoDataFrames + sjoin + distancias)
python -m src.geo_integration

# 3. Métricas distritales (baseline + alternativa + índice compuesto)
python -m src.metrics

# 4. Visualizaciones estáticas
python -m src.viz_static

# 5. Mapas (choropleth + folium)
python -m src.viz_geospatial

# 6. App Streamlit
streamlit run app/streamlit_app.py
```

## Resultados clave

- **1,873** distritos analizados · **7,951** IPRESS con coordenadas válidas · **787** con actividad de emergencia en 2024.
- **78 %** de los distritos (1,457 / 1,873) **no cuentan** con ningún IPRESS que haya reportado actividad de emergencia en 2024.
- **Total de atenciones 2024:** ≈ 15.8 millones.
- **Distancia mediana** desde un centro poblado al IPRESS de emergencia más cercano: **22.8 km**; máxima: 377 km (Amazonía).
- **Índice de cobertura de emergencia en [0, 1]** — 1 = mejor atendido, 0 = peor. Se construye como promedio simple de 3 dimensiones normalizadas a [0, 1]: oferta (densidad de emergencia por km², log-normalizada), actividad (log-atenciones, normalizada) y acceso (share de CPs con emergencia ≤ 30 km).
- **Top distritos peor atendidos (cobertura ≈ 0):** Ayabaca (Piura), Pisacoma (Puno), Jacas Grande (Huánuco), Ocoyo (Huancavelica) — todos con cero oferta, cero actividad y cero acceso simultáneamente.
- **Mejor atendidos (cobertura ≈ 0.95):** Jesús María (0.97), Bellavista (0.96), Arequipa cercado (0.95), Lima cercado (0.92), Miraflores (0.90) — ningún distrito alcanza el 1 perfecto.
- **Sensibilidad al cambio de especificación:** Spearman ρ = **0.891** entre el ranking baseline (umbral 30 km, densidad por km²) y la alternativa (umbral 15 km, oferta por centro poblado). La conclusión cualitativa se mantiene, pero hay reordenamientos notables en el tramo medio del ranking.

## Video explicativo

Ver `video/README.md` para el link (≤4 min).

## Estrategia de branches

Todo el desarrollo ocurrió en branches `feature/*`, mergeadas a `develop` y de ahí un único PR a `main`. Ver `BRANCHES.md` para el detalle de cada branch.
