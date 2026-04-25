# Diccionario de datos (`data/processed/`)

Documenta cada parquet/geoparquet producido por el pipeline de limpieza
(`python -m src.cleaning`) y por la integración geoespacial
(`python -m src.geospatial`). Cumple con el deliverable *data dictionary*
exigido por el rubric en la Tarea 1.

## Convenciones

- **UBIGEO**: código de 6 dígitos (DEP+PROV+DIST), zero-padded en string.
- **codigo**: identificador del establecimiento (IPRESS) o centro poblado, string sin padding.
- **CRS**: todas las geometrías en `EPSG:4326` (WGS84) excepto donde se indique.

---

## 1. `ipress.parquet` (T1 — IPRESS MINSA limpios)

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `codigo` | string | Código único del establecimiento (`CODIGO_UNICO` en el CSV fuente) |
| `nombre` | string | Nombre del establecimiento |
| `ubigeo` | string (6 chars) | UBIGEO declarado por el MINSA, zero-padded |
| `institucion` | string | Sector institucional (MINSA, ESSALUD, FFAA, Privado, etc.) |
| `clasificacion` | string | Clasificación del IPRESS (puesto, posta, hospital, etc.) |
| `categoria` | string | Categoría administrativa (I-1, I-2, II-E, …) |
| `lat` | float | Latitud en grados decimales (rango Perú: ~ −18.5 a 0.5) |
| `lon` | float | Longitud en grados decimales (rango Perú: ~ −81.5 a −68.0) |

**Origen**: `data/raw/IPRESS.csv` (datosabiertos.gob.pe).
**Filas**: 7 951 (de 20 819 originales — se descartaron filas sin coords válidas o con coords fuera de Perú).
**Quirk**: en el CSV fuente las columnas `NORTE` y `ESTE` están **invertidas** (NORTE = longitud, ESTE = latitud); `cleaning.clean_ipress` las remapea explícitamente.

---

## 2. `emergencies.parquet` (T1 — Producción de emergencias SUSALUD)

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `codigo` | string | Código del IPRESS (`CO_IPRESS` en el CSV fuente) |
| `ubigeo` | string (6 chars) | UBIGEO del IPRESS según SUSALUD |
| `atenciones` | float | Total de atenciones de emergencia agregadas anualmente para 2024 |
| `atendidos` | float | Total de pacientes atendidos agregados anualmente para 2024 |

**Origen**: `data/raw/emergencias_2024.csv` (datos.susalud.gob.pe).
**Filas**: 4 293 (IPRESS-año, agregando los 250 000 registros mensuales × sexo × edad).
**Quirk**: el marcador `NE_0001` en celdas con `atenciones < umbral de privacidad` se parsea como `NaN` y se cuenta como 0 en la agregación anual.
**Total nacional 2024**: ~15.8 M atenciones efectivas no anonimizadas.

---

## 3. `centros_poblados.parquet` (T1 — Centros poblados INEI)

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `ubigeo` | string (6 chars) | UBIGEO derivado de los primeros 6 dígitos del campo `CÓDIGO` |
| `codigo` | string | Código completo INEI del CP (10+ dígitos: DEP+PROV+DIST+CCPP) |
| `nombre` | string | Nombre del centro poblado |
| `categoria` | string | Categoría INEI (Centro Poblado Menor, etc.) |
| `poblacion` | int | **Peso uniforme = 1** (el shapefile no trae población) |
| `geometry` | Point | Geometría del CP en EPSG:4326 |

**Origen**: `data/raw/CCPP/*.shp` (extraído del ZIP `CCPP_0.zip` de datosabiertos.gob.pe).
**Filas**: 136 587 centros poblados.
**Quirk**: el shapefile no incluye UBIGEO ni población. UBIGEO se reconstruye de `CÓDIGO[:6]`; población se asume uniforme = 1 (ver `methodology.md` § Limitaciones).

---

## 4. `distritos.parquet` (T1 — Límites distritales)

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `ubigeo` | string (6 chars) | UBIGEO del distrito (`IDDIST` en el shapefile fuente) |
| `distrito` | string | Nombre del distrito |
| `provincia` | string | Provincia que contiene al distrito |
| `departamento` | string | Departamento que contiene al distrito |
| `geometry` | MultiPolygon | Polígono distrital en EPSG:4326 |

**Origen**: `data/raw/DISTRITOS.shp` (repo `d2cml-ai/Data-Science-Python`).
**Filas**: 1 873 distritos.

---

## 5. `ipress_geo.parquet` (T2 — IPRESS con actividad y UBIGEO geométrico)

Versión enriquecida de `ipress.parquet` después del *spatial join* y la integración con SUSALUD.

| Columna | Tipo | Descripción |
| --- | --- | --- |
| Todas las de `ipress.parquet` | — | Columnas heredadas |
| `geometry` | Point | Construida a partir de `lat`, `lon` |
| `atenciones` | int | Atenciones de emergencia 2024 (0 si el IPRESS no aparece en SUSALUD) |
| `is_emergency` | bool | True si `atenciones > 0` (definición de "IPRESS de emergencia") |
| `ubigeo_geom` | string | UBIGEO del distrito donde cae geográficamente vía sjoin |
| `ubigeo_final` | string | UBIGEO usado en agregación distrital (prefiere `ubigeo_geom`; fallback al declarado) |

**Filas**: 7 951.
**Total IPRESS de emergencia**: 787 (`is_emergency = True`).

---

## 6. `ccpp_with_distance.parquet` (T2 — CPs con distancia a emergencia más cercana)

| Columna | Tipo | Descripción |
| --- | --- | --- |
| Todas las de `centros_poblados.parquet` | — | Columnas heredadas |
| `dist_to_emergency_km` | float | Distancia euclidiana al IPRESS de emergencia más cercano (UTM 18S, en km) |
| `nearest_emergency_codigo` | string | `codigo` del IPRESS asignado por KDTree |

**Filas**: 136 587.
**Estadísticas nacionales**: mediana 22.8 km, media 30.0 km, máximo 377.2 km.

---

## 7. `district_integrated.parquet` (T2 — Agregados distritales)

Joining de IPRESS, emergencias y CPs al nivel distrital (1 873 filas).

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `ubigeo`, `distrito`, `provincia`, `departamento` | — | Heredadas de `distritos.parquet` |
| `geometry` | MultiPolygon | Polígono distrital |
| `n_facilities` | int | Total de IPRESS asignados al distrito (cualquier categoría) |
| `n_emergency_facilities` | int | IPRESS con `is_emergency = True` |
| `total_atenciones` | int | Suma de atenciones 2024 dentro del distrito |
| `n_ccpp` | int | Número de centros poblados en el distrito |
| `mean_dist_km`, `median_dist_km`, `max_dist_km` | float | Estadísticas de la distancia CP → emergencia |

---

## 8. `district_metrics.parquet` (T3 — Índice de cobertura)

Versión final con todas las métricas baseline y alternativa.

| Columna | Tipo | Descripción |
| --- | --- | --- |
| Todas las de `district_integrated.parquet` | — | — |
| `area_km2` | float | Área del distrito en km² (calculada en UTM 18S) |
| `n_facilities_per_100km2` | float | Densidad: IPRESS / 100 km² |
| `n_emergency_per_100km2` | float | Densidad: IPRESS de emergencia / 100 km² |
| `n_facilities_per_ccpp` | float | IPRESS / nº de CPs (alternativa) |
| `n_emergency_per_ccpp` | float | IPRESS de emergencia / nº de CPs (alternativa) |
| `share_cp_within_30km` | float ∈ [0, 1] | Fracción de CPs con emergencia ≤ 30 km (baseline) |
| `share_cp_within_15km` | float ∈ [0, 1] | Fracción de CPs con emergencia ≤ 15 km (alternativa) |
| `log_atenciones` | float | `ln(1 + total_atenciones)` |
| `supply_01_baseline` | float ∈ [0, 1] | Oferta normalizada (log-transform + min-max) |
| `activity_01` | float ∈ [0, 1] | Actividad normalizada (compartida baseline/alt) |
| `access_01_baseline` | float ∈ [0, 1] | Acceso normalizado (= `share_cp_within_30km`) |
| `coverage_index_baseline` | float ∈ [0, 1] | **Índice principal**: promedio de los 3 anteriores |
| `supply_01_alt` | float ∈ [0, 1] | Oferta alternativa (por CP, log-transform + min-max) |
| `activity_01_alt` | float ∈ [0, 1] | Actividad alternativa (= `activity_01`) |
| `access_01_alt` | float ∈ [0, 1] | Acceso alternativo (= `share_cp_within_15km`) |
| `coverage_index_alt` | float ∈ [0, 1] | Índice alternativo: promedio de los 3 anteriores |

**Sanity check**: ambos índices verifican `0 ≤ coverage ≤ 1` mediante un `assert` en `metrics.main()`.

---

## Tablas CSV en `output/tables/`

| Archivo | Descripción |
| --- | --- |
| `district_metrics_baseline.csv` | Snapshot de `district_metrics.parquet` con columnas baseline |
| `district_metrics_alternative.csv` | Snapshot con columnas de la especificación alternativa |
| `sensitivity_ranking.csv` | Comparación de rankings: rank baseline, rank alternativo, rank_diff por distrito |

**Spearman(rank baseline, rank alternativa)** = **0.891**.
