# Metodología

## Pregunta de investigación

¿En qué distritos del Perú el acceso a servicios de emergencia en salud es más desigual, considerando simultáneamente la presencia física de establecimientos, la actividad real de atención, y la distancia desde los centros poblados?

## Datasets

| # | Dataset | Cobertura temporal | Nivel de observación |
| --- | --- | --- | --- |
| 1 | IPRESS MINSA | Último snapshot publicado | Establecimiento de salud |
| 2 | Producción de emergencias SUSALUD | Series por período | Establecimiento × mes |
| 3 | Centros Poblados INEI | Último censo disponible | Centro poblado |
| 4 | DISTRITOS.shp | — | Distrito |

## Limpieza (T1)

- Normalización de UBIGEO a 6 dígitos, zero-padded.
- Deduplicación por código institucional de IPRESS.
- Filtro de coordenadas válidas (rango Perú: lat ∈ [-18.5, 0], lon ∈ [-81.5, -68.5]).
- Conversión a parquet / geoparquet en `data/processed/` para carga rápida.

## Integración geoespacial (T2)

- Sistemas de referencia: los datos se manejan en WGS84 (EPSG:4326) para visualización, y se reproyectan a UTM 18S (EPSG:32718) para cualquier cálculo de distancias en metros.
- `sjoin(op="within")` entre puntos IPRESS y polígonos distritales para asignar UBIGEO cuando el dato venga inconsistente.
- Distancia al IPRESS de emergencia más cercano desde cada centro poblado, con KDTree sobre coordenadas proyectadas.

## Métricas distritales (T3)

### Baseline
- `n_facilities_per_10k`: IPRESS totales / población × 10 000
- `mean_dist_cp_to_emergency_km`: promedio simple de la distancia CP → emergencia más cercana
- `emergency_activity`: suma de atenciones en el período de análisis
- `access_share_30km`: fracción de centros poblados con emergencia ≤ 30 km

### Alternativa (sensibilidad)
- Solo IPRESS con actividad > 0 (no solo existencia formal)
- Distancia **ponderada por población** del CP (pesos: población del CP)
- `access_share_15km`: umbral más exigente (15 km)

### Índice compuesto de subatención
Z-score de 3 dimensiones (supply, activity, access), promedio invertido → mayor valor = más desatendido.

## Outputs

- `outputs/tables/district_metrics_baseline.csv`
- `outputs/tables/district_metrics_alternative.csv`
- `outputs/tables/sensitivity_ranking.csv`
- `outputs/figures/*.png`
- `outputs/maps/*.html` (folium) + `*.png` (matplotlib)

## Limitaciones

- La producción de emergencias puede tener subregistro en IPRESS rurales.
- Los centros poblados tienen censo discreto; la población entre censos es interpolada y no siempre precisa.
- El umbral de 30 km es convencional; no captura tiempo real de viaje (topografía, vías).
