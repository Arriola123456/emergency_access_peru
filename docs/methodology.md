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
- Deduplicación de IPRESS por `CODIGO_UNICO`.
- Filtro de coordenadas válidas (rango Perú: lat ∈ [-18.5, 0.5], lon ∈ [-81.5, -68.0]).
- Conversión a parquet / geoparquet en `data/processed/` para carga rápida.

### Quirks del dato fuente (documentados en el pipeline)

1. **IPRESS MINSA — coordenadas NORTE/ESTE invertidas.** En el CSV fuente, la columna llamada `NORTE` contiene longitudes (rango -81..-68) y `ESTE` contiene latitudes (-18..0). El loader las remapea explícitamente (`NORTE → lon`, `ESTE → lat`). Solo ~38 % de los 20 819 establecimientos trae coordenadas válidas; los demás se descartan para análisis espacial.

2. **SUSALUD — anonimización con `NE_0001`.** Las columnas `NRO_TOTAL_ATENCIONES` y `NRO_TOTAL_ATENDIDOS` tienen el marcador `NE_0001` cuando la celda es < umbral de privacidad (publicación agregada por sexo-edad). Estos se parsean como NaN y se cuentan como 0 al agregar anualmente — la agregación sigue siendo informativa para la métrica distrital pero subestima modestamente los IPRESS pequeños.

3. **Centros Poblados INEI — sin UBIGEO ni población.** El shapefile trae un `CÓDIGO` compuesto (10+ dígitos DEP+PROV+DIST+CCPP); el UBIGEO distrital se deriva de los primeros 6 dígitos. No trae población del CP; se usa **peso uniforme (1)** en el baseline. Una extensión con población censal (INEI CPV 2017) pasaría a especificación alternativa.

4. **Separadores heterogéneos.** IPRESS usa `,`, SUSALUD usa `;`, ambos en encoding `latin-1`. El pipeline lo hard-codea por archivo.

## Integración geoespacial (T2)

- Sistemas de referencia: los datos se manejan en WGS84 (EPSG:4326) para visualización, y se reproyectan a UTM 18S (EPSG:32718) para cualquier cálculo de distancias en metros.
- `sjoin(op="within")` entre puntos IPRESS y polígonos distritales para asignar UBIGEO cuando el dato venga inconsistente.
- Distancia al IPRESS de emergencia más cercano desde cada centro poblado, con KDTree sobre coordenadas proyectadas.

## Métricas distritales (T3)

### Índice de cobertura de emergencia — escala [0, 1]

Se construye **un único escalar por distrito** a partir de tres dimensiones.
Cada dimensión se normaliza a `[0, 1]` con min-max (con `log1p` previo para las
dimensiones sesgadas) y se promedia. Interpretación directa:

- **`I_d = 0`** → el distrito está en el piso simultáneamente en oferta, actividad
  y acceso (peor atendido).
- **`I_d = 1`** → el distrito es el máximo observado en las tres dimensiones a la
  vez (cota superior, inalcanzable en la práctica).

### Especificación baseline
- Oferta: `n_emergency_per_100km2` (IPRESS con emergencia por 100 km²), log-normalizada.
- Actividad: `log(1 + total_atenciones)`, normalizada.
- Acceso: `share_cp_within_30km` (fracción de CPs con IPRESS de emergencia ≤ 30 km). Ya vive en [0, 1], no requiere transformación.

$$I^{\text{base}}_d = \frac{\widetilde{\text{oferta}}_d + \widetilde{\text{actividad}}_d + \widetilde{\text{acceso}}^{30}_d}{3}$$

### Especificación alternativa (sensibilidad)
- Oferta: `n_emergency_per_ccpp` (IPRESS con emergencia por centro poblado), log-normalizada.
- Actividad: igual que baseline.
- Acceso: `share_cp_within_15km` (umbral más estricto).

$$I^{\text{alt}}_d = \frac{\widetilde{\text{oferta}}^{\text{alt}}_d + \widetilde{\text{actividad}}_d + \widetilde{\text{acceso}}^{15}_d}{3}$$

### Normalización min-max con log previo

Para una variable sesgada $X$:

$$\widetilde{X}_d = \frac{\ln(1 + X_d) - \min_d \ln(1 + X_d)}{\max_d \ln(1 + X_d) - \min_d \ln(1 + X_d)} \in [0, 1]$$

### Sensibilidad

Correlación de Spearman entre los dos rankings (rank 1 = peor atendido = menor coverage):

$$\rho_S = \text{Spearman}(\text{rank}(I^{\text{base}}), \text{rank}(I^{\text{alt}})) = 0.891$$

## Outputs

- `outputs/tables/district_metrics_baseline.csv`
- `outputs/tables/district_metrics_alternative.csv`
- `outputs/tables/sensitivity_ranking.csv`
- `outputs/figures/*.png`
- `outputs/maps/*.html` (folium) + `*.png` (matplotlib)

## Limitaciones

- La producción de emergencias puede tener subregistro en IPRESS rurales; la actividad medida sesga hacia centros urbanos con mejor registro.
- Los centros poblados no traen población del INEI en el shapefile público usado; se usa peso uniforme (1 por CP).
- El umbral de 30 km (baseline) y 15 km (alternativa) es convencional; no captura tiempo real de viaje (topografía, vías, estado de trochas).
- Distancia euclidiana en UTM, no de viaje.
- Normalización min-max dependiente de la muestra: el máximo observado define el "1" de la escala. Incorporar o excluir distritos podría mover la escala para todos.
- Índice compuesto con **pesos iguales** (1/3 por dimensión) — elección de simplicidad, no de principio técnico.
- Un solo año (2024), sin capturar tendencias ni shocks temporales.
