# Estrategia de branches

Todo el desarrollo sigue un flujo feature → develop → main. `main` solo recibe el PR final.

| Branch | Descripción | Tarea rúbrica |
| --- | --- | --- |
| `main` | Rama estable; solo recibe el merge final | — |
| `develop` | Integración de todas las features antes del merge a `main` | — |
| `feature/00-setup` | Scaffolding: estructura de carpetas, `.gitignore`, `requirements.txt`, README, skeletons de `src/` | — |
| `feature/t1-data-ingestion` | Descarga reproducible de los 4 datasets + limpieza a parquet/geoparquet | T1 (3 pts) |
| `feature/t2-geo-integration` | GeoDataFrames, reproyecciones, `sjoin` IPRESS↔distritos, distancia nearest-facility | T2 (3 pts) |
| `feature/t3-district-metrics` | Métricas baseline + alternativa + índice compuesto + sensibilidad | T3 (3 pts) |
| `feature/t4-static-viz` | Barplots, scatter, heatmap con matplotlib/seaborn | T4 (2 pts) |
| `feature/t5-geospatial-outputs` | Choropleth estático + folium interactivo + mapa bivariado | T5 (2 pts) |
| `feature/t6-streamlit-app` | App con los 4 tabs obligatorios | T6 (4 pts) |
| `feature/video-and-docs` | README final, methodology.md, placeholder del video | Video (3 pts) |
| `chore/qa-reproduce` | Verificación end-to-end antes del merge a main | — |
