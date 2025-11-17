# Algoritmo Skatter - Estimación Geoestadística

Análisis geoestadístico utilizando el algoritmo SKATER (Spatial 'K'luster Analysis by Tree Edge Removal) para clustering espacial de datos geológicos.

## Descripción

Este proyecto implementa un análisis de clustering espacial utilizando:
- **SKATER**: Algoritmo de clustering espacial basado en árboles de expansión mínima
- **KNN**: Matriz de conectividad espacial mediante K-Nearest Neighbors
- **Clasificación**: Random Forest y KNN para clasificar puntos fantasma

## Estructura del Proyecto

```
.
├── main.py              # Script principal con el análisis completo
├── data/                # Datos de entrada
│   ├── bd_dm_cmp_entry.csv
│   └── 1_recpeso.xlsx
├── src espacial/        # Código fuente adicional
├── src2/                # Código fuente adicional
└── README.md
```

## Requisitos

Las dependencias principales incluyen:
- pandas
- geopandas
- spopt
- libpysal
- matplotlib
- scikit-learn
- numpy
- shapely

## Uso

Ejecutar el script principal:

```bash
python main.py
```

El análisis incluye:
1. Carga de datos desde CSV
2. Creación de geometrías 2D (X-Z)
3. Construcción de matriz de conectividad espacial (KNN)
4. Aplicación del algoritmo SKATER
5. Visualización de clusters
6. Clasificación de puntos fantasma

## Resultados

El script genera visualizaciones de:
- Distribución espacial de datos
- Clusters identificados por SKATER
- Clasificación de puntos fantasma mediante Random Forest y KNN

## Autor

Proyecto de tesina sobre estimación geoestadística.

