# Análisis Completo de Módulos del Proyecto

## 📚 Índice
1. [src/clustering.py](#1-srcclusteringpy)
2. [src/interpolacion.py](#2-srcinterpolacionpy)
3. [src/estimacion.py](#3-srcestimacionpy)
4. [src/visualizacion.py](#4-srcvisualizacionpy)

---

## 1. src/clustering.py

### Propósito
Agrupa datos espaciales en dominios homogéneos usando K-means ponderado. Combina información espacial (coordenadas X, Z) con el atributo de interés (starkey_min).

### Clase: `ClusterKmeans`

#### `__init__(n_clusters=4, w_spatial=0.0)`
**Entradas:**
- `n_clusters` (int): Número de grupos/clusters a crear (default: 4)
- `w_spatial` (float): Peso espacial entre 0 y 1 (default: 0.0)
  - `w_spatial=0.0` → Solo considera el atributo
  - `w_spatial=1.0` → Solo considera posición espacial
  - `w_spatial=0.65` → 65% posición, 35% atributo

**Qué hace:**
- Inicializa scalers para normalizar coordenadas y atributos
- Prepara estructura para almacenar resultados

#### `fit(x, z, attr)`
**Entradas:**
- `x` (array): Coordenadas X de los puntos
- `z` (array): Coordenadas Z de los puntos
- `attr` (array): Valores del atributo (ej: starkey_min)

**Qué hace:**
1. Guarda datos originales
2. Normaliza coordenadas (X, Z) y atributo por separado
3. Aplica pesos: `coords * w_spatial + attr * (1 - w_spatial)`
4. Ejecuta K-means sobre las features combinadas
5. Asigna cada punto a un cluster

**Retorna:** `self` (para encadenar métodos)

#### `get_stats()`
**Entradas:** Ninguna (usa datos internos)

**Qué hace:**
- Calcula estadísticas por cluster:
  - `n_points`: Cantidad de puntos
  - `mean`: Media del atributo
  - `std`: Desviación estándar
  - `efecto_proporcional`: CV = std/mean

**Retorna:** `dict` con estadísticas por cluster

#### `summary_plot()`
**Entradas:** Ninguna

**Qué hace:**
- Imprime resumen estadístico por cluster en consola

**Retorna:** `None`

#### `get_global_metrics()`
**Entradas:** Ninguna

**Qué hace:**
- Calcula métricas agregadas de todos los clusters:
  - Promedios, mínimos y máximos de std, CV, tamaño

**Retorna:** `dict` con métricas globales

---

## 2. src/interpolacion.py

### Propósito
Interpola los clusters en una grilla regular para delimitar dominios espaciales continuos. Usa KNN Classifier para predecir a qué cluster pertenece cada punto de la grilla.

### Clase: `InterpoladorEspacial`

#### `__init__(clusterer, n_neighbors=5, n_points=100)`
**Entradas:**
- `clusterer` (ClusterKmeans): Objeto ya entrenado con `fit()`
- `n_neighbors` (int): Vecinos para KNN (default: 5)
- `n_points` (int): Resolución de grilla (default: 100×100)

**Qué hace:**
- Valida que el clusterer esté entrenado
- Inicializa KNN Classifier y scaler

#### `crear_grid()`
**Entradas:** Ninguna (usa datos del clusterer)

**Qué hace:**
- Calcula rangos de X y Z con 2% de margen
- Crea arrays lineales con `n_points` valores

**Retorna:** `(x_range, z_range)` tupla de arrays

#### `interpolar()`
**Entradas:** Ninguna

**Qué hace:**
1. Crea grilla 2D con `np.meshgrid()`
2. Entrena KNN Classifier con puntos originales y sus clusters
3. Predice cluster para cada punto de la grilla
4. Guarda resultado en `self.clusters_interpolados`

**Retorna:** `self`

#### `get_info()`
**Entradas:** Ninguna

**Qué hace:**
- Retorna diccionario con información de la interpolación

**Retorna:** `dict` con métricas

#### `print_info()`
**Entradas:** Ninguna

**Qué hace:**
- Imprime información de la interpolación en consola

**Retorna:** `None`

---

## 3. src/estimacion.py

### Propósito
Estima valores del atributo en puntos nuevos usando métodos geoestadísticos. Contiene dos clases:
- `EstimadorEspacial`: Estimación global (sin considerar clusters)
- `EstimadorPorCluster`: Estimación independiente por cluster

### Clase: `EstimadorEspacial`

#### `__init__(metodo='knn', n_neighbors=10)`
**Entradas:**
- `metodo` (str): Método de estimación ('knn' por ahora)
- `n_neighbors` (int): Vecinos para KNN (default: 10)

**Qué hace:**
- Inicializa estimador KNN con ponderación por distancia

#### `fit(x, z, attr)`
**Entradas:**
- `x` (array): Coordenadas X
- `z` (array): Coordenadas Z
- `attr` (array): Valores del atributo

**Qué hace:**
1. Normaliza coordenadas
2. Entrena KNN Regressor con pesos por distancia

**Retorna:** `self`

#### `predict(x, z)`
**Entradas:**
- `x` (array): Coordenadas X nuevas
- `z` (array): Coordenadas Z nuevas

**Qué hace:**
- Predice valores del atributo usando KNN

**Retorna:** `array` de predicciones

### Clase: `EstimadorPorCluster`

#### `__init__(n_neighbors=10)`
**Entradas:**
- `n_neighbors` (int): Vecinos para KNN por cluster

**Qué hace:**
- Inicializa diccionarios para almacenar modelos por cluster

#### `fit(x, z, attr, clusters)`
**Entradas:**
- `x`, `z`, `attr`: Datos espaciales
- `clusters` (array): Asignación de cluster de cada punto

**Qué hace:**
- Entrena un modelo KNN independiente para cada cluster
- Ajusta `n_neighbors` si hay pocos puntos en un cluster

**Retorna:** `self`

#### `predict(x, z, clusters)`
**Entradas:**
- `x`, `z`: Coordenadas nuevas
- `clusters` (array): Cluster asignado a cada punto nuevo

**Qué hace:**
- Usa el modelo correspondiente a cada cluster para predecir

**Retorna:** `array` de predicciones

---

## 4. src/visualizacion.py

### Propósito
Genera visualizaciones profesionales de clusters, interpolaciones y comparaciones.

### Clase: `VisualizadorClusters`

#### `__init__(carpeta_salida='results/figures', estilo='seaborn-v0_8-darkgrid', dpi=150)`
**Entradas:**
- `carpeta_salida` (str): Carpeta para guardar figuras
- `estilo` (str): Estilo de matplotlib
- `dpi` (int): Resolución de imágenes

**Qué hace:**
- Crea carpeta de salida si no existe
- Configura estilos y paletas de colores

#### `plot_clusters(clusterer, ...)`
**Entradas:**
- `clusterer`: Objeto ClusterKmeans entrenado
- `titulo`, `guardar`, `nombre_archivo`, `mostrar`: Opciones de visualización

**Qué hace:**
- Crea scatter plot con colores discretos por cluster
- Agrega leyenda y métricas

**Retorna:** `(fig, ax)`

#### `plot_atributo_real(clusterer, ...)`
**Entradas:** Similar a `plot_clusters`

**Qué hace:**
- Visualiza el atributo original con colormap continuo

**Retorna:** `(fig, ax)`

#### `plot_comparacion(clusterer, ...)`
**Entradas:** Similar

**Qué hace:**
- Compara clusters vs atributo real lado a lado

**Retorna:** `(fig, axes)`

#### `plot_interpolacion(interpolador, ...)`
**Entradas:**
- `interpolador`: Objeto InterpoladorEspacial interpolado

**Qué hace:**
- Muestra contornos de clusters interpolados en grilla

**Retorna:** `(fig, ax)`

#### `crear_dashboard(clusterer, ...)`
**Entradas:** Similar

**Qué hace:**
- Crea panel 2×2 con:
  1. Efecto proporcional (CV)
  2. Probability plot lognormal
  3. Boxplots por cluster
  4. Mapa de clusters

**Retorna:** `fig`

---

## 🔗 Flujo de Trabajo Típico

```
1. ClusterKmeans.fit() → Agrupa datos
2. InterpoladorEspacial.interpolar() → Delimita dominios
3. EstimadorEspacial.fit() → Entrena modelo global
   O
   EstimadorPorCluster.fit() → Entrena modelos por cluster
4. predict() → Estima valores nuevos
5. VisualizadorClusters → Genera gráficos
```

---

## 📝 Notas Importantes

- **Normalización**: Todos los módulos normalizan coordenadas para evitar sesgos por escalas diferentes
- **Pesos espaciales**: `w_spatial` controla el balance entre posición y atributo en clustering
- **KNN**: Usa ponderación por distancia (vecinos cercanos pesan más)
- **Validación**: Todos los métodos validan que los objetos estén entrenados antes de usar

