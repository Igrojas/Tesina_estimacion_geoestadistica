#%%
"""
============================================================
EJEMPLO SENCILLO: Uso de todos los módulos de src
============================================================

Este script muestra cómo usar:
1. ClusterKmeans (clustering.py)
2. InterpoladorEspacial (interpolacion.py)
3. EstimadorEspacial (estimacion.py)
4. VisualizadorClusters (visualizacion.py)
"""
# ============================================================
# IMPORTS
# ============================================================
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar y recargar todos los módulos de src
import importlib

from src import clustering, interpolacion, estimacion, visualizacion

importlib.reload(clustering)
importlib.reload(interpolacion)
importlib.reload(estimacion)
importlib.reload(visualizacion)

from src.clustering import ClusterKmeans
from src.interpolacion import InterpoladorEspacial
from src.estimacion import EstimadorEspacial
from src.visualizacion import VisualizadorClusters
#%%
# ============================================================
# PASO 1: CARGAR DATOS
# ============================================================
print("="*70)
print("PASO 1: CARGAR DATOS")
print("="*70)

df = pd.read_csv("data/raw/bd_dm_cmp_entry.csv", sep=";")
columnas = ["midx", "midy", "midz", "starkey_min"]
df = df[columnas].copy()

x = df['midx'].values
z = df['midz'].values
atributo = df['starkey_min'].values

print(f"Datos cargados: {len(df)} puntos")

#%%
# ============================================================
# PASO 2: CLUSTERING (clustering.py)
# ============================================================
print("\n" + "="*70)
print("PASO 2: CLUSTERING - ClusterKmeans")
print("="*70)

# Crear clusterer
clusterer = ClusterKmeans(n_clusters=5, w_spatial=0.8)

# Entrenar
clusterer.fit(x, z, atributo)

# Ver estadísticas
stats = clusterer.get_stats()
print("\nEstadísticas por cluster:")
for i, stat in stats.items():
    print(f"  Cluster {i}: {stat['n_points']} puntos, media={stat['mean']:.2f}")

#%%
# ============================================================
# PASO 3: INTERPOLACIÓN (interpolacion.py)
# ============================================================
print("\n" + "="*70)
print("PASO 3: INTERPOLACIÓN - InterpoladorEspacial")
print("="*70)

# Crear interpolador
interpolador = InterpoladorEspacial(
    clusterer=clusterer,
    n_neighbors=10,
    n_points=100
)

# Interpolar
interpolador.interpolar()

# Ver información
interpolador.print_info()

#%%
# ============================================================
# PASO 4: ESTIMACIÓN (estimacion.py)
# ============================================================
print("\n" + "="*70)
print("PASO 4: ESTIMACIÓN - EstimadorEspacial")
print("="*70)

# Crear estimador
estimador = EstimadorEspacial(metodo='knn', n_neighbors=10)

# Entrenar
estimador.fit(x, z, atributo)

# Predecir en algunos puntos nuevos (ejemplo: primeros 10 puntos)
x_nuevos = x[:10]
z_nuevos = z[:10]
predicciones = estimador.predict(x_nuevos, z_nuevos)

print(f"\nPredicciones para 10 puntos:")
print(f"  Reales:    {atributo[:10]}")
print(f"  Predichos: {predicciones}")

#%%
# ============================================================
# PASO 5: VISUALIZACIÓN (visualizacion.py)
# ============================================================
print("\n" + "="*70)
print("PASO 5: VISUALIZACIÓN - VisualizadorClusters")
print("="*70)

# Crear visualizador
viz = VisualizadorClusters(carpeta_salida='results/figures')

# Visualizar clusters
print("\nGenerando visualización de clusters...")
viz.plot_clusters(clusterer, mostrar=True, guardar=False)

# Visualizar interpolación
print("\nGenerando visualización de interpolación...")
viz.plot_interpolacion(interpolador, mostrar=True, guardar=False)

# Comparación
print("\nGenerando comparación...")
viz.plot_comparacion(clusterer, mostrar=True, guardar=False)

#%%
# ============================================================
# RESUMEN
# ============================================================
print("\n" + "="*70)
print("RESUMEN")
print("="*70)
print("""
✅ Todos los módulos utilizados:

1. ClusterKmeans (clustering.py)
   - fit(x, z, attr) → Agrupa datos en clusters
   - get_stats() → Obtiene estadísticas

2. InterpoladorEspacial (interpolacion.py)
   - interpolar() → Interpola clusters en grilla
   - print_info() → Muestra información

3. EstimadorEspacial (estimacion.py)
   - fit(x, z, attr) → Entrena modelo KNN
   - predict(x, z) → Predice valores

4. VisualizadorClusters (visualizacion.py)
   - plot_clusters() → Visualiza clusters
   - plot_interpolacion() → Visualiza interpolación
   - plot_comparacion() → Compara resultados
""")

print("="*70)
print("✅ EJEMPLO COMPLETADO")
print("="*70)

