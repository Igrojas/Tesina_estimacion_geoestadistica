#%%

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.clustering import ClusterKmeans
import importlib

VisualizadorClusters = importlib.reload(importlib.import_module('src.visualizacion')).VisualizadorClusters

print("✅ Imports exitosos")

df = pd.read_csv('data/raw/com_p_plt_entry 1.csv', sep=',')

columnas = ["midx", "midy", "midz", "cut"]
df = df[columnas].copy()
df = df.sample(frac=0.01)

# Eliminación de outliers para 'cut'
# Usamos el método IQR para mayor robustez frente a distribución sesgada
Q1 = df['cut'].quantile(0.25)
Q3 = df['cut'].quantile(0.75)
IQR = Q3 - Q1
filtro = (df['cut'] >= (Q1 - 1.5 * IQR)) & (df['cut'] <= (Q3 + 1.5 * IQR))
df = df[filtro]

x = df['midx'].values
y = df['midy'].values
z = df['midz'].values
atributo = df['cut'].values

print(f"📊 Datos cargados: {len(df)} puntos")

clusterer = ClusterKmeans(n_clusters=3, w_spatial=0.25)
clusterer.fit(x, y, z, atributo)

visualizador = VisualizadorClusters()
# visualizador.plot_clusters(clusterer, guardar=False)
# visualizador.plot_atributo_real(clusterer, guardar=False)
# visualizador.plot_comparacion(clusterer, guardar=False)
# visualizador.crear_dashboard(clusterer, guardar=False, nombre_atributo="cut")
df = visualizador.plot_clusters_convexhull_with_ghost(clusterer, n_ghost=5000, guardar=False)


# w_spatial_values = [0.25, 0.5, 0.75, 1.0]

# n_clusters_list = [3, 4, 5, 6]
# from itertools import product
# for n_clust, w_spatial in product(n_clusters_list, w_spatial_values):
#     clusterer = ClusterKmeans(n_clusters=n_clust, w_spatial=w_spatial)
#     clusterer.fit(x, y, z, atributo)
#     visualizador.crear_dashboard(clusterer, guardar=True, nombre_atributo="cut")

# %%
