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

df = pd.read_csv("../data/raw/bd_dm_cmp_entry.csv", sep=";")

columnas = ["midx", "midy", "midz", "starkey_min", "bwi_kwh_tc"]
df = df[columnas].copy()

x = df['midx'].values
z = df['midz'].values
atributo = df['bwi_kwh_tc'].values

print(f"📊 Datos cargados: {len(df)} puntos")

clusterer = ClusterKmeans(n_clusters=4, w_spatial=0.65)
clusterer.fit(x, z, atributo)

visualizador = VisualizadorClusters()
# visualizador.plot_clusters(clusterer)
# visualizador.plot_atributo_real(clusterer)
# visualizador.plot_comparacion(clusterer)
# visualizador.crear_dashboard(clusterer, guardar=False)



w_spatial_values = [0.25, 0.5, 0.75, 1.0]
n_clusters_list = [3, 4, 5, 6]
from itertools import product
for n_clust, w_spatial in product(n_clusters_list, w_spatial_values):
    clusterer = ClusterKmeans(n_clusters=n_clust, w_spatial=w_spatial)
    clusterer.fit(x, z, atributo)
    visualizador.crear_dashboard(clusterer, guardar=True)
