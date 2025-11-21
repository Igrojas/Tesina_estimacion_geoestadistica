#%%

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.clustering import ClusterKmeans
from src.visualizacion import VisualizadorClusters  # 🆕 NUEVO

print("✅ Imports exitosos")
# %%
df = pd.read_csv("data/raw/bd_dm_cmp_entry.csv", sep=";")
columnas = ["midx", "midy", "midz", "starkey_min"]
df = df[columnas].copy()

x = df['midx'].values
z = df['midz'].values
atributo = df['starkey_min'].values

print(f"📊 Datos cargados: {len(df)} puntos")
# %%
