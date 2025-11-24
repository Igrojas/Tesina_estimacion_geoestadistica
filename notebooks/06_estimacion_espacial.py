#%%

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import importlib

warnings.filterwarnings('ignore')

# Importar módulos
from src import clustering, interpolacion, estimacion
from src.clustering import ClusterKmeans
from src.interpolacion import InterpoladorEspacial
from src.estimacion import EstimadorPorCluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def recargar_modulos():
    """Recarga los módulos personalizados para aplicar cambios sin reiniciar kernel"""
    importlib.reload(clustering)
    importlib.reload(interpolacion)
    importlib.reload(estimacion)
    # Re-importar las clases después de recargar
    globals()['ClusterKmeans'] = clustering.ClusterKmeans
    globals()['InterpoladorEspacial'] = interpolacion.InterpoladorEspacial
    globals()['EstimadorPorCluster'] = estimacion.EstimadorPorCluster
    print("✅ Módulos recargados correctamente")

print("✅ Imports exitosos")
print("💡 Tip: Si modificas clases en src/, ejecuta recargar_modulos() para aplicar cambios")
recargar_modulos()

df = pd.read_csv("data/raw/bd_dm_cmp_entry.csv", sep=";")
columnas = ["midx", "midy", "midz", "starkey_min"]
df = df[columnas].copy()

x = df['midx'].values
z = df['midz'].values
atributo = df['starkey_min'].values

print(f"📊 Datos cargados: {len(df)} puntos")

# ============================================================
# CELDA 3: Train/Test split
# ============================================================
indices = np.arange(len(x))
train_idx, test_idx = train_test_split(indices, test_size=0.2)

x_train, x_test = x[train_idx], x[test_idx]
z_train, z_test = z[train_idx], z[test_idx]
attr_train, attr_test = atributo[train_idx], atributo[test_idx]

# ============================================================
# CELDA 4: CLUSTERING (definir dominios)
# ============================================================
clusterer = ClusterKmeans(n_clusters=4, w_spatial=0.7)
clusterer.fit(x_train, z_train, attr_train)
clusters_train = clusterer.clusters

# Asignar clusters a test
# (Necesitas predecir clusters de test - esto falta en tu código actual)
# Por ahora, re-clusterizar con todos los datos
clusterer_full = ClusterKmeans(n_clusters=3, w_spatial=0.98)
clusterer_full.fit(x, z, atributo)
clusters = clusterer_full.clusters
clusters_train = clusters[train_idx]
clusters_test = clusters[test_idx]

print(f"Clusters definidos: {clusterer_full.n_clusters}")


# ============================================================
# CELDA 5: ESTIMACIÓN por cluster
# ============================================================
# Entrenar estimador
estimador = EstimadorPorCluster(n_neighbors=50)
estimador.fit(x_train, z_train, attr_train, clusters_train)

# Predecir en test
pred = estimador.predict(x_test, z_test, clusters_test)

# Métricas
mae = mean_absolute_error(attr_test, pred)
rmse = np.sqrt(mean_squared_error(attr_test, pred))
r2 = r2_score(attr_test, pred)

print(f"\nMétricas:")
print(f"  MAE:  {mae:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  R²:   {r2:.3f}")

# ============================================================
# CELDA 6: DELIMITAR + ESTIMAR en grilla
# ============================================================
# Re-entrenar con todos los datos
estimador_full = EstimadorPorCluster(n_neighbors=3)
estimador_full.fit(x, z, atributo, clusters)

# Delimitar zonas (interpolar clusters)
interpolador = InterpoladorEspacial(clusterer_full, n_neighbors=20, n_points=100)
interpolador.interpolar()

# Estimar en grilla respetando límites
xx, zz, valores = estimador_full.estimar_grilla(interpolador)

print(f"\nEstimación en grilla: {xx.shape}")

# ============================================================
# CELDA 7: Visualización mínima
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Clusters (dominios)

ax = axes[0]
ax.scatter(x, z, c=clusters, cmap='viridis', s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
ax.set_title('Dominios', fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Z')

# Panel 2: Límites interpolados
ax = axes[1]
ax.contourf(interpolador.xx, interpolador.zz, interpolador.clusters_interpolados,
           levels=np.arange(6+1)-0.5, cmap='viridis', alpha=0.5)
ax.scatter(x, z, c='black', s=10, alpha=0.5)
ax.set_title('Límites Delimitados', fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Z')

# Panel 3: Estimación dentro de límites
ax = axes[2]
contour = ax.contourf(xx, zz, valores, levels=20, cmap='RdYlBu_r')
ax.contour(interpolador.xx, interpolador.zz, interpolador.clusters_interpolados,
          levels=np.arange(6+1)-0.5, colors='black', linewidths=1.5, alpha=0.6)
ax.scatter(x, z, c='white', s=8, alpha=0.6, edgecolors='black', linewidth=0.3)
ax.set_title('Estimación por Dominio', fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.colorbar(contour, ax=ax, label='Estimado')

plt.tight_layout()
plt.show()

print("✅ Completado")

