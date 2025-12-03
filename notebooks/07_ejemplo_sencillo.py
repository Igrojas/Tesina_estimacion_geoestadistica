#%%
# ============================================================
# PASO 1: IMPORTS Y CONFIGURACIÓN
# ============================================================
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src import clustering, estimacion, visualizacion, interpolacion
import importlib
importlib.reload(clustering)
importlib.reload(estimacion)
importlib.reload(visualizacion)
importlib.reload(interpolacion)

from src.clustering import ClusterKmeans
from src.estimacion import EstimadorEspacial
from src.visualizacion import VisualizadorClusters
from src.interpolacion import InterpoladorEspacial

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("="*70)
print("📚 EJEMPLO SENCILLO - Pipeline de Estimación Espacial")
print("="*70)

#%%
# ============================================================
# PASO 2: CARGA DE DATOS
# ============================================================
print("\n" + "="*70)
print("📂 PASO 1: CARGAR DATOS")
print("="*70)

# Cargar datos desde CSV
df = pd.read_csv("data/raw/bd_dm_cmp_entry.csv", sep=";")

# Seleccionar columnas relevantes
columnas = ["midx", "midy", "midz", "starkey_min"]
df = df[columnas].copy()

# Extraer arrays numpy
x = df['midx'].values      # Coordenada X (horizontal)
z = df['midz'].values      # Coordenada Z (profundidad/vertical)
atributo = df['starkey_min'].values  # Valor a predecir

print(f"\n✅ Datos cargados:")
print(f"   • Total de puntos: {len(df)}")
print(f"   • Rango X: [{x.min():.1f}, {x.max():.1f}]")
print(f"   • Rango Z: [{z.min():.1f}, {z.max():.1f}]")
print(f"   • Atributo (starkey_min):")
print(f"      - Media: {atributo.mean():.2f}")
print(f"      - Min: {atributo.min():.2f}, Max: {atributo.max():.2f}")

# Visualización rápida
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(x, z, c=atributo, cmap='RdYlBu_r', s=30, alpha=0.8)
ax.set_title('Distribución Espacial del Atributo', fontweight='bold')
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
plt.colorbar(scatter, ax=ax, label='starkey_min')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# ============================================================
# PASO 3: CLUSTERING (Agrupar datos en dominios)
# ============================================================
print("\n" + "="*70)
print("🔵 PASO 2: CLUSTERING ESPACIAL")
print("="*70)

print("\n📌 ¿Qué hace el clustering?")
print("   Agrupa los puntos en dominios homogéneos considerando:")
print("   • Ubicación espacial (X, Z)")
print("   • Valor del atributo (starkey_min)")

# Configurar parámetros
n_clusters = 5              # Número de grupos a crear
w_spatial = 0.8             # 65% peso espacial, 35% peso atributo

print(f"\n⚙️  Parámetros:")
print(f"   • Número de clusters: {n_clusters}")
print(f"   • Peso espacial: {w_spatial}")

# Crear y entrenar el clusterer
clusterer = ClusterKmeans(n_clusters=n_clusters, w_spatial=w_spatial)
clusterer.fit(x, z, atributo)

print("\n✅ Clustering completado")

# Ver estadísticas por cluster
stats = clusterer.get_stats()
print("\n📊 Estadísticas por cluster:")
for i, stat in stats.items():
    print(f"   Cluster {i}: {stat['n_points']} puntos, "
          f"media={stat['mean']:.2f}, std={stat['std']:.2f}")

visualizador = VisualizadorClusters()
# visualizador.plot_clusters(clusterer)
# visualizador.plot_atributo_real(clusterer)
# visualizador.plot_comparacion(clusterer)
visualizador.crear_dashboard(clusterer)

#%%

# Interpolar
interpolador = InterpoladorEspacial(clusterer,
                             n_neighbors=20,   
                             n_points=100)
interpolador.interpolar()
df_interpolado = interpolador.get_dataframe()
interpolador.print_info()

visualizador.plot_interpolacion(interpolador)
#%%
# Quiero los del cluster 1 o los del tipo original (por ejemplo, tipo='original')
cluster_1 = df_interpolado[(df_interpolado['cluster'] == 0) & (df_interpolado['tipo'] == 'original')]
print(cluster_1)

import random
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios':[2,1]})
ax_full, ax_zoom = axes

# --- GRÁFICO COMPLETO --

sc1 = ax_full.scatter(cluster_1['x'], cluster_1['z'], c=cluster_1['variable'], cmap='RdYlBu_r', s=30, alpha=0.8)
ax_full.set_title('Cluster 1 - Completo')
ax_full.set_xlabel('X')
ax_full.set_ylabel('Z')
plt.colorbar(sc1, ax=ax_full, label="Variable")
ax_full.grid(alpha=0.3)

# Elegimos un punto random dentro del cluster que NO sea un punto original
cluster_1_interp = df_interpolado[(df_interpolado['cluster'] == 0) & (df_interpolado['tipo'] != 'original')]
if not cluster_1_interp.empty:
    punto_random = cluster_1_interp.sample(1, random_state=random.randint(0, 10000)).iloc[0]
    x_c, z_c = punto_random['x'], punto_random['z']

    # Calcula distancias a todos los puntos originales en cluster 1
    dx = cluster_1['x'] - x_c
    dz = cluster_1['z'] - z_c
    distancias = np.sqrt(dx**2 + dz**2)

    # Creamos el DataFrame de distancias
    df_distancias = cluster_1.copy()
    df_distancias['dist_centro'] = distancias.values

    print("DataFrame con distancias al punto central:")
    print(df_distancias[['x', 'z', 'dist_centro']])

    # Zoom: Tomar los puntos originales más cercanos (por ejemplo, los 10 más cercanos)
    n_zoom = min(10, len(df_distancias))
    puntos_mas_cercanos = df_distancias.nsmallest(n_zoom, 'dist_centro')

    # --- MARCAMOS EN AMBOS PANELES ---
    # Punto random (centro)
    ax_full.scatter([x_c], [z_c], color='lime', s=120, edgecolor='k', linewidth=2, marker='*', label='Pto. centro (no original)')
    ax_zoom.scatter([x_c], [z_c], color='lime', s=120, edgecolor='k', linewidth=2, marker='*', label='Pto. centro (no original)')

    # Puntos más cercanos
    ax_full.scatter(puntos_mas_cercanos['x'], puntos_mas_cercanos['z'], facecolors='none', edgecolors='orange', s=90, linewidths=2, label=f'{n_zoom} p. más cercanos')
    ax_zoom.scatter(puntos_mas_cercanos['x'], puntos_mas_cercanos['z'], facecolors='none', edgecolors='orange', s=90, linewidths=2, label=f'{n_zoom} p. más cercanos')

    # Trazar líneas en el panel de zoom
    for _, row in puntos_mas_cercanos.iterrows():
        ax_zoom.plot([x_c, row['x']], [z_c, row['z']], color='gray', linestyle='-', linewidth=1.8, alpha=0.75, zorder=2)

    # ---- ZOOM PANEL -------
    ax_zoom.scatter(cluster_1['x'], cluster_1['z'], c=cluster_1['variable'], cmap='RdYlBu_r', s=25, alpha=0.5)
    ax_zoom.set_title('Cluster 1 - Zoom\nal punto central')
    ax_zoom.set_xlabel('X')
    ax_zoom.set_ylabel('Z')
    ax_zoom.grid(alpha=0.3)

    # Zoom automático al área alrededor del centro y sus puntos cercanos
    padding = max(puntos_mas_cercanos['dist_centro'].max(), 1)
    ax_zoom.set_xlim(x_c - padding, x_c + padding)
    ax_zoom.set_ylim(z_c - padding, z_c + padding)
    ax_zoom.legend(loc='best')
else:
    print("⚠️ No existen puntos interpolados en el cluster 1.")

plt.tight_layout()
plt.show()
# %%
