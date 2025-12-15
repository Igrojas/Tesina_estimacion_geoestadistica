#%%
import sys
sys.path.append('../')  # Para importar desde src/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import importlib
ClusterKmeans = importlib.reload(importlib.import_module('src.clustering')).ClusterKmeans

df = pd.read_csv('data/raw/com_p_plt_entry 1.csv', sep=',')
# Filtrar columnas
midx = "midx"
midy = "midy"
midz = "midz"
cut  = "cus"

df = df[[midx, midy, midz, cut]].copy()
df = df.sample(frac=0.01) 

# Extraer coordenadas y atributo
x = df[midx].values
y = df[midy].values
z = df[midz].values
atributo = df[cut].values

print(f"📊 Datos cargados: {len(df)} puntos")
print(f"📏 Rango X: [{x.min():.0f}, {x.max():.0f}]")
print(f"📏 Rango Y: [{y.min():.0f}, {y.max():.0f}]")
print(f"📏 Rango Z: [{z.min():.0f}, {z.max():.0f}]")
print(f"📊 Rango atributo: [{atributo.min():.2f}, {atributo.max():.2f}]")

# ============================================================
# CELDA 3: Crear objeto (sin entrenar)
# ============================================================
clusterer = ClusterKmeans(n_clusters=7, w_spatial=0.8)

# Ver objeto sin entrenar
print(clusterer)

# ============================================================
# CELDA 4: Entrenar
# ============================================================
clusterer.fit(x, y, z, atributo)

# Ver objeto entrenado
print(clusterer)

# ============================================================
# CELDA 5: Obtener resultados
# ============================================================
# Opción 1: Acceder directamente
clusters = clusterer.clusters
print(f"Clusters asignados: {clusters[:20]}...")

# Opción 2: Usar método
clusterer.summary_plot()

# ============================================================
# CELDA 6: Visualización básica
# ============================================================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

# Subplot 1: Clusters en 3D
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
sc1 = ax1.scatter(x, y, z, c=clusters, cmap='viridis',
                  s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.7, pad=0.1, label='Cluster')
ax1.set_title(f'Clustering K-means 3D (k={clusterer.n_clusters})')
ax1.set_xlabel('X (midx)')
ax1.set_ylabel('Y (midy)')
ax1.set_zlabel('Z (midz)')
ax1.grid(alpha=0.3)
ax1.view_init(elev=20, azim=40)

# Subplot 2: Atributo real en 3D
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc2 = ax2.scatter(x, y, z, c=atributo, cmap='RdYlBu_r',
                  s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.7, pad=0.1, label='starkey_min')
ax2.set_title('Atributo Real 3D')
ax2.set_xlabel('X (midx)')
ax2.set_ylabel('Y (midy)')
ax2.set_zlabel('Z (midz)')
ax2.grid(alpha=0.3)
ax2.view_init(elev=20, azim=40)

plt.tight_layout()
plt.show()

# ============================================================
# CELDA 7: Estadísticas detalladas
# ============================================================
stats = clusterer.get_stats()

# Convertir a DataFrame para mejor visualización
import pandas as pd
df_stats = pd.DataFrame(stats).T
df_stats.index.name = 'Cluster'

# print("\n📊 TABLA DE ESTADÍSTICAS")
# print(df_stats)

# Guardar en Excel
# df_stats.to_excel('../results/tables/01_estadisticas_clusters.xlsx')
# print("\n✅ Tabla guardada en: results/tables/01_estadisticas_clusters.xlsx")

# ============================================================
# CELDA 8: Probar diferentes valores de k
# ============================================================
#%%

valores_k = [3, 4, 5, 6, 7, 8]

fig = plt.figure(figsize=(18, 10))

for idx, k in enumerate(valores_k):
    # Crear y entrenar
    clust = ClusterKmeans(n_clusters=k, w_spatial=0.9)
    clust.fit(x, y, z, atributo)
    
    # Visualizar
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    ax.scatter(x, y, z, c=clust.clusters, cmap='viridis',
               s=30, alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.set_title(f'k = {k}', fontweight='bold', fontsize=14)
    ax.set_xlabel('X (midx)')
    ax.set_ylabel('Y (midy)')
    ax.set_zlabel('Z (midz)')
    ax.grid(alpha=0.3)
    
    # Agregar métrica
    stats = clust.get_stats()
    std_prom = np.mean([s['mean'] for s in stats.values()])
    ax.text2D(0.02, 0.98, f'Mean prom: {std_prom:.2f}',
              transform=ax.transAxes,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
# plt.savefig('../results/figures/01_comparacion_k_valores.png', dpi=150, bbox_inches='tight')
plt.show()

# print("✅ Comparación guardada en: results/figures/01_comparacion_k_valores.png")

#%%

clusterer_tradicional = ClusterKmeans(n_clusters=5, w_spatial=0.0)
clusterer_balanceado = ClusterKmeans(n_clusters=5, w_spatial=0.5)
clusterer_espacial = ClusterKmeans(n_clusters=5, w_spatial=0.8)

print("Entrenando clusterers...")
clusterer_tradicional.fit(x, y, z, atributo)
clusterer_balanceado.fit(x, y, z, atributo)
clusterer_espacial.fit(x, y, z, atributo)

print("Clusterers entrenados.")

print("Obteniendo resultados...")

print("\n" + "="*70)
print("📊 COMPARACIÓN DE ENFOQUES")
print("="*70)

print("\n1️⃣ CLUSTERING TRADICIONAL (peso=0.0, solo atributo)")
clusterer_tradicional.summary_plot()

print("\n2️⃣ CLUSTERING BALANCEADO (peso=0.5)")
clusterer_balanceado.summary_plot()

print("\n3️⃣ CLUSTERING ESPACIAL (peso=0.8)")
clusterer_espacial.summary_plot()
# ============================================================
# CELDA 6: Visualización comparativa
# ============================================================

# Configuración de los tres modelos
modelos = [
    (clusterer_tradicional, "Tradicional (w=0.0)\nSolo Atributo"),
    (clusterer_balanceado, "Balanceado (w=0.5)\n50% Espacio + 50% Atributo"),
    (clusterer_espacial, "Espacial (w=0.8)\n80% Espacio + 20% Atributo")
]

from mpl_toolkits.mplot3d import Axes3D  # importa para habilitar proyección 3D
from matplotlib import colors

# Usamos una ListedColormap para colores de clusters enteros
cluster_cmap = colors.ListedColormap(
    ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
)

fig = plt.figure(figsize=(20, 10))
axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]

for idx, (modelo, titulo) in enumerate(modelos):
    ax = axes[idx]
    n_clusters = len(set(modelo.clusters))  # número de clusters

    # Graficar clusters (en 3D) usando colormap discreto y norm entero
    scatter = ax.scatter(
        x, y, z,
        c=modelo.clusters,
        cmap=cluster_cmap,
        norm=colors.BoundaryNorm(boundaries=np.arange(-0.5, n_clusters+0.5, 1), ncolors=n_clusters),
        s=50, alpha=0.7, edgecolor='k', linewidth=0.5
    )

    ax.set_title(titulo, fontweight='bold', fontsize=12)
    ax.set_xlabel('X (midx)')
    ax.set_ylabel('Y (midy)')
    ax.set_zlabel('Z (midz)')
    ax.grid(alpha=0.3)

    # Agregar métrica
    metricas = modelo.get_global_metrics()
    ax.text2D(0.02, 0.98, f"Std prom: {metricas['std_prom']:.2f}",
              transform=ax.transAxes,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Colorbar con ticks para valores de cluster (enteros)
    cbar = plt.colorbar(
        scatter,
        ax=ax,
        label='Cluster',
        shrink=0.6,
        aspect=10,
        pad=0.1,
        ticks=np.arange(n_clusters)
    )
    cbar.ax.set_yticklabels([str(i) for i in range(n_clusters)])

plt.tight_layout()
# plt.savefig('../results/figures/02_comparacion_pesos.png', dpi=150, bbox_inches='tight')
plt.show()
#%%