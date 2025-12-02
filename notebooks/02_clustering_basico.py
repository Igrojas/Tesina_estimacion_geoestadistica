#%%
import sys
sys.path.append('../')  # Para importar desde src/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.clustering import ClusterKmeans

df = pd.read_csv('../data/raw/bd_dm_cmp_entry.csv', sep=';')
# Filtrar columnas
columnas = ["midx", "midy", "midz", "starkey_min"]
df = df[columnas].copy()

# Extraer coordenadas y atributo
x = df['midx'].values
z = df['midz'].values
atributo = df['starkey_min'].values

print(f"📊 Datos cargados: {len(df)} puntos")
print(f"📏 Rango X: [{x.min():.0f}, {x.max():.0f}]")
print(f"📏 Rango Z: [{z.min():.0f}, {z.max():.0f}]")
print(f"📊 Rango atributo: [{atributo.min():.2f}, {atributo.max():.2f}]")

# ============================================================
# CELDA 3: Crear objeto (sin entrenar)
# ============================================================
clusterer = ClusterKmeans(n_clusters=5)

# Ver objeto sin entrenar
print(clusterer)

# ============================================================
# CELDA 4: Entrenar
# ============================================================
clusterer.fit(x, z, atributo)

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
plt.figure(figsize=(12, 5))

# Subplot 1: Clusters
plt.subplot(1, 2, 1)
scatter = plt.scatter(x, z, c=clusters, cmap='viridis', 
                     s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Clustering K-means (k={clusterer.n_clusters})')
plt.xlabel('X (midx)')
plt.ylabel('Z (midz)')
plt.grid(alpha=0.3)

# Subplot 2: Atributo real
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(x, z, c=atributo, cmap='RdYlBu_r',
                       s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
plt.colorbar(scatter2, label='starkey_min')
plt.title('Atributo Real')
plt.xlabel('X (midx)')
plt.ylabel('Z (midz)')
plt.grid(alpha=0.3)

plt.tight_layout()
# plt.savefig('../results/figures/01_clustering_basico.png', dpi=150, bbox_inches='tight')
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

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, k in enumerate(valores_k):
    # Crear y entrenar
    clust = ClusterKmeans(n_clusters=k)
    clust.fit(x, z, atributo)
    
    # Visualizar
    axes[idx].scatter(x, z, c=clust.clusters, cmap='viridis',
                     s=30, alpha=0.7, edgecolor='k', linewidth=0.3)
    axes[idx].set_title(f'k = {k}', fontweight='bold', fontsize=14)
    axes[idx].set_xlabel('X (midx)')
    axes[idx].set_ylabel('Z (midz)')
    axes[idx].grid(alpha=0.3)
    
    # Agregar métrica
    stats = clust.get_stats()
    std_prom = np.mean([s['mean'] for s in stats.values()])
    axes[idx].text(0.02, 0.98, f'Mean prom: {std_prom:.2f}',
                   transform=axes[idx].transAxes,
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
clusterer_tradicional.fit(x, z, atributo)
clusterer_balanceado.fit(x, z, atributo)
clusterer_espacial.fit(x, z, atributo)

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
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Configuración de los tres modelos
modelos = [
    (clusterer_tradicional, "Tradicional (w=0.0)\nSolo Atributo"),
    (clusterer_balanceado, "Balanceado (w=0.5)\n50% Espacio + 50% Atributo"),
    (clusterer_espacial, "Espacial (w=0.8)\n80% Espacio + 20% Atributo")
]

for idx, (modelo, titulo) in enumerate(modelos):
    ax = axes[idx]
    
    # Graficar clusters
    scatter = ax.scatter(x, z, c=modelo.clusters, cmap='viridis',
                        s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
    
    ax.set_title(titulo, fontweight='bold', fontsize=12)
    ax.set_xlabel('X (midx)')
    ax.set_ylabel('Z (midz)')
    ax.grid(alpha=0.3)
    
    # Agregar métrica
    metricas = modelo.get_global_metrics()
    ax.text(0.02, 0.98, f"Std prom: {metricas['std_prom']:.2f}",
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.colorbar(scatter, ax=ax, label='Cluster')

plt.tight_layout()
    # plt.savefig('../results/figures/02_comparacion_pesos.png', dpi=150, bbox_inches='tight')
plt.show()