#%% 
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import importlib
ClusterKmeans = importlib.reload(importlib.import_module('src.clustering')).ClusterKmeans
InterpoladorEspacial = importlib.reload(importlib.import_module('src.interpolacion')).InterpoladorEspacial
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

clusterer = ClusterKmeans(n_clusters=5, w_spatial=0.65)
clusterer.fit(x, y, z, atributo)

clusterer.summary_plot()

interpolador = InterpoladorEspacial(clusterer,
                             n_neighbors=20,   
                             n_points=20)

interpolador.interpolar()
interpolador.print_info()

viz = VisualizadorClusters()
viz.plot_interpolacion(interpolador)


#%%
viz.plot_comparacion_interpolacion(clusterer, interpolador)


valores_knn = [20,30,40,50]
interpoladores = {}
for n_neighbors in valores_knn:
    interpt = InterpoladorEspacial(clusterer,
                             n_neighbors=n_neighbors,   
                             n_points=100)
    interpt.interpolar()
    interpoladores[n_neighbors] = interpt

viz.plot_comparacion_n_neighbors(interpoladores)

#%%

configuraciones = [
    {'n_clusters': 5, 'w_spatial': 0.5},
    {'n_clusters': 5, 'w_spatial': 0.8},
    # {'n_clusters': 5, 'w_spatial': 0.7},
]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for idx, config in enumerate(configuraciones):
    # Clustering
    clust = ClusterKmeans(**config)
    clust.fit(x, z, atributo)
    
    # Interpolación
    interp = InterpoladorEspacial(clust, n_neighbors=25, n_points=100)
    interp.interpolar()
    
    # Visualizar clusters
    ax = axes[idx, 0]
    scatter = ax.scatter(x, y, z, c=clust.clusters, cmap='viridis',
                        s=40, alpha=0.8, edgecolors='k', linewidth=0.5)
    ax.set_title(f'Clusters (w={config["w_spatial"]:.1f})',
                fontweight='bold', fontsize=12)
    ax.set_xlabel('X (midx)')
    ax.set_ylabel('Z (midz)')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax)
    
    # Visualizar interpolación
    ax = axes[idx, 1]
    contour = ax.contourf(interp.xx, interp.yy, interp.zz, interp.clusters_interpolados,
                         levels=np.arange(clust.n_clusters + 1) - 0.5,
                         cmap='viridis', alpha=0.4)
    ax.scatter(x, y, z, c=clust.clusters, cmap='viridis',
              s=40, alpha=0.9, edgecolors='k', linewidth=0.5, zorder=10)
    ax.set_title(f'Interpolación (w={config["w_spatial"]:.1f})',
                fontweight='bold', fontsize=12)
    ax.set_xlabel('X (midx)')
    ax.set_ylabel('Y (midy)')
    ax.set_ylabel('Z (midz)')
    ax.grid(alpha=0.3)
    plt.colorbar(contour, ax=ax)

plt.tight_layout()
# plt.savefig('../results/figures/comparacion_configs_interpolacion.png',
#            dpi=150, bbox_inches='tight')
plt.show()

print("✅ Comparación guardada")