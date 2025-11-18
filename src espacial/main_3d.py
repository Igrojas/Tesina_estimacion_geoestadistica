#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

df = pd.read_csv("data/bd_dm_cmp_entry.csv", sep=";")
list_cols = ["midx","midy","midz", "starkey_min"]
df = df[list_cols].copy()

x_coords = df['midx'].values
y_coords = df['midy'].values
z_coords = df['midz'].values
attr = df['starkey_min'].values

# Aplicar cluster k means espacial
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(df[['midx', 'midy', 'midz']])
attr_scaled = scaler.fit_transform(df[['starkey_min']])

# %%
w = 0.5
k = 6
coords_weight = w * coords_scaled[:, :3]
attr_weight = (1 - w) * attr_scaled[:, 0].reshape(-1, 1)
#%%
# Combinar
features = np.hstack([coords_weight, attr_weight])
kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit_predict(features)
df['cluster'] = clusters
#%%

import plotly.express as px

fig = px.scatter_3d(
    df,
    x='midx', y='midy', z='midz',
    color='cluster',    
    color_continuous_scale='inferno',
    title='Clusters K-Means Espacial 3D',
    # labels={'midx': 'X (midx)', 'midy': 'Y (midy)', 'midz': 'Z (midz)', 'cluster': 'Cluster'},
    opacity=0.7
)

fig.update_traces(marker=dict(size=3))
fig.update_layout(
    legend_title_text='Cluster',
    scene=dict(
        xaxis_title="X (midx)",
        yaxis_title="Y (midy)",
        zaxis_title="Z (midz)"
    )
)
fig.show()
# %%
def kmeans_ponderado(x, z, atributo, n_clusters, peso_espacial=0.5):
    """
    K-means con ponderación entre espacio y atributo
    
    Parámetros:
    -----------
    peso_espacial : float (0 a 1)
        - peso_espacial = 0: Solo considera atributo (clustering tradicional)
        - peso_espacial = 0.5: Balance 50/50 entre espacio y atributo
        - peso_espacial = 1: Solo considera coordenadas (espacial puro)
    """
    
    # Estandarizar por separado
    scaler_coords = StandardScaler()
    scaler_attr = StandardScaler()
    
    coords_scaled = scaler_coords.fit_transform(np.column_stack([x, z]))
    attr_scaled = scaler_attr.fit_transform(atributo.reshape(-1, 1))
    
    # Aplicar ponderación
    coords_weighted = coords_scaled * peso_espacial
    attr_weighted = attr_scaled * (1 - peso_espacial)
    
    # Combinar features ponderadas
    features_ponderadas = np.column_stack([coords_weighted, attr_weighted])
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_ponderadas)
    
    return clusters, features_ponderadas

pesos = [0.0, 0.3, 0.5, 0.7, 1.0]
n_clusters = 6

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, peso in enumerate(pesos):
    clusters_w, _ = kmeans_ponderado(
        x_coords, z_coords, attr, 
        n_clusters=n_clusters, 
        peso_espacial=peso
    )
    
    axes[idx].scatter(x_coords, z_coords, c=clusters_w, cmap='viridis',
                      s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[idx].set_xlabel('X (midx)')
    axes[idx].set_ylabel('Z (midz)')
    axes[idx].set_title(f'Peso Espacial = {peso:.1f}\n' + 
                        f'{"100% Atributo" if peso==0 else "100% Espacial" if peso==1 else f"{int(peso*100)}% Espacial, {int((1-peso)*100)}% Atributo"}',
                        fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    
    # Calcular homogeneidad promedio
    stds = []
    for i in range(n_clusters):
        mask = clusters_w == i
        if mask.sum() > 1:
            stds.append(attr[mask].std())
    
    axes[idx].text(0.02, 0.98, f'Std promedio: {np.mean(stds):.3f}',
                   transform=axes[idx].transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Atributo real
axes[5].scatter(x_coords, z_coords, c=attr, cmap='RdYlBu_r',
                s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[5].set_xlabel('X (midx)')
axes[5].set_ylabel('Z (midz)')
axes[5].set_title('ATRIBUTO REAL\n(starkey_min)', fontweight='bold')
axes[5].grid(True, alpha=0.3)
plt.colorbar(axes[5].scatter(x_coords, z_coords, c=attr, cmap='RdYlBu_r'), 
             ax=axes[5], label='starkey_min')

plt.tight_layout()
plt.show()