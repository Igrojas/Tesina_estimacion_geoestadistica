#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Cluster Algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("../data/bd_dm_cmp_entry.csv", sep=";")
# %%

x_coords = df['midx']
y_coords = df['midy']
z_coords = df['midz']
attr = df['starkey_min']

#%%

coords_xz = df[['midx', 'midz']]

n_cluster = 4

kmeans_spacial = KMeans(n_clusters=n_cluster)
clusters_spacial = kmeans_spacial.fit_predict(coords_xz)

df['cluster_spacial'] = clusters_spacial
# %%

unique, counts = np.unique(clusters_spacial, return_counts=True)
print(f"Número de clusters: {len(unique)}")
print(f"Número de puntos por cluster: {counts}")
for cluster_id, count in zip(unique, counts):
    print(f"Cluster {cluster_id}: {count} puntos")

# %%

plt.figure(figsize=(10, 6))
palette = sns.color_palette("Dark2", n_colors=df['cluster_spacial'].nunique())
sns.scatterplot(x='midx', y='midz', hue='cluster_spacial', data=df, palette=palette)
plt.grid(True, alpha=0.5)
plt.title('Clusters K-Means Espacial')
plt.xlabel('X')
plt.ylabel('Z')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%



def kmeans_ponderado(x,z,attr, peso_espacial=0.5, n_clusters=4):

    scaler_coords = StandardScaler()
    scaler_attr = StandardScaler()

    coords_scaled = scaler_coords.fit_transform(np.column_stack([x, z]))
    attr_scaled = scaler_attr.fit_transform(np.array([attr]).reshape(-1, 1))

    # Aca la ponderacion
    coords_weight = peso_espacial * coords_scaled   
    attr_weight = (1 - peso_espacial) * attr_scaled

    features = np.column_stack([coords_weight, attr_weight])

    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(features)

    return clusters,features


pesos = [0.5, 0.6, 0.75, 0.85, 1.0]
n_clusters = 6

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, peso in enumerate(pesos):
    clusters_w, features = kmeans_ponderado(
        x_coords, z_coords, attr, 
        n_clusters=n_clusters, 
        peso_espacial=peso
    )
    
    axes[idx].scatter(x_coords, z_coords, c=clusters_w, cmap='viridis',
                      s=50, alpha=0.6, edgecolors='k', linewidth=0.8)
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
# %%

df_clusters = df[['midx', 'midz', 'starkey_min']]
df_clusters['cluster_spacial'] = clusters_spacial



# Graficar todos los boxplots de 'starkey_min' por cluster en un solo gráfico
plt.figure(figsize=(8, 6))
palette = sns.color_palette("dark", n_colors=df_clusters['cluster_spacial'].nunique())
sns.boxplot(x='cluster_spacial', y='starkey_min', data=df_clusters, palette=palette)
plt.title('Distribución de Starkey por cluster')
plt.xlabel('Cluster')
plt.ylabel('Starkey')
plt.grid(True, alpha=0.3)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

def graficar_cdf_completa(datos, titulo='', xlabel='Valor', color='steelblue'):
    """
    Grafica CDF con percentiles marcados
    """
    # Ordenar datos
    datos_ordenados = np.sort(datos)
    n = len(datos_ordenados)
    probabilidades = np.arange(1, n + 1) / n
    
    # Calcular percentiles clave
    percentiles = [10, 25, 50, 75, 90]
    valores_percentiles = np.percentile(datos, percentiles)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Graficar CDF
    ax.plot(datos_ordenados, probabilidades * 100, 
            linewidth=2.5, color=color, label='CDF')
    
    # Agregar líneas de percentiles
    colores_percentiles = ['red', 'orange', 'green', 'orange', 'red']
    
    for p, val, col in zip(percentiles, valores_percentiles, colores_percentiles):
        ax.axvline(val, color=col, linestyle='--', alpha=0.6, linewidth=1.5)
        ax.axhline(p, color=col, linestyle='--', alpha=0.6, linewidth=1.5)
        ax.plot(val, p, 'o', color=col, markersize=8, 
                label=f'P{p}: {val:.3f}', zorder=5)
    
    # Etiquetas y formato
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel('Probabilidad Acumulada (%)', fontsize=13, fontweight='bold')
    ax.set_title(titulo if titulo else f'CDF de {xlabel}', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.5, linestyle=':')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 105])
    
    # Agregar cuadro con estadísticas
    stats_text = f'N = {len(datos)}\n'
    stats_text += f'Media = {np.mean(datos):.3f}\n'
    stats_text += f'Mediana = {np.median(datos):.3f}\n'
    stats_text += f'Std = {np.std(datos):.3f}\n'
    stats_text += f'Min = {np.min(datos):.3f}\n'
    stats_text += f'Max = {np.max(datos):.3f}'
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10,
            family='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return valores_percentiles

# Usar la función
percentiles = graficar_cdf_completa(
    df['starkey_min'].values,
    titulo='Distribución Acumulada de Starkey Mineral',
    xlabel='starkey_min'
)
