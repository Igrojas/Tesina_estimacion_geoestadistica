#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/bd_dm_cmp_entry.csv", sep=";")
list_cols = ["midx","midy","midz", "starkey_min"]
df = df[list_cols].copy()

x_coords = df['midx'].values
z_coords = df['midz'].values
attr = df['starkey_min'].values

# Aplicar cluster k means espacial

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
coords_scaled = scaler.fit_transform(df[['midx', 'midz']])
attr_scaled = scaler.fit_transform(df[['starkey_min']])
# %%
w = 0.7
k = 5
coords_weight = w * coords_scaled
attr_weight = (1 - w) * attr_scaled

# Combinar
features = np.hstack([coords_weight, attr_weight])
kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit_predict(features)
df['cluster'] = clusters
#%%

sns.scatterplot(x='midx', y='midz', hue='cluster', data=df, palette='viridis')
plt.grid(True, alpha=0.3)
plt.title('Clusters K-Means Espacial')
plt.xlabel('X')
plt.ylabel('Z')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
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

pesos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_clusters = 4

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
#%%
# ============================ Estadisticas por cluster ============================

# Calcular y graficar media vs desviación estándar de starkey_min por cluster

df["cluster"] = clusters_w
medias = []
stds = []
clusters = []

for i in range(df["cluster"].nunique()):
    grupo = df[df["cluster"] == i]["starkey_min"]
    media = grupo.mean()
    std = grupo.std()
    ep = std / media
    medias.append(media)
    stds.append(std)
    clusters.append(i)
    print(f"Cluster {i} - media: {media:.2f} - std: {std:.2f} - ep: {ep:.2f}")

plt.figure(figsize=(8,5))
plt.scatter(medias, stds, color='blue', alpha=0.7)
for i, (x, y) in enumerate(zip(medias, stds)):
    plt.text(x, y, str(i), fontsize=10, ha='right', va='bottom', color='dimgray')
plt.xlabel('Media de starkey_min')
plt.ylabel('Desviación estándar de starkey_min')
plt.title('Media vs Desviación Estándar por Cluster')
plt.grid(alpha=0.3)

# Ajuste de línea de tendencia
z = np.polyfit(medias, stds, 1)
p = np.poly1d(z)
medias_line = np.linspace(min(medias), max(medias), 100)
plt.plot(medias_line, p(medias_line), 'r--', label='Tendencia')
plt.legend()
plt.tight_layout()
plt.show()
#%%
from scipy.stats import lognorm
from matplotlib.cm import get_cmap

# Graficar Probability Lognormal Plot para todos los clusters

n_clusters = df["cluster"].nunique()
cmap = get_cmap("tab10")

plt.figure(figsize=(10, 6))

for i in range(n_clusters):
    data = df[df["cluster"] == i]["starkey_min"].dropna().values
    if len(data) < 3:
        continue
    # Ajustar distribución log-normal
    shape, loc, scale = lognorm.fit(data, floc=0)
    sorted_data = np.sort(data)
    # Probabilidad de no excedencia
    prob = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
    # Valor teórico lognormal para las probabilidades
    theo = lognorm.ppf(prob, shape, loc=loc, scale=scale)
    color = cmap(i % 10)
    # Probability plot: eje X = datos reales ordenados, eje Y = probabilidad acumulada (escala log-normal teórica)
    plt.plot(sorted_data, prob, marker='o', linestyle='', color=color, label=f'Cluster {i} datos')
    plt.plot(theo, prob, linestyle='-', color=color, alpha=0.7, label=f'Cluster {i} lognorm')

plt.xlabel('starkey_min')
plt.ylabel('Probabilidad no excedencia')
plt.title(f'Probability Lognormal Plot por Cluster - starkey_min - {n_clusters} clusters')
plt.legend(title="Clusters", fontsize=10, title_fontsize=11)
plt.yscale('logit')
plt.grid(alpha=0.25, which='both')
plt.tight_layout()
plt.show()
# %%
import seaborn as sns

# Paleta profesional: "Set2" es sobria, alternativa: "colorblind", "pastel", o crear una custom.
professional_palette = sns.color_palette("colorblind", n_colors=n_clusters)

plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    x="cluster", 
    y="starkey_min", 
    data=df, 
    palette=professional_palette,
    boxprops=dict(alpha=1)
)

# Calcular y graficar la media de cada cluster
cluster_means = df.groupby("cluster")["starkey_min"].mean().sort_index()
for i, mean in enumerate(cluster_means):
    ax.scatter(i, mean, color='firebrick', s=40, marker='D', label="Media" if i == 0 else "", zorder=10, edgecolor='black')

plt.xlabel('Cluster')
plt.ylabel('starkey_min')
plt.title(f'Boxplots de starkey_min por Cluster - {n_clusters} clusters')
plt.grid(axis='y', alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
# Solo un entry para la media
if "Media" in labels:
    ax.legend(["Media"], loc="best", fontsize=10, facecolor="white", frameon=True)
plt.tight_layout()
plt.show()