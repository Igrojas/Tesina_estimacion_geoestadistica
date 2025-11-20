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

pesos = [0.6]
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
#%%
# ============================ Estadisticas por cluster ============================
from scipy.stats import lognorm
from matplotlib.cm import get_cmap
import seaborn as sns

def resumen_graficas_clusters(df, clusters_w):
    """
    Muestra tres análisis gráficos principales (stats, lognormal, boxplot) 
    de 'starkey_min' por cluster, en un solo subplot 2x2.
    """
    df = df.copy()
    df["cluster"] = clusters_w
    n_clusters = df["cluster"].nunique()
    cmap = get_cmap("tab10")
    professional_palette = sns.color_palette("colorblind", n_colors=n_clusters)

    # Preparar estadísticas de clusters
    medias = []
    stds = []
    clusters_list = []
    for i in range(n_clusters):
        grupo = df[df["cluster"] == i]["starkey_min"]
        media = grupo.mean()
        std = grupo.std()
        ep = std / media
        medias.append(media)
        stds.append(std)
        clusters_list.append(i)
        print(f"Cluster {i} - media: {media:.2f} - std: {std:.2f} - ep: {ep:.2f}")

    # Set up main plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Media vs Desv. Estándar por Cluster
    ax = axes[0, 0]
    ax.scatter(medias, stds, color='blue', alpha=0.7)
    for i, (x, y) in enumerate(zip(medias, stds)):
        ax.text(x, y, str(i), fontsize=10, ha='right', va='bottom', color='dimgray')
    ax.set_xlabel('Media de starkey_min')
    ax.set_ylabel('Desviación estándar de starkey_min')
    ax.set_title('Media vs Desv.Std. por Cluster')
    ax.grid(alpha=0.3)
    # Línea tendencia
    z = np.polyfit(medias, stds, 1)
    p = np.poly1d(z)
    medias_line = np.linspace(min(medias), max(medias), 100)
    ax.plot(medias_line, p(medias_line), 'r--', label='Tendencia')
    ax.legend()

    # 2. Probability Lognormal Plot por Cluster
    ax = axes[0, 1]
    for i in range(n_clusters):
        data = df[df["cluster"] == i]["starkey_min"].dropna().values
        if len(data) < 3:
            continue
        shape, loc, scale = lognorm.fit(data, floc=0)
        sorted_data = np.sort(data)
        prob = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
        theo = lognorm.ppf(prob, shape, loc=loc, scale=scale)
        color = cmap(i % 10)
        ax.plot(sorted_data, prob, marker='o', linestyle='', color=color, label=f'Cluster {i} datos')
        ax.plot(theo, prob, linestyle='-', color=color, alpha=0.7, label=f'Cluster {i} lognorm')
    ax.set_xlabel('starkey_min')
    ax.set_ylabel('Probabilidad no excedencia')
    ax.set_title(f'Probability Lognormal Plot\npor Cluster')
    ax.set_yscale('logit')
    ax.grid(alpha=0.25, which='both')
    handles, labels = ax.get_legend_handles_labels()
    # Solo mostrar una leyenda con "datos" y "lognorm"
    shown = set()
    new_handles, new_labels = [], []
    for handle, label in zip(handles, labels):
        key = label.split(" ")[0]+"-"+label.split(" ")[-1]
        if key not in shown:
            shown.add(key)
            new_handles.append(handle)
            new_labels.append(label)
    ax.legend(new_handles, new_labels, fontsize=8, loc="best", frameon=True)

    # 3. Boxplot de starkey_min por Cluster
    ax = axes[1, 0]
    sns.boxplot(
        x="cluster", 
        y="starkey_min", 
        data=df, 
        palette=professional_palette,
        boxprops=dict(alpha=1),
        ax=ax
    )
    # Media de cada cluster
    cluster_means = df.groupby("cluster")["starkey_min"].mean().sort_index()
    for i, mean in enumerate(cluster_means):
        ax.scatter(i, mean, color='firebrick', s=40, marker='D', label="Media" if i == 0 else "", zorder=10, edgecolor='black')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('starkey_min')
    ax.set_title(f'Boxplots de starkey_min\npor Cluster')
    ax.grid(axis='y', alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if "Media" in labels:
        ax.legend(["Media"], loc="best", fontsize=9, facecolor="white", frameon=True)

    # 4. Empty or another basic info
    axes[1,1].axis('off')
    axes[1,1].text(0.5, 0.5, 
        f"Clusters: {n_clusters}\nN: {len(df)}\nAtributo: starkey_min", 
        ha='center', va='center', fontsize=14, color='dimgray'
    )

    plt.tight_layout()
    plt.show()

# Para usarla simplemente:
#
resumen_graficas_clusters(df, clusters_w)

# %%

N_points = 100

x_range = np.linspace(x_coords.min(), x_coords.max(), N_points)
z_range = np.linspace(z_coords.min(), z_coords.max(), N_points)

xx, zz = np.meshgrid(x_range, z_range)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

n_neighbors_list = [5, 10, 15, 20]

X_train = np.column_stack((x_coords, z_coords))
y_train = clusters_w
X_ghost = np.column_stack((xx.flatten(), zz.flatten()))

# Estandarizar X_train y X_ghost con el mismo scaler
scaler_knn = StandardScaler()
X_train_scaled = scaler_knn.fit_transform(X_train)
X_ghost_scaled = scaler_knn.transform(X_ghost)

fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True, dpi=150)
axes = axes.flatten()
for idx, n_neighbors in enumerate(n_neighbors_list):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Entrenar y predecir usando datos estandarizados
    knn.fit(X_train_scaled, y_train)
    ghost_clusters_knn = knn.predict(X_ghost_scaled)
    sc = axes[idx].scatter(xx, zz, c=ghost_clusters_knn, cmap='viridis', s=10, alpha=0.6)
    axes[idx].set_title(f'n_neighbors = {n_neighbors}', fontweight='bold')
    axes[idx].set_xlabel('midx (X)')
    if idx in [0, 2]:
        axes[idx].set_ylabel('midz (Z)')
    # También estandarizar x_coords y z_coords para visualizar sobre el mismo espacio,
    # pero para graficar puntos originales los usamos en su sistema original (sin escalar)
    axes[idx].scatter(x_coords, z_coords, c=clusters_w, cmap='viridis',
                        s=50, alpha=1, edgecolors='k', linewidth=0.5)
        
    # axes[idx].grid(alpha=0)

fig.suptitle('Clusters KNN para distintos n_neighbors', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label="Cluster")
plt.show()
# %%
