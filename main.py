#%%
import pandas as pd
import geopandas as gpd
from spopt.region import Skater
import libpysal
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
from sklearn.metrics import pairwise as skm
import numpy
# Cargar datos desde Excel
df = pd.read_csv("data/bd_dm_cmp_entry.csv", sep=";")

list_cols = ["midx","midy","midz", "starkey_min"]
df = df[list_cols].copy()

# %%
print("=" * 60)
print("PASO 1: Datos cargados")
print("=" * 60)
print(f"Cantidad de registros: {len(df)}")
print(f"\nRangos de coordenadas:")
print(f"X: {df['midx'].min():.2f} - {df['midx'].max():.2f}")
print(f"Y: {df['midy'].min():.2f} - {df['midy'].max():.2f}")
print(f"Z: {df['midz'].min():.2f} - {df['midz'].max():.2f}")
print(f"Starkey: {df['starkey_min'].min():.2f} - {df['starkey_min'].max():.2f}")

# %%
# ===== PASO 2: CREAR GEOMETRÍAS 2D (X-Z) =====
# Usamos X (este-oeste) y Z (profundidad)
geometry_xz = [Point(x, z) for x, z in zip(df['midx'], df['midz'])]
gdf_xz = gpd.GeoDataFrame(df, geometry=geometry_xz, crs=None)

print("\n" + "=" * 60)
print("PASO 2: GeoDataFrame 2D (X-Z) creado")
print("=" * 60)
print(gdf_xz.head())

# Visualizar datos en 2D
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(df['midx'], df['midz'], 
                     c=df['starkey_min'], 
                     cmap='viridis', 
                     s=20, 
                     alpha=0.6)
ax.set_xlabel('X (Este-Oeste)', fontsize=12)
ax.set_ylabel('Z (Profundidad)', fontsize=12)
ax.set_title('Vista 2D: Sección X-Z coloreada por starkey_min', fontsize=14)
plt.colorbar(scatter, ax=ax, label='starkey_min')
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('paso2_datos_xz.png', dpi=150)
# print("✓ Gráfico guardado: paso2_datos_xz.png")
# %%
# ===== PASO 3: MATRIZ DE CONECTIVIDAD ESPACIAL =====
print("\n" + "=" * 60)
print("PASO 3: Crear matriz de conectividad espacial (W)")
print("=" * 60)

# KNN: cada punto conectado con sus k vecinos más cercanos
k_neighbors = 80
w = libpysal.weights.KNN.from_dataframe(gdf_xz, k=k_neighbors)

print(f"✓ Matriz W creada con K={k_neighbors} vecinos")
print(f"  - Número de observaciones: {w.n}")
print(f"  - Total de conexiones: {w.n * k_neighbors}")
print(f"  - Vecinos del punto 0: {w.neighbors[0]}")

# Visualizar conectividad
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(df['midx'], df['midz'], c='teal', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

# Dibujar conexiones (solo algunas para no saturar)
sample_indices = np.random.choice(len(df), size=min(50, len(df)), replace=False)
for idx in sample_indices:
    x0, z0 = df.iloc[idx]['midx'], df.iloc[idx]['midz']
    for neighbor_idx in w.neighbors[idx]:
        x1, z1 = df.iloc[neighbor_idx]['midx'], df.iloc[neighbor_idx]['midz']
        ax.plot([x0, x1], [z0, z1], 'r-', alpha=0.5, linewidth=1)

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Z', fontsize=12)
ax.set_title(f'Conectividad Espacial (KNN k={k_neighbors})', fontsize=14)
plt.tight_layout()
plt.grid(True, alpha=0.3)
# %%
# ===== PASO 4: APLICAR SKATER =====
print("\n" + "=" * 60)
print("PASO 4: Aplicar algoritmo SKATER")
print("=" * 60)

# Variable para clustering (starkey_min)
attrs_name = ['starkey_min',
              'midx',
              'midz'
              ]

# Número de clusters deseados
n_clusters = 6

print(f"Configuración:")
print(f"  - Variable de atributo: {attrs_name}")
print(f"  - Número de clusters: {n_clusters}")
print(f"  - Método de conectividad: KNN (k={k_neighbors})")


spanning_forest_kwds = {
    "dissimilarity": skm.manhattan_distances,
    "affinity": None,
    "reduction": numpy.sum,
    "center": numpy.mean,
    "verbose": False,
}


# Aplicar SKATER
model = Skater(
    gdf_xz,
    w,
    attrs_name,
    n_clusters=n_clusters,
    floor=5,  # Mínimo de observaciones por cluster
    trace=False,
    spanning_forest_kwds=spanning_forest_kwds,

)
model.solve()

print("\n✓ SKATER ejecutado exitosamente")
print(f"  - Clusters creados: {len(np.unique(model.labels_))}")

# Agregar labels al dataframe
gdf_xz['cluster'] = model.labels_

# Estadísticas por cluster
print("\n" + "=" * 60)
print("ESTADÍSTICAS POR CLUSTER")
print("=" * 60)
for cluster_id in sorted(gdf_xz['cluster'].unique()):
    cluster_data = gdf_xz[gdf_xz['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(f"  - Cantidad de puntos: {len(cluster_data)}")
    print(f"  - Starkey promedio: {cluster_data['starkey_min'].mean():.2f}")
    print(f"  - Starkey std: {cluster_data['starkey_min'].std():.2f}")
    print(f"  - Rango X: [{cluster_data['midx'].min():.0f}, {cluster_data['midx'].max():.0f}]")
    print(f"  - Rango Z: [{cluster_data['midz'].min():.0f}, {cluster_data['midz'].max():.0f}]")

#%%
# ===== PASO 5: VISUALIZAR RESULTADOS =====
print("\n" + "=" * 60)
print("PASO 5: Visualizar clusters")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Crear una máscara para los puntos con datos conocidos
known_mask = ~numpy.isnan(gdf_xz['starkey_min'])

# Subplot 1: Clusters
# Luego dibujamos los clusters con datos conocidos
scatter1 = axes[0].scatter(gdf_xz.loc[known_mask, 'midx'], gdf_xz.loc[known_mask, 'midz'], 
                           c=gdf_xz.loc[known_mask, 'cluster'], 
                           cmap='tab10', 
                           s=30, 
                           alpha=1,
                           edgecolors='black',
                           linewidth=0.5)

axes[0].set_xlabel('X (Este-Oeste)', fontsize=12)
axes[0].set_ylabel('Z (Profundidad)', fontsize=12)
axes[0].set_title(f'Clusters SKATER (n={n_clusters})', fontsize=14)
plt.colorbar(scatter1, ax=axes[0], label='Cluster ID')
axes[0].grid(True, alpha=0.3)

# Subplot 2: Variable original con puntos fantasma
# Crear una malla de puntos fantasma para rellenar el espacio
x_min, x_max = gdf_xz['midx'].min(), gdf_xz['midx'].max()
z_min, z_max = gdf_xz['midz'].min(), gdf_xz['midz'].max()
x_grid = numpy.linspace(x_min, x_max, 100)
z_grid = numpy.linspace(z_min, z_max, 100)
xx, zz = numpy.meshgrid(x_grid, z_grid)
ghost_points_x = xx.flatten()
ghost_points_z = zz.flatten()

# Dibujar los puntos fantasma primero (en un color muy claro)
axes[1].scatter(ghost_points_x, ghost_points_z, 
               c='black', 
               marker='.',
               s=15, 
               alpha=0.2,
               label='Puntos fantasma',
               zorder=0)

# Luego dibujamos los clusters con datos conocidos (igual que en el primer subplot)
scatter2 = axes[1].scatter(gdf_xz.loc[known_mask, 'midx'], gdf_xz.loc[known_mask, 'midz'], 
                           c=gdf_xz.loc[known_mask, 'cluster'], 
                           cmap='tab10', 
                           s=30, 
                           alpha=1,
                           edgecolors='black',
                           linewidth=0.5,
                           zorder=10)
axes[1].set_xlabel('X (Este-Oeste)', fontsize=12)
axes[1].set_ylabel('Z (Profundidad)', fontsize=12)
axes[1].set_title(f'Clusters SKATER con puntos fantasma (n={n_clusters})', fontsize=14)
plt.colorbar(scatter2, ax=axes[1], label='Cluster ID')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('paso5_resultados_skater.png', dpi=150)
print("✓ Gráfico guardado: paso5_resultados_skater.png")

plt.show()

print("\n" + "=" * 60)
print("✓ ANÁLISIS COMPLETO")
print("=" * 60)
# %%

#%%
# ===== PASO 6: CLASIFICACIÓN DE PUNTOS FANTASMA =====
print("\n" + "=" * 60)
print("PASO 6: Clasificación de puntos fantasma")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Preparar datos de entrenamiento (puntos conocidos)
X_train = gdf_xz.loc[known_mask, ['midx', 'midz']].values
y_train = gdf_xz.loc[known_mask, 'cluster'].values

# Preparar datos de puntos fantasma para predecir
X_ghost = np.column_stack((ghost_points_x, ghost_points_z))

print("Entrenando modelos de clasificación...")

# Método 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)
ghost_clusters_rf = rf_model.predict(X_ghost)

# Método 2: K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
ghost_clusters_knn = knn_model.predict(X_ghost)

# Visualizar resultados
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest
scatter_rf = axes[0].scatter(ghost_points_x, ghost_points_z, 
                            c=ghost_clusters_rf, 
                            cmap='tab10', 
                            s=15, 
                            alpha=0.5,
                            marker='.')
# Puntos originales
axes[0].scatter(gdf_xz.loc[known_mask, 'midx'], gdf_xz.loc[known_mask, 'midz'], 
               c=gdf_xz.loc[known_mask, 'cluster'], 
               cmap='tab10', 
               s=30, 
               alpha=1,
               edgecolors='black',
               linewidth=0.5)
axes[0].set_xlabel('X (Este-Oeste)', fontsize=12)
axes[0].set_ylabel('Z (Profundidad)', fontsize=12)
axes[0].set_title('Clasificación con Random Forest', fontsize=14)
plt.colorbar(scatter_rf, ax=axes[0], label='Cluster ID')
axes[0].grid(True, alpha=0.3)

# K-Nearest Neighbors
scatter_knn = axes[1].scatter(ghost_points_x, ghost_points_z, 
                             c=ghost_clusters_knn, 
                             cmap='tab10', 
                             s=15, 
                             alpha=0.5,
                             marker='.')
# Puntos originales
axes[1].scatter(gdf_xz.loc[known_mask, 'midx'], gdf_xz.loc[known_mask, 'midz'], 
               c=gdf_xz.loc[known_mask, 'cluster'], 
               cmap='tab10', 
               s=30, 
               alpha=1,
               edgecolors='black',
               linewidth=0.5)
axes[1].set_xlabel('X (Este-Oeste)', fontsize=12)
axes[1].set_ylabel('Z (Profundidad)', fontsize=12)
axes[1].set_title('Clasificación con K-Nearest Neighbors', fontsize=14)
plt.colorbar(scatter_knn, ax=axes[1], label='Cluster ID')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paso6_clasificacion_puntos_fantasma.png', dpi=150)
print("✓ Gráfico guardado: paso6_clasificacion_puntos_fantasma.png")

# Evaluación de modelos (validación cruzada)
from sklearn.model_selection import cross_val_score

print("\nEvaluación de modelos:")
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
knn_scores = cross_val_score(knn_model, X_train, y_train, cv=5)

print(f"Random Forest - Precisión media: {rf_scores.mean():.4f} (±{rf_scores.std():.4f})")
print(f"KNN - Precisión media: {knn_scores.mean():.4f} (±{knn_scores.std():.4f})")

print("\nComparación de distribución de clusters:")
print("Clusters originales:")
print(pd.Series(y_train).value_counts().sort_index())
print("\nClusters asignados por Random Forest:")
print(pd.Series(ghost_clusters_rf).value_counts().sort_index())
print("\nClusters asignados por KNN:")
print(pd.Series(ghost_clusters_knn).value_counts().sort_index())

plt.show()

print("\n" + "=" * 60)
print("✓ CLASIFICACIÓN DE PUNTOS FANTASMA COMPLETA")
print("=" * 60)
# %%
