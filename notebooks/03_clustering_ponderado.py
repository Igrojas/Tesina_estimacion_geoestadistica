#%%
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.clustering import ClusterKmeans

print("✅ Imports exitosos")

# ============================================================
# CELDA 2: Cargar datos
# ============================================================
df = pd.read_csv("../data/raw/bd_dm_cmp_entry.csv", sep=";")
columnas = ["midx", "midy", "midz", "starkey_min"]
df = df[columnas].copy()

x = df['midx'].values
z = df['midz'].values
atributo = df['starkey_min'].values

print(f"📊 Datos cargados: {len(df)} puntos")
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

#%%

# ============================================================
# CELDA 7: Explorar rango de pesos
# ============================================================
# Probar muchos pesos diferentes
pesos_a_probar = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

resultados_pesos = []

for idx, peso in enumerate(pesos_a_probar):
    ax = axes[idx]
    
    # Crear y entrenar
    clust = ClusterKmeans(n_clusters=6, w_spatial=peso)
    clust.fit(x, z, atributo)
    
    # Visualizar
    ax.scatter(x, z, c=clust.clusters, cmap='viridis',
              s=30, alpha=0.7, edgecolor='k', linewidth=0.3)
    
    # Título con interpretación
    if peso == 0.0:
        interpretacion = "Solo Atributo"
    elif peso == 1.0:
        interpretacion = "Solo Espacio"
    else:
        interpretacion = f"{int(peso*100)}% Espacio\n{int((1-peso)*100)}% Atributo"
    
    ax.set_title(f'peso = {peso:.1f}\n{interpretacion}', 
                fontweight='bold', fontsize=11)
    ax.set_xlabel('X (midx)')
    ax.set_ylabel('Z (midz)')
    ax.grid(alpha=0.3)
    
    # Métricas
    metricas = clust.get_global_metrics()
    ax.text(0.02, 0.98, f"Std: {metricas['std_prom']:.2f}",
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Guardar para análisis
    resultados_pesos.append({
        'peso_espacial': peso,
        'std_promedio': metricas['std_prom'],
        'cv_promedio': metricas['cv_prom']
    })

# Ocultar el último subplot vacío
axes[-1].axis('off')

plt.tight_layout()
plt.show()
#%%

# ============================================================
# CELDA 8: Análisis cuantitativo de pesos
# ============================================================
df_pesos = pd.DataFrame(resultados_pesos)

print("\n📊 TABLA: Impacto del peso espacial")
print(df_pesos.to_string(index=False))

# Gráfico de línea: Std vs Peso
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Std promedio vs peso
axes[0].plot(df_pesos['peso_espacial'], df_pesos['std_promedio'], 
            marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0].set_xlabel('Peso Espacial', fontsize=12)
axes[0].set_ylabel('Std Promedio', fontsize=12)
axes[0].set_title('Impacto del Peso Espacial en Homogeneidad', fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Balance (0.5)')
axes[0].legend()

# Subplot 2: CV promedio vs peso
axes[1].plot(df_pesos['peso_espacial'], df_pesos['cv_promedio'],
            marker='s', linewidth=2, markersize=8, color='coral')
axes[1].set_xlabel('Peso Espacial', fontsize=12)
axes[1].set_ylabel('CV Promedio', fontsize=12)
axes[1].set_title('Impacto del Peso Espacial en CV', fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Balance (0.5)')
axes[1].legend()

plt.tight_layout()
plt.show()
#%%

# ============================================================
# CELDA 9: Encontrar peso óptimo
# ============================================================
# Basado en minimizar std_promedio
peso_optimo = df_pesos.loc[df_pesos['std_promedio'].idxmin()]

print(f"\n{'='*70}")
print(f"🏆 PESO ÓPTIMO (basado en mínima Std promedio)")
print(f"{'='*70}")
print(f"  • Peso espacial: {peso_optimo['peso_espacial']:.1f}")
print(f"  • Std promedio: {peso_optimo['std_promedio']:.2f}")
print(f"  • CV promedio: {peso_optimo['cv_promedio']:.2f}")
print(f"{'='*70}")

# Crear modelo con peso óptimo
clusterer_optimo = ClusterKmeans(
    n_clusters=6, 
    w_spatial=peso_optimo['peso_espacial']
)
clusterer_optimo.fit(x, z, atributo)

print(f"\n🔧 Modelo óptimo creado:")
print(clusterer_optimo)
clusterer_optimo.summary_plot()

# ============================================================
# CELDA 10: Comparación visual final
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Clustering óptimo
ax = axes[0, 0]
scatter = ax.scatter(x, z, c=clusterer_optimo.clusters, cmap='viridis',
                    s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
ax.set_title(f'CLUSTERING ÓPTIMO\npeso={peso_optimo["peso_espacial"]:.1f}', 
            fontweight='bold', fontsize=14)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster')

# Panel 2: Atributo real
ax = axes[0, 1]
scatter2 = ax.scatter(x, z, c=atributo, cmap='RdYlBu_r',
                     s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
ax.set_title('ATRIBUTO REAL\n(starkey_min)', fontweight='bold', fontsize=14)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
plt.colorbar(scatter2, ax=ax, label='starkey_min')

# Panel 3: Std por cluster
ax = axes[1, 0]
stats = clusterer_optimo.get_stats()
clusters_ids = list(stats.keys())
stds = [stats[i]['std'] for i in clusters_ids]

bars = ax.bar(clusters_ids, stds, color='steelblue', alpha=0.7, edgecolor='k')
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Std', fontsize=12)
ax.set_title('Desviación Estándar por Cluster', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.axhline(np.mean(stds), color='red', linestyle='--', 
          label=f'Promedio: {np.mean(stds):.2f}')
ax.legend()

# Panel 4: Tamaño de clusters
ax = axes[1, 1]
n_puntos = [stats[i]['n_points'] for i in clusters_ids]

bars = ax.bar(clusters_ids, n_puntos, color='coral', alpha=0.7, edgecolor='k')
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Número de Puntos', fontsize=12)
ax.set_title('Tamaño de Clusters', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("✅ PASO 3 COMPLETADO")
print("="*70)