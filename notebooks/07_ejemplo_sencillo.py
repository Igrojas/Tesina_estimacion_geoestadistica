"""
============================================================
EJEMPLO SENCILLO: Entendiendo el Pipeline de Estimación
============================================================

Este script es una versión simplificada y comentada del pipeline completo.
Ideal para entender paso a paso cómo funciona cada componente.

Autor: Sistema
Fecha: 2024
"""

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

# Importar nuestros módulos
from src.clustering import ClusterKmeans
from src.estimacion import EstimadorEspacial
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
df = pd.read_csv("../data/raw/bd_dm_cmp_entry.csv", sep=";")

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
scatter = ax.scatter(x, z, c=atributo, cmap='RdYlBu_r', s=30, alpha=0.6)
ax.set_title('Distribución Espacial del Atributo', fontweight='bold')
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
plt.colorbar(scatter, ax=ax, label='starkey_min')
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
w_spatial = 0.65             # 65% peso espacial, 35% peso atributo

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

# Visualizar clusters
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(x, z, c=clusterer.clusters, cmap='viridis', 
                    s=30, alpha=0.6, edgecolors='k', linewidth=0.3)
ax.set_title(f'Clusters Espaciales (k={n_clusters})', fontweight='bold')
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.show()

#%%
# ============================================================
# PASO 4: DIVISIÓN TRAIN/TEST
# ============================================================
print("\n" + "="*70)
print("🔀 PASO 3: DIVIDIR DATOS EN TRAIN/TEST")
print("="*70)

print("\n📌 ¿Por qué dividir?")
print("   • Train (80%): Para entrenar el modelo")
print("   • Test (20%): Para evaluar el modelo (datos nunca vistos)")

# Crear índices
indices = np.arange(len(x))

# Dividir estratificando por cluster (mantiene proporciones)
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,           # 20% para test
    random_state=42,          # Semilla para reproducibilidad
    stratify=clusterer.clusters  # Mantener proporción de clusters
)

# Extraer subconjuntos
x_train, x_test = x[train_idx], x[test_idx]
z_train, z_test = z[train_idx], z[test_idx]
attr_train, attr_test = atributo[train_idx], atributo[test_idx]
clusters_train = clusterer.clusters[train_idx]

print(f"\n✅ División completada:")
print(f"   • Train: {len(x_train)} puntos ({len(x_train)/len(x)*100:.1f}%)")
print(f"   • Test: {len(x_test)} puntos ({len(x_test)/len(x)*100:.1f}%)")

# Visualizar división
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_train, z_train, c='blue', s=20, alpha=0.5, 
          label=f'Train ({len(x_train)})', edgecolors='k', linewidth=0.2)
ax.scatter(x_test, z_test, c='red', s=60, alpha=0.8, marker='s',
          label=f'Test ({len(x_test)})', edgecolors='k', linewidth=0.5)
ax.set_title('División Train/Test', fontweight='bold')
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# ============================================================
# PASO 5: ESTIMACIÓN CON KNN
# ============================================================
print("\n" + "="*70)
print("🎯 PASO 4: ESTIMACIÓN CON KNN")
print("="*70)

print("\n📌 ¿Qué hace KNN?")
print("   K-Nearest Neighbors estima el valor de un punto usando")
print("   el promedio ponderado de sus K vecinos más cercanos.")
print("   • Vecinos cercanos pesan más que lejanos")

# Configurar KNN
n_neighbors = 10
print(f"\n⚙️  Parámetros:")
print(f"   • Número de vecinos (k): {n_neighbors}")

# Crear y entrenar estimador
estimador = EstimadorEspacial(metodo='knn', n_neighbors=n_neighbors)
estimador.fit(x_train, z_train, attr_train)

print(f"\n✅ Modelo entrenado con {len(x_train)} puntos")

# Predecir en test
print("\n🔮 Prediciendo valores en conjunto de test...")
predicciones = estimador.predict(x_test, z_test)

# Calcular métricas
mae = mean_absolute_error(attr_test, predicciones)
r2 = r2_score(attr_test, predicciones)

print("\n📊 RESULTADOS:")
print(f"   • MAE (Error Absoluto Medio): {mae:.3f}")
print(f"     → En promedio, nos equivocamos en {mae:.3f} unidades")
print(f"   • R² (Coeficiente de Determinación): {r2:.3f}")
print(f"     → El modelo explica {r2*100:.1f}% de la varianza")

# Visualizar resultados
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Predicho vs Real
ax = axes[0]
ax.scatter(attr_test, predicciones, alpha=0.6, s=50, edgecolors='k', linewidth=0.3)
min_val = min(attr_test.min(), predicciones.min())
max_val = max(attr_test.max(), predicciones.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
       label='Predicción perfecta')
ax.set_title(f'Predicho vs Real\nMAE={mae:.2f}, R²={r2:.3f}', fontweight='bold')
ax.set_xlabel('Valor Real')
ax.set_ylabel('Valor Predicho')
ax.legend()
ax.grid(alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Panel 2: Errores
ax = axes[1]
errores = attr_test - predicciones
ax.hist(errores, bins=30, alpha=0.7, color='steelblue', edgecolor='k')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
ax.axvline(errores.mean(), color='orange', linestyle='--', linewidth=2,
          label=f'Media = {errores.mean():.2f}')
ax.set_title('Distribución de Errores', fontweight='bold')
ax.set_xlabel('Error (Real - Predicho)')
ax.set_ylabel('Frecuencia')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

#%%
# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*70)
print("📝 RESUMEN")
print("="*70)

print(f"""
✅ PIPELINE COMPLETADO

1. DATOS: {len(df)} puntos cargados
2. CLUSTERING: {n_clusters} dominios identificados
3. TRAIN/TEST: {len(x_train)} train, {len(x_test)} test
4. ESTIMACIÓN: KNN con k={n_neighbors}
5. RESULTADOS: MAE={mae:.3f}, R²={r2:.3f}

📌 INTERPRETACIÓN:
   • MAE={mae:.3f}: En promedio, el error es de {mae:.3f} unidades
   • R²={r2:.3f}: El modelo explica {r2*100:.1f}% de la variabilidad
   
🎯 PRÓXIMOS PASOS:
   • Probar diferentes valores de k (vecinos)
   • Comparar estimación global vs por cluster
   • Visualizar predicciones en el espacio
""")

print("="*70)
print("✅ EJEMPLO COMPLETADO")
print("="*70)

