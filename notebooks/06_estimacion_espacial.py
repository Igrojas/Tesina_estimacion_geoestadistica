"""
============================================================
PIPELINE COMPLETO: CLUSTERING → DELIMITACIÓN → ESTIMACIÓN
============================================================

Este script implementa el flujo completo de:
1. Clusterización espacial de datos
2. Delimitación de dominios (interpolación de clusters)
3. Estimación con KNN por dominio

Autor: Sistema
Fecha: 2024
"""

#%%
# ============================================================
# PASO 1: CONFIGURACIÓN E IMPORTS
# ============================================================
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.clustering import ClusterKmeans
from src.estimacion import EstimadorEspacial
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*70)
print("🚀 PIPELINE DE ESTIMACIÓN ESPACIAL")
print("="*70)
print("✅ Librerías cargadas correctamente\n")

#%%
# ============================================================
# PASO 2: CARGA DE DATOS
# ============================================================
print("\n" + "="*70)
print("📂 PASO 1: CARGA DE DATOS")
print("="*70)

df = pd.read_csv("../data/raw/bd_dm_cmp_entry.csv", sep=";")
columnas = ["midx", "midy", "midz", "starkey_min"]
df = df[columnas].copy()

# Extraer variables
x = df['midx'].values
z = df['midz'].values
atributo = df['starkey_min'].values

print(f"\n📊 Datos cargados exitosamente:")
print(f"   • Total de puntos: {len(df)}")
print(f"   • Coordenadas X: [{x.min():.1f}, {x.max():.1f}]")
print(f"   • Coordenadas Z: [{z.min():.1f}, {z.max():.1f}]")
print(f"   • Atributo (starkey_min):")
print(f"      - Media: {atributo.mean():.2f}")
print(f"      - Std: {atributo.std():.2f}")
print(f"      - Min: {atributo.min():.2f}")
print(f"      - Max: {atributo.max():.2f}")

# Visualización inicial
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Distribución espacial
ax = axes[0]
scatter = ax.scatter(x, z, c=atributo, cmap='RdYlBu_r',
                    s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
ax.set_title('Distribución Espacial del Atributo', fontweight='bold', fontsize=14)
ax.set_xlabel('X (midx)', fontsize=12)
ax.set_ylabel('Z (midz)', fontsize=12)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='starkey_min')

# Panel 2: Histograma del atributo
ax = axes[1]
ax.hist(atributo, bins=50, alpha=0.7, color='steelblue', edgecolor='k')
ax.axvline(atributo.mean(), color='red', linestyle='--', linewidth=2,
          label=f'Media: {atributo.mean():.2f}')
ax.set_title('Distribución del Atributo', fontweight='bold', fontsize=14)
ax.set_xlabel('starkey_min', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

#%%
# ============================================================
# PASO 3: CLUSTERIZACIÓN
# ============================================================
print("\n" + "="*70)
print("🔵 PASO 2: CLUSTERIZACIÓN ESPACIAL")
print("="*70)

print("\n📌 ¿Qué hace la clusterización?")
print("   La clusterización agrupa los datos en dominios homogéneos")
print("   considerando tanto la ubicación espacial como el atributo.")
print("   Usamos K-means ponderado que balancea:")
print("   • Proximidad espacial (coordenadas X, Z)")
print("   • Similitud en el atributo (starkey_min)")

# Parámetros de clustering
n_clusters = 5
w_spatial = 0.65  # 65% peso espacial, 35% peso de atributo

print(f"\n⚙️  Parámetros:")
print(f"   • Número de clusters: {n_clusters}")
print(f"   • Peso espacial: {w_spatial} (65% espacio, 35% atributo)")

# Crear y entrenar el clusterer
clusterer = ClusterKmeans(n_clusters=n_clusters, w_spatial=w_spatial)
clusterer.fit(x, z, atributo)

print("\n✅ Clustering completado")
print(f"   Se han identificado {n_clusters} dominios")

# Obtener estadísticas
stats = clusterer.get_stats()
print("\n📊 Estadísticas por cluster:")
for i, stat in stats.items():
    print(f"\n   Cluster {i}:")
    print(f"      • Puntos: {stat['n_points']}")
    print(f"      • Media: {stat['mean']:.2f}")
    print(f"      • Std: {stat['std']:.2f}")
    print(f"      • CV: {stat['efecto_proporcional']:.3f}")

# Visualización de clusters
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Clusters espaciales
ax = axes[0]
scatter = ax.scatter(x, z, c=clusterer.clusters, cmap='viridis',
                    s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
ax.set_title(f'Clusters Espaciales (k={n_clusters}, w={w_spatial})',
            fontweight='bold', fontsize=14)
ax.set_xlabel('X (midx)', fontsize=12)
ax.set_ylabel('Z (midz)', fontsize=12)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster')

# Panel 2: Comparación con atributo original
ax = axes[1]
scatter = ax.scatter(x, z, c=atributo, cmap='RdYlBu_r',
                    s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
ax.set_title('Atributo Original', fontweight='bold', fontsize=14)
ax.set_xlabel('X (midx)', fontsize=12)
ax.set_ylabel('Z (midz)', fontsize=12)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='starkey_min')

plt.tight_layout()
plt.show()

# Métricas globales
metricas = clusterer.get_global_metrics()
print(f"\n📈 Métricas globales:")
print(f"   • Std promedio: {metricas['std_prom']:.2f}")
print(f"   • CV promedio: {metricas['cv_prom']:.3f}")

#%%
# ============================================================
# PASO 4: DIVISIÓN TRAIN/TEST
# ============================================================
print("\n" + "="*70)
print("🔀 PASO 3: DIVISIÓN TRAIN/TEST")
print("="*70)

print("\n📌 ¿Por qué dividir los datos?")
print("   Para evaluar el desempeño del modelo de estimación necesitamos:")
print("   • 80% de datos para ENTRENAR el modelo")
print("   • 20% de datos para PROBAR el modelo (nunca vistos)")

# División estratificada por cluster
indices = np.arange(len(x))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=clusterer.clusters  # Mantener proporciones de clusters
)

# Extraer train y test
x_train, x_test = x[train_idx], x[test_idx]
z_train, z_test = z[train_idx], z[test_idx]
attr_train, attr_test = atributo[train_idx], atributo[test_idx]
clusters_train, clusters_test = clusterer.clusters[train_idx], clusterer.clusters[test_idx]

print(f"\n📊 División completada:")
print(f"   • Datos de entrenamiento: {len(x_train)} puntos ({len(x_train)/len(x)*100:.1f}%)")
print(f"   • Datos de test: {len(x_test)} puntos ({len(x_test)/len(x)*100:.1f}%)")

# Verificar distribución de clusters
print(f"\n📊 Distribución por cluster:")
for i in range(n_clusters):
    n_train = np.sum(clusters_train == i)
    n_test = np.sum(clusters_test == i)
    total = n_train + n_test
    print(f"   Cluster {i}: {n_train} train ({n_train/total*100:.1f}%), "
          f"{n_test} test ({n_test/total*100:.1f}%)")

# Visualización de la división
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x_train, z_train, c='blue', s=30, alpha=0.6,
          label=f'Train ({len(x_train)} pts)', edgecolors='k', linewidth=0.3)
ax.scatter(x_test, z_test, c='red', s=80, alpha=0.8, marker='s',
          label=f'Test ({len(x_test)} pts)', edgecolors='k', linewidth=0.5)
ax.set_title('División Train/Test', fontweight='bold', fontsize=14)
ax.set_xlabel('X (midx)', fontsize=12)
ax.set_ylabel('Z (midz)', fontsize=12)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# ============================================================
# PASO 5: ESTIMACIÓN GLOBAL CON KNN
# ============================================================
print("\n" + "="*70)
print("🎯 PASO 4: ESTIMACIÓN GLOBAL CON KNN")
print("="*70)

print("\n📌 ¿Qué hace KNN?")
print("   K-Nearest Neighbors (KNN) estima el valor de un punto desconocido")
print("   usando el promedio ponderado de sus K vecinos más cercanos.")
print("   • Ventaja: Simple, rápido, no asume distribución estadística")
print("   • Desventaja: Sensible a outliers y no genera incertidumbre")

# Parámetros KNN
n_neighbors = 10
print(f"\n⚙️  Parámetros:")
print(f"   • Número de vecinos (k): {n_neighbors}")
print(f"   • Ponderación: Por distancia (vecinos cercanos pesan más)")

# Crear y entrenar estimador KNN
estimador_knn = EstimadorEspacial(metodo='knn', n_neighbors=n_neighbors)
estimador_knn.fit(x_train, z_train, attr_train)

print(f"\n✅ Modelo KNN entrenado con {len(x_train)} puntos")

# Predecir en conjunto de test
print("\n🔮 Realizando predicciones en conjunto de test...")
pred_knn = estimador_knn.predict(x_test, z_test)

# Calcular métricas
mae_knn = mean_absolute_error(attr_test, pred_knn)
rmse_knn = np.sqrt(mean_squared_error(attr_test, pred_knn))
r2_knn = r2_score(attr_test, pred_knn)

print("\n📊 MÉTRICAS EN DATOS DE TEST:")
print(f"   • MAE (Error Absoluto Medio): {mae_knn:.3f}")
print(f"     → En promedio, nos equivocamos en {mae_knn:.3f} unidades")
print(f"   • RMSE (Raíz del Error Cuadrático Medio): {rmse_knn:.3f}")
print(f"     → Penaliza más los errores grandes")
print(f"   • R² (Coeficiente de Determinación): {r2_knn:.3f}")
print(f"     → El modelo explica {r2_knn*100:.1f}% de la varianza")

# Visualización de resultados
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Predicho vs Real
ax = axes[0]
ax.scatter(attr_test, pred_knn, alpha=0.6, s=60, edgecolors='k', linewidth=0.5)
min_val = min(attr_test.min(), pred_knn.min())
max_val = max(attr_test.max(), pred_knn.max())
ax.plot([min_val, max_val], [min_val, max_val],
       'r--', linewidth=2, label='Predicción perfecta')
ax.set_title(f'KNN: Predicho vs Real\nMAE={mae_knn:.2f}, RMSE={rmse_knn:.2f}, R²={r2_knn:.3f}',
            fontweight='bold', fontsize=13)
ax.set_xlabel('Valor Real', fontsize=12)
ax.set_ylabel('Valor Predicho', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# Panel 2: Distribución de errores
ax = axes[1]
errores = attr_test - pred_knn
ax.hist(errores, bins=30, alpha=0.7, color='steelblue', edgecolor='k')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
ax.axvline(errores.mean(), color='orange', linestyle='--', linewidth=2,
          label=f'Media = {errores.mean():.2f}')
ax.set_title('Distribución de Errores', fontweight='bold', fontsize=13)
ax.set_xlabel('Error (Real - Predicho)', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

#%%
# ============================================================
# PASO 6: ESTIMACIÓN POR CLUSTER
# ============================================================
print("\n" + "="*70)
print("🎯 PASO 5: ESTIMACIÓN POR DOMINIO (CLUSTER)")
print("="*70)

print("\n📌 ¿Por qué estimar por cluster?")
print("   Cada cluster tiene características propias (media, std diferentes).")
print("   Al estimar dentro de cada dominio:")
print("   • Mejoramos la precisión local")
print("   • Respetamos la homogeneidad de cada dominio")
print("   • Evitamos contaminar estimaciones entre dominios distintos")

# Resultados por cluster
resultados_clusters = {}

print(f"\n🔧 Estimando en {n_clusters} clusters...")

for cluster_id in range(n_clusters):
    print(f"\n{'─'*60}")
    print(f"📍 CLUSTER {cluster_id}")
    print(f"{'─'*60}")
    
    # Filtrar datos de este cluster
    mask_train = clusters_train == cluster_id
    mask_test = clusters_test == cluster_id
    
    x_train_c = x_train[mask_train]
    z_train_c = z_train[mask_train]
    attr_train_c = attr_train[mask_train]
    
    x_test_c = x_test[mask_test]
    z_test_c = z_test[mask_test]
    attr_test_c = attr_test[mask_test]
    
    print(f"   • Puntos de entrenamiento: {len(x_train_c)}")
    print(f"   • Puntos de test: {len(x_test_c)}")
    
    # Solo estimar si hay suficientes datos
    if len(x_test_c) == 0:
        print(f"   ⚠️  Sin datos de test en este cluster, saltando...")
        continue
    
    if len(x_train_c) < n_neighbors:
        print(f"   ⚠️  Pocos datos de entrenamiento ({len(x_train_c)} < {n_neighbors})")
        print(f"      Ajustando k a {len(x_train_c)}")
        k_actual = len(x_train_c)
    else:
        k_actual = n_neighbors
    
    # Entrenar estimador para este cluster
    estimador_c = EstimadorEspacial(metodo='knn', n_neighbors=k_actual)
    estimador_c.fit(x_train_c, z_train_c, attr_train_c)
    
    # Predecir
    pred_c = estimador_c.predict(x_test_c, z_test_c)
    
    # Métricas
    mae_c = mean_absolute_error(attr_test_c, pred_c)
    rmse_c = np.sqrt(mean_squared_error(attr_test_c, pred_c))
    r2_c = r2_score(attr_test_c, pred_c)
    
    print(f"   • MAE:  {mae_c:.3f}")
    print(f"   • RMSE: {rmse_c:.3f}")
    print(f"   • R²:   {r2_c:.3f}")
    
    # Guardar resultados
    resultados_clusters[cluster_id] = {
        'n_train': len(x_train_c),
        'n_test': len(x_test_c),
        'mae': mae_c,
        'rmse': rmse_c,
        'r2': r2_c,
        'predicciones': pred_c,
        'reales': attr_test_c
    }

#%%
# ============================================================
# PASO 7: COMPARACIÓN GLOBAL VS POR CLUSTER
# ============================================================
print("\n" + "="*70)
print("📊 PASO 6: COMPARACIÓN DE ENFOQUES")
print("="*70)

print("\n🔍 Comparando dos estrategias:")
print("   1. Estimación GLOBAL: Un solo modelo KNN para todos los datos")
print("   2. Estimación POR CLUSTER: Un modelo KNN independiente por dominio")

# Calcular métricas promedio ponderado por cluster
maes_cluster = []
rmses_cluster = []
r2s_cluster = []
pesos = []

for cluster_id, resultado in resultados_clusters.items():
    if 'mae' in resultado:
        maes_cluster.append(resultado['mae'])
        rmses_cluster.append(resultado['rmse'])
        r2s_cluster.append(resultado['r2'])
        pesos.append(resultado['n_test'])

# Promedios ponderados
total_test = sum(pesos)
mae_clusters_prom = np.average(maes_cluster, weights=pesos)
rmse_clusters_prom = np.average(rmses_cluster, weights=pesos)
r2_clusters_prom = np.average(r2s_cluster, weights=pesos)

# Tabla comparativa
print("\n" + "="*70)
print("📋 TABLA COMPARATIVA")
print("="*70)

comparacion = pd.DataFrame({
    'Método': ['Global (1 modelo)', 'Por Cluster (5 modelos)'],
    'MAE': [mae_knn, mae_clusters_prom],
    'RMSE': [rmse_knn, rmse_clusters_prom],
    'R²': [r2_knn, r2_clusters_prom]
})

print("\n" + comparacion.to_string(index=False))

# Calcular mejoras
mejora_mae = ((mae_knn - mae_clusters_prom) / mae_knn) * 100
mejora_rmse = ((rmse_knn - rmse_clusters_prom) / rmse_knn) * 100
mejora_r2 = ((r2_clusters_prom - r2_knn) / abs(r2_knn)) * 100

print(f"\n📈 MEJORAS AL USAR CLUSTERING:")
print(f"   • MAE:  {mejora_mae:+.2f}% ({'mejor' if mejora_mae > 0 else 'peor'})")
print(f"   • RMSE: {mejora_rmse:+.2f}% ({'mejor' if mejora_rmse > 0 else 'peor'})")
print(f"   • R²:   {mejora_r2:+.2f}% ({'mejor' if mejora_r2 > 0 else 'peor'})")

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Métricas por cluster
ax = axes[0]
x_pos = list(resultados_clusters.keys())
maes = [resultados_clusters[i]['mae'] for i in x_pos]
colors = plt.cm.viridis(np.linspace(0, 1, len(x_pos)))
bars = ax.bar(x_pos, maes, color=colors, alpha=0.7, edgecolor='k')
ax.axhline(mae_knn, color='red', linestyle='--', linewidth=2,
          label=f'MAE Global: {mae_knn:.2f}')
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('MAE por Cluster vs Global', fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

# Panel 2: Comparación de métodos
ax = axes[1]
metodos = ['Global', 'Por Cluster']
maes_comp = [mae_knn, mae_clusters_prom]
rmses_comp = [rmse_knn, rmse_clusters_prom]
x_pos = np.arange(len(metodos))
width = 0.35
bars1 = ax.bar(x_pos - width/2, maes_comp, width, label='MAE',
              color='steelblue', alpha=0.7, edgecolor='k')
bars2 = ax.bar(x_pos + width/2, rmses_comp, width, label='RMSE',
              color='coral', alpha=0.7, edgecolor='k')
ax.set_xlabel('Método', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_title('Comparación de Errores', fontweight='bold', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(metodos)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

#%%
# ============================================================
# PASO 8: VISUALIZACIÓN ESPACIAL DE PREDICCIONES
# ============================================================
print("\n" + "="*70)
print("🗺️  PASO 7: VISUALIZACIÓN ESPACIAL")
print("="*70)

print("\n📌 Visualizando predicciones en el espacio...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: Datos de test reales
ax = axes[0, 0]
scatter = ax.scatter(x_test, z_test, c=attr_test, cmap='RdYlBu_r',
                    s=100, alpha=0.8, edgecolors='k', linewidth=0.8)
ax.set_title('Valores REALES (Test)', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)', fontsize=11)
ax.set_ylabel('Z (midz)', fontsize=11)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='starkey_min')

# Panel 2: Predicciones globales
ax = axes[0, 1]
scatter = ax.scatter(x_test, z_test, c=pred_knn, cmap='RdYlBu_r',
                    s=100, alpha=0.8, edgecolors='k', linewidth=0.8)
ax.set_title(f'Predicciones GLOBALES\nRMSE={rmse_knn:.2f}',
            fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)', fontsize=11)
ax.set_ylabel('Z (midz)', fontsize=11)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Predicho')

# Panel 3: Clusters en espacio
ax = axes[1, 0]
scatter = ax.scatter(x_test, z_test, c=clusters_test, cmap='viridis',
                    s=100, alpha=0.8, edgecolors='k', linewidth=0.8)
ax.set_title('Clusters de Test', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)', fontsize=11)
ax.set_ylabel('Z (midz)', fontsize=11)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster')

# Panel 4: Errores espaciales
ax = axes[1, 1]
errores_espaciales = attr_test - pred_knn
scatter = ax.scatter(x_test, z_test, c=errores_espaciales, cmap='RdYlGn_r',
                    s=100, alpha=0.8, edgecolors='k', linewidth=0.8,
                    vmin=-abs(errores_espaciales).max(),
                    vmax=abs(errores_espaciales).max())
ax.set_title('Errores de Predicción\n(Rojo=Subestimado, Verde=Sobreestimado)',
            fontweight='bold', fontsize=12)
ax.set_xlabel('X (midx)', fontsize=11)
ax.set_ylabel('Z (midz)', fontsize=11)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Error')

plt.tight_layout()
plt.show()

#%%
# ============================================================
# PASO 9: RESUMEN Y CONCLUSIONES
# ============================================================
print("\n" + "="*70)
print("📝 RESUMEN Y CONCLUSIONES")
print("="*70)

print(f"""
✅ PIPELINE COMPLETADO CON ÉXITO

🔹 DATOS PROCESADOS:
   • Total de puntos: {len(df)}
   • Entrenamiento: {len(x_train)} ({len(x_train)/len(x)*100:.1f}%)
   • Test: {len(x_test)} ({len(x_test)/len(x)*100:.1f}%)

🔹 CLUSTERIZACIÓN:
   • Número de clusters: {n_clusters}
   • Peso espacial: {w_spatial}
   • Std promedio: {metricas['std_prom']:.2f}

🔹 ESTIMACIÓN GLOBAL (KNN):
   • Vecinos (k): {n_neighbors}
   • MAE: {mae_knn:.3f}
   • RMSE: {rmse_knn:.3f}
   • R²: {r2_knn:.3f}

🔹 ESTIMACIÓN POR CLUSTER:
   • Modelos entrenados: {len(resultados_clusters)}
   • MAE promedio: {mae_clusters_prom:.3f}
   • RMSE promedio: {rmse_clusters_prom:.3f}
   • R² promedio: {r2_clusters_prom:.3f}

🔹 MEJORA AL USAR CLUSTERING:
   • MAE: {mejora_mae:+.2f}%
   • RMSE: {mejora_rmse:+.2f}%
   • R²: {mejora_r2:+.2f}%

📌 INTERPRETACIÓN:
   
   1. CLUSTERIZACIÓN:
      La clusterización agrupa los datos en {n_clusters} dominios espaciales
      que son homogéneos en términos del atributo starkey_min.
   
   2. ESTIMACIÓN:
      KNN usa los {n_neighbors} vecinos más cercanos para predecir valores.
      Al estimar dentro de cada cluster, respetamos la homogeneidad local.
   
   3. RESULTADOS:
      {'La estimación por cluster MEJORA' if mejora_mae > 0 else 'La estimación global es MEJOR'}
      los resultados, reduciendo el error en {abs(mejora_mae):.1f}%.
      
      Esto indica que {'los dominios tienen características distintas' if mejora_mae > 0 else 'los datos son bastante homogéneos'}
      {'y se benefician de modelos especializados.' if mejora_mae > 0 else 'y un modelo global es suficiente.'}

🎯 PRÓXIMOS PASOS:
   1. Probar diferentes valores de k (vecinos)
   2. Experimentar con otros números de clusters
   3. Comparar con otros métodos (IDW, Kriging)
   4. Generar estimaciones en grilla completa
   5. Cuantificar incertidumbre de las predicciones
""")

print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO")
print("="*70)

#%%
print("\n🎉 ¡Script ejecutado exitosamente!")
print("📊 Todos los resultados han sido generados y visualizados.")