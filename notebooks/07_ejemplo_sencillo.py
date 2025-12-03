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

from src import clustering, estimacion, visualizacion, interpolacion
import importlib
importlib.reload(clustering)
importlib.reload(estimacion)
importlib.reload(visualizacion)
importlib.reload(interpolacion)

from src.clustering import ClusterKmeans
from src.estimacion import EstimadorEspacial
from src.visualizacion import VisualizadorClusters
from src.interpolacion import InterpoladorEspacial

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
scatter = ax.scatter(x, z, c=atributo, cmap='RdYlBu_r', s=30, alpha=0.8)
ax.set_title('Distribución Espacial del Atributo', fontweight='bold')
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
plt.colorbar(scatter, ax=ax, label='starkey_min')
plt.grid(alpha=0.3)
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
n_clusters = 4              # Número de grupos a crear
w_spatial = 0.8             # 65% peso espacial, 35% peso atributo

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

visualizador = VisualizadorClusters()
# visualizador.plot_clusters(clusterer)
# visualizador.plot_atributo_real(clusterer)
# visualizador.plot_comparacion(clusterer)
visualizador.crear_dashboard(clusterer)

#%%

# Interpolar
interpolador = InterpoladorEspacial(clusterer,
                             n_neighbors=20,   
                             n_points=100)
interpolador.interpolar()
df_interpolado = interpolador.get_dataframe()
interpolador.print_info()

visualizador.plot_interpolacion(interpolador)

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# Asumiendo que tu dataframe se llama 'df'
# df = pd.read_csv('tu_archivo.csv') o como lo tengas cargado

# =============================================================================
# 1. PREPARACIÓN DE DATOS
# =============================================================================

# Separar datos originales y fantasmas
df_original = df_interpolado[df_interpolado['tipo'] == 'original'].copy()
df_fantasma = df_interpolado[df_interpolado['tipo'] == 'fantasma'].copy()

print(f"Datos originales: {len(df_original)}")
print(f"Datos fantasma: {len(df_fantasma)}")
print(f"Clusters únicos: {sorted(df_interpolado['cluster'].unique())}")

# =============================================================================
# 2. FUNCIÓN PARA ESTIMAR UN CLUSTER ESPECÍFICO
# =============================================================================

def estimar_cluster_knn(df_original, df_fantasma, cluster_id, n_neighbors=5, 
                        cv_folds=5, n_neighbors_range=range(3, 21)):
    """
    Estima valores para puntos fantasma de un cluster usando KNN
    
    Parámetros:
    -----------
    df_original : DataFrame con datos originales
    df_fantasma : DataFrame con datos fantasma
    cluster_id : ID del cluster a estimar
    n_neighbors : número de vecinos (si es None, se optimiza)
    cv_folds : número de folds para validación cruzada
    n_neighbors_range : rango de K para optimizar
    
    Retorna:
    --------
    dict con resultados de estimación y métricas
    """
    
    # Filtrar datos del cluster específico
    orig_cluster = df_original[df_original['cluster'] == cluster_id].copy()
    fant_cluster = df_fantasma[df_fantasma['cluster'] == cluster_id].copy()
    
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*70}")
    print(f"Puntos originales: {len(orig_cluster)}")
    print(f"Puntos fantasma: {len(fant_cluster)}")
    
    # Preparar datos de entrenamiento
    X_train = orig_cluster[['x', 'z']].values
    y_train = orig_cluster['variable'].values
    
    # Preparar datos para predicción
    X_fantasma = fant_cluster[['x', 'z']].values
    
    # -------------------------------------------------------------------------
    # OPTIMIZACIÓN DE K (si no se especifica)
    # -------------------------------------------------------------------------
    if n_neighbors is None:
        print("\nOptimizando número de vecinos K...")
        best_k = None
        best_score = -np.inf
        cv_scores = []
        
        for k in n_neighbors_range:
            if k >= len(orig_cluster):
                break
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            # Validación cruzada con score R²
            scores = cross_val_score(knn, X_train, y_train, 
                                    cv=min(cv_folds, len(orig_cluster)),
                                    scoring='r2')
            mean_score = scores.mean()
            cv_scores.append((k, mean_score))
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        
        n_neighbors = best_k
        print(f"Mejor K encontrado: {best_k} (R² CV: {best_score:.4f})")
    
    # -------------------------------------------------------------------------
    # VALIDACIÓN CRUZADA PARA EVALUAR EL MODELO
    # -------------------------------------------------------------------------
    print(f"\nEntrenando KNN con K={n_neighbors}...")
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    
    # Realizar validación cruzada para obtener métricas
    kfold = KFold(n_splits=min(cv_folds, len(orig_cluster)), shuffle=True, random_state=42)
    
    cv_predictions = []
    cv_actuals = []
    
    for train_idx, test_idx in kfold.split(X_train):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_test_fold = X_train[test_idx]
        y_test_fold = y_train[test_idx]
        
        knn_temp = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        knn_temp.fit(X_train_fold, y_train_fold)
        
        y_pred_fold = knn_temp.predict(X_test_fold)
        cv_predictions.extend(y_pred_fold)
        cv_actuals.extend(y_test_fold)
    
    cv_predictions = np.array(cv_predictions)
    cv_actuals = np.array(cv_actuals)
    
    # Calcular métricas de validación cruzada
    mae_cv = mean_absolute_error(cv_actuals, cv_predictions)
    rmse_cv = np.sqrt(mean_squared_error(cv_actuals, cv_predictions))
    r2_cv = r2_score(cv_actuals, cv_predictions)
    mape_cv = np.mean(np.abs((cv_actuals - cv_predictions) / cv_actuals)) * 100
    
    print(f"\nMÉTRICAS DE VALIDACIÓN CRUZADA:")
    print(f"  MAE  (Error Absoluto Medio):        {mae_cv:.4f}")
    print(f"  RMSE (Raíz del Error Cuadrático):   {rmse_cv:.4f}")
    print(f"  R²   (Coeficiente Determinación):   {r2_cv:.4f}")
    print(f"  MAPE (Error Porcentual Absoluto):   {mape_cv:.2f}%")
    
    # -------------------------------------------------------------------------
    # ENTRENAMIENTO FINAL Y PREDICCIÓN EN FANTASMAS
    # -------------------------------------------------------------------------
    knn.fit(X_train, y_train)
    
    # Predecir en puntos fantasma
    y_fantasma_pred = knn.predict(X_fantasma)
    
    print(f"\nEstadísticas de predicciones en fantasmas:")
    print(f"  Mínimo: {y_fantasma_pred.min():.2f}")
    print(f"  Máximo: {y_fantasma_pred.max():.2f}")
    print(f"  Media:  {y_fantasma_pred.mean():.2f}")
    print(f"  Desv:   {y_fantasma_pred.std():.2f}")
    
    # Retornar resultados
    resultados = {
        'cluster_id': cluster_id,
        'n_neighbors': n_neighbors,
        'X_train': X_train,
        'y_train': y_train,
        'X_fantasma': X_fantasma,
        'y_fantasma_pred': y_fantasma_pred,
        'cv_predictions': cv_predictions,
        'cv_actuals': cv_actuals,
        'metricas': {
            'MAE': mae_cv,
            'RMSE': rmse_cv,
            'R2': r2_cv,
            'MAPE': mape_cv
        }
    }
    
    return resultados

# =============================================================================
# 3. FUNCIÓN PARA GRAFICAR RESULTADOS DE UN CLUSTER
# =============================================================================

def graficar_cluster(resultados):
    """Grafica resultados de estimación para un cluster"""
    
    cluster_id = resultados['cluster_id']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ----- Subplot 1: Mapa espacial -----
    ax1 = axes[0]
    
    # Puntos originales
    scatter1 = ax1.scatter(resultados['X_train'][:, 0], 
                          resultados['X_train'][:, 1],
                          c=resultados['y_train'], 
                          s=100, 
                          cmap='viridis',
                          edgecolors='black',
                          linewidth=1.5,
                          label='Datos Originales',
                          marker='o',
                          vmin=min(resultados['y_train'].min(), 
                                  resultados['y_fantasma_pred'].min()),
                          vmax=max(resultados['y_train'].max(), 
                                  resultados['y_fantasma_pred'].max()))
    
    # Puntos fantasma estimados
    scatter2 = ax1.scatter(resultados['X_fantasma'][:, 0], 
                          resultados['X_fantasma'][:, 1],
                          c=resultados['y_fantasma_pred'], 
                          s=100, 
                          cmap='viridis',
                          edgecolors='red',
                          linewidth=1.5,
                          label='Fantasmas Estimados',
                          marker='s',
                          vmin=min(resultados['y_train'].min(), 
                                  resultados['y_fantasma_pred'].min()),
                          vmax=max(resultados['y_train'].max(), 
                                  resultados['y_fantasma_pred'].max()))
    
    plt.colorbar(scatter1, ax=ax1, label='Variable')
    ax1.set_xlabel('Coordenada X', fontsize=12)
    ax1.set_ylabel('Coordenada Z', fontsize=12)
    ax1.set_title(f'Distribución Espacial - Cluster {cluster_id}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # ----- Subplot 2: Validación Cruzada (Real vs Predicho) -----
    ax2 = axes[1]
    
    ax2.scatter(resultados['cv_actuals'], 
               resultados['cv_predictions'],
               alpha=0.6, 
               s=80,
               edgecolors='black',
               linewidth=0.5)
    
    # Línea 1:1
    min_val = min(resultados['cv_actuals'].min(), resultados['cv_predictions'].min())
    max_val = max(resultados['cv_actuals'].max(), resultados['cv_predictions'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Línea 1:1')
    
    # Añadir métricas en el gráfico
    metricas = resultados['metricas']
    texto_metricas = f"R² = {metricas['R2']:.3f}\nRMSE = {metricas['RMSE']:.3f}\nMAE = {metricas['MAE']:.3f}"
    ax2.text(0.05, 0.95, texto_metricas, 
            transform=ax2.transAxes, 
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Valores Reales', fontsize=12)
    ax2.set_ylabel('Valores Predichos (CV)', fontsize=12)
    ax2.set_title(f'Validación Cruzada - Cluster {cluster_id}', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # ----- Subplot 3: Histograma de Errores -----
    ax3 = axes[2]
    
    errores = resultados['cv_predictions'] - resultados['cv_actuals']
    
    ax3.hist(errores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    ax3.axvline(x=errores.mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Media = {errores.mean():.3f}')
    
    ax3.set_xlabel('Error (Predicho - Real)', fontsize=12)
    ax3.set_ylabel('Frecuencia', fontsize=12)
    ax3.set_title(f'Distribución de Errores - Cluster {cluster_id}', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. ESTIMACIÓN PARA UN CLUSTER ESPECÍFICO
# =============================================================================

# Ejemplo: estimar cluster 0
print("\n" + "="*70)
print("ESTIMACIÓN PARA UN CLUSTER ESPECÍFICO")
print("="*70)

cluster_a_estimar = 0  # Cambia esto por el cluster que quieras

resultados_cluster_0 = estimar_cluster_knn(
    df_original, 
    df_fantasma, 
    cluster_id=cluster_a_estimar,
    n_neighbors=None,  # None para optimizar automáticamente
    cv_folds=5
)

# Graficar resultados
graficar_cluster(resultados_cluster_0)

# =============================================================================
# 5. ESTIMACIÓN PARA TODOS LOS CLUSTERS
# =============================================================================

print("\n" + "="*70)
print("ESTIMACIÓN PARA TODOS LOS CLUSTERS")
print("="*70)

clusters_unicos = sorted(df_original['cluster'].unique())
resultados_todos = {}

for cluster_id in clusters_unicos:
    resultados_todos[cluster_id] = estimar_cluster_knn(
        df_original, 
        df_fantasma, 
        cluster_id=cluster_id,
        n_neighbors=None,  # Optimiza K automáticamente
        cv_folds=5
    )

# =============================================================================
# 6. GRÁFICA CONSOLIDADA DE TODOS LOS CLUSTERS
# =============================================================================

def graficar_todos_clusters(resultados_todos, df_original, df_fantasma):
    """Crea gráficas consolidadas para todos los clusters"""
    
    n_clusters = len(resultados_todos)
    
    # Figura 1: Comparación de métricas entre clusters
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
    
    clusters = list(resultados_todos.keys())
    metricas_nombres = ['MAE', 'RMSE', 'R2', 'MAPE']
    
    for idx, metrica in enumerate(metricas_nombres):
        valores = [resultados_todos[c]['metricas'][metrica] for c in clusters]
        
        ax = axes1[idx]
        bars = ax.bar(clusters, valores, edgecolor='black', linewidth=1.5, alpha=0.7)
        
        # Colorear barras
        for bar in bars:
            bar.set_color('steelblue')
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel(metrica, fontsize=12)
        ax.set_title(f'{metrica} por Cluster', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras
        for i, v in enumerate(valores):
            ax.text(clusters[i], v, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Figura 2: Mapa espacial consolidado
    fig2, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Combinar todos los datos
    all_X_train = []
    all_y_train = []
    all_clusters_train = []
    
    all_X_fantasma = []
    all_y_fantasma = []
    all_clusters_fantasma = []
    
    for cluster_id, res in resultados_todos.items():
        all_X_train.append(res['X_train'])
        all_y_train.append(res['y_train'])
        all_clusters_train.extend([cluster_id] * len(res['y_train']))
        
        all_X_fantasma.append(res['X_fantasma'])
        all_y_fantasma.append(res['y_fantasma_pred'])
        all_clusters_fantasma.extend([cluster_id] * len(res['y_fantasma_pred']))
    
    all_X_train = np.vstack(all_X_train)
    all_y_train = np.hstack(all_y_train)
    
    all_X_fantasma = np.vstack(all_X_fantasma)
    all_y_fantasma = np.hstack(all_y_fantasma)
    
    # Graficar originales
    scatter1 = ax.scatter(all_X_train[:, 0], all_X_train[:, 1],
                         c=all_y_train, 
                         s=120, 
                         cmap='viridis',
                         edgecolors='black',
                         linewidth=2,
                         label='Datos Originales',
                         marker='o',
                         vmin=min(all_y_train.min(), all_y_fantasma.min()),
                         vmax=max(all_y_train.max(), all_y_fantasma.max()))
    
    # Graficar fantasmas
    scatter2 = ax.scatter(all_X_fantasma[:, 0], all_X_fantasma[:, 1],
                         c=all_y_fantasma, 
                         s=120, 
                         cmap='viridis',
                         edgecolors='red',
                         linewidth=2,
                         label='Fantasmas Estimados',
                         marker='s',
                         vmin=min(all_y_train.min(), all_y_fantasma.min()),
                         vmax=max(all_y_train.max(), all_y_fantasma.max()))
    
    cbar = plt.colorbar(scatter1, ax=ax)
    cbar.set_label('Variable', fontsize=13, fontweight='bold')
    
    ax.set_xlabel('Coordenada X', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coordenada Z', fontsize=13, fontweight='bold')
    ax.set_title('Mapa de Estimación - Todos los Clusters', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Figura 3: Real vs Predicho consolidado
    fig3, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for idx, (cluster_id, res) in enumerate(resultados_todos.items()):
        ax.scatter(res['cv_actuals'], 
                  res['cv_predictions'],
                  alpha=0.6, 
                  s=100,
                  edgecolors='black',
                  linewidth=0.5,
                  label=f'Cluster {cluster_id}',
                  color=colors[idx])
    
    # Línea 1:1
    all_cv_actuals = np.hstack([res['cv_actuals'] for res in resultados_todos.values()])
    all_cv_predictions = np.hstack([res['cv_predictions'] for res in resultados_todos.values()])
    
    min_val = min(all_cv_actuals.min(), all_cv_predictions.min())
    max_val = max(all_cv_actuals.max(), all_cv_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Línea 1:1')
    
    # Métricas globales
    mae_global = mean_absolute_error(all_cv_actuals, all_cv_predictions)
    rmse_global = np.sqrt(mean_squared_error(all_cv_actuals, all_cv_predictions))
    r2_global = r2_score(all_cv_actuals, all_cv_predictions)
    
    texto_metricas = f"MÉTRICAS GLOBALES:\nR² = {r2_global:.3f}\nRMSE = {rmse_global:.3f}\nMAE = {mae_global:.3f}"
    ax.text(0.05, 0.95, texto_metricas, 
            transform=ax.transAxes, 
            fontsize=13,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontweight='bold')
    
    ax.set_xlabel('Valores Reales', fontsize=13, fontweight='bold')
    ax.set_ylabel('Valores Predichos (CV)', fontsize=13, fontweight='bold')
    ax.set_title('Validación Cruzada - Todos los Clusters', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resumen de métricas
    print("\n" + "="*70)
    print("RESUMEN DE MÉTRICAS POR CLUSTER")
    print("="*70)
    print(f"{'Cluster':<10} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'MAPE (%)':<12} {'K':<5}")
    print("-"*70)
    for cluster_id, res in resultados_todos.items():
        m = res['metricas']
        print(f"{cluster_id:<10} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} {m['R2']:<12.4f} {m['MAPE']:<12.2f} {res['n_neighbors']:<5}")
    
    print("\n" + "="*70)
    print("MÉTRICAS GLOBALES (TODOS LOS CLUSTERS)")
    print("="*70)
    print(f"MAE Global:  {mae_global:.4f}")
    print(f"RMSE Global: {rmse_global:.4f}")
    print(f"R² Global:   {r2_global:.4f}")

# Ejecutar visualización consolidada
graficar_todos_clusters(resultados_todos, df_original, df_fantasma)

# =============================================================================
# 7. GUARDAR RESULTADOS
# =============================================================================

# Crear DataFrame con las estimaciones de los fantasmas
estimaciones_final = []

for cluster_id, res in resultados_todos.items():
    fant_cluster = df_fantasma[df_fantasma['cluster'] == cluster_id].copy()
    fant_cluster['variable_estimada'] = res['y_fantasma_pred']
    estimaciones_final.append(fant_cluster)

df_estimaciones = pd.concat(estimaciones_final, ignore_index=True)

# Mostrar primeras filas
print("\n" + "="*70)
print("PRIMERAS ESTIMACIONES")
print("="*70)
print(df_estimaciones.head(10))

# Opcional: Guardar a archivo
# df_estimaciones.to_csv('estimaciones_knn.csv', index=False)
# print("\nEstimaciones guardadas en 'estimaciones_knn.csv'")
# %%
