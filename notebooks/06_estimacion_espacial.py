# ============================================================
# CELDA 1: Setup e instalación
# ============================================================
# Si no tienes pykrige instalado, ejecuta:
# !pip install pykrige
#%%
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.estimacion import EstimadorEspacial
from sklearn.model_selection import train_test_split

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
print(f"📏 Atributo - Media: {atributo.mean():.2f}, Std: {atributo.std():.2f}")
print(f"📏 Atributo - Min: {atributo.min():.2f}, Max: {atributo.max():.2f}")

# ============================================================
# CELDA 3: Dividir datos en train/test
# ============================================================
# Para validación, separamos 20% para test
indices = np.arange(len(x))
train_idx, test_idx = train_test_split(
    indices, 
    test_size=0.2, 
    random_state=42
)

x_train, x_test = x[train_idx], x[test_idx]
z_train, z_test = z[train_idx], z[test_idx]
attr_train, attr_test = atributo[train_idx], atributo[test_idx]

print(f"\n📊 División de datos:")
print(f"   • Entrenamiento: {len(x_train)} puntos")
print(f"   • Test: {len(x_test)} puntos")

# Visualizar train/test
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x_train, z_train, c='blue', s=30, alpha=0.6, label='Train', edgecolors='k', linewidth=0.3)
ax.scatter(x_test, z_test, c='red', s=80, alpha=0.8, marker='s', label='Test', edgecolors='k', linewidth=0.5)
ax.set_xlabel('X (midx)', fontsize=12)
ax.set_ylabel('Z (midz)', fontsize=12)
ax.set_title('División Train/Test', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# CELDA 4: MÉTODO 1 - KNN (K-Nearest Neighbors)
# ============================================================
print("\n" + "="*70)
print("🔵 MÉTODO 1: K-NEAREST NEIGHBORS (KNN)")
print("="*70)

# Crear estimador KNN
estimador_knn = EstimadorEspacial(metodo='knn', n_neighbors=10)

# Entrenar
estimador_knn.fit(x_train, z_train, attr_train)
print(estimador_knn)

# Predecir en puntos de test
pred_knn = estimador_knn.predict(x_test, z_test)

# Calcular errores
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_knn = mean_absolute_error(attr_test, pred_knn)
rmse_knn = np.sqrt(mean_squared_error(attr_test, pred_knn))
r2_knn = r2_score(attr_test, pred_knn)

print(f"\n📊 Métricas en datos de TEST:")
print(f"   • MAE:  {mae_knn:.3f}")
print(f"   • RMSE: {rmse_knn:.3f}")
print(f"   • R²:   {r2_knn:.3f}")

# ============================================================
# CELDA 5: MÉTODO 2 - IDW (Inverse Distance Weighting)
# ============================================================
print("\n" + "="*70)
print("🟢 MÉTODO 2: INVERSE DISTANCE WEIGHTING (IDW)")
print("="*70)

# Crear estimador IDW
estimador_idw = EstimadorEspacial(metodo='idw', power=2)

# Entrenar
estimador_idw.fit(x_train, z_train, attr_train)
print(estimador_idw)

# Predecir
pred_idw = estimador_idw.predict(x_test, z_test)

# Métricas
mae_idw = mean_absolute_error(attr_test, pred_idw)
rmse_idw = np.sqrt(mean_squared_error(attr_test, pred_idw))
r2_idw = r2_score(attr_test, pred_idw)

print(f"\n📊 Métricas en datos de TEST:")
print(f"   • MAE:  {mae_idw:.3f}")
print(f"   • RMSE: {rmse_idw:.3f}")
print(f"   • R²:   {r2_idw:.3f}")

# ============================================================
# CELDA 6: MÉTODO 3 - KRIGING (Ordinary Kriging)
# ============================================================
print("\n" + "="*70)
print("🟡 MÉTODO 3: ORDINARY KRIGING")
print("="*70)

# Crear estimador Kriging
estimador_kriging = EstimadorEspacial(
    metodo='kriging', 
    variogram_model='spherical'
)

# Entrenar (esto puede tomar unos segundos)
estimador_kriging.fit(x_train, z_train, attr_train)
print(estimador_kriging)

# Predecir (con varianza)
pred_kriging, var_kriging = estimador_kriging.predict(
    x_test, z_test, 
    return_variance=True
)

# Métricas
mae_kriging = mean_absolute_error(attr_test, pred_kriging)
rmse_kriging = np.sqrt(mean_squared_error(attr_test, pred_kriging))
r2_kriging = r2_score(attr_test, pred_kriging)

print(f"\n📊 Métricas en datos de TEST:")
print(f"   • MAE:  {mae_kriging:.3f}")
print(f"   • RMSE: {rmse_kriging:.3f}")
print(f"   • R²:   {r2_kriging:.3f}")

# ============================================================
# CELDA 7: Comparación de métodos (Tabla)
# ============================================================
print("\n" + "="*70)
print("📊 COMPARACIÓN DE MÉTODOS")
print("="*70)

resultados = pd.DataFrame({
    'Método': ['KNN', 'IDW', 'Kriging'],
    'MAE': [mae_knn, mae_idw, mae_kriging],
    'RMSE': [rmse_knn, rmse_idw, rmse_kriging],
    'R²': [r2_knn, r2_idw, r2_kriging]
})

# Identificar mejor método por métrica
resultados['Mejor_MAE'] = resultados['MAE'] == resultados['MAE'].min()
resultados['Mejor_RMSE'] = resultados['RMSE'] == resultados['RMSE'].min()
resultados['Mejor_R2'] = resultados['R²'] == resultados['R²'].max()

print("\n" + resultados.to_string(index=False))

# Guardar
# resultados.to_excel('../results/tables/05_comparacion_metodos.xlsx', index=False)
print("\n✅ Tabla guardada en: results/tables/05_comparacion_metodos.xlsx")

# ============================================================
# CELDA 8: Gráficos de comparación - Predicho vs Real
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metodos = [
    ('KNN', pred_knn, mae_knn, rmse_knn, r2_knn),
    ('IDW', pred_idw, mae_idw, rmse_idw, r2_idw),
    ('Kriging', pred_kriging, mae_kriging, rmse_kriging, r2_kriging)
]

for idx, (nombre, pred, mae, rmse, r2) in enumerate(metodos):
    ax = axes[idx]
    
    # Scatter predicho vs real
    ax.scatter(attr_test, pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Línea diagonal (predicción perfecta)
    min_val = min(attr_test.min(), pred.min())
    max_val = max(attr_test.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', linewidth=2, label='Predicción perfecta')
    
    # Títulos y etiquetas
    ax.set_title(f'{nombre}\nMAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}',
                fontweight='bold', fontsize=11)
    ax.set_xlabel('Valor Real', fontsize=11)
    ax.set_ylabel('Valor Predicho', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Aspect ratio igual
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
# plt.savefig('../results/figures/05_comparacion_predicciones.png', 
#            dpi=150, bbox_inches='tight')
plt.show()

print("✅ Gráfico guardado: results/figures/05_comparacion_predicciones.png")

# ============================================================
# CELDA 9: Distribución de errores
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

errores = [
    ('KNN', attr_test - pred_knn),
    ('IDW', attr_test - pred_idw),
    ('Kriging', attr_test - pred_kriging)
]

for idx, (nombre, error) in enumerate(errores):
    ax = axes[idx]
    
    # Histograma de errores
    ax.hist(error, bins=30, alpha=0.7, color='steelblue', edgecolor='k')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    ax.axvline(error.mean(), color='orange', linestyle='--', linewidth=2, 
              label=f'Media = {error.mean():.2f}')
    
    ax.set_title(f'Distribución de Errores - {nombre}', 
                fontweight='bold', fontsize=12)
    ax.set_xlabel('Error (Real - Predicho)', fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
# plt.savefig('../results/figures/05_distribucion_errores.png',
#            dpi=150, bbox_inches='tight')
plt.show()

print("✅ Gráfico guardado: results/figures/05_distribucion_errores.png")

# ============================================================
# CELDA 10: Estimación en grilla completa - KRIGING
# ============================================================
print("\n" + "="*70)
print("🗺️  ESTIMACIÓN EN GRILLA COMPLETA (KRIGING)")
print("="*70)

# Usar TODOS los datos para la grilla final
estimador_kriging_completo = EstimadorEspacial(
    metodo='kriging',
    variogram_model='spherical'
)
estimador_kriging_completo.fit(x, z, atributo)

# Estimar en grilla
print("🔧 Generando grilla de estimación...")
xx, zz, valores_kriging, varianza_kriging = estimador_kriging_completo.estimar_grilla(
    n_points=100,
    return_variance=True
)

print(f"✅ Grilla generada: {xx.shape}")

# ============================================================
# CELDA 11: Visualización de la estimación Kriging
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: Datos originales
ax = axes[0, 0]
scatter1 = ax.scatter(x, z, c=atributo, cmap='RdYlBu_r',
                     s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
ax.set_title('Datos Originales', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
plt.colorbar(scatter1, ax=ax, label='starkey_min')

# Panel 2: Estimación Kriging
ax = axes[0, 1]
contour = ax.contourf(xx, zz, valores_kriging, levels=20, cmap='RdYlBu_r')
ax.scatter(x, z, c='black', s=10, alpha=0.5, label='Datos')
ax.set_title('Estimación Kriging', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
ax.legend()
plt.colorbar(contour, ax=ax, label='Estimado')

# Panel 3: Varianza de Kriging (incertidumbre)
ax = axes[1, 0]
contour_var = ax.contourf(xx, zz, varianza_kriging, levels=20, cmap='Reds')
ax.scatter(x, z, c='black', s=10, alpha=0.5, label='Datos')
ax.set_title('Varianza de Kriging (Incertidumbre)', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
ax.legend()
plt.colorbar(contour_var, ax=ax, label='Varianza')

# Panel 4: Desviación estándar (raíz de varianza)
ax = axes[1, 1]
std_kriging = np.sqrt(varianza_kriging)
contour_std = ax.contourf(xx, zz, std_kriging, levels=20, cmap='Oranges')
ax.scatter(x, z, c='black', s=10, alpha=0.5, label='Datos')
ax.set_title('Desviación Estándar de Kriging', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
ax.legend()
plt.colorbar(contour_std, ax=ax, label='Std Dev')

plt.tight_layout()
# plt.savefig('../results/figures/05_estimacion_kriging_completa.png',
#            dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualización guardada: results/figures/05_estimacion_kriging_completa.png")

# ============================================================
# CELDA 12: Estadísticas de la estimación
# ============================================================
print("\n" + "="*70)
print("📊 ESTADÍSTICAS DE LA ESTIMACIÓN KRIGING")
print("="*70)

print(f"\n🔹 Datos Originales:")
print(f"   • Media: {atributo.mean():.2f}")
print(f"   • Std:   {atributo.std():.2f}")
print(f"   • Min:   {atributo.min():.2f}")
print(f"   • Max:   {atributo.max():.2f}")

print(f"\n🔹 Estimación Kriging:")
print(f"   • Media: {valores_kriging.mean():.2f}")
print(f"   • Std:   {valores_kriging.std():.2f}")
print(f"   • Min:   {valores_kriging.min():.2f}")
print(f"   • Max:   {valores_kriging.max():.2f}")

print(f"\n🔹 Incertidumbre (Std Dev):")
print(f"   • Media: {std_kriging.mean():.2f}")
print(f"   • Min:   {std_kriging.min():.2f}")
print(f"   • Max:   {std_kriging.max():.2f}")

# ============================================================
# CELDA 13: Comparación de todas las estimaciones en grilla
# ============================================================
print("\n🔧 Generando estimaciones con todos los métodos...")

# KNN en grilla
estimador_knn_completo = EstimadorEspacial(metodo='knn', n_neighbors=10)
estimador_knn_completo.fit(x, z, atributo)
xx_knn, zz_knn, valores_knn = estimador_knn_completo.estimar_grilla(n_points=100)

# IDW en grilla
estimador_idw_completo = EstimadorEspacial(metodo='idw', power=2)
estimador_idw_completo.fit(x, z, atributo)
xx_idw, zz_idw, valores_idw = estimador_idw_completo.estimar_grilla(n_points=100)

print("✅ Estimaciones completadas")

# Visualizar comparación
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: Datos originales
ax = axes[0, 0]
scatter = ax.scatter(x, z, c=atributo, cmap='RdYlBu_r',
                    s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
ax.set_title('DATOS ORIGINALES', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='starkey_min')

# Panel 2: KNN
ax = axes[0, 1]
contour = ax.contourf(xx_knn, zz_knn, valores_knn, levels=20, cmap='RdYlBu_r')
ax.scatter(x, z, c='black', s=5, alpha=0.3)
ax.set_title('ESTIMACIÓN KNN', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
plt.colorbar(contour, ax=ax, label='Estimado')

# Panel 3: IDW
ax = axes[1, 0]
contour = ax.contourf(xx_idw, zz_idw, valores_idw, levels=20, cmap='RdYlBu_r')
ax.scatter(x, z, c='black', s=5, alpha=0.3)
ax.set_title('ESTIMACIÓN IDW', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
plt.colorbar(contour, ax=ax, label='Estimado')

# Panel 4: Kriging
ax = axes[1, 1]
contour = ax.contourf(xx, zz, valores_kriging, levels=20, cmap='RdYlBu_r')
ax.scatter(x, z, c='black', s=5, alpha=0.3)
ax.set_title('ESTIMACIÓN KRIGING', fontweight='bold', fontsize=13)
ax.set_xlabel('X (midx)')
ax.set_ylabel('Z (midz)')
ax.grid(alpha=0.3)
plt.colorbar(contour, ax=ax, label='Estimado')

plt.tight_layout()
# plt.savefig('../results/figures/05_comparacion_todos_metodos.png',
#            dpi=150, bbox_inches='tight')
plt.show()

print("✅ Comparación guardada: results/figures/05_comparacion_todos_metodos.png")

# ============================================================
# CELDA 14: Validación cruzada (opcional, puede tardar)
# ============================================================
print("\n" + "="*70)
print("🔄 VALIDACIÓN CRUZADA (5-FOLD)")
print("="*70)

# KNN
print("\n🔵 Validando KNN...")
cv_knn = estimador_knn_completo.validacion_cruzada(n_folds=5)

# IDW
print("🟢 Validando IDW...")
cv_idw = estimador_idw_completo.validacion_cruzada(n_folds=5)

# Kriging (puede tardar más)
print("🟡 Validando Kriging...")
cv_kriging = estimador_kriging_completo.validacion_cruzada(n_folds=5)

print("\n" + "="*70)
print("📊 RESULTADOS DE VALIDACIÓN CRUZADA")
print("="*70)

cv_resultados = pd.DataFrame({
    'Método': ['KNN', 'IDW', 'Kriging'],
    'MAE': [cv_knn['MAE'], cv_idw['MAE'], cv_kriging['MAE']],
    'MAE_std': [cv_knn['MAE_std'], cv_idw['MAE_std'], cv_kriging['MAE_std']],
    'RMSE': [cv_knn['RMSE'], cv_idw['RMSE'], cv_kriging['RMSE']],
    'RMSE_std': [cv_knn['RMSE_std'], cv_idw['RMSE_std'], cv_kriging['RMSE_std']],
    'R²': [cv_knn['R2'], cv_idw['R2'], cv_kriging['R2']],
    'R²_std': [cv_knn['R2_std'], cv_idw['R2_std'], cv_kriging['R2_std']]
})

print("\n" + cv_resultados.to_string(index=False))

# Guardar
# cv_resultados.to_excel('../results/tables/05_validacion_cruzada.xlsx', index=False)
print("\n✅ Resultados guardados: results/tables/05_validacion_cruzada.xlsx")

# ============================================================
# CELDA 15: Conclusiones
# ============================================================
print("\n" + "="*70)
print("📝 CONCLUSIONES")
print("="*70)

mejor_metodo = resultados.loc[resultados['RMSE'].idxmin(), 'Método']

print(f"""
✅ MEJOR MÉTODO (según RMSE en test): {mejor_metodo}

🔹 INTERPRETACIÓN:
   • KNN: Rápido, simple, pero puede sobre-ajustar
   • IDW: Intermedio, depende del parámetro 'power'
   • Kriging: Más robusto, cuantifica incertidumbre

🔹 VENTAJA DE KRIGING:
   • Proporciona varianza de estimación
   • Permite identificar zonas de alta/baja incertidumbre
   • Es el estándar en geoestadística minera

🔹 PRÓXIMOS PASOS:
   1. Combinar clustering + estimación
   2. Estimar por dominio (cluster)
   3. Comparar estimación global vs por dominios
""")

print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO")
print("="*70)