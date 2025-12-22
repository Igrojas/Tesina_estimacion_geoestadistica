#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from typing import Tuple, Optional, Dict

warnings.filterwarnings('ignore')

# Kriging
try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    print("⚠️  Instalar pykrige: pip install pykrige")
    PYKRIGE_AVAILABLE = False

#%%
# ═══════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════

def cargar_y_preparar_datos(ruta: str,
                            variable_objetivo: str,
                            aplicar_log: bool = False,
                            filtro_min: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Carga y prepara los datos para análisis
    
    Parámetros
    ----------
    ruta : str
        Ruta al archivo CSV
    variable_objetivo : str
        Nombre de la variable objetivo (ej: 'cus', 'starkey_min', etc.)
    aplicar_log : bool
        Si True, aplica logaritmo a la variable objetivo
    filtro_min : float
        Valor mínimo para filtrar datos (solo valores > filtro_min)
        
    Retorna
    -------
    x, y, z, atributo, nombre_variable
    """
    print("="*70)
    print("📁 CARGANDO DATOS")
    print("="*70)
    
    df = pd.read_csv(ruta, sep=';')
    print(f"  ✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Verificar que la variable existe
    if variable_objetivo not in df.columns:
        raise ValueError(f"Variable '{variable_objetivo}' no encontrada en los datos. "
                        f"Columnas disponibles: {list(df.columns)}")
    
    # Seleccionar columnas necesarias
    columnas_necesarias = ["midx", "midy", "midz", variable_objetivo]
    df = df[columnas_necesarias].copy()
    
    # Limpiar datos
    mask = (df[variable_objetivo] > filtro_min) & np.isfinite(df[columnas_necesarias]).all(axis=1)
    df = df[mask].reset_index(drop=True)
    
    # Extraer variables
    x = df['midx'].values
    y = df['midy'].values
    z = df['midz'].values
    atributo = df[variable_objetivo].values
    
    # Aplicar logaritmo si se solicita
    nombre_variable = variable_objetivo
    if aplicar_log:
        atributo = np.log(atributo)
        nombre_variable = f"log({variable_objetivo})"
    
    print(f"  ✓ Datos válidos: {len(df)} puntos")
    print(f"  ✓ Variable objetivo: {nombre_variable}")
    print(f"  ✓ Rango: [{atributo.min():.4f}, {atributo.max():.4f}]")
    print(f"  ✓ Media: {atributo.mean():.4f}, Desv.Est.: {atributo.std():.4f}")
    
    return x, y, z, atributo, nombre_variable

def entrenar_knn(X_train: np.ndarray,
                 y_train: np.ndarray,
                 n_neighbors: int = 15,
                 weights: str = 'distance',
                 p: int = 2) -> KNeighborsRegressor:
    """Entrena modelo KNN"""
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    knn.fit(X_train, y_train)
    return knn

def entrenar_kriging(x_train: np.ndarray,
                     y_train: np.ndarray,
                     z_train: np.ndarray,
                     variogram_model: str = 'spherical',
                     nlags: int = 15) -> Optional[OrdinaryKriging]:
    """Entrena modelo de Kriging"""
    if not PYKRIGE_AVAILABLE:
        return None
    
    # Escalar coordenadas
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_train.reshape(-1, 1)).flatten()
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    ok = OrdinaryKriging(
        x_scaled, y_scaled, z_train,
        variogram_model=variogram_model,
        nlags=nlags,
        verbose=False,
        enable_plotting=False
    )
    
    return ok, scaler_x, scaler_y

def predecir_kriging(ok: OrdinaryKriging,
                     scaler_x: StandardScaler,
                     scaler_y: StandardScaler,
                     x_pred: np.ndarray,
                     y_pred: np.ndarray,
                     tipo: str = 'points') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predice usando kriging
    
    tipo: 'points' o 'grid'
    """
    x_scaled = scaler_x.transform(x_pred.reshape(-1, 1)).flatten()
    y_scaled = scaler_y.transform(y_pred.reshape(-1, 1)).flatten()
    
    if tipo == 'points':
        z_pred, var_pred = ok.execute('points', x_scaled, y_scaled)
    else:  # grid
        z_pred, var_pred = ok.execute('grid', x_scaled, y_scaled)
    
    return z_pred, var_pred

def visualizar_resultados(x: np.ndarray,
                         y: np.ndarray,
                         atributo: np.ndarray,
                         grid_x: np.ndarray,
                         grid_y: np.ndarray,
                         z_knn_grid: np.ndarray,
                         z_kriging_grid: Optional[np.ndarray],
                         var_kriging_grid: Optional[np.ndarray],
                         y_test: np.ndarray,
                         y_pred_knn: np.ndarray,
                         y_pred_kriging: Optional[np.ndarray],
                         r2_knn: float,
                         rmse_knn: float,
                         r2_kriging: Optional[float],
                         rmse_kriging: Optional[float],
                         nombre_variable: str):
    """Genera todas las visualizaciones"""
    
    print("\n📊 Generando visualizaciones...\n")
    
    vmin = atributo.min()
    vmax = atributo.max()
    
    # Figura 1: Comparación de predicciones
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Datos observados
    ax = axes[0]
    sc = ax.scatter(x, y, c=atributo, cmap='turbo', s=15, 
                    vmin=vmin, vmax=vmax, edgecolors='black', linewidths=0.3)
    ax.set_title('1. Datos Observados', fontsize=12, fontweight='bold')
    ax.set_xlabel('midx (Este)', fontsize=10)
    ax.set_ylabel('midy (Norte)', fontsize=10)
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label=nombre_variable)
    
    # Predicción KNN
    ax = axes[1]
    im = ax.imshow(z_knn_grid, origin='lower',
                   extent=(x.min(), x.max(), y.min(), y.max()),
                   aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax)
    ax.scatter(x, y, c='white', s=10, alpha=0.3, edgecolors='black', linewidths=0.3)
    ax.set_title(f'2. Predicción KNN\n(R²={r2_knn:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('midx (Este)', fontsize=10)
    ax.set_ylabel('midy (Norte)', fontsize=10)
    plt.colorbar(im, ax=ax, label=nombre_variable)
    
    # Predicción Kriging
    ax = axes[2]
    if z_kriging_grid is not None:
        im = ax.imshow(z_kriging_grid, origin='lower',
                       extent=(x.min(), x.max(), y.min(), y.max()),
                       aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax.scatter(x, y, c='white', s=5, alpha=0.3, edgecolors='black', linewidths=0.3)
        r2_str = f"{r2_kriging:.3f}" if r2_kriging is not None else "N/A"
        ax.set_title(f'3. Predicción Kriging\n(R²={r2_str})', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label=nombre_variable)
    else:
        ax.text(0.5, 0.5, 'Kriging no disponible', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('3. Predicción Kriging', fontsize=12, fontweight='bold')
    ax.set_xlabel('midx (Este)', fontsize=10)
    ax.set_ylabel('midy (Norte)', fontsize=10)
    
    plt.suptitle(f'Comparación: KNN vs Kriging - {nombre_variable}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Figura 2: Validación (scatter plots)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KNN
    ax = axes[0]
    ax.scatter(y_test, y_pred_knn, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    min_val = min(y_test.min(), y_pred_knn.min())
    max_val = max(y_test.max(), y_pred_knn.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('Observado', fontsize=11)
    ax.set_ylabel('Predicho', fontsize=11)
    ax.set_title(f'KNN: Predicción vs Real\n(R²={r2_knn:.3f}, RMSE={rmse_knn:.2f})', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Kriging
    ax = axes[1]
    if y_pred_kriging is not None:
        ax.scatter(y_test, y_pred_kriging, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
        min_val = min(y_test.min(), y_pred_kriging.min())
        max_val = max(y_test.max(), y_pred_kriging.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        r2_str = f"{r2_kriging:.3f}" if r2_kriging is not None else "N/A"
        rmse_str = f"{rmse_kriging:.2f}" if rmse_kriging is not None else "N/A"
        ax.set_title(f'Kriging: Predicción vs Real\n(R²={r2_str}, RMSE={rmse_str})', 
                    fontsize=12, fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Kriging no disponible', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Kriging: Predicción vs Real', fontsize=12, fontweight='bold')
    ax.set_xlabel('Observado', fontsize=11)
    ax.set_ylabel('Predicho', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Validación: Predicción vs Observado - {nombre_variable}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Figura 3: Incertidumbre (solo Kriging)
    if var_kriging_grid is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Varianza
        ax = axes[0]
        im = ax.imshow(var_kriging_grid, origin='lower',
                       extent=(x.min(), x.max(), y.min(), y.max()),
                       aspect='auto', cmap='YlOrRd')
        ax.scatter(x, y, c='blue', s=5, alpha=0.4, marker='x')
        ax.set_title('Incertidumbre (Varianza)', fontsize=12, fontweight='bold')
        ax.set_xlabel('midx (Este)', fontsize=10)
        ax.set_ylabel('midy (Norte)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Varianza')
        
        # Desviación estándar
        ax = axes[1]
        im = ax.imshow(np.sqrt(var_kriging_grid), origin='lower',
                       extent=(x.min(), x.max(), y.min(), y.max()),
                       aspect='auto', cmap='YlOrRd')
        ax.scatter(x, y, c='blue', s=5, alpha=0.4, marker='x')
        ax.set_title('Incertidumbre (Desv. Est.)', fontsize=12, fontweight='bold')
        ax.set_xlabel('midx (Este)', fontsize=10)
        ax.set_ylabel('midy (Norte)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Desv. Est.')
        
        plt.suptitle(f'Incertidumbre de Kriging - {nombre_variable}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

#%%
# ═══════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main(ruta: str = 'data/raw/bd_dm_cmp_entry.csv',
         variable_objetivo: str = 'cus',
         aplicar_log: bool = False,
         filtro_min: float = 0.0,
         test_size: float = 0.3,
         random_state: int = 42,
         # Parámetros KNN
         n_neighbors: int = 15,
         knn_weights: str = 'distance',
         knn_p: int = 2,
         # Parámetros Kriging
         variogram_model: str = 'spherical',
         nlags: int = 15,
         # Parámetros visualización
         grid_size: int = 100,
         mostrar_graficos: bool = True):
    """
    Función principal para ejecutar análisis KNN y Kriging
    
    Parámetros
    ----------
    ruta : str
        Ruta al archivo CSV
    variable_objetivo : str
        Nombre de la variable objetivo ('cus', 'starkey_min', etc.)
    aplicar_log : bool
        Si True, aplica logaritmo a la variable objetivo
    filtro_min : float
        Valor mínimo para filtrar datos
    test_size : float
        Proporción de datos para test (0.0 a 1.0)
    random_state : int
        Semilla para reproducibilidad
    n_neighbors : int
        Número de vecinos para KNN
    knn_weights : str
        Tipo de pesos para KNN ('uniform' o 'distance')
    knn_p : int
        Parámetro p para distancia Minkowski (1=Manhattan, 2=Euclidiana)
    variogram_model : str
        Modelo de variograma ('spherical', 'exponential', 'gaussian', 'linear')
    nlags : int
        Número de lags para variograma
    grid_size : int
        Tamaño de la malla para predicciones
    mostrar_graficos : bool
        Si True, muestra los gráficos
    """
    
    # ── 1. Cargar y preparar datos ──
    x, y, z, atributo, nombre_variable = cargar_y_preparar_datos(
        ruta=ruta,
        variable_objetivo=variable_objetivo,
        aplicar_log=aplicar_log,
        filtro_min=filtro_min
    )
    
    # Visualización inicial
    if mostrar_graficos:
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(x, y, c=atributo, cmap='turbo', s=20, 
                        edgecolors='black', linewidths=0.3)
        plt.xlabel("midx (Este)", fontsize=11)
        plt.ylabel("midy (Norte)", fontsize=11)
        plt.colorbar(sc, label=nombre_variable)
        plt.title(f"Distribución Espacial de {nombre_variable}", 
                 fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ── 2. Dividir datos ──
    print("\n" + "="*70)
    print("✂️  DIVIDIENDO DATOS")
    print("="*70)
    
    scaler_coord = StandardScaler()
    coords = np.column_stack([x, y])
    coords_scaled = scaler_coord.fit_transform(coords)
    
    X_train, X_test, y_train, y_test = train_test_split(
        coords_scaled, atributo, test_size=test_size, random_state=random_state
    )
    
    print(f"  ✓ Train: {len(X_train)} puntos")
    print(f"  ✓ Test: {len(X_test)} puntos")
    
    # ── 3. Entrenar KNN ──
    print("\n" + "="*70)
    print("🔍 APLICANDO K-NEAREST NEIGHBORS (KNN)")
    print("="*70)
    
    knn = entrenar_knn(X_train, y_train, n_neighbors=n_neighbors, 
                       weights=knn_weights, p=knn_p)
    y_pred_knn = knn.predict(X_test)
    
    r2_knn = r2_score(y_test, y_pred_knn)
    rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    
    print(f"  ✓ KNN entrenado (k={n_neighbors}, weights={knn_weights}, p={knn_p})")
    print(f"  ✓ R²: {r2_knn:.4f}")
    print(f"  ✓ RMSE: {rmse_knn:.4f}")
    
    # ── 4. Entrenar Kriging ──
    print("\n" + "="*70)
    print("🗺️  APLICANDO KRIGING")
    print("="*70)
    
    if PYKRIGE_AVAILABLE:
        x_train_orig = scaler_coord.inverse_transform(X_train)[:, 0]
        y_train_orig = scaler_coord.inverse_transform(X_train)[:, 1]
        
        resultado_kriging = entrenar_kriging(
            x_train_orig, y_train_orig, y_train,
            variogram_model=variogram_model,
            nlags=nlags
        )
        
        if resultado_kriging is not None:
            ok, scaler_x, scaler_y = resultado_kriging
            
            # Predecir en test
            x_test_orig = scaler_coord.inverse_transform(X_test)[:, 0]
            y_test_orig = scaler_coord.inverse_transform(X_test)[:, 1]
            y_pred_kriging, var_kriging = predecir_kriging(
                ok, scaler_x, scaler_y, x_test_orig, y_test_orig, tipo='points'
            )
            
            r2_kriging = r2_score(y_test, y_pred_kriging)
            rmse_kriging = np.sqrt(mean_squared_error(y_test, y_pred_kriging))
            
            print(f"  ✓ Kriging entrenado (modelo={variogram_model}, nlags={nlags})")
            print(f"  ✓ R²: {r2_kriging:.4f}")
            print(f"  ✓ RMSE: {rmse_kriging:.4f}")
            
            # Parámetros del variograma
            params = ok.variogram_model_parameters
            print(f"  ✓ Variograma - Sill: {params[0]:.4f}, Range: {params[1]:.4f}, Nugget: {params[2]:.4f}")
        else:
            y_pred_kriging = None
            var_kriging = None
            r2_kriging = None
            rmse_kriging = None
            ok = None
            scaler_x = None
            scaler_y = None
    else:
        print("  ⚠️  Kriging no disponible (pykrige no instalado)")
        y_pred_kriging = None
        var_kriging = None
        r2_kriging = None
        rmse_kriging = None
        ok = None
        scaler_x = None
        scaler_y = None
    
    # ── 5. Comparación de resultados ──
    print("\n" + "="*70)
    print("📊 COMPARACIÓN DE RESULTADOS")
    print("="*70)
    print(f"{'Método':<15} {'R²':<10} {'RMSE':<10}")
    print("-" * 35)
    print(f"{'KNN':<15} {r2_knn:<10.4f} {rmse_knn:<10.4f}")
    if r2_kriging is not None:
        print(f"{'Kriging':<15} {r2_kriging:<10.4f} {rmse_kriging:<10.4f}")
    print("="*70)
    
    # ── 6. Predicciones en malla ──
    if mostrar_graficos:
        print("\n🗺️  Generando predicciones en malla...")
        gridx = np.linspace(x.min(), x.max(), grid_size)
        gridy = np.linspace(y.min(), y.max(), grid_size)
        grid_x, grid_y = np.meshgrid(gridx, gridy)
        
        # KNN en malla
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        grid_points_scaled = scaler_coord.transform(grid_points)
        z_knn_grid = knn.predict(grid_points_scaled).reshape(grid_x.shape)
        
        # Kriging en malla
        if ok is not None and scaler_x is not None and scaler_y is not None:
            gridx_scaled = scaler_x.transform(gridx.reshape(-1, 1)).flatten()
            gridy_scaled = scaler_y.transform(gridy.reshape(-1, 1)).flatten()
            z_kriging_grid, var_kriging_grid = ok.execute('grid', gridx_scaled, gridy_scaled)
        else:
            z_kriging_grid = None
            var_kriging_grid = None
        
        # Visualizar
        visualizar_resultados(
            x, y, atributo, grid_x, grid_y,
            z_knn_grid, z_kriging_grid, var_kriging_grid,
            y_test, y_pred_knn, y_pred_kriging,
            r2_knn, rmse_knn, r2_kriging, rmse_kriging,
            nombre_variable
        )
    
    print("\n✅ Análisis completado!")
    
    return {
        'r2_knn': r2_knn,
        'rmse_knn': rmse_knn,
        'r2_kriging': r2_kriging,
        'rmse_kriging': rmse_kriging,
        'nombre_variable': nombre_variable
    }

#%%
# ═══════════════════════════════════════════════════════════
# EJECUCIÓN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # ═══════════════════════════════════════════════════════
    # CONFIGURACIÓN - CAMBIA ESTOS PARÁMETROS AQUÍ
    # ═══════════════════════════════════════════════════════
    
    resultados = main(
        # ── Datos ──
        ruta='data/raw/bd_dm_cmp_entry.csv',
        variable_objetivo='cus',  # Cambia aquí: 'cus', 'starkey_min', etc.
        aplicar_log=False,  # True para aplicar logaritmo
        filtro_min=0.0,  # Valor mínimo para filtrar
        
        # ── División de datos ──
        test_size=0.3,
        random_state=42,
        
        # ── Parámetros KNN ──
        n_neighbors=15,  # Número de vecinos
        knn_weights='distance',  # 'uniform' o 'distance'
        knn_p=2,  # 1=Manhattan, 2=Euclidiana
        
        # ── Parámetros Kriging ──
        variogram_model='spherical',  # 'spherical', 'exponential', 'gaussian', 'linear'
        nlags=15,  # Número de lags
        
        # ── Visualización ──
        grid_size=100,  # Tamaño de malla
        mostrar_graficos=True
    )
    
    print(f"\n🎯 Resultados finales:")
    print(f"   KNN - R²: {resultados['r2_knn']:.4f}, RMSE: {resultados['rmse_knn']:.4f}")
    if resultados['r2_kriging'] is not None:
        print(f"   Kriging - R²: {resultados['r2_kriging']:.4f}, RMSE: {resultados['rmse_kriging']:.4f}")
