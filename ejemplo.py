#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def cargar_datos(path, columnas, frac_muestreo=0.05, filtro_midy=7.4e6, random_state=0):
    df = pd.read_csv(path, sep=",")
    df = df[columnas].copy()
    df = df.sample(frac=frac_muestreo, random_state=random_state).reset_index(drop=True)
    df = df[df['midy'] > filtro_midy].reset_index(drop=True)
    mask = (df['cus'] > 0) & np.isfinite(df[['midx', 'midy', 'midz', 'cus']]).all(axis=1)
    df = df[mask].reset_index(drop=True)
    return df

def preparar_variables(df):
    x = df['midx'].values
    y = df['midy'].values
    z = df['midz'].values
    atributo = np.log(df['cus'].values)
    return x, y, z, atributo

def escalar_variables(x, y, z, atributo):
    scaler_coord = StandardScaler()
    coords = np.column_stack((x, y, z))
    coords_scaled = scaler_coord.fit_transform(coords)
    scaler_attr = StandardScaler()
    atributo_scaled = scaler_attr.fit_transform(atributo.reshape(-1, 1)).flatten()
    return coords_scaled, atributo_scaled, scaler_coord, scaler_attr

def dividir_train_test(coords_scaled, atributo_scaled, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        coords_scaled, atributo_scaled, test_size=test_size
    )
    return X_train, X_test, y_train, y_test

def entrenar_knn(X_train, y_train, n_neighbors=30):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", p=2)
    knn.fit(X_train, y_train)
    return knn

def evaluar_modelo(knn, X_test, y_test, scaler_attr):
    y_pred_scaled = knn.predict(X_test)
    y_pred = scaler_attr.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_true = scaler_attr.inverse_transform(y_test.reshape(-1,1)).flatten()
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\nValidación 80/20 - R2: {r2:.3f}, RMSE: {rmse:.3f}")
    return y_true, y_pred, r2, rmse

def crear_malla(coords_scaled, scaler_coord, grid_size=100):
    min_vals = coords_scaled[:, :2].min(axis=0)
    max_vals = coords_scaled[:, :2].max(axis=0)
    gridx = np.linspace(min_vals[0], max_vals[0], grid_size)
    gridy = np.linspace(min_vals[1], max_vals[1], grid_size)
    grid_x, grid_y = np.meshgrid(gridx, gridy)
    grid_z = np.zeros_like(grid_x) # El midz (z) para la malla; podrías interpolar o tomar un valor medio si quisieras
    grid_points_scaled = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()))
    grid_orig = scaler_coord.inverse_transform(grid_points_scaled)
    gridx_orig = grid_orig[:,0].reshape(grid_x.shape)
    gridy_orig = grid_orig[:,1].reshape(grid_y.shape)
    gridz_orig = grid_orig[:,2].reshape(grid_z.shape)
    return grid_x, grid_y, grid_z, grid_points_scaled, gridx_orig, gridy_orig, gridz_orig

def estimar_en_malla(knn, grid_points_scaled, scaler_attr, grid_shape):
    z_knn_scaled = knn.predict(grid_points_scaled).reshape(grid_shape)
    z_knn = scaler_attr.inverse_transform(z_knn_scaled.reshape(-1, 1)).reshape(grid_shape)
    return z_knn

def graficar_resultados(x, y, z, atributo, gridx_orig, gridy_orig, z_knn, titulo_malla="Estimación KNN (escala original)"):
    cmap = "turbo"
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Datos originales
    scatter0 = axs[0].scatter(x, y, c=atributo, cmap=cmap)
    axs[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.2)
    axs[0].set_title('Datos Sondajes (log cus)', fontsize=13)
    axs[0].set_xlabel('midx', fontsize=11)
    axs[0].set_ylabel('midy', fontsize=11)
    axs[0].tick_params(axis='both', labelsize=10)
    cbar0 = plt.colorbar(scatter0, ax=axs[0], orientation='vertical')
    cbar0.set_label('log(cus)', fontsize=11)

    # 2. Estimación KNN en grilla
    img = axs[1].imshow(
        z_knn, origin='lower',
        extent=(gridx_orig.min(), gridx_orig.max(), gridy_orig.min(), gridy_orig.max()),
        aspect='auto', cmap=cmap
    )
    axs[1].scatter(x, y, c='k', s=5, alpha=0.25, label='Datos')
    axs[1].set_title(titulo_malla, fontsize=13)
    axs[1].set_xlabel('midx', fontsize=11)
    axs[1].set_ylabel('midy', fontsize=11)
    axs[1].tick_params(axis='both', labelsize=10)
    cbar1 = plt.colorbar(img, ax=axs[1], orientation='vertical')
    cbar1.set_label('log(cus) estimado', fontsize=11)

    plt.tight_layout()
    plt.show()

def main(ruta, columnas, frac_muestreo=0.05, n_neighbors=30, grid_size=100):
    # 1. Cargar y preparar datos
    df = cargar_datos(ruta, columnas, frac_muestreo=frac_muestreo)
    x, y, z, atributo = preparar_variables(df)
    print(f"📊 Datos cargados: {len(df)} puntos")
    print(f"Estadísticas de log(cus): min={atributo.min():.3f}, max={atributo.max():.3f}, mean={atributo.mean():.3f}")
    # 2. Escalar
    coords_scaled, atributo_scaled, scaler_coord, scaler_attr = escalar_variables(x, y, z, atributo)
    # 3. Split train/test
    X_train, X_test, y_train, y_test = dividir_train_test(coords_scaled, atributo_scaled)
    # 4. Entrenar KNN
    knn = entrenar_knn(X_train, y_train, n_neighbors=n_neighbors)
    # 5. Validar modelo sobre test
    y_true, y_pred, r2, rmse = evaluar_modelo(knn, X_test, y_test, scaler_attr)
    # 6. Crear malla y estimar en la grilla
    grid_x, grid_y, grid_z, grid_points_scaled, gridx_orig, gridy_orig, gridz_orig = crear_malla(coords_scaled, scaler_coord, grid_size=grid_size)
    z_knn = estimar_en_malla(knn, grid_points_scaled, scaler_attr, grid_x.shape)
    # 7. Graficar resultados
    graficar_resultados(x, y, z, atributo, gridx_orig, gridy_orig, z_knn)

    # 8. Adicional: comparar histograma de datos reales (de entrenamiento) vs. estimaciones en la malla completa
    import seaborn as sns
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    
    # Scatter pred vs real TEST
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].set_xlabel('log(cus) Test Real')
    axes[0].set_ylabel('log(cus) Test Estimado')
    axes[0].set_title('KNN: Predicción vs Real (Test)')
    min_plot = min(np.min(y_true), np.min(y_pred))
    max_plot = max(np.max(y_true), np.max(y_pred))
    axes[0].plot([min_plot, max_plot], [min_plot, max_plot], 'k--')
    axes[0].grid(alpha=0.2)
    
    # Histogramas: comparar todos los datos de entrenamiento vs estimado en toda la malla
    # Real: use la variable "atributo" (todos los datos log(cus) originales)
    # Estimado: puntos de la grilla ("z_knn" aplanado)
    bins = np.linspace(
        min(np.min(atributo), np.min(z_knn)),
        max(np.max(atributo), np.max(z_knn)),
        30
    )
    n_real, _, _ = axes[1].hist(atributo, bins=bins, alpha=0.7, label='Entrenamiento (Real)', color='royalblue', edgecolor='black')
    n_estimado, _, _ = axes[1].hist(z_knn.flatten(), bins=bins, alpha=0.7, label='Malla (Estimado)', color='tomato', edgecolor='black')
    axes[1].set_xlabel('log(cus)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title(
        f'Histograma: Entrenamiento vs Malla\n'
        f'Total Real: {len(atributo)}, Total Estimado: {z_knn.size}'
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Opcional: Cambia frac_muestreo a voluntad
    main(
        ruta='data/raw/com_p_plt_entry 1.csv',
        columnas=['midx', 'midy', 'midz', 'cus'],
        frac_muestreo=0.15,    # Fracción de datos a muestrear
        n_neighbors=5,        # Número de vecinos para KNN
        grid_size=100,         # Tamaño de grilla para el mapa de estimación
    )
