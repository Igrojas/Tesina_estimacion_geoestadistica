#%%
"""
ANÁLISIS DE ANISOTROPÍA ESPACIAL CON VARIogramas DIRECCIONALES

Este script permite:
1. Analizar anisotropía espacial usando variogramas direccionales
2. Comparar KNN vs Kriging
3. Visualizar resultados en diferentes direcciones

CONCEPTO DE ANISOTROPÍA:
────────────────────────
- ISOTROPÍA: La correlación espacial es igual en todas las direcciones
- ANISOTROPÍA: La correlación varía según la dirección

Ejemplo práctico:
- Si el RANGE es mayor en dirección 0° (Este) que en 90° (Norte),
  significa que hay más correlación en dirección Este-Oeste.
- Esto puede indicar estructuras geológicas, vetas minerales, etc.

USO:
────
1. Para analizar anisotropía:
   resultados = analizar_anisotropia(variable_objetivo='cus')

2. Para análisis completo (KNN + Kriging):
   resultados = main(variable_objetivo='cus')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from typing import Tuple, Optional, Dict

# Scipy para estadísticas
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy no disponible, detección automática de log limitada")

warnings.filterwarnings('ignore')

# Kriging
try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_AVAILABLE = True
except ImportError:
    print("⚠️  Instalar pykrige: pip install pykrige")
    PYKRIGE_AVAILABLE = False

# PyTorch para Redes Neuronales
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  Instalar PyTorch: pip install torch")
    TORCH_AVAILABLE = False

#%%
# ═══════════════════════════════════════════════════════════
# RED NEURONAL PARA ESTIMACIÓN ESPACIAL
# ═══════════════════════════════════════════════════════════

class SpatialDataset(Dataset):
    """Dataset para datos espaciales"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SpatialNeuralNetwork(nn.Module):
    """
    Red neuronal para estimación espacial
    
    Arquitectura:
    - Input: coordenadas (x, y) o (x, y, z)
    - Capas ocultas: configurables
    - Output: valor de la variable objetivo
    """
    def __init__(self, input_dim: int = 2, 
                 hidden_dims: list = [64, 32, 16],
                 dropout: float = 0.2,
                 activation: str = 'relu'):
        super(SpatialNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Construir capas
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Función de activación
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            
            # Dropout para regularización
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Capa de salida
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

def entrenar_red_neuronal(X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None,
                         hidden_dims: list = [64, 32, 16],
                         dropout: float = 0.2,
                         learning_rate: float = 0.001,
                         batch_size: int = 32,
                         epochs: int = 100,
                         patience: int = 10,
                         verbose: bool = True) -> Tuple[nn.Module, Dict]:
    """
    Entrena una red neuronal para estimación espacial
    
    Parámetros
    ----------
    X_train : array
        Coordenadas de entrenamiento (n_samples, n_features)
    y_train : array
        Valores objetivo de entrenamiento
    X_val, y_val : arrays, optional
        Datos de validación
    hidden_dims : list
        Dimensiones de capas ocultas [64, 32, 16]
    dropout : float
        Tasa de dropout (0-1)
    learning_rate : float
        Tasa de aprendizaje
    batch_size : int
        Tamaño del batch
    epochs : int
        Número máximo de épocas
    patience : int
        Early stopping patience
    verbose : bool
        Mostrar progreso
        
    Retorna
    -------
    model : nn.Module
        Modelo entrenado
    history : dict
        Historial de entrenamiento
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch no está disponible")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear datasets
    train_dataset = SpatialDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val is not None:
        val_dataset = SpatialDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo
    input_dim = X_train.shape[1]
    model = SpatialNeuralNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)
    
    # Loss y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5)
    
    # Historial
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    # Early stopping
    no_improve = 0
    
    if verbose:
        print(f"\n  🧠 Entrenando red neuronal en {device}")
        print(f"     Arquitectura: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1")
        print(f"     Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entrenamiento
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validación
        if X_val is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                no_improve = 0
                # Guardar mejor modelo
                best_model_state = model.state_dict().copy()
            else:
                no_improve += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"     Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, "
                      f"Val Loss={val_loss:.6f}")
            
            if no_improve >= patience:
                if verbose:
                    print(f"     ⏹️  Early stopping en epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"     Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}")
    
    model.eval()
    if verbose:
        print(f"     ✅ Entrenamiento completado (mejor epoch: {history['best_epoch']+1})")
    
    return model, history

def predecir_red_neuronal(model: nn.Module,
                          X: np.ndarray,
                          batch_size: int = 512) -> np.ndarray:
    """
    Predice usando la red neuronal
    
    Parámetros
    ----------
    model : nn.Module
        Modelo entrenado
    X : array
        Coordenadas para predecir
    batch_size : int
        Tamaño del batch para predicción
        
    Retorna
    -------
    predictions : array
        Predicciones
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch no está disponible")
    
    device = next(model.parameters()).device
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)
            batch_pred = model(batch_X).cpu().numpy()
            predictions.append(batch_pred)
    
    return np.concatenate(predictions)

#%%
# ═══════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════

def cargar_y_preparar_datos(ruta: str,
                            variable_objetivo: str,
                            aplicar_log: Optional[bool] = None,
                            filtro_min: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Carga y prepara los datos para análisis
    
    Parámetros
    ----------
    ruta : str
        Ruta al archivo CSV
    variable_objetivo : str
        Nombre de la variable objetivo (ej: 'cus', 'starkey_min', 'cut', etc.)
    aplicar_log : bool, optional
        Si True, aplica logaritmo a la variable objetivo.
        Si None, detecta automáticamente si debe aplicar log (recomendado para 'cut' y variables con alta asimetría)
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
    atributo_original = df[variable_objetivo].values
    
    # Detectar si debe aplicar logaritmo automáticamente
    if aplicar_log is None:
        # Variables que típicamente requieren logaritmo
        variables_con_log = ['cut', 'cus', 'cu', 'au', 'ag', 'pb', 'zn']
        
        # Verificar si el nombre de la variable sugiere aplicar log
        aplicar_log_auto = variable_objetivo.lower() in [v.lower() for v in variables_con_log]
        
        # También verificar asimetría (coeficiente de asimetría > 1 sugiere distribución log-normal)
        if not aplicar_log_auto and len(atributo_original) > 10 and SCIPY_AVAILABLE:
            skewness = stats.skew(atributo_original)
            # Si la asimetría es alta (> 1.5), probablemente necesita log
            if skewness > 1.5:
                aplicar_log_auto = True
                print(f"     (asimetría detectada: {skewness:.2f})")
        
        aplicar_log = aplicar_log_auto
        
        if aplicar_log:
            print(f"  ℹ️  Se detectó que '{variable_objetivo}' se beneficia de transformación logarítmica")
            print(f"     (variable comúnmente log-normal o alta asimetría detectada)")
    elif aplicar_log:
        print(f"  ℹ️  Aplicando transformación logarítmica a '{variable_objetivo}' (especificado manualmente)")
    
    # Aplicar logaritmo si corresponde
    nombre_variable = variable_objetivo
    if aplicar_log:
        atributo = np.log(atributo_original)
        nombre_variable = f"log({variable_objetivo})"
    else:
        atributo = atributo_original
    
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
                         nombre_variable: str,
                         z_nn_grid: Optional[np.ndarray] = None,
                         y_pred_nn: Optional[np.ndarray] = None,
                         r2_nn: Optional[float] = None,
                         rmse_nn: Optional[float] = None):
    """Genera todas las visualizaciones"""
    
    print("\n📊 Generando visualizaciones...\n")
    
    vmin = atributo.min()
    vmax = atributo.max()
    
    # Determinar número de columnas según métodos disponibles
    n_metodos = 1  # KNN siempre está
    if z_kriging_grid is not None:
        n_metodos += 1
    if z_nn_grid is not None:
        n_metodos += 1
    
    # Figura 1: Comparación de predicciones
    fig, axes = plt.subplots(1, n_metodos + 1, figsize=(6 * (n_metodos + 1), 5))
    if n_metodos == 1:
        axes = axes.reshape(1, -1)
    
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
    idx = 2
    if z_kriging_grid is not None:
        ax = axes[idx]
        im = ax.imshow(z_kriging_grid, origin='lower',
                       extent=(x.min(), x.max(), y.min(), y.max()),
                       aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax.scatter(x, y, c='white', s=5, alpha=0.3, edgecolors='black', linewidths=0.3)
        r2_str = f"{r2_kriging:.3f}" if r2_kriging is not None else "N/A"
        ax.set_title(f'{idx+1}. Predicción Kriging\n(R²={r2_str})', fontsize=12, fontweight='bold')
        ax.set_xlabel('midx (Este)', fontsize=10)
        ax.set_ylabel('midy (Norte)', fontsize=10)
        plt.colorbar(im, ax=ax, label=nombre_variable)
        idx += 1
    
    # Predicción Red Neuronal
    if z_nn_grid is not None:
        ax = axes[idx]
        im = ax.imshow(z_nn_grid, origin='lower',
                       extent=(x.min(), x.max(), y.min(), y.max()),
                       aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax.scatter(x, y, c='white', s=5, alpha=0.3, edgecolors='black', linewidths=0.3)
        r2_str = f"{r2_nn:.3f}" if r2_nn is not None else "N/A"
        ax.set_title(f'{idx+1}. Predicción Red Neuronal\n(R²={r2_str})', fontsize=12, fontweight='bold')
        ax.set_xlabel('midx (Este)', fontsize=10)
        ax.set_ylabel('midy (Norte)', fontsize=10)
        plt.colorbar(im, ax=ax, label=nombre_variable)
    
    titulo = 'Comparación: KNN'
    if z_kriging_grid is not None:
        titulo += ' vs Kriging'
    if z_nn_grid is not None:
        titulo += ' vs Red Neuronal'
    titulo += f' - {nombre_variable}'
    
    plt.suptitle(titulo, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Figura 2: Validación (scatter plots)
    n_valid = 1  # KNN siempre
    if y_pred_kriging is not None:
        n_valid += 1
    if y_pred_nn is not None:
        n_valid += 1
    
    fig, axes = plt.subplots(1, n_valid, figsize=(7 * n_valid, 5))
    if n_valid == 1:
        axes = [axes]
    
    idx = 0
    # KNN
    ax = axes[idx]
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
    idx += 1
    
    # Kriging
    if y_pred_kriging is not None:
        ax = axes[idx]
        ax.scatter(y_test, y_pred_kriging, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
        min_val = min(y_test.min(), y_pred_kriging.min())
        max_val = max(y_test.max(), y_pred_kriging.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        r2_str = f"{r2_kriging:.3f}" if r2_kriging is not None else "N/A"
        rmse_str = f"{rmse_kriging:.2f}" if rmse_kriging is not None else "N/A"
        ax.set_title(f'Kriging: Predicción vs Real\n(R²={r2_str}, RMSE={rmse_str})', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_xlabel('Observado', fontsize=11)
        ax.set_ylabel('Predicho', fontsize=11)
        ax.grid(alpha=0.3)
        idx += 1
    
    # Red Neuronal
    if y_pred_nn is not None:
        ax = axes[idx]
        ax.scatter(y_test, y_pred_nn, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
        min_val = min(y_test.min(), y_pred_nn.min())
        max_val = max(y_test.max(), y_pred_nn.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        r2_str = f"{r2_nn:.3f}" if r2_nn is not None else "N/A"
        rmse_str = f"{rmse_nn:.2f}" if rmse_nn is not None else "N/A"
        ax.set_title(f'Red Neuronal: Predicción vs Real\n(R²={r2_str}, RMSE={rmse_str})', 
                    fontsize=12, fontweight='bold')
        ax.legend()
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
# VARIogramas DIRECCIONALES Y ANISOTROPÍA
# ═══════════════════════════════════════════════════════════

"""
EXPLICACIÓN: VARIogramas DIRECCIONALES Y ANISOTROPÍA

¿Qué es la anisotropía espacial?
────────────────────────────────
La anisotropía espacial ocurre cuando la correlación entre puntos varía según la DIRECCIÓN.
Por ejemplo:
- En geología: las estructuras pueden extenderse más en una dirección (ej: este-oeste)
- En minería: las vetas pueden seguir una dirección preferencial
- En agricultura: el suelo puede variar más rápido en una dirección que en otra

¿Cómo se estudia?
─────────────────
1. Se calculan variogramas en diferentes DIRECCIONES (0°, 45°, 90°, 135°, etc.)
2. Se comparan los parámetros (range, sill) entre direcciones
3. Si hay diferencias significativas → hay anisotropía

Direcciones comunes:
- 0° (Este): dirección horizontal (→)
- 45° (Noreste): diagonal
- 90° (Norte): dirección vertical (↑)
- 135° (Noroeste): diagonal

Si el RANGE es mayor en una dirección → hay más correlación en esa dirección
"""

def calcular_variogramas_direccionales(x: np.ndarray,
                                       y: np.ndarray,
                                       z: np.ndarray,
                                       direcciones: list = [0, 45, 90, 135],
                                       tolerancia_angular: float = 22.5,
                                       nlags: int = 15,
                                       lag_step: Optional[float] = None) -> Dict:
    """
    Calcula variogramas experimentales en diferentes direcciones
    
    Parámetros
    ----------
    x, y : arrays
        Coordenadas espaciales
    z : array
        Valores de la variable
    direcciones : list
        Ángulos en grados (0=Este, 90=Norte, 45=NE, 135=NW)
    tolerancia_angular : float
        Tolerancia en grados para cada dirección (±tolerancia)
    nlags : int
        Número de lags
    lag_step : float, optional
        Tamaño del lag (si None, se calcula automáticamente)
        
    Retorna
    -------
    dict con variogramas por dirección
    """
    if not PYKRIGE_AVAILABLE:
        print("⚠️  pykrige no disponible para variogramas direccionales")
        return {}
    
    # No necesitamos importar nada de pykrige para variogramas direccionales
    # Calculamos manualmente usando scipy
    resultados = {}
    
    # Calcular distancia máxima
    distancias = np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2)
    if lag_step is None:
        lag_step = distancias / (nlags * 2)
    
    print("\n" + "="*70)
    print("🧭 CALCULANDO VARIogramas DIRECCIONALES")
    print("="*70)
    
    for direccion in direcciones:
        print(f"\n  📐 Dirección {direccion}° (tolerancia ±{tolerancia_angular}°)")
        
        try:
            # Calcular variograma experimental manualmente en esta dirección
            # Nota: pykrige no tiene una función directa para variogramas direccionales
            # Por lo tanto, calculamos manualmente
            lags, semivariances = calcular_variograma_experimental_direccional(
                x, y, z, direccion, tolerancia_angular, nlags, lag_step
            )
            
            # Ajustar modelo teórico
            from scipy.optimize import curve_fit
            
            def modelo_esferico(h, sill, range_val, nugget):
                """Modelo esférico de variograma"""
                h = np.array(h)
                gamma = np.zeros_like(h)
                mask1 = h == 0
                mask2 = (h > 0) & (h < range_val)
                mask3 = h >= range_val
                gamma[mask1] = nugget
                gamma[mask2] = nugget + sill * (1.5 * (h[mask2]/range_val) - 0.5 * (h[mask2]/range_val)**3)
                gamma[mask3] = nugget + sill
                return gamma
            
            # Ajustar parámetros
            try:
                popt, _ = curve_fit(
                    modelo_esferico, lags, semivariances,
                    p0=[np.max(semivariances), np.max(lags)/2, 0.0],
                    bounds=([0, 0, 0], [np.max(semivariances)*2, np.max(lags)*2, np.max(semivariances)])
                )
                sill, range_val, nugget = popt
            except:
                # Valores por defecto si falla el ajuste
                sill = np.max(semivariances)
                range_val = np.max(lags) / 2
                nugget = 0.0
            
            resultados[direccion] = {
                'lags': lags,
                'semivariances': semivariances,
                'sill': sill,
                'range': range_val,
                'nugget': nugget,
                'n_pares': len(lags)
            }
            
            print(f"     ✓ Sill: {sill:.4f}, Range: {range_val:.2f}, Nugget: {nugget:.4f}")
            print(f"     ✓ Puntos en variograma: {len(lags)}")
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
            resultados[direccion] = None
    
    return resultados

def calcular_variograma_experimental_direccional(x: np.ndarray,
                                                 y: np.ndarray,
                                                 z: np.ndarray,
                                                 direccion: float,
                                                 tolerancia: float,
                                                 nlags: int,
                                                 lag_step: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula variograma experimental en una dirección específica
    
    Parámetros
    ----------
    direccion : float
        Ángulo en grados (0=Este, 90=Norte)
    tolerancia : float
        Tolerancia angular en grados
    """
    # Convertir dirección a radianes
    dir_rad = np.deg2rad(direccion)
    tol_rad = np.deg2rad(tolerancia)
    
    # Calcular todas las distancias y ángulos entre pares
    n = len(x)
    lags_list = []
    semivariances_list = []
    
    # Calcular distancia máxima
    dist_max = np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2)
    lag_max = dist_max / 2
    
    # Crear bins para los lags
    lag_bins = np.linspace(0, lag_max, nlags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
    
    # Para cada bin de lag
    for i in range(len(lag_centers)):
        lag_min = lag_bins[i]
        lag_max_bin = lag_bins[i + 1]
        
        # Encontrar pares de puntos en esta dirección y distancia
        pares_en_bin = []
        
        for j in range(n):
            for k in range(j + 1, n):
                # Vector entre puntos
                dx = x[k] - x[j]
                dy = y[k] - y[j]
                distancia = np.sqrt(dx**2 + dy**2)
                angulo = np.arctan2(dy, dx)
                
                # Normalizar ángulo a [0, 2π]
                if angulo < 0:
                    angulo += 2 * np.pi
                
                # Verificar si está en el rango de distancia
                if lag_min <= distancia < lag_max_bin:
                    # Verificar si está en la dirección correcta
                    # Comparar con dirección y dirección opuesta (180°)
                    dir1 = dir_rad
                    dir2 = (dir_rad + np.pi) % (2 * np.pi)
                    
                    # Calcular diferencia angular
                    diff1 = abs(angulo - dir1)
                    diff1 = min(diff1, 2 * np.pi - diff1)
                    diff2 = abs(angulo - dir2)
                    diff2 = min(diff2, 2 * np.pi - diff2)
                    
                    if diff1 <= tol_rad or diff2 <= tol_rad:
                        pares_en_bin.append((j, k, distancia))
        
        # Calcular semivarianza para este bin
        if len(pares_en_bin) > 0:
            semivarianzas = []
            for j, k, _ in pares_en_bin:
                semivarianzas.append(0.5 * (z[j] - z[k])**2)
            
            semivarianza_promedio = np.mean(semivarianzas)
            lags_list.append(lag_centers[i])
            semivariances_list.append(semivarianza_promedio)
    
    return np.array(lags_list), np.array(semivariances_list)

def visualizar_variogramas_direccionales(resultados: Dict,
                                        nombre_variable: str = "Variable"):
    """
    Visualiza variogramas en diferentes direcciones
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    direcciones = sorted([d for d in resultados.keys() if resultados[d] is not None])
    colores = plt.cm.tab10(np.linspace(0, 1, len(direcciones)))
    
    # Gráfico 1: Variogramas superpuestos
    ax = axes[0]
    for idx, direccion in enumerate(direcciones):
        datos = resultados[direccion]
        ax.scatter(datos['lags'], datos['semivariances'], 
                  color=colores[idx], s=30, alpha=0.6, 
                  label=f'{direccion}°', edgecolors='black', linewidths=0.5)
        
        # Modelo teórico
        h_range = np.linspace(0, datos['lags'].max() * 1.5, 100)
        gamma = []
        for h in h_range:
            if h == 0:
                gamma.append(datos['nugget'])
            elif h < datos['range']:
                gamma.append(datos['nugget'] + datos['sill'] * 
                            (1.5 * (h / datos['range']) - 0.5 * (h / datos['range'])**3))
            else:
                gamma.append(datos['nugget'] + datos['sill'])
        ax.plot(h_range, gamma, color=colores[idx], linewidth=2, linestyle='--')
    
    ax.set_xlabel('Distancia (h)', fontsize=11)
    ax.set_ylabel('Semivarianza γ(h)', fontsize=11)
    ax.set_title('Variogramas por Dirección', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Gráfico 2: Comparación de Ranges
    ax = axes[1]
    ranges = [resultados[d]['range'] for d in direcciones]
    ax.bar([f"{d}°" for d in direcciones], ranges, 
          color=colores, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Range', fontsize=11)
    ax.set_xlabel('Dirección', fontsize=11)
    ax.set_title('Range por Dirección', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Gráfico 3: Comparación de Sills
    ax = axes[2]
    sills = [resultados[d]['sill'] for d in direcciones]
    ax.bar([f"{d}°" for d in direcciones], sills, 
          color=colores, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Sill', fontsize=11)
    ax.set_xlabel('Dirección', fontsize=11)
    ax.set_title('Sill por Dirección', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Gráfico 4: Diagrama polar de Range (anisotropía)
    ax = axes[3]
    ax = plt.subplot(2, 2, 4, projection='polar')
    
    # Convertir direcciones a radianes
    angles = np.deg2rad(direcciones)
    ranges_polar = ranges
    
    # Duplicar para cerrar el círculo
    angles_closed = np.concatenate([angles, [angles[0]]])
    ranges_closed = np.concatenate([ranges_polar, [ranges_polar[0]]])
    
    ax.plot(angles_closed, ranges_closed, 'o-', linewidth=2, markersize=8, color='red')
    ax.fill(angles_closed, ranges_closed, alpha=0.25, color='red')
    ax.set_theta_zero_location('E')  # 0° apunta al Este
    ax.set_theta_direction(1)  # Sentido antihorario
    ax.set_title('Anisotropía Espacial\n(Range por Dirección)', 
                fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.suptitle(f'Análisis de Anisotropía - {nombre_variable}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Tabla resumen
    print("\n" + "="*70)
    print("📊 RESUMEN DE VARIogramas DIRECCIONALES")
    print("="*70)
    print(f"{'Dirección':<12} {'Range':<12} {'Sill':<12} {'Nugget':<12} {'N Pares':<10}")
    print("-" * 60)
    for direccion in direcciones:
        datos = resultados[direccion]
        print(f"{direccion:>3}°{'':<8} {datos['range']:>10.2f}  {datos['sill']:>10.4f}  "
              f"{datos['nugget']:>10.4f}  {datos['n_pares']:>8}")
    
    # Detectar anisotropía
    ranges_array = np.array(ranges)
    ratio_anisotropia = ranges_array.max() / ranges_array.min()
    
    print(f"\n  📐 Ratio de Anisotropía (Range_max/Range_min): {ratio_anisotropia:.2f}")
    if ratio_anisotropia > 1.5:
        print(f"  ⚠️  Se detecta ANISOTROPÍA significativa")
        direccion_max = direcciones[np.argmax(ranges_array)]
        direccion_min = direcciones[np.argmin(ranges_array)]
        print(f"     → Mayor correlación en dirección {direccion_max}°")
        print(f"     → Menor correlación en dirección {direccion_min}°")
    else:
        print(f"  ✓ Los datos son aproximadamente ISOTRÓPICOS (sin dirección preferencial)")

#%%
# ═══════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main(ruta: str = 'data/raw/bd_dm_cmp_entry.csv',
         variable_objetivo: str = 'cus',
         aplicar_log: Optional[bool] = None,
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
         # Parámetros Red Neuronal
         usar_red_neuronal: bool = True,
         nn_hidden_dims: list = [64, 32, 16],
         nn_dropout: float = 0.2,
         nn_learning_rate: float = 0.001,
         nn_batch_size: int = 32,
         nn_epochs: int = 100,
         nn_patience: int = 10,
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
        Nombre de la variable objetivo ('cus', 'starkey_min', 'cut', etc.)
    aplicar_log : bool, optional
        Si True, aplica logaritmo a la variable objetivo.
        Si None (default), detecta automáticamente si debe aplicar log
        (recomendado para 'cut', 'cus' y variables con alta asimetría)
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
    
    # ── 5. Entrenar Red Neuronal ──
    print("\n" + "="*70)
    print("🧠 APLICANDO RED NEURONAL")
    print("="*70)
    
    if usar_red_neuronal and TORCH_AVAILABLE:
        # Dividir train en train y val para early stopping
        X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state
        )
        
        model_nn, history_nn = entrenar_red_neuronal(
            X_train_nn, y_train_nn,
            X_val=X_val_nn, y_val=y_val_nn,
            hidden_dims=nn_hidden_dims,
            dropout=nn_dropout,
            learning_rate=nn_learning_rate,
            batch_size=nn_batch_size,
            epochs=nn_epochs,
            patience=nn_patience,
            verbose=True
        )
        
        # Predecir en test
        y_pred_nn = predecir_red_neuronal(model_nn, X_test)
        
        r2_nn = r2_score(y_test, y_pred_nn)
        rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
        
        print(f"  ✓ Red neuronal entrenada")
        print(f"  ✓ R²: {r2_nn:.4f}")
        print(f"  ✓ RMSE: {rmse_nn:.4f}")
    elif usar_red_neuronal and not TORCH_AVAILABLE:
        print("  ⚠️  Red neuronal no disponible (PyTorch no instalado)")
        y_pred_nn = None
        r2_nn = None
        rmse_nn = None
        model_nn = None
    else:
        y_pred_nn = None
        r2_nn = None
        rmse_nn = None
        model_nn = None
    
    # ── 6. Comparación de resultados ──
    print("\n" + "="*70)
    print("📊 COMPARACIÓN DE RESULTADOS")
    print("="*70)
    print(f"{'Método':<15} {'R²':<10} {'RMSE':<10}")
    print("-" * 35)
    print(f"{'KNN':<15} {r2_knn:<10.4f} {rmse_knn:<10.4f}")
    if r2_kriging is not None:
        print(f"{'Kriging':<15} {r2_kriging:<10.4f} {rmse_kriging:<10.4f}")
    if r2_nn is not None:
        print(f"{'Red Neuronal':<15} {r2_nn:<10.4f} {rmse_nn:<10.4f}")
    print("="*70)
    
    # ── 7. Predicciones en malla ──
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
        
        # Red Neuronal en malla
        if model_nn is not None:
            z_nn_grid = predecir_red_neuronal(model_nn, grid_points_scaled).reshape(grid_x.shape)
        else:
            z_nn_grid = None
        
        # Visualizar
        visualizar_resultados(
            x, y, atributo, grid_x, grid_y,
            z_knn_grid, z_kriging_grid, var_kriging_grid,
            y_test, y_pred_knn, y_pred_kriging,
            r2_knn, rmse_knn, r2_kriging, rmse_kriging,
            nombre_variable,
            z_nn_grid=z_nn_grid,
            y_pred_nn=y_pred_nn,
            r2_nn=r2_nn,
            rmse_nn=rmse_nn
        )
    
    print("\n✅ Análisis completado!")
    
    return {
        'r2_knn': r2_knn,
        'rmse_knn': rmse_knn,
        'r2_kriging': r2_kriging,
        'rmse_kriging': rmse_kriging,
        'r2_nn': r2_nn,
        'rmse_nn': rmse_nn,
        'nombre_variable': nombre_variable
    }

#%%
# ═══════════════════════════════════════════════════════════
# EJECUCIÓN
# ═══════════════════════════════════════════════════════════

#%%
# ═══════════════════════════════════════════════════════════
# FUNCIÓN PARA ANÁLISIS DE ANISOTROPÍA
# ═══════════════════════════════════════════════════════════

def analizar_anisotropia(ruta: str = 'data/raw/bd_dm_cmp_entry.csv',
                        variable_objetivo: str = 'cus',
                        aplicar_log: Optional[bool] = None,
                        filtro_min: float = 0.0,
                        direcciones: list = [0, 45, 90, 135],
                        tolerancia_angular: float = 22.5,
                        nlags: int = 15):
    """
    Analiza anisotropía espacial usando variogramas direccionales
    
    Parámetros
    ----------
    ruta : str
        Ruta al archivo CSV
    variable_objetivo : str
        Variable a analizar
    aplicar_log : bool
        Aplicar logaritmo
    filtro_min : float
        Filtro mínimo
    direcciones : list
        Direcciones a analizar (en grados)
    tolerancia_angular : float
        Tolerancia angular para cada dirección
    nlags : int
        Número de lags
    """
    # Cargar datos
    x, y, z, atributo, nombre_variable = cargar_y_preparar_datos(
        ruta=ruta,
        variable_objetivo=variable_objetivo,
        aplicar_log=aplicar_log,
        filtro_min=filtro_min
    )
    
    # Calcular variogramas direccionales
    resultados = calcular_variogramas_direccionales(
        x, y, atributo,
        direcciones=direcciones,
        tolerancia_angular=tolerancia_angular,
        nlags=nlags
    )
    
    # Visualizar
    visualizar_variogramas_direccionales(resultados, nombre_variable)
    
    return resultados

#%%
# ═══════════════════════════════════════════════════════════
# EJECUCIÓN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # ═══════════════════════════════════════════════════════
    # OPCIÓN 1: ANÁLISIS DE ANISOTROPÍA
    # ═══════════════════════════════════════════════════════
    # 
    # Descomenta para analizar anisotropía:
    resultados_anisotropia = analizar_anisotropia(
        ruta='data/raw/bd_dm_cmp_entry.csv',
        variable_objetivo='starkey_min',  # o 'starkey_min'
        direcciones=[0, 45, 90, 135],  # Direcciones a analizar
        tolerancia_angular=22.5,  # ±22.5° de tolerancia
        nlags=15
    )
    
    # ═══════════════════════════════════════════════════════
    # OPCIÓN 2: KNN Y KRIGING (ANÁLISIS COMPLETO)
    # ═══════════════════════════════════════════════════════
    
    resultados = main(
        # ── Datos ──
        ruta='data/raw/bd_dm_cmp_entry.csv',
        variable_objetivo='cut',  # Cambia aquí: 'cus', 'starkey_min', 'cut', etc.
        aplicar_log=None,  # None=auto (detecta automáticamente para 'cut', 'cus', etc.), True=forzar log, False=sin log
        filtro_min=0.0,  # Valor mínimo para filtrar
        
        # ── División de datos ──
        test_size=0.3,
        # random_state=42,
        
        # ── Parámetros KNN ──
        n_neighbors=10,  # Número de vecinos
        knn_weights='distance',  # 'uniform' o 'distance'
        knn_p=2,  # 1=Manhattan, 2=Euclidiana
        
        # ── Parámetros Kriging ──
        variogram_model='gaussian',  # 'spherical', 'exponential', 'gaussian', 'linear'
        nlags=15,  # Número de lags
        
        # ── Parámetros Red Neuronal ──
        usar_red_neuronal=True,  # True para usar red neuronal
        nn_hidden_dims=[64, 32, 16],  # Arquitectura: [64, 32, 16] = 3 capas ocultas
        nn_dropout=0.2,  # Dropout para regularización (0-1)
        nn_learning_rate=0.001,  # Tasa de aprendizaje
        nn_batch_size=32,  # Tamaño del batch
        nn_epochs=100,  # Número máximo de épocas
        nn_patience=10,  # Early stopping patience
        
        # ── Visualización ──
        grid_size=100,  # Tamaño de malla
        mostrar_graficos=True
    )
    
    print(f"\n🎯 Resultados finales:")
    print(f"   KNN - R²: {resultados['r2_knn']:.4f}, RMSE: {resultados['rmse_knn']:.4f}")
    if resultados['r2_kriging'] is not None:
        print(f"   Kriging - R²: {resultados['r2_kriging']:.4f}, RMSE: {resultados['rmse_kriging']:.4f}")
    if resultados['r2_nn'] is not None:
        print(f"   Red Neuronal - R²: {resultados['r2_nn']:.4f}, RMSE: {resultados['rmse_nn']:.4f}")
#%%