"""
Módulo de estimación espacial
Versión 1: KNN, IDW y Kriging Ordinario
"""


import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from pykrige.rk import OrdinaryKriging
from scipy.spatial.distance import cdist

class EstimadorEspacial:

    def __init__(self, metodo='knn', **kwargs):


        metodos_validos = ['knn', 'idw', 'kriging']
        if metodo not in metodos_validos:
            raise ValueError(f"Método de estimación inválido: {metodo}. Valores válidos: {metodos_validos}")

        self.metodo = metodo
        self.kwargs = kwargs

        if metodo == 'knn':
            self.n_neighbors = kwargs.get('n_neighbors', 10)
            self.scaler = StandardScaler()
            self.modelo = None
        
        elif metodo == 'idw':
            self.power = kwargs.get('power', 2)

        elif metodo == 'kriging':
            self.variogram_model = kwargs.get('variogram_model', 'spherical')
            self.modelo = None

        
        self.x_train = None
        self.z_train = None
        self.attr_train = None
        self.ajustado = False

    def fit(self,x,z,attr):

        self.x_train = x
        self.z_train = z
        self.attr_train = attr

        if self.metodo == 'knn':

            coords = np.column_stack([x,z])
            coords_scaled = self.scaler.fit_transform(coords)

            self.modelo = KNeighborsRegressor(
                n_neighbors=self.n_neighbors, 
                weights='distance')
            self.modelo.fit(coords_scaled, attr)

        elif self.metodo == 'idw':
            pass
        
        elif self.metodo == 'kriging':
            print(f"Ajustando variograma {self.variogram_model}...")
            self.modelo = OrdinaryKriging(
                x,z,attr,
                variogram_model=self.variogram_model,
                verbose=False,
                enable_plotting=False,
                **{k:v for k,v in self.kwargs.items() if k != 'variogram_model'}
            )

            print("Kriging ajustado exitosamente")

        self.ajustado = True
        return self

    def predict(self, x_pred, z_pred, return_variance=False):

        if not self.ajustado:
            raise ValueError("El modelo no está ajustado. Por favor, ajuste el modelo antes de predecir.")

        if self.metodo == 'knn':
            coords_pred = np.column_stack([x_pred, z_pred])
            coords_pred_scaled = self.scaler.transform(coords_pred)
            valores = self.modelo.predict(coords_pred_scaled)

            if return_variance:
                print("KNN no devuelve varianza")
                return valores, None
            return valores

        elif self.metodo == 'idw':
            valores = self._idw_predict(x_pred, z_pred)

            if return_variance:
                print("⚠️ IDW no calcula varianza, retornando None")
                return valores, None
            return valores

        elif self.metodo == 'kriging':

            if len(x_pred.shape) == 1:
                valores, varianza = self.modelo.execute('points', x_pred, z_pred)
            else:
                valores,varianza = self.modelo.execute('grid', x_pred, z_pred)

            if return_variance:
                return valores, varianza
            return valores

    def _idw_predict(self, x_pred, z_pred):
        """
        Implementación de Inverse Distance Weighting.

        Fórmula: valor = Σ(wi * vi) / Σ(wi)
        donde wi = 1 / distancia^power
        """
        coords_train = np.column_stack([self.x_train, self.z_train])
        coords_pred = np.column_stack([x_pred, z_pred])

        distancias = cdist(coords_pred, coords_train)

        distancias[distancias == 0] = 1e-10

        pesos = 1 / distancias**self.power
        pesos_norm = pesos / pesos.sum(axis=1, keepdims=True)

        valores = (pesos_norm * self.attr_train).sum(axis=1)

        return valores

    def estimar_grilla(self, n_points =100, return_variance=False):

        if not self.ajustado:
            raise ValueError("❌ Primero ejecuta fit()")

        x_min, x_max = self.x_train.min(), self.x_train.max()
        z_min, z_max = self.z_train.min(), self.z_train.max()

        # Margen
        margen_x = (x_max - x_min) * 0.02
        margen_z = (z_max - z_min) * 0.02

        x_range = np.linspace(x_min - margen_x, x_max + margen_x, n_points)
        z_range = np.linspace(z_min - margen_z, z_max + margen_z, n_points)

        if self.metodo == 'kriging':
            # Kriging puede trabajar directamente con rangos
            valores, varianza = self.modelo.execute('grid', x_range, z_range)
            xx, zz = np.meshgrid(x_range, z_range)
            
            if return_variance:
                return xx, zz, valores, varianza
            return xx, zz, valores
        
        else:
            # Para KNN e IDW, crear meshgrid primero
            xx, zz = np.meshgrid(x_range, z_range)
            x_flat = xx.flatten()
            z_flat = zz.flatten()
            
            valores_flat = self.predict(x_flat, z_flat)
            valores = valores_flat.reshape(xx.shape)
            
            if return_variance:
                return xx, zz, valores, None
            return xx, zz, valores

    def validacion_cruzada(self, n_folds=5):
        """
        Realiza validación cruzada para evaluar el modelo.
        
        Parámetros:
        -----------
        n_folds : int
            Número de folds para cross-validation
        
        Retorna:
        --------
        dict : Métricas de validación (MAE, RMSE, R2)
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        if not self.ajustado:
            raise ValueError("❌ Primero ejecuta fit()")
        
        # Preparar datos
        coords = np.column_stack([self.x_train, self.z_train])
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        maes = []
        rmses = []
        r2s = []
        
        for train_idx, test_idx in kf.split(coords):
            # Split
            x_tr, x_te = self.x_train[train_idx], self.x_train[test_idx]
            z_tr, z_te = self.z_train[train_idx], self.z_train[test_idx]
            attr_tr, attr_te = self.atributo_train[train_idx], self.atributo_train[test_idx]
            
            # Crear modelo temporal
            estimador_temp = EstimadorEspacial(metodo=self.metodo, **self.kwargs)
            estimador_temp.fit(x_tr, z_tr, attr_tr)
            
            # Predecir
            pred = estimador_temp.predict(x_te, z_te)
            
            # Métricas
            maes.append(mean_absolute_error(attr_te, pred))
            rmses.append(np.sqrt(mean_squared_error(attr_te, pred)))
            r2s.append(r2_score(attr_te, pred))
        
        return {
            'MAE': np.mean(maes),
            'MAE_std': np.std(maes),
            'RMSE': np.mean(rmses),
            'RMSE_std': np.std(rmses),
            'R2': np.mean(r2s),
            'R2_std': np.std(r2s)
        }
    
    def get_info(self):
        """Información del estimador"""
        info = {
            'metodo': self.metodo,
            'ajustado': self.ajustado,
        }
        
        if self.ajustado:
            info['n_puntos_entrenamiento'] = len(self.x_train)
        
        if self.metodo == 'knn':
            info['n_neighbors'] = self.n_neighbors
        elif self.metodo == 'idw':
            info['power'] = self.power
        elif self.metodo == 'kriging':
            info['variogram_model'] = self.variogram_model
        
        return info
    
    def __repr__(self):
        """Representación del objeto"""
        estado = "✅ ajustado" if self.ajustado else "⏳ no ajustado"
        return f"EstimadorEspacial(metodo='{self.metodo}', {estado})"