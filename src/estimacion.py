#%%

"""
Módulo de estimación espacial por cluster
Versión 1: Solo KNN, estimación por dominio geometalúrgico
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class EstimadorEspacial:
    """
    Estimador espacial genérico que puede usar diferentes métodos.
    Actualmente solo soporta KNN.
    
    Parámetros:
    -----------
    metodo : str, default='knn'
        Método de estimación ('knn' por ahora)
    n_neighbors : int, default=10
        Número de vecinos para KNN
    """
    
    def __init__(self, metodo='knn', n_neighbors=10):
        self.metodo = metodo
        self.n_neighbors = n_neighbors
        self.modelo = None
        self.scaler = StandardScaler()
        self.ajustado = False
        
    def fit(self, x, z, attr):
        """
        Entrena el estimador con datos espaciales.
        
        Parámetros:
        -----------
        x : array-like
            Coordenadas X
        z : array-like
            Coordenadas Z
        attr : array-like
            Valores del atributo a predecir
        """
        if self.metodo == 'knn':
            coords = np.column_stack([x, z])
            coords_scaled = self.scaler.fit_transform(coords)
            
            self.modelo = KNeighborsRegressor(
                n_neighbors=self.n_neighbors,
                weights='distance'
            )
            self.modelo.fit(coords_scaled, attr)
        else:
            raise ValueError(f"Método '{self.metodo}' no soportado")
        
        self.ajustado = True
        return self
    
    def predict(self, x, z):
        """
        Predice valores para nuevas coordenadas.
        
        Parámetros:
        -----------
        x : array-like
            Coordenadas X
        z : array-like
            Coordenadas Z
            
        Retorna:
        --------
        array : Predicciones
        """
        if not self.ajustado:
            raise ValueError("El modelo no está ajustado. Ejecuta fit() primero.")
        
        coords = np.column_stack([x, z])
        coords_scaled = self.scaler.transform(coords)
        return self.modelo.predict(coords_scaled)

class EstimadorPorCluster:

    def __init__(self, n_neighbors=10):

        self.n_neighbors = n_neighbors

        self.modelos = {}
        self.scalers = {}

        self.datos_por_cluster = {}

        self.ajustado = False
        self.n_clusters = None

    def fit(self, x, z, attr, clusters):

        self.n_clusters = len(np.unique(clusters))

        for cluster_id in np.unique(clusters):

            mask = clusters == cluster_id
            x_cluster = x[mask]
            z_cluster = z[mask]
            attr_cluster = attr[mask]

            n_puntos = len(x_cluster)

            print(f" \n Cluster {cluster_id}: {n_puntos} puntos")


            if n_puntos < self.n_neighbors:
                print(f"❌ Cluster {cluster_id}: Número de puntos ({n_puntos}) menor que n_neighbors ({self.n_neighbors})")
                n_neighbors_ajustado = max(1, n_puntos - 1)
            else:
                n_neighbors_ajustado = self.n_neighbors

            self.datos_por_cluster[cluster_id] = {
                'x': x_cluster,
                'z': z_cluster,
                'attributo': attr_cluster,
                'n_puntos': n_puntos,
            }

            coords = np.column_stack([x_cluster, z_cluster])

            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)

            modelo = KNeighborsRegressor(
                n_neighbors=n_neighbors_ajustado,
                weights='distance')

            modelo.fit(coords_scaled, attr_cluster)

            self.modelos[cluster_id] = modelo
            self.scalers[cluster_id] = scaler

        self.ajustado = True
        return self

    def predict(self, x, z, clusters):

        if not self.ajustado:
            raise ValueError("El modelo no está ajustado. Por favor, ajuste el modelo antes de predecir.")

        valores = np.zeros(len(x))

        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id

            if cluster_id not in self.modelos:
                print(f"❌ Cluster {cluster_id}: No se encontró el modelo")
                valores[mask] = np.nan
                continue

            x_cluster = x[mask]
            z_cluster = z[mask]
            coords = np.column_stack([x_cluster, z_cluster])

            coords_scaled = self.scalers[cluster_id].transform(coords)

            pred = self.modelos[cluster_id].predict(coords_scaled)

            valores[mask] = pred

        return valores

    def estimar_grilla(self, clusters_grilla, n_points=100):

        if not self.ajustado:
            raise ValueError("El modelo no está ajustado. Por favor, ajuste el modelo antes de estimar la grilla.")

        xx = clusters_grilla.xx
        zz = clusters_grilla.zz
        clusters_interp = clusters_grilla.clusters_interpolados

        x_flat = xx.flatten()
        z_flat = zz.flatten()
        clusters_flat = clusters_interp.flatten()

        valores_flat = self.predict(x_flat, z_flat, clusters_flat)

        valores = valores_flat.reshape(xx.shape)

        return xx,zz,valores

    def get_estadisticas_cluster(self):

        if not self.ajustado:
            raise ValueError("El modelo no está ajustado. Por favor, ajuste el modelo antes de obtener las estadísticas de los clusters.")

        stats = {}

        for cluster_id, datos in self.datos_por_cluster.items():
            attr = datos['attributo']
            stats[cluster_id] = {
                'n_puntos': datos['n_puntos'],
                'media': attr.mean(),
                'std': attr.std(),
                'min': attr.min(),
                'max': attr.max(),
                'cv': attr.std() / attr.mean() if attr.mean() != 0 else 0
            }

        return stats

    def print_summary(self):

        if not self.ajustado:
            print("El modelo no está ajustado. Por favor, ajuste el modelo antes de imprimir el resumen.")
            return

    def imprimir_resumen(self):
        """Imprime resumen del estimador"""
        if not self.ajustado:
            print("❌ Modelo no entrenado")
            return
        
        stats = self.get_estadisticas_cluster()
        
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN ESTIMADOR POR CLUSTER")
        print(f"{'='*70}")
        print(f"⚙️  Configuración:")
        print(f"    • N° clusters: {self.n_clusters}")
        print(f"    • N° vecinos KNN: {self.n_neighbors}")
        
        print(f"\n📈 Estadísticas por Cluster:")
        for cluster_id, stat in stats.items():
            print(f"\n   Cluster {cluster_id}:")
            print(f"      • Puntos: {stat['n_puntos']}")
            print(f"      • Media: {stat['media']:.2f}")
            print(f"      • Std: {stat['std']:.2f}")
            print(f"      • Rango: [{stat['min']:.2f}, {stat['max']:.2f}]")
        
        print(f"\n{'='*70}\n")
    
    def __repr__(self):
        """Representación del objeto"""
        estado = "✅ ajustado" if self.ajustado else "⏳ no ajustado"
        if self.ajustado:
            return f"EstimadorPorCluster(n_neighbors={self.n_neighbors}, clusters={self.n_clusters}, {estado})"
        return f"EstimadorPorCluster(n_neighbors={self.n_neighbors}, {estado})"