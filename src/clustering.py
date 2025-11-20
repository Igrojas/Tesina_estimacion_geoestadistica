"""Módulo de clustering."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

class ClusterKmeans:
    def __init__(self, n_clusters=4):

        # Parametros
        self.n_clusters = n_clusters

        # Objetos Internos
        self.scaler_coords = StandardScaler()
        self.scaler_attr = StandardScaler()
        self.kmeans = None

        # Resultados
        self.clusters = None
        self.x_original = None
        self.y_original = None
        self.z_original = None
        self.attr_original = None
        self.ajustado = False

    def fit(self, x, z, attr):

        self.x_original = x
        # self.y_original = y
        self.z_original = z
        self.attr_original = attr

        coords = np.column_stack([x, z])
        coords_scaled = self.scaler_coords.fit_transform(coords)

        attr_scaled = self.scaler_attr.fit_transform(attr.reshape(-1, 1))

        features = np.column_stack([coords_scaled, attr_scaled])

        self.kmeans = KMeans(
                            n_clusters=self.n_clusters, 
                            random_state=42, 
                            n_init=10)

        self.clusters = self.kmeans.fit_predict(features)

        self.ajustado = True

        return self


    def get_stats(self):

        if not self.ajustado:
            raise ValueError("El modelo no ha sido ajustado. Por favor, ajuste el modelo antes de obtener las estadísticas.")

        stats = {}

        for i in range(self.n_clusters):
            mask = self.clusters == i
            cluster_attr = self.attr_original[mask]

            mean = cluster_attr.mean()
            std = cluster_attr.std()
            efecto_proporcional = std / mean

            stats[i] = {
                'n_points': int(mask.sum()),
                'mean': float(mean),
                'std': float(std),
                'efecto_proporcional': float(efecto_proporcional),
            }
        
        return stats

    def summary_plot(self):

        if not self.ajustado:
            raise ValueError("El modelo no ha sido ajustado. Por favor, ajuste el modelo antes de obtener el resumen gráfico.")

        stats = self.get_stats()

        print(f"\n{'='*60}")
        print(f"📊 RESUMEN DE CLUSTERING (k={self.n_clusters})")
        print(f"{'='*60}")
        
        for i, stat in stats.items():
            print(f"\nCluster {i}:")
            print(f"  • Puntos: {stat['n_points']}")
            print(f"  • Media: {stat['mean']:.2f}")
            print(f"  • Std: {stat['std']:.2f}")
            print(f"  • Efecto Proporcional: {stat['efecto_proporcional']:.2f}")



