"""Módulo de clustering."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

class ClusterKmeans:
    def __init__(self, n_clusters=4, w_spatial=0.0):

        # Parametros
        self.n_clusters = n_clusters
        self.w_spatial = w_spatial

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

    def fit(self, x, y, z, attr):

        self.x_original = x
        self.y_original = y
        self.z_original = z
        self.attr_original = attr

        coords = np.column_stack([x, y, z])
        coords_scaled = self.scaler_coords.fit_transform(coords)

        attr_scaled = self.scaler_attr.fit_transform(attr.reshape(-1, 1))

        coords_weighted = coords_scaled * self.w_spatial
        attr_weighted = attr_scaled * (1 - self.w_spatial)

        features = np.column_stack([coords_weighted, attr_weighted])

        self.kmeans = KMeans(
                            n_clusters=self.n_clusters,
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

    def get_global_metrics(self):

        if not self.ajustado:
            raise ValueError("El modelo no ha sido ajustado. Por favor, ajuste el modelo antes de obtener las métricas globales.")
        
        stats = self.get_stats()

        stds = [s['std'] for s in stats.values()]
        cvs = [s['efecto_proporcional'] for s in stats.values()]
        n_points = [s['n_points'] for s in stats.values()]

        std_prom = np.mean(stds)
        cv_prom = np.mean(cvs)
        n_points_prom = np.mean(n_points)

        return {
            'std_prom': float(std_prom),
            'std_min': float(min(stds)),
            'std_max': float(max(stds)),
            'cv_prom': float(cv_prom),
            'cv_min': float(min(cvs)),
            'cv_max': float(max(cvs)),
            'n_points_min': float(min(n_points)),
            'n_points_max': float(max(n_points)),
        }


