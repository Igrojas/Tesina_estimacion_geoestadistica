"""Módulo de interpolación.
Primera interpolación espacial.
Con KNN
"""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime

class InterpoladorEspacial:

    def __init__(self, clusterer, n_neighbors=5, n_points=100):
        
        if not clusterer.ajustado:
            raise ValueError("El clusterer no está ajustado")

        

        self.clusterer = clusterer
        self.n_neighbors = n_neighbors
        self.n_points = n_points

        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

        self.xx = None
        self.zz = None
        self.clusters_interpolados = None
        self.interpolado = False

    def crear_grid(self):

        x_min = self.clusterer.x_original.min()
        x_max = self.clusterer.x_original.max()
        z_min = self.clusterer.z_original.min()
        z_max = self.clusterer.z_original.max()

        margen_x = (x_max - x_min) * 0.02
        margen_z = (z_max - z_min) * 0.02

        x_range = np.linspace(x_min - margen_x, x_max + margen_x, self.n_points)
        z_range = np.linspace(z_min - margen_z, z_max + margen_z, self.n_points)
    
        return x_range, z_range

    def interpolar(self):

        print("Interpolando... con KNN....")

        x_range, z_range = self.crear_grid()
        self.xx, self.zz = np.meshgrid(x_range, z_range)

        X_train = np.column_stack((
            self.clusterer.x_original,
            self.clusterer.z_original
        ))

        y_train = self.clusterer.clusters

        X_train_scaled = self.scaler.fit_transform(X_train)

        X_ghost = np.column_stack((
            self.xx.flatten(),
            self.zz.flatten()
        ))
        X_ghost_scaled = self.scaler.transform(X_ghost)

        self.clusters_interpolados = self.knn.fit(X_train_scaled, y_train).predict(X_ghost_scaled).reshape(self.xx.shape)

        self.interpolado = True

        print("Interpolación exitosa")
        return self

    def get_info(self):

        if not self.interpolado:
            return {"estado": "No Interpolado"}

        return {
            'n_neighbors': self.n_neighbors,
            'n_points': self.n_points,
            'shape_grilla': self.xx.shape,
            'n_puntos_grilla': self.xx.size,
            'n_puntos_originales': len(self.clusterer.x_original),
            'n_clusters': self.clusterer.n_clusters,
        }
    def print_info(self):
        """Imprime información de la interpolación"""
        info = self.get_info()
        
        if info.get('estado') == 'No interpolado':
            print("❌ Interpolación no realizada. Ejecuta interpolar() primero.")
            return
        
        print(f"\n{'='*60}")
        print(f"📍 INFORMACIÓN DE INTERPOLACIÓN")
        print(f"{'='*60}")
        print(f"⚙️  KNN neighbors: {info['n_neighbors']}")
        print(f"📐 Grilla: {info['shape_grilla']} ({info['n_puntos_grilla']:,} puntos)")
        print(f"📊 Puntos originales: {info['n_puntos_originales']:,}")
        print(f"🎯 Clusters: {info['n_clusters']}")
        print(f"{'='*60}\n")

    def comparar_n_neighbors(self, lista_n_neighbors):
        """
        Compara diferentes valores de n_neighbors.
        
        Parámetros:
        -----------
        lista_n_neighbors : list
            Lista de valores de n_neighbors a probar
        
        Retorna:
        --------
        dict : {n_neighbors: InterpoladorEspacial}
        """
        print(f"🔧 Comparando {len(lista_n_neighbors)} configuraciones de n_neighbors...")
        
        resultados = {}
        
        for n_neigh in lista_n_neighbors:
            # Crear nuevo interpolador
            interp = InterpoladorEspacial(
                self.clusterer,
                n_neighbors=n_neigh,
                n_points=self.n_points
            )
            interp.interpolar()
            resultados[n_neigh] = interp
        
        print(f"✅ Comparación completada")
        
        return resultados
    
    def __repr__(self):
        """Representación del objeto"""
        estado = "✅ interpolado" if self.interpolado else "⏳ no interpolado"
        return (f"InterpoladorEspacial(n_neighbors={self.n_neighbors}, "
                f"n_points={self.n_points}, {estado})")