"""Módulo de interpolación.
Primera interpolación espacial.
Con KNN
"""

import numpy as np
import pandas as pd
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
        self.yy = None
        self.zz = None
        self.clusters_interpolados = None
        self.interpolado = False

    def crear_grid(self):
        """
        Crea un grid regular de 25x25x25 puntos dentro del bounding box de los datos originales,
        y determina cuáles de esos puntos caen dentro del convex hull de los puntos originales.

        Returns
        -------
        x_range, y_range, z_range : np.ndarray
            Vectores 1D para cada eje, delimitando el grid regular.
            También guarda en self.puntos_dentro_hull el conjunto de puntos del grid que están dentro del convex hull.
        """
        from scipy.spatial import ConvexHull, Delaunay

        # Stack de puntos originales
        puntos = np.column_stack((
            self.clusterer.x_original,
            self.clusterer.y_original,
            self.clusterer.z_original
        ))

        # Calcular bounding box minimal
        x_min, x_max = self.clusterer.x_original.min(), self.clusterer.x_original.max()
        y_min, y_max = self.clusterer.y_original.min(), self.clusterer.y_original.max()
        z_min, z_max = self.clusterer.z_original.min(), self.clusterer.z_original.max()

        # Definir SIEMPRE un grid de 25x25x25
        grid_points = 25
        x_range = np.linspace(x_min, x_max, grid_points)
        y_range = np.linspace(y_min, y_max, grid_points)
        z_range = np.linspace(z_min, z_max, grid_points)

        # Crear TODOS los puntos del grid regular
        Xg, Yg, Zg = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        puntos_grid = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

        # Calcular el convex hull y su representación "Delaunay" para queries eficientes
        hull = ConvexHull(puntos)
        delaunay = Delaunay(puntos[hull.vertices])

        # Consultar: ¿está cada punto del grid dentro del hull? (True/False)
        inside = delaunay.find_simplex(puntos_grid) >= 0
        puntos_dentro = puntos_grid[inside]

        # Guardar los puntos realmente dentro del hull para su uso posterior
        self.puntos_dentro_hull = puntos_dentro
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        return x_range, y_range, z_range

    def interpolar(self):
        print("Interpolando... con KNN....")

        x_range, y_range, z_range = self.crear_grid()
        
        # Fix: Set indexing="ij" for correct shape order
        self.xx, self.yy, self.zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')

        X_train = np.column_stack((
            self.clusterer.x_original,
            self.clusterer.y_original,
            self.clusterer.z_original
        ))

        y_train = self.clusterer.clusters

        X_train_scaled = self.scaler.fit_transform(X_train)

        X_ghost = np.column_stack((
            self.xx.flatten(),
            self.yy.flatten(),
            self.zz.flatten()
        ))
        X_ghost_scaled = self.scaler.transform(X_ghost)

        # Fix: Use self.xx.shape (should be a tuple of 3 ints), so reshape only to that
        self.clusters_interpolados = self.knn.fit(X_train_scaled, y_train).predict(X_ghost_scaled).reshape(self.xx.shape)

        self.interpolado = True

        print("Interpolación exitosa")
        return self

    def get_dataframe(self):
        """
        Retorna un DataFrame con puntos originales y fantasmas.

        Retorna:
        --------
        pd.DataFrame
            DataFrame con columnas: x, y, z, variable, cluster, tipo
            - tipo='original' para puntos conocidos
            - tipo='fantasma' para puntos interpolados
        """
        if not self.interpolado:
            raise ValueError("Debe ejecutar interpolar() antes de obtener el DataFrame")
        
        df_originales = pd.DataFrame({
            'x': self.clusterer.x_original,
            'y': self.clusterer.y_original,
            'z': self.clusterer.z_original,
            'variable': self.clusterer.attr_original,
            'cluster': self.clusterer.clusters,
            'tipo': 'original'
        })
        
        df_fantasmas = pd.DataFrame({
            'x': self.xx.flatten(),
            'y': self.yy.flatten(),
            'z': self.zz.flatten(),
            'variable': np.nan,  # No tenemos el valor de la variable para puntos fantasmas
            'cluster': self.clusters_interpolados.flatten(),
            'tipo': 'fantasma'
        })
        
        df_completo = pd.concat([df_originales, df_fantasmas], ignore_index=True)
        return df_completo

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

        if info.get('estado') == 'No Interpolado':
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