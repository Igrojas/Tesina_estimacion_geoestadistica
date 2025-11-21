"""
Módulo de visualización para clustering espacial
Versión 1: Visualizaciones básicas y comparativas
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.stats import lognorm


class VisualizadorClusters:
    """
    Clase para visualizar resultados de clustering espacial.
    
    Esta clase encapsula toda la lógica de visualización, permitiendo
    crear gráficos consistentes y profesionales con una línea de código.
    
    Parámetros:
    -----------
    carpeta_salida : str, default='results/figures'
        Carpeta donde guardar las figuras
    estilo : str, default='seaborn-v0_8-darkgrid'
        Estilo de matplotlib a usar
    dpi : int, default=150
        Resolución de las figuras guardadas
    
    Ejemplo:
    --------
    >>> viz = VisualizadorClusters()
    >>> viz.plot_clusters(clusterer)
    >>> viz.plot_comparacion(clusterer)
    >>> viz.crear_dashboard(clusterer)
    """
    
    def __init__(self, carpeta_salida='results/figures', 
                 estilo='seaborn-v0_8-darkgrid', dpi=150):
        """
        Inicializa el visualizador.
        
        Crea la carpeta de salida si no existe y configura estilos.
        """
        self.carpeta_salida = Path(carpeta_salida)
        self.carpeta_salida.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Configurar estilo
        try:
            plt.style.use(estilo)
        except:
            plt.style.use('default')
        
        # Paleta de colores profesional
        self.cmap_clusters = 'viridis'
        self.cmap_atributo = 'RdYlBu_r'
    
    def plot_clusters(self, clusterer, titulo=None, guardar=True, 
                     nombre_archivo=None, mostrar=True):
        """
        Visualiza los clusters en 2D.
        
        Parámetros:
        -----------
        clusterer : ClusterizadorKMeans
            Objeto ya entrenado
        titulo : str, optional
            Título personalizado
        guardar : bool, default=True
            Si True, guarda la figura
        nombre_archivo : str, optional
            Nombre del archivo (si None, genera automáticamente)
        mostrar : bool, default=True
            Si True, muestra la figura
        
        Retorna:
        --------
        fig, ax : Figure y Axes de matplotlib
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(
            clusterer.x_original, 
            clusterer.z_original,
            c=clusterer.clusters,
            cmap=self.cmap_clusters,
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        
        # Título
        if titulo is None:
            titulo = (f'Clustering K-means\n'
                     f'k={clusterer.n_clusters}, peso={clusterer.peso_espacial:.2f}')
        ax.set_title(titulo, fontweight='bold', fontsize=14)
        
        # Etiquetas
        ax.set_xlabel('X (midx)', fontsize=12)
        ax.set_ylabel('Z (midz)', fontsize=12)
        ax.grid(alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=11)
        
        # Agregar métrica en esquina
        metricas = clusterer.get_metricas_globales()
        texto_metrica = f"Std prom: {metricas['std_promedio']:.2f}"
        ax.text(0.02, 0.98, texto_metrica,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10)
        
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"clusters_k{clusterer.n_clusters}_w{int(clusterer.peso_espacial*100)}_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Figura guardada: {ruta}")
        
        # Mostrar
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, ax
    
    def plot_atributo_real(self, clusterer, titulo=None, guardar=True,
                          nombre_archivo=None, mostrar=True):
        """
        Visualiza el atributo real en 2D.
        
        Útil para comparar con los clusters obtenidos.
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(
            clusterer.x_original,
            clusterer.z_original,
            c=clusterer.atributo_original,
            cmap=self.cmap_atributo,
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        
        # Título
        if titulo is None:
            titulo = 'Atributo Real\n(starkey_min)'
        ax.set_title(titulo, fontweight='bold', fontsize=14)
        
        # Etiquetas
        ax.set_xlabel('X (midx)', fontsize=12)
        ax.set_ylabel('Z (midz)', fontsize=12)
        ax.grid(alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('starkey_min', fontsize=11)
        
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"atributo_real_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Figura guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, ax
    
    def plot_comparacion(self, clusterer, guardar=True, nombre_archivo=None, mostrar=True):
        """
        Compara clusters vs atributo real lado a lado.
        
        Parámetros:
        -----------
        clusterer : ClusterizadorKMeans
            Objeto entrenado
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        # Crear figura con 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel 1: Clusters
        ax = axes[0]
        scatter1 = ax.scatter(
            clusterer.x_original,
            clusterer.z_original,
            c=clusterer.clusters,
            cmap=self.cmap_clusters,
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        ax.set_title(f'Clusters (k={clusterer.n_clusters}, w={clusterer.peso_espacial:.2f})',
                    fontweight='bold', fontsize=13)
        ax.set_xlabel('X (midx)', fontsize=11)
        ax.set_ylabel('Z (midz)', fontsize=11)
        ax.grid(alpha=0.3)
        plt.colorbar(scatter1, ax=ax, label='Cluster')
        
        # Agregar métrica
        metricas = clusterer.get_metricas_globales()
        ax.text(0.02, 0.98, f"Std prom: {metricas['std_promedio']:.2f}",
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Panel 2: Atributo real
        ax = axes[1]
        scatter2 = ax.scatter(
            clusterer.x_original,
            clusterer.z_original,
            c=clusterer.atributo_original,
            cmap=self.cmap_atributo,
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        ax.set_title('Atributo Real (starkey_min)',
                    fontweight='bold', fontsize=13)
        ax.set_xlabel('X (midx)', fontsize=11)
        ax.set_ylabel('Z (midz)', fontsize=11)
        ax.grid(alpha=0.3)
        plt.colorbar(scatter2, ax=ax, label='starkey_min')
        
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"comparacion_k{clusterer.n_clusters}_w{int(clusterer.peso_espacial*100)}_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Figura guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, axes
    
    def plot_estadisticas(self, clusterer, guardar=True, nombre_archivo=None, mostrar=True):
        """
        Visualiza estadísticas descriptivas por cluster.
        
        Muestra:
        - Std por cluster
        - Tamaño de clusters
        - Boxplots de atributo
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        stats = clusterer.get_stats()
        n_clusters = clusterer.n_clusters
        
        # Extraer datos
        clusters_ids = list(stats.keys())
        stds = [stats[i]['std'] for i in clusters_ids]
        n_puntos = [stats[i]['n_puntos'] for i in clusters_ids]
        medias = [stats[i]['media'] for i in clusters_ids]
        
        # Crear figura con 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Panel 1: Std por cluster
        ax = axes[0]
        bars = ax.bar(clusters_ids, stds, color='steelblue', alpha=0.7, edgecolor='k')
        ax.axhline(np.mean(stds), color='red', linestyle='--', linewidth=2,
                  label=f'Promedio: {np.mean(stds):.2f}')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Desviación Estándar', fontsize=12)
        ax.set_title('Std por Cluster', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        # Panel 2: Tamaño de clusters
        ax = axes[1]
        bars = ax.bar(clusters_ids, n_puntos, color='coral', alpha=0.7, edgecolor='k')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Número de Puntos', fontsize=12)
        ax.set_title('Tamaño de Clusters', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        
        # Panel 3: Media vs Std
        ax = axes[2]
        ax.scatter(medias, stds, s=200, c=clusters_ids, cmap=self.cmap_clusters,
                  alpha=0.7, edgecolors='k', linewidth=2)
        for i, (m, s) in enumerate(zip(medias, stds)):
            ax.text(m, s, str(i), fontsize=10, ha='center', va='center',
                   fontweight='bold', color='white')
        ax.set_xlabel('Media de Atributo', fontsize=12)
        ax.set_ylabel('Desviación Estándar', fontsize=12)
        ax.set_title('Media vs Std por Cluster', fontweight='bold', fontsize=13)
        ax.grid(alpha=0.3)
        
        # Línea de tendencia
        if len(medias) > 2:
            z = np.polyfit(medias, stds, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(medias), max(medias), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label='Tendencia')
            ax.legend()
        
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"estadisticas_k{clusterer.n_clusters}_w{int(clusterer.peso_espacial*100)}_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Figura guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, axes
    
    def plot_boxplots(self, clusterer, guardar=True, nombre_archivo=None, mostrar=True):
        """
        Crea boxplots del atributo por cluster.
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        # Preparar datos
        import pandas as pd
        df_temp = pd.DataFrame({
            'atributo': clusterer.atributo_original,
            'cluster': clusterer.clusters
        })
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Boxplot
        palette = sns.color_palette(self.cmap_clusters, clusterer.n_clusters)
        sns.boxplot(x='cluster', y='atributo', data=df_temp, palette=palette, ax=ax)
        
        # Agregar medias
        stats = clusterer.get_stats()
        medias = [stats[i]['media'] for i in range(clusterer.n_clusters)]
        ax.scatter(range(clusterer.n_clusters), medias, 
                  color='red', s=100, marker='D', zorder=10,
                  label='Media', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Atributo (starkey_min)', fontsize=12)
        ax.set_title(f'Distribución de Atributo por Cluster (k={clusterer.n_clusters})',
                    fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"boxplots_k{clusterer.n_clusters}_w{int(clusterer.peso_espacial*100)}_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Figura guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, ax
    
    def crear_dashboard(self, clusterer, guardar=True, nombre_archivo=None, mostrar=True):
        """
        Crea un dashboard completo con todas las visualizaciones.
        
        Incluye:
        - Clusters espaciales
        - Atributo real
        - Estadísticas
        - Boxplots
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        # Crear figura grande con subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # === PANEL 1: Clusters (grande) ===
        ax1 = fig.add_subplot(gs[0:2, 0])
        scatter1 = ax1.scatter(
            clusterer.x_original, clusterer.z_original,
            c=clusterer.clusters, cmap=self.cmap_clusters,
            s=40, alpha=0.7, edgecolors='k', linewidth=0.3
        )
        ax1.set_title(f'Clusters (k={clusterer.n_clusters}, w={clusterer.peso_espacial:.2f})',
                     fontweight='bold', fontsize=13)
        ax1.set_xlabel('X (midx)')
        ax1.set_ylabel('Z (midz)')
        ax1.grid(alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # === PANEL 2: Atributo real (grande) ===
        ax2 = fig.add_subplot(gs[0:2, 1])
        scatter2 = ax2.scatter(
            clusterer.x_original, clusterer.z_original,
            c=clusterer.atributo_original, cmap=self.cmap_atributo,
            s=40, alpha=0.7, edgecolors='k', linewidth=0.3
        )
        ax2.set_title('Atributo Real (starkey_min)', fontweight='bold', fontsize=13)
        ax2.set_xlabel('X (midx)')
        ax2.set_ylabel('Z (midz)')
        ax2.grid(alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='starkey_min')
        
        # === PANEL 3: Métricas ===
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        metricas = clusterer.get_metricas_globales()
        texto_metricas = (
            f"📊 MÉTRICAS GLOBALES\n\n"
            f"Std promedio: {metricas['std_promedio']:.2f}\n"
            f"Std rango: [{metricas['std_min']:.2f}, {metricas['std_max']:.2f}]\n"
            f"CV promedio: {metricas['cv_promedio']:.2f}\n"
            f"Puntos/cluster: [{metricas['n_puntos_min']}, {metricas['n_puntos_max']}]"
        )
        ax3.text(0.1, 0.5, texto_metricas, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # === PANEL 4: Std por cluster ===
        ax4 = fig.add_subplot(gs[1, 2])
        stats = clusterer.get_stats()
        stds = [stats[i]['std'] for i in range(clusterer.n_clusters)]
        ax4.bar(range(clusterer.n_clusters), stds, color='steelblue', alpha=0.7, edgecolor='k')
        ax4.axhline(np.mean(stds), color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Cluster', fontsize=10)
        ax4.set_ylabel('Std', fontsize=10)
        ax4.set_title('Std por Cluster', fontweight='bold', fontsize=11)
        ax4.grid(axis='y', alpha=0.3)
        
        # === PANEL 5: Boxplots (completo inferior) ===
        ax5 = fig.add_subplot(gs[2, :])
        import pandas as pd
        df_temp = pd.DataFrame({
            'atributo': clusterer.atributo_original,
            'cluster': clusterer.clusters
        })
        palette = sns.color_palette(self.cmap_clusters, clusterer.n_clusters)
        sns.boxplot(x='cluster', y='atributo', data=df_temp, palette=palette, ax=ax5)
        
        medias = [stats[i]['media'] for i in range(clusterer.n_clusters)]
        ax5.scatter(range(clusterer.n_clusters), medias,
                   color='red', s=80, marker='D', zorder=10, edgecolor='black')
        ax5.set_xlabel('Cluster', fontsize=11)
        ax5.set_ylabel('Atributo (starkey_min)', fontsize=11)
        ax5.set_title('Distribución de Atributo por Cluster', fontweight='bold', fontsize=12)
        ax5.grid(axis='y', alpha=0.3)
        
        # Título general
        fig.suptitle(
            f'Dashboard Completo - Clustering K-means\n'
            f'k={clusterer.n_clusters}, peso_espacial={clusterer.peso_espacial:.2f}',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"dashboard_k{clusterer.n_clusters}_w{int(clusterer.peso_espacial*100)}_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Dashboard guardado: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def __repr__(self):
        """Representación del objeto"""
        return f"VisualizadorClusters(carpeta='{self.carpeta_salida}', dpi={self.dpi})"