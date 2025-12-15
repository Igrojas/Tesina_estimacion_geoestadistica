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
import scipy.stats as stats
import pandas as pd


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
                 estilo='default', dpi=150):
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
        
        # Paleta de colores profesional para variables continuas
        self.cmap_clusters = 'tab10'  # Cambiar por defecto de clusters, pero sólo usaremos para palette discreta
        self.cmap_atributo = 'RdYlBu_r'
    
    def get_cluster_palette(self, n_clusters):
        """Devuelve una lista de colores fijos, uno para cada cluster, siempre igual para cada cantidad de clusters."""
        # tab10 soporta hasta 10 colores, tab20 hasta 20
        if n_clusters <= 10:
            palette = sns.color_palette("tab10", n_clusters)
        elif n_clusters <= 20:
            palette = sns.color_palette("tab20", n_clusters)
        else:
            # Si hay más de 20 clusters, usar husl para un rango alto
            palette = sns.color_palette("husl", n_clusters)
        return palette

    def plot_clusters(self, clusterer, titulo=None, guardar=True, 
                     nombre_archivo=None, mostrar=True, nombre_atributo="starkey_min"):
        """
        Visualiza los clusters en 2D usando colores discretos para cada cluster!
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        # Paleta discreta: colores fijos para cada cluster
        n_clusters = clusterer.n_clusters
        palette = self.get_cluster_palette(n_clusters)
        cluster_color_dict = {i: palette[i] for i in range(n_clusters)}
        cluster_colors = np.array([cluster_color_dict[c] for c in clusterer.clusters])
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot, con color fijo por cluster
        scatter = ax.scatter(
            clusterer.x_original, 
            clusterer.y_original,
            clusterer.z_original,
            c=cluster_colors,
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        
        # Título
        if titulo is None:
            titulo = (f'Clustering K-means\n'
                     f'k={clusterer.n_clusters}, peso={clusterer.w_spatial:.2f}')
        ax.set_title(str(titulo), fontweight='bold', fontsize=14)
        
        # Etiquetas
        ax.set_xlabel('X (midx)', fontsize=12)
        ax.set_ylabel('Z (midz)', fontsize=12)
        ax.grid(alpha=0.3)
        
        # Leyenda: color por cluster
        unique_clusters = np.unique(clusterer.clusters)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_color_dict[k], markeredgecolor='k',
                              markersize=10, label=f'Cluster {k}') for k in unique_clusters]
        ax.legend(handles=handles, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Agregar métrica en esquina
        metricas = clusterer.get_global_metrics()
        texto_metrica = f"Std prom: {metricas['std_prom']:.2f}"
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
                nombre_archivo = f"clusters_k{clusterer.n_clusters}_w{int(clusterer.w_spatial*100)}_{timestamp}.png"
            
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
                          nombre_archivo=None, mostrar=True, nombre_atributo="starkey_min"):
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
            clusterer.y_original,
            clusterer.z_original,
            c=clusterer.attr_original,
            cmap=self.cmap_atributo,
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        
        # Título
        if titulo is None:
            titulo = f'Atributo Real\n({nombre_atributo})'
        ax.set_title(str(titulo), fontweight='bold', fontsize=14)
        
        # Etiquetas
        ax.set_xlabel('X (midx)', fontsize=12)
        ax.set_ylabel('Z (midz)', fontsize=12)
        ax.grid(alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(str(nombre_atributo), fontsize=11)
        
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
    
    def plot_comparacion(self, clusterer, guardar=True, nombre_archivo=None, mostrar=True, nombre_atributo="starkey_min"):
        """
        Compara clusters vs atributo real lado a lado.
        
        Parámetros:
        -----------
        clusterer : ClusterizadorKMeans
            Objeto entrenado
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        n_clusters = clusterer.n_clusters
        palette = self.get_cluster_palette(n_clusters)
        cluster_color_dict = {i: palette[i] for i in range(n_clusters)}
        cluster_colors = np.array([cluster_color_dict[c] for c in clusterer.clusters])
        
        # Crear figura con 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel 1: Clusters - COLORES DISCRETOS
        ax = axes[0]
        scatter1 = ax.scatter(
            clusterer.x_original,
            clusterer.z_original,
            c=cluster_colors,
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        ax.set_title(f'Clusters (k={clusterer.n_clusters}, w={clusterer.w_spatial:.2f})',
                    fontweight='bold', fontsize=13)
        ax.set_xlabel('X (midx)', fontsize=11)
        ax.set_ylabel('Z (midz)', fontsize=11)
        ax.grid(alpha=0.3)
        # Leyenda
        unique_clusters = np.unique(clusterer.clusters)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_color_dict[k], markeredgecolor='k',
                              markersize=10, label=f'Cluster {k}') for k in unique_clusters]
        ax.legend(handles=handles, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Agregar métrica
        metricas = clusterer.get_global_metrics()
        ax.text(0.02, 0.98, f"Std prom: {metricas['std_prom']:.2f}",
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Panel 2: Atributo real (continuous variable - cmap)
        ax = axes[1]
        scatter2 = ax.scatter(
            clusterer.x_original,
            clusterer.z_original,
            c=clusterer.attr_original,
            cmap=self.cmap_atributo,
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        # Generaliza el título y etiqueta del colorbar
        ax.set_title(f'Atributo Real ({nombre_atributo})', fontweight='bold', fontsize=13)
        ax.set_xlabel('X (midx)', fontsize=11)
        ax.set_ylabel('Z (midz)', fontsize=11)
        ax.grid(alpha=0.3)
        plt.colorbar(scatter2, ax=ax, label=str(nombre_atributo))
        
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"comparacion_k{clusterer.n_clusters}_w{int(clusterer.w_spatial*100)}_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Figura guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, axes
    
    def plot_estadisticas(self, clusterer, guardar=True, nombre_archivo=None, mostrar=True, nombre_atributo="starkey_min"):
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
        
        # Paleta fija para los clusters
        palette = self.get_cluster_palette(n_clusters)
        cluster_color_dict = {i: palette[i] for i in range(n_clusters)}
        cluster_colors = [cluster_color_dict[i] for i in clusters_ids]
        
        # Crear figura con 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Panel 1: Std por cluster
        ax = axes[0]
        bars = ax.bar(clusters_ids, stds, color=cluster_colors, alpha=0.7, edgecolor='k')
        ax.axhline(np.mean(stds), color='red', linestyle='--', linewidth=2,
                  label=f'Promedio: {np.mean(stds):.2f}')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Desviación Estándar', fontsize=12)
        ax.set_title('Std por Cluster', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        # Panel 2: Tamaño de clusters
        ax = axes[1]
        bars = ax.bar(clusters_ids, n_puntos, color=cluster_colors, alpha=0.7, edgecolor='k')  # mismo color por cluster
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Número de Puntos', fontsize=12)
        ax.set_title('Tamaño de Clusters', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        
        # Panel 3: Media vs Std, puntos con los colores de cada cluster, SIN cmap continuous
        ax = axes[2]
        ax.scatter(medias, stds, s=200, c=cluster_colors,
                  alpha=0.7, edgecolors='k', linewidth=2)
        for i, (m, s) in enumerate(zip(medias, stds)):
            ax.text(m, s, str(clusters_ids[i]), fontsize=10, ha='center', va='center',
                   fontweight='bold', color='white')
        ax.set_xlabel(f'Media de {nombre_atributo}', fontsize=12)
        ax.set_ylabel('Desviación Estándar', fontsize=12)
        ax.set_title(f'Media vs Std por Cluster', fontweight='bold', fontsize=13)
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
    
    def plot_boxplots(self, clusterer, guardar=True, nombre_archivo=None, mostrar=True, nombre_atributo="starkey_min"):
        """
        Crea boxplots del atributo por cluster.
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        # Preparar datos
        import pandas as pd
        df_temp = pd.DataFrame({
            'atributo': clusterer.attr_original,
            'cluster': clusterer.clusters
        })
        
        n_clusters = clusterer.n_clusters
        palette = self.get_cluster_palette(n_clusters)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Boxplot con colores fijos por cluster (palette)
        sns.boxplot(x='cluster', y='atributo', data=df_temp, palette=palette, ax=ax)
        
        # Agregar medias
        stats = clusterer.get_stats()
        medias = [stats[i]['media'] for i in range(clusterer.n_clusters)]
        ax.scatter(range(clusterer.n_clusters), medias,
                  color='red', s=100, marker='D', zorder=10,
                  label='Media', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel(f'Atributo ({nombre_atributo})', fontsize=12)
        ax.set_title(f'Distribución de {nombre_atributo} por Cluster (k={clusterer.n_clusters})',
                    fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"boxplots_k{clusterer.n_clusters}_w{int(clusterer.w_spatial*100)}_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Figura guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, ax
    
    def crear_dashboard(self, clusterer, guardar=True, nombre_archivo=None, mostrar=True, nombre_atributo="starkey_min"):
        """
        Crea un dashboard (2x2) con visualizaciones de clusters, mostrando todas las leyendas completas y específicas para cada gráfica.

        Orden:
        1. Efecto proporcional (Coeficiente de variación por cluster)
        2. Probability Plot normal por cluster (usando log(atributo) en vez de atributo directo)
        3. Boxplots del atributo por cluster
        4. Mapa 3D de clusters (scatter 3D)
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")
        
        import numpy as np
        import pandas as pd
        from scipy.stats import norm
        import matplotlib.gridspec as gridspec
        import seaborn as sns

        stats_clusters = clusterer.get_stats()
        n_clusters = clusterer.n_clusters

        df_temp = pd.DataFrame({
            'atributo': clusterer.attr_original,
            'cluster': clusterer.clusters
        })

        # Paleta de colores consistente
        palette = self.get_cluster_palette(n_clusters)
        cluster_color_dict = {i: palette[i] for i in range(n_clusters)}  # cluster_id -> color RGB

        # --- Usamos GridSpec para controlar espacio de subplots ---
        fig = plt.figure(figsize=(18, 10))
        # Grid de 2x2, pero la 3D ocupa toda la derecha (filas 0:2 y col 1)
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1.1, 1.6], height_ratios=[1, 1])

        # (0, 0) Efecto proporcional
        ax0 = fig.add_subplot(gs[0, 0])
        cvs = [stats_clusters[i].get('efecto_proporcional', np.nan) for i in range(n_clusters)]
        clusters_x = np.arange(n_clusters)
        legend_handles = []
        for i in range(n_clusters):
            sc = ax0.scatter(clusters_x[i], cvs[i], color=cluster_color_dict[i], s=80, zorder=3, label=f'Cluster {i}')
            legend_handles.append(sc)
        z = np.polyfit(clusters_x, cvs, 1)
        p = np.poly1d(z)
        ln, = ax0.plot(clusters_x, p(clusters_x), color='crimson', linestyle='--', linewidth=2, label='Tendencia lineal')
        for i, cv in enumerate(cvs):
            ax0.text(i, cv + 0.001, f"{cv:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
        ax0.set_xlabel('Cluster', fontsize=10)
        ax0.set_ylabel('Coef. de Variación\n(std/media)', fontsize=10)
        ax0.set_title('Efecto proporcional (CV)', fontweight='bold', fontsize=12)
        ax0.grid(axis='y', alpha=0.3)
        handles = legend_handles + [ln]
        labels = [f'Cluster {i}' for i in range(n_clusters)] + ['Tendencia lineal']
        ax0.legend(handles, labels, fontsize=8, frameon=True, loc='upper right')

        # (0, 1) Probability Plot NORMAL sobre log(atributo)
        ax1 = fig.add_subplot(gs[0, 1])

        # Parámetros para etiquetas percentiles
        percentile_labels = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 99.5, 99.9]
        z_ticks = norm.ppf(np.array(percentile_labels) / 100)
        cluster_handles = []
        cluster_labels = []
        for i in range(n_clusters):
            data = df_temp[df_temp["cluster"] == i]["atributo"].dropna().values
            # Usar únicamente valores > 0 para aplicar log
            data = data[data > 0]
            if len(data) < 3:
                continue
            log_data = np.log(data)
            # Ordenar
            datos_sorted = np.sort(log_data)
            n = len(datos_sorted)
            percentiles = (np.arange(1, n + 1) - 0.5) / n * 100
            # Convertir percentiles a z-scores (cuantiles normales)
            z_scores = norm.ppf(percentiles / 100)
            color = cluster_color_dict[i]
            # Graficar puntos en normal prob plot
            handle = ax1.plot(datos_sorted, z_scores, 'o', markersize=4, color=color, label=f'Cluster {i}')[0]
            cluster_handles.append(handle)
            cluster_labels.append(f'Cluster {i}')
        ax1.set_yticks(z_ticks)
        ax1.set_yticklabels([f'{p}' for p in percentile_labels])
        ax1.set_xlabel(f'log(Atributo) (log({nombre_atributo}))', fontsize=10)
        ax1.set_ylabel('Frecuencia acumulada', fontsize=10)
        ax1.set_title(f'Gráfico de probabilidad normal (log) [{nombre_atributo}]', fontsize=12, fontweight='bold')
        # ax1.set_xscale('log')
        ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, which='both')
        if cluster_handles:
            ax1.legend(cluster_handles, cluster_labels, fontsize=8, loc='best', frameon=True)

        # (1, 0) Boxplots del atributo
        ax2 = fig.add_subplot(gs[1, 0])
        sns.boxplot(x='cluster', y='atributo', data=df_temp, palette=palette, ax=ax2, width=0.75, fliersize=2.5)
        sc = None
        for i in range(n_clusters):
            media_val = stats_clusters[i].get('mean', np.nan)
            sc = ax2.scatter(i, media_val,
                        color='crimson', s=40, marker='D', zorder=10,
                        edgecolor='black', linewidth=0.4, label='Media' if i == 0 else "")
        ax2.set_xlabel('Cluster', fontsize=10)
        ax2.set_ylabel(f'Atributo\n({nombre_atributo})', fontsize=10)
        ax2.set_title(f'Boxplot por cluster ({nombre_atributo})', fontweight='bold', fontsize=12)
        ax2.grid(axis='y', alpha=0.28)
        if sc is not None:
            ax2.legend([sc], ['Media'], loc='best', fontsize=8)

        # (0:2, 2) Scatter 3D de clusters (ocupa ambas filas derecha)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax3 = fig.add_subplot(gs[:, 2], projection='3d')
        x3d = clusterer.x_original
        y3d = clusterer.y_original
        z3d = clusterer.z_original
        cluster_colors = np.array([cluster_color_dict[c] for c in clusterer.clusters])
        sc3d = ax3.scatter(
            x3d, y3d, z3d,
            c=cluster_colors,
            s=30,
            alpha=0.83,
            edgecolors='k',
            linewidth=0.5)
        titulo3d = (f'Clustering K-means (3D)\nk={clusterer.n_clusters}, peso={clusterer.w_spatial:.2f}')
        ax3.set_title(titulo3d, fontweight='bold', fontsize=13, pad=9)
        ax3.set_xlabel('X (midx)', fontsize=10)
        ax3.set_ylabel('Y (midy)', fontsize=10)
        ax3.set_zlabel('Z (midz)', fontsize=10)
        ax3.grid(alpha=0.26)
        # Leyenda 3D
        unique_clusters = np.unique(clusterer.clusters)
        handles_3d = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=cluster_color_dict[k], markeredgecolor='k',
                                markersize=8, label=f'Cluster {k}') for k in unique_clusters]
        ax3.legend(handles=handles_3d, title='Cluster', loc='best', fontsize=9, frameon=True)
        # Métrica
        metricas = clusterer.get_global_metrics()
        texto_metrica = f"Std prom: {metricas['std_prom']:.2f}"
        ax3.text2D(0.04, 0.97, texto_metrica,
                transform=ax3.figure.transFigure,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)

        # Título global
        fig.suptitle(
            f"Dashboard 2x2 - Clustering K-means\nk={n_clusters}, peso_espacial={clusterer.w_spatial:.2f}, atributo={nombre_atributo}",
            fontsize=17, fontweight='bold', y=0.995)

        plt.subplots_adjust(left=0.042, right=0.98, bottom=0.06, top=0.92, hspace=0.23, wspace=0.18)

        # Guardar
        if guardar:
            if nombre_archivo is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"dashboard2x2_3d_k{n_clusters}_w{int(clusterer.w_spatial*100)}_{timestamp}.png"

            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Dashboard guardado: {ruta}")

        if mostrar:
            plt.show()
        else:
            plt.close()
        

        return fig
        
    def plot_interpolacion(self, interpolador, mostrar_puntos=True,
                           guardar=True, nombre_archivo=None, mostrar=True, nombre_atributo="starkey_min"):
        """
        Visualiza la interpolación KNN en 3D interactivo usando plotly,
        mostrando los puntos nuevos creados y los originales, junto al cluster asignado.
        Ahora también muestra cada bloque/grupo interpolado como una "nube sólida" mediante superficies 3D Mesh con color por grupo (más sólido).
        Además, se siguen mostrando las fronteras entre clusters como superficies semitransparentes.

        Nota: No muestra ni guarda la gráfica si mostrar=False o guardar=False respectivamente.
        Así se evita doble visualización si se llama desde otros scripts o notebooks.
        """
        import plotly.graph_objects as go
        import numpy as np

        if not interpolador.interpolado:
            raise ValueError("❌ El interpolador no está interpolado")

        clusterer = interpolador.clusterer
        xx, yy, zz = interpolador.xx, interpolador.yy, interpolador.zz
        clusters_interp = interpolador.clusters_interpolados

        n_clusters = clusterer.n_clusters
        from plotly.colors import sample_colorscale, find_intermediate_color
        color_scale = "Viridis"
        color_list = sample_colorscale(color_scale, [i/(n_clusters-1) if n_clusters > 1 else 0 for i in range(n_clusters)])

        # -- Puntos originales
        if mostrar_puntos:
            puntos_orig = go.Scatter3d(
                x=clusterer.x_original,
                y=clusterer.y_original,
                z=clusterer.z_original,
                mode='markers',
                marker=dict(
                    size=5,
                    color=clusterer.clusters,
                    colorscale=color_scale,
                    opacity=0.92,
                    line=dict(width=1, color='black')
                ),
                name="Puntos originales",
                text=[f"Cluster: {c}" for c in clusterer.clusters],
                hoverinfo='text'
            )
        else:
            puntos_orig = None

        # -- Puntos de la grilla interpolados (todos los puntos creados)
        x_interp = xx.flatten()
        y_interp = yy.flatten()
        z_interp = zz.flatten()
        clusters_interp_flat = clusters_interp.flatten()

        puntos_interpolados = go.Scatter3d(
            x=x_interp,
            y=y_interp,
            z=z_interp,
            mode='markers',
            marker=dict(
                size=3,
                color=clusters_interp_flat,
                colorscale=color_scale,
                opacity=0.18,  # aún más bajo, para que no opaque las Mesh (bloques sólidos)
                line=dict(width=0.3, color='gray')
            ),
            name="Puntos interpolados",
            text=[f"Cluster: {c}" for c in clusters_interp_flat],
            hoverinfo='skip'
        )

        # ---------- Mesh3D: bloques sólidos para cada grupo/clúster -------------
        # Usamos marching_cubes para cada cluster para obtener el "volumen" (malla) de cada grupo y seccionarlo como sólido.
        meshes_clusters = []
        try:
            from skimage.measure import marching_cubes
            for k in range(n_clusters):
                mask = (clusters_interp == k).astype(float)
                if np.sum(mask) > 8:
                    # Obtenemos la malla 3D del bloque cluster k
                    verts, faces, normals, values = marching_cubes(mask, level=0.5)
                    # Convertir a coordenadas reales
                    scale_x = (xx.max() - xx.min()) / (xx.shape[0] - 1) if xx.shape[0] > 1 else 1
                    scale_y = (yy.max() - yy.min()) / (yy.shape[1] - 1) if yy.shape[1] > 1 else 1
                    scale_z = (zz.max() - zz.min()) / (zz.shape[2] - 1) if zz.shape[2] > 1 else 1
                    verts_real = np.zeros_like(verts)
                    verts_real[:, 0] = verts[:, 0] * scale_x + xx.min()
                    verts_real[:, 1] = verts[:, 1] * scale_y + yy.min()
                    verts_real[:, 2] = verts[:, 2] * scale_z + zz.min()
                    surface_color = color_list[k] if isinstance(color_list[k], str) else f'rgb{color_list[k][:3]}'
                    # Aumentamos la opacidad para que se vea más sólido, pero no completamente opaco
                    mesh = go.Mesh3d(
                        x=verts_real[:, 0],
                        y=verts_real[:, 1],
                        z=verts_real[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        name=f"Grupo Interpolado {k}",
                        color=surface_color,
                        opacity=0.9,  # Más sólido, sin ser completamente opaco
                        showscale=False,
                        hoverinfo="skip"
                    )
                    meshes_clusters.append(mesh)
        except ImportError:
            print("⚠️ skimage no está instalado. No se mostrarán bloques sólidos.")
        except Exception as e:
            print(f"⚠️ Error generando bloques sólidos de clusters: {e}")

        # ---------- Fronteras / delimitaciones entre clusters (como antes, solo para contorno visual) -------------
        superficies_delimitacion = []
        try:
            from skimage.measure import marching_cubes
            for k in range(n_clusters - 1):
                mask = (clusters_interp == k).astype(float)
                if np.sum(mask) > 8:
                    verts, faces, normals, values = marching_cubes(mask, level=0.5)
                    scale_x = (xx.max() - xx.min()) / (xx.shape[0] - 1) if xx.shape[0] > 1 else 1
                    scale_y = (yy.max() - yy.min()) / (yy.shape[1] - 1) if yy.shape[1] > 1 else 1
                    scale_z = (zz.max() - zz.min()) / (zz.shape[2] - 1) if zz.shape[2] > 1 else 1
                    verts_real = np.zeros_like(verts)
                    verts_real[:, 0] = verts[:, 0] * scale_x + xx.min()
                    verts_real[:, 1] = verts[:, 1] * scale_y + yy.min()
                    verts_real[:, 2] = verts[:, 2] * scale_z + zz.min()
                    # El color de frontera: más oscuro (o negro) y más transparente
                    superficie = go.Mesh3d(
                        x=verts_real[:, 0],
                        y=verts_real[:, 1],
                        z=verts_real[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        name=f"Delimitación cluster {k}",
                        opacity=0.12,
                        color='black',
                        showscale=False,
                        hoverinfo='skip'
                    )
                    superficies_delimitacion.append(superficie)
        except ImportError:
            print("⚠️ skimage no está instalado. No se mostrarán superficies de delimitación.")
        except Exception as e:
            print(f"⚠️ Error generando delimitaciones: {e}")

        # Figura
        data = []
        if mostrar_puntos and puntos_orig is not None:
            data.append(puntos_orig)
        # Primero los bloque sólidos (Mesh3d por cluster), luego los puntos interplados, luego las fronteras
        data.extend(meshes_clusters)
        data.append(puntos_interpolados)
        data.extend(superficies_delimitacion)

        titulo = (f'Interpolación Espacial (KNN, 3D) plotly<br>'
                  f'k={clusterer.n_clusters}, peso={clusterer.w_spatial:.2f}, '
                  f'n_neighbors={interpolador.n_neighbors}<br>'
                  f'<span style="font-size:13px">(Los bloques sólidos son cada grupo/cluster. Bordes delimitan clusters)</span>')

        layout = go.Layout(
            title=titulo,
            scene=dict(
                xaxis_title='X (midx)',
                yaxis_title='Y (midy)',
                zaxis_title='Z (midz)'
            ),
            margin=dict(r=10, l=10, b=10, t=60),
            legend=dict(
                x=0.05, y=0.98,
                bgcolor='rgba(255,255,255,0.7)',
                font_size=11
            )
        )

        fig = go.Figure(data=data, layout=layout)

        # Solo guarda si se solicita
        if guardar:
            if nombre_archivo is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"interpolacion3d_plotly_k{clusterer.n_clusters}_knn{interpolador.n_neighbors}_{timestamp}.html"
            ruta = self.carpeta_salida / nombre_archivo
            fig.write_html(str(ruta))
            print(f"✅ Interpolación 3D interactiva guardada: {ruta}")

        # Solo muestra si se solicita (y no mostrar por default si mostrar=False).
        if mostrar:
            fig.show()

        # Si mostrar=False y guardar=False, solo retorna el objeto plotly.Figure sin mostrar ni guardar
        return fig

    def plot_comparacion_interpolacion(self, clusterer, interpolador,
                                      guardar=True, nombre_archivo=None, mostrar=True, nombre_atributo="starkey_min"):
        """
        Compara clusters originales vs interpolación lado a lado.
        
        Parámetros:
        -----------
        clusterer : ClusterizadorKMeans
            Clustering original
        interpolador : InterpoladorEspacial
            Interpolación realizada
        """
        if not interpolador.interpolado:
            raise ValueError("❌ Primero ejecuta interpolador.interpolar()")
        
        # Crear figura con 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # === PANEL 1: Solo clusters ===
        ax = axes[0]
        scatter1 = ax.scatter(
            clusterer.x_original,
            clusterer.z_original,
            c=clusterer.clusters,
            cmap=self.cmap_clusters,
            s=50,
            alpha=0.8,
            edgecolors='k',
            linewidth=0.5
        )
        ax.set_title('Clusters Originales', fontweight='bold', fontsize=13)
        ax.set_xlabel('X (midx)')
        ax.set_ylabel('Z (midz)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter1, ax=ax, label='Cluster')
        
        # === PANEL 2: Interpolación ===
        ax = axes[1]
        contour = ax.contourf(
            interpolador.xx,
            interpolador.zz,
            interpolador.clusters_interpolados,
            levels=np.arange(clusterer.n_clusters + 1) - 0.5,
            cmap=self.cmap_clusters,
            alpha=0.5
        )
        ax.scatter(
            clusterer.x_original,
            clusterer.z_original,
            c=clusterer.clusters,
            cmap=self.cmap_clusters,
            s=50,
            alpha=1,
            edgecolors='k',
            linewidth=0.8,
            zorder=10
        )
        ax.set_title(f'Interpolación (KNN n={interpolador.n_neighbors})',
                    fontweight='bold', fontsize=13)
        ax.set_xlabel('X (midx)')
        ax.set_ylabel('Z (midz)')
        ax.grid(alpha=0.3)
        plt.colorbar(contour, ax=ax, label='Cluster')
        
        # === PANEL 3: Atributo real ===
        ax = axes[2]
        scatter3 = ax.scatter(
            clusterer.x_original,
            clusterer.z_original,
            c=clusterer.attr_original,
            cmap=self.cmap_atributo,
            s=50,
            alpha=0.8,
            edgecolors='k',
            linewidth=0.5
        )
        ax.set_title(f'Atributo Real ({nombre_atributo})', fontweight='bold', fontsize=13)
        ax.set_xlabel('X (midx)')
        ax.set_ylabel('Z (midz)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter3, ax=ax, label=str(nombre_atributo))
        
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"comparacion_interpolacion_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Comparación guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, axes
    
    def plot_comparacion_n_neighbors(self, interpoladores_dict,
                                    guardar=True, nombre_archivo=None, mostrar=True, nombre_atributo="starkey_min"):
        """
        Compara diferentes valores de n_neighbors en una grilla.
        
        Parámetros:
        -----------
        interpoladores_dict : dict
            {n_neighbors: InterpoladorEspacial}
        """
        n_configs = len(interpoladores_dict)
        
        # Determinar layout de subplots
        if n_configs <= 4:
            nrows, ncols = 2, 2
        elif n_configs <= 6:
            nrows, ncols = 2, 3
        else:
            nrows, ncols = 3, 3
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        axes = axes.flatten() if n_configs > 1 else [axes]
        
        for idx, (n_neigh, interp) in enumerate(sorted(interpoladores_dict.items())):
            if idx >= len(axes):
                break

            
            ax = axes[idx]
            
            # Interpolación
            contour = ax.contourf(
                interp.xx,
                interp.zz,
                interp.clusters_interpolados,
                levels=np.arange(interp.clusterer.n_clusters + 1) - 0.5,
                cmap=self.cmap_clusters,
                alpha=0.4
            )
            
            # Puntos originales
            ax.scatter(
                interp.clusterer.x_original,
                interp.clusterer.z_original,
                c=interp.clusterer.clusters,
                cmap=self.cmap_clusters,
                s=30,
                alpha=0.9,
                edgecolors='k',
                linewidth=0.5,
                zorder=10
            )
            
            ax.set_title(f'n_neighbors = {n_neigh}', fontweight='bold', fontsize=12)
            ax.set_xlabel('X (midx)', fontsize=10)
            ax.set_ylabel('Z (midz)', fontsize=10)
            ax.grid(alpha=0.3)
        
        # Ocultar ejes sobrantes
        for idx in range(n_configs, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'Comparación de n_neighbors en KNN\nAtributo: {nombre_atributo}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"comparacion_knn_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Comparación n_neighbors guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, axes

    # INSERT_YOUR_CODE
    def plot_clusters_convexhull_with_ghost(self, clusterer, n_ghost=200, mostrar=True, guardar=False, nombre_archivo=None):
        """
        Muestra la gráfica 3D de los clusters, dibuja el ConvexHull y agrega puntos "ghost" equiespaciados dentro del hull.
        Además, retorna un DataFrame combinando los originales y los "ghost"
        
        Parámetros:
        -----------
        clusterer : objeto clusterer ya ajustado
        n_ghost : int, default=200
            Número de puntos fantasmas (ghost) equiespaciados a generar dentro del ConvexHull
        mostrar : bool, default=True
            Si True, muestra la figura
        guardar : bool, default=False
            Si True, guarda la figura en self.carpeta_salida
        nombre_archivo : str, default=None
            Nombre de archivo para guardar la figura
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from scipy.spatial import ConvexHull, Delaunay

        import numpy as np
        import pandas as pd

        # 1. Extraer los puntos originales
        X = np.column_stack((clusterer.x_original, clusterer.y_original, clusterer.z_original))

        # 2. Crear el ConvexHull y Delaunay para test de pertenencia
        hull = ConvexHull(X)
        delaunay = Delaunay(X[hull.vertices])

        # 3. Scatter 3D de los puntos originales, coloreados por cluster
        n_clusters = clusterer.n_clusters
        palette = self.get_cluster_palette(n_clusters)
        cluster_color_dict = {i: palette[i] for i in range(n_clusters)}
        cluster_colors = np.array([cluster_color_dict[c] for c in clusterer.clusters])

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            clusterer.x_original, clusterer.y_original, clusterer.z_original,
            c=cluster_colors, s=35, alpha=0.95, edgecolors='k', linewidth=0.5, label="Datos"
        )

        # 4. Graficar el ConvexHull (superficie)
        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])  # cerrar el triángulo
            ax.plot(X[simplex, 0], X[simplex, 1], X[simplex, 2], c='tab:gray', alpha=0.6, lw=1.3, label=None)

        # 5. Generar puntos ghost dentro del hull
        # Generamos una nube 3D "dense" en el bounding box del hull
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        n_grid = max(30, int(np.cbrt(n_ghost * 10)))  # más de n_ghost en malla para filtrar después

        gx = np.linspace(mins[0], maxs[0], n_grid)
        gy = np.linspace(mins[1], maxs[1], n_grid)
        gz = np.linspace(mins[2], maxs[2], n_grid)
        mgx, mgy, mgz = np.meshgrid(gx, gy, gz)
        ghost_grid = np.column_stack([mgx.ravel(), mgy.ravel(), mgz.ravel()])

        # Filtrar solo los que caen dentro del hull
        in_hull_mask = delaunay.find_simplex(ghost_grid) >= 0
        ghost_candidates = ghost_grid[in_hull_mask]

        # Tomar n_ghost puntos equiespaciados del conjunto filtrado
        if len(ghost_candidates) > n_ghost:
            stride = max(1, int(len(ghost_candidates) / n_ghost))
            ghost_points = ghost_candidates[::stride][:n_ghost]
        else:
            ghost_points = ghost_candidates

        # 6. Graficar los puntos ghost
        ax.scatter(
            ghost_points[:,0], ghost_points[:,1], ghost_points[:,2],
            c='aqua', s=12, alpha=0.75, marker='^', label='Ghost'
        )
        
        # 7. Extras (leyenda, títulos)
        ax.set_title("Clusters 3D + ConvexHull + Puntos Ghost", fontweight='bold', fontsize=13)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        handles = [plt.Line2D([0],[0], marker='o', color='w', label=f'Cluster {k}',
                    markerfacecolor=cluster_color_dict[k], markeredgecolor='k', markersize=8) for k in range(n_clusters)]
        handles.append(plt.Line2D([0],[0], color='tab:gray', lw=1.3, label='ConvexHull'))
        handles.append(plt.Line2D([0],[0], marker='^', color='aqua', label='Ghost', markerfacecolor='aqua',
                                    markeredgecolor='k', markersize=7, linestyle='None'))
        ax.legend(handles=handles, fontsize=8, frameon=True, loc='best')

        ax.grid(alpha=0.28)

        plt.tight_layout()

        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"convexhull_ghost_{timestamp}.png"
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Figura convexhull+ghost guardada en: {ruta}")

        if mostrar:
            plt.show()
        else:
            plt.close()

        # --- Generar DataFrame combinando puntos originales y ghost ---
        # Puntos originales
        df_original = pd.DataFrame({
            'x': clusterer.x_original,
            'y': clusterer.y_original,
            'z': clusterer.z_original,
            'atributo': clusterer.attr_original,
            'cluster': clusterer.clusters
        })
        # Puntos ghost
        n_ghost_real = ghost_points.shape[0]
        df_ghost = pd.DataFrame({
            'x': ghost_points[:,0],
            'y': ghost_points[:,1],
            'z': ghost_points[:,2],
            'atributo': np.nan,
            'cluster': np.nan
        })

        df_combined = pd.concat([df_original, df_ghost], ignore_index=True)

        return fig, ax, ghost_points, df_combined
