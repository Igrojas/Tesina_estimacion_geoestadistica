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
                     nombre_archivo=None, mostrar=True):
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
        ax.set_title(titulo, fontweight='bold', fontsize=14)
        
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
            c=clusterer.attr_original,
            cmap=self.cmap_atributo,   # Dejar continuous cmap para variables continuas
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
                nombre_archivo = f"comparacion_k{clusterer.n_clusters}_w{int(clusterer.w_spatial*100)}_{timestamp}.png"
            
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
                nombre_archivo = f"boxplots_k{clusterer.n_clusters}_w{int(clusterer.w_spatial*100)}_{timestamp}.png"
            
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
        Crea un dashboard (2x2) con visualizaciones de clusters, mostrando todas las leyendas completas y específicas para cada gráfica.

        Orden:
        1. Efecto proporcional (Coeficiente de variación por cluster)
        2. Lognormal Probability Plot por cluster
        3. Boxplots del atributo por cluster
        4. Mapa de clusters igual que plot_clusters
        """
        if not clusterer.ajustado:
            raise ValueError("❌ El clusterer debe estar entrenado")

        import numpy as np
        import pandas as pd
        from scipy.stats import lognorm

        # Preparar datos y métricas
        stats_clusters = clusterer.get_stats()
        n_clusters = clusterer.n_clusters

        df_temp = pd.DataFrame({
            'atributo': clusterer.attr_original,
            'cluster': clusterer.clusters
        })

        # Paleta fija para clusters: cada cluster tiene el mismo color en todos los ejes
        palette = self.get_cluster_palette(n_clusters)
        cluster_color_dict = {i: palette[i] for i in range(n_clusters)}  # cluster_id -> color RGB

        # ==== FIGURA 2x2 ====
        fig, axs = plt.subplots(2, 2, figsize=(20, 14))

        # --- (0,0) Efecto proporcional (Coeficiente de variación por cluster)
        ax0 = axs[0, 0]
        cvs = [stats_clusters[i].get('efecto_proporcional', np.nan) for i in range(n_clusters)]
        clusters_x = np.arange(n_clusters)
        legend_handles = []
        legend_labels = []
        # Cada punto con color de cluster y leyenda para cada uno
        for i in range(n_clusters):
            sc = ax0.scatter(clusters_x[i], cvs[i], color=cluster_color_dict[i], s=110, zorder=3, label=f'Cluster {i}')
            legend_handles.append(sc)
            legend_labels.append(f'Cluster {i}')
        # Linea de tendencia (crimson)
        z = np.polyfit(clusters_x, cvs, 1)
        p = np.poly1d(z)
        ln, = ax0.plot(clusters_x, p(clusters_x), color='crimson', linestyle='--', linewidth=2, label='Tendencia lineal')
        legend_handles.append(ln)
        legend_labels.append('Tendencia lineal')
        # Etiquetas de puntos, también con color
        for i, cv in enumerate(cvs):
            ax0.text(i, cv + 0.001, f"{cv:.2f}", ha='center', va='bottom', fontsize=16, fontweight='bold', color='black')
        ax0.set_xlabel('Cluster', fontsize=11)
        ax0.set_ylabel('Coef. de Variación (std/media)', fontsize=11)
        ax0.set_title('Efecto Proporcional (CV) por Cluster', fontweight='bold', fontsize=13)
        ax0.grid(axis='y', alpha=0.3)
        # Mostrar leyenda completa de clusters + tendencia
        ax0.legend(legend_handles, legend_labels)

        # --- (0,1) Probability lognormal plot por cluster
        ax1 = axs[0, 1]
        all_handles = []
        all_labels = []
        for i in range(n_clusters):
            data = df_temp[df_temp["cluster"] == i]["atributo"].dropna().values
            if len(data) < 3:
                continue
            shape, loc, scale = lognorm.fit(data, floc=0)
            sorted_data = np.sort(data)
            prob = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
            theo = lognorm.ppf(prob, shape, loc=loc, scale=scale)
            color = cluster_color_dict[i]
            line_data, = ax1.plot(sorted_data, prob, marker='o', linestyle='', color=color, label=f'Cluster {i} datos')
            line_theo, = ax1.plot(theo, prob, linestyle='-', color=color, alpha=0.7, label=f'Cluster {i} lognorm')
            all_handles.extend([line_data, line_theo])
            all_labels.extend([f'Cluster {i} datos', f'Cluster {i} lognorm'])
        ax1.set_xlabel('starkey_min')
        ax1.set_ylabel('Probabilidad no excedencia')
        ax1.set_title(f'Probability Lognormal Plot\npor Cluster')
        ax1.set_yscale('logit')
        ax1.grid(alpha=0.25, which='both')
        if all_handles:
            ax1.legend(all_handles, all_labels, fontsize=8, loc="best", frameon=True)

        # --- (1,0) Boxplots de atributo por cluster
        ax2 = axs[1, 0]
        import seaborn as sns
        sns.boxplot(x='cluster', y='atributo', data=df_temp, palette=palette, ax=ax2)
        scatter_handles = []
        for i in range(n_clusters):
            media_val = stats_clusters[i].get('mean', np.nan)
            sc = ax2.scatter(i, media_val,
                        color='crimson', s=50, marker='D', zorder=10,
                        edgecolor='black', linewidth=0.5, label='Media')
            scatter_handles.append(sc)
        ax2.set_xlabel('Cluster', fontsize=11)
        ax2.set_ylabel('Atributo (starkey_min)', fontsize=11)
        ax2.set_title('Distribución de Atributo por Cluster', fontweight='bold', fontsize=13)
        ax2.grid(axis='y', alpha=0.3)
        if scatter_handles:
            ax2.legend([scatter_handles[0]], ['Media'], loc='best')

        # --- (1,1) El mismo gráfico que plot_clusters
        ax3 = axs[1, 1]
        # PREPARACIÓN de los datos igual que en plot_clusters
        cluster_colors = np.array([cluster_color_dict[c] for c in clusterer.clusters])
        scatter = ax3.scatter(
            clusterer.x_original, 
            clusterer.z_original,
            c=cluster_colors,
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        # Título como plot_clusters
        titulo = (f'Clustering K-means\n'
                  f'k={clusterer.n_clusters}, peso={clusterer.w_spatial:.2f}')
        ax3.set_title(titulo, fontweight='bold', fontsize=14)
        ax3.set_xlabel('X (midx)', fontsize=12)
        ax3.set_ylabel('Z (midz)', fontsize=12)
        ax3.grid(alpha=0.3)
        unique_clusters = np.unique(clusterer.clusters)
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=cluster_color_dict[k], markeredgecolor='k',
                              markersize=10, label=f'Cluster {k}') for k in unique_clusters]
        ax3.legend(handles=handles, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        metricas = clusterer.get_global_metrics()
        texto_metrica = f"Std prom: {metricas['std_prom']:.2f}"
        ax3.text(0.02, 0.98, texto_metrica,
                 transform=ax3.transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 fontsize=10)

        # --- Título
        fig.suptitle(f"Dashboard 2x2 - Clustering K-means\nk={n_clusters}, peso_espacial={clusterer.w_spatial:.2f}",
                     fontsize=18, fontweight='bold', y=0.99)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Guardar
        if guardar:
            if nombre_archivo is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"dashboard2x2_k{n_clusters}_w{int(clusterer.w_spatial*100)}_{timestamp}.png"

            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Dashboard guardado: {ruta}")

        if mostrar:
            plt.show()
        else:
            plt.close()

        return fig


    def plot_interpolacion(self, interpolador, mostrar_puntos=True,
                            guardar=True, nombre_archivo=None, mostrar=True):


        if not interpolador.interpolado:
            raise ValueError("❌ El interpolador no está interpolado")

        clusterer = interpolador.clusterer

        fig, ax = plt.subplots(figsize=(10, 6))

        contour = ax.contourf(
            interpolador.xx,
            interpolador.zz,
            interpolador.clusters_interpolados,
            levels=np.arange(clusterer.n_clusters + 1) - 0.5,
            cmap = self.cmap_clusters,
            alpha = 0.4,
        )

        if mostrar_puntos:
            scatter = ax.scatter(
                clusterer.x_original,
                clusterer.z_original,
                c=clusterer.clusters,
                cmap=self.cmap_clusters,
                s=50,
                alpha=0.9,
                edgecolors='k',
                linewidth=0.8,
                zorder=10
            )

        titulo = (f'Interpolación Espacial (KNN)\n'
                 f'k={clusterer.n_clusters}, peso={clusterer.w_spatial:.2f}, '
                 f'n_neighbors={interpolador.n_neighbors}')
        ax.set_title(titulo, fontweight='bold', fontsize=13)
        
        # Etiquetas
        ax.set_xlabel('X (midx)', fontsize=12)
        ax.set_ylabel('Z (midz)', fontsize=12)
        ax.grid(alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Cluster', fontsize=11)
        
        # Información
        info = interpolador.get_info()
        texto = (f"Grilla: {info['n_points']}×{info['n_points']}\n"
                f"KNN neighbors: {info['n_neighbors']}")
        ax.text(0.02, 0.98, texto,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               fontsize=9)
        
        plt.tight_layout()
        
        # Guardar
        if guardar:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"interpolacion_k{clusterer.n_clusters}_knn{interpolador.n_neighbors}_{timestamp}.png"
            
            ruta = self.carpeta_salida / nombre_archivo
            plt.savefig(ruta, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Interpolación guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig, ax

    def plot_comparacion_interpolacion(self, clusterer, interpolador,
                                      guardar=True, nombre_archivo=None, mostrar=True):
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
        ax.set_title('Atributo Real', fontweight='bold', fontsize=13)
        ax.set_xlabel('X (midx)')
        ax.set_ylabel('Z (midz)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter3, ax=ax, label='starkey_min')
        
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
                                    guardar=True, nombre_archivo=None, mostrar=True):
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
        
        fig.suptitle('Comparación de n_neighbors en KNN',
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