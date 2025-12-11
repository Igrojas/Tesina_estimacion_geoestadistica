#%%
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import numpy as np


df = pd.read_csv('data/raw/com_p_plt_entry 1.csv', sep=',')

df.head()

print(f"N Filas: {df.shape[0]}")
print(f"N Columnas: {df.shape[1]}")

df.columns
drops_cols = ["compid","geocod", "dhid",
                "topx","topy","topz",
                "botx","boty","botz",
                "length", "from","to",
                "bound","zmin","zmin01","ore"]

df = df.drop(columns=drops_cols)
display(df.describe())
#%%
def boxplot_variables_subplot(
    df, 
    coord_cols=["midx", "midy", "midz"], 
    save_fig=False, 
    fig_path="figs_boxplots/",
    dpi=300
):
    """
    Crea un gráfico de subplots con un boxplot por cada variable numérica (sin coordenadas).
    Limpia ceros y -99/-99.0. Estético, sobrio y con título individual por variable.
    Permite guardar la figura opcionalmente.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    # Crear carpeta si no existe
    if save_fig and not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Selección numérica sin coordenadas
    cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in coord_cols]
    if not cols:
        print("No hay columnas numéricas (sin coord) para graficar.")
        return

    # Preprocesar: limpiar datos por columna
    cleaned_data = {}
    for col in cols:
        d = df[col].replace([-99, -99.0], np.nan)
        d = d.where(d != 0, np.nan).dropna()
        if d.empty:
            print(f"{col}: Sin datos válidos (todos NaN o ceros o -99).")
            continue
        cleaned_data[col] = d
    if not cleaned_data:
        print("No hay datos válidos para graficar.")
        return

    n_vars = len(cleaned_data)

    # Determinar la cuadricula del subplot
    if n_vars == 1:
        nrows, ncols = 1, 1
    else:
        ncols = min(4, n_vars)
        nrows = math.ceil(n_vars / ncols)

    # Tamaño de figura adaptado
    # 4 pulgadas por subplot aproximadamente (pero nunca menos de 3.5 ni más de 5)
    width = max(3.5, min(ncols * 4, 20))
    height = max(3.5, min(nrows * 4, 21))

    # Estilo sobrio para Tesis
    sns.set_theme(style="whitegrid", palette="muted", font="serif", rc={
        "axes.labelweight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "#444",
        "grid.linewidth": 0.7,
        "axes.linewidth": 1.25
    })
    box_color = "#555F63"
    median_color = "#C94821"
    flier_color = "#888B8D"
    boxprops    = dict(facecolor=box_color, edgecolor="#333333", linewidth=1.7, alpha=0.68)
    whiskerprops= dict(color="#333333", linewidth=1.15)
    capprops    = dict(color="#333333", linewidth=1.1)
    medianprops = dict(color=median_color, linewidth=2.2)
    flierprops  = dict(marker='o', markersize=3, markerfacecolor=flier_color, markeredgecolor=flier_color, alpha=0.44)

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), dpi=dpi, squeeze=False)
    axes = axes.flatten()

    for i, (col, d) in enumerate(cleaned_data.items()):
        ax = axes[i]
        sns.boxplot(
            y=d,
            ax=ax,
            color=box_color,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            flierprops=flierprops,
            medianprops=medianprops,
            width=0.37,
            showmeans=False,
            orient='v'
        )
        ax.set_title(f"{col}", fontsize=13, weight='bold', pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Valor", fontsize=11, weight='bold', labelpad=6)
        ax.grid(axis="y", ls="--", alpha=0.18)
        sns.despine(ax=ax, left=False, bottom=True)
    # Quitar ejes vacíos si sobran
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=1.5, w_pad=1.6, h_pad=1.2)
    fig.suptitle("Boxplots por variable numérica", fontsize=16, weight='bold', y=1.03)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(top=0.92)
    
    if save_fig:
        fname = os.path.join(fig_path, f"boxplots_subplots_tesis.png")
        plt.savefig(fname, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.show()

boxplot_variables_subplot(df)

#%%
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

# Elegir aleatoriamente solo el 50% de los datos para graficar
df_sample = df.sample(frac=0.01)
print(df_sample.shape)
# Gráfica 3D principal de los puntos (midx, midy, midz) usando el subconjunto
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    df_sample["midx"], 
    df_sample["midy"], 
    df_sample["midz"], 
    s=3, 
    c=df_sample["midz"], 
    cmap='viridis', 
    alpha=0.6
)
ax.set_xlabel("midx")
ax.set_ylabel("midy")
ax.set_zlabel("midz")
ax.set_title("Dispersión 3D de midx, midy, midz (50% de los datos)", fontsize=14, weight='bold', pad=16)
plt.tight_layout()
plt.show()

# Subplots 1x3: proyecciones XY, XZ, YZ usando el subconjunto
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# XY
axes[0].scatter(df_sample["midx"], df_sample["midy"], s=3, c=df_sample["midz"], cmap="viridis", alpha=0.5)
axes[0].set_xlabel("midx")
axes[0].set_ylabel("midy")
axes[0].set_title("Proyección XY")
axes[0].grid(True, alpha=0.15)

# XZ
axes[1].scatter(df_sample["midx"], df_sample["midz"], s=3, c=df_sample["midy"], cmap="viridis", alpha=0.5)
axes[1].set_xlabel("midx")
axes[1].set_ylabel("midz")
axes[1].set_title("Proyección XZ")
axes[1].grid(True, alpha=0.15)

# YZ
axes[2].scatter(df_sample["midy"], df_sample["midz"], s=3, c=df_sample["midx"], cmap="viridis", alpha=0.5)
axes[2].set_xlabel("midy")
axes[2].set_ylabel("midz")
axes[2].set_title("Proyección YZ")
axes[2].grid(True, alpha=0.15)

plt.tight_layout()
plt.suptitle("Proyecciones 2D de midx, midy, midz (50% de los datos)", fontsize=15, weight='bold', y=1.02)
plt.subplots_adjust(top=0.87)
plt.show()




#%%
import seaborn as sns

def filtrar_columnas_no_cero(df_input, coord_cols=["midx", "midy", "midz"], porcentaje_min=0, verbose=True):
    """
    Filtra y devuelve un df solo con columnas donde al menos porcentaje_min% de los datos NO son ceros.
    Grafica el resumen de completitud, cantidad de ceros, y cantidad/porcentaje de -99 sobre los valores distintos de cero por variable.
    """
    # Quitamos columnas de coordenadas para el chequeo
    df_stats = df_input.drop(columns=coord_cols, errors='ignore')
    total_counts = df_stats.shape[0]

    # Calcular cantidad y porcentaje de valores NO cero por columna
    count_no_cero = (df_stats != 0).sum()
    porcentaje_no_cero = count_no_cero / total_counts * 100

    # También calcular para cada columna, cuántos valores son -99, y de esos, cuántos están en los que no son cero
    count_menos99 = (df_stats == -99).sum()
    # Solo de los NO cero, cuántos son -99
    # Para evitar división por cero colocamos 0 cuando count_no_cero es 0
    count_menos99_en_no_cero = ((df_stats == -99) & (df_stats != 0)).sum()
    porcentaje_menos99_sobre_no_cero = count_menos99_en_no_cero.divide(count_no_cero.where(count_no_cero > 0)).fillna(0) * 100

    # Columnas a conservar: más del porcentaje_min% NO cero
    cols_good = porcentaje_no_cero[porcentaje_no_cero >= porcentaje_min].index.tolist()
    df_filtrado = df_input[cols_good + [c for c in coord_cols if c in df_input.columns]]

    if verbose:
        print(f"Columnas conservadas ({len(cols_good)}): {cols_good}")
        
    # Recalcular métricas solo sobre las columnas seleccionadas
    df_stats_filtrado = df_filtrado.drop(columns=coord_cols, errors='ignore')
    non_null_counts = df_stats_filtrado.notnull().sum()
    null_counts = df_stats_filtrado.isnull().sum()
    zero_counts = (df_stats_filtrado == 0).sum()
    total_counts_filtrado = df_stats_filtrado.shape[0]
    # Para las que quedan, recalcular métricas - utilizar subset de counts ya calculados
    count_no_cero_filtrado = count_no_cero[cols_good]
    count_menos99_en_no_cero_filtrado = count_menos99_en_no_cero[cols_good]
    porcentaje_menos99_sobre_no_cero_filtrado = porcentaje_menos99_sobre_no_cero[cols_good]

    summary = pd.DataFrame({
        'No Nulos': non_null_counts,
        'Nulos': null_counts,
        'Ceros': zero_counts,
        'Total': total_counts_filtrado,
        'Porcentaje No Nulo': (non_null_counts / total_counts_filtrado * 100).round(2),
        'Porcentaje Ceros': (zero_counts / total_counts_filtrado * 100).round(2),
        'Porcentaje No Cero': (count_no_cero_filtrado / total_counts_filtrado * 100).round(2),
        'No Cero': count_no_cero_filtrado,
        '“-99 en No 0”': count_menos99_en_no_cero_filtrado,
        '“-99 s/No 0 (%)”': porcentaje_menos99_sobre_no_cero_filtrado.round(2)
    })

    print("Resumen de completitud de datos, cantidad de ceros y -99 por variable (solo columnas filtradas):")
    print(summary)

    # Visualización
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator

    no_null_color = "#2E4053"      # Azul oscuro
    zero_color = "#AEB6BF"         # Gris profesional
    menos99_color = "#D35400"      # Naranja para -99

    fig, ax1 = plt.subplots(figsize=(15, 7))
    bar_width = 0.6

    # Barras de valores no nulos, ceros y -99 en los no ceros
    bars1 = ax1.bar(summary.index, summary['No Nulos'], color=no_null_color, width=bar_width, label='Valores no nulos', edgecolor="black", zorder=3)
    bars2 = ax1.bar(summary.index, summary['Ceros'], color=zero_color, width=bar_width*0.65, label='Valores cero', edgecolor="black", zorder=4)
    # Nueva capa: -99 (dentro de los no ceros)
    bars3 = ax1.bar(summary.index, summary['“-99 en No 0”'], color=menos99_color, width=bar_width*0.4, label='“-99 en No 0”', edgecolor="black", zorder=5)

    # Etiquetas en barras
    for rect in bars1:
        height = rect.get_height()
        ax1.annotate(f'{int(height)}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, color=no_null_color, fontweight='bold')
    for rect in bars2:
        height = rect.get_height()
        if height > 0:
            ax1.annotate(f'{int(height)}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, color=zero_color, fontweight='bold')

    for rect in bars3:
        height = rect.get_height()
        if height > 0:
            ax1.annotate(f'{int(height)}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, color=menos99_color, fontweight='bold')

    ax1.set_xticklabels(summary.index, rotation=45, ha='right', fontsize=11)
    ax1.set_ylabel("Cantidad de valores", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Variables", fontsize=12, fontweight='bold')
    ax1.set_title("Columnas filtradas (>%.1f%% datos no cero): Completitud, ceros y -99 en no cero\n" % porcentaje_min, fontsize=16, fontweight='bold', color="#273746")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.grid(axis='y', linestyle='-', alpha=0.25, zorder=0)
    ax1.set_facecolor('white')
    fig.patch.set_facecolor('white')

    custom_legend = [
        mpatches.Patch(color=no_null_color, label='Valores no nulos'),
        mpatches.Patch(color=zero_color, label='Valores cero'),
        mpatches.Patch(color=menos99_color, label='“-99 en No 0”')
    ]
    ax1.legend(handles=custom_legend, fontsize=12, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.show()

    print(
        "\nExplicación:\n"
        f"Se han conservado únicamente las columnas donde más del {porcentaje_min}% de los datos NO son ceros.\n"
        "Ahora también puedes ver cuántos de esos NO ceros corresponden al código -99, tanto como cantidad como porcentaje.\n"
        "El gráfico y tabla resumen permiten identificar variables con mucha sparcity de ceros y/o valores especiales (-99) y ayudan a depurar el dataframe para el análisis."
    )

    return df_filtrado

# Uso: devuelve el nuevo dataframe filtrado y grafica
df_filtrado_no_cero = filtrar_columnas_no_cero(df,porcentaje_min= 15)
#%%
import seaborn as sns
import numpy as np
# 1. Filtrar solo filas donde la columna "ph" no sea nula
df_ph = df_filtrado_no_cero[df_filtrado_no_cero['ph'].notnull()].copy()

# 2. Seleccionar variables de interés geoestadístico (incluyendo coordenadas)
geo_vars = [
    'midx', 'midy', 'midz', 'cut', 'cus', 'cuscn', 'cussec', 
    'cur', 'ca', 'ph', 'au', 'ag', 'fe', 'frm'
]
geo_vars_presente = [col for col in geo_vars if col in df_ph.columns]
df_geo = df_ph[geo_vars_presente]

# 3. Estadísticas descriptivas
print("=== Estadísticas descriptivas ===")
display(df_geo.describe().T)

# --- ANALISIS: Estacionariedad respecto a las coordenadas ---
# Queremos 1x3 subplots por variable (cut, cus, etc.), comparando contra midx, midy, midz, SOLO si el valor a graficar es != 0.
from matplotlib import gridspec

est_vars = [v for v in geo_vars_presente if v not in ['midx', 'midy', 'midz']]
coord_vars = ['midx', 'midy', 'midz']

for var in est_vars:
    df_var = df_geo[df_geo[var] != 0]  # filtrar ceros
    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(1, 3)
    axes = [plt.subplot(gs[i]) for i in range(3)]
    for i, coord in enumerate(coord_vars):
        sns.scatterplot(
            data=df_var, x=coord, y=var, alpha=0.45, ax=axes[i],
            edgecolor=None, s=25
        )
        axes[i].set_title(f"{var} vs {coord}", fontsize=13, fontweight='bold')
        axes[i].set_xlabel(coord, fontsize=11)
        axes[i].set_ylabel(var if i == 0 else "", fontsize=11)
        axes[i].grid(axis='both', linestyle='--', linewidth=0.4, alpha=0.14)
        axes[i].set_facecolor('#f9f9fd')
    plt.suptitle(f'Estacionariedad de "{var}" respecto a coordenadas', fontsize=15, fontweight='bold', color="#22426e")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    sns.despine()
    plt.show()

# --- Matriz de correlación SIN coordenadas ---
vars_sin_coords = [v for v in geo_vars_presente if v not in ['midx', 'midy', 'midz']]
if len(vars_sin_coords) > 1:
    df_no_coord = df_geo[vars_sin_coords]
    corr_matrix = df_no_coord.corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
        linewidths=0.5, mask=mask, square=True, cbar_kws={"shrink": .8},
        annot_kws={"fontsize":10}
    )
    plt.title("Matriz de correlación (variables geoestadísticas SIN coordenadas)", fontsize=16, fontweight='bold', color="#193151")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # SCATTERS de correlaciones fuertes (>0.5, absoluto, y excluyendo la diagonal)
    strong_corrs = []
    n = len(vars_sin_coords)
    for i in range(n):
        for j in range(i+1, n):
            v1, v2 = vars_sin_coords[i], vars_sin_coords[j]
            corr_val = corr_matrix.loc[v1, v2]
            if abs(corr_val) > 0.5:
                strong_corrs.append((v1, v2, corr_val))

    if strong_corrs:
        ncols = 2 if len(strong_corrs) > 1 else 1
        nrows = (len(strong_corrs) + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs_flat = axs.flatten() if axs.ndim>0 else [axs]
        for idx, (v1, v2, corr_val) in enumerate(strong_corrs):
            ax = axs_flat[idx]
            df_pair = df_no_coord[(df_no_coord[v1] != 0) & (df_no_coord[v2] != 0)]
            sns.scatterplot(data=df_pair, x=v1, y=v2, alpha=0.45, ax=ax, s=30)
            ax.set_title(f"{v1} vs {v2} (r={corr_val:.2f})", fontsize=13, fontweight='bold', color="#284e56")
            ax.set_xlabel(v1, fontsize=11)
            ax.set_ylabel(v2, fontsize=11)
            ax.grid(axis='both', linestyle=':', linewidth=0.4, alpha=0.13)
            ax.set_facecolor('#f8fafd')
        # Remove empty axes
        for j in range(len(strong_corrs), len(axs_flat)):
            fig.delaxes(axs_flat[j])
        plt.suptitle("Relaciones altamente correlacionadas (>0.5)", fontsize=16, fontweight='bold', color="#0e344a")
        plt.tight_layout(rect=[0,0,1,0.97])
        sns.despine()
        plt.show()
    else:
        print('-- No hay correlaciones fuertes (>0.5) entre las variables numéricas sin coordenadas --')

# Ejemplo directo de relación entre ph y elevación (midz) -- OMITIDO, ya contemplado arriba
#%%
def plot_pca_side_by_side(df, coord_cols=["midx", "midy", "midz"]):
    """
    Muestra dos subplots: 
    - Izquierda: datos proyectados en PC1 vs PC2 (PCA).
    - Derecha: igual que izquierda pero añade las CARGAS (flechas/loadings de variables).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Seleccionar columnas numéricas que no sean coordenadas
    num_cols = [col for col in df.columns 
                if (df[col].dtype in [np.float64, np.float32, np.int64, np.int32]) 
                and (col not in coord_cols)]

    # Filtrar filas que no tengan -99 o -99.0 en ninguna variable seleccionada antes del PCA (los ceros se consideran válidos)
    df_pca = df[num_cols].replace([-99, -99.0], np.nan)
    df_pca = df_pca.loc[~(df_pca.isna().any(axis=1))].copy()

    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca.values)

    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_

    # Loadings: shape [n_variables, 2] para PC1 y PC2
    loadings = pca.components_.T[:, :2]

    # Tema de seaborn
    sns.set_theme(style="whitegrid", palette="muted", font="serif")

    # Hacer el subplot lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: solo scatter de PC1 vs PC2
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], alpha=0.45, s=32, edgecolor="k", ax=axs[0])
    axs[0].set_xlabel(f"PC1 ({explained_var[0]:.1%} var. expl.)", fontsize=12, fontweight="bold")
    axs[0].set_ylabel(f"PC2 ({explained_var[1]:.1%} var. expl.)", fontsize=12, fontweight="bold")
    axs[0].set_title("PCA: Proyección en PC1 vs PC2", fontsize=15, fontweight='bold', color="#193151")
    axs[0].grid(axis="both", linestyle=":", linewidth=0.4, alpha=0.15)
    axs[0].set_facecolor("#f8fafd")

    # Right: scatter + loadings
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], alpha=0.45, s=32, edgecolor="k", ax=axs[1])
    # Escalado para flechas (mismo método que original)
    scalefactor = 2.5 * np.max(np.abs(X_pca[:, :2])) / (np.max(np.abs(loadings)) + 1e-8)
    for i, (name, vec) in enumerate(zip(num_cols, loadings)):
        axs[1].arrow(0, 0, vec[0]*scalefactor, vec[1]*scalefactor, 
                     color='#DB5229', alpha=0.72, 
                     head_width=0.28, head_length=0.25, linewidth=2, length_includes_head=True)
        axs[1].text(vec[0]*scalefactor*1.11, vec[1]*scalefactor*1.11, name, 
                    color="#1C315E", fontsize=11, weight='bold', ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#BBB", lw=0.6, alpha=0.8))
    axs[1].set_xlabel(f"PC1 ({explained_var[0]:.1%} var. expl.)", fontsize=12, fontweight="bold")
    axs[1].set_ylabel(f"PC2 ({explained_var[1]:.1%} var. expl.)", fontsize=12, fontweight="bold")
    axs[1].set_title("PCA: PC1 vs PC2 + cargas (loadings)", fontsize=15, fontweight='bold', color="#193151")
    axs[1].grid(axis="both", linestyle=":", linewidth=0.4, alpha=0.17)
    axs[1].set_facecolor("#f8fafd")

    # Mejorar espaciado/estilo
    plt.tight_layout(w_pad=4)
    sns.despine()
    plt.show()

    # Adicional (opcional): gráfico de varianza explicada
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[f"C{i+1}" for i in range(len(explained_var))], y=explained_var, color="#2B538C")
    plt.plot(range(len(explained_var)), np.cumsum(explained_var), marker="o", color="#E67E22", linewidth=2, label="Acumulada")
    plt.ylabel("Varianza explicada")
    plt.xlabel("Componente principal")
    plt.title("Varianza explicada por componente (PCA)", fontsize=16, fontweight='bold', color="#193151")
    plt.xticks(rotation=40)
    for i, v in enumerate(explained_var):
        plt.text(i, v + 0.01, f"{v:.2%}", ha="center", va="bottom", fontsize=9, color="#1C315E")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Heatmap de cargas (PC1 y PC2)
    plt.figure(figsize=(9, 5))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm",
                yticklabels=num_cols, xticklabels=["PC1", "PC2"],
                cbar_kws={"shrink": .7}, annot_kws={"fontsize":10})
    plt.title("Cargas de variables en PC1 y PC2", fontsize=15, fontweight='bold', color="#193151")
    plt.tight_layout()
    plt.show()

# Para usar:
plot_pca_side_by_side(df_filtrado_no_cero)
#%%

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def kmeans_comparacion_3d(
    df, 
    ks=(6, 7, 8),   # Cantidades de clusters a probar
    coord_cols=["midx", "midy", "midz"], 
    incluir_coords=True,
    random_state=42,
    graficar=True
):
    """
    Ejecuta KMeans para cada valor de k en ks.
    Devuelve un diccionario: para cada k, un dataframe original completo (todas las variables originales, sin filas NA en las usadas para clustering) y la asignación de cluster como columna nueva llamada "cluster".

    Si graficar=True, también muestra los subplots 3D con los clusters encontrados.

    Coord_cols: columnas de coordenadas para visualizar.
    """
    resultados = {}

    # Asegurarse de que las coords espaciales están en las columnas a usar para clustering
    if incluir_coords:
        # Nos aseguramos de que todas las columnas (incluyendo coords espaciales) están presentes una única vez
        use_cols = []
        seen = set()
        for col in list(df.columns) + [c for c in coord_cols if c not in df.columns]:
            if col not in seen:
                use_cols.append(col)
                seen.add(col)
    else:
        # Aseguramos que las coord_cols NO estén en use_cols
        use_cols = [col for col in df.columns if col not in coord_cols]

    # Drop NA para estas variables (solo filas completas en estas columnas serán usadas para clustering)
    X = df[use_cols].dropna()

    # Comprobar que coord_cols están presentes en X para graficar (siempre deben estar)
    for coord in coord_cols:
        if coord not in X.columns:
            # Si falta, buscamos obtener la columna correspondiente del df original
            X[coord] = df.loc[X.index, coord]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Coordenadas de las filas NO nulas (después de dropna)
    midx = X["midx"]
    midy = X["midy"]
    midz = X["midz"]

    if graficar:
        fig = plt.figure(figsize=(6 * len(ks), 6))

    for i, k in enumerate(ks):
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=random_state, n_init=15)
        cluster_labels = kmeans.fit_predict(X_scaled)
        # Dataframe con resultado: el dataframe original completo (todas sus columnas)
        # pero SOLO en los índices (filas) usados para clustering (sin los NA)
        # Se le agrega la columna 'cluster'
        df_full = df.loc[midx.index].copy()  # conservar todas las columnas originales
        df_full["cluster"] = cluster_labels
        resultados[f"cluster_k{k}"] = df_full
        # Graficar:
        if graficar:
            ax = fig.add_subplot(1, len(ks), i + 1, projection='3d')
            ax.scatter(
                midx, midy, midz, 
                c=cluster_labels, cmap='Set2', s=15, alpha=0.68, edgecolor="k", linewidth=0.15
            )
            ax.set_xlabel("midx")
            ax.set_ylabel("midy")
            ax.set_zlabel("midz")
            ax.set_title(f"KMeans: k={k}")
            ax.grid(alpha=0.13)
    if graficar:
        plt.suptitle(f"KMeans clustering 3D para k={ks}", fontsize=16, fontweight='bold', color="#194a85")
        plt.tight_layout(w_pad=2.9)
        plt.subplots_adjust(top=0.85)
        plt.show()

    return resultados

# Ejemplo de uso:
# Para pedir solo 3 clusters distintos (por ejemplo k=6, 7, 8) y obtener un dataframe completo para cada uno:
df_filtrado_sample = df_filtrado_no_cero.sample(frac=0.01)
clusters_por_k = kmeans_comparacion_3d(df_filtrado_sample, ks=(6,7,8), incluir_coords=True, graficar=True)

# Acceder a cada dataframe completo por número de clusters:
df_clusters_k6 = clusters_por_k["cluster_k6"]
df_clusters_k7 = clusters_por_k["cluster_k7"]
df_clusters_k8 = clusters_por_k["cluster_k8"]

# Mostrar un ejemplo (primeras filas para k=6, 7, 8)
df_clusters_k6.head()
df_clusters_k7.head()
df_clusters_k8.head()


# %%
# === Dashboard 2x2 para clusters en 3D (adaptado del dashboard de 2D) ===
def dashboard_clusters_3d_variable(
    df, attr_col, show=True, nombre=None,
    elev=18, azim=135
):
    """
    Dashboard tipo 2x2 para análisis de clusters en 3D.

    Paneles:
    1. Coeficiente de variación (CV) por cluster
    2. Probability lognormal plot por cluster (de atributo)
    3. Boxplot del atributo por cluster
    4. Mapa 3D de clusters

    Parámetros:
    -----------
    df: DataFrame con las columnas 'midx','midy','midz', 'cluster' y una columna numérica para estadísticas
    attr_col: str. Nombre de la columna de atributo (ej: "cus").
    show: bool, mostrar figura
    nombre: str o None, si se quiere guardar figura
    elev, azim: vista inicial 3D

    Devuelve:
    ---------
    fig, axs (axs es np.array 2x2)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy.stats import lognorm
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.lines import Line2D
    import warnings

    # --- Validaciones y datos
    cols_necesarias = ["midx", "midy", "midz", "cluster", attr_col]
    for col in cols_necesarias:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida '{col}'")

    # --- Info de clusters
    clusters_uni = sorted(df["cluster"].unique())
    n_clusters = len(clusters_uni)
    palette = sns.color_palette("tab10", n_colors=n_clusters)
    cluster_color_dict = {k: palette[i] for i, k in enumerate(clusters_uni)}

    # --- Paneles
    fig, axs = plt.subplots(2, 2, figsize=(22, 13))
    plt.subplots_adjust(wspace=0.36, hspace=0.32, left=0.05, right=0.97, top=0.92, bottom=0.08)

    ## ---- 1: Efecto Proporcional (CV por cluster)
    ax0 = axs[0, 0]
    c_stats = []
    for c in clusters_uni:
        datos = df.loc[df["cluster"] == c, attr_col].dropna()
        if len(datos) == 0:
            c_stats.append(dict(mean=np.nan, std=np.nan, cv=np.nan))
        else:
            mean = datos.mean()
            std = datos.std(ddof=1)
            cv = std / mean if mean != 0 else np.nan
            c_stats.append(dict(mean=mean, std=std, cv=cv))
    cvs = [d["cv"] for d in c_stats]
    sc_handles = []
    for idx, c in enumerate(clusters_uni):
        sc = ax0.scatter(idx, cvs[idx], color=cluster_color_dict[c], s=100, zorder=3, label=f'Cluster {c}')
        sc_handles.append(sc)
        ax0.text(idx, cvs[idx]+0.003, f"{cvs[idx]:.2f}", ha='center', fontsize=14, fontweight='bold')
    # Tendencia lineal
    # Manejar el caso de datos NaN en cvs para evitar el error de SVD en np.polyfit
    cvs_for_polyfit = np.array(cvs, dtype=float)
    idx_valid = ~np.isnan(cvs_for_polyfit)
    x_valid = np.arange(n_clusters)[idx_valid]
    y_valid = cvs_for_polyfit[idx_valid]
    # Si hay menos de 2 puntos válidos, NO intentar ajustar la recta (para evitar LinAlgError)
    line = None
    # --- FIX: Importar y usar RankWarning de numpy.polynomial.polyutils (no np.RankWarning)
    try:
        from numpy.polynomial.polyutils import RankWarning as PolyRankWarning
        rank_warning_type = PolyRankWarning
    except ImportError:
        rank_warning_type = Warning  # fallback, in practice should not happen
    if np.sum(idx_valid) >= 2:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=rank_warning_type
                )
                z = np.polyfit(x_valid, y_valid, 1)
                p = np.poly1d(z)
            y_fit = p(np.arange(n_clusters))
            line, = ax0.plot(np.arange(n_clusters), y_fit, color='crimson', linestyle='--', lw=2, label='Tendencia lineal')
        except np.linalg.LinAlgError:
            # No dibujar la tendencia si hay error de SVD
            line = None
    ax0.set_xlabel('Cluster')
    ax0.set_ylabel('Coef. de Variación (std/media)')
    ax0.set_title('Efecto Proporcional (CV) por Cluster', fontsize=13, fontweight='bold')
    ax0.grid(axis='y', alpha=0.28)
    if line is not None:
        ax0.legend([*sc_handles, line], [f'Cluster {ck}' for ck in clusters_uni] + ['Tendencia lineal'])
    else:
        ax0.legend(sc_handles, [f'Cluster {ck}' for ck in clusters_uni])

    ## ---- 2: Probability lognormal plot por cluster
    ax1 = axs[0, 1]
    all_handles = []
    all_labels = []
    for idx, c in enumerate(clusters_uni):
        data = df.loc[df["cluster"] == c, attr_col].dropna().values
        # --- EVITAR ERROR: necesitamos que todos los valores sean >0 para lognormal y al menos 3 datos válidos ---
        data = data[data > 0]
        if len(data) < 3:
            continue
        try:
            shape, loc, scale = lognorm.fit(data, floc=0)
            # Verificar por seguridad que fitting no usó loc/scale inválidos y que el ajuste es válido
            # (redundante si data es >0 y floc=0, pero por robustez)
            if not np.all((data - loc) / scale > 0):
                continue
        except Exception as e:
            # Si el ajuste falla, evitar el cluster y continuar
            continue

        sorted_data = np.sort(data)
        prob = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
        try:
            theo = lognorm.ppf(prob, shape, loc=loc, scale=scale)
            # A veces, si ppf falla por shape/loss numérico, puede devolver nan/inf. Así que verificamos
            if np.any(~np.isfinite(theo)):
                continue
        except Exception:
            continue

        color = cluster_color_dict[c]
        dots, = ax1.plot(sorted_data, prob, marker='o', linestyle='', color=color, label=f'Cluster {c} datos', markersize=4)
        theo_ln, = ax1.plot(theo, prob, linestyle='-', color=color, alpha=0.7, linewidth=2, label=f'Cluster {c} lognorm')
        all_handles.extend([dots, theo_ln])
        all_labels.extend([f'Cluster {c} datos', f'Cluster {c} lognorm'])
    
    ax1.set_xlabel(attr_col)
    ax1.set_ylabel('Probabilidad no excedencia')
    ax1.set_title(f'Probability Lognormal Plot\npor Cluster')
    # ========== CORRECCIÓN PRINCIPAL ==========
    ax1.set_xscale('log')  # Escala logarítmica en eje X
    # ==========================================
    ax1.set_yscale('logit')
    ax1.grid(alpha=0.25, which='both')
    if all_handles:
        ax1.legend(all_handles, all_labels, fontsize=8, loc="best", frameon=True)

    ## ---- 3: Boxplot atributo por cluster
    ax2 = axs[1, 0]
    sns.boxplot(x="cluster", y=attr_col, data=df, palette=palette, ax=ax2)
    mu_handles = []
    for idx, c in enumerate(clusters_uni):
        media_val = c_stats[idx]['mean']
        sc = ax2.scatter(idx, media_val,
                         color='crimson', s=54, marker='D', zorder=11,
                         edgecolor='black', linewidth=0.55, label='Media')
        mu_handles.append(sc)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel(attr_col)
    ax2.set_title('Distribución de atributo por Cluster', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.25)
    if mu_handles:
        ax2.legend([mu_handles[0]], ['Media'], loc='best')

    ## ---- 4: Mapa 3D de clusters
    ax3 = axs[1, 1] = fig.add_subplot(2, 2, 4, projection='3d')
    x = df["midx"].values
    y = df["midy"].values
    z_vals = df["midz"].values
    clusters = df["cluster"].values
    colors = [cluster_color_dict[c] for c in clusters]
    sc3d = ax3.scatter(
        x, y, z_vals, c=colors,
        s=42, alpha=0.76, edgecolors='k', linewidth=0.49
    )
    ax3.set_xlabel("midx", fontsize=12)
    ax3.set_ylabel("midy", fontsize=12)
    ax3.set_zlabel("midz", fontsize=12)
    ax3.set_title(f"Muestra: clusters 3D (K={n_clusters})", fontsize=14, fontweight='bold')
    ax3.view_init(elev=elev, azim=azim)
    # Leyenda
    handles3d = [Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=cluster_color_dict[ck], markeredgecolor='k',
                        markersize=10, label=f'Cluster {ck}')
                 for ck in clusters_uni]
    ax3.legend(handles=handles3d, title="Cluster", loc='best', fontsize=10)

    # --- Título general
    attr_name = attr_col.replace('_', ' ') if attr_col else ''
    fig.suptitle(f"Dashboard 2x2 - Clusters en espacio 3D\n({attr_name})", fontsize=19, fontweight='bold', y=0.993)

    if nombre:
        fig.savefig(nombre, dpi=160, bbox_inches='tight')
        print(f"✅ Dashboard guardado: {nombre}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axs

# Ejemplo de uso:
# Para la variable "cus" como estadísticas:
dashboard_clusters_3d_variable(df_clusters_k7, attr_col="cus", show=True)
#%%

# Estadísticas descriptivas para 'cus' en cada cluster (k=6)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# DataFrame de referencia y columna de análisis
df_ = df_clusters_k6
col = "ca"

# Calcular estadísticas descriptivas por cluster
stats_por_cluster = df_.groupby("cluster")[col].agg([
    ("media", "mean"),
    ("mediana", "median"),
    ("std", lambda x: x.std(ddof=0)),
    ("min", "min"),
    ("max", "max"),
    ("Q1", lambda x: x.quantile(0.25)),
    ("Q3", lambda x: x.quantile(0.75)),
    ("count", "count")
])
stats_por_cluster["cv"] = stats_por_cluster["std"] / stats_por_cluster["media"]

print("Estadísticas descriptivas de 'cus' por cluster:")
print(stats_por_cluster.round(3))

# Histograma de 'cus' por cluster con cuadro de estadísticas en la gráfica
fig, axes = plt.subplots(2, 3, figsize=(21, 9))
axes = axes.flatten()
sns.set_theme(style="whitegrid", font_scale=1.09)

for idx, (cl, ax) in enumerate(zip(sorted(df_["cluster"].unique()), axes)):
    datos = df_.loc[df_["cluster"] == cl, col].dropna()
    sns.histplot(datos, bins=25, ax=ax, color="#2B8EF5", edgecolor="#183155", alpha=0.87)

    stats = stats_por_cluster.loc[cl]
    texto = (
        f"Media:   {stats['media']:.2f}\n"
        f"Mediana: {stats['mediana']:.2f}\n"
        f"Std:     {stats['std']:.2f}\n"
        f"CV:      {stats['cv']:.2f}\n"
        f"Q1:      {stats['Q1']:.2f}\n"
        f"Q3:      {stats['Q3']:.2f}\n"
        f"Min:     {stats['min']:.2f}\n"
        f"Max:     {stats['max']:.2f}\n"
        f"n:       {int(stats['count'])}"
    )
    # Cuadro estético con estadísticas
    ax.text(
        0.98, 0.97, texto, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.33", facecolor="#F5F6FA", edgecolor="#2B8EF5", alpha=0.87),
        transform=ax.transAxes, family="monospace"
    )
    ax.set_title(f"Cluster {cl}", fontsize=15, fontweight='bold', color="#21355A")
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.grid(axis='y', alpha=0.19)

# Quitar ejes vacíos
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(f"Histogramas de '{col}' por cluster (k=6) y estadísticas descriptivas", fontsize=20, fontweight='bold', color="#193151", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
sns.despine()
plt.show()
