#%%
import pandas as pd
import matplotlib.pyplot as plt

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
df_sample = df.sample(frac=0.1)
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
