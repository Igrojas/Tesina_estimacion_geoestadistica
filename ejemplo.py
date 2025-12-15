#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# Cargar datos
df = pd.read_csv('data/raw/com_p_plt_entry 1.csv', sep=',')
attr = df["cus"]
attr_name = "cut"
datos = attr[attr > 0]

# Ordenar los datos
datos_sorted = np.sort(datos)
n = len(datos_sorted)

# Calcular percentiles y z-scores
percentiles = (np.arange(1, n + 1) - 0.5) / n * 100
z_scores = norm.ppf(percentiles / 100)

# Crear figura con fondo blanco y grilla
fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
ax.set_facecolor('white')

# Graficar puntos
ax.plot(datos_sorted, z_scores, 'ko', markersize=3, alpha=0.6)

# Escala logarítmica en X
ax.set_xscale('log')

# Configurar eje Y (probabilidad normal)
percentile_labels = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 
                     60, 70, 80, 90, 95, 98, 99, 99.5, 99.8, 99.9, 99.95, 99.99]
z_ticks = norm.ppf(np.array(percentile_labels) / 100)
ax.set_yticks(z_ticks)
ax.set_yticklabels([f'{p}' for p in percentile_labels], fontsize=9)

# Límites
ax.set_ylim([norm.ppf(0.01/100), norm.ppf(99.99/100)])

# Títulos
ax.set_title('Gráfico de probabilidad lognormal', fontsize=14, fontweight='bold')
ax.set_xlabel(attr_name, fontsize=11)
ax.set_ylabel('Probs', fontsize=11)

# Grid detallada como en la imagen
ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5, color='gray')
ax.grid(True, which='minor', linestyle='-', linewidth=0.4, alpha=0.3, color='lightgray')

plt.tight_layout()
plt.show()
# %%
