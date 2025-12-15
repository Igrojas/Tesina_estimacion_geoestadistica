#%%
import numpy as np
import pyvista as pv
from pykrige.ok3d import OrdinaryKriging3D

# --- PASO 1: Generar Datos Sintéticos (Simulando Taladros) ---
# En un caso real, esto vendría de tu CSV de sondajes (compositos)
data = np.array([
    # x, y, z, ley (ej. %Cu)
    [10, 10, 0, 0.1],
    [90, 10, 0, 0.2],
    [10, 90, 0, 0.1],
    [90, 90, 0, 0.3],
    [50, 50, 20, 2.5], # Núcleo de alta ley
    [50, 50, -20, 2.2],
    [30, 30, 10, 1.5],
    [70, 70, 10, 1.8],
    [20, 80, -10, 0.6],
    [80, 20, -10, 0.5],
    [25, 25, 15, 1.7],
    [75, 75, 15, 1.9],
    [40, 60, 5, 0.95],
    [65, 35, 5, 0.88],
    [50, 80, -15, 1.0],
    [80, 50, -15, 0.7],
    [20, 20, 20, 0.3],
    [80, 80, -20, 0.4],
    [60, 40, 0, 0.9],
    [40, 60, -5, 1.2],
    [25, 75, 5, 0.85],
    [75, 25, -5, 0.95],
    [60, 60, 25, 2.8],
    [40, 40, -25, 2.1],
    [50, 50, 0, 2.0],    # Más puntos en el núcleo
    [55, 55, 10, 2.4],
    [45, 45, -10, 2.3],
    [50, 50, 15, 2.6]
])

# Separar coordenadas y valores (asegurar tipo float64)
x_coords = data[:, 0].astype(np.float64)
y_coords = data[:, 1].astype(np.float64)
z_coords = data[:, 2].astype(np.float64)
values = data[:, 3].astype(np.float64)

# --- PASO 2: Estimación con Kriging Ordinario 3D (PyKrige) ---
# Definir la grilla del modelo de bloques (asegurar tipo float64)
grid_x = np.arange(0, 100, 5, dtype=np.float64) # Bloques de 5m
grid_y = np.arange(0, 100, 5, dtype=np.float64)
grid_z = np.arange(-30, 30, 5, dtype=np.float64)

# Configurar Kriging
ok3d = OrdinaryKriging3D(
    x_coords, y_coords, z_coords, values,
    variogram_model='exponential', # O 'spherical', 'gaussian', 'exponential'
    verbose=False,
    enable_plotting=False
)

# Ejecutar estimación
# k3d: array 3D con las estimaciones (leyes)
# ss3d: array 3D con la varianza de kriging
k3d, ss3d = ok3d.execute('grid', grid_x, grid_y, grid_z)

# --- PASO 3: Visualización Profesional 3D (PyVista) ---

# Crear una grilla estructurada en PyVista compatible con los datos
# Nota: PyKrige devuelve datos en orden (z, y, x), a veces requiere transponer dependiendo del setup
grid = pv.RectilinearGrid(grid_x, grid_y, grid_z)

# Aplanar el array de estimación para asignarlo a la grilla (orden Fortran 'F' suele alinear mejor z,y,x)
grid["Ley_Cu"] = k3d.flatten(order='F') 

# Filtrar para ver solo el "mineral" (ej. ley de corte > 0.5%)
thresholded_grid = grid.threshold(0.9)

# Configurar el plotter
plotter = pv.Plotter()
plotter.add_mesh(thresholded_grid, cmap="jet", show_edges=True, opacity=1, label="Cuerpo Mineralizado")
plotter.add_mesh(pv.PolyData(data[:, :3]), color="red", point_size=10, render_points_as_spheres=True, label="Sondajes")
plotter.add_axes()
plotter.add_legend()
plotter.show_grid()

print("Abriendo ventana de visualización 3D...")
plotter.show()


# %%
