
#%%

import numpy as np

# Definir la matriz A con 4 filas y 6 columnas
A = np.array([
    [ 1, -1, -1,  0,  0,  0],  # Ecuación 1-7a: Balance divisor
    [ 0,  1,  0, -1,  0,  0],  # Ecuación 1-7b: Balance válvula
    [ 0,  0,  1,  0, -1,  0],  # Ecuación 1-7c: Balance intercambiador
    [ 0,  0,  0,  1,  1, -1]   # Ecuación 1-7d: Balance mezclador
])

print("---- Caso 1: Los 6 flujos medidos ----")
rango_total = np.linalg.matrix_rank(A)
print(f"Rango de la matriz A (todos los flujos): {rango_total}")

# Caso 2: Solo flujo 1 y 2 medido, los no medidos son 3,4,5,6
indices_no_medidos_2 = [2, 3, 4, 5]  # Python index: columnas 3 a 6 (de flujo 3 a 6)
A_u2 = A[:, indices_no_medidos_2]
print("\n---- Caso 2: Solo Flujo 1 y 2 medido ----")
print(f"Matriz A_u (no medidos):\n{A_u2}")
rango_A_u2 = np.linalg.matrix_rank(A_u2)
print(f"Rango de la matriz A_u (flujos 3,4,5,6 no medidos): {rango_A_u2}")

# Caso 3: Solo flujo 1 y 6 medido, los no medidos son 2,3,4,5
indices_no_medidos_3 = [1, 2, 3, 4]  # Python index: columnas 2 a 5 (de flujo 2 a 5)
A_u3 = A[:, indices_no_medidos_3]
print("\n---- Caso 3: Solo Flujo 1 y 6 medido ----")
print(f"Matriz A_u (no medidos):\n{A_u3}")
rango_A_u3 = np.linalg.matrix_rank(A_u3)
print(f"Rango de la matriz A_u (flujos 2,3,4,5 no medidos): {rango_A_u3}")



    
# %%
