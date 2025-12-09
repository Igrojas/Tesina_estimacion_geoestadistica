
#%%

import numpy as np

# Definir la matriz A con 4 filas y 6 columnas
A = np.array([
    [ 1, -1, -1,  0,  0,  0],  # Ecuación 1-7a: Balance divisor
    [ 0,  1,  0, -1,  0,  0],  # Ecuación 1-7b: Balance válvula
    [ 0,  0,  1,  0, -1,  0],  # Ecuación 1-7c: Balance intercambiador
    [ 0,  0,  0,  1,  1, -1]   # Ecuación 1-7d: Balance mezclador
])

# Definir el conjunto general de flujos
F = list(range(1, 7))

# Flujos medidos y no medidos
flujos_medidos = [1, 6]           # M
flujos_no_medidos = [2, 3, 4, 5]  # U

# Extraer columnas correspondientes a medidos y no medidos
A_x = A[:, [i-1 for i in flujos_medidos]]
A_u = A[:, [i-1 for i in flujos_no_medidos]]

m = A.shape[0]
n = A_x.shape[1]
p = A_u.shape[1]

s = np.linalg.matrix_rank(A_u)
t = m - s

print(f"p = {p} Flujos no medidos")
print(f"Rango de A_u: {s}")
print(f"t = {t} Grados de redundancia")

# Aplicar factorización QR a A_u
Q, R = np.linalg.qr(A_u)

print("\nFactorización QR de A_u:")
print("Matriz Q:")
print(Q)
print("Matriz R:")
print(np.round(R, 2))







    
# %%
