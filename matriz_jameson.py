
#%%
import numpy as np

# ==============================================================================
# MATRIZ DE INCIDENCIA DE NODOS
# ==============================================================================
# Filas: Nodos 3, 4, 5, 6, 7
# Columnas: Flujos 4, 5, 6, 7, 8, 9, 10, 11, 12, 13

# Etiquetas de flujos (para referencia)
flujos = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Etiquetas de nodos (para referencia)
nodos = [3, 4, 5, 6, 7]

# Matriz A de balances (5 nodos × 10 flujos)
A = np.array([
    # Flujo:  4   5   6   7   8   9  10  11  12  13
    [        1,  0,  0,  0,  0,  0,  0,  0, -1, -1],  # Nodo 3
    [        0, -1, -1,  0,  0,  0,  0,  0,  0,  1],  # Nodo 4
    [        0,  0,  1, -1,  0,  0,  1,  0,  0,  0],  # Nodo 5
    [        0,  0,  0,  1, -1,  0,  0, -1,  0,  0],  # Nodo 6
    [        0,  0,  0,  0,  1, -1, -1,  0,  0,  0]   # Nodo 7
])

# Flujos a eliminar: 5, 9, 11, 12
flujos_quitar = [4, 5, 9, 11, 12]

# Obtener los índices de columnas a quitar
indices_quitar = [flujos.index(f) for f in flujos_quitar]

# Construir la matriz A_u eliminando esos flujos/columnas
A_u = np.delete(A, indices_quitar, axis=1)
flujos_u = [f for i, f in enumerate(flujos) if i not in indices_quitar]

print("="*80)
print("MATRIZ A_u (Sin flujos 5, 9, 11, 12)")
print("="*80)
print(f"\nFlujos restantes (columnas de A_u): {flujos_u}")
print(f"Matriz A_u ({A_u.shape[0]} nodos × {A_u.shape[1]} flujos):")
print(A_u)
print(f"\nRango de la matriz A_u: {np.linalg.matrix_rank(A_u)}")



    
# %%
