#%%
import numpy as np

def analizar_matriz(A, flujos_medidos):
    """
    Recibe una matriz A (numpy array) y la lista de flujos medidos.
    Calcula y muestra p, rango de A_u, t y la factorización QR de A_u.
    Devuelve un diccionario con los mismos resultados.
    """
    n_flujos = A.shape[1]
    F = list(range(1, n_flujos + 1))
    flujos_no_medidos = [i for i in F if i not in flujos_medidos]

    # Extraer columnas correspondientes a medidos y no medidos
    A_x = A[:, [i-1 for i in flujos_medidos]]
    A_u = A[:, [i-1 for i in flujos_no_medidos]]

    m = A.shape[0]
    n = A_x.shape[1]
    p = A_u.shape[1]

    s = np.linalg.matrix_rank(A_u)
    # Lógica de t de acuerdo a la instrucción
    if s == p:
        t = m - p
    else:
        t = m - s

    print(f"p = {p} Flujos no medidos")
    print(f"Rango de A_u: {s}")
    print(f"t = {t} Grados de redundancia")

    # Verificación solicitada: si s = p
    if s == p:
        print("VERIFICACIÓN: El rango de A_u (s) ES IGUAL al número de flujos no medidos (p).")
    else:
        print("ADVERTENCIA: El rango de A_u (s) NO es igual al número de flujos no medidos (p).")

    # Aplicar factorización QR a A_u
    Q, R = np.linalg.qr(A_u)

    # Redondeo adicional para ver claramente los ceros aproximados
    Q_round = np.round(Q, 2)
    R_round = np.round(R, 2)

    print("\nFactorización QR de A_u:")
    print("Matriz Q (redondeada):")
    print(Q_round)
    print("Matriz R (redondeada):")
    print(R_round)

    # Devolver los mismos resultados, pero con Q y R redondeadas para fácil inspección de ceros aproximados
    return {
        'p': p,
        'flujos_no_medidos': flujos_no_medidos,
        'rango_A_u': s,
        't': t,
        'Q': Q_round,
        'R': R_round,
        'A_x': A_x,
        'A_u': A_u
    }

# Ejemplo de uso:
if __name__ == "__main__":
    A = np.array([
        [        1,  -1,  -1,  0,  0,  0,  0,  0, 0, 0],  # Nodo 3
        [        0,   1, 0,  -1,  -1,  0,  0,  0,  0,  0],  # Nodo 4
        [        0,   0,  0, 0,  1,  -1,  0,  0,  1,  0],  # Nodo 5
        [        0,   0,  0,  0, 0,  1,  -1, -1,  0,  0],  # Nodo 6
        [        0,   0,  0,  0,  0, 0, 0,  1,  -1,  -1]   # Nodo 7
    ])
    flujos_medidos = [2,4,5,6]

    dict_resultados = analizar_matriz(A, flujos_medidos)
#%%
dict_resultados["A_u"]
dict_resultados["A_x"]
dict_resultados["Q"][:1,2].T

# print(np.dot(dict_resultados["Q"][:,-1].T, dict_resultados["A_x"]))
print(np.dot(dict_resultados["Q"][:,-1].T, dict_resultados["A_u"]))






    
# %%

    
# %%
