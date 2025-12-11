"""
SOLUCIÓN ANALÍTICA POR SEPARACIÓN DE VARIABLES:

1. dy/dt = y(1-y)
2. Separar: dy/[y(1-y)] = dt
3. Integrar ambos lados: ∫dy/[y(1-y)] = ∫dt
4. Usar fracciones parciales: 1/[y(1-y)] = 1/y + 1/(1-y)
5. ∫(1/y + 1/(1-y))dy = ∫dt
6. ln|y| - ln|1-y| = t + C
7. ln|y/(1-y)| = t + C
8. y/(1-y) = A·e^t  (donde A = e^C)
9. Despejar y: y = A·e^t/(1 + A·e^t)
10. Con y(0)=0.2: 0.2 = A/(1+A) → A = 0.25
11. Solución: y(t) = 0.25·e^t/(1 + 0.25·e^t)
   = (0.2·e^t)/(1 - 0.2 + 0.2·e^t)
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
y0 = 0.2            # condición inicial
t0, tf = 0.0, 1.0   # intervalo
h = 0.2             # paso
n_steps = int((tf - t0) / h)  # =5
t = np.linspace(t0, tf, n_steps + 1)

# Función f(t, y) = y(1 - y)
def f(t, y):
    return y * (1 - y)

# Método de Euler explícito
y_euler = np.zeros(n_steps + 1)
y_euler[0] = y0
for k in range(n_steps):
    y_euler[k + 1] = y_euler[k] + h * f(t[k], y_euler[k])

# Solución analítica
def y_exact(t):
    return (y0 * np.exp(t)) / (1 - y0 + y0 * np.exp(t))


y_a = y_exact(t)

# Imprimir tabla de resultados
print("k\tt_k\t\ty_Euler\t\ty_Exact\t\tError abs")
for k, tk in enumerate(t):
    err = abs(y_euler[k] - y_a[k])
    print(f"{k}\t{tk:.1f}\t\t{y_euler[k]:.6f}\t{y_a[k]:.6f}\t{err:.6e}")

# Graficar comparación
plt.figure(figsize=(8,5))
plt.plot(t, y_a, label="Solución exacta", linewidth=2)
plt.plot(t, y_euler, "o-", label=f"Euler (h={h})")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Euler vs Solución exacta (Ecuación separable dy/dt = y(1-y))")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("comparacion_euler.png")
print("\nGráfico guardado como 'comparacion_euler.png'")
plt.show()
