import numpy as np
import matplotlib.pyplot as plt

# Explicit Euler method for 1st order ODEs
def explicitEuler(y0: float, a: float, b: float, n: int, fprime: callable)->np.ndarray:
    h = (b - a) / n
    t = a
    y = np.zeros((n + 1, 1))
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * fprime(t, y[i])
        t = t + h
    return y.flatten()

# Newton's method for numerically solving non-linear algebraic equations of the form: 
# c + h * f(t, x) - x = 0
def newton(t: float, c: float, h: float, x0: float, f: callable, error: float=10e-8)->float:
    step: float = 10e-5 # step for numerical differentiation
    f_value: float = c + h * f(t, x0) - x0
    f_prime: float = h * (f(t, x0 + step) - f(t, x0)) / step - 1
    x: float = x0 - f_value / f_prime
    while np.abs(x - x0) >= error:
        x0 = x
        f_value = c + h * f(t, x0) - x0
        f_prime = h * (f(t, x0 + step) - f(t, x0)) / step - 1
        x = x0 - f_value / f_prime
    return x

# Implicit Euler method for 1st order ODEs
def implicitEuler(y0: float, a: float, b: float, n: int, fprime: callable)->np.ndarray:
    h = (b - a) / n
    t = a + h
    y = np.zeros((n + 1, 1))
    y[0] = y0
    for i in range(n):
        y[i + 1] = newton(t, y[i], h, y[i], fprime)
        t = t + h
    return y.flatten()

def rungeKutta2(y0: float, a: float, b: float, n: int, fprime: callable)->np.ndarray:
    h = (b - a) / n
    t = a 
    y = np.zeros((n + 1, 1))
    y[0] = y0
    for i in range(n):
        k = y[i] + h * fprime(t, y[i])
        y[i + 1] = y[i] + 0.5 * h * (fprime(t, y[i]) + fprime(t + h, k)) 
        t = t + h
    return y.flatten()

# Testing
def func(t: float, y: float)->float:
    return 2 - t + y

N: int = 50
A: float = 0.
B: float = 1.
y_init: float = 1.
t: np.ndarray = np.linspace(A, B, N + 1)
yExact: np.ndarray = 2 * np.exp(t) + t - 1
yAppr1: np.ndarray = explicitEuler(y_init, A, B, N, func)
yAppr2: np.ndarray = implicitEuler(y_init, A, B, N, func)
yAppr3: np.ndarray = rungeKutta2(y_init, A, B, N, func)

#plt.plot(t, yExact, color='blue')
plt.plot(t, yExact - yAppr1, color='red')
plt.plot(t, yExact - yAppr2, color='green')
plt.plot(t, yExact - yAppr3, color='purple')
plt.show()
