import numpy as np

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
        k1 = fprime(t, y[i])
        k2 = fprime(t + h, y[i] + h * k1)
        y[i + 1] = y[i] + 0.5 * h * (k1 + k2) 
        t = t + h
    return y.flatten()

def rungeKutta3(y0: float, a: float, b: float, n: int, fprime: callable)->np.ndarray:
    h = (b - a) / n
    t = a 
    y = np.zeros((n + 1, 1))
    y[0] = y0
    for i in range(n):
        k1 = fprime(t, y[i])
        k2 = fprime(t + h / 2, y[i] + h * k1 / 2)
        k3 = fprime(t + h, y[i] - h * k1 + 2 * h * k2)
        y[i + 1] = y[i] + h * (k1 + 4 * k2 + k3) / 6
        t = t + h
    return y.flatten()
