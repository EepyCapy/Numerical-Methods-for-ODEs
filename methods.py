import numpy as np

# Explicit Euler method for 1st order ODEs
def explicitEuler(y0: float, a: float, b: float, n: int, fprime: callable)->np.ndarray:
    dt = (b - a) / n
    t = a
    y = np.zeros((n + 1, 1))
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + dt * fprime(t, y[i])
        t = t + dt
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
    dt = (b - a) / n
    t = a + dt
    y = np.zeros((n + 1, 1))
    y[0] = y0
    for i in range(n):
        y[i + 1] = newton(t, y[i], dt, y[i], fprime)
        t = t + dt
    return y.flatten()

# Runge Kutta Methods for 1st order ODEs
def rungeKutta2(y0: float, a: float, b: float, n: int, fprime: callable)->np.ndarray:
    dt = (b - a) / n
    t = a 
    y = np.zeros((n + 1, 1))
    y[0] = y0
    for i in range(n):
        k1 = fprime(t, y[i])
        k2 = fprime(t + dt, y[i] + dt * k1)
        y[i + 1] = y[i] + 0.5 * dt * (k1 + k2) 
        t = t + dt
    return y.flatten()

def rungeKutta3(y0: float, a: float, b: float, n: int, fprime: callable)->np.ndarray:
    dt = (b - a) / n
    t = a 
    y = np.zeros((n + 1, 1))
    y[0] = y0
    for i in range(n):
        k1 = fprime(t, y[i])
        k2 = fprime(t + dt / 2, y[i] + dt * k1 / 2)
        k3 = fprime(t + dt, y[i] - dt * k1 + 2 * dt * k2)
        y[i + 1] = y[i] + dt * (k1 + 4 * k2 + k3) / 6
        t = t + dt
    return y.flatten()

def rungeKutta4(y0: float, a: float, b: float, n: int, fprime: callable)->np.ndarray:
    dt = (b - a) / n
    t = a 
    y = np.zeros((n + 1, 1))
    y[0] = y0
    for i in range(n):
        k1 = fprime(t, y[i])
        k2 = fprime(t + dt / 2, y[i] + dt * k1 / 2)
        k3 = fprime(t + dt / 2, y[i] + dt * k2 / 2)
        k4 = fprime(t + dt, y[i] + dt * k3)
        y[i + 1] = y[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + dt
    return y.flatten()

# Methods for 2nd order ODEs
# 2nd order ODEs that have the form of Newton's 2nd Law
# x''(t) = a(x, x', t)
# with initial conditions: x(0)=x0, x'(0)=v(0)=v0
# can be turned into a system of 1st order ODEs
# v'(t) = a(x, x', t)
# x'(t) = v(t)
# with the same initial conditions

# The following methods only work for forces/accelerations that depend only on the position and time: a=a(x, t)

# Euler's method for 2nd order ODEs
# Returns both the position (unknown function) and the velocity (1st derivative)
def euler(t0: float, tf: float, x0: float, v0: float, n: int, acceleration: callable)->tuple[np.ndarray, np.ndarray]:
    dt = (tf - t0) / n
    t = t0
    x = np.zeros((n + 1, 1))
    v = np.zeros((n + 1, 1))
    x[0] = x0
    v[0] = v0
    for i in range(n):
        v[i + 1] = v[i] + acceleration(x[i], t) * dt
        x[i + 1] = x[i] + v[i] * dt
        t = t + dt
    return x.flatten(), v.flatten()

# Euler-Cromer method for 2nd order ODEs
# Returns both the position (unknown function) and the velocity (1st derivative)
def euler_cromer(t0: float, tf: float, x0: float, v0: float, n: int, acceleration: callable)->tuple[np.ndarray, np.ndarray]:
    dt = (tf - t0) / n
    t = t0
    x = np.zeros((n + 1, 1))
    v = np.zeros((n + 1, 1))
    x[0] = x0
    v[0] = v0
    for i in range(n):
        v[i + 1] = v[i] + acceleration(x[i], t) * dt
        x[i + 1] = x[i] + v[i + 1] * dt
        t = t + dt
    return x.flatten(), v.flatten()

# Verlet method for 2nd order ODEs
# Returns both the position (unknown function) and the velocity (1st derivative)
def verlet(t0: float, tf: float, x0: float, v0: float, n: int, acceleration: callable)->tuple[np.ndarray, np.ndarray]:
    dt = (tf - t0) / n
    t = t0
    x = np.zeros((n + 1, 1))
    v = np.zeros((n + 1, 1))
    x[0] = x0
    v[0] = v0
    xOneStepBack = x0 - v0 * dt + 0.5 * acceleration(x0, t) * dt * dt
    x[1] = 2 * x0 - xOneStepBack + acceleration(x0, t) * dt * dt
    t = t + dt
    for i in range(1, n):
        x[i + 1] = 2 * x[i] - x[i - 1] + acceleration(x[i], t) * dt * dt
        v[i] = 0.5 * (x[i + 1] - x[i - 1]) / dt
        t = t + dt
    v[n] = (x[n] - x[n - 1]) / dt
    return x.flatten(), v.flatten()

# Runge-Kutta 4 method for 2nd order ODEs
# Returns both the position (unknown function) and the velocity (1st derivative)
def rungeKuttaPT(t0: float, tf: float, x0: float, v0: float, n: int, acceleration: callable)->tuple[np.ndarray, np.ndarray]:
    dt = (tf - t0) / n
    t = t0
    x = np.zeros((n + 1, 1))
    v = np.zeros((n + 1, 1))
    x[0] = x0
    v[0] = v0
    for i in range(n):
        k11 = v[i]
        k21 = acceleration(x[i], t)
        k12 = v[i] + dt * k21 / 2
        k22 = acceleration(x[i] + dt * k11 / 2, t + dt / 2)
        k13 = v[i] + dt * k22 / 2
        k23 = acceleration(x[i] + dt * k12 / 2, t + dt / 2)
        k14 = v[i] + dt * k23
        k24 = acceleration(x[i] + dt * k13, t + dt)
        x[i + 1] = x[i] + dt * (k11 + 2 * k12 + 2 * k13 + k14) / 6
        v[i + 1] = v[i] + dt * (k21 + 2 * k22 + 2 * k23 + k24) / 6
        t = t + dt
    return x.flatten(), v.flatten()

# Runge-Kutta 4 method for 2nd order ODEs with dependence on position, velocity and time
# Returns both the position (unknown function) and the velocity (1st derivative)
def rungeKutta(t0: float, tf: float, x0: float, v0: float, n: int, acceleration: callable)->tuple[np.ndarray, np.ndarray]:
    dt = (tf - t0) / n
    t = t0
    x = np.zeros((n + 1, 1))
    v = np.zeros((n + 1, 1))
    x[0] = x0
    v[0] = v0
    for i in range(n):
        k11 = v[i]
        k21 = acceleration(x[i], v[i], t)
        k12 = v[i] + dt * k21 / 2
        k22 = acceleration(x[i] + dt * k11 / 2, v[i] + dt * k21 / 2, t + dt / 2)
        k13 = v[i] + dt * k22 / 2
        k23 = acceleration(x[i] + dt * k12 / 2, v[i] + dt * k22 / 2, t + dt / 2)
        k14 = v[i] + dt * k23
        k24 = acceleration(x[i] + dt * k13, v[i] + dt * k23, t + dt)
        x[i + 1] = x[i] + dt * (k11 + 2 * k12 + 2 * k13 + k14) / 6
        v[i + 1] = v[i] + dt * (k21 + 2 * k22 + 2 * k23 + k24) / 6
        t = t + dt
    return x.flatten(), v.flatten()