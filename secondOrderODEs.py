import numpy as np
from scipy.special import airy, gamma

# ODE 1 - Simple harmonic Oscillator
# x'' + omega**2 * x = 0, x(0)=x0, x'(0)=v0
def f1(x: float, t: float)->float:
    omega = 1
    return -omega * omega * x    

# Solution 1 fot x0=1, v0=0: x(t)=cos(omega*t)
def solution1(t: float | np.ndarray)->float | np.ndarray:
    omega = 1
    return np.cos(omega * t)

# ODE 2
# x'' - omega**2 * x = 0, x(0)=x0, x'(0)=v0
def f2(x: float, t: float)->float:
    omega = 1
    return omega * omega * x 

# Solution 2 fot x0=0, v0=1: x(t)=sinh(omega*t)
def solution2(t: float | np.ndarray)->float | np.ndarray:
    omega = 1
    return np.sinh(omega * t)

# ODE 3
# x'' -x = e**(-0.1 * t), x(0)=1, x'(0)=0
def f3(x: float, t: float)->float:
    return -x + np.e ** (-0.1 * t)

# Solution 3: x(t)=cos(t)+0.990099*e**(-0.1*t)
def solution3(t: float | np.ndarray)->float | np.ndarray:
    return 0.0990099 * np.sin(t) + 0.00990099 * np.cos(t) + 0.990099 * np.e ** (-0.1 * t)

# ODE 4 - Airy Equation
#  x'' - t * x = 0, x(0)=1, x'(0)=0
def f4(x: float, t: float)->float:
    return t * x

# # Solution 4: x(t)=0.5*3**(1/6)*Gamma(2/3)*(sqrt(3) Ai(t) + Bi(t)) 
def solution4(t: float | np.ndarray)->float | np.ndarray:
    ai, bi = airy(t)[0], airy(t)[2]
    return 0.5 * 3**(1/6) * gamma(2/3) * (np.sqrt(3) * ai + bi) 
