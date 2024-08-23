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

# Solution 4: x(t)=0.5*3**(1/6)*Gamma(2/3)*(sqrt(3) Ai(t) + Bi(t)) 
def solution4(t: float | np.ndarray)->float | np.ndarray:
    ai, bi = airy(t)[0], airy(t)[2]
    return 0.5 * 3**(1/6) * gamma(2/3) * (np.sqrt(3) * ai + bi) 

# ODE 5 - Mass thrown upwards with no air resistance
# z'' = g/m, z(0)=z0, z'(0)=v0
def f5(z: float, t: float)->float:
    g = 9.81
    return -g 

# Solution 5: z(t)=z0+v0*t-(g/m)*t**2
def solution5(t: float | np.ndarray, z0: float, v0: float, g: float)->float | np.ndarray:
    return z0 + v0 * t - 0.5 * g * t ** 2

# ODE 6 - Another Airy Equation
#  x'' + t * x = 0, x(-2)=Ai(2), x'(-2)=-Ai'(2)
def f6(x: float, t: float)->float:
    return -t * x

# Solution 6: x(t)=Ai(-t)
def solution6(t: float | np.ndarray)->float | np.ndarray:
    return airy(-t)[0]

# ODE 7 - Dampened Harmonic Oscillator
# x'' + 2*b*x' + omega**2*x = 0, x(0)=1, x'(0)=0
def f7(x: float, v: float, t: float)->float:
    m = 0.5
    s = 13.
    r = 0.5
    omega2 = s / m
    b = 0.5 * r / m
    return -2 * b * v - omega2 * x

# Solution 7: x(t)=e**(-b*t)*((b/omega')*sin(omega'*t)+cos(omega'*t))
def solution7(t: float | np.ndarray)->float | np.ndarray:
    m = 0.5
    s = 13.
    r = 0.5
    omega2 = s / m
    b = 0.5 * r / m
    freq = np.sqrt(omega2 - b * b) #omega'
    return np.e**(-b * t) * (b / freq * np.sin(freq*t) + np.cos(freq*t))

# ODE 8 - Forced Oscillator
# x'' + 2*b*x' + omega**2*x = (F0/m)*cos(w*t), x(0)=1, x'(0)=0
def f8(x: float, v: float, t: float)->float:
    m = 0.5
    s = 13.
    r = 0.5
    omega2 = s / m
    b = 0.5 * r / m
    F0 = 1. 
    w = 4.
    return -2 * b * v - omega2 * x + F0 / m * np.cos(w * t)

# Solution 8
def solution8(t: float | np.ndarray)->float | np.ndarray:
    a = np.cos(5.07445*t)
    b = 0.827586*np.exp(-0.5*t) - 0.0701585*np.sin(1.07445*t) - 0.00119296*np.sin(9.07445*t) + 0.150763*np.cos(1.07445*t) + 0.0216508*np.cos(9.07445*t)
    d = np.sin(5.07445*t)
    e = 0.0271815*np.exp(-0.5*t) + 0.150763*np.sin(1.07445*t) + 0.0216508*np.sin(9.07445*t) + 0.0701585*np.cos(1.07445*t) + 0.00119296*np.cos(9.07445*t)
    return a * b + d * e

# ODE 9 - Mass thrown upwards with air resistence
# z'' = -g -k*z'/m, z(0)=0, z'(0)=v0
def f9(x: float, v: float, t: float)->float:
    k = 1.
    m = 5.
    g = 9.81
    return -g - k * v / m

# Solution 9
def solution9(t: float | np.ndarray)->float | np.ndarray:
    k = 1.
    m = 5.
    g = 9.81
    v0= 200.
    b = k / m
    c = m * g / k
    a = c + v0
    return a * (1 - np.exp(-b * t)) / b - c * t