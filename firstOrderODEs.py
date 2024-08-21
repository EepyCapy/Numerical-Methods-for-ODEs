import numpy as np
# ODE 1
# y' = 2 - t + y, y(0)=1
def f1(t: float, y: float)->float:
    return 2 - t + y

def solution1(t: float | np.ndarray)->float | np.ndarray:
    return 2 * np.exp(t) + t - 1

# ODE 2
# y' = 4 * t - 2 * y / t, y(1)=2
def f2(t: float, y: float)->float:
    return 4 * t - 2 * y / t

def solution2(t: float | np.ndarray)->float | np.ndarray:
    return t * t + 1 / (t * t)

# ODE 3
# y' = 0.5 * (3 * t * t + 4 * t + 2) / (y - 1), y(-1.95)=0.461367
def f3(t: float, y: float)->float:
    return 0.5 * (3 * t * t + 4 * t + 2) / (y - 1)

def solution3(t: float | np.ndarray)->float | np.ndarray:
    return 1 - np.sqrt(t ** 3 + 2 * t * t + 2 * t + 4)

# ODE 4
# y' = (4 * t - t ** 3) / (4 + y ** 3), y(0)=1
def f4(t: float, y: float)->float:
    return (4 * t - t ** 3) / (4 + y ** 3)
