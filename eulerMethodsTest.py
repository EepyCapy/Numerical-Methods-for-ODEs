from methods import explicitEuler, implicitEuler
from functions import *
import numpy as np
import matplotlib.pyplot as plt

n: int = 50
# ODE 1
leftBound: float = 0.
rightBound: float = 1.
y0: float = 1.
t1: np.ndarray = np.linspace(leftBound, rightBound, n + 1)
y1_exact: np.ndarray = solution1(t1)
y1_exEuler: np.ndarray = explicitEuler(y0, leftBound, rightBound, n, f1)
y1_imEuler: np.ndarray = implicitEuler(y0, leftBound, rightBound, n, f1)
# ODE 2
leftBound: float = 1.
rightBound: float = 3.
y0: float = 2.
t2: np.ndarray = np.linspace(leftBound, rightBound, n + 1)
y2_exact: np.ndarray = solution2(t2)
y2_exEuler: np.ndarray = explicitEuler(y0, leftBound, rightBound, n, f2)
y2_imEuler: np.ndarray = implicitEuler(y0, leftBound, rightBound, n, f2)
# ODE 3
leftBound: float = -1.95
rightBound: float = 1.
y0: float = 0.461367
t3: np.ndarray = np.linspace(leftBound, rightBound, n + 1)
y3_exact: np.ndarray = solution3(t3)
y3_exEuler: np.ndarray = explicitEuler(y0, leftBound, rightBound, n, f3)
y3_imEuler: np.ndarray = implicitEuler(y0, leftBound, rightBound, n, f3)
# ODE 4
leftBound: float = 0.
rightBound: float = 3. #2.85
y0: float = 1.
t4: np.ndarray = np.linspace(leftBound, rightBound, n + 1)
y4_exact: np.ndarray = np.array([solution4(t) for t in t4]).flatten()
y4_exEuler: np.ndarray = explicitEuler(y0, leftBound, rightBound, n, f4)
y4_imEuler: np.ndarray = implicitEuler(y0, leftBound, rightBound, n, f4)

# Plotting
fig, ax = plt.subplots(2, 2)
# ODE 1 graph and errors
#ax[0][0].plot(t1, y1_exact, color='blue')
#ax[0][0].plot(t1, y1_exEuler, color='red')
#ax[0][0].plot(t1, y1_imEuler, color='green')
#ax[1][0].plot(t1, y1_exact - y1_exEuler, color='red')
#ax[1][0].plot(t1, y1_exact - y1_imEuler, color='green')
# ODE 2 graph and errors
#ax[0][1].plot(t1, y2_exact, color='blue')
#ax[0][1].plot(t1, y2_exEuler, color='red')
#ax[0][1].plot(t1, y2_imEuler, color='green')
#ax[1][1].plot(t1, y2_exact - y2_exEuler, color='red')
#ax[1][1].plot(t1, y2_exact - y2_imEuler, color='green')
# ODE 3 graph and errors
ax[0][0].plot(t3, y3_exact, color='blue')
ax[0][0].plot(t3, y3_exEuler, color='red')
ax[0][0].plot(t3, y3_imEuler, color='green')
ax[1][0].plot(t3, y3_exact - y3_exEuler, color='red')
ax[1][0].plot(t3, y3_exact - y3_imEuler, color='green')
# ODE 4 graph and errors
ax[0][1].plot(t4, y4_exact, color='blue')
ax[0][1].plot(t4, y4_exEuler, color='red')
ax[0][1].plot(t4, y4_imEuler, color='green')
ax[1][1].plot(t4, y4_exact - y4_exEuler, color='red')
ax[1][1].plot(t4, y4_exact - y4_imEuler, color='green')

plt.show()
