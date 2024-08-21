from methods import euler, euler_cromer, verlet, rungeKuttaPT
from secondOrderODEs import *
import numpy as np
import matplotlib.pyplot as plt

n: int = 500
# ODE 1
t0: float = 0.
tf: float = 2 * np.pi
x0: float = 1.
v0: float = 0.
t1: np.ndarray = np.linspace(t0, tf, n + 1)
x1_exact: np.ndarray = solution1(t1)
x1_euler: np.ndarray = euler(t0, tf, x0, v0, n, f1)[0]
x1_cromer: np.ndarray = euler_cromer(t0, tf, x0, v0, n, f1)[0]
x1_verlet: np.ndarray = verlet(t0, tf, x0, v0, n, f1)[0]
# ODE 2
t0: float = 0.
tf: float = 10.
x0: float = 0.
v0: float = 1.
t2: np.ndarray = np.linspace(t0, tf, n + 1)
x2_exact: np.ndarray = solution2(t2)
x2_euler: np.ndarray = euler(t0, tf, x0, v0, n, f2)[0]
x2_cromer: np.ndarray = euler_cromer(t0, tf, x0, v0, n, f2)[0]
x2_verlet: np.ndarray = verlet(t0, tf, x0, v0, n, f2)[0]
x2_rk: np.ndarray =  rungeKuttaPT(t0, tf, x0, v0, n, f2)[0]
# ODE 3
t0: float = 0.
tf: float = 10.
x0: float = 1.
v0: float = 0.
t3: np.ndarray = np.linspace(t0, tf, n + 1)
x3_exact: np.ndarray = solution3(t3)
x3_euler: np.ndarray = euler(t0, tf, x0, v0, n, f3)[0]
x3_cromer: np.ndarray = euler_cromer(t0, tf, x0, v0, n, f3)[0]
x3_verlet: np.ndarray = verlet(t0, tf, x0, v0, n, f3)[0]
x3_rk: np.ndarray =  rungeKuttaPT(t0, tf, x0, v0, n, f3)[0]
# ODE 4
t0: float = 0.
tf: float = 2.
x0: float = 1.
v0: float = 0.
t4: np.ndarray = np.linspace(t0, tf, n + 1)
x4_exact: np.ndarray = solution4(t4)
x4_euler: np.ndarray = euler(t0, tf, x0, v0, n, f4)[0]
x4_cromer: np.ndarray = euler_cromer(t0, tf, x0, v0, n, f4)[0]
x4_verlet: np.ndarray = verlet(t0, tf, x0, v0, n, f4)[0]
x4_rk: np.ndarray =  rungeKuttaPT(t0, tf, x0, v0, n, f4)[0]

# Plotting
fig, ax = plt.subplots(2,2)
# ODE 1
#ax[0][0].plot(t1, x1_exact, color='blue')
#ax[0][0].plot(t1, x1_euler, color='red')
#ax[0][0].plot(t1, x1_cromer, color='green')
#ax[0][0].plot(t1, x1_verlet, color='purple')
#ax[1][0].plot(t1, x1_exact - x1_euler, color='red')
#ax[1][0].plot(t1, x1_exact - x1_cromer, color='green')
#ax[1][0].plot(t1, x1_exact - x1_verlet, color='purple')
# ODE 2
#ax[0][1].plot(t2, x2_exact, color='blue')
#ax[0][1].plot(t2, x2_euler, color='red')
#ax[0][1].plot(t2, x2_cromer, color='green')
#ax[0][1].plot(t2, x2_verlet, color='purple')
#ax[0][1].plot(t2, x2_rk, color='black', ls='--')
#ax[1][1].plot(t2, x2_exact - x2_euler, color='red')
#ax[1][1].plot(t2, x2_exact - x2_cromer, color='green')
#ax[1][1].plot(t2, x2_exact - x2_verlet, color='purple')
#ax[1][1].plot(t2, x2_exact - x2_rk, color='black', ls='--')
# ODE 3
ax[0][0].plot(t3, x3_exact, color='blue')
ax[0][0].plot(t3, x3_euler, color='red')
ax[0][0].plot(t3, x3_cromer, color='green')
ax[0][0].plot(t3, x3_verlet, color='purple')
ax[0][0].plot(t3, x3_rk, color='black', ls='--')
ax[1][0].plot(t3, x3_exact - x3_euler, color='red')
ax[1][0].plot(t3, x3_exact - x3_cromer, color='green')
ax[1][0].plot(t3, x3_exact - x3_verlet, color='purple')
ax[1][0].plot(t3, x3_exact - x3_rk, color='black', ls='--')
# ODE 4
ax[0][1].plot(t4, x4_exact, color='blue')
ax[0][1].plot(t4, x4_euler, color='red')
ax[0][1].plot(t4, x4_cromer, color='green')
ax[0][1].plot(t4, x4_verlet, color='purple')
ax[0][1].plot(t4, x4_rk, color='black', ls='--')
ax[1][1].plot(t4, x4_exact - x4_euler, color='red')
ax[1][1].plot(t4, x4_exact - x4_cromer, color='green')
ax[1][1].plot(t4, x4_exact - x4_verlet, color='purple')
ax[1][1].plot(t4, x4_exact - x4_rk, color='black', ls='--')

plt.show()
