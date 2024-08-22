import numpy as np
import matplotlib.pyplot as plt
from methods import verlet, rungeKuttaPT
from secondOrderODEs import *
from scipy.special import airy

fig, ax = plt.subplots(2,2)
n: list[int] = [10, 100, 1000, 10000, 100000]
colors = ['y', 'r', 'g', 'c', 'm']
# ODE 5 - Mass thrown upwards with no air resistance
""" g: float = 9.81
z0: float = 0.
v0: float = 10.
t0: float = 0.
tf: float = 2 * v0 / g
t: float = np.linspace(t0, tf, n[-1]+1)
z_exact = solution5(t, z0, v0, g)
ax[0][0].plot(t, z_exact, color='b')
ax[0][1].plot(t, z_exact, color='b')
for i in range(len(n)):
    t = np.linspace(t0, tf, n[i]+1)
    z_exact = solution5(t, z0, v0, g)
    z_verlet = verlet(t0, tf, z0, v0, n[i], f5)[0]
    z_rk = rungeKuttaPT(t0, tf, z0, v0, n[i], f5)[0]
    ax[0][0].plot(t, z_verlet, color=colors[i])
    ax[1][0].plot(t, z_exact - z_verlet, color=colors[i])
    ax[0][1].plot(t, z_rk, color=colors[i])
    ax[1][1].plot(t, z_exact - z_rk, color=colors[i]) """
# ODE 6 - Airy Equation
x0: float = airy(3)[0]
v0: float = -airy(3)[1]
t0: float = -3.
tf: float = 7.
t: float = np.linspace(t0, tf, n[-1]+1)
x_exact = solution6(t)
ax[0][0].plot(t, x_exact, color='b')
ax[0][1].plot(t, x_exact, color='b')
for i in range(len(n)):
    t = np.linspace(t0, tf, n[i]+1)
    x_exact = solution6(t)
    x_verlet = verlet(t0, tf, x0, v0, n[i], f6)[0]
    x_rk = rungeKuttaPT(t0, tf, x0, v0, n[i], f6)[0]
    ax[0][0].plot(t, x_verlet, color=colors[i], ls='--')
    ax[1][0].plot(t, x_exact - x_verlet, color=colors[i], label=f'n={n[i]}')
    ax[0][1].plot(t, x_rk, color=colors[i], ls='--')
    ax[1][1].plot(t, x_exact - x_rk, color=colors[i], label=f'n={n[i]}')
ax[0][0].set_title('Verlet Approximations')
ax[0][1].set_title('Runge-Kutta 4 Approximations')
ax[1][0].set_title('Verlet Errors')
ax[1][1].set_title('Runge-Kutta 4 Errors')
ax[1][0].set_yscale('log')
ax[1][1].set_yscale('log')
ax[1][0].legend()
ax[1][1].legend()

plt.show()