from methods import rungeKutta
from secondOrderODEs import f7, solution7, f8, solution8, f9, solution9
from errorQudrature import rectangleRule
import numpy as np
import matplotlib.pyplot as plt

fps: int = 60
t0: float = 0.
tf: int = 30
n: int = fps * tf
x0=1
v0=0
t: np.ndarray = np.linspace(t0, tf, n + 1)
fig, ax = plt.subplots(2, 3)
# ODE 7 - Dampened Harmonic Oscillator
x_exact: np.ndarray = solution7(t)
x_appr: np.ndarray = rungeKutta(t0, tf, x0, v0, n ,f7)[0]
error: np.ndarray = x_exact - x_appr
meanAbsErrorApproximation: float = rectangleRule(t, error)
ax[0][0].plot(t, x_exact, color='DarkRed', label='Exact Solution')
ax[0][0].plot(t, x_appr, ls='--', label='Runge-Kutta 4')
ax[1][0].plot(t, error, label='error')
ax[1][0].plot([t[0], t[-1]],[meanAbsErrorApproximation, meanAbsErrorApproximation], color='r', ls='--', label='Mean Abs. Error')
ax[0][0].set_title(f'Dampened Harmonic Oscillator with step dt={round(tf / n, 4)}')
# ODE 8 - Forced Oscillator
x_exact: np.ndarray = solution8(t)
x_appr: np.ndarray = rungeKutta(t0, tf, x0, v0, n ,f8)[0]
error: np.ndarray = x_exact - x_appr
meanAbsErrorApproximation: float = rectangleRule(t, error)
ax[0][1].plot(t, x_exact, color='DarkRed', label='Exact Solution')
ax[0][1].plot(t, x_appr, ls='--', label='Runge-Kutta 4')
ax[1][1].plot(t, error, label='error')
ax[1][1].plot([t[0], t[-1]],[meanAbsErrorApproximation, meanAbsErrorApproximation], color='r', ls='--', label='Mean Abs. Error')
ax[0][1].set_title(f'Forced Oscillator with step dt={round(tf / n, 4)}')
# ODE 9
z0: float = 0.
v0: float = 200.
x_exact: np.ndarray = solution9(t)
x_appr: np.ndarray = rungeKutta(t0, tf, z0, v0, n ,f9)[0]
error: np.ndarray = x_exact - x_appr
meanAbsErrorApproximation: float = rectangleRule(t, error)
ax[0][2].plot(t, x_exact, color='DarkRed', label='Exact Solution')
ax[0][2].plot(t, x_appr, ls='--', label='Runge-Kutta 4')
ax[1][2].plot(t, error, label='error')
ax[1][2].plot([t[0], t[-1]],[meanAbsErrorApproximation, meanAbsErrorApproximation], color='r', ls='--', label='Mean Abs. Error')
ax[0][2].set_title(f'Air Resistence with step dt={round(tf / n, 4)}')

for axes1 in ax:
    for axes2 in axes1:
        axes2.legend()

fig.set_size_inches(18.5, 9.5)
fig.tight_layout()
plt.show()
