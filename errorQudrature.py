import numpy as np

def rectangleRule(t: np.ndarray, errors: np.ndarray)->float:
    absErrors = np.abs(errors)
    integral = 0
    for i in range(len(t) - 1):
        integral += absErrors[i] * (t[i + 1] - t[i])
    return integral / (t[-1] - t[0])

if __name__ == "__main__":
    l = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    for num in l:
        t = np.linspace(0, 2 * np.pi, num)
        y = np.sin(t)
        integral=rectangleRule(t, y)
        print(f'n={num}: integral={integral}')
    for num in l:
        t = np.linspace(0, 1., num)
        y = np.exp(t)
        integral=rectangleRule(t, y)
        print(f'n={num}: integral={integral}')
    print(np.e - 1)
    print(np.sqrt(2) / 2)