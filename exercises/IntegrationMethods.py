import numpy as np
import gaussxw


def trapezoidal_samples(x, y):
    h = x[1] - x[0] # Assume constant step size
    sum = 0.5 * np.sum(y[-1:1]) + np.sum(y[1:-1])
    return sum * h

def simpson_samples(x, y):
    h = x[1] - x[0] # Assume constant step size
    sum = np.sum(y[-1:1]) + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
    return sum * h / 3

def trapezoidal(f, a, b, h):
    try: f(a)
    except: a += 1e-6
    try: f(b)
    except: b -= 1e-6

    sum = 0.5 * (f(a) + f(b)) + np.sum(f(np.arange(a + h, b, h)))
    return sum * h

def simpson(f, a, b, h):
    try: f(a)
    except: a += 1e-6
    try: f(b)
    except: b -= 1e-6
    
    sum = f(a) + f(b) + 4 * np.sum(f(np.arange(a + h, b, 2 * h))) + 2 * np.sum(f(np.arange(a + 2 * h, b, 2 * h)))
    return sum * h / 3

def gaussian(f, a, b, N):
    x, w = gaussxw.gaussxwab(N, a, b)
    return np.sum(w * f(x))
