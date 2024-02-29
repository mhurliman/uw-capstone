import numpy as np
import matplotlib.pyplot as plt
from functools import partial as bind
import sys
import types

def euler(f, x, a, b, N):
    h = (b - a) / N

    tpoints = np.arange(a, b, h)
    xpoints = []
    for t in tpoints:
        xpoints.append(x)
        x += h * f(x, t)

    return tpoints, xpoints

def rk4(f, x, a, b, N):
    h = (b - a) / N

    tpoints = np.arange(a, b, h)
    xpoints = []
    for t in tpoints:
        xpoints.append(x)

        k1 = h * f(x, t)
        k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(x + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(x + k3, t + h)

        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return tpoints, xpoints

def ex_8_1a():
    f = lambda x, t : np.select([np.trunc(2 * t) % 2 == 0], [1], -1)

    x = 0.0

    N = 10000
    a = 0.0
    b = 10.0
    
    tpoints, xpoints = rk4(f, x, a, b, N)

    plt.plot(tpoints, xpoints)
    plt.show()

def ex_8_1b():
    j = lambda t : np.select([np.trunc(2 * t) % 2 == 0], [1], -1)
    f2 = lambda x, t, RC : (1/RC) * (j(t) - x)

    Vout = 0

    N = 10000
    a = 0
    b = 10
    h = b / N

    t1, x1 = rk4(bind(f2, RC=0.01), Vout, a, b, N)
    t2, x2 = rk4(bind(f2, RC=0.1), Vout, a, b, N)
    t3, x3 = rk4(bind(f2, RC=1), Vout, a, b, N)

    plt.plot(t1, x1, label='RC = 0.01')
    plt.plot(t2, x2, label='RC = 0.1')
    plt.plot(t3, x3, label='RC = 1.0')

    plt.xlabel('Time (s)')
    plt.ylabel('Voltage Out')
    plt.legend()

    plt.show()
    


def ex():
    f = lambda x, t: -x**3 + np.sin(t)

    a = 0
    b = 10
    N = 20
    x = 0

    t1, x1 = euler(f, x, a, b, N)
    t2, x2 = rk4(f, x, a, b, N)

    fig, axs = plt.subplots(2, 1,  layout='constrained')
    axs[0].plot(t1, x1)
    axs[1].plot(t2, x2)
    plt.show()


default = ex

if __name__ == '__main__':
    if len(sys.argv) == 1:
        default()

    name = sys.argv[1]
    name = 'ex_' + name

    this = sys.modules[__name__]
    f = getattr(this, name)

    if isinstance(f, types.FunctionType):
        f()