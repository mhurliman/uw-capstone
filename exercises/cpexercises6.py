import numpy as np

import matplotlib.pyplot as plt
import math
import cmath
import sys
import types

import IntegrationMethods as im

def ex_6_1():
    A = np.array([
        [4, -1, -1, -1],
        [-1, 3, 0, -1],
        [-1, 0, 3, -1],
        [-1, -1, -1, 4]
    ])

    b = np.array([5, 0, 0, 0])

    print(np.linalg.solve(A, b))

def ex_6_5():
    A = np.array([
        [complex(1.5, 1000), complex(0, -1000), 0],
        [complex(0, -1000), complex(1.5, 1500), complex(0, -500)],
        [0, complex(0, -500), complex(1.5, 500)]
    ], complex)

    b = np.array([3, 1.5, 3])

    V = np.linalg.solve(A, b)

    for i in range(len(V)):
        r, theta = cmath.polar(V[i])
        theta *= 180 / np.pi

        print('V{}: Amplitude - {}, Phase - {}'.format(i, r, theta))

def ex_6_9b():
    N = 10

    h = 6.62607015e-34 # J * s
    hB = h / (2 * np.pi)
    L = 5e-10          # m
    a = 10           # eV
    M = 9.1094e-31   # kg
    C = 1.6022e19    # J -> eV

    def f(m, n):
        condlist = [
            m == n,
            (m & 1) != (n & 1),
        ]
        choicelist = [
            hB**2 * n**2 * np.pi**2 / (2 * M * L**2) * C + a / 2,
            (2 * a / L**2) * (-2 * L / np.pi)**2 * (m * n / (m**2 - n**2)**2),
        ]
        return np.select(condlist, choicelist, 0)

    r = np.arange(N) + 1
    m, n = np.meshgrid(r, r, indexing='ij')
    H = f(m, n)

    e, v = np.linalg.eigh(H)
    v = np.transpose(v)

    x = np.linspace(0, L, 100)
    xi, ni = np.meshgrid(x, r, indexing='ij')

    psi = lambda x, n, v, i : np.sqrt(2 / L) * np.choose(n - 1, v[i]) * np.sin(np.pi * n * x / L)

    for i in range(3):
        p = np.square(psi(xi, ni, v, i))

        plt.plot(x, np.sum(p, axis=1))

        print(im.simpson_samples(x, p))


    plt.show()

    #plt.show()

   ## psi(xi, np.choose(v), r)
   # np.choose()

def ex_6_b():
    h = 6.62607015e-34 # J * s
    
    L = 1e-10         # m
    a = 10            # eV
    M = 9.1093837e-31 # kg
    c = 299792458     # m / s
    C = 1.6022e19     # J -> eV
    
    def g(n):
        return h**2 * n**2 / (8 * M * c**2 * L**2) * C

    print(g(1))



default = ex_6_9b

if __name__ == '__main__':
    if len(sys.argv) == 1:
        default()

    else:
        name = sys.argv[1]
        name = 'ex_' + name

        this = sys.modules[__name__]
        f = getattr(this, name)

        if isinstance(f, types.FunctionType):
            f()
