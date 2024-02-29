from math import pi
import math
import numbers
import numpy as np
import matplotlib.pyplot as plt
from functools import partial as bind

import IntegrationMethods as im

def ex_5_1():
    data = np.loadtxt('velocities.txt', np.float64)
    x, y = np.split(data, 2, axis=1)
    plt.plot(x, y)

    sum = im.impson_samples(x, y)
    print(sum)

    ys = np.zeros(x.shape[0])
    for i in range(1, x.shape[0]):
        ys[i] = im.simpson_samples(x, y[:i])

    plt.plot(x, ys)
    plt.show()

def ex_5_2():
    def f(x):
        return x**4 - 2 * x + 1

    N = 1000
    r = (0, 2)
    h = (r[1] - r[0]) / N

    sum = im.simpson(f, r[0], r[1], h)

    print(sum)

    x = np.arange(0, 2, h)
    y = f(x)

    plt.plot(x, y)
    plt.show()

def ex_5_3():
    def f(t):
        return np.exp(-t**2)

    def E(x, h, i):
        return i(f, 0, x, h)

    r = 3
    step = 0.1
    N = int(r/step)
    ys = np.zeros(N)

    x = np.arange(0, r, step)
    h = 0.001

    for i in range(0, N):
        x[i] = i * step
        ys[i] = E(x[i], h, im.simpson)

    plt.plot(x, ys)
    plt.show()

def ex_5_4():
    def f(theta, m, x):
        return np.cos(m * theta - x * np.sin(theta))

    def J(m, x, N):
        if isinstance(x, numbers.Number):
            x = [x]

        return np.array([im.trapezoidal(bind(f, m=m, x=xi), 0, pi, pi / N) for xi in x])

    r = 20
    step = 0.1

    xs = np.arange(0, r, step)
    for i in range(4):
        j0 = J(i, xs, 1000)
        plt.plot(xs, j0, label='J{}'.format(i))

    plt.legend()
    plt.show()

def ex_5_4b():
    def J(m, x, N, i):
        f = lambda t : np.cos(m * t - x * np.sin(t))
        return i(f, 0, pi, pi / N)

    def I(r, l):
        f = lambda x : np.square(J(1, x, 1000, im.simpson) / x)
        vf = np.vectorize(f)

        return vf(2 * pi / l * r)

    r = 1e-6
    N = 100
    h = 2 * r / N
    l = 5e-7

    x = np.arange(-r, r, h)
    y = np.arange(-r, r, h)

    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = I(r, l)

    plt.imshow(z, vmax=1e-1, extent=[x.min(), x.max(), y.min(), y.max()])
    plt.show()

def ex_5_6():
    f = lambda x : x**4 - 2*x + 1

    R = 2

    N = 10
    h = float(R) / N
    I1 = im.trapezoidal(f, 0, R, h)
    print(I1)

    N = N * 2
    h = float(R) / N
    I2 = im.trapezoidal(f, 0, R, h)
    print(I2)

    error = abs(I2 - I1) / 3

    print('Error: ', error)

def ex_5_7a():
    f = lambda x : np.sin((100 * x)**0.5)**2

    error_target = 1e-6

    N = 1
    h = 1 / N
    I1 = im.trapezoidal(f, 0, 1, h)

    i = 0
    e = np.inf
    while e > error_target:
        N = N * 2
        h = 1 / N
        I2 = im.trapezoidal(f, 0, 1, h)

        e = abs(I2 - I1) / 3
        I1 = I2
        i += 1

    print('Iterations: ', i)
    print('Integral: ', I1)
    print('Error: ', e)

def ex_5_7b():
    f = lambda x : np.sin((100 * x)**0.5)**2

    R = [None]
    R[0] = [im.trapezoidal(f, 0, 1, 1)]

    i = 1
    N = 2
    e = np.inf
    error_target = 1e-6
    while abs(e) > error_target:
        R.append([0] * (i + 1))

        R[i][0] = im.trapezoidal(f, 0, 1, 1 / N)

        for j in range(1, i+1):
            e = (R[i][j - 1] - R[i - 1][j - 1]) / (4**j - 1)
            R[i][j] = R[i][j - 1] + e

        i += 1
        N = N * 2

    print(e)
    for i in range(len(R)):
        #print(len(R[i]))
        print(R[i])

def ex_5_8():
    f = lambda x : np.sin((100 * x)**0.5)**2
    S = lambda f, a, b, h: (f(a) + f(b) + 2 * np.sum(f(np.arange(a, b, 2 * h)))) / 3
    T = lambda f, a, b, h : 2 * np.sum(f(np.arange(a + h, b, 2 * h))) / 3

    error_target = 1e-6

    a = 0
    b = 1
    N = 2
    h = (b - a) / N

    Si = S(f, a, b, h)
    Ti = T(f, a, b, h)

    I0 = h * (Si + 2 * Ti)

    it = 0
    e = np.inf
    while abs(e) > error_target:
        N = N * 2
        h = (b - a) / N

        Si = Si + Ti
        Ti = T(f, a, b, h)

        I1 = h * (Si + 2 * Ti)
        e = (I1 - I0) / 15
        I0 = I1
        it += 1

    print(it)
    print(I0)

def ex_5_9():
    f = lambda x : x**4 * np.exp(x) / (np.exp(x) - 1)**2
    
    def Cv(T, V, p, thD, N):
        Kb = 1.380649e-23
        return 9 * V * p * Kb * (T / thD)**3 * im.gaussian(f, 0, thD / T, 50)
    
    V = 1000 * 10**-6
    p = 6.022e28
    thD = 428

    arr = []
    for T in range(5, 500):
        arr.append(Cv(T, V, p, thD, 50))

    plt.plot(range(5, 500), arr)
    plt.xlabel('K°')
    plt.ylabel('Heat Capacity')
    plt.show()

def ex_5_10():
    f = lambda x, a : 1.0 / np.sqrt(a**4 - x**4)
    t = lambda a, m, N : np.sqrt(8 * m) * im.gaussian(bind(f, a=a), 0, a, N)
    vt = np.vectorize(t)

    M = 50    # Steps for independent variable 'a'
    h = 2 / M # Step size 

    a = np.arange(h, 2, h) # Start at 'h' since t(a=0) is undefined/complex

    m = 1  # Mass
    N = 20 # Gaussian quadrature steps
    T = vt(a, m, N)

    plt.plot(a, T)
    plt.xlabel('a')
    plt.ylabel('Period (s)')
    plt.show()

def ex_5_12():
    kB = 1.380649e-23 # J / K
    c = 299792458
    h = 6.62607015e-34
    hb = h / (2 * pi)

    f = lambda z : z**3 / (1 - z)**5 / (np.exp(z / (1 - z)) - 1)
    ft = lambda T, N: kB**4 / (4 * pi**2 * c**2 * hb**3) * T**4 * im.gaussian(f, 1e-6, 1 - 1e-6, N)

    N = 50
    sigma = kB**4 / (4 * pi**2 * c**2 * hb**3) * im.gaussian(f, 1e-6, 1 - 1e-6, N)
    print(sigma)


def H(n, x):
    if n > 1:
        return 2 * x * H(n-1, x) - 2 * n * H(n-2, x)
    elif n == 1:
        return 2 * x
    else:
        return 1

def psi(n, x):
    return np.exp(-x**2/2) * H(n, x) / np.sqrt(2**n * math.factorial(n) * np.sqrt(pi))

def ex_5_13a():
    N = 100
    x = np.linspace(-4, 4, N)
    for i in range(4):
        y = psi(i, x)
        plt.plot(x, y, label='psi_{}'.format(i))
    
    plt.legend()
    plt.show()

def ex_5_13b():
    N = 100
    x = np.linspace(-10, 10, N)
    y = psi(30, x)

    plt.plot(x, y, label='psi_30')
    plt.legend()
    plt.show()

def ex_5_13c():
    f = lambda n, z: (1 + z**2) / (1 - z**2)**2 * (z / (1 - z**2))**2 * psi(n,  z / (1 - z**2))

    N = 100
    m = im.gaussian(bind(f, 4), -1, 1, N)

    print(np.sqrt(m))

def finite_differences(f, x):
    h = (x.max() - x.min()) / len(x)
    return (f(x + h / 2) - f(x - h / 2)) / h

def ex_5_15():
    f = lambda x : 1 + 0.5 * np.tanh(2 * x)

    N = 100
    x = np.linspace(-2, 2, N)
    y = f(x)
    y2 = finite_differences(f, x)

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.show()


ex_5_15()
