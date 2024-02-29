
from functools import partial as bind
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
import types
from time import time
import vpython as vp
import dcst

def ex_9_1a():
    M = 100
    V = 1.0
    target = 1e-6

    f = lambda p, x, y: (p[x + 1, y] + p[x - 1, y] + p[x, y + 1] + p[x, y - 1]) / 4

    phi = np.zeros([M + 1, M + 1], float)
    phi[:, 0] = V

    phip = np.copy(phi)

    it = 0
    t = time()

    # Jacobi iteration
    delta = float('inf')

    while delta > target:
        for y in np.arange(1, M):
            for x in np.arange(1, M):
                phip[x, y] = f(phi, x, y)

        delta = np.max(np.abs(phip - phi))
        phi, phip = phip, phi

        # Print out progress every 100 iterations
        it += 1
        if it % 100 == 0:
            print(delta)

    print("{}s elapsed - {} Iterations".format(time() - t, it))

    phi = np.transpose(phi) # x, y indexing
    plt.imshow(phi)
    plt.show()


def ex_9_1b():
    M = 100
    V = 1.0
    target = 1e-6

    f = lambda p, x, y: (p[x + 1, y] + p[x - 1, y] + p[x, y + 1] + p[x, y - 1]) / 4

    p = np.zeros([M + 1, M + 1], float)
    p[60:80, 20:40] = V
    p[20:40, 60:80] = -V

    phi = np.copy(p)
    phip = np.copy(phi)

    it = 0
    t = time()

    # Jacobi iteration
    delta = float('inf')
    
    while delta > target:
        for y in np.arange(1, M):
            for x in np.arange(1, M):
                phip[x, y] = f(phi, x, y) + p[x, y] / 4

        delta = np.max(np.abs(phip - phi))
        phi, phip = phip, phi

        # Print out progress every 100 iterations
        it += 1
        if it % 100 == 0:
            print(delta)

    print("{}s elapsed - {} Iterations".format(time() - t, it))

    phi = np.transpose(phi) # x, y indexing
    plt.imshow(phi)
    plt.show()


def ex_9_2():
    M = 100
    V = 1.0
    target = 1e-6
    omega = 0.9

    f = lambda p, x, y: (p[x + 1, y] + p[x - 1, y] + p[x, y + 1] + p[x, y - 1]) / 4

    phi = np.zeros([M + 1, M + 1], float)
    phi[:, 0] = V

    it = 0
    t = time()

    # Gauss-Seidel iteration
    delta = float('inf')
    
    while delta > target:
        phip = np.copy(phi)

        for y in np.arange(1, M):
            for x in np.arange(1, M):
                phi[x, y] = (1 + omega) * f(phi, x, y) - omega * phi[x, y]

        delta = np.max(np.abs(phip - phi))

        # Print out progress every 100 iterations
        it += 1
        if it % 100 == 0:
            print(delta)

    print("{}s elapsed - {} Iterations".format(time() - t, it))

    phi = np.transpose(phi)
    plt.imshow(phi)
    plt.show()

def ex_9_3():
    M = 100
    V = 1.0
    target = 1e-6
    omega = 0.9

    f = lambda p, x, y: (p[x + 1, y] + p[x - 1, y] + p[x, y + 1] + p[x, y - 1]) / 4

    phi = np.zeros([M + 1, M + 1], float)
    phi[20, 20:80] = V
    phi[80, 20:80] = V

    mask = phi == 0

    it = 0
    t = time()

    # Gauss-Seidel iteration
    delta = float('inf')

    while delta > target:
        phip = np.copy(phi)

        for y in np.arange(1, M):
            for x in np.arange(1, M):
                if mask[x, y]:
                    phi[x, y] = (1 + omega) * f(phi, x, y) - omega * phi[x, y]

        delta = np.max(np.abs(phip - phi))

        it += 1
        if it % 100 == 0:
            print(delta)
            
    print("{}s elapsed - {} Iterations".format(time() - t, it))

    phi = np.transpose(phi)
    plt.imshow(phi)
    plt.title('Voltage')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()

def ex_9_4():
    D = 0.1 # m^2 day^-1

    L = 20       # Simulation depth in meters
    N = 100      # Number of spatial grid points
    a = L / N    # Grid spacing in meters

    T = 10 * 365 # Simulation duration in days
    h = 0.2      # Timestep in days

    # When choosing h & a we must keep in mind the stability condition h <= a**2 / (2*D)
    assert(h <= a**2 / (2 * D))

    d = np.full(N + 1, 10, float)
    d[-1] = 11
    dp = np.copy(d)

    plot_times = [9, 9.25, 9.5, 9.75]
    plot_labels = ['spring', 'summer', 'fall', 'winter']
    plot_times.sort()
    plot_i = 0

    f = lambda t : 10 + 12 * np.sin(2 * np.pi * t / 365)

    for t in np.arange(0, T, h):
        d[0] = f(t)
        dp[1:-1] = h * D / a**2 * np.convolve(d, [1, -2, 1], 'valid') + d[1:-1]

        d, dp = dp, d

        if plot_i < len(plot_times) and t >= plot_times[plot_i] * 365:
            plt.plot(np.linspace(0, L, N + 1), d, label=plot_labels[plot_i])
            plot_i += 1

    plt.legend()
    plt.show()

def ex_9_5():
    # phi(x, t + h) = phi(x, t) + h * psi(x, t)
    # psi(x, t + h) = psi(x, t) + h * v**2 / a**2 * (phi(x + a, t) + phi(x - a, t) - 2 * phi(x, t))
    L = 100       # cm
    d = 10        # cm
    C = 1         # ms^-1
    sigma = 30    # cm
    h = 1e-3      # Timestep (ms)
    v = 100       # ms^-1

    N = 50      # Number of spatial grid points
    a = L / N    # Grid spacing in meters


    psi_0 = lambda x : C * (x * (L - x)) / L**2 * np.exp(-(x - d)**2 / (2 * sigma**2))

    x = np.linspace(0, L, N + 1)

    phi = np.zeros(N + 1, float)
    psi = psi_0(x)

    phi[0] = phi[-1] = 0
    psi[0] = psi[-1] = 0

    phip = np.copy(phi)
    psip = np.copy(psi)

    g = vp.graph(ymin=-1e-1, ymax=1e-1, title='t = 0 ms')
    c = vp.gcurve()

    for t in np.arange(0, 100, h):
        vp.rate(30)
        phip = phi + h * psi
        psip[1:-1] = h * v**2 / a**2 * np.convolve(phi, [1, -2, 1], 'valid') + psi[1:-1]

        phi, phip = phip, phi
        psi, psip = psip, psi

        bl = np.dstack((x, phi))
        bl = bl.squeeze()

        c.data=bl.tolist()
        g.title = 't = {:.3f} ms'.format(t)

def triagonal(a, b, c, n):
    a = np.complex128(a)
    b = np.complex128(b)
    c = np.complex128(c)
    return np.diag([a]*(n-1), -1) + np.diag([b]*n, 0) + np.diag([c]*(n-1), 1)

from decimal import Decimal

def ex_9_8():
    N = 1000
    L = 1e-8 # m
    a = L / N

    h = 1e-18 # s

    x_0 = L / 2
    sigma = 1e-10 # m
    k = 5e10 # m^-1

    hb = 1.054571817e-34 # J s 
    m = 9.109e-31 # kg

    a_1 = 1 + h * 1j * hb / (2 * m * a**2)
    a_2 = -h * 1j * hb / (4 * m * a**2)

    b_1 = 1 - h * 1j * hb / (2 * m * a**2)
    b_2 = h * 1j * hb / (4 * m * a**2)

    A = triagonal(a_2, a_1, a_2, N+1)
    B = triagonal(b_2, b_1, b_2, N+1)
    Z = np.matmul(np.linalg.inv(A), B)

    psi_0 = lambda x: np.exp(-(x - x_0)**2 / (2 * sigma**2)) * np.exp(1j * k * x)

    x = np.linspace(0, L, N+1, True)
    psi = psi_0(x)

    g = vp.graph(ymin = -1, ymax = 1, title='t = 0 fs')
    c = vp.gcurve()

    t = 0
    while True:
        t += h
        vp.rate(90)

        psip = np.matmul(Z, psi)

        psip, psi = psi, psip

        bl = np.dstack((x, np.abs(psi)))
        bl = bl.squeeze()

        c.data = bl.tolist()
        g.title = 't = {:.3f} fs'.format(t * 10**15)

def ex_9_9():
    N = 1000
    L = 1e-8 # m
    a = L / N

    h = 1e-18 # s
    T = h * 1000

    x_0 = L / 2
    sigma = 1e-10 # m
    K = 5e10 # m^-1

    hb = 1.054571817e-34 # J s 
    m = 9.109e-31 # kg

    E = lambda k: (np.pi * hb * k)**2 / (2 * m * L**2)
    psi_k = lambda x, t, k: np.sin(np.pi * k * x / L) * np.exp(1j * E(k) * t / hb)

    psi_0 = lambda x: np.exp(-(x - x_0)**2 / (2 * sigma**2)) * np.exp(1j * K * x)

    x = np.linspace(0, L, N, True)
    n = np.arange(0, len(x))
    psi = psi_0(x)

    psi_real = np.real(psi)
    psi_imag = np.imag(psi)

    bk = dcst.dst(psi_real) + 1j * dcst.dst(psi_imag)

    # bk * np.exp(1j * np.pi**2 * hb * )

    # dcst.idst()
    # psi_t(x, n, alpha, beta, k, 1e-16)

    # plt.plot(x, )
    # plt.show()


default = ex_9_9

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
