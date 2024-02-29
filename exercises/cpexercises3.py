from math import sqrt, sin, pi
import numpy as np
import matplotlib.pyplot as plt

import sys
import types

def draw_circular():
    data = np.loadtxt("circular.txt", float)
    plt.imshow(data, origin="lower", extent=[0, 10, 0, 5], aspect=2.0)
    plt.gray()
    plt.show()

def draw_wave_interference():
    wavelength = 5.0
    k = 2 * pi / wavelength
    xi0 = 1.0
    separation = 20.0
    side = 100.0
    points = 500
    spacing = side / points

    x1 = side / 2 + separation / 2
    y1 = side / 2
    x2 = side / 2 - separation / 2
    y2 = side / 2

    xi = np.empty([points, points], float)

    for i in range(points):
        y = spacing * i

        for j in range(points):
            x = spacing * j

            r1 = sqrt((x - x1)**2 + (y - y1)**2)
            r2 = sqrt((x - x2)**2 + (y - y2)**2)

            xi[i, j] = xi0 * sin(k * r1)  + xi0 * sin(k * r2)

    plt.imshow(xi, origin="lower", extent=[0, side, 0, side])
    plt.gray()
    plt.show()

def ex3_1():
    data = np.loadtxt('sunspots.txt', float)
    x, y = np.hsplit(data, 2)

    r = 5

    Y = np.zeros_like(y)
    for k in range(len(y)):
        start = max(0, k - r)
        end = min(k + r + 1, len(y))
        Y[k] = np.sum(y[start : end]) / (end - start)

    plt.plot(x, y, Y)
    plt.plot(x, Y)
    plt.xlim(0, 1000)
    plt.xlabel('Time (months)')
    plt.ylabel('Count')
    plt.show()

def ex3_2():
    def deltoid(theta):
        x = 2 * np.cos(theta) + np.cos(2 * theta)
        y = 2 * np.sin(theta) - np.sin(2 * theta)
        return x, y

    def p2c(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def galilean_spiral(theta):
        return theta**2

    def feys_function(theta):
        return np.exp(np.cos(theta)) - 2 * np.cos(4 * theta) + np.sin(theta/12)**5

    n = 1000
    fig, axs = plt.subplots(3, 1, layout='constrained')

    domain = 2 * pi
    theta = np.arange(0, domain, domain / n)
    x, y = deltoid(theta)
    axs[0].plot(x, y)

    domain = 10 * pi
    theta = np.arange(0, domain, domain / n)
    r = galilean_spiral(theta)
    x, y = p2c(r, theta)
    axs[1].plot(x, y)

    domain = 24 * pi
    theta = np.arange(0, domain, domain / n)
    r = feys_function(theta)
    x, y = p2c(r, theta)
    axs[2].plot(x, y)

    plt.show()

def ex3_3():
    data = np.loadtxt('stm.txt', float)

    h, w = data.shape

    x = np.arange(w)
    y = np.arange(h)

    xv, yv = np.meshgrid(x, y, indexing='xy')

    plt.contourf(xv, yv, data)
    plt.colorbar()
    plt.show()

def ex3_4_a():
    x_ = np.linspace(-5, 5, 11)
    y_ = np.linspace(-5, 5, 11)
    z_ = np.linspace(-5, 5, 11)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')

    b = (x + y + z) % 2 == 0

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x[b], y[b], z[b], marker='o')
    ax.scatter(x[~b], y[~b], z[~b], marker='^')

    plt.show()


def ex3_4_b():
    n = 0.5
    a_ = np.arange(n)
    a_ = np.repeat(a_, 4)

    x_ = np.linspace(-n, n, int(2 * n + 1))
    y_ = np.linspace(-n, n, int(2 * n + 1))
    z_ = np.linspace(-n, n, int(2 * n + 1))

    x2_ = np.linspace(-n + 0.5, n + 0.5, int(2 * n + 1))
    y2_ = np.linspace(-n + 0.5, n + 0.5, int(2 * n + 1))
    z2_ = np.linspace(-n + 0.5, n + 0.5, int(2 * n + 1))

    x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')
    x2, y2, z2 = np.meshgrid(x2_, y2_, z_, indexing='xy')
    x3, y3, z3 = np.meshgrid(x2_, y_, z2_, indexing='xy')
    x4, y4, z4 = np.meshgrid(x_, y2_, z2_, indexing='xy')

    xn = np.append(x, [x2, x3, x4])
    yn = np.append(y, [y2, y3, y4])
    zn = np.append(z, [z2, z3, z4])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xn, yn, zn,  marker='o')

    plt.show()

def ex3_5():
    pass # Can't download package 'visual' on the plane

def logistic_map(r, x):
    return r * x * (1.0 - x)

def ex3_6_a():
    iterations = 1000
    dr = 0.01

    rd = np.arange(1, 4, dr)
    xd = np.empty(rd.shape[0] * iterations)
    offset = 0

    for r in rd:
        x = 0.5
        for i in range(iterations):
            x = logistic_map(r, x)

        for i in range(iterations):
            xd[offset + i] = x
            x = logistic_map(r, x)

        offset += iterations

    rd = np.repeat(rd, iterations)

    plt.scatter(rd, xd)
    plt.show()

def ex3_6_b():
    iterations = 1000
    dr = 0.01

    rd = np.arange(1, 4, dr)
    domain = rd.shape[0]
    xp = np.full(domain, 0.5)

    for i in range(iterations):
        xp = logistic_map(rd, xp)

    xdd = list(range(iterations))
    xdd[0] = xp
    for i in range(0, iterations-1):
        xdd[i+1] = logistic_map(rd, xdd[i])

    rd = np.tile(rd, iterations)
    xd = np.stack(xdd)

    plt.scatter(rd, xd)
    plt.show()

def mandelbrot(z, c):
    return z**2 + c

def ex3_7():
    n = 1000
    bounds = 2
    step = 2 * bounds / n
    x = np.arange(-bounds, bounds, step)

    cx, cy = np.meshgrid(x, x, indexing='xy')
    c = cx + complex(0, 1) * cy
    z = np.zeros((n, n), np.complex128)

    iterations = 100
    data = np.ones((n, n), np.int32)
    for i in range(iterations):
        z = mandelbrot(z, c)

        data += np.choose(np.abs(z) > 2.0, [0, 1])

    data = np.log(data)

    plt.imshow(data)
    plt.show()

def least_squares_fit(x, y):
    Ni = 1.0 / x.shape[0]
    Ex = Ni * np.sum(x)
    Ey = Ni * np.sum(y)
    Exx = Ni * np.sum(np.square(x))
    Exy = Ni * np.sum(x * y)
    
    d = 1.0 / (Exx - np.square(Ex))
    m = (Exy - Ex * Ey) * d
    c = (Exx * Ey - Ex * Exy) * d

    return m, c

def ex3_8():
    data = np.loadtxt('millikan.txt', np.float64)
    x, y = np.split(data, 2, axis=1)

    m, c = least_squares_fit(x, y)
    y2 = m * x + c

    plt.scatter(x, y)
    plt.plot(x, y2)
    plt.show()

def ex3_8d():
    data = np.loadtxt('millikan.txt', np.float64)
    x, y = np.split(data, 2, axis=1)

    e = np.float64(1.602 * 10**-19)

    m, c = least_squares_fit(x, y)
    
    print(m * e)


default = ex3_8d

if __name__ == '__main__':
    if len(sys.argv) == 1:
        default()

    name = sys.argv[1]
    name = 'ex' + name

    this = sys.modules[__name__]
    f = getattr(this, name)

    if isinstance(f, types.FunctionType):
        f()
