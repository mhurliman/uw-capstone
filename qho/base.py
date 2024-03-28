import numpy as np
from matplotlib import (
    pyplot as plt,
    animation as anim,
    axes,
    cm
)

# Creates a tridiagonal matrix with a, b, c as the repeated elements along the diagonal
def triagonal(a, b, c, n):
    return np.diag([a]*(n-1), -1) + np.diag([b]*n, 0) + np.diag([c]*(n-1), 1)

def qho1d():
    N = 1000

    xmax = 6
    dx =  2 * xmax / N

    f = 8.88e13
    w = 1 # 2 * np.pi * f

    m = 1 # 9.109e-31 # kg
    hb = 1.0 # 1.054571817e-34 # J s 
    hbEv = 1 # 6.5821220e-16 # Ev s

    c = 3 # Energy levels to map

    x = np.linspace(-xmax, xmax, N+1)

    U = 0.5 * m * w**2 * x**2
    T = -hbEv**2 / (2 * m * dx**2) * triagonal(1, -2, 1, N+1)
    H = T + np.diag(U)

    e, v = np.linalg.eigh(H)

    ymax = np.max(abs(np.reshape(v[:, c-1], N+1))) + e[c-1]
    ymin = -np.max(abs(np.reshape(v[:, 0], N+1))) + e[0]

    fig, ax = plt.subplots()
    wave = ax.plot(x, np.zeros([N+1, c+1]))
    ax.set(ylim=[ymin, ymax], xlabel='x (m)', ylabel='psi(x, t)')

    dt = (1.0 / 33) / e[0] * np.pi * 2

    wave[c].set_ydata(U)

    def update(frame):
        t = frame * dt
        ax.set(title='t = {}'.format(t))

        y = np.real(v[:, :c] * np.exp(-1j * e[:c] * t) + e[:c])

        for i, w in enumerate(wave):
            if i < c:
                w.set_ydata(y[:, i])

    an = anim.FuncAnimation(fig=fig, func=update, interval=33)
    plt.show()


def qho2d():
    N = 50

    xmax = 6
    dx =  2 * xmax / N

    f = 8.88e13
    w = 2 * np.pi * f

    m = 9.109e-31 # kg
    hb = 1.054571817e-34 # J s 
    hbEv = 6.5821220e-16 # Ev s

    c = 5 # Energy levels to map
    f = lambda p : -xmax + p * dx

    xi, yi = np.arange(N + 1), np.arange(N + 1)

    xis = np.tile(xi, N + 1)
    yis = np.repeat(yi, N + 1)

    # Fancy method of laying out T
    xis0, xis1 = np.meshgrid(xis, xis)
    xx = xis0 - xis1
    x0, x1 = xx == 0, abs(xx) == 1

    yis0, yis1 = np.meshgrid(yis, yis)
    yy = yis0 - yis1
    y0, y1 = yy == 0, abs(yy) == 1

    T = -hbEv**2 / (2 * m * dx**2) * np.select([x0 & y0, (x1 & y0) | (x0 & y1)], [-4, 1], 0)

    xs, ys = f(xis), f(yis)
    U = 0.5 * m * w**2 * (xs**2 + ys**2)

    H = T + np.diag(U)
    e, v = np.linalg.eigh(H)

    vc = np.reshape(v[:, c], (N + 1, N + 1))
    ec = e[c]
    m = np.max(abs(vc))

    print(e[1] - e[0])
    print(e[:10])

    x, y = f(xi), f(yi)
    xis, yis = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(xis, yis, np.real(vc) + ec, cmap='viridis')

    dt = (1.0 / 33) / ec * 2 * np.pi

    def update(frame):
        t = frame * dt

        ax.clear()
        ax.set(zlim=[ec-m, ec+m], xlabel='x (m)', ylabel='y (m)', zlabel='psi(x, t)', title='t = {} (s)'.format(t))

        ax.plot_surface(xis, yis, np.real(vc * np.exp(-1j * ec * t)) + ec, cmap='viridis')


    an = anim.FuncAnimation(fig=fig, func=update, interval=33)
    plt.show()

def qho3d():
    N = 12
    xmax = 2.5
    dx =  2 * xmax / N

    f = 8.88e13
    w = 2 * np.pi * f

    m = 9.109e-31 # kg
    hb = 1.054571817e-34 # J s 
    hbEv = 6.5821220e-16 # Ev s

    c = 1 # Energy level to map

    f = lambda p : -xmax + p * dx

    xi = np.arange(N + 1)
    yi, zi = np.copy(xi), np.copy(xi)

    xis = np.tile(xi, (N + 1)**2)
    yis = np.tile(np.repeat(yi, N + 1), N + 1)
    zis = np.repeat(zi, (N + 1)**2)

    # Fancy method of laying out T
    xis0, xis1 = np.meshgrid(xis, xis)
    xx = xis0 - xis1
    x0, x1 = xx == 0, abs(xx) == 1

    yis0, yis1 = np.meshgrid(yis, yis)
    yy = yis0 - yis1
    y0, y1 = yy == 0, abs(yy) == 1

    zis0, zis1 = np.meshgrid(zis, zis)
    zz = zis0 - zis1
    z0, z1 = zz == 0, abs(zz) == 1

    T = -hbEv**2 / (2 * m * dx**2) * np.select([x0 & y0 & z0, (x1 & y0 & z0) | (x0 & y1 & z0) | (x0 & y0 & z1)], [-6, 1], 0)

    xs, ys, zs = f(xis), f(yis), f(zis)
    U = 0.5 * m * w**2 * (xs**2 + ys**2 + zs**2)

    H = T + np.diag(U)
    e, v = np.linalg.eigh(H)

    vc = np.reshape(v[:, c], (N + 1, N + 1, N + 1))
    ec = e[c]

    print(e[1] - e[0])
    print(e[:20])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xis, yis, zis = np.meshgrid(f(xi), f(yi), f(zi))
    ax.scatter(xis, yis, zis, c=np.real(vc) + ec, cmap='viridis')

    dt = (1.0 / 33) / ec * 2 * np.pi

    def update(frame):
        t = frame * dt

        ax.clear()
        ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='y (m)', title='t = {} (s)'.format(t))

        ax.scatter(xis, yis, zis, c=np.real(vc * np.exp(-1j * ec * t)) + ec, cmap='viridis')


    an = anim.FuncAnimation(fig=fig, func=update, interval=33)
    plt.show()


default = qho1d

if __name__ == '__main__':
    import sys
    import types

    if len(sys.argv) == 1:
        default()

    else:
        name = sys.argv[1]
        name = 'qho' + name

        this = sys.modules[__name__]
        f = getattr(this, name)

        if isinstance(f, types.FunctionType):
            f()
