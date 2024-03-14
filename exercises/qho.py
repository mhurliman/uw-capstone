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

    L = 5
    a =  2 * L / N

    f = 8.88e13
    w = 2 * np.pi * f

    m = 9.109e-31 # kg
    hb = 1.054571817e-34 # J s 
    hbEv = 6.5821220e-16 # Ev s

    c = 3 # Energy levels to map

    x = np.linspace(-L, L, N + 1)

    U = m * w**2 * x**2
    T = -hbEv**2 / (2 * m * a**2) * triagonal(1, -2, 1, N + 1)
    H = T + np.diag(U)

    e, v = np.linalg.eigh(H)

    vc = np.reshape(v[:, 0], N + 1)
    ec = e[0]
    m = np.max(abs(vc) + ec)

    print(e[1] - e[0])
    print(ec)

    fig, ax = plt.subplots()
    wave = ax.plot(x, np.zeros([N+1, c+1]))
    ax.set(ylim=[-m, m], xlabel='x (m)', ylabel='psi(x, t)')

    h = 1.0 / 33 / ec * 2 * np.pi

    wave[c].set_ydata(U)

    def update(frame):
        t = frame * h
        ax.set(title='t = {}'.format(t))

        y = np.real(v[:, :c] * np.exp(1j * e[:c] * t)) + e[:c]

        for i, w in enumerate(wave):
            if i < c:
                w.set_ydata(y[:, i])

    an = anim.FuncAnimation(fig=fig, func=update, interval=1)
    plt.show()


def qho2d():
    N = 40
    L = 1e-8
    a =  L / N

    f = 8.88e13
    w = 2 * np.pi * f

    m = 9.109e-31 # kg
    hb = 1.054571817e-34 # J s 
    hbEv = 6.5821220e-16 # Ev s

    c = 2 # Energy levels to map
    f = lambda p : -L / 2 + p * a

    plot_args = {
        'rstride': 1, 'cstride': 1, 'cmap': 'viridis', 'linewidth': 0.01, 
        'antialiased': True, 'color': 'w', 'shade': True 
    }

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

    T = -hbEv**2 / (2 * m * a**2) * np.select([x0 & y0, (x1 & y0) | (x0 & y1)], [-4, 1], 0)

    xs, ys = f(xis), f(yis)
    U = m * w**2 * (xs**2 + ys**2)

    H = T + np.diag(U)
    e, v = np.linalg.eigh(H)

    vc = np.reshape(v[:, c], (N + 1, N + 1))
    ec = e[c]
    m = np.max(abs(vc) + ec)

    print(e[:10])
    print(e[1] - e[0])

    x, y = f(xi), f(yi)
    xis, yis = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(xis, yis, np.real(vc) + ec, **plot_args)

    h = 1.0 / 33 / ec * 2 * np.pi

    def update(frame):
        t = frame * h

        ax.clear()
        ax.set(zlim=[-m, m], xlabel='x (m)', ylabel='y (m)', zlabel='psi(x, t)', title='t = {} (s)'.format(t))

        ax.plot_surface(xis, yis, np.real(vc * np.exp(1j * ec * t)) + ec, **plot_args)


    an = anim.FuncAnimation(fig=fig, func=update, interval=33)
    plt.show()

def qho3d():
    N = 8
    L = 1e-8
    a =  L / N

    f = 8.88e13
    w = 2 * np.pi * f

    m = 9.109e-31 # kg
    hb = 1.054571817e-34 # J s 
    hbEv = 6.5821220e-16 # Ev s

    c = 2 # Energy levels to map
    f = lambda p : -L / 2 + p * a

    plot_args = {
        'rstride': 1, 'cstride': 1, 'cmap': 'viridis', 'linewidth': 0.01, 
        'antialiased': True, 'color': 'w', 'shade': True 
    }

    xi = np.arange(N + 1)
    yi, zi = np.copy(xi), np.copy(xi)

    xis = np.tile(xi, N + 1)
    yis = np.tile(np.repeat(yi, N + 1), N + 1)
    zis = np.repeat(zi, (N + 1)**3)

    # Fancy method of laying out T
    xis0, xis1 = np.meshgrid(xis, xis)
    xx = xis0 - xis1
    x0, x1 = xx == 0, abs(xx) == 1

    yis0, yis1 = np.meshgrid(yis, yis)
    yy = yis0 - yis1
    y0, y1 = yy == 0, abs(yy) == 1

    zis0, zis1 = np.meshgrid(yis, yis)
    zz = zis0 - zis1
    z0, z1 = zz == 0, abs(zz) == 1

    T = -hbEv**2 / (2 * m * a**2) * np.select([x0 & y0 & z0, (x1 & y0 & z0) | (x0 & y1 & z0) | (x0 & y0 & z1)], [-6, 1], 0)

    xs, ys, zs = f(xis), f(yis), f(zis)
    U = m * w**2 * (xs**2 + ys**2 + zs**2)

    H = T + np.diag(U)
    e, v = np.linalg.eigh(H)

    vc = np.reshape(v[:, c], (N + 1, N + 1, N + 1))
    ec = e[c]
    m = np.max(abs(vc) + ec)

    print(e[:10])
    print(e[1] - e[0])

    x, y, z = f(xi), f(yi), f(zi)
    xis, yis, zis = np.meshgrid(x, y, z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(xis, yis, np.real(vc) + ec, **plot_args)

    h = 1.0 / 33 / ec * 2 * np.pi

    def update(frame):
        t = frame * h

        ax.clear()
        ax.set(zlim=[-m, m], xlabel='x (m)', ylabel='y (m)', zlabel='psi(x, t)', title='t = {} (s)'.format(t))

        ax.plot_surface(xis, yis, np.real(vc * np.exp(1j * ec * t)) + ec, **plot_args)


    an = anim.FuncAnimation(fig=fig, func=update, interval=33)
    plt.show()

if __name__ == '__main__':
    qho3d()
