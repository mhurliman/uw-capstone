import numpy as np
from matplotlib import (
    pyplot as plt,
    animation as anim,
)

# Creates a tridiagonal matrix with a, b, c as the repeated elements along the diagonal
def triagonal(a, b, c, n):
    return np.diag(np.full(n-1, a), -1) + np.diag(np.full(n, b), 0) + np.diag(np.full(n-1, c), 1)

def gaussian1d(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(1/2) * ((x - mu) / sigma)**2)

def gaussian2d(x, y, mu, sigma):
    return gaussian1d(x, mu, sigma) * gaussian1d(y, mu, sigma)

def gaussian3d(x, y, z, mu, sigma):
    return gaussian2d(x, y, mu, sigma) * gaussian1d(z, mu, sigma)

def decompose(basis, psi, dx):
    b = np.trapz(basis * psi[:, np.newaxis], dx=dx, axis=0)
    b /= np.trapz(np.sum(b * basis, axis=1), dx=dx)
    return b

def spectral(k):
    I = np.eye(k.shape[0])
    return np.matmul(np.matmul(np.fft.ifft(I), np.diag(k)), np.fft.fft(I))

def qho1d():
    N = 10

    xmax = 5
    dx =  2 * xmax / N
    dk = np.pi / xmax
    kmax = dk * N / 2

    f = 8.88e13
    w = 2 * np.pi * f

    m = 9.109e-31 # kg
    hb = 1.054571817e-34 * 6.242e18 # J s * EV/J

    c = N  # Energy levels to cutoff

    x = np.linspace(-xmax, xmax, N + 1)
    k = np.roll(np.linspace(-kmax, kmax, N + 1), N // 2 + 1)

    U = 0.5 * m * w**2 * x**2
    T = hb**2 * k**2 / (2 * m)

    H = spectral(T) + np.diag(U)

    e, v = np.linalg.eigh(H)
    vc =  v[:, :c]
    ec = e[:c]

    psi = gaussian1d(x, -1, 1)
    Ak = decompose(vc, psi, dx)

    E = np.sum(np.abs(Ak)**2 * ec) # Total energy

    xbound = [-xmax, xmax]
    ybound = [-1, 5]

    fig, ax = plt.subplots()
    wave = ax.plot(x, np.zeros([N + 1, 2]))
    ax.set(xlim=xbound, ylim=ybound, xlabel='x (m)', ylabel='psi(x, t)')

    wave[0].set_ydata(U)

    dF = 33
    dt = (1.0 / dF) / ec[0] * np.pi * 2 

    def update(frame):
        t = frame * dt
        ax.set(title='t = {:.2f}'.format(t))

        psi_t = np.sum(Ak * vc * np.exp(-1j * ec * t), axis=1)
        y = np.abs(psi_t)

        wave[1].set_ydata(y)

    an = anim.FuncAnimation(fig=fig, func=update, interval=dF, cache_frame_data=False)
    plt.show()


def qho2d():
    N = 20

    xmax = 6
    dx =  2 * xmax / N
    dk = np.pi / xmax
    kmax = dk * (N / 2)

    f = 8.88e13
    w = 1 #2 * np.pi * f

    m = 1 #9.109e-31 # kg
    hb = 1 #1.054571817e-34 * 6.242e18 # J s * (EV / J) 

    c = N # Energy levels to map

    k = np.roll(np.linspace(-kmax, kmax, N + 1), N // 2 + 1)
    x = np.linspace(-xmax, xmax, N + 1)

    kxx, kyy = np.tile(k, N + 1), np.repeat(k, N + 1)
    xx, yy = np.tile(x, N + 1), np.repeat(x, N + 1)

    T = hb**2 / (2 * m) * (kxx**2 + kyy**2)
    U = 0.5 * m * w**2 * (xx**2 + yy**2)

    H = spectral(T) + np.diag(U)

    e, v = np.linalg.eigh(H)

    vc = np.reshape(v[:, :c], (N + 1, N + 1, c))
    ec = e[:c]

    xs, ys = np.meshgrid(x, x)

    psi = gaussian2d(xs, ys, -1, 1)
    Ak = np.trapz(np.trapz(vc * psi[:, :, np.newaxis], dx=dx, axis=0), dx=dx, axis=0)
    Ak /= np.trapz(np.trapz(np.sum(Ak * vc, axis=2), dx=dx, axis=0), dx=dx, axis=0)

    psi_0 = Ak * vc

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    dt = (1.0 / 33) / ec[0] * 2 * np.pi

    def update(frame):
        t = frame * dt

        ax.clear()
        ax.set(zlim=[-1, 1], xlabel='x (m)', ylabel='y (m)', zlabel='psi(x, t)', title='t = {} (s)'.format(t))

        psi = np.sum(psi_0 * np.exp(-1j * ec * t), axis=2)
        z = np.abs(psi)
        ax.plot_surface(xs, ys, z, cmap='viridis')


    an = anim.FuncAnimation(fig=fig, func=update, interval=33, cache_frame_data=False)
    plt.show()

def qho3d():
    N = 12
    xmax = 5
    dx =  2 * xmax / N

    f = 8.88e13
    w = 2 * np.pi * f

    m = 9.109e-31 # kg
    hb = 1.054571817e-34 * 6.242e18 # J s 

    c = N # Energy level to map

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

    T = -hb**2 / (2 * m * dx**2) * np.select([x0 & y0 & z0, (x1 & y0 & z0) | (x0 & y1 & z0) | (x0 & y0 & z1)], [-6, 1], 0)

    xs, ys, zs = f(xis), f(yis), f(zis)
    U = 0.5 * m * w**2 * (xs**2 + ys**2 + zs**2)

    H = T + np.diag(U)
    e, v = np.linalg.eigh(H)

    vc = np.reshape(v[:, :c], (N + 1, N + 1, N + 1, c))
    ec = e[:c]

    xis, yis, zis = np.meshgrid(f(xi), f(yi), f(zi))

    psi = gaussian3d(xis, yis, zis, -1, 1)
    Ak = np.trapz(np.trapz(np.trapz(vc * psi[:, :, :, np.newaxis], dx=dx, axis=0), dx=dx, axis=0), dx=dx, axis=0)
    Ak /= np.trapz(np.trapz(np.trapz(np.sum(Ak * vc, axis=2), dx=dx, axis=0), dx=dx, axis=0), dx=dx, axis=0)
    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    dF = 33
    dt = 2 * np.pi / (ec[0] * dF)

    def update(frame):
        t = frame * dt

        ax.clear()
        ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title='t = {} (s)'.format(t))

        psi = np.sum(Ak * vc * np.exp(-1j * ec * t), axis=3)
        cf = np.abs(psi)
        ax.scatter(xis, yis, zis, c=cf, cmap='viridis')


    an = anim.FuncAnimation(fig=fig, func=update, interval=dF)
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
