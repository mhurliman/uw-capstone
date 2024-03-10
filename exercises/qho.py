import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Creates a tridiagonal matrix with a, b, c as the repeated elements along the diagonal
def triagonal(a, b, c, n):
    return np.diag([a]*(n-1), -1) + np.diag([b]*n, 0) + np.diag([c]*(n-1), 1)

def qho1d():
    N = 1000
    L = 50
    a = L / N
    h = 1

    w = 2e-2
    m = 1#9.109e-31 # kg
    hb = 1#1.054571817e-34 # J s 

    c = 10 # Energy levels to map

    x = np.linspace(-L, L, N+1)

    U = m * w**2 * x**2
    T = -hb**2 / (2 * m * a**2) * triagonal(1, -2, 1, N+1)
    H = T + np.diag(U)

    e, v = np.linalg.eigh(H)

    fig, ax = plt.subplots()
    wave = ax.plot(x, np.zeros([N+1, c]))
    ax.set(ylim=[-1, 1], xlabel='x (m)', ylabel='psi(x, t)')

    def update(frame):
        t = frame * h
        ax.set(title='t = {}'.format(t))

        y = np.real(v[:, :c] * np.exp(1j * e[:c] * t)) + e[:c]

        for i, w in enumerate(wave):
            w.set_ydata(y[:, i])

    an = anim.FuncAnimation(fig=fig, func=update, interval=1)

    plt.show()

def qho2d():
    N = 1000
    L = 3

    w = 1e-4
    m = 1 #9.109e-31 # kg
    hb = 1 #1.054571817e-34 # J s 

    psi = lambda x, n: np.exp(1j * np.pi * x)

    U = lambda x : m * w**2 * x**2
    a = -hb**2 / (2 * m)

    T = a * triagonal(1, -2, 1, N + 1)
    x = np.linspace(-L, L, N + 1)

    H = T + np.diag(U(x))

    e, v = np.linalg.eigh(H)

    for i in range(3):
        psiv = v[:, i] + e[i] * np.ones_like(x)
        plt.plot(x, psiv)
        

    plt.show()

if __name__ == '__main__':
    qho1d()
