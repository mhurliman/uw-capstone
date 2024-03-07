import numpy as np
import matplotlib.pyplot as plt

# Creates a tridiagonal matrix with a, b, c as the repeated elements along the diagonal
def triagonal(a, b, c, n):
    return np.diag([a]*(n-1), -1) + np.diag([b]*n, 0) + np.diag([c]*(n-1), 1)

def qho():
    N = 2000
    L = 3

    w = 1
    m = 9.109e-31 # kg
    hb = 1.054571817e-34 # J s 


    U = lambda x : m * w**2 * x**2
    a = -hb**2 / (2 * m)

    A = a * triagonal(1, -2, 1, N + 1)
    x = np.linspace(-L, L, N + 1)

    

if __name__ == '__main__':
    qho()
