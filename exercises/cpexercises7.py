import numpy as np
import matplotlib.pyplot as plt
import dcst
import sys
import types

def ex_7_1():

    N = 1000
    N2 = N/2
    squarewave = lambda x : np.floor(x / N2)
    sawtooth = lambda x : x
    modsine = lambda x : np.sin(np.pi * x / N) * np.sin(20 * np.pi * x / N)
    f = sawtooth

    x = np.arange(N)
    
    y1 = squarewave(x)
    y2 = sawtooth(x)
    y3 = modsine(x)
    
    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(x, y1)
    axs[0].plot(x, y2)
    axs[0].plot(x, y3)

    k1 = np.abs(np.fft.fft(y1))
    k2 = np.abs(np.fft.fft(y2))
    k3 = np.abs(np.fft.fft(y3))

    axs[1].plot(x[:int(N2)], k1[:int(N2)])
    axs[1].plot(x[:int(N2)], k2[:int(N2)])
    axs[1].plot(x[:int(N2)], k3[:int(N2)])

    plt.show()

def ex_7_2():
    data = np.loadtxt('sunspots.txt', float)
    x, y = np.hsplit(data, 2)

    x = x.squeeze()
    y = y.squeeze()

    k = np.abs(np.fft.fft(y))**2
    N2 = len(k)//2

    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(x, y)
    axs[1].plot(range(-N2, N2+1), np.roll(k, N2))

    kmax = k[1:N2].argmax()
    k2 = np.zeros(len(k))
    k2[kmax] = 100

    print(kmax)

    y2 = np.abs(np.fft.ifft(k2))

    axs[0].plot(x, 100* np.sin(2 * np.pi * kmax * x / len(x)))

    plt.show()

def find_frequency(samples, sample_rate):

    duration = len(samples) / sample_rate

    k = np.abs(np.fft.fft(samples))**2
    kmax = np.argmax(k)

    return kmax / duration

def ex_7_3():
    sample_rate = 44100

    y1 = np.loadtxt('trumpet.txt', float)
    y2 = np.loadtxt('piano.txt', float)
    f1 = find_frequency(y1, sample_rate)
    f2 = find_frequency(y2, sample_rate)
    
    print('Trumpet frequency: ', f1, ' hz')
    print('Piano frequency: ', f2, ' hz')


    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(range(len(y1)), y1)
    axs[1].plot(range(len(y2)), y2)
    plt.show()

def ex_7_4():
    y = np.loadtxt('dow.txt', float)

    k = np.fft.rfft(y)

    m1 = np.copy(k)
    m2 = np.copy(k)
    m1[len(k) // 10:] = 0
    m2[len(k) // 50:] = 0

    y1 = np.fft.irfft(m1)
    y2 = np.fft.irfft(m2)

    plt.plot(range(len(y)), y)
    plt.plot(range(len(y1)), np.real(y1))
    plt.plot(range(len(y2)), np.real(y2))

    plt.show()

def ex_7_5():
    f = lambda x : np.select([np.trunc(2 * x) % 2 == 0], [1], -1)

    N = 1000
    x = np.linspace(0, 1, N, False)
    y = f(x)

    k = np.fft.fft(y)
    m = np.copy(k)
    m[10:] = 0

    y2 = np.fft.ifft(m)

    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(x, y)
    axs[0].plot(x, y2)
    axs[1].plot(range(len(k)), np.real(k))
    axs[1].plot(range(len(m)), np.real(m))
    plt.show()

def ex_7_6a():
    y = np.loadtxt('dow2.txt', float)

    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(range(y.size), y)

    k = np.fft.fft(y)
    k[y.size // 50:] = 0
    y2 = np.fft.ifft(k)

    axs[1].plot(range(y2.size), y2)

    plt.show()

def ex_7_6b():
    y = np.loadtxt('dow2.txt', float)

    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(range(y.size), y)

    k = dcst.dct(y)
    k[y.size // 50:] = 0
    y2 = dcst.idct(k)

    axs[1].plot(range(y2.size), y2)

    plt.show()


default = ex_7_6b

if __name__ == '__main__':
    if len(sys.argv) == 1:
        default()

    name = sys.argv[1]
    name = 'ex_' + name

    this = sys.modules[__name__]
    f = getattr(this, name)

    if isinstance(f, types.FunctionType):
        f()
