
from array import array
from functools import partial
import math
import matplotlib.pyplot as plt
import numpy as np

def sine(x, freq):
    return np.sin(2 * math.pi * x * freq + math.pi / 2)

def composite_sine(x, freqs):
    return sum([np.sin(2 * math.pi * x * f + math.pi / 2) for f in freqs])

def generate_samples(f, rate, duration):
    x_vals = np.arange(0, duration, 1/rate)
    y_vals = f(x_vals)

    return np.stack([x_vals, y_vals])

def write_to_file(path, samples):
    if not path.endswith('.bin'):
        path.append('.bin')

    with open(path, 'wb') as f:
        length = len(samples[0])
        f.write(length.to_bytes(4, 'little')) # Sample count

        array('f', [float(x) for x in samples[0]]).tofile(f)
        array('f', [float(y) for y in samples[1]]).tofile(f)

def plot_samples(samples, rate, freq, plot_count):
    sample_range = math.ceil(rate/freq * plot_count)
    plt.plot(samples[0][:sample_range], samples[1][:sample_range])
    plt.xlabel('time')

# Simple sine wave signal generation
def test1():
    FREQ = 440 # A
    SAMPLE_RATE_LO = 880 # Nyquist Frequency
    SAMPLE_RATE_HI = 16000 # 16 kh
    SAMPLE_DURATION = 0.5 # seconds
    PLOT_COUNT = 10 # cycles

    lo_samples = generate_samples(partial(sine, freq=FREQ), SAMPLE_RATE_LO, SAMPLE_DURATION)
    hi_samples = generate_samples(partial(sine, freq=FREQ), SAMPLE_RATE_HI, SAMPLE_DURATION)

    write_to_file("samples_{}hz.bin".format(FREQ), lo_samples)

    plot_samples(lo_samples, SAMPLE_RATE_LO, FREQ, PLOT_COUNT)
    plot_samples(hi_samples, SAMPLE_RATE_HI, FREQ, PLOT_COUNT)

    plt.show()

# Composite sine wave generation
def test2():
    FREQ = [5, 19, 54, 212, 300]
    SAMPLE_RATE_LO = 880
    SAMPLE_RATE_HI = 16000 # 16 kh
    SAMPLE_DURATION = 0.5 # seconds
    PLOT_COUNT = 10 # cycles

    lo_samples = generate_samples(partial(composite_sine, freqs=FREQ), SAMPLE_RATE_LO, SAMPLE_DURATION)
    hi_samples = generate_samples(partial(composite_sine, freqs=FREQ), SAMPLE_RATE_HI, SAMPLE_DURATION)

    write_to_file("samples_composite.bin", lo_samples)

    plot_samples(lo_samples, SAMPLE_RATE_LO, 300, PLOT_COUNT)
    plot_samples(hi_samples, SAMPLE_RATE_HI, 300, PLOT_COUNT)

    plt.show()

test1()
