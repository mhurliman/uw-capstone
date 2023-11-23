
from functools import partial
import math
import matplotlib.pyplot as plt
import numpy as np

def sine(x, freq):
    return np.sin(2 * math.pi * x * freq)

def generate_samples(f, rate, duration):
    x_vals = np.arange(0, duration, 1/rate)
    y_vals = f(x_vals)

    return np.stack([x_vals, y_vals])

def write_to_file(path, samples):
    if not path.endswith('.csv'):
        path.append('.csv')

    with open(path, 'w') as f:
        f.write(len(samples[0])) # Sample count
        f.write(','.join([str(x) for x in samples[0]])) # X samples
        f.write('\n')
        f.write(','.join([str(y) for y in samples[1]])) # Y samples

def plot_samples(samples, rate, freq, plot_count):
    sample_range = math.ceil(rate/freq * plot_count)
    plt.plot(samples[0][:sample_range], samples[1][:sample_range])
    plt.xlabel('time')

def test():
    FREQ = 440 # A
    SAMPLE_RATE_LO = 880
    SAMPLE_RATE_HI = 16000 # 16 kh
    SAMPLE_DURATION = 0.5 # seconds
    PLOT_COUNT = 10 # cycles

    lo_samples = generate_samples(partial(sine, freq=FREQ), SAMPLE_RATE_LO, SAMPLE_DURATION)
    hi_samples = generate_samples(partial(sine, freq=FREQ), SAMPLE_RATE_HI, SAMPLE_DURATION)

    write_to_file("samples_440hz.csv", lo_samples)

    plot_samples(lo_samples, SAMPLE_RATE_LO, FREQ, PLOT_COUNT)
    plot_samples(hi_samples, SAMPLE_RATE_HI, FREQ, PLOT_COUNT)

    plt.show()
