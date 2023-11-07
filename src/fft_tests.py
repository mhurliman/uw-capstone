
import math
import numpy as np
import matplotlib.pyplot as plt

FREQ = 440 # A
SAMPLE_RATE_LO = 880
SAMPLE_RATE_HI = 16000 # 16 kh
SAMPLE_DURATION = 0.5 # seconds
PLOT_COUNT = 10 # cycles

def func(x):
    return np.sin(2 * math.pi * x * FREQ)

def sample_func(f, rate, duration):
    x_vals = np.arange(0, duration, 1/rate)
    y_vals = f(x_vals)

    return np.stack([x_vals, y_vals])

def write_to_file(samples):
    filename = 'file.txt'

    with open(filename, 'w') as f:
        f.write(','.join([str(x) for x in samples[0]]))
        f.write('\n')
        f.write(','.join([str(y) for y in samples[1]]))

def plot_samples(samples, rate, freq, plot_count):
    sample_range = math.ceil(rate/freq * plot_count)
    plt.plot(samples[0][:sample_range], samples[1][:sample_range])


lo_samples = sample_func(func, SAMPLE_RATE_LO, SAMPLE_DURATION)
hi_samples = sample_func(func, SAMPLE_RATE_HI, SAMPLE_DURATION)

write_to_file(lo_samples)

plot_samples(lo_samples, SAMPLE_RATE_LO, FREQ, PLOT_COUNT)
plot_samples(hi_samples, SAMPLE_RATE_HI, FREQ, PLOT_COUNT)

plt.xlabel('time')
plt.show()
