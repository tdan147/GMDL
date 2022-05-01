import numpy as np
from ex9 import sample_gibbs


def ex10():
    latice_length = 100
    sweeps = 50
    temps = [1, 1.5, 2]
    for temp in temps:
        sample = sample_gibbs(latice_length, temp, sweeps)
        y = sample + 2 * np.random.standard_normal((100, 100))
