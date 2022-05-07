import numpy as np
from ex9 import sample_gibbs, calc_sum_neigh
from copy import copy
import matplotlib.pyplot as plt
from ex7 import create_subtitle


def ex10():
    latice_length = 100
    sweeps = 50
    temps = [1, 1.5, 2]
    images = []
    images_titles = ["x", "y", "posterior", "ICM", "MLE"]
    temp_title = ["temp=1", "temp=1.5", "temp=2"]
    for temp in temps:
        sample = sample_gibbs(latice_length, temp, sweeps)
        y = sample + 2 * np.random.standard_normal((latice_length, latice_length))
        posterior_sample = sample_gibbs_posterior(latice_length, temp, sweeps, y, sweep)
        icm_sample = sample_gibbs_posterior(latice_length, temp, sweeps, y, sweep_icm)
        mle_sample = np.sign(y)
        images.append([sample, y, posterior_sample, icm_sample, mle_sample])
    fig, axs = plt.subplots(3, 5, figsize=(20, 8))
    for i in range(3):
        for j in range(5):
            if i == 0:
                axs[i, j].title.set_text(images_titles[j])
            if j == 0:
                axs[i, j].set_ylabel(temp_title[i])
            if j == 1:
                axs[i, j].imshow(images[i][j], interpolation='None', cmap="gray")
            else:
                axs[i, j].imshow(images[i][j], interpolation='None', vmin=-1, vmax=1, cmap="gray")
            axs[i, j].set_xticks([]), axs[i, j].set_yticks([])

    plt.show()


def calc_prob_posterior(sample, i, j, temp, y):
    sum_neigh = calc_sum_neigh(sample, i, j)
    pos_prob = np.exp(1/temp * sum_neigh - 1/8 * (y[i][j] - 1)**2)
    neg_prob = np.exp(-1/temp * sum_neigh- 1/8 * (y[i][j] + 1)**2)
    z_temp = pos_prob + neg_prob
    return [neg_prob / z_temp, pos_prob / z_temp]




def sweep(latice_length, padded_sample, temp, y):
    old_padded_sample = copy(padded_sample)
    for i in range(1,latice_length+1):
        for j in range(1, latice_length+1):
            prob = calc_prob_posterior(old_padded_sample, i, j, temp, y)
            padded_sample[i][j] = np.random.choice([-1, 1], p=prob)
    return padded_sample


def sweep_icm(latice_length, padded_sample, temp, y):
    old_padded_sample = copy(padded_sample)
    for i in range(1,latice_length+1):
        for j in range(1, latice_length+1):
            prob = calc_prob_posterior(old_padded_sample, i, j, temp, y)
            padded_sample[i][j] = np.argmax(prob) * 2 - 1
    return padded_sample


def sample_gibbs_posterior(latice_length, temp, num_of_sweeps, y, sweep_func):
    sample = np.random.randint(0,2, (latice_length,latice_length))*2-1
    padded_sample = np.pad(sample, ((1,1),(1,1)), 'constant')
    for i in range(num_of_sweeps):
        padded_sample = sweep_func(latice_length, padded_sample, temp, y)
    return padded_sample[1:-1, 1:-1]


def independent_samples_posterior(latice_length, temp, num_of_samples, num_of_sweeps, y):
    samples = []
    for i in range(num_of_samples):
        if i % 100 == 0:
            print(f'sample number: {i}')
        samples.append(sample_gibbs_posterior(latice_length, temp, num_of_sweeps, y))
    return samples


# ex10()