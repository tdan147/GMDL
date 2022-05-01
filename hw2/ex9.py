import numpy as np
from copy import copy
from ex7 import calc_expectation


def independent_samples(latice_length, temp, num_of_samples, num_of_sweeps):
    samples = []
    for i in range(num_of_samples):
        if i % 100 == 0:
            print(f'sample number: {i}')
        samples.append(sample_gibbs(latice_length, temp, num_of_sweeps))
    return samples


def sample_gibbs(latice_length, temp, num_of_sweeps):
    sample = np.random.randint(0,2, (latice_length,latice_length))*2-1
    padded_sample = np.pad(sample, ((1,1),(1,1)), 'constant')
    for i in range(num_of_sweeps):
        padded_sample = sweep(latice_length, padded_sample, temp)
    return padded_sample[1:-1, 1:-1]



def sweep(latice_length, padded_sample, temp):
    old_padded_sample = copy(padded_sample)
    for i in range(1,latice_length):
        for j in range(1, latice_length):
            prob = calc_prob(old_padded_sample, i, j, temp)
            padded_sample[i][j] = np.random.choice([-1, 1], p=prob)
    return padded_sample


def calc_prob(sample, i, j, temp):
    sum_neigh = calc_sum_neigh(sample, i, j)
    pos_prob = np.exp(1/temp * sum_neigh)
    neg_prob = np.exp(-1/temp * sum_neigh)
    z_temp = pos_prob + neg_prob
    return [pos_prob / z_temp, neg_prob / z_temp]


def calc_sum_neigh(sample, i, j):
    return sample[i+1][j] + sample[i-1][j] + sample[i][j+1] + sample[i][j-1]


def main():
    temps = [1, 1.5, 2]
    for temp in temps:
        samples = independent_samples(8, temp, 10000, 25)
        expectations_1_2 = calc_expectation(samples, 1, 2)
        expectations_1_8 = calc_expectation(samples, 1, 8)
        print(f'temp= {temp}')
        print(f'method1: empirical expectation for entries 1,2 is: {expectations_1_2}')
        print(f'method1: empirical expectation for entries 1,8 is: {expectations_1_8}')
        expectations_1_2, expectations_1_8 = ergodicity(8, temp, 25000, 100)
        print(f'method2: empirical expectation for entries 1,2 is: {expectations_1_2}')
        print(f'method2: empirical expectation for entries 1,8 is: {expectations_1_8}')



main()


def ergodicity(latice_length, temp, num_of_sweeps, burnin):
    remaining_sweeps = num_of_sweeps - burnin
    sample = sample_gibbs(latice_length, temp, burnin)
    padded_sample = np.pad(sample, ((1,1),(1,1)), 'constant')
    x_1_2, x_1_8 = 0, 0
    for i in range(remaining_sweeps):
        padded_sample = sweep(latice_length, padded_sample, temp)
        x_1_2 = (i * x_1_2 + padded_sample[1][1] * padded_sample[2][2]) / (i + 1)
        x_1_8 = (i * x_1_8 + padded_sample[1][1] * padded_sample[8][8]) / (i + 1)
    return x_1_2, x_1_8