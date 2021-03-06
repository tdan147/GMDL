import numpy as np


p_D = np.asarray([0.6, 0.4])
p_I = np.asarray([0.7, 0.3])
p_S_I = np.asarray([[0.95, 0.05], [0.2, 0.8]])
p_G_D_I = np.asarray([[0.3,0.4,0.3], [0.05, 0.25, 0.7], [0.9, 0.08, 0.02], [0.5, 0.3, 0.2]])
p_L_G = np.asarray([[0.1, 0.9], [0.4, 0.6], [0.99, 0.01]])


f3_G_msg = p_L_G.sum(axis=1)
f1_G = [0, 0, 0]
f2_I = p_S_I.sum(axis=1)
I_f1 = f2_I
f1_I = [0.0, 0.0]
indexes = [[0, 1], [2, 3]]

def marginal_s():
    G_f2 = f3_G_msg
    for i in range(2):
        for j in range(3):
            f1_I[0] += p_I[0] * p_D[i] * p_G_D_I[0:2][i][j] * G_f2[j]
            f1_I[1] += p_I[1] * p_D[i] * p_G_D_I[2:4][i][j] * G_f2[j]
    I_f3 = f1_I
    f2_S = np.asarray([0.0,0.0])
    for i in range(2):
        f2_S[0] += p_S_I[i][0] * I_f3[i]
        f2_S[1] += p_S_I[i][1] * I_f3[i]
    return f2_S

def marginal_l():

    for i in range(2):
        for j in range(2):
            f1_G[0] += p_D[j] * p_I[i] * p_G_D_I[indexes[i][j]][0] * I_f1[i]
            f1_G[1] += p_D[j] * p_I[i] * p_G_D_I[indexes[i][j]][1] * I_f1[i]
            f1_G[2] += p_D[j] * p_I[i] * p_G_D_I[indexes[i][j]][2] * I_f1[i]
    G_f3 = f1_G
    f3_L = np.asarray([0.0, 0.0])
    for i in range(3):
        f3_L[0] += p_L_G[i][0] * G_f3[i]
        f3_L[1] += p_L_G[i][1] * G_f3[i]
    return f3_L

def marginal_g():
    prop_margin = f3_G_msg * f1_G
    return prop_margin / prop_margin.sum()


def marginal_D():
    G_f1 = f3_G_msg
    f1_D = [0, 0]
    for g in range(3):
        for i in range(2):
            f1_D[0] += p_D[0] * G_f1[g] * I_f1[i]
            f1_D[1] += p_D[1] * G_f1[g] * I_f1[i]
    f1_D = np.asarray(f1_D)
    return f1_D / f1_D.sum()

def marginal_I():
    prop_margin = f1_I * f2_I
    return prop_margin / prop_margin.sum()


def main():
    print(f'S: {marginal_s()}')
    print(f'L: {marginal_l()}')
    print(f'G: {marginal_g()}')
    print(f'D: {marginal_D()}')
    print(f'I: {marginal_I()}')

main()