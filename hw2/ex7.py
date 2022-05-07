from matplotlib.gridspec import SubplotSpec

from hw2 import F, G, y2row
import numpy as np
import matplotlib.pyplot as plt



def calc_T(latice_length, temp):
    row_length = 2 ** latice_length
    T = []
    for i in range(latice_length-1):
        T_prev = np.ones(row_length) if i == 0 else T[i-1]
        T.append(calc_T_k(T_prev, latice_length, temp))
    G_last = np.asarray([G(y2row(i, width=latice_length), temp) for i in range(row_length)])
    T.append(T[-1] @ G_last)
    return T


def calc_T_k(T_prev, latice_length, temp):
    row_length = 2 ** latice_length
    T_k = np.zeros(row_length)
    for i in range(row_length):
        sum_T = 0
        i_2_row = y2row(i, width=latice_length)
        for j in range(row_length):
            j_2_row = y2row(j, width=latice_length)
            sum_T += T_prev[j] * G(j_2_row, temp) * F(j_2_row,i_2_row,temp)
        T_k[i] = sum_T
    return T_k


def calc_p(T, latice_length, temp):
    row_length = 2 ** latice_length
    G_last = np.asarray(list(map(lambda x: G(y2row(x, width=latice_length), temp), np.arange(row_length))))
    p_reveresed = [(G_last * T[-2]) / T[-1]]
    for i in range(6,-1,-1):
        T_prev = np.ones(row_length) if i == 0 else T[i-1]
        p_reveresed.append(calc_p_k(T[i], T_prev, latice_length, temp))
    return p_reveresed


def calc_p_k(T_curr, T_prev, latice_length,  temp):
    row_length = 2 ** latice_length
    p_k = np.zeros((row_length, row_length))
    for i in range(row_length):
        i_2_row = y2row(i, width=latice_length)
        for j in range(row_length):
            j_2_row = y2row(j, width=latice_length)
            p_k[i][j] = T_prev[i] * G(i_2_row, temp) * F(i_2_row, j_2_row, temp) / T_curr[j]
    return p_k


def sample_images(p, latice_length):
    row_length = 2 ** latice_length
    y_reversed = np.zeros(latice_length, dtype=int)
    y_reversed[0] = np.random.choice(row_length, p=p[0])
    for i in range(1,latice_length):
        y_reversed[i] = np.random.choice(row_length, p=p[i][:, y_reversed[i-1]])
    y = np.flip(y_reversed)
    image = np.asarray(list(map(lambda y: y2row(y), y)))
    return image

def plot(images):
    fig, axs = plt.subplots(3, 10, figsize=(20,8))
    temp_title = ["temp=1", "temp=1.5", "temp=2"]
    for i in range(3):
        for j in range(10):
            if j == 0:
                axs[i, j].set_ylabel(temp_title[i])
            axs[i, j].imshow(images[i][j], interpolation='None', vmin=-1, vmax=1, cmap="gray")
            axs[i, j].set_xticks([]), axs[i, j].set_yticks([])
    plt.show()

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold', fontsize=20)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

def ex7():
    latice_length = 8
    temps = [1, 1.5, 2]
    images = []
    for temp in temps:
        T = calc_T(latice_length, temp)
        p = calc_p(T, latice_length, temp)
        temp_images = [sample_images(p, latice_length) for i in range(10)]
        images.append(temp_images)
    plot(images)


def ex8():
    latice_length = 8
    temps = [1, 1.5, 2]
    for temp in temps:
        T = calc_T(latice_length, temp)
        p = calc_p(T, latice_length, temp)
        temp_images = [sample_images(p, latice_length) for i in range(10000)]
        expectations_1_2 = calc_expectation(temp_images, 1,2)
        expectations_1_8 = calc_expectation(temp_images, 1,8)
        print(f'temp= {temp}')
        print(f'empirical expectation for entries 1,2 is: {expectations_1_2}')
        print(f'empirical expectation for entries 1,8 is: {expectations_1_8}')


def calc_expectation(images, entry1, entry2):
    X_entry1 = np.asarray([images[i][entry1-1][entry1-1] for i in range(len(images))])
    X_entry2 = np.asarray([images[i][entry2-1][entry2-1] for i in range(len(images))])
    return X_entry1 @ X_entry2 / 10000


# ex8()