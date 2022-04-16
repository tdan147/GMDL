import numpy as np
from itertools import product


def G(row_s, temp):  # first task of 1.1
    return np.exp((1/temp) * (row_s[:-1] @ row_s[1:]))


def F(row_s, row_t, temp):  # second task of 1.1
    return np.exp((1 / temp) * (row_s @ row_t))


def ex3():
    temps = [1,1.5,2]
    z_temp = np.zeros(3)
    perms = list(product([-1, 1], repeat=4))
    for perm in perms:
        m = np.asarray(perm).reshape(2,2)
        for (ind, temp) in enumerate(temps):
            z_temp[ind] += F(m[0], m[1], temp) * F(m[:, 0], m[:, 1], temp)
    print(z_temp)


def ex4():
    temps = [1, 1.5, 2]
    z_temp = np.zeros(3)
    perms = list(product([-1, 1], repeat=9))
    for perm in perms:
        m = np.asarray(perm).reshape(3,3)
        for (ind, temp) in enumerate(temps):
            z_temp[ind] += F(m[0], m[1], temp) * F(m[1], m[2], temp) * F(m[:, 0], m[:, 1], temp) * F(m[:, 1], m[:, 2], temp)
    print(z_temp)


def y2row(y,width=8):
    """
    y: an integer in (0,...,(2**width)-1)
    """
    if not 0<=y<=(2**width)-1:
        raise ValueError(y)
    my_str=np.binary_repr(y,width=width)
    my_list = list(map(int,my_str)) # Python 3
    my_array = np.asarray(my_list)
    my_array[my_array==0]=-1
    row=my_array
    return row

def ex5():
    temps = [1, 1.5, 2]
    z_temp = np.zeros(3)
    y_values = [0,1,2,3]
    for y1 in y_values:
        for y2 in y_values:
            y1_row = y2row(y1, width=2)
            y2_row = y2row(y2, width=2)
            for (ind, temp) in enumerate(temps):
                z_temp[ind] += G(y1_row,temp) * G(y2_row, temp) * F(y1_row, y2_row, temp)
    print(z_temp)

def ex6():
    temps = [1, 1.5, 2]
    z_temp = np.zeros(3)
    y_values = [i for i in range(8)]
    for y1 in y_values:
        for y2 in y_values:
            for y3 in y_values:
                y1_row = y2row(y1, width=3)
                y2_row = y2row(y2, width=3)
                y3_row = y2row(y3, width=3)
                for (ind, temp) in enumerate(temps):
                    z_temp[ind] += G(y1_row,temp) * G(y2_row, temp) * G(y3_row, temp) * \
                                   F(y1_row, y2_row, temp) * F(y2_row, y3_row, temp)
    print(z_temp)

