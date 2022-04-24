import numpy as np
import matplotlib.pyplot as plt


def Q_norm(Q, x1, x2):
    return np.sqrt(x1 ** 2 * Q[0, 0] + x2 ** 2 * Q[1, 1] + 2 * x1 * x2 * Q[0, 1])


def plot_l1_norm():
    x2, x1 = np.mgrid[-2:2:0.01, -2:2:0.01]
    fig, ax = plt.subplots()
    fig.suptitle('l1 distance')
    plt.imshow(np.logical_and((np.abs(x1) + np.abs(x2)) <= 1.001, (np.abs(x1) + np.abs(x2)) >= 0.999), cmap='binary', extent=[-2,2,-2,2])
    plt.title('d(x, 0) = 1')
    plt.show()
    plt.show()


def plot_norm(Q, title):
    x2, x1 = np.mgrid[-2:2:0.01, -2:2:0.01]
    fig, ax = plt.subplots()
    fig.suptitle(title)
    plt.title(title)
    plt.imshow(np.logical_and(Q_norm(Q, x1, x2) <= 1.015,(Q_norm(Q, x1, x2) >= 0.985)), cmap='binary',extent=[-2,2,-2,2])
    plt.title('d(x, 0) = 1')
    plt.show()



plot_norm(np.eye(2), 'l2 distance')
plot_l1_norm()
plot_norm(np.array([[9, 0], [0, 1]]), 'Q1 distance')
plot_norm(np.array([[9, 2], [2, 1]]), 'Q2 distance')
plot_norm(np.array([[9, -2], [-2, 1]]), 'Q3 distance')
