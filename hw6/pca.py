# Import Libraries
import scipy.linalg
import utils
import numpy as np


def PCA(X,components):  # each sample is a column
    d = X.shape[0]
    mean = np.mean(X, axis=1).reshape(d, 1)
    X_mean = X - mean
    correlation = X_mean @ X_mean.T
    W, V = scipy.linalg.eigh(correlation, subset_by_index=[d-components, d-1])
    V_flipped = np.flip(V, axis=1)  # TODO validate vectors are columns
    return V_flipped @ V_flipped.T @ X


def main():
    data, labels = utils.get_pca_data()
    pca = PCA(data, 2)
    utils.scatter_plot(pca, labels, 10)

if __name__ == "__main__":
    main()

