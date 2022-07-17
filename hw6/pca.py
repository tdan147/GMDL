# Import Libraries
import scipy.linalg
import utils
import numpy as np
from sklearn.preprocessing import StandardScaler


def PCA(X,components):
    d = X.shape[1]
    X_mean = X - StandardScaler().fit(X).mean_
    correlation = X_mean.T @ X_mean
    W, V = scipy.linalg.eigh(correlation, subset_by_index=[d-components, d-1])
    V_flipped = np.flip(V, axis=1)
    return X @ V_flipped

def main():
    data, labels = utils.get_pca_data()
    pca = PCA(data, 2)
    utils.scatter_plot(pca.T, labels, 10)

if __name__ == "__main__":
    main()

