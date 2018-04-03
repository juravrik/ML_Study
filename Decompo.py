import numpy as np


class PCA:

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        Z = X - np.mean(X, axis=0)
        self.covariance = np.cov(Z.T, bias=1)
        w, v = np.linalg.eig(self.covariance)
        self.components = v[:, :self.n_components]

    def transform(self, X):
        return np.dot(X, self.components)
