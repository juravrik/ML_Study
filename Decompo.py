import numpy as np


class PCA:

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        Z = X - np.mean(X, axis=0)
        self.covariance = np.cov(Z.T, bias=1)
        w, v = np.linalg.eig(self.covariance)
        d = dict(zip(w, [v[:, i] for i in range(len(w))]))
        d = sorted(d.items(), key=lambda x: x[0], reverse=True)
        v = np.vstack([i[1] for i in d]).T
        self.components = v[:, :self.n_components]

    def transform(self, X):
        return np.dot(X, self.components)
