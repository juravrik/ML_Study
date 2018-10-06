import numpy as np

class Cos:
    '''
    cos類似度によるマルチクラス識別器
    '''

    def fit(self, X, y):
        self.mean_vec = {i:np.mean(X[np.where(y==i)[0], :], axis=0) for i in np.unique(y)}

    def predict(self, x):
        l = []
        for i in x:
            def _cossim(a):
                return ((a @ i)/(np.linalg.norm(i)*np.linalg.norm(a)))**2

            d = dict(zip(self.mean_vec.keys(), list(map(_cossim, self.mean_vec.values()))))
            l.append(max(d.items(), key=lambda x:x[1])[0])
        return np.array(l)


class Maharanobis:
    def fit(self, X, y):
        self.mean_vec = {i:np.mean(X[np.where(y==i)[0], :], axis=0) for i in np.unique(y)}

    def predict(self, x):
        l = []

        c = np.cov(x.T)
        for i in x:
            def _maha(a):
                t = np.reshape((i-a), ((i-a).shape[0], 1))
                return np.sqrt(((t.T @ np.linalg.inv(c)) @ t))

            d = dict(zip(self.mean_vec.keys(), list(map(_maha, self.mean_vec.values()))))
            l.append(min(d, key=d.get))
        return np.array(l)
