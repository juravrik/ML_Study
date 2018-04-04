import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sns
import BinClass
import LineReg
import Decompo


def fourier():
    '''
    標準偏差0.3の正規分布によるノイズをかけたsin関数をデータセットとして使用し、フーリエ級数での回帰
    '''

    x = np.arange(0, 10).reshape(10, 1)
    noize = np.random.normal(loc=0, scale=0.3, size=10).reshape(10, 1)
    y = np.sin(x) + noize
    pyplot.scatter(x, y, label='data')
    prd = LineReg.Fourier(m=3, lam=1.5)
    prd.fit(x, y)
    pyplot.plot(x, np.sin(x), label='true')
    pyplot.plot(x, prd.predict(x), label='pred')
    pyplot.legend()
    pyplot.show()


def perseptron():
    '''
    '''

    noize = np.random.normal(loc=0, scale=0.3, size=40).reshape(20, 2)
    zeros = np.zeros([10, 1])
    ones = np.ones([10, 1])
    negative = np.hstack([ones,  zeros])
    positive = np.hstack([zeros, ones])
    x = np.vstack([negative, positive]) + noize
    y = np.vstack([np.zeros(10), np.ones(10)]).reshape(20, 1)
    x /= x.max(axis=0)

    pyplot.scatter(x[:10, 0], x[:10, 1], marker='x')
    pyplot.scatter(x[10:, 0], x[10:, 1], marker='o')

    clf = BinClass.Perseptron(eta=1, itr=100)
    clf.fit(x, y)

    x = np.arange(-0.1, 1, 0.01)
    y = -(clf.w[0]+clf.w[1]*x)/clf.w[2]
    pyplot.plot(x, y, label='Boundary')

    pyplot.show()


if __name__ == '__main__':
    sns.set()
    if '--fourier' in sys.argv:
        fourier()

    # if '--logistic' in sys.argv:
    #    logistic()

    if '--perseptron' in sys.argv:
        perseptron()

    # if '--pca' in sys.argv:
        # pca()
