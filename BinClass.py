import numpy as np


class Perceptron:
    """
    SGDによる単層パーセプトロン
    """

    def __init__(self, eta, itr):
        """
        param
        -----
        * eta: 学習率
        * itr: 勾配法の繰り返し回数
        """
        self.eta = eta
        self.itr = itr

    def fit(self, x, y):
        """
        確率的勾配効果法によるパーセプトロン学習
        """
        self.w = np.random.rand(x.shape[1]+1, 1)
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        y = np.where(y == 0, -1, 1)

        def conv(x):
            return np.where(np.dot(x, self.w) > 0, 1, -1)

        for i in range(self.itr):
            tmp_idx = np.random.choice(range(x.shape[0]))
            tmp_x = x[tmp_idx, :]
            tmp_y = y[tmp_idx, :]

            grad = (self.eta * tmp_y * tmp_x).reshape(tmp_x.shape[0], 1)
            self.w = np.where(conv(tmp_x) == tmp_y, self.w, self.w + grad)

    def predict(self, x):
        """
        予測ラベルの出力
        """
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        return np.where(np.dot(x, self.w) > 0, 1, 0)
